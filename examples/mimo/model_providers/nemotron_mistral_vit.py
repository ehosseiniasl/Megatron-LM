# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Nemotron6-MoE and Mistral-ViT provider for heterogeneous MIMO."""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

import torch

from examples.mimo.model_providers import MimoProvider
from examples.mimo.model_providers.nemotron_moe_vlm import (
    language_model_spec,
    nemotron_projection_layer_spec,
)
from examples.mimo.model_providers.radio_encoder import (
    _base_config,
    _disable_gtp,
    _dtype,
    _make_dense_non_hybrid,
)
from examples.mimo.utils.hetero import get_grid_dim_size
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.models.vision.vit_model import ViTModel
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_size

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:  # pragma: no cover - TE always present in the CI container
    TEColumnParallelLinear = None
    TERowParallelLinear = None

if TYPE_CHECKING:
    from examples.mimo.training.topology import HeteroTopology

NEMOTRON_MISTRAL_VIT_MODEL_PROVIDER = "nemotron-moe-mistral-vit"
MISTRAL_VIT_ENCODER_MODULE_NAME = "mistral_vit_encoder"

_MISTRAL_VIT_PROJECTION_HIDDEN_SIZES = {"410m": 1024, "1.6b": 2560}


def _uncompiled_fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU without the global MCore torch.compile decorator."""
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


class MimoMistralViTModel(ViTModel):
    """Mistral ViT that retains its explicit MIMO tensor-parallel group."""

    # `name` is named explicitly rather than left to **kwargs so that callers can see
    # this model takes one; MIMO only threads a quantization path into constructors
    # that declare the argument.
    def __init__(self, *args, pg_collection=None, name=None, **kwargs):
        super().__init__(*args, pg_collection=pg_collection, name=name, **kwargs)
        if pg_collection is not None:
            self.tp_group = pg_collection.tp


def mistral_vit_vision_config(
    args: argparse.Namespace, tp_size: int, pp_size: int
) -> TransformerConfig:
    """Build the selected Mistral-ViT configuration."""
    config = deepcopy(_base_config(args))
    bf16, dtype = _dtype(args)
    model_size = getattr(args, "mistral_vit_model_size", "410m")
    if model_size == "410m":
        config.num_layers = 24
        config.hidden_size = 1024
        config.ffn_hidden_size = 4096
        config.kv_channels = 64
    elif model_size == "1.6b":
        config.num_layers = 30
        config.hidden_size = 2048
        config.ffn_hidden_size = 6144
        config.kv_channels = 128
    else:
        raise ValueError(f"Unsupported Mistral-ViT model size: {model_size}")
    config.num_attention_heads = 16
    config.num_query_groups = 16
    config.normalization = "RMSNorm"
    config.layernorm_epsilon = 1.0e-5
    config.activation_func = torch.nn.functional.silu
    config.gated_linear_unit = True
    config.add_bias_linear = False
    config.add_qkv_bias = False
    config.rotary_interleaved = True
    config.qk_layernorm = False
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.layernorm_zero_centered_gamma = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    if args.mimo_vision_encoder_attention_backend is not None:
        config.attention_backend = args.mimo_vision_encoder_attention_backend
    if args.mimo_vision_encoder_flash_attention_version is not None:
        config.flash_attention_version = args.mimo_vision_encoder_flash_attention_version
    config.apply_rope_fusion = False
    config.mtp_num_layers = None
    _make_dense_non_hybrid(config)
    config.params_dtype = dtype
    config.pipeline_dtype = dtype
    config.bf16 = bf16
    config.tensor_model_parallel_size = tp_size
    config.pipeline_model_parallel_size = pp_size
    config.context_parallel_size = 1
    _disable_gtp(config)
    config.sequence_parallel = False
    return config


def mistral_vit_encoder_spec(
    args: argparse.Namespace,
    vision_config: TransformerConfig,
    pg_collection: ProcessGroupCollection | None,
) -> ModuleSpec:
    """Build the bare MCore ViT with the fixed native-image geometry."""
    patch_dim = getattr(args, "patch_dim", 14)
    spatial_merge_size = getattr(args, "spatial_merge_size", 2)
    return ModuleSpec(
        module=MimoMistralViTModel,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
            "patch_dim": patch_dim,
            "img_h": args.img_h,
            "img_w": args.img_w,
            "add_class_token": False,
            "class_token_len": 0,
            "ln_pre": True,
            "pos_emb_type": "rope2d",
            "use_merger": spatial_merge_size > 1,
            "spatial_merge_size": spatial_merge_size,
            "pg_collection": pg_collection,
        },
    )


def mistral_vit_projection_config(
    args: argparse.Namespace, tp_size: int, base_config: TransformerConfig | None = None
) -> TransformerConfig:
    """Build the fixed non-GLU MLP projection to the Nemotron hidden width."""
    config = deepcopy(base_config if base_config is not None else _base_config(args))
    config.num_layers = 1
    config.hidden_size = int(args.hidden_size)
    config.num_attention_heads = 1
    model_size = getattr(args, "mistral_vit_model_size", "410m")
    config.ffn_hidden_size = _MISTRAL_VIT_PROJECTION_HIDDEN_SIZES[model_size]
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.add_bias_linear = False
    _make_dense_non_hybrid(config)
    config.context_parallel_size = 1
    config.pipeline_model_parallel_size = 1
    config.tensor_model_parallel_size = tp_size
    if base_config is None:
        _disable_gtp(config)
    config.sequence_parallel = False
    config.tensor_parallel_num_sequence_shards = None
    return config


def mistral_vit_mlp2_projection_config(
    args: argparse.Namespace, tp_size: int, base_config: TransformerConfig | None = None
) -> TransformerConfig:
    """Build ``W2(FastGELU(W1(z)))`` with both hidden widths matching the decoder."""
    config = mistral_vit_projection_config(args, tp_size, base_config)
    config.ffn_hidden_size = int(args.hidden_size)
    config.activation_func = _uncompiled_fast_gelu
    return config


def mistral_vit_projection_layer_spec() -> MLPSubmodules:
    """Return the tensor-parallel linears for the fixed MLP projector."""
    if TEColumnParallelLinear is None or TERowParallelLinear is None:
        raise RuntimeError("TEColumnParallelLinear and TERowParallelLinear are required")
    return MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear)


def _mistral_vit_projection_spec(
    args: argparse.Namespace,
    input_size: int,
    tp_size: int,
    pg_collection: ProcessGroupCollection | None,
    base_config: TransformerConfig | None = None,
) -> ModuleSpec:
    """Build the Mistral-ViT projection for its owning module grid."""
    projection_type = getattr(args, "vision_projection_type", "mlp")
    projection_submodules = (
        nemotron_projection_layer_spec().submodules
        if projection_type == "affine"
        else mistral_vit_projection_layer_spec()
    )
    return ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": (
                mistral_vit_mlp2_projection_config(args, tp_size, base_config)
                if projection_type == "mlp2"
                else mistral_vit_projection_config(args, tp_size, base_config)
            ),
            "submodules": projection_submodules,
            "projector_type": "mlp" if projection_type == "mlp2" else projection_type,
            "input_size": input_size,
            "pg_collection": pg_collection,
        },
    )


def mistral_vit_submodules_spec(
    args: argparse.Namespace,
    pg_collection: ProcessGroupCollection | None,
    encoder_grid: HyperCommGrid,
) -> ModuleSpec:
    """Build the fixed Mistral-ViT encoder and projector submodules."""
    pp_group = getattr(pg_collection, "pp", None) if pg_collection is not None else None
    tp_group = getattr(pg_collection, "tp", None) if pg_collection is not None else None
    if pg_collection is None:
        tp_size = get_grid_dim_size(encoder_grid, "tp")
        pp_size = get_grid_dim_size(encoder_grid, "pp")
    else:
        assert (
            pp_group is not None and tp_group is not None
        ), "encoder pg_collection is missing the required pp/tp group"
        tp_size = get_pg_size(tp_group)
        pp_size = get_pg_size(pp_group)

    vision_config = mistral_vit_vision_config(args, tp_size, pp_size)
    vision_encoder_spec = mistral_vit_encoder_spec(args, vision_config, pg_collection)
    input_projections = []
    if not getattr(args, "mimo_run_input_projections_on_llm_ranks", False):
        input_projections.append(
            _mistral_vit_projection_spec(args, vision_config.hidden_size, tp_size, pg_collection)
        )
    return ModuleSpec(
        module=VisionModalitySubmodules,
        params={"pg_collection": pg_collection},
        submodules={
            "encoders": {MISTRAL_VIT_ENCODER_MODULE_NAME: vision_encoder_spec},
            "input_projections": input_projections,
        },
    )


def mistral_vit_language_input_projection_spec(
    args: argparse.Namespace,
    pg_collection: ProcessGroupCollection | None,
    language_grid: HyperCommGrid,
    language_spec: ModuleSpec,
) -> ModuleSpec | None:
    """Build the Mistral-ViT projection on the first language stage when requested."""
    if not getattr(args, "mimo_run_input_projections_on_llm_ranks", False):
        return None

    del language_grid
    vision_config = mistral_vit_vision_config(args, getattr(args, "mimo_encoder_tp", 1), 1)
    language_config = language_spec.params["config"]
    return _mistral_vit_projection_spec(
        args,
        vision_config.hidden_size,
        language_config.tensor_model_parallel_size,
        pg_collection,
        language_config,
    )


def mistral_vit_special_token_ids(args: argparse.Namespace) -> dict[str, int]:
    """Replace only native img tokens with Mistral-ViT embeddings."""
    return {MISTRAL_VIT_ENCODER_MODULE_NAME: args.image_token_id}


def _mistral_vit_bridge_recv_shape(batch: dict, hidden_size: int) -> tuple[int, int]:
    """Derive the local projected-vision receive shape from precomputed token indices."""
    token_indices = batch["modality_token_indices"][MISTRAL_VIT_ENCODER_MODULE_NAME]
    return token_indices.numel(), hidden_size


def build_nemotron_mistral_vit_communicator(
    args: argparse.Namespace, topology: "HeteroTopology"
) -> MultiModulePipelineCommunicator:
    """Wire the fixed Mistral-ViT encoder to the Nemotron language grid."""
    language_grid = topology.grids[MIMO_LANGUAGE_MODULE_KEY]
    language_config = language_model_spec(args, None, language_grid).params["config"]
    projections_on_llm_ranks = getattr(args, "mimo_run_input_projections_on_llm_ranks", False)
    bridge_hidden_size = int(language_config.hidden_size)
    if projections_on_llm_ranks:
        vision_config = mistral_vit_vision_config(args, getattr(args, "mimo_encoder_tp", 1), 1)
        bridge_hidden_size = int(vision_config.hidden_size)
    bridge_requires_backward = not (getattr(args, "freeze_vit", False) and projections_on_llm_ranks)
    return MultiModulePipelineCommunicator(
        topology.grids,
        {MISTRAL_VIT_ENCODER_MODULE_NAME: [MIMO_LANGUAGE_MODULE_KEY], MIMO_LANGUAGE_MODULE_KEY: []},
        language_config,
        dim_mapping={"s": 0, "h": 2, "b": 1},
        module_output_ndim={MISTRAL_VIT_ENCODER_MODULE_NAME: 2},
        bridge_comm_dtypes={MISTRAL_VIT_ENCODER_MODULE_NAME: language_config.params_dtype},
        bridge_recv_shape_fns=(
            {
                MISTRAL_VIT_ENCODER_MODULE_NAME: partial(
                    _mistral_vit_bridge_recv_shape, hidden_size=bridge_hidden_size
                )
            }
            if getattr(args, "mimo_bridge_skip_shape_exchange", False)
            else None
        ),
        bridge_requires_backward={MISTRAL_VIT_ENCODER_MODULE_NAME: bridge_requires_backward},
        timeout=topology.timeout,
    )


def nemotron_mistral_vit_provider() -> MimoProvider:
    """Return the Nemotron6-MoE and selectable Mistral-ViT provider."""
    return MimoProvider(
        encoder_module_names=(MISTRAL_VIT_ENCODER_MODULE_NAME,),
        language_spec=language_model_spec,
        encoder_specs={MISTRAL_VIT_ENCODER_MODULE_NAME: mistral_vit_submodules_spec},
        special_token_ids=mistral_vit_special_token_ids,
        build_communicator=build_nemotron_mistral_vit_communicator,
        language_input_projection_specs={
            MISTRAL_VIT_ENCODER_MODULE_NAME: mistral_vit_language_input_projection_spec
        },
    )
