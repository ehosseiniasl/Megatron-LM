# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain vision language model."""

# Capture entrypoint execution BEFORE heavy imports, after interpreter startup.
import time

_PROGRAM_START_TIME = time.time()

import os
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.multimodal_dataset import MockMultimodalDataset, MultimodalDatasetConfig
from megatron.core.enums import ModelType
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.models.multimodal import context_parallel
from megatron.core.models.multimodal.llava_model import DEFAULT_IMAGE_TOKEN_INDEX, LLaVAModel
from megatron.core.models.multimodal.llava_spec import (
    decoder_model_with_local_default_spec,
    decoder_model_with_transformer_engine_default_spec,
)
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_local_spec,
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules, import_module
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.datasets.vlm import add_vlm_data_args, build_vlm_dataset_provider


@dataclass(frozen=True)
class VisionModelSpec:
    model_type: str
    checkpoint_dir: str
    img_h: int = 224
    img_w: int = 224
    patch_dim: int = 16
    class_token_len: int = 8
    pixel_shuffle: bool = False
    spatial_merge_size: int = 1
    radio_force_eval_mode: bool = False
    dynamic_resolution: bool = False
    dynamic_resolution_max_patches: int = 0
    strict_checkpoint: bool = False
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_query_groups: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    kv_channels: Optional[int] = None
    normalization: Optional[str] = None
    activation_func: Optional[str] = None
    layernorm_epsilon: Optional[float] = None
    qk_layernorm: Optional[bool] = None
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    gated_linear_unit: bool = False
    rotary_interleaved: Optional[bool] = None

    @property
    def has_transformer_overrides(self):
        return self.num_layers is not None


_VISION_MODEL_SPECS = {
    "radio": VisionModelSpec(model_type="radio", checkpoint_dir="c_radio_vit_h"),
    "post-c-radio-omni": VisionModelSpec(
        model_type="radio",
        checkpoint_dir="post-c-radio-omni",
        img_h=512,
        img_w=512,
        class_token_len=10,
        pixel_shuffle=True,
        radio_force_eval_mode=True,
        dynamic_resolution=True,
        dynamic_resolution_max_patches=1024,
        strict_checkpoint=True,
        num_layers=32,
        hidden_size=1280,
        num_attention_heads=16,
        num_query_groups=16,
        ffn_hidden_size=5120,
        kv_channels=80,
        normalization="LayerNorm",
        activation_func="fast_gelu",
        layernorm_epsilon=1e-6,
        qk_layernorm=False,
    ),
    "mistral-vit": VisionModelSpec(
        model_type="mistral-vit",
        checkpoint_dir="vision_encoder_410M_mcore",
        img_h=1540,
        img_w=1540,
        patch_dim=14,
        class_token_len=0,
        spatial_merge_size=2,
        dynamic_resolution=True,
        dynamic_resolution_max_patches=12100,
        strict_checkpoint=True,
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=16,
        ffn_hidden_size=4096,
        kv_channels=64,
        normalization="RMSNorm",
        activation_func="silu",
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        add_qkv_bias=False,
        gated_linear_unit=True,
        rotary_interleaved=True,
    ),
    "mistral-vit-1p6b": VisionModelSpec(
        model_type="mistral-vit",
        checkpoint_dir="vision_encoder_mcore",
        img_h=1540,
        img_w=1540,
        patch_dim=14,
        class_token_len=0,
        spatial_merge_size=2,
        dynamic_resolution=True,
        dynamic_resolution_max_patches=12100,
        strict_checkpoint=True,
        num_layers=30,
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=16,
        ffn_hidden_size=6144,
        kv_channels=128,
        normalization="RMSNorm",
        activation_func="silu",
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        add_qkv_bias=False,
        gated_linear_unit=True,
        rotary_interleaved=True,
    ),
}

_VISION_TRANSFORMER_CONFIG_ATTRS = (
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "ffn_hidden_size",
    "kv_channels",
    "normalization",
    "layernorm_epsilon",
    "qk_layernorm",
)


def _language_model_args(args):
    transformer_impl = getattr(args, "vlm_language_transformer_impl", None)
    if transformer_impl is None:
        transformer_impl = args.transformer_impl
    # The inference-optimized language stack always needs its spec remapped, even when
    # --transformer-impl already selects it globally: the training specs build the dense
    # TopKRouter, whose [tokens, num_experts] routing map the inference MoE dispatchers
    # cannot consume.
    if transformer_impl == args.transformer_impl and transformer_impl != "inference_optimized":
        return args

    language_args = deepcopy(args)
    language_args.transformer_impl = transformer_impl
    if transformer_impl == "inference_optimized":
        spec = list(args.spec) if args.spec is not None else None
        inference_specs = {
            "hybrid_stack_spec": "hybrid_inference_stack_spec",
            "mamba_stack_spec": "hybrid_inference_stack_spec",
            "gated_delta_product_stack_spec": "gated_delta_product_inference_stack_spec",
            "gdp_stack_spec": "gated_delta_product_inference_stack_spec",
            # Already inference-optimized; keep the remapping idempotent.
            "hybrid_inference_stack_spec": "hybrid_inference_stack_spec",
            "gated_delta_product_inference_stack_spec": (
                "gated_delta_product_inference_stack_spec"
            ),
        }
        try:
            spec[-1] = inference_specs[spec[-1]]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"No inference-optimized VLM language spec for {args.spec!r}") from exc
        language_args.spec = spec
    return language_args


def _vision_transformer_impl(args):
    """Return the backend to use for the vision tower and the vision projection.

    Neither has an inference-optimized variant, and the inference-optimized backend is
    itself built on Transformer Engine modules, so ``inference_optimized`` maps to the
    Transformer Engine specs here. Only an explicit ``local`` request selects the
    non-TE specs, which support LayerNorm alone.
    """
    return "local" if args.transformer_impl == "local" else "transformer_engine"


def _vision_spec(args):
    return _VISION_MODEL_SPECS[args.vision_model_type]


def _vision_model_type(args):
    return _vision_spec(args).model_type


def _vision_pixel_shuffle(args):
    return _vision_spec(args).pixel_shuffle


def _vision_spatial_merge_size(args):
    return _vision_spec(args).spatial_merge_size


def _vision_force_eval_mode(args):
    return _vision_spec(args).radio_force_eval_mode


def _vision_dynamic_resolution(args):
    return _vision_spec(args).dynamic_resolution


def _vision_dynamic_resolution_max_patches(args):
    if args.vision_dynamic_resolution_max_patches is not None:
        if args.vision_dynamic_resolution_max_patches < 0:
            raise ValueError("--vision-dynamic-resolution-max-patches must be non-negative")
        return args.vision_dynamic_resolution_max_patches
    return _vision_spec(args).dynamic_resolution_max_patches


def _vision_strict_checkpoint(args):
    return _vision_spec(args).strict_checkpoint


def _vision_class_token_len(args):
    if getattr(args, "fp8", None):
        return 32 if getattr(args, "fp8_recipe", None) == "mxfp8" else 16
    return _vision_spec(args).class_token_len


def _resolve_vlm_encoder_args(args):
    spec = _vision_spec(args)
    if args.img_h == 224:
        args.img_h = spec.img_h
    if args.img_w == 224:
        args.img_w = spec.img_w
    if args.patch_dim == 16:
        args.patch_dim = spec.patch_dim
    if (
        args.load_vision_from is None
        and args.vision_encoder_checkpoint_root is not None
        and getattr(args, "load", None) is None
    ):
        args.load_vision_from = os.path.join(
            args.vision_encoder_checkpoint_root, spec.checkpoint_dir
        )
    return args


def _resolve_vlm_runtime_args(args):
    args = _resolve_vlm_encoder_args(args)
    checkpoint_dir = getattr(args, "save", None) or getattr(args, "load", None)
    if args.dataloader_type == "external" and checkpoint_dir is not None:
        args.dataloader_save = os.path.join(os.path.dirname(checkpoint_dir), "dataloader")
    return args


def _dense_vision_config_from_base(config):
    """Copy a decoder-derived config and remove language-only architecture settings."""
    from examples.mimo.model_providers.radio_encoder import _disable_gtp, _make_dense_non_hybrid

    config = deepcopy(config)
    _make_dense_non_hybrid(config)
    _disable_gtp(config)
    return config


def _mlp2_projection_module_spec(args, vision_model_type, vision_transformer_config):
    """Build the MIMO Mistral-ViT ``mlp2`` projector for the LLaVA runtime."""
    if not vision_model_type.startswith("mistral-vit"):
        raise ValueError("--vision-projection-type mlp2 is supported only with Mistral-ViT")

    from examples.mimo.model_providers.nemotron_mistral_vit import (
        mistral_vit_mlp2_projection_config,
        mistral_vit_projection_layer_spec,
    )
    from megatron.core.models.vision.multimodal_projector import MultimodalProjector

    projection_input_size = vision_transformer_config.hidden_size
    if _vision_pixel_shuffle(args):
        projection_input_size *= 4
    return ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": mistral_vit_mlp2_projection_config(
                args, args.tensor_model_parallel_size
            ),
            "submodules": mistral_vit_projection_layer_spec(),
            "projector_type": "mlp",
            "input_size": projection_input_size,
        },
    )


def _apply_vision_spec_to_config(config, args):
    spec = _vision_spec(args)
    if not spec.has_transformer_overrides:
        return config
    for attr in _VISION_TRANSFORMER_CONFIG_ATTRS:
        value = getattr(spec, attr)
        if value is not None:
            setattr(config, attr, value)
    if spec.activation_func == "fast_gelu":
        from megatron.core.activations import fast_gelu

        config.activation_func = fast_gelu
    elif spec.activation_func == "silu":
        config.activation_func = torch.nn.functional.silu
    elif spec.activation_func is not None:
        raise ValueError(f"Unknown vision activation function {spec.activation_func!r}")
    config.add_bias_linear = spec.add_bias_linear
    config.add_qkv_bias = spec.add_qkv_bias
    config.gated_linear_unit = spec.gated_linear_unit
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.layernorm_zero_centered_gamma = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.apply_rope_fusion = False
    if spec.rotary_interleaved is not None:
        config.rotary_interleaved = spec.rotary_interleaved
    return config


def _tp_slice(tensor, param_shape, tp_rank, tp_size):
    if tp_size == 1 or tensor.shape == param_shape:
        return tensor
    if tensor.shape[0] != param_shape[0]:
        start = tp_rank * param_shape[0]
        return tensor[start : start + param_shape[0], ...]
    if len(tensor.shape) > 1 and tensor.shape[1] != param_shape[1]:
        start = tp_rank * param_shape[1]
        return tensor[:, start : start + param_shape[1]]
    return tensor


def _remap_vision_checkpoint_key(key):
    key = re.sub(
        r"(decoder\.layers\.\d+)\.self_attention\.linear_qkv\.layer_norm_weight$",
        r"\1.input_layernorm.weight",
        key,
    )
    key = re.sub(
        r"(decoder\.layers\.\d+)\.self_attention\.linear_qkv\.layer_norm_bias$",
        r"\1.input_layernorm.bias",
        key,
    )
    key = re.sub(
        r"(decoder\.layers\.\d+)\.mlp\.linear_fc1\.layer_norm_weight$",
        r"\1.pre_mlp_layernorm.weight",
        key,
    )
    key = re.sub(
        r"(decoder\.layers\.\d+)\.mlp\.linear_fc1\.layer_norm_bias$",
        r"\1.pre_mlp_layernorm.bias",
        key,
    )
    return key


def _load_vision_from_checkpoint(model, ckpt_dir, strict=False):
    import torch.distributed.checkpoint as dist_checkpointing
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    from megatron.core import mpu

    tracker = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tracker):
        with open(tracker) as tracker_file:
            iteration = int(tracker_file.read().strip())
        ckpt_dir = os.path.join(ckpt_dir, f"iter_{iteration:07d}")

    print_rank_0(f"[load-vision-from] Loading ViT from {ckpt_dir}")
    reader = FileSystemReader(ckpt_dir)
    checkpoint_metadata = reader.read_metadata().state_dict_metadata
    prefix = "model.vision_model."
    load_state_dict = {
        key: torch.empty(metadata.size, dtype=metadata.properties.dtype)
        for key, metadata in checkpoint_metadata.items()
        if key.startswith(prefix) and isinstance(metadata, TensorStorageMetadata)
    }
    if not load_state_dict:
        raise RuntimeError(f"No '{prefix}*' tensors found in {ckpt_dir}")

    dist_checkpointing.load(load_state_dict, storage_reader=reader)

    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    model_tensors = dict(model.vision_model.named_parameters())
    model_tensors.update(dict(model.vision_model.named_buffers()))

    loaded = 0
    loaded_names = set()
    skipped_keys = []
    mismatched_keys = []
    for checkpoint_key, tensor in load_state_dict.items():
        name = checkpoint_key[len(prefix) :]
        param = model_tensors.get(name)
        if param is None:
            remapped_name = _remap_vision_checkpoint_key(name)
            param = model_tensors.get(remapped_name)
            if param is not None:
                name = remapped_name
        if param is None:
            skipped_keys.append(checkpoint_key)
            continue
        tensor = _tp_slice(tensor, param.shape, tp_rank, tp_size)
        if tensor.shape != param.shape:
            mismatched_keys.append((checkpoint_key, tuple(tensor.shape), tuple(param.shape)))
            continue
        param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))
        loaded += 1
        loaded_names.add(name)

    missing_params = [
        name for name, _ in model.vision_model.named_parameters() if name not in loaded_names
    ]
    if mismatched_keys or (strict and (skipped_keys or missing_params)):
        details = []
        if skipped_keys:
            details.append(f"skipped checkpoint keys: {skipped_keys[:8]}")
        if mismatched_keys:
            details.append(f"shape mismatches: {mismatched_keys[:8]}")
        if missing_params:
            details.append(f"missing model parameters: {missing_params[:8]}")
        raise RuntimeError(
            f"Vision checkpoint load from {ckpt_dir} was not strict: " + "; ".join(details)
        )

    print_rank_0(
        f"[load-vision-from] ViT loaded ({loaded}/{len(load_state_dict)} tensors"
        f"{f', {len(skipped_keys)} skipped' if skipped_keys else ''})"
    )


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}
    return loss, num_tokens, report


def model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    parallel_output=True,
    config=None,
    pg_collection=None,
) -> LLaVAModel:
    """Builds the model.

    Note: currently, only LLaVA model is supported. Follow-up changes will make this configurable.

    Args:
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism). Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        parallel_output (bool): Enable model parallel output.

    Returns:
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): A multimodal model
    """
    args = get_args()
    vision_model_type = _vision_model_type(args)

    assert args.ckpt_format in (
        "torch",
        "torch_dist",
    ), "Only ckpt-format torch or torch_dist is supported for VLM training currently."
    assert not (
        args.context_parallel_size > 1 and args.pipeline_model_parallel_size > 1
    ), "PP+CP is not yet supported by this script. \
    Current mock dataset does not support natively packed sequence dataset required for correct PP comm shapes."

    num_image_embeddings = get_num_image_embeddings(
        args.img_h,
        args.img_w,
        args.patch_dim,
        vision_model_type,
        args.disable_vision_class_token,
        class_token_len=_vision_class_token_len(args),
        pixel_shuffle=_vision_pixel_shuffle(args),
        use_tile_tags=False,
        spatial_merge_size=_vision_spatial_merge_size(args),
    )

    old_seq_length = args.seq_length
    # dataloader-seq-length is required to determine the length of text seq len
    if args.dataloader_seq_length is None:
        args.dataloader_seq_length = args.seq_length

    # decoder_seq_len denotes the language model sequence length.
    decoder_seq_len = args.dataloader_seq_length + num_image_embeddings

    # seq_length and encoder_seq_length denote the vision model sequence length. Override if the user provided something else.
    args.seq_length = args.encoder_seq_length = num_image_embeddings
    if torch.distributed.get_rank() == 0 and old_seq_length != args.seq_length:
        warnings.warn(
            f"Changed seq_length and encoder_seq_length (vision model sequence length) from {old_seq_length} to num_image_tokens ({num_image_embeddings})"
        )
    mp_padding_needed = context_parallel.get_padding(
        decoder_seq_len,
        args.context_parallel_size,
        args.tensor_model_parallel_size,
        args.sequence_parallel,
        args.decoder_tp_comm_overlap,
        args.decoder_seq_length,
    )
    args.decoder_seq_length = decoder_seq_len + mp_padding_needed

    args.max_position_embeddings = max(args.max_position_embeddings, args.decoder_seq_length)

    print_rank_0('building a multimodal model ...')
    language_args = _language_model_args(args)
    if config is None:
        language_transformer_config = core_transformer_config_from_args(language_args)
    else:
        language_transformer_config = config
    language_transformer_config.language_model_type = args.language_model_type
    if args.decoder_num_layers is not None:
        language_transformer_config.num_layers = args.decoder_num_layers
    else:
        language_transformer_config.num_layers = args.num_layers
    if args.decoder_tp_comm_overlap:
        assert (
            args.transformer_impl == "transformer_engine"
        ), "TransformerEngine is needed to support Decoder TP Comm overlap"
        language_transformer_config.tp_comm_overlap = args.decoder_tp_comm_overlap

    if language_args.spec is not None:
        language_transformer_layer_spec = import_module(language_args.spec)
    elif language_args.transformer_impl == "transformer_engine":
        language_transformer_layer_spec = decoder_model_with_transformer_engine_default_spec(
            args.num_experts, args.moe_grouped_gemm
        )
    else:  # transformer_impl == "local"
        language_transformer_layer_spec = decoder_model_with_local_default_spec(
            args.num_experts, args.moe_grouped_gemm
        )

    # Prepare mask type for any required padding to support CP/SP sequence sharding.
    if mp_padding_needed > 0:
        if (
            language_transformer_layer_spec.submodules.self_attention.params.get(
                'attn_mask_type', ''
            )
            == AttnMaskType.causal
        ):
            language_transformer_layer_spec.submodules.self_attention.params['attn_mask_type'] = (
                AttnMaskType.padding_causal
            )
        elif (
            language_transformer_layer_spec.submodules.self_attention.params.get(
                'attn_mask_type', ''
            )
            == AttnMaskType.no_mask
        ):
            language_transformer_layer_spec.submodules.self_attention.params['attn_mask_type'] = (
                AttnMaskType.padding
            )

    vision_transformer_impl = _vision_transformer_impl(args)
    if vision_transformer_impl == "transformer_engine":
        vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
    else:  # transformer_impl == "local"
        vision_transformer_layer_spec = get_vit_layer_with_local_spec()

    # TODO: Make these configurable via input .yaml config.
    vision_transformer_config = _dense_vision_config_from_base(language_transformer_config)
    vision_transformer_config.num_layers = args.encoder_num_layers
    vision_transformer_config = _apply_vision_spec_to_config(vision_transformer_config, args)
    vision_transformer_config.first_pipeline_num_layers = None
    vision_transformer_config.last_pipeline_num_layers = None
    vision_transformer_config.vision_model_type = vision_model_type
    vision_transformer_config.class_token_len = _vision_class_token_len(args)
    vision_transformer_config.context_parallel_size = 1  # Force CP=1 for Vision Transformer
    if vision_transformer_config.sequence_parallel:
        print_rank_0("> Disabling Sequence parallelism in Vision Transformer. Not yet supported")
        vision_transformer_config.sequence_parallel = False
    if vision_transformer_config.tp_comm_overlap:
        print_rank_0("> Disabling TP Comm overlap in Vision Transformer. Not yet supported")
        vision_transformer_config.tp_comm_overlap = False

    vision_projection_type = args.vision_projection_type
    vision_projection_config = _dense_vision_config_from_base(language_transformer_config)
    if args.vision_projection_ffn_hidden_size is not None:
        if args.vision_projection_ffn_hidden_size <= 0:
            raise ValueError("--vision-projection-ffn-hidden-size must be positive")
        vision_projection_config.ffn_hidden_size = args.vision_projection_ffn_hidden_size
    if vision_model_type.startswith("mistral-vit"):
        # Mistral checkpoints use a non-GLU two-layer projector; keep the width override.
        if args.vision_projection_ffn_hidden_size is None:
            vision_projection_config.ffn_hidden_size = vision_projection_config.hidden_size
        vision_projection_config.gated_linear_unit = False
        vision_projection_config.bias_activation_fusion = False
    vision_projection_config.context_parallel_size = 1  # Force CP=1 for Vision Projection
    if vision_projection_config.sequence_parallel:
        print_rank_0("> Disabling Sequence parallelism in Vision Projection. Not yet supported")
        vision_projection_config.sequence_parallel = False
    if vision_projection_config.tp_comm_overlap:
        print_rank_0("> Disabling TP Comm overlap in Vision Projection. Not yet supported")
        vision_projection_config.tp_comm_overlap = False

    # Vision Encoder and Projection should live on PP rank0
    vision_transformer_config.pipeline_model_parallel_size = 1
    vision_projection_config.pipeline_model_parallel_size = 1

    vision_projection_modules = deepcopy(
        get_submodules(
            get_mlp_module_spec(
                use_te=vision_transformer_impl == "transformer_engine",
                num_experts=None,
            )
        )
    )
    if vision_model_type.startswith("mistral-vit"):
        if vision_transformer_impl == "transformer_engine":
            vision_projection_modules.linear_fc1 = TEColumnParallelLinear
            vision_projection_modules.linear_fc2 = TERowParallelLinear
        else:
            vision_projection_modules.linear_fc1 = ColumnParallelLinear
            vision_projection_modules.linear_fc2 = RowParallelLinear
    if vision_projection_type == "affine":
        vision_projection_modules.linear_fc1 = ColumnParallelLinear

    vision_projection_module_spec = None
    if vision_projection_type == "mlp2":
        vision_projection_module_spec = _mlp2_projection_module_spec(
            args, vision_model_type, vision_transformer_config
        )

    language_max_sequence_length = args.decoder_seq_length
    if args.context_parallel_size > 1:
        if args.use_packed_sequence or mp_padding_needed > 0:
            # Use THD data format
            language_max_sequence_length = args.decoder_seq_length * args.micro_batch_size
    image_token_index = getattr(args, "image_token_id", None)
    if image_token_index is None:
        image_token_index = DEFAULT_IMAGE_TOKEN_INDEX

    model = LLaVAModel(
        language_transformer_config=language_transformer_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=language_max_sequence_length,
        vision_transformer_config=vision_transformer_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        drop_vision_class_token=args.disable_vision_class_token,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_modules,
        vision_projection_type=vision_projection_type,
        vision_projection_module_spec=vision_projection_module_spec,
        parallel_output=parallel_output,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        language_rope_scaling=args.use_rope_scaling,
        language_rope_scaling_factor=args.rope_scaling_factor,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
        add_encoder=parallel_state.is_pipeline_first_stage(),
        add_decoder=True,
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        hybrid_layer_pattern=args.hybrid_layer_pattern,
        logit_dtype=args.logit_dtype,
        pixel_shuffle=_vision_pixel_shuffle(args),
        spatial_merge_size=_vision_spatial_merge_size(args),
        radio_force_eval_mode=_vision_force_eval_mode(args),
        dynamic_resolution=_vision_dynamic_resolution(args),
        image_token_index=image_token_index,
    )

    model.freeze(
        freeze_language_model=args.freeze_LM,
        freeze_vision_model=args.freeze_ViT,
        freeze_vision_projection=False,
    )
    if args.load_vision_from is not None:
        _load_vision_from_checkpoint(
            model, args.load_vision_from, strict=_vision_strict_checkpoint(args)
        )

    return model


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train, validation, and test sets.

    Returns:
        train_ds, val_ds, test_ds (megatron.core.datasets.multimodal_dataset.MockMultimodalDataset): Train, validation, and test datasets, respectively.
    """
    args = get_args()

    config = MultimodalDatasetConfig(
        random_seed=args.seed,
        split=args.split,
        sequence_length=args.dataloader_seq_length,
        tokenizer=get_tokenizer(),
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        image_h=args.img_h,
        image_w=args.img_w,
        preprocess_func=_preprocess_data_for_llava,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
    )

    print_rank_0("> building train, validation, and test datasets for multimodal ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        MockMultimodalDataset,
        train_val_test_num_samples,
        lambda: parallel_state.get_tensor_model_parallel_rank() == 0,
        config,
    ).build()

    print_rank_0("> finished creating multimodal datasets ...")

    return train_ds, valid_ds, test_ds


def _preprocess_data_for_llava(data):
    """Preprocess data sample to the format expected by a LLaVA model.

    Note: This doesn't support all the different modes in the official LLaVA repo yet.

    Args:
        data (dict): Data sample with keys like 'image', 'tokens', etc.

    Returns:
        data (dict): Processed data sample suitable for the model.
    """
    # Prepend image token index to tokens.
    data["tokens"] = torch.cat(
        [
            DEFAULT_IMAGE_TOKEN_INDEX
            * torch.ones(1, dtype=data["tokens"].dtype, device=data["tokens"].device),
            data["tokens"],
        ]
    )
    # Prepend labels accordingly.
    data["labels"] = torch.cat([data["tokens"][1].unsqueeze(0), data["labels"]])
    # Zero loss mask for the image token index.
    data["loss_mask"] = torch.cat(
        [
            torch.zeros(1, dtype=data["loss_mask"].dtype, device=data["loss_mask"].device),
            data["loss_mask"],
        ]
    )
    # Add one more position id.
    data["position_ids"] = torch.cat(
        [data["position_ids"], data["position_ids"][-1].unsqueeze(0) + 1]
    )

    return data


def get_batch(data_iterator):
    """Generate a batch.

    Args:
        data_iterator: Iterable dataset.

    Returns:
        sample: A data sample with images, tokens, etc.
    """
    args = get_args()
    cp_size = args.context_parallel_size
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_i_keys = ["tokens", "position_ids", "labels"]
    if args.dataloader_type == "external":
        data_i_keys.append("num_image_tiles")
    data_i = tensor_parallel.broadcast_data(data_i_keys, data, torch.int64)
    data_p = None
    if args.dataloader_type == "external":
        data_p = tensor_parallel.broadcast_data(
            [
                "cu_seqlens",
                "max_seqlen",
                "imgs_sizes",
                "vision_cu_seqlens",
                "vision_max_seqlen",
            ],
            data,
            torch.int32,
        )
    data_f = tensor_parallel.broadcast_data(["image", "loss_mask"], data, torch.float32)

    packed_seq_params = None
    image_token_mask = None
    # Create batch with tokens and position_ids for CP sharding.
    tokens = data_i["tokens"].long()
    position_ids = data_i["position_ids"].long()
    labels = data_i["labels"].long()
    loss_mask = data_f["loss_mask"].float()
    images = data_f["image"].float()
    num_image_tiles = (
        data_i["num_image_tiles"].long() if args.dataloader_type == "external" else None
    )
    imgs_sizes = data_p["imgs_sizes"] if data_p is not None else None
    if imgs_sizes is not None and imgs_sizes.shape == torch.Size([1, 1]):
        imgs_sizes = None
    if (
        num_image_tiles is not None
        and num_image_tiles.shape[0] == 1
        and num_image_tiles[0].item() == 0
    ):
        num_image_tiles = torch.empty(0, dtype=torch.long, device=num_image_tiles.device)

    if data_p is not None and data_p["cu_seqlens"].shape != torch.Size([1, 1]):
        cu_seqlens = data_p["cu_seqlens"]
        max_seqlen = data_p["max_seqlen"]
        assert cu_seqlens.shape[0] == max_seqlen.shape[0] == 1, (
            "micro-batch-size must be 1 for packed VLM external batches"
        )
        cu_seqlens = cu_seqlens[0]
        max_seqlen = int(max_seqlen[0].max().item())
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

    vision_packed_seq_params = None
    if data_p is not None and data_p["vision_cu_seqlens"].shape != torch.Size([1, 1]):
        vision_cu_seqlens = data_p["vision_cu_seqlens"]
        if vision_cu_seqlens.dim() == 2:
            assert vision_cu_seqlens.shape[0] == 1, (
                "micro-batch-size must be 1 for packed VLM vision batches"
            )
            vision_cu_seqlens = vision_cu_seqlens[0]
        vision_max_seqlen = int(data_p["vision_max_seqlen"].max().item())
        vision_packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=vision_cu_seqlens,
            cu_seqlens_kv=vision_cu_seqlens,
            max_seqlen_q=vision_max_seqlen,
            max_seqlen_kv=vision_max_seqlen,
        )

    if (cp_size > 1 or args.sequence_parallel) and packed_seq_params is None:
        # Calculate the number of image embedding tokens will be added to text tokens
        num_image_embeddings_per_tile = get_num_image_embeddings(
            args.img_h,
            args.img_w,
            args.patch_dim,
            _vision_model_type(args),
            args.disable_vision_class_token,
            class_token_len=_vision_class_token_len(args),
            pixel_shuffle=_vision_pixel_shuffle(args),
            use_tile_tags=False,
            spatial_merge_size=_vision_spatial_merge_size(args),
        )
        # Pad to make sure the text sequence can be sharded equally by CP chunks.
        image_token_mask = tokens == DEFAULT_IMAGE_TOKEN_INDEX
        num_images_per_sample = torch.sum(image_token_mask, dim=-1)
        img_seq_len = (
            num_image_embeddings_per_tile * num_images_per_sample - num_images_per_sample
        ).max()
        mp_padding_needed_for_text = context_parallel.get_padding(
            tokens.shape[1] + img_seq_len,
            args.context_parallel_size,
            args.tensor_model_parallel_size,
            args.sequence_parallel,
            args.decoder_tp_comm_overlap,
            args.decoder_seq_length,
        )
        if mp_padding_needed_for_text > 0:
            tokens, position_ids, labels, loss_mask = [
                torch.nn.functional.pad(item, (0, mp_padding_needed_for_text))
                for item in (tokens, position_ids, labels, loss_mask)
            ]
        packed_seq_params = context_parallel.get_packed_seq_params(
            tokens, img_seq_len, mp_padding_needed_for_text, cp_size, args.use_packed_sequence
        )

        if packed_seq_params.qkv_format == 'thd':
            # Reshape from [B,S] to [T,1]
            tokens = tokens.contiguous().view(tokens.shape[0] * tokens.shape[1]).unsqueeze(0)
            position_ids = (
                position_ids.contiguous()
                .view(position_ids.shape[0] * position_ids.shape[1])
                .unsqueeze(0)
            )
            labels = labels.view(labels.shape[0] * labels.shape[1]).unsqueeze(0)
            loss_mask = loss_mask.view(loss_mask.shape[0] * loss_mask.shape[1]).unsqueeze(0)

    attention_mask = None  # Use the attention mask type defined in layer spec. Typically no mask for the vision model and causal mask for the vision model.

    return (
        tokens,
        position_ids,
        labels,
        images,
        loss_mask,
        attention_mask,
        packed_seq_params,
        vision_packed_seq_params,
        num_image_tiles,
        imgs_sizes,
    )


def forward_step(data_iterator, model: LLaVAModel):
    """Forward training step.

    Args:
        data_iterator: Iterable dataset.
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): Multimodal model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    (
        tokens,
        position_ids,
        labels,
        images,
        loss_mask,
        attention_mask,
        packed_seq_params,
        vision_packed_seq_params,
        num_image_tiles,
        imgs_sizes,
    ) = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor, loss_mask = model(
        images,
        tokens,
        position_ids,
        attention_mask,
        labels,
        loss_mask,
        packed_seq_params=packed_seq_params,
        vision_packed_seq_params=vision_packed_seq_params,
        num_image_tiles=num_image_tiles,
        imgs_sizes=imgs_sizes,
    )

    return output_tensor, partial(loss_func, loss_mask)


def add_vlm_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='vision language model specific arguments')
    group.add_argument(
        '--freeze-LM', action='store_true', default=False, help="Freeze language model weights"
    )
    group.add_argument(
        '--freeze-ViT', action='store_true', default=False, help="Freeze vision model (ViT) weights"
    )
    group.add_argument(
        "--disable-vision-class-token",
        action="store_true",
        default=False,
        help="Drop vision model class token",
    )
    group.add_argument(
        "--language-model-type",
        type=str,
        required=True,
        help="Language decoder family. Hybrid values such as nemotron6-moe build HybridModel.",
    )
    group.add_argument(
        "--vision-model-type",
        choices=sorted(_VISION_MODEL_SPECS),
        default="radio",
        help="Vision model type.",
    )
    group.add_argument(
        "--vision-dynamic-resolution-max-patches",
        type=int,
        default=None,
        help=(
            "Override the selected vision preset's dynamic-resolution max patch budget. "
            "Defaults to the preset value; 0 disables the cap."
        ),
    )
    group.add_argument(
        "--vision-projection-type",
        choices=["mlp", "mlp2", "affine"],
        default="mlp",
        help="Projection module from frozen vision features to language hidden size.",
    )
    group.add_argument(
        "--vision-projection-ffn-hidden-size",
        type=int,
        default=None,
        help=(
            "Intermediate hidden width for the MLP vision projector. Defaults to the "
            "language model FFN hidden size, except mistral-vit defaults to the "
            "language hidden size."
        ),
    )
    group.add_argument(
        "--dataloader-seq-length",
        type=int,
        help="Make dataloader to produce sequences of specific length.",
    )
    group.add_argument(
        "--decoder-tp-comm-overlap",
        action="store_true",
        default=False,
        help="Enables the overlap of "
        "Tensor parallel communication and GEMM kernels in Decoder only. "
        "Please provide decoder-seq-length when using this feature.",
    )
    group.add_argument(
        "--use-packed-sequence", action="store_true", default=False, help="Use packed sequence"
    )
    group.add_argument(
        "--load-vision-from",
        type=str,
        default=None,
        help="Load only vision model weights from this checkpoint directory.",
    )
    group.add_argument(
        "--vision-encoder-checkpoint-root",
        type=str,
        default=None,
        help="Root directory for built-in vision encoder checkpoints.",
    )
    group.add_argument(
        "--tokenizer-prompt-format",
        type=str,
        default="nemotron-4",
        help="Prompt format used by MultimodalTokenizer.",
    )
    group.add_argument(
        "--image-tag-type",
        type=str,
        choices=["nvlm", "internvl", ""],
        default="",
        help="Image tag style used by MultimodalTokenizer.",
    )
    parser = add_vlm_data_args(parser)
    return parser


def llava_embedding_ranks(pp_ranks):
    """LLaVA's embedding ranks consist of the first and last ranks of the pipeline.
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    first_rank = pp_ranks[0]
    last_rank = pp_ranks[-1]

    if len(pp_ranks) == 1:
        return [first_rank]
    else:
        return [first_rank, last_rank]


def llava_position_embedding_ranks(pp_ranks):
    """LLaVA's positional embeddings are on the first rank stage
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    return [pp_ranks[0]]


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    train_valid_test_datasets_provider.is_distributed = True

    args = parse_and_validate_args(
        extra_args_provider=add_vlm_extra_args,
        args_defaults={
            'tokenizer_type': 'GPT2BPETokenizer',
            'special_tokens': [],
            'force_system_message': False,
        },
    )
    args = _resolve_vlm_runtime_args(args)
    dataset_provider = build_vlm_dataset_provider(
        args,
        train_valid_test_datasets_provider,
        vision_model_type=_vision_model_type(args),
        class_token_len=_vision_class_token_len(args),
        pixel_shuffle=_vision_pixel_shuffle(args),
        spatial_merge_size=_vision_spatial_merge_size(args),
        dynamic_resolution=_vision_dynamic_resolution(args),
        dynamic_resolution_max_patches=_vision_dynamic_resolution_max_patches(args),
    )
    full_config = pretrain_cfg_container_from_args(args)
    pretrain(
        full_config,
        dataset_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        model_provider,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
    )
