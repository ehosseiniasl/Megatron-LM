# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for the Nemotron6-MoE and Mistral-ViT provider."""

import argparse
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from examples.mimo.model_providers import resolve_provider
from examples.mimo.model_providers.nemotron_mistral_vit import (
    MISTRAL_VIT_ENCODER_MODULE_NAME,
    NEMOTRON_MISTRAL_VIT_MODEL_PROVIDER,
    MimoMistralViTModel,
    build_nemotron_mistral_vit_communicator,
    mistral_vit_language_input_projection_spec,
)
from examples.mimo.model_providers.nemotron_moe_vlm import (
    NEMOTRON_MODEL_PROVIDER,
    add_model_provider_args,
)
from examples.mimo.model_providers.radio_encoder import RADIO_ENCODER_MODULE_NAME
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_model import ViTModel
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import set_attention_backend
from tests.unit_tests.test_utilities import clear_nvte_env_vars

_NVTE_BACKEND_ENV_VARS = ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")
_NVTE_FLASH_VERSION_ENV_VARS = tuple(f"NVTE_FLASH_ATTN_V{version}" for version in (2, 3, 4))
_BACKEND_ENV_VALUES = {
    AttnBackend.local: ("0", "0", "0"),
    AttnBackend.flash: ("1", "0", "0"),
    AttnBackend.fused: ("0", "1", "0"),
    AttnBackend.unfused: ("0", "0", "1"),
    AttnBackend.auto: ("1", "1", "1"),
}


@pytest.fixture(autouse=True)
def _restore_nvte_attention_env():
    with patch.dict(os.environ):
        clear_nvte_env_vars()
        yield


def _transformer_config(**overrides):
    return TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=4, **overrides)


def _provider_args(**overrides):
    values = {
        "bf16": True,
        "fp16": False,
        "hidden_size": 2688,
        "image_token_id": 10,
        "img_h": 1540,
        "img_w": 1540,
        "mimo_bridge_skip_shape_exchange": False,
        "mimo_encoder_tp": 1,
        "mimo_run_input_projections_on_llm_ranks": False,
        "mimo_vision_encoder_attention_backend": None,
        "mimo_vision_encoder_flash_attention_version": None,
        "mistral_vit_model_size": "410m",
        "params_dtype": torch.bfloat16,
        "vision_projection_type": "mlp",
        "freeze_vit": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _provider_base_config(**overrides):
    values = {
        "activation_func": torch.nn.functional.gelu,
        "attention_backend": AttnBackend.unfused,
        "flash_attention_version": 2,
        "expert_tensor_parallel_size": None,
        "fp32_residual_connection": False,
        "params_dtype": torch.bfloat16,
        "pipeline_dtype": torch.bfloat16,
        # Read by _make_dense_non_hybrid when it strips residual-stream recompute.
        "recompute_modules": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _assert_gtp_disabled(config, tp_size=1):
    assert config.context_parallel_size == 1
    assert config.tensor_parallel_num_sequence_shards is None
    assert config.gtp_weight_remat_size == 1
    assert config.tensor_parallel_num_weight_shards == tp_size
    assert config.expert_gtp_weight_remat_size == 1
    assert config.expert_tensor_parallel_num_weight_shards == tp_size


@pytest.mark.parametrize("backend", [None, *AttnBackend])
def test_vision_encoder_attention_backend_arg_uses_full_enum(backend):
    parser = argparse.ArgumentParser()
    add_model_provider_args(parser)
    cli_args = [] if backend is None else ["--mimo-vision-encoder-attention-backend", backend.name]
    args = parser.parse_args(cli_args)

    assert args.mimo_vision_encoder_attention_backend is backend


@pytest.mark.parametrize("backend", [*AttnBackend, "unfused"])
def test_set_attention_backend_accepts_all_enums_and_string(backend):
    set_attention_backend(_transformer_config(attention_backend=backend))

    expected_backend = AttnBackend[backend] if isinstance(backend, str) else backend
    assert (
        tuple(os.environ[name] for name in _NVTE_BACKEND_ENV_VARS)
        == _BACKEND_ENV_VALUES[expected_backend]
    )
    assert all(name not in os.environ for name in _NVTE_FLASH_VERSION_ENV_VARS)


def test_set_attention_backend_rejects_process_wide_conflicts():
    set_attention_backend(
        _transformer_config(attention_backend=AttnBackend.flash, flash_attention_version=3)
    )
    assert tuple(os.environ[name] for name in _NVTE_FLASH_VERSION_ENV_VARS) == ("0", "1", "0")

    conflicts = (
        _transformer_config(attention_backend=AttnBackend.fused, flash_attention_version=3),
        _transformer_config(attention_backend=AttnBackend.flash, flash_attention_version=2),
    )
    for config in conflicts:
        with pytest.raises(AssertionError, match="process-wide"):
            set_attention_backend(config)


@pytest.mark.parametrize(
    ("encoder_backend", "encoder_flash_version"),
    ((None, None), (AttnBackend.flash, 4), (AttnBackend.fused, None)),
)
def test_fixed_410m_vision_tower_and_projector(monkeypatch, encoder_backend, encoder_flash_version):
    from examples.mimo.model_providers import nemotron_mistral_vit

    monkeypatch.setattr(
        nemotron_mistral_vit,
        "_base_config",
        lambda _args: _provider_base_config(
            context_parallel_size=4, tensor_parallel_num_sequence_shards=2
        ),
    )
    monkeypatch.setattr(nemotron_mistral_vit, "mistral_vit_projection_layer_spec", object)

    args = _provider_args(
        mimo_vision_encoder_attention_backend=encoder_backend,
        mimo_vision_encoder_flash_attention_version=encoder_flash_version,
    )
    provider = resolve_provider(SimpleNamespace(model_provider=NEMOTRON_MISTRAL_VIT_MODEL_PROVIDER))
    spec = provider.encoder_specs[MISTRAL_VIT_ENCODER_MODULE_NAME](args, None, None)
    encoder = spec.submodules["encoders"][MISTRAL_VIT_ENCODER_MODULE_NAME]
    vision_config = encoder.params["transformer_config"]
    projection = spec.submodules["input_projections"][0]

    assert provider.encoder_module_names == (MISTRAL_VIT_ENCODER_MODULE_NAME,)
    assert provider.special_token_ids(args) == {MISTRAL_VIT_ENCODER_MODULE_NAME: 10}
    assert encoder.module is MimoMistralViTModel
    assert issubclass(encoder.module, ViTModel)
    assert (
        vision_config.num_layers,
        vision_config.hidden_size,
        vision_config.ffn_hidden_size,
        vision_config.num_attention_heads,
        vision_config.num_query_groups,
        vision_config.kv_channels,
    ) == (24, 1024, 4096, 16, 16, 64)
    assert vision_config.normalization == "RMSNorm"
    assert vision_config.activation_func is torch.nn.functional.silu
    assert vision_config.gated_linear_unit is True
    assert vision_config.add_bias_linear is False
    assert vision_config.add_qkv_bias is False
    assert vision_config.rotary_interleaved is True
    assert vision_config.hidden_dropout == 0.0
    assert vision_config.attention_dropout == 0.0
    assert vision_config.attention_backend is (encoder_backend or AttnBackend.unfused)
    assert vision_config.flash_attention_version == (encoder_flash_version or 2)
    assert encoder.params["patch_dim"] == 14
    assert encoder.params["add_class_token"] is False
    assert encoder.params["use_merger"] is True
    assert encoder.params["spatial_merge_size"] == 2
    assert projection.params["projector_type"] == "mlp"
    assert projection.params["input_size"] == 1024
    assert projection.params["config"].ffn_hidden_size == 1024
    assert projection.params["config"].hidden_size == 2688
    _assert_gtp_disabled(vision_config)
    _assert_gtp_disabled(projection.params["config"])


def test_affine_projector_honors_projection_type(monkeypatch):
    from examples.mimo.model_providers import nemotron_mistral_vit

    affine_submodules = object()
    monkeypatch.setattr(nemotron_mistral_vit, "_base_config", lambda _args: _provider_base_config())
    monkeypatch.setattr(
        nemotron_mistral_vit,
        "nemotron_projection_layer_spec",
        lambda: SimpleNamespace(submodules=affine_submodules),
    )

    args = _provider_args(vision_projection_type="affine")
    provider = resolve_provider(SimpleNamespace(model_provider=NEMOTRON_MISTRAL_VIT_MODEL_PROVIDER))
    spec = provider.encoder_specs[MISTRAL_VIT_ENCODER_MODULE_NAME](args, None, None)
    projection = spec.submodules["input_projections"][0]

    assert projection.params["projector_type"] == "affine"
    assert projection.params["submodules"] is affine_submodules
    _assert_gtp_disabled(projection.params["config"])


def test_mlp2_projector_uses_decoder_width_fast_gelu_and_no_bias(monkeypatch):
    from examples.mimo.model_providers import nemotron_mistral_vit

    mlp_submodules = object()
    decoder_activation = torch.nn.functional.gelu
    monkeypatch.setattr(
        nemotron_mistral_vit,
        "_base_config",
        lambda _args: _provider_base_config(activation_func=decoder_activation),
    )
    monkeypatch.setattr(
        nemotron_mistral_vit, "mistral_vit_projection_layer_spec", lambda: mlp_submodules
    )

    args = _provider_args(mistral_vit_model_size="1.6b", vision_projection_type="mlp2")
    provider = resolve_provider(SimpleNamespace(model_provider=NEMOTRON_MISTRAL_VIT_MODEL_PROVIDER))
    spec = provider.encoder_specs[MISTRAL_VIT_ENCODER_MODULE_NAME](args, None, None)
    encoder = spec.submodules["encoders"][MISTRAL_VIT_ENCODER_MODULE_NAME]
    projection = spec.submodules["input_projections"][0]
    config = projection.params["config"]

    assert encoder.params["transformer_config"].hidden_size == 2048
    assert projection.module is MultimodalProjector
    assert projection.params["projector_type"] == "mlp"
    assert projection.params["submodules"] is mlp_submodules
    assert projection.params["input_size"] == 2048
    assert config.hidden_size == args.hidden_size
    assert config.ffn_hidden_size == args.hidden_size
    assert config.activation_func is nemotron_mistral_vit._uncompiled_fast_gelu
    activation_input = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0])
    expected = (
        0.5
        * activation_input
        * (
            1.0
            + torch.tanh(
                activation_input
                * 0.7978845608
                * (1.0 + 0.044715 * activation_input * activation_input)
            )
        )
    )
    torch.testing.assert_close(config.activation_func(activation_input), expected)
    assert config.activation_func_tanh_clamp_scale is None
    assert config.gated_linear_unit is False
    assert config.add_bias_linear is False
    _assert_gtp_disabled(encoder.params["transformer_config"])
    _assert_gtp_disabled(config)


def test_mimo_mistral_vit_real_constructor_configures_backend(monkeypatch):
    from megatron.core.models.vision import vit_model

    class StubTransformerBlock(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()

    monkeypatch.setattr(vit_model, "TransformerBlock", StubTransformerBlock)
    tp_group = object()
    model = MimoMistralViTModel(
        transformer_config=_transformer_config(attention_backend=AttnBackend.fused),
        transformer_layer_spec=object(),
        ln_pre=False,
        pos_emb_type="none",
        pg_collection=SimpleNamespace(tp=tp_group),
    )

    assert model.tp_group is tp_group
    assert tuple(os.environ[name] for name in _NVTE_BACKEND_ENV_VARS) == ("0", "1", "0")


def test_language_rank_placement_uses_language_parallelism(monkeypatch):
    from examples.mimo.model_providers import nemotron_mistral_vit

    projection_submodules = object()
    monkeypatch.setattr(nemotron_mistral_vit, "_base_config", lambda _args: _provider_base_config())
    monkeypatch.setattr(
        nemotron_mistral_vit, "mistral_vit_projection_layer_spec", lambda: projection_submodules
    )

    args = _provider_args(
        mimo_run_input_projections_on_llm_ranks=True,
        mistral_vit_model_size="1.6b",
        vision_projection_type="mlp2",
    )
    provider = resolve_provider(SimpleNamespace(model_provider=NEMOTRON_MISTRAL_VIT_MODEL_PROVIDER))
    encoder_spec = provider.encoder_specs[MISTRAL_VIT_ENCODER_MODULE_NAME](args, None, None)

    assert encoder_spec.submodules["input_projections"] == []

    language_config = _provider_base_config(
        tensor_model_parallel_size=2,
        tensor_parallel_num_weight_shards=64,
        gtp_weight_remat_size=32,
        expert_tensor_parallel_num_weight_shards=8,
        expert_gtp_weight_remat_size=4,
        context_parallel_size=8,
        tensor_parallel_num_sequence_shards=8,
        fp8_param=True,
    )
    language_spec = SimpleNamespace(params={"config": language_config})
    pg_collection = SimpleNamespace(tp=object())
    projection = mistral_vit_language_input_projection_spec(
        args, pg_collection, None, language_spec
    )
    config = projection.params["config"]

    assert projection.params["input_size"] == 2048
    assert projection.params["pg_collection"] is pg_collection
    assert projection.params["submodules"] is projection_submodules
    assert config.tensor_model_parallel_size == 2
    assert config.tensor_parallel_num_weight_shards == 64
    assert config.gtp_weight_remat_size == 32
    assert config.context_parallel_size == 1
    assert config.tensor_parallel_num_sequence_shards is None
    assert config.fp8_param is True
    assert config.activation_func is nemotron_mistral_vit._uncompiled_fast_gelu
    assert provider.language_input_projection_specs[MISTRAL_VIT_ENCODER_MODULE_NAME] is (
        mistral_vit_language_input_projection_spec
    )


@pytest.mark.parametrize(
    "enabled,language_projection,freeze_vit,expected_width,expected_backward",
    [
        (False, False, False, 2688, True),
        (False, True, True, 2048, False),
        (True, False, True, 2688, True),
        (True, True, False, 2048, True),
        (True, True, True, 2048, False),
    ],
    ids=[
        "legacy",
        "legacy_frozen_source",
        "encoder_projection",
        "language_projection",
        "receiver_derived_frozen_source",
    ],
)
def test_build_communicator_wires_bridge_contract(
    monkeypatch, enabled, language_projection, freeze_vit, expected_width, expected_backward
):
    import examples.mimo.model_providers.nemotron_mistral_vit as provider

    captured = {}
    communicator = object()
    language_grid = object()
    topology = SimpleNamespace(
        grids={MISTRAL_VIT_ENCODER_MODULE_NAME: object(), MIMO_LANGUAGE_MODULE_KEY: language_grid},
        timeout=None,
    )
    args = _provider_args(
        mimo_bridge_skip_shape_exchange=enabled,
        mimo_run_input_projections_on_llm_ranks=language_projection,
        mistral_vit_model_size="1.6b",
        freeze_vit=freeze_vit,
    )
    language_config = SimpleNamespace(
        hidden_size=2688, params_dtype=torch.bfloat16, pipeline_dtype=torch.float32
    )

    monkeypatch.setattr(
        provider,
        "language_model_spec",
        lambda args, pg_collection, grid: SimpleNamespace(params={"config": language_config}),
    )
    monkeypatch.setattr(
        provider,
        "mistral_vit_vision_config",
        lambda _args, _tensor_parallel_size, _pipeline_parallel_size: SimpleNamespace(
            hidden_size=2048
        ),
    )

    def capture_communicator(*args, **kwargs):
        captured.update(kwargs)
        return communicator

    monkeypatch.setattr(provider, "MultiModulePipelineCommunicator", capture_communicator)

    assert build_nemotron_mistral_vit_communicator(args, topology) is communicator
    assert captured["bridge_comm_dtypes"] == {MISTRAL_VIT_ENCODER_MODULE_NAME: torch.bfloat16}
    assert captured["bridge_requires_backward"] == {
        MISTRAL_VIT_ENCODER_MODULE_NAME: expected_backward
    }
    shape_fns = captured["bridge_recv_shape_fns"]
    if enabled:
        batch = {
            "modality_token_indices": {MISTRAL_VIT_ENCODER_MODULE_NAME: torch.tensor([2, 5, 9])}
        }
        assert shape_fns[MISTRAL_VIT_ENCODER_MODULE_NAME](batch) == (3, expected_width)
    else:
        assert shape_fns is None


def test_radio_provider_contract_remains_unchanged():
    provider = resolve_provider(SimpleNamespace(model_provider=NEMOTRON_MODEL_PROVIDER))

    assert provider.encoder_module_names == (RADIO_ENCODER_MODULE_NAME,)
