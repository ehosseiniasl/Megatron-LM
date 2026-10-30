# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import pretrain_vlm
from megatron.core.models.vision.multimodal_projector import MultimodalProjector


def test_vlm_args_accept_mlp2_projection():
    parser = argparse.ArgumentParser()
    pretrain_vlm.add_vlm_extra_args(parser)

    args = parser.parse_args(
        ["--language-model-type", "nemotron6-moe", "--vision-projection-type", "mlp2"]
    )

    assert args.vision_projection_type == "mlp2"


def test_vlm_dense_configs_match_mimo_training_isolation():
    config = SimpleNamespace(
        activation_func_tanh_clamp_scale=16.0,
        fp32_residual_connection=True,
        params_dtype="bf16",
        pipeline_dtype="fp32",
        tensor_model_parallel_size=2,
        tensor_parallel_num_weight_shards=8,
        gtp_weight_remat_size=4,
        expert_tensor_parallel_size=1,
        expert_tensor_parallel_num_weight_shards=4,
        expert_gtp_weight_remat_size=4,
        mtp_num_layers=2,
        mtp_use_repeated_layer=True,
        num_moe_experts=256,
        moe_ffn_hidden_size=3712,
        moe_shared_expert_intermediate_size=3712,
        moe_grouped_gemm=True,
        moe_router_fusion=True,
        moe_permute_fusion=True,
        moe_shared_expert_overlap=True,
        moe_shortcut_connection=True,
        moe_shortcut_parallel=True,
        moe_shortcut_post_norm=True,
        is_hybrid_model=True,
        use_fused_weighted_squared_relu=True,
        wide_residual=object(),
        residual_stream_recompute_num_layers=4,
        recompute_modules=["moe_act", "residual_stream", "shortcut_pre_mlp_layernorm"],
        offload_modules=["mlp", "shortcut_post_norm"],
    )

    vision_config = pretrain_vlm._dense_vision_config_from_base(config)

    assert vision_config is not config
    assert config.wide_residual is not None
    assert vision_config.activation_func_tanh_clamp_scale is None
    assert vision_config.fp32_residual_connection is False
    assert vision_config.pipeline_dtype == "bf16"
    assert vision_config.tensor_parallel_num_weight_shards == 2
    assert vision_config.gtp_weight_remat_size == 1
    assert vision_config.expert_tensor_parallel_num_weight_shards == 1
    assert vision_config.expert_gtp_weight_remat_size == 1
    assert vision_config.mtp_num_layers is None
    assert vision_config.mtp_use_repeated_layer is False
    assert vision_config.num_moe_experts is None
    assert vision_config.moe_shortcut_connection is False
    assert vision_config.moe_shortcut_parallel is False
    assert vision_config.moe_shortcut_post_norm is False
    assert vision_config.is_hybrid_model is False
    assert vision_config.wide_residual is None
    assert vision_config.residual_stream_recompute_num_layers is None
    assert vision_config.recompute_modules == ["moe_act"]
    assert vision_config.offload_modules == ["mlp"]


def test_mistral_vit_mlp2_uses_mimo_projection(monkeypatch):
    from examples.mimo.model_providers import nemotron_mistral_vit

    projection_config = object()
    projection_submodules = object()
    config_builder = MagicMock(return_value=projection_config)
    monkeypatch.setattr(nemotron_mistral_vit, "mistral_vit_mlp2_projection_config", config_builder)
    monkeypatch.setattr(
        nemotron_mistral_vit, "mistral_vit_projection_layer_spec", lambda: projection_submodules
    )

    args = SimpleNamespace(vision_model_type="mistral-vit-1p6b", tensor_model_parallel_size=2)
    spec = pretrain_vlm._mlp2_projection_module_spec(
        args,
        vision_model_type="mistral-vit",
        vision_transformer_config=SimpleNamespace(hidden_size=2048),
    )

    assert spec.module is MultimodalProjector
    config_builder.assert_called_once_with(args, 2)
    assert spec.params["config"] is projection_config
    assert spec.params["submodules"] is projection_submodules
    assert spec.params["projector_type"] == "mlp"
    assert spec.params["input_size"] == 2048


def test_mistral_vit_1p6b_preset_matches_mimo_provider():
    spec = pretrain_vlm._VISION_MODEL_SPECS["mistral-vit-1p6b"]

    assert spec.model_type == "mistral-vit"
    assert spec.num_layers == 30
    assert spec.hidden_size == 2048
    assert spec.ffn_hidden_size == 6144
    assert spec.kv_channels == 128
    assert spec.spatial_merge_size == 2


def _vlm_args(transformer_impl, vlm_language_transformer_impl, spec):
    return SimpleNamespace(
        transformer_impl=transformer_impl,
        vlm_language_transformer_impl=vlm_language_transformer_impl,
        spec=spec,
    )


def test_language_spec_is_inference_optimized_when_both_impls_agree():
    # A global --transformer-impl inference_optimized must still swap the training stack
    # spec: the training specs build the dense TopKRouter, whose [tokens, num_experts]
    # routing map the inference MoE dispatchers cannot consume.
    base_spec = ["megatron.core.models.hybrid.hybrid_layer_specs", "hybrid_stack_spec"]
    args = _vlm_args("inference_optimized", "inference_optimized", base_spec)

    language_args = pretrain_vlm._language_model_args(args)

    assert language_args is not args
    assert language_args.transformer_impl == "inference_optimized"
    assert language_args.spec[-1] == "hybrid_inference_stack_spec"
    assert args.spec == base_spec  # caller's args left untouched


def test_language_spec_remap_is_idempotent():
    args = _vlm_args(
        "inference_optimized",
        None,
        ["megatron.core.models.hybrid.hybrid_layer_specs", "hybrid_inference_stack_spec"],
    )

    language_args = pretrain_vlm._language_model_args(args)

    assert language_args.spec[-1] == "hybrid_inference_stack_spec"


def test_language_args_unchanged_without_inference_optimized():
    spec = ["megatron.core.models.hybrid.hybrid_layer_specs", "hybrid_stack_spec"]
    args = _vlm_args("transformer_engine", "transformer_engine", spec)

    assert pretrain_vlm._language_model_args(args) is args

    args = _vlm_args("transformer_engine", None, spec)

    assert pretrain_vlm._language_model_args(args) is args


def test_language_spec_remap_rejects_unknown_spec():
    args = _vlm_args("inference_optimized", None, ["some.module", "custom_stack_spec"])

    with pytest.raises(ValueError, match="No inference-optimized VLM language spec"):
        pretrain_vlm._language_model_args(args)


def test_inference_optimized_vision_tower_uses_transformer_engine():
    # The vision tower has no inference-optimized spec, and its local spec only supports
    # LayerNorm, so RMSNorm vision encoders must fall back to the Transformer Engine spec.
    for transformer_impl in ("transformer_engine", "inference_optimized"):
        args = SimpleNamespace(transformer_impl=transformer_impl)
        assert pretrain_vlm._vision_transformer_impl(args) == "transformer_engine"

    assert pretrain_vlm._vision_transformer_impl(SimpleNamespace(transformer_impl="local")) == (
        "local"
    )
