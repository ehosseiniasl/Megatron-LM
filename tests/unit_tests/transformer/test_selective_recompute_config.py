# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig


def _make_transformer_config(**kwargs) -> TransformerConfig:
    return TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4, **kwargs)


def _make_expert_recompute_config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_moe_experts=2,
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=True,
        fine_grained_activation_offloading=True,
        offload_modules=["fused_group_mlp"],
        activation_offload_fraction=0.75,
        recompute_granularity="selective",
        recompute_modules=["moe_act", "moe_experts"],
    )
    kwargs.update(overrides)
    return _make_transformer_config(**kwargs)


def test_resident_expert_recompute_accepts_shortcut_moe():
    config = _make_expert_recompute_config(moe_shortcut_connection=True)
    assert "moe_experts" in config.recompute_modules


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"moe_grouped_gemm": False}, "moe_act|TE grouped experts"),
        ({"use_transformer_engine_op_fuser": False}, "op fuser"),
        ({"fine_grained_activation_offloading": False}, "fused_group_mlp offloading"),
        ({"offload_modules": ["expert_fc1", "moe_act"]}, "fused_group_mlp offloading"),
        ({"recompute_modules": ["moe_experts", "moe"]}, "whole-moe"),
        ({"moe_paged_stash": True}, "paged stash"),
        ({"moe_ncclep_zero_copy": True}, "zero-copy"),
        ({"delay_wgrad_compute": True}, "deferred expert weight gradients"),
        ({"delta_offload_bytes_across_pp_ranks": 1024}, "delta_offload_bytes_across_pp_ranks=0"),
        ({"cuda_graph_impl": "local"}, "cuda_graph_impl='none'"),
    ],
)
def test_resident_expert_recompute_rejects_unsupported_paths(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_expert_recompute_config(**overrides)


@pytest.mark.parametrize("granularity, modules", [(None, None), ("selective", [])])
def test_disabled_expert_recompute_does_not_require_offloading_or_op_fuser(granularity, modules):
    config = _make_transformer_config(
        recompute_granularity=granularity,
        recompute_modules=modules,
        moe_grouped_gemm=False,
        use_transformer_engine_op_fuser=False,
        fine_grained_activation_offloading=False,
    )
    assert "moe_experts" not in config.recompute_modules


def test_attention_output_recompute_can_share_expert_offloading():
    config = _make_expert_recompute_config(recompute_modules=["moe_experts", "core_attn_output"])
    assert "core_attn_output" in config.recompute_modules


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"multi_latent_attention": True}, "MLA"),
        ({"attention_output_gate": True}, "attention output gating"),
        ({"cuda_graph_impl": "local"}, "cuda_graph_impl='none'"),
        (
            {"fine_grained_activation_offloading": True, "offload_modules": ["core_attn"]},
            "cannot offload core_attn",
        ),
        ({"fp8": "e4m3", "fp8_recipe": "delayed"}, "Delayed scaling"),
    ],
)
def test_attention_output_recompute_rejects_unsupported_paths(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_transformer_config(
            recompute_granularity="selective", recompute_modules=["core_attn_output"], **overrides
        )


def test_qkv_projection_recompute_accepts_mxfp8_hybrid_recompute_and_expert_offloading(monkeypatch):
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_config.is_te_min_version", lambda version: True
    )
    config = _make_expert_recompute_config(
        fp8="e4m3",
        fp8_recipe="mxfp8",
        gtp_weight_remat_size=64,
        expert_gtp_weight_remat_size=36,
        moe_shortcut_connection=True,
        moe_shortcut_parallel=True,
        moe_shared_expert_intermediate_size=128,
        wide_residual=WideResidualConfig(num_streams=3),
        residual_stream_recompute_num_layers=16,
        recompute_modules=[
            "residual_stream",
            "moe_act",
            "moe_experts",
            "gdp_in_proj",
            "gdp_qkv",
            "shared_experts",
            "mlp",
            "core_attn",
            "core_attn_output",
            "qkv_proj",
        ],
    )
    assert "qkv_proj" in config.recompute_modules
    assert config.offload_modules == ["fused_group_mlp"]


def test_qkv_projection_recompute_requires_attention_output_recompute():
    with pytest.raises(ValueError, match="qkv_proj recompute requires core_attn_output"):
        _make_transformer_config(
            recompute_granularity="selective", recompute_modules=["core_attn", "qkv_proj"]
        )


@pytest.mark.parametrize("use_residual_stream", [False, True])
def test_qkv_projection_recompute_rejects_standalone_layernorm(use_residual_stream):
    modules = ["core_attn_output", "qkv_proj", "layernorm"]
    if use_residual_stream:
        modules.append("residual_stream")
    with pytest.raises(ValueError, match="qkv_proj.*standalone layernorm"):
        _make_transformer_config(recompute_granularity="selective", recompute_modules=modules)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"multi_latent_attention": True}, "MLA"),
        ({"attention_output_gate": True}, "attention output gating"),
        ({"fused_single_qkv_rope": True}, "fused_single_qkv_rope"),
        ({"delay_wgrad_compute": True}, "deferred weight gradients"),
        ({"cuda_graph_impl": "local"}, "cuda_graph_impl='none'"),
        ({"cuda_graph_impl": "full_iteration"}, "cuda_graph_impl='none'"),
        ({"enable_cuda_graph": True}, "cuda_graph_impl='none'"),
        ({"external_cuda_graph": True}, "cuda_graph_impl='none'"),
        (
            {"fine_grained_activation_offloading": True, "offload_modules": ["qkv_linear"]},
            "cannot offload qkv_linear",
        ),
        (
            {"fine_grained_activation_offloading": True, "offload_modules": ["core_attn"]},
            "cannot offload core_attn",
        ),
        (
            {"fine_grained_activation_offloading": True, "offload_modules": ["attn_proj"]},
            "cannot offload core_attn or attn_proj",
        ),
        ({"fp8": "e4m3", "fp8_recipe": "delayed"}, "Delayed scaling"),
    ],
)
def test_qkv_projection_recompute_rejects_unsupported_paths(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_transformer_config(
            recompute_granularity="selective",
            recompute_modules=["core_attn_output", "qkv_proj"],
            **overrides,
        )


def test_qkv_projection_recompute_rejects_old_fp8_checkpoint_support(monkeypatch):
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_config.is_te_min_version", lambda version: False
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_config.get_te_version", lambda: "2.5.0"
    )
    with pytest.raises(ValueError, match="qkv_proj.*transformer-engine>=2.6.0dev0"):
        _make_transformer_config(
            recompute_granularity="selective",
            recompute_modules=["core_attn_output", "qkv_proj"],
            fp8="e4m3",
            fp8_recipe="mxfp8",
        )
