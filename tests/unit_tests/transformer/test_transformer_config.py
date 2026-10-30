# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.transformer_config import TransformerConfig


def _make_transformer_config(**kwargs) -> TransformerConfig:
    return TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4, **kwargs)


def test_squared_relu_tanh_clamp_accepts_supported_configuration():
    config = _make_transformer_config(
        activation_func=squared_relu,
        activation_func_tanh_clamp_scale=16.0,
        use_fused_weighted_squared_relu=True,
    )

    assert config.activation_func_tanh_clamp_scale == 16.0


@pytest.mark.parametrize("clamp_scale", [0.0, -1.0, float("inf"), float("nan")])
def test_squared_relu_tanh_clamp_rejects_invalid_scale(clamp_scale):
    with pytest.raises(ValueError, match="must be finite and positive"):
        _make_transformer_config(
            activation_func=squared_relu, activation_func_tanh_clamp_scale=clamp_scale
        )


@pytest.mark.parametrize(
    "overrides",
    [{"activation_func": F.gelu}, {"activation_func": squared_relu, "gated_linear_unit": True}],
)
def test_squared_relu_tanh_clamp_rejects_other_activation_layouts(overrides):
    with pytest.raises(ValueError, match="requires non-gated squared-ReLU"):
        _make_transformer_config(activation_func_tanh_clamp_scale=16.0, **overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"activation_func_clamp_value": 10.0}, "cannot both be set"),
        ({"bias_activation_fusion": True}, "bias_activation_fusion"),
        ({"use_te_activation_func": True}, "use_te_activation_func"),
    ],
)
def test_squared_relu_tanh_clamp_rejects_bypassing_fused_paths(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_transformer_config(
            activation_func=squared_relu, activation_func_tanh_clamp_scale=16.0, **overrides
        )


def test_squared_relu_tanh_clamp_allows_te_op_fuser():
    """The op fuser is a runtime capability question, not a config-validity one.

    Whether the fused path can clamp depends on the installed TE and cuDNN frontend, so
    rejecting the combination here would make one config valid on one node and invalid on
    another -- and, because checkpoint load force-restores this field, would report a stale
    dependency as an invalid config on resume. TEGroupedMLP._is_fused_impl_supported owns that
    decision and falls back to the unfused clamped path when it cannot fuse.
    """
    config = _make_transformer_config(
        activation_func=squared_relu,
        activation_func_tanh_clamp_scale=16.0,
        use_fused_weighted_squared_relu=True,
        use_transformer_engine_op_fuser=True,
    )

    assert config.activation_func_tanh_clamp_scale == 16.0
    assert config.use_transformer_engine_op_fuser


def _make_overlap_config(mtp_num_layers: int | None) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
        bf16=True,
        mtp_num_layers=mtp_num_layers,
    )


@pytest.mark.parametrize("mtp_num_layers", [None, 0, 1])
def test_ep_a2a_overlap_accepts_supported_mtp_layer_counts(mtp_num_layers: int | None):
    config = _make_overlap_config(mtp_num_layers)

    assert config.mtp_num_layers == mtp_num_layers


@pytest.mark.parametrize("mtp_num_layers", [-1, 2])
def test_ep_a2a_overlap_rejects_unsupported_mtp_layer_counts(mtp_num_layers: int):
    with pytest.raises(AssertionError, match="MTP supports at most one layer"):
        _make_overlap_config(mtp_num_layers)


def test_batch_invariant_backend_rejects_unknown_value_at_construction():
    # Programmatic construction bypasses argparse's Literal choices, so
    # __post_init__ must catch typos before model init.
    with pytest.raises(AssertionError, match="Unknown batch_invariant_backend"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            batch_invariant_mode=True,
            batch_invariant_backend="te-native",
        )


def test_gdp_num_householder_defaults_to_three():
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)

    assert config.gdp_num_householder == 3


def test_gdp_num_householder_accepts_positive_values():
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, gdp_num_householder=5
    )

    assert config.gdp_num_householder == 5


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        pytest.param(
            {"moe_shortcut_connection": True},
            AssertionError,
            "requires MoE to be enabled",
            id="requires-moe",
        ),
        pytest.param(
            {"num_moe_experts": 2, "moe_shortcut_parallel": True},
            AssertionError,
            "requires moe_shortcut_connection",
            id="parallel-requires-shortcut",
        ),
        pytest.param(
            {"num_moe_experts": 2, "moe_shortcut_post_norm": True},
            ValueError,
            "requires moe_shortcut_connection",
            id="post-norm-requires-shortcut",
        ),
        pytest.param(
            {
                "num_moe_experts": 2,
                "moe_shortcut_connection": True,
                "recompute_granularity": "full",
            },
            ValueError,
            "not supported with full activation recomputation",
            id="full-recompute",
        ),
        pytest.param(
            {
                "num_moe_experts": 2,
                "moe_shortcut_connection": True,
                "recompute_granularity": "selective",
                "recompute_modules": ["moe"],
            },
            ValueError,
            "Whole-MoE selective recomputation is not supported",
            id="whole-moe-selective-recompute",
        ),
        pytest.param(
            {"num_moe_experts": 2, "moe_shortcut_connection": True, "moe_layer_recompute": True},
            ValueError,
            "Whole-MoE selective recomputation is not supported",
            id="deprecated-whole-moe-recompute",
        ),
        pytest.param(
            {
                "num_moe_experts": 2,
                "moe_shortcut_connection": True,
                "moe_shared_expert_overlap": True,
            },
            ValueError,
            "mutually exclusive",
            id="shared-expert-overlap",
        ),
    ],
)
def test_shortcut_rejects_incompatible_configurations(overrides, error, message):
    with pytest.raises(error, match=message):
        TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            **overrides,
        )


def test_shortcut_accepts_local_block_inference_cuda_graphs():
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_shortcut_connection=True,
        cuda_graph_impl="local",
        inference_cuda_graph_scope=InferenceCudaGraphScope.block,
    )

    assert config.cuda_graph_impl == "local"
    assert config.inference_cuda_graph_scope == InferenceCudaGraphScope.block


@pytest.mark.parametrize(
    ("cuda_graph_impl", "inference_cuda_graph_scope"),
    [
        pytest.param("local", InferenceCudaGraphScope.layer, id="local-layer"),
        pytest.param("transformer_engine", InferenceCudaGraphScope.none, id="transformer-engine"),
        pytest.param("full_iteration", InferenceCudaGraphScope.none, id="full-iteration"),
    ],
)
def test_shortcut_rejects_unvalidated_cuda_graph_modes(cuda_graph_impl, inference_cuda_graph_scope):
    with pytest.raises(AssertionError, match="supported only with"):
        TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_moe_experts=2,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_shortcut_connection=True,
            cuda_graph_impl=cuda_graph_impl,
            inference_cuda_graph_scope=inference_cuda_graph_scope,
        )


@pytest.mark.parametrize("num_householder", [0, -1])
def test_gdp_num_householder_rejects_non_positive_values(num_householder: int):
    with pytest.raises(ValueError, match="gdp_num_householder must be positive"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            gdp_num_householder=num_householder,
        )


def _make_mxfp8_wire_config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="ncclep",
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=True,
        moe_dispatch_fwd_dtype='mxfp8',
        moe_combine_bwd_dtype='mxfp8',
        bf16=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_mxfp8_wire_dtypes_accept_valid_ncclep_config():
    config = _make_mxfp8_wire_config()

    assert config.moe_dispatch_fwd_dtype == 'mxfp8'
    assert config.moe_combine_bwd_dtype == 'mxfp8'


def test_mxfp8_wire_dtypes_accept_a2a_overlap():
    # The 1F1B a2a overlap schedule only moves/stages the dispatch output as an opaque block,
    # which the plain-tensor MXFP8 carrier survives; the combination is deliberately allowed.
    config = _make_mxfp8_wire_config(overlap_moe_expert_parallel_comm=True)

    assert config.overlap_moe_expert_parallel_comm


@pytest.mark.parametrize(
    "overrides",
    [
        dict(moe_flex_dispatcher_backend="hybridep"),
        dict(moe_token_dispatcher_type="alltoall", moe_flex_dispatcher_backend=None),
    ],
)
def test_mxfp8_wire_dtypes_reject_non_ncclep_dispatcher(overrides):
    with pytest.raises(ValueError, match="require the 'ncclep' flex"):
        _make_mxfp8_wire_config(**overrides)


@pytest.mark.parametrize(
    "overrides", [dict(use_transformer_engine_op_fuser=False), dict(moe_grouped_gemm=False)]
)
def test_mxfp8_wire_dtypes_require_op_fuser_grouped_gemm(overrides):
    with pytest.raises(ValueError, match="require BOTH"):
        _make_mxfp8_wire_config(**overrides)


def test_mxfp8_2d_quantization_accepts_mxfp8_recipe():
    config = _make_transformer_config(fp8="e4m3", fp8_recipe="mxfp8", mxfp8_2d_quantization=True)

    assert config.mxfp8_2d_quantization


@pytest.mark.parametrize(
    ("config_kwargs", "match"),
    [
        ({"fp8_recipe": "mxfp8"}, "together with fp8 mode"),
        ({"fp8": "e4m3", "fp8_recipe": "delayed"}, "requires fp8_recipe='mxfp8'"),
        (
            {"fp8": "e4m3", "fp8_recipe": "mxfp8", "moe_single_grouped_weight": True},
            "does not support moe_single_grouped_weight",
        ),
    ],
)
def test_mxfp8_2d_quantization_rejects_incompatible_fp8_config(config_kwargs, match):
    with pytest.raises(ValueError, match=match):
        _make_transformer_config(mxfp8_2d_quantization=True, **config_kwargs)


def test_offload_modules_fused_group_mlp_excludes_unfused_expert_knobs():
    """fused_group_mlp and expert_fc1 + moe_act are equivalent spellings, so combining them is
    rejected as redundant."""
    common = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=2,
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=True,
        fine_grained_activation_offloading=True,
    )
    TransformerConfig(offload_modules=["fused_group_mlp"], **common)
    TransformerConfig(offload_modules=["expert_fc1", "moe_act"], **common)
    with pytest.raises(ValueError, match="cannot be combined with expert_fc1 or moe_act"):
        TransformerConfig(offload_modules=["fused_group_mlp", "moe_act"], **common)
