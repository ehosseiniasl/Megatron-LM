# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from collections.abc import Generator

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_utils import fused_topk_with_score_function_supports_qb
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import InitType, TransformerConfig
from megatron.core.utils import orthogonal_init_method
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def pg_collection() -> Generator[ProcessGroupCollection, None, None]:
    """Use real process groups for the replicated router."""
    Utils.initialize_model_parallel(1, 1)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


def _config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=64,
        moe_router_topk=2,
        moe_aux_loss_coeff=0,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_router_dtype='fp32',
        add_bias_linear=False,
    )
    kwargs.update(overrides)
    kwargs.setdefault(
        'moe_router_init_method',
        orthogonal_init_method(math.sqrt(kwargs['num_moe_experts'] / kwargs['hidden_size'])),
    )
    return TransformerConfig(**kwargs)


def _assert_scaled_orthogonal(weight: torch.Tensor) -> None:
    rows, cols = weight.shape
    value = weight.detach().double().cpu()
    gram = value @ value.T if rows <= cols else value.T @ value
    scale_sq = rows / cols
    tolerance = 5e-3 if weight.dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(
        gram,
        torch.eye(min(rows, cols), dtype=torch.float64) * scale_sq,
        rtol=0,
        atol=scale_sq * tolerance,
    )


@pytest.mark.parametrize(
    ('init_type', 'fp32_weights'), [(InitType.MEGATRON_LM, False), (InitType.MISTRAL, True)]
)
def test_orthogonal_init_and_reset(
    pg_collection: ProcessGroupCollection, init_type: InitType, fp32_weights: bool
) -> None:
    """The override works with either model init and after casting/wrapping the gate."""
    config = _config(init_type=init_type, moe_router_fp32_weights=fp32_weights)
    router = TopKRouter(config, pg_collection)
    expected_dtype = torch.float32 if fp32_weights else torch.bfloat16
    assert router.weight.dtype == expected_dtype
    _assert_scaled_orthogonal(router.weight)

    wrapped = Float16Module(config, router.cuda())
    wrapped.module.reset_parameters()
    assert router.weight.dtype == expected_dtype
    _assert_scaled_orthogonal(router.weight)


@pytest.mark.parametrize('init_type', list(InitType))
def test_disabled_preserves_initialization_and_rng(
    pg_collection: ProcessGroupCollection, init_type: InitType
) -> None:
    """Disabling the override preserves both the old weights and subsequent RNG draws."""
    config = _config(init_type=init_type, moe_router_init_method=None)
    torch.manual_seed(1234)
    expected = torch.empty(8, 32)
    if init_type == InitType.MISTRAL:
        torch.nn.init.xavier_normal_(expected)
    else:
        assert config.init_method is not None
        config.init_method(expected)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(1234)
    router = TopKRouter(config, pg_collection)
    assert torch.equal(router.weight, expected.bfloat16())
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_skipped_initialization_preserves_weights(pg_collection: ProcessGroupCollection) -> None:
    """perform_initialization=False still prevents reset from overwriting weights."""
    router = TopKRouter(_config(), pg_collection)
    original = router.weight.detach().clone()
    rng = torch.get_rng_state()
    router.config.perform_initialization = False
    router.reset_parameters()
    assert torch.equal(router.weight, original)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.skipif(
    not fused_topk_with_score_function_supports_qb, reason="TE fused QB router ops not available"
)
def test_ultra_router_forward_backward(pg_collection: ProcessGroupCollection) -> None:
    """Exercise Ultra's full router dimensions, fused top-k and global-batch QB."""
    config = _config(
        hidden_size=10240,
        num_attention_heads=96,
        kv_channels=256,
        num_query_groups=1,
        num_moe_experts=512,
        moe_ffn_hidden_size=10240,
        moe_latent_size=3072,
        moe_router_topk=10,
        moe_router_topk_scaling_factor=3.16,
        moe_router_score_function='sigmoid',
        moe_router_load_balancing_type='quantile_balancing',
        moe_router_quantile_balancing_estimation_scope='global_batch',
        moe_router_qb_num_bins=1000,
        moe_router_fusion=True,
        calculate_per_token_loss=True,
    )
    torch.manual_seed(1234)
    router = TopKRouter(config, pg_collection).cuda()
    router.set_layer_number(1)
    wrapped = Float16Module(config, router)
    _assert_scaled_orthogonal(router.weight)
    reference = router.weight.detach().clone()
    torch.distributed.broadcast(reference, src=0)
    assert torch.equal(router.weight, reference)

    inputs = torch.randn(16, 1, 10240, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    probs, routing_map = wrapped(inputs)
    assert torch.all(routing_map.sum(dim=-1) == 10)
    torch.testing.assert_close(probs.sum(dim=-1), torch.full_like(probs.sum(dim=-1), 3.16))
    (probs * torch.randn_like(probs)).sum().backward()
    assert torch.isfinite(router.weight.grad).all() and router.weight.grad.abs().sum() > 0
    assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0
