# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed.finalize_model_grads import (
    _update_router_expert_bias_with_quantile,
    capture_global_batch_quantile_router_state,
    reset_model_temporary_tensors,
    restore_global_batch_quantile_router_state,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import checkpoint
from megatron.core.transformer.moe.moe_utils import (
    get_updated_expert_bias_with_quantile,
    pad_routing_map,
    topk_routing_with_score_function,
)
from megatron.core.transformer.moe.paged_stash import PagedStashRunner
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_router_score_function="sigmoid",
        moe_router_load_balancing_type="quantile_balancing",
        moe_router_quantile_balancing_estimation_scope="global_batch",
        moe_router_qb_bias_space="logit",
        moe_aux_loss_coeff=0.0,
        moe_router_qb_num_bins=1000,
        add_bias_linear=False,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _route(logits, bias, **kwargs):
    return topk_routing_with_score_function(
        logits,
        2,
        score_function="sigmoid",
        expert_bias=bias,
        qb_bias_space="logit",
        scaling_factor=3.16,
        **kwargs,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"moe_router_load_balancing_type": "none"},
        {"moe_router_quantile_balancing_estimation_scope": "micro_batch"},
        {"moe_router_qb_logit_initial_bound": 0},
        {"moe_router_qb_logit_initial_bound": float("nan")},
        {"moe_router_qb_logit_initial_bound": float("inf")},
        {"moe_router_qb_bias_space": "wrong"},
    ],
)
def test_logit_qb_rejects_unsupported_config(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_logit_qb_cli():
    from megatron.training.arguments import add_megatron_arguments

    parser = argparse.ArgumentParser()
    add_megatron_arguments(parser)
    args = parser.parse_args(
        ["--moe-router-qb-bias-space", "logit", "--moe-router-qb-logit-initial-bound", "32"]
    )
    assert args.moe_router_qb_bias_space == "logit"
    assert args.moe_router_qb_logit_initial_bound == 32


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_logit_selection_and_original_score_gradients(dtype):
    # Expert 0 has a tiny sigmoid score, but is selected by its logit-space bias.
    # Expert 2 saturates sigmoid to 1; raw logits still distinguish it from expert 3.
    logits = torch.tensor([[-20.0, -2.0, 20.0, 19.0]], dtype=dtype, requires_grad=True)
    bias = torch.tensor([42.0, 0.0, 0.0, 0.0])
    probs, indices = _route(logits, bias, dense_output=True)
    assert indices.tolist() == [[0, 2]]
    assert probs[0, 0] > 0
    reference_logits = logits.detach().clone().requires_grad_(True)
    selected = torch.sigmoid(reference_logits.float()[:, [0, 2]])
    expected = (3.16 * selected / (selected.sum(-1, keepdim=True) + 1e-20)).to(dtype)
    torch.testing.assert_close(probs, expected)
    upstream = torch.tensor([[1.0, -0.5]], dtype=dtype)
    (probs * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    assert logits.grad[0, 0] != 0
    assert logits.grad[0, 1] == 0


def test_bias_space_changes_selection_not_weight_function():
    logits = torch.tensor([[-3.0, -2.0, 0.0, 2.0]])
    bias = torch.tensor([1.0, 0.0, 0.0, 0.0])
    _, score_indices = topk_routing_with_score_function(
        logits, 2, score_function="sigmoid", expert_bias=bias, dense_output=True
    )
    _, logit_indices = _route(logits, bias, dense_output=True)
    assert score_indices.tolist() == [[0, 3]]
    assert logit_indices.tolist() == [[3, 2]]


def test_logit_histogram_matches_independent_margin_reference():
    logits = torch.tensor(
        [[-18.0, -1.0, 2.0, 4.0], [-12.0, 0.0, -3.0, 6.0], [-14.0, 2.0, 4.0, -1.0]]
    )
    bias = torch.tensor([15.0, -2.0, 0.0, -1.0])
    bounds = torch.tensor([-32.0, 32.0])
    histogram = torch.zeros(4, 128, dtype=torch.int32)
    span = torch.zeros(())
    for microbatch in logits.split(1):
        _route(microbatch, bias, qb_histogram=histogram, qb_bin_bounds=bounds, qb_logit_span=span)
    expected = torch.zeros_like(histogram)
    for row in logits:
        cutoff = sorted((row + bias).tolist(), reverse=True)[2]
        for expert, value in enumerate(row.tolist()):
            index = max(0, min(127, int((cutoff - value + 32.0) // 0.5)))
            expected[expert, index] += 1
    assert torch.equal(histogram, expected)
    assert span == 22.0


def test_adaptive_bounds_and_empty_batch():
    histogram = torch.tensor([[0, 2, 2, 0], [2, 2, 0, 0]], dtype=torch.int32)
    bias, bounds = get_updated_expert_bias_with_quantile(
        histogram, torch.tensor([-1.0, 1.0]), torch.zeros(2), 1, logit_span=torch.tensor(20.0)
    )
    torch.testing.assert_close(bounds, torch.stack((bias.min() - 22.0, bias.max() + 22.0)))
    unchanged_bias, unchanged_bounds = get_updated_expert_bias_with_quantile(
        histogram * 0, bounds, bias, 1, logit_span=torch.tensor(0.0)
    )
    torch.testing.assert_close(unchanged_bias, bias)
    torch.testing.assert_close(unchanged_bounds, bounds)


def test_common_logit_offsets_do_not_change_qb_histogram():
    logits = torch.tensor([[-18.0, -1.0, 2.0, 4.0], [-12.0, 0.0, -3.0, 6.0]])
    bias = torch.tensor([15.0, -2.0, 0.0, -1.0])
    histograms = []
    for shifted in (logits, logits + torch.tensor([[100.0], [-100.0]])):
        histogram = torch.zeros(4, 1000, dtype=torch.int32)
        span = torch.zeros(())
        _route(
            shifted,
            bias,
            qb_histogram=histogram,
            qb_bin_bounds=torch.tensor([-32.0, 32.0]),
            qb_logit_span=span,
        )
        histograms.append(histogram)
        assert span == 22.0
    assert torch.equal(*histograms)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_logits_do_not_poison_next_batch_bounds(invalid):
    logits = torch.tensor([[invalid, -1.0, 2.0, 4.0], [-12.0, 0.0, -3.0, 6.0]])
    bias = torch.tensor([15.0, -2.0, 0.0, -1.0])
    bounds = torch.tensor([-32.0, 32.0])
    histogram = torch.zeros(4, 1000, dtype=torch.int32)
    span = torch.zeros(())
    _route(logits, bias, qb_histogram=histogram, qb_bin_bounds=bounds, qb_logit_span=span)
    assert span == float("inf")
    updated_bias, updated_bounds = get_updated_expert_bias_with_quantile(
        histogram, bounds, bias, 2, logit_span=span
    )
    torch.testing.assert_close(updated_bias, bias)
    torch.testing.assert_close(updated_bounds, bounds)


@pytest.fixture(params=[False, True], ids=["torch", "te"])
def router(request):
    from megatron.core.extensions.transformer_engine import (
        fused_topk_with_score_function_supports_logit_bias,
    )

    if request.param and not fused_topk_with_score_function_supports_logit_bias:
        pytest.skip("Installed TE lacks logit-space bias")
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(1, 1)
    router = TopKRouter(
        _config(moe_router_fusion=request.param), ProcessGroupCollection.use_mpu_process_groups()
    ).cuda()
    yield router
    Utils.destroy_model_parallel()


def test_logit_qb_pooled_update_repeated_invocation_and_rollback(router):
    rank = dist.get_rank()
    # Different ranges across DP, and two invocations of one shared router (MTP).
    logits = torch.randn(
        32, 4, device="cuda", generator=torch.Generator(device="cuda").manual_seed(123 + rank)
    ) * (rank + 1)
    before = capture_global_batch_quantile_router_state([router])
    router.quantile_balancing(logits[:16])
    router.quantile_balancing(logits[16:])
    assert torch.equal(router.qb_histogram.sum(-1), torch.full((4,), 32, device="cuda"))
    pooled = router.qb_histogram.clone()
    span = router.qb_logit_span.clone()
    dist.all_reduce(pooled)
    dist.all_reduce(span, op=dist.ReduceOp.MAX)
    expected_bias, expected_bounds = get_updated_expert_bias_with_quantile(
        pooled, router.qb_bin_bounds, router.expert_bias, 2, logit_span=span
    )
    _update_router_expert_bias_with_quantile([router], router.config, (dist.group.WORLD,))
    torch.testing.assert_close(router.expert_bias, expected_bias)
    torch.testing.assert_close(router.qb_bin_bounds, expected_bounds)
    restore_global_batch_quantile_router_state(before)
    assert router.expert_bias.count_nonzero() == 0
    assert router.qb_histogram.count_nonzero() == 0
    assert router.qb_logit_span == 0
    torch.testing.assert_close(router.qb_bin_bounds, torch.tensor([-16.0, 16.0], device="cuda"))


def test_nonfinite_rank_skips_global_bias_update(router):
    logits = torch.randn(32, 4, device="cuda")
    if dist.get_rank() == 0:
        logits[0, 0] = float("nan")
    router.quantile_balancing(logits)
    _update_router_expert_bias_with_quantile([router], router.config, (dist.group.WORLD,))
    assert router.expert_bias.count_nonzero() == 0
    torch.testing.assert_close(router.qb_bin_bounds, torch.tensor([-16.0, 16.0], device="cuda"))


def test_logit_qb_recompute_eval_and_frozen(router):
    logits = torch.randn(32, 4, device="cuda", requires_grad=True)
    router.eval()
    router.quantile_balancing(logits)
    router.train()
    router.frozen_expert_bias = True
    router.quantile_balancing(logits)
    router.frozen_expert_bias = False
    with torch.no_grad():
        router.quantile_balancing(logits)
    assert router.qb_logit_span == 0
    assert router.qb_histogram.count_nonzero() == 0
    probs, _ = checkpoint(router.quantile_balancing, False, logits)
    assert router.qb_logit_span == 0
    probs.square().sum().backward()
    assert router.qb_logit_span > 0
    assert torch.equal(router.qb_histogram.sum(-1), torch.full((4,), 32, device="cuda"))
    reset_model_temporary_tensors(router.config, [router])
    assert router.qb_logit_span == 0
    assert router.qb_histogram.count_nonzero() == 0
    router.quantile_balancing(logits)
    runner = PagedStashRunner.__new__(PagedStashRunner)
    runner.moe_layers = [type("Layer", (), {"router": router})()]
    runner._reset_qb_histograms()
    assert router.qb_logit_span == 0
    assert router.qb_histogram.count_nonzero() == 0


def test_logit_checkpoint_roundtrip_and_reject_wrong_units(router):
    router.expert_bias.copy_(torch.tensor([-12.0, 3.0, 4.0, 5.0], device="cuda"))
    router.qb_bin_bounds.copy_(torch.tensor([-40.0, 40.0], device="cuda"))
    state = {k: v.clone() for k, v in router.state_dict().items()}
    assert "logit_expert_bias" in state and "expert_bias" not in state
    assert "qb_logit_span" not in state and "qb_histogram" not in state
    router.expert_bias.zero_()
    router.load_state_dict(state)
    torch.testing.assert_close(router.expert_bias, state["logit_expert_bias"])
    score_router = TopKRouter(
        _config(moe_router_qb_bias_space="score"), ProcessGroupCollection.use_mpu_process_groups()
    ).cuda()
    for strict in (False, True):
        with pytest.raises(RuntimeError, match="score-space"):
            score_router.load_state_dict(state, strict=strict)
        with pytest.raises(RuntimeError, match="score-space"):
            router.load_state_dict(score_router.state_dict(), strict=strict)
    sharded = router.sharded_state_dict(metadata={"dp_cp_group": dist.group.WORLD})
    assert "logit_expert_bias" in sharded
    assert "logit_qb_bin_bounds" in sharded


def test_logit_inference_agrees_with_training_route(router, monkeypatch):
    inference = InferenceTopKRouter(
        router.config, ProcessGroupCollection.use_mpu_process_groups()
    ).cuda()
    inference.load_state_dict(router.state_dict())
    # Exercise the specialized dense route without paying compiler startup in a unit test.
    monkeypatch.setattr(inference, "_compiled_topk_routing", topk_routing_with_score_function)
    hidden = torch.randn(16, 1, 16, device="cuda")
    with torch.no_grad():
        sparse, mask = router(hidden)
        dense, indices = inference._forward(hidden)
    torch.testing.assert_close(dense, sparse.gather(1, indices))
    assert torch.equal(mask, torch.zeros_like(mask).scatter(1, indices, True))


def test_fused_logit_qb_rejects_older_te(monkeypatch):
    from megatron.core.transformer.moe import moe_utils

    monkeypatch.setattr(moe_utils, "HAVE_TE", True)
    monkeypatch.setattr(moe_utils, "fused_topk_with_score_function", lambda **kwargs: None)
    monkeypatch.setattr(moe_utils, "fused_topk_with_score_function_supports_logit_bias", False)
    with pytest.raises(ValueError, match="does not support fused logit-space bias"):
        _route(torch.zeros(2, 4), torch.zeros(4), fused=True)


def test_fused_logit_qb_matches_reference():
    from megatron.core.extensions.transformer_engine import (
        fused_topk_with_score_function_supports_logit_bias,
    )

    if not fused_topk_with_score_function_supports_logit_bias:
        pytest.skip("Installed TE lacks logit-space bias")
    logits = torch.tensor([[-20.0, -2.0, 20.0, 19.0]], device="cuda", requires_grad=True)
    bias = torch.tensor([42.0, 0.0, 0.0, 0.0], device="cuda")
    bounds = torch.tensor([-64.0, 64.0], device="cuda")
    results = []
    for fused in (False, True):
        z = logits.detach().clone().requires_grad_(True)
        hist = torch.zeros(4, 128, dtype=torch.int32, device="cuda")
        span = torch.zeros((), device="cuda")
        probs, mask = _route(
            z, bias, fused=fused, qb_histogram=hist, qb_bin_bounds=bounds, qb_logit_span=span
        )
        (probs * torch.tensor([[1.0, -0.5, 0.25, 2.0]], device="cuda")).sum().backward()
        results.append((probs, mask, hist, span, z.grad))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=5e-6)


@pytest.mark.parametrize("space", ["score", "logit"])
@pytest.mark.parametrize("with_histogram", [False, True])
def test_padding_preserves_fused_router_backward(space, with_histogram):
    from megatron.core.extensions.transformer_engine import (
        fused_topk_with_score_function_supports_logit_bias,
    )

    if not fused_topk_with_score_function_supports_logit_bias:
        pytest.skip("Installed TE lacks logit-space bias")
    logits = torch.tensor(
        [[2.0, 1.0, -1.0, -2.0]] * 11 + [[-2.0, -1.0, 1.0, 2.0]] * 5,
        device="cuda",
        requires_grad=True,
    )
    bias = torch.zeros(4, device="cuda")
    kwargs = {}
    if with_histogram:
        kwargs.update(
            qb_histogram=torch.zeros(4, 100, device="cuda", dtype=torch.int32),
            qb_bin_bounds=torch.tensor([-8.0, 8.0], device="cuda"),
        )
    probs, mask = topk_routing_with_score_function(
        logits,
        2,
        score_function="sigmoid",
        expert_bias=bias,
        qb_bias_space=space,
        fused=True,
        **kwargs,
    )
    original = mask.clone()
    expected_grad = torch.autograd.grad(probs.square().sum(), logits, retain_graph=True)[0]
    padded = pad_routing_map(mask, 8)
    assert torch.equal(mask, original)
    assert padded.is_contiguous()
    assert torch.equal(padded.sum(0), torch.tensor([16, 16, 8, 8], device="cuda"))
    probs.square().sum().backward()
    torch.testing.assert_close(logits.grad, expected_grad)
