# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.tensor_observation import (
    capture_tensor_observations,
    suspend_tensor_observations,
)
from megatron.core.transformer.moe.contribution_diagnostics import (
    SOURCE,
    contribution_energy_payload,
    contribution_rank_map,
    moe_contribution_mtp_depth,
    observe_moe_contributions,
    ranked_contribution_sums,
    returned_contribution_sums,
    row_moments,
    validate_moe_contribution_config,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import permute
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock
from megatron.training.tensor_metrics.core import (
    MetricSite,
    MetricTensor,
    RankRelation,
    Replica,
    Shard,
    TensorMetricExecutor,
)
from megatron.training.tensor_metrics.moe_metrics import MoEContributionsMetric


def _config(**overrides):
    return SimpleNamespace(
        **{
            'moe_token_dispatcher_type': 'alltoall',
            'moe_expert_capacity_factor': None,
            'moe_latent_size': None,
            'sequence_parallel': False,
            'moe_router_topk': 2,
            'moe_router_padding_for_quantization': False,
            'moe_permute_fusion': False,
            'batch_invariant_mode': False,
            **overrides,
        }
    )


def _emit(probs, selected, contributions, shared, *, depth=None):
    """Reference token/expert population in arbitrary expert-id order."""
    topk = int(selected[0].sum())
    rank_map, routing = contribution_rank_map(probs, selected, topk)
    # Expert-major storage, as returned by dispatch before expert outputs merge.
    pairs = selected.T.nonzero()
    expert, token = pairs.unbind(1)
    routing.ranks = rank_map[token, expert]
    weighted = contributions[token, expert]
    sums = ranked_contribution_sums(weighted, routing)
    routed = contributions.sum(1)
    observed = []
    with capture_tensor_observations(lambda *a: observed.append(a), frozenset({SOURCE})):
        with moe_contribution_mtp_depth(depth) if depth is not None else nullcontext():
            observe_moe_contributions(
                SimpleNamespace(config=_config()), routed, shared, routing, sums
            )
    return observed[0]


def _value(observation, relations=()):
    return MetricTensor(
        observation[3], (MetricSite('decoder.layers.0.mlp.' + observation[1], SOURCE),), relations
    )


def _results(values, groups=None):
    metric = MoEContributionsMetric()
    executor = TensorMetricExecutor(groups or {})
    return {r.label: r.tensor for r in executor.run(metric, values)}


def test_weight_rank_ignores_expert_id_and_preserves_selected_zero_weights():
    p = torch.tensor([[0.5, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, False, True, True], [False, True, True, True]])
    rank_map, meta = contribution_rank_map(p, mask, 3)
    torch.testing.assert_close(rank_map, torch.tensor([[1.0, 0.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]]))
    torch.testing.assert_close(meta.weight_sums, torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64))
    assert meta.zero_weight_tokens == 1


def test_actual_outputs_determine_rank_energy_not_only_router_weights():
    p = torch.tensor([[0.25, 0.75], [0.8, 0.2]])
    mask = torch.ones_like(p, dtype=torch.bool)
    # First token: low router weight has the larger output. Second: cancellation.
    c = torch.tensor([[[3.0, 4.0], [0.0, 2.0]], [[1.0, 0.0], [-1.0, 0.0]]], requires_grad=True)
    shared = torch.tensor([[0.0, 6.0], [0.0, 2.0]], requires_grad=True)
    obs = _emit(p, mask, c, shared)
    assert not obs[3].requires_grad
    s = _results([_value(obs)])
    f = lambda name: s['decoder.layers.0/' + name]
    assert f('shared-rms') == pytest.approx(20**0.5)
    assert f('routed-rms') == pytest.approx((45 / 2) ** 0.5)
    assert f('shared-over-routed-rms') == pytest.approx((40 / 45) ** 0.5)
    assert f('shared-routed-alignment') == pytest.approx(36 / (40 * 45) ** 0.5)
    assert f('top-over-last-rms') == pytest.approx((5 / 26) ** 0.5)
    assert f('rank_1/energy-share') == pytest.approx(5 / 31)
    assert f('rank_2/energy-share') == pytest.approx(26 / 31)
    assert f('rank_1/mean-weight-share') == pytest.approx(0.775)
    assert f('routed-zero-fraction') == 0.5


def test_pool_squares_before_ratios_and_keep_mtp_depths_separate():
    def sample(shared, routed, depth):
        return _value(
            _emit(
                torch.ones(1, 1),
                torch.ones(1, 1, dtype=torch.bool),
                torch.tensor([[[routed]]]),
                torch.tensor([[shared]]),
                depth=depth,
            )
        )

    values = [sample(2.0, 1.0, 1), sample(0.0, 3.0, 1), sample(5.0, 1.0, 2)]
    s = _results(values)
    assert s['decoder.layers.0/mtp_1/shared-over-routed-rms'] == pytest.approx((4 / 10) ** 0.5)
    assert s['decoder.layers.0/mtp_2/shared-over-routed-rms'] == 5
    assert s['decoder.layers.0/mtp_1/tokens'] == 2
    assert s['decoder.layers.0/mtp_2/tokens'] == 1


def test_top_three_share_uses_output_energy_and_weight_scale_is_preserved():
    p = torch.tensor([[0.4, 0.3, 0.2, 0.1]]) * 3.16
    c = torch.tensor([[[4.0], [3.0], [2.0], [1.0]]]) * 3.16
    obs = _emit(p, torch.ones_like(p, dtype=torch.bool), c, torch.ones(1, 1))
    s = _results([_value(obs)])
    assert s['decoder.layers.0/top-three-energy-share'] == pytest.approx(29 / 30)
    assert s['decoder.layers.0/rank_1/rms'] == pytest.approx(4 * 3.16)
    assert s['decoder.layers.0/rank_1/mean-weight-share'] == pytest.approx(0.4)
    assert s['decoder.layers.0/shared-over-routed-rms'] == pytest.approx(1 / 31.6)


def test_real_repeated_mtp_loop_labels_each_depth_and_resets_per_microbatch():
    p = torch.ones(1, 1)
    _, meta = contribution_rank_map(p, p.bool(), 1)
    meta.ranks = torch.ones(1)
    sums = ranked_contribution_sums(p, meta)
    owner = SimpleNamespace(config=_config())

    def layer(**kw):
        observe_moe_contributions(owner, p, p, meta, sums)
        return (
            kw['hidden_states'],
            kw['input_ids'],
            kw['position_ids'],
            kw['padding_mask'],
            kw['mtp_input_mask'],
        )

    block = SimpleNamespace(
        config=SimpleNamespace(mtp_detach_heads=False, mtp_hsm=False, mtp_num_layers=2),
        training=True,
        vp_stage=None,
        pp_rank=0,
        mtp_use_repeated_layer=True,
        layers=[layer],
    )
    observed = []
    with patch(
        'megatron.core.transformer.multi_token_prediction.get_mtp_layer_offset', return_value=0
    ):
        with capture_tensor_observations(lambda *a: observed.append(a), frozenset({SOURCE})):
            for _ in range(2):
                MultiTokenPredictionBlock.forward(block, None, None, torch.ones(1, 1, 1), None)
            observe_moe_contributions(owner, p, p, meta, sums)
    assert [v[1] for v in observed] == [
        'moe_contributions.mtp_1',
        'moe_contributions.mtp_2',
        'moe_contributions.mtp_1',
        'moe_contributions.mtp_2',
        'moe_contributions',
    ]


def test_zero_denominators_are_explicit_and_no_shared_expert_is_supported():
    obs = _emit(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, 3), None)
    s = _results([_value(obs)])
    for key in (
        'shared-over-routed-rms',
        'shared-routed-alignment',
        'top-over-last-rms',
        'top-three-energy-share',
    ):
        assert torch.isnan(s['decoder.layers.0/' + key])
    for key in (
        'zero-shared-energy',
        'zero-routed-energy',
        'zero-last-rank-energy',
        'zero-individual-energy',
        'zero-weight-sum-fraction',
    ):
        assert s['decoder.layers.0/' + key] == 1


def test_rank_zero_padding_does_not_pollute_energy_or_zero_fractions():
    _, meta = contribution_rank_map(
        torch.tensor([[1.0, 0.0]]), torch.ones(1, 2, dtype=torch.bool), 2
    )
    meta.ranks = torch.tensor([1.0, 0.0, 2.0, 0.0])
    sums, zeros = ranked_contribution_sums(
        torch.tensor([[3.0, 4.0], [100.0, 100.0], [0.0, 0.0], [0.0, 0.0]]), meta
    )
    torch.testing.assert_close(sums, torch.tensor([25.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(zeros, torch.tensor([0.0, 1.0], dtype=torch.float64))


def test_tiny_nonzero_outputs_are_not_counted_as_exact_zero():
    moments = row_moments(torch.tensor([[1e-30], [0.0]]))
    # Squaring can underflow in FP32; zero-output counts inspect the values themselves.
    torch.testing.assert_close(moments[1], torch.tensor([0.0, 1.0]))


def test_empty_token_shard_and_no_grad_forward():
    rank_map, meta = contribution_rank_map(
        torch.empty(0, 4), torch.empty(0, 4, dtype=torch.bool), 2
    )
    assert rank_map.shape == (0, 4)
    meta.ranks = torch.empty(0)
    sums = ranked_contribution_sums(torch.empty(0, 3), meta)
    observed = []
    with (
        torch.no_grad(),
        capture_tensor_observations(lambda *a: observed.append(a), frozenset({SOURCE})),
    ):
        observe_moe_contributions(
            SimpleNamespace(config=_config()), torch.empty(0, 3), None, meta, sums
        )
    s = _results([_value(observed[0])])
    assert s['decoder.layers.0/empty-population'] == 1
    assert torch.isnan(s['decoder.layers.0/shared-rms'])


def test_sharded_populations_reduce_moments_and_replicas_do_not(monkeypatch):
    obs = _emit(
        torch.ones(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[[2.0]]]),
        torch.tensor([[1.0]]),
    )
    value = _value(obs, (RankRelation('dp', Shard(0)), RankRelation('tp', Replica())))
    calls = []
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda group: 2)

    def reduce(tensor, op, group):
        calls.append(group)
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, 'all_reduce', reduce)
    s = _results([value], {'dp': 'dp', 'tp': 'tp'})
    assert calls == ['dp']
    assert s['decoder.layers.0/tokens'] == 2
    assert s['decoder.layers.0/shared-over-routed-rms'] == 0.5


@pytest.mark.parametrize(
    'kwargs',
    [
        {'moe_token_dispatcher_type': 'flex'},
        {'moe_expert_capacity_factor': 1.1},
        {
            'moe_token_dispatcher_type': 'flex',
            'moe_flex_dispatcher_backend': 'hybridep',
            'expert_tensor_parallel_size': 2,
        },
        {
            'moe_token_dispatcher_type': 'flex',
            'moe_flex_dispatcher_backend': 'hybridep',
            'moe_expert_rank_capacity_factor': 1.5,
        },
        {'moe_latent_size': 64, 'moe_shared_expert_overlap': True},
        {
            'moe_token_dispatcher_type': 'flex',
            'moe_flex_dispatcher_backend': 'hybridep',
            'moe_hybridep_pad_uneven_dispatch_inputs': True,
        },
    ],
)
def test_unsupported_layouts_fail_explicitly(kwargs):
    with pytest.raises(NotImplementedError):
        validate_moe_contribution_config(_config(**kwargs))


@pytest.mark.parametrize('overlap', [False, True])
def test_dispatch_and_shared_combine_do_not_change_output_or_gradients(overlap):
    p = torch.tensor([[0.7, 0.3], [0.1, 0.9]], requires_grad=True)
    mask = torch.ones_like(p, dtype=torch.bool)
    x = torch.tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    shared = torch.tensor([[1.0, -1.0], [2.0, 1.0]], requires_grad=True)
    observed = []
    dispatcher = object.__new__(MoEAlltoAllTokenDispatcher)
    dispatcher.config = _config()
    dispatcher.shared_experts = None
    dispatcher.drop_and_pad = False
    dispatcher.preprocess = lambda routing: routing.sum(0)
    dispatcher._maybe_dtoh_and_synchronize = lambda point, values: values
    dispatcher.num_out_tokens = 4
    owner = SimpleNamespace(config=_config(), token_dispatcher=dispatcher)

    def run(enabled):
        with capture_tensor_observations(
            lambda *a: observed.append(a), frozenset({SOURCE}) if enabled else frozenset()
        ):
            inputs, weights = dispatcher.dispatch_preprocess(x, mask, p)
            # No communication is needed in a one-rank layout. Distinct expert transforms.
            weighted = inputs * weights[:, None] * torch.tensor([[1.0], [1.0], [2.0], [2.0]])
            if overlap:
                dispatcher.shared_experts = SimpleNamespace(get_output=lambda: shared)
            result = MoELayer.postprocess(owner, weighted, None if overlap else shared)
            dispatcher.shared_experts = None
        grads = torch.autograd.grad(result.square().sum(), (x, p, shared))
        return result, grads

    baseline, old_grads = run(False)
    result, new_grads = run(True)
    torch.testing.assert_close(result, baseline, rtol=0, atol=0)
    for a, b in zip(old_grads, new_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(observed) == 1
    assert dispatcher._contribution_routing is None
    assert not observed[0][3].requires_grad


def test_suspended_recomputation_has_no_observations():
    observed = []
    with capture_tensor_observations(lambda *a: observed.append(a), frozenset({SOURCE})):
        with (
            suspend_tensor_observations(),
            patch(
                'megatron.core.transformer.moe.contribution_diagnostics.row_moments',
                side_effect=AssertionError('recomputation must not measure'),
            ),
        ):
            observe_moe_contributions(None, None, None, None, None)
    assert observed == []


def test_dispatch_alignment_padding_keeps_zero_weight_selection_distinct():
    import megatron.core.transformer.moe.token_dispatcher as dispatcher_module

    dispatcher = object.__new__(MoEAlltoAllTokenDispatcher)
    dispatcher.config = _config(moe_router_padding_for_quantization=True)
    dispatcher.shared_experts = None
    dispatcher.drop_and_pad = False

    def preprocess(routing):
        dispatcher.num_out_tokens = int(routing.sum())
        return routing.sum(0)

    dispatcher.preprocess = preprocess
    dispatcher._maybe_dtoh_and_synchronize = lambda point, values: values
    p = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.25, 0.75], [0.0, 0.0, 1.0]])
    mask = torch.tensor(
        [[True, False, True], [True, True, False], [False, True, True], [True, False, True]]
    )
    with patch.object(dispatcher_module, 'get_align_size_for_quantization', return_value=4):
        with capture_tensor_observations(lambda *a: None, frozenset({SOURCE})):
            dispatcher.dispatch_preprocess(torch.ones(4, 2), mask, p)
    meta = dispatcher._contribution_routing
    assert meta.valid_tokens.sum() == 4
    assert (meta.ranks == 0).sum() == 4
    assert (meta.ranks == 1).sum() == (meta.ranks == 2).sum() == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel validation')
@pytest.mark.parametrize('hidden', [3, 3072, 6144, 7168, 10240])
def test_cuda_row_reductions_match_fp32_reference(hidden):
    a = torch.randn(5, hidden, device='cuda', dtype=torch.bfloat16)
    b = torch.randn_like(a)
    expected = row_moments(a.cpu(), b.cpu()).cuda()
    torch.testing.assert_close(row_moments(a, b), expected, rtol=2e-5, atol=2e-4)
    torch.testing.assert_close(row_moments(a), expected[[0, 3]], rtol=2e-5, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='TE permutation validation')
@pytest.mark.parametrize('fused', [False, True])
def test_permuted_scalar_ranks_match_actual_output_order(fused):
    from megatron.core.transformer.moe.moe_utils import pad_routing_map

    p = torch.rand(128, 8, device='cuda')
    chosen = p.topk(2, dim=-1).indices
    selected = torch.zeros_like(p, dtype=torch.bool).scatter_(1, chosen, True)
    p *= selected
    rank_map, _ = contribution_rank_map(p, selected, 2)
    padded = pad_routing_map(selected, 16)
    rows = int(padded.sum())
    x = torch.arange(128, device='cuda', dtype=torch.float32)[:, None].expand(-1, 32).contiguous()
    outputs, weights, _, _, _ = permute(x, padded, probs=p, num_out_tokens=rows, fused=fused)
    _, ranks, _, _, _ = permute(x, padded, probs=rank_map, num_out_tokens=rows, fused=fused)
    # The rank's router weight must agree with the weight accompanying that exact row.
    tokens = outputs[:, 0].long()
    sorted_weights = p.sort(dim=-1, descending=True).values[:, :2]
    real = ranks > 0
    torch.testing.assert_close(weights[real], sorted_weights[tokens[real], ranks[real].long() - 1])
    assert torch.all(weights[~real] == 0)


@pytest.mark.parametrize("latent", [None, 64])
def test_hybridep_config_supported(latent):
    validate_moe_contribution_config(
        _config(
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            expert_tensor_parallel_size=1,
            moe_latent_size=latent,
        )
    )


def test_returned_energy_channel_preserves_ranking_zeros_and_ignores_padding():
    p = torch.tensor([[0.0, 0.8, 0.2, 0.0], [0.5, 0.0, 0.0, 0.5]])
    mask = torch.tensor([[False, True, True, False], [True, False, False, True]])
    _, routing = contribution_rank_map(p, mask, 2)
    # Poison non-selections; valid zero and underflowing nonzero remain distinct.
    c = torch.tensor([[[99.0], [3.0], [0.0], [99.0]], [[1e-30], [99.0], [99.0], [4.0]]])
    payload = contribution_energy_payload(c.reshape(-1, 1)).reshape(2, 2, 2)
    energy, zeros = returned_contribution_sums(payload, routing)
    torch.testing.assert_close(energy, torch.tensor([9.0, 16.0], dtype=torch.float64))
    torch.testing.assert_close(zeros, torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert not energy.requires_grad


@pytest.mark.parametrize('ep', [1, 8, 64])
def test_returned_energy_channel_512_experts_crosses_ep_boundaries(ep):
    # Exercise global expert ids at partition boundaries, including a selected
    # zero-weight expert. Poison non-selections and a padded trailing token.
    experts = torch.tensor([511, 0, 255, 8, 63, 64, 256, 7, 504, 127])
    weights = torch.arange(10, dtype=torch.float32).flip(0)
    probs = torch.zeros(2, 512)
    selected = torch.zeros_like(probs, dtype=torch.bool)
    payload = torch.full((3, 512), float('nan'))
    expected = torch.arange(1, 11, dtype=torch.float64)
    for token in range(2):
        ids = experts.roll(token)
        probs[token, ids] = weights
        selected[token, ids] = True
        payload[token, ids] = expected.float() * (token + 1)
        payload[token, ids[-1]] = -1  # Exact-zero output of the last real selection.
    _, routing = contribution_rank_map(probs, selected, 10)
    energy, zeros = returned_contribution_sums(payload.reshape(3, ep, 512 // ep), routing)
    expected *= 3
    expected[-1] = 0
    torch.testing.assert_close(energy, expected)
    torch.testing.assert_close(zeros, torch.tensor([0.0] * 9 + [2.0], dtype=torch.float64))


def test_latent_projection_shared_comparison_and_rank_labels():
    p = torch.tensor([[0.7, 0.3], [0.1, 0.9]], requires_grad=True)
    mask = torch.ones_like(p, dtype=torch.bool)
    x = torch.tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    shared = torch.tensor([[1.0, -1.0, 2.0], [2.0, 1.0, 3.0]], requires_grad=True)
    projection = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], requires_grad=True)
    config = _config(moe_latent_size=2, moe_use_norm_before_up_proj=True)
    dispatcher = object.__new__(MoEAlltoAllTokenDispatcher)
    dispatcher.config = config
    dispatcher.shared_experts = None
    dispatcher.drop_and_pad = False
    dispatcher.preprocess = lambda routing: routing.sum(0)
    dispatcher._maybe_dtoh_and_synchronize = lambda point, values: values
    dispatcher.num_out_tokens = 4

    class Norm(torch.nn.Module):
        def forward(self, y):
            return y / (y.square().mean(-1, keepdim=True) + 1e-5).sqrt()

    owner = SimpleNamespace(
        config=config,
        token_dispatcher=dispatcher,
        fc2_norm=Norm(),
        fc2_latent_proj=lambda y: (y @ projection, None),
    )
    observed = []

    def run(enabled):
        with capture_tensor_observations(
            lambda *a: observed.append(a), frozenset({SOURCE}) if enabled else frozenset()
        ):
            inputs, weights = dispatcher.dispatch_preprocess(x, mask, p)
            weighted = inputs * weights[:, None]
            result = MoELayer.postprocess(owner, weighted, shared)
        return result, torch.autograd.grad(result.square().sum(), (x, p, shared, projection))

    reference, old_grads = run(False)
    result, grads = run(True)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for a, b in zip(grads, old_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    s = _results([_value(observed[0])])
    actual_routed = reference.detach() - shared.detach()
    expected = shared.square().sum() / actual_routed.square().sum()
    torch.testing.assert_close(
        s['decoder.layers.0/shared-over-routed-rms'], expected.double().sqrt()
    )
    assert 'decoder.layers.0/latent/top-over-last-rms' in s
    assert 'decoder.layers.0/latent/rank_1/rms' in s
    assert 'decoder.layers.0/rank_1/rms' not in s


def test_hybridep_side_channel_is_detached_and_keeps_backward(monkeypatch):
    import megatron.core.transformer.moe.fused_a2a as fused

    class Buffer:
        @staticmethod
        def combine_with_unpermute(*, hidden, probs=None, **kwargs):
            return hidden.clone(), None if probs is None else probs.clone()

        @staticmethod
        def dispatch_with_permute(*, hidden, **kwargs):
            return hidden.clone(), None, None, None, None

    monkeypatch.setattr(fused, "_hybrid_ep_buffer", Buffer())
    x = torch.randn(4, 3, requires_grad=True)
    payload = torch.randn(4, requires_grad=True)
    reference = fused.HybridEPCombine.apply(x, None, None, None, False, None)
    output, returned = fused.HybridEPCombine.apply(x, None, None, None, False, payload)
    assert not returned.requires_grad
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    grad = torch.autograd.grad(output.square().sum(), x)[0]
    torch.testing.assert_close(grad, 2 * x, rtol=0, atol=0)
    assert payload.grad is None
