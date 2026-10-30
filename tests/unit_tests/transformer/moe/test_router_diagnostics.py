# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe.router_diagnostics import (
    ROUTER_DIAGNOSTIC_CHANNEL_COUNT,
    RouterDiagnosticChannel,
    build_router_diagnostics,
)
from tests.unit_tests.test_utilities import Utils


def test_build_router_diagnostics_compacts_per_sequence_scores_loads_and_bias():
    scores = torch.tensor(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.4, 0.4, 0.2],
            [0.7, 0.2, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    aux_map = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.bool
    )
    actual_map = torch.tensor(
        [[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=torch.bool
    )
    padding_mask = torch.tensor([False, False, False, True, True, False])

    diagnostics = build_router_diagnostics(
        scores,
        aux_map,
        actual_map,
        torch.tensor([0.1, -0.1, 0.0]),
        seq_length=3,
        batch_size=2,
        padding_mask=padding_mask,
    )

    assert diagnostics.shape == (2, ROUTER_DIAGNOSTIC_CHANNEL_COUNT, 3)
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.MEAN_SCORE],
        torch.tensor([[0.5, 0.35, 0.15], [0.15, 0.35, 0.5]]),
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.AUX_LOAD],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]),
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.ACTUAL_LOAD],
        torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]),
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.EXPERT_BIAS],
        torch.tensor([[0.1, -0.1, 0.0], [0.1, -0.1, 0.0]]),
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.AUX_ACTUAL_OVERLAP, 0], torch.tensor([0.5, 1.0])
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.VALID_TOKEN_COUNT, 0], torch.tensor([2.0, 2.0])
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.TOPK_BOUNDARY_RELATIVE_MARGIN, 0],
        torch.tensor([0.25, 0.55714285]),
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.ACTUAL_COUNT],
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
    )


def test_build_router_diagnostics_uses_zero_margin_without_a_topk_boundary():
    scores = torch.tensor([[0.5, 0.5], [0.6, 0.4]])
    routing_map = torch.tensor([[True, True], [False, False]])

    diagnostics = build_router_diagnostics(
        scores,
        routing_map,
        routing_map,
        None,
        seq_length=1,
        batch_size=2,
        padding_mask=torch.tensor([False, True]),
    )

    margins = diagnostics[:, RouterDiagnosticChannel.TOPK_BOUNDARY_RELATIVE_MARGIN, 0]
    torch.testing.assert_close(margins, torch.zeros(2))
    assert torch.isfinite(margins).all()


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("with_padding", [False, True])
def test_sequence_parallel_diagnostics_match_complete_sequences(tp_size, with_padding):
    """SP sums must precede sequence normalization, including unequal denominators."""
    if Utils.world_size % tp_size:
        pytest.skip("The distributed test world must be divisible by TP.")
    Utils.initialize_model_parallel(tp_size, 1)
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        seq_length, batch_size, num_experts = 8, 3, 4
        generator = torch.Generator(device="cuda").manual_seed(71)
        scores = torch.rand(
            seq_length, batch_size, num_experts, generator=generator, device="cuda"
        ).softmax(dim=-1)
        aux_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(
            -1, scores.topk(2, dim=-1).indices, True
        )
        actual_map = aux_map.roll(1, dims=-1)
        # Boundary counts and assignment counts need their own denominators, not
        # valid-token counts or a fixed top-k. Exercise all-selected/no-selected rows.
        aux_map[:2, 0] = True
        aux_map[4:, 1] = False
        actual_map[2:5, 0] = False
        padding = torch.zeros((seq_length, batch_size), device="cuda", dtype=torch.bool)
        if with_padding:
            padding[:4, 0] = True  # Some ranks have no valid tokens in this sequence.
            padding[-1, 1] = True
            padding[:, 2] = True  # Also exercise a globally empty sequence.
        bias = torch.tensor([-0.3, 0.1, 0.2, 0.9], device="cuda")
        scores.requires_grad_()
        expected = build_router_diagnostics(
            scores.flatten(0, 1),
            aux_map.flatten(0, 1),
            actual_map.flatten(0, 1),
            bias,
            seq_length,
            batch_size,
            padding.flatten() if with_padding else None,
        )
        local = slice(
            group.rank() * seq_length // tp_size, (group.rank() + 1) * seq_length // tp_size
        )
        actual = build_router_diagnostics(
            scores[local].flatten(0, 1),
            aux_map[local].flatten(0, 1),
            actual_map[local].flatten(0, 1),
            bias,
            seq_length // tp_size,
            batch_size,
            padding[local].flatten() if with_padding else None,
            sequence_parallel_group=group,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            actual[:, RouterDiagnosticChannel.EXPERT_BIAS], bias.expand(batch_size, -1)
        )
        assert not actual.requires_grad
        assert torch.isfinite(actual).all()
    finally:
        Utils.destroy_model_parallel()


def test_complete_sequence_diagnostics_do_not_communicate(monkeypatch):
    """Omitting the SP group must never implicitly reduce across the world group."""

    def unexpected_collective(*args, **kwargs):
        pytest.fail("Complete local sequences do not need a diagnostic collective.")

    monkeypatch.setattr(torch.distributed, "all_reduce", unexpected_collective)
    scores = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    routing = torch.tensor([[True, False], [False, True]])
    build_router_diagnostics(scores, routing, routing, seq_length=2, batch_size=1)
