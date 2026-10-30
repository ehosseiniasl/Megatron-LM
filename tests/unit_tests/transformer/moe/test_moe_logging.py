# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker

ORTHOGONALITY = "router_orthogonality_loss"
AUXILIARY = "load_balancing_loss"


def _record_metrics():
    tracker = MoEMetricsTracker()
    # Three selected routers, including one with a genuinely zero penalty.
    for layer, value in [(1, 6.0), (3, 0.0), (5, 12.0)]:
        tracker.record(ORTHOGONALITY, torch.tensor(value), layer, num_layers=8)
    for layer, value in [(1, 8.0), (7, 16.0)]:
        tracker.record(AUXILIARY, torch.tensor(value), layer, num_layers=8)
    return tracker


@pytest.mark.parametrize(
    "layer_counts, expected_orthogonality", [(None, 1.5), ({ORTHOGONALITY: 3}, 3.0)]
)
def test_metric_layer_counts_only_override_selected_metric(layer_counts, expected_orthogonality):
    """The denominator counts selected routers, including zero penalties, not nonzero slots."""
    tracker = _record_metrics()

    values = tracker._aggregate(
        loss_scale=0.5,
        num_moe_layers=6,
        metric_names=[ORTHOGONALITY, AUXILIARY],
        metric_layer_counts=layer_counts,
    )

    torch.testing.assert_close(values[ORTHOGONALITY], torch.tensor(expected_orthogonality))
    torch.testing.assert_close(values[AUXILIARY], torch.tensor(2.0))


def test_zero_contributors_report_explicit_zero():
    tracker = _record_metrics()

    values = tracker._aggregate(0.5, 6, [ORTHOGONALITY], metric_layer_counts={ORTHOGONALITY: 0})

    assert isinstance(values[ORTHOGONALITY], torch.Tensor)
    torch.testing.assert_close(values[ORTHOGONALITY], torch.tensor(0.0))


def test_negative_metric_layer_count_is_rejected():
    tracker = _record_metrics()

    with pytest.raises(ValueError):
        tracker._aggregate(0.5, 6, [ORTHOGONALITY], metric_layer_counts={ORTHOGONALITY: -1})


def test_report_uses_metric_layer_counts_for_writer_and_loss_accumulator(monkeypatch):
    tracker = _record_metrics()
    synchronized = []
    monkeypatch.setattr(
        tracker, "_sync_metrics", lambda names, groups: synchronized.append(list(names))
    )
    logged = {}
    writer = SimpleNamespace(
        add_scalar=lambda name, value, step: logged.update({name: (value.clone(), step)})
    )
    total_loss = {ORTHOGONALITY: torch.tensor(4.0)}

    message = tracker.report(
        loss_scale=0.5,
        iteration=7,
        writer=writer,
        track_names=[ORTHOGONALITY, AUXILIARY],
        num_moe_layers=6,
        metric_layer_counts={ORTHOGONALITY: 3},
        total_loss_dict=total_loss,
    )

    assert synchronized == [[ORTHOGONALITY, AUXILIARY]]
    assert message == ""
    for name, expected in [(ORTHOGONALITY, 3.0), (AUXILIARY, 2.0)]:
        torch.testing.assert_close(logged[name][0], torch.tensor(expected))
        assert logged[name][1] == 7
    torch.testing.assert_close(total_loss[ORTHOGONALITY], torch.tensor(7.0))
    torch.testing.assert_close(total_loss[AUXILIARY], torch.tensor(2.0))
    assert all(torch.count_nonzero(entry.values) == 0 for entry in tracker.metrics.values())


def test_empty_rank_initializes_full_metric_vector_for_zero_selection(monkeypatch):
    tracker = MoEMetricsTracker()
    initialize = tracker.ensure_initialized
    monkeypatch.setattr(
        tracker, "ensure_initialized", lambda name, size: initialize(name, size, device="cpu")
    )
    synchronized = []
    monkeypatch.setattr(
        tracker,
        "_sync_metrics",
        lambda names, groups: synchronized.append(tracker.metrics[ORTHOGONALITY].values.clone()),
    )
    total_loss = {}

    tracker.report(
        loss_scale=0.5,
        iteration=1,
        force_initialize=True,
        track_names=[ORTHOGONALITY],
        num_layers=8,
        num_moe_layers=6,
        metric_layer_counts={ORTHOGONALITY: 0},
        total_loss_dict=total_loss,
    )

    assert len(synchronized) == 1
    torch.testing.assert_close(synchronized[0], torch.zeros(8))
    torch.testing.assert_close(total_loss[ORTHOGONALITY], torch.tensor(0.0))
    torch.testing.assert_close(tracker.metrics[ORTHOGONALITY].values, torch.zeros(8))
