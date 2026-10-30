# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training source attribution, weighted reduction, and observer scheduling."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.tensor_observation import capture_tensor_observations
from megatron.training.tensor_metrics import observer as observer_mod
from megatron.training.tensor_metrics.core import (
    MetricResult,
    MetricSite,
    MetricTensor,
    RankRelation,
    Replica,
    Shard,
    TensorMetricExecutor,
)
from megatron.training.tensor_metrics.data_source_loss import (
    DataSourceLossMetric,
    capture_data_source_metrics,
    configured_data_source_names,
    data_source_metrics_requested,
    observe_data_source_loss,
)

SOURCES = ("000-a", "001-b", "002-absent")


def _group(size=1):
    return SimpleNamespace(size=lambda: size, rank=lambda: 0)


def test_observer_weights_tokens_across_microbatches_and_owns_prepared_state(monkeypatch):
    monkeypatch.setattr(observer_mod, "get_nvtx_range", lambda: lambda *a, **kw: nullcontext())
    captured = {}
    observer = observer_mod.build_tensor_metric_observer(
        ["train-loss-by-data-source:2"],
        data_source_names=SOURCES,
        result_sink=lambda metric, results, step: captured.update(
            {r.label: r.tensor.item() for r in results}
        ),
    )
    model = torch.nn.Module()
    pg = SimpleNamespace(cp=_group(), tp=_group(), gtp_remat=_group(), dp_cp=_group())
    for iteration in (0, 1):
        with observer.observe_forward_backward(
            model=[model], iteration=iteration, pg_collection=pg
        ):
            for losses, mask, ids in (
                ([[2, 4, 9, float("nan")]], [[1, 1, 1, 0]], [[0, 0, 1, 2]]),
                ([[12]], [[1]], [[0]]),
            ):
                losses = torch.tensor(losses, dtype=torch.float32, requires_grad=True)
                observe_data_source_loss(model, losses, torch.tensor(mask), torch.tensor(ids))
                with torch.no_grad():
                    losses.fill_(999)
        observer(model=[model], optimizer=None, iteration=iteration, pg_collection=pg)
        if iteration == 0:
            assert captured == {}
    assert captured == {"source/000-a": 6, "source/001-b": 9, "global": 6.75}
    # Encoder-only ranks have no loss observations and must start no source reductions.
    captured.clear()
    with observer.observe_forward_backward(model=[model], iteration=3, pg_collection=pg):
        pass
    observer(model=[model], optimizer=None, iteration=3, pg_collection=pg)
    assert captured == {}


def test_source_reduction_merges_remote_sources_without_counting_tp_replicas(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    dp, tp, calls = object(), object(), []

    def all_reduce(tensor, op, group):
        assert group is dp and op == torch.distributed.ReduceOp.SUM
        calls.append(tensor.numel())
        tensor.add_(tensor.new_tensor([[12, 1], [9, 3], [0, 0]]).reshape_as(tensor))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    value = MetricTensor(
        torch.tensor([[6, 2], [0, 0], [0, 0]], dtype=torch.float64),
        (MetricSite("data_source_loss", "data_source_loss"),),
        (RankRelation("tp", Replica()), RankRelation("dp", Shard(None))),
    )
    results = TensorMetricExecutor({"tp": tp, "dp": dp}).run(DataSourceLossMetric(SOURCES), [value])
    assert calls == [6]
    assert {r.label: r.tensor.item() for r in results} == {
        "source/000-a": 6,
        "source/001-b": 3,
        "global": 4.5,
    }


@pytest.mark.parametrize(
    "ids", [torch.tensor([[3]]), torch.tensor([[-1]]), torch.tensor([[0.5]]), torch.tensor([0])]
)
def test_invalid_token_provenance_is_rejected(ids):
    with capture_data_source_metrics(SOURCES, SimpleNamespace(cp=_group())):
        with pytest.raises(ValueError, match="source IDs"):
            observe_data_source_loss(torch.nn.Module(), torch.ones(1, 1), torch.ones(1, 1), ids)


def test_source_context_resets_after_failure_and_rejects_unsharded_cp_metadata():
    with pytest.raises(RuntimeError, match="forward failed"):
        with capture_data_source_metrics(SOURCES, SimpleNamespace(cp=_group())):
            raise RuntimeError("forward failed")
    observed = []
    with capture_tensor_observations(lambda *args: observed.append(args), {"data_source_loss"}):
        observe_data_source_loss(
            torch.nn.Module(),
            torch.ones(1, 1),
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.int64),
        )
    assert observed == []
    with pytest.raises(NotImplementedError, match="CP=1"):
        with capture_data_source_metrics(SOURCES, SimpleNamespace(cp=_group(2))):
            pass


def test_zero_counts_are_omitted_and_large_counts_keep_precision():
    metric = DataSourceLossMetric(SOURCES)
    count = 2**40 + 1
    results = metric.finalize_results(
        [
            MetricResult(
                torch.tensor([[count * 3, count], [0, 0], [0, 0]], dtype=torch.float64), "global"
            )
        ]
    )
    assert [(r.label, r.tensor.item()) for r in results] == [("source/000-a", 3), ("global", 3)]
    assert metric.finalize_results([MetricResult(torch.zeros(3, 2), "global")]) == []


def test_source_names_follow_config_order_and_metric_requires_names():
    args = SimpleNamespace(
        _mistral_pretokenized_config=SimpleNamespace(
            sources=[
                SimpleNamespace(prefix_path="/data/github/merged_123b/chunks_00000-00001"),
                SimpleNamespace(prefix_path="/data/github/merged_123b/chunks_00001-00002"),
                SimpleNamespace(prefix_path="/data/image source"),
            ]
        )
    )
    assert configured_data_source_names(args) == ("000-github", "001-github", "002-image_source")
    assert configured_data_source_names(SimpleNamespace(dataset_provider="mock")) == ("000-mock",)
    assert data_source_metrics_requested(["train-loss-by-data-source:10"])
    assert not data_source_metrics_requested(["layer-residual-accumulator-l2:10"])
    with pytest.raises(ValueError, match="require MIMO"):
        configured_data_source_names(SimpleNamespace())
    with pytest.raises(ValueError, match="source names"):
        observer_mod.build_tensor_metric_observer(["train-loss-by-data-source:1"])
    with pytest.raises(ValueError, match="source names"):
        DataSourceLossMetric(["a", "a"])
