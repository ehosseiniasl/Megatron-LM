# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression coverage for the production online metrics integration."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.tensor_observation import capture_tensor_observations, observe_layer_residuals
from megatron.training.global_vars import _WandbSharedWriter
from megatron.training.tensor_metrics import (
    MetricSite,
    MetricTensor,
    RankRelation,
    Shard,
    TensorMetricExecutor,
)
from megatron.training.tensor_metrics import observer as observer_mod
from megatron.training.tensor_metrics import scalar_logging
from megatron.training.tensor_metrics.activation_metrics import (
    LoggingRankLocalExpert0ReluHealthMetric,
)
from megatron.training.tensor_metrics.residual_metrics import LayerResidualSummaryMetric
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.training.test_tensor_metric_observer import (
    _fp32_optimizer,
    _layer_model,
    _pg_collection,
)


@pytest.fixture(autouse=True)
def distributed_cuda():
    Utils.initialize_distributed()


def _residual_results(populations, *, distributed=False):
    relations = (RankRelation("dp", Shard(1)),) if distributed else ()
    item = MetricTensor(
        populations,
        (MetricSite("decoder.layers.0.residual_summary", "residual_summary"),),
        relations,
    )
    groups = {"dp": torch.distributed.group.WORLD} if distributed else {}
    return {
        result.label: result.tensor
        for result in TensorMetricExecutor(groups).run(LayerResidualSummaryMetric(), (item,))
    }


@pytest.mark.parametrize("distributed", (False, True))
@pytest.mark.parametrize("outlier", (0.0, 64.0))
def test_residual_histogram_preserves_zero_mass(distributed, outlier):
    norms = torch.zeros(1001, device="cuda")
    norms[-1] = outlier
    populations = torch.stack((norms, norms, torch.zeros_like(norms)))
    if distributed and torch.distributed.get_rank() != 0:
        populations = populations[:, :0]

    results = _residual_results(populations, distributed=distributed)

    for scope in ("decoder.layers.0", "global"):
        for population in ("accumulator", "contribution"):
            for quantile in ("p75", "p90", "p95"):
                assert results[f"{scope}/{population}/{quantile}"].item() == 0.0
            assert results[f"{scope}/{population}/max"].item() == outlier
            torch.testing.assert_close(
                results[f"{scope}/{population}/rms"],
                torch.tensor(outlier / 1001**0.5, device="cuda"),
            )


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("population_index", (0, 1, 2))
@pytest.mark.parametrize("distributed", (False, True))
def test_nonfinite_residual_invalidates_only_its_population(bad, population_index, distributed):
    populations = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-0.5, 0.5]], device="cuda")
    if not distributed or torch.distributed.get_rank() == 0:
        populations[population_index, -1] = bad
    elif torch.distributed.get_rank() > 1:
        # Include empty ranks as well as a rank with finite observations. Ranks must
        # agree on the histogram layout despite seeing different local populations.
        populations = populations[:, :0]

    results = _residual_results(populations, distributed=distributed)

    invalid_population = ("accumulator", "contribution", "correlation")[population_index]
    for label, value in results.items():
        if f"/{invalid_population}/" in label:
            assert torch.isnan(value).item()
        else:
            assert torch.isfinite(value).item()


@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
def test_residual_histogram_handles_an_entirely_nonfinite_population(bad):
    results = _residual_results(torch.full((3, 2), bad, device="cuda"), distributed=True)
    assert all(torch.isnan(value).item() for value in results.values())


def test_residual_summary_includes_last_layer_boundaries():
    values = [
        MetricTensor(
            torch.tensor([[3.0, 4.0]], device="cuda"),
            (MetricSite("decoder.last_layer", "residual_summary_boundary"),),
        ),
        MetricTensor(
            torch.tensor([[0.0, 2.0]], device="cuda"),
            (MetricSite("decoder.last_layer_widereadout", "residual_summary_boundary"),),
        ),
    ]

    results = {
        result.label: result.tensor
        for result in TensorMetricExecutor({}).run(LayerResidualSummaryMetric(), values)
    }

    assert set(results) == {
        f"{boundary}/{statistic}"
        for boundary in ("last-layer", "last-layer-wide-readout")
        for statistic in ("rms", "p75", "p90", "p95", "max")
    }
    torch.testing.assert_close(results["last-layer/rms"], torch.tensor(12.5**0.5, device="cuda"))
    torch.testing.assert_close(
        results["last-layer-wide-readout/rms"], torch.tensor(2.0**0.5, device="cuda")
    )
    for statistic in ("p75", "p90", "p95", "max"):
        assert results[f"last-layer/{statistic}"].item() == 4.0
        assert results[f"last-layer-wide-readout/{statistic}"].item() == 2.0


def _expert_values(layer, tokens):
    return [
        MetricTensor(
            tensor.unsqueeze(0),
            (MetricSite(f"decoder.layers.{layer}.mlp.local_experts.0.{name}", kind),),
        )
        for name, kind, tensor in (
            ("pre_relu", "expert_pre_relu", tokens),
            ("post_relu", "expert_post_relu", tokens.relu()),
        )
    ]


def test_empty_logging_expert_is_omitted_without_diluting_populated_experts():
    metric = LoggingRankLocalExpert0ReluHealthMetric()
    executor = TensorMetricExecutor({})
    empty = _expert_values(0, torch.empty(0, 2, device="cuda"))
    populated = _expert_values(1, torch.tensor([[-1.0, 2.0], [3.0, -4.0]], device="cuda"))

    assert executor.run(metric, empty) == []
    expected = executor.run(metric, populated)
    actual = executor.run(metric, empty + populated)

    assert [result.label for result in actual] == [result.label for result in expected]
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left.tensor, right.tensor)


def test_empty_microbatch_does_not_remove_later_expert_observations():
    metric = LoggingRankLocalExpert0ReluHealthMetric()
    executor = TensorMetricExecutor({})
    empty = _expert_values(0, torch.empty(0, 2, device="cuda"))
    populated = _expert_values(0, torch.tensor([[-1.0, 2.0]], device="cuda"))

    actual = executor.run(metric, empty + populated)
    expected = executor.run(metric, populated)

    assert [result.label for result in actual] == [result.label for result in expected]
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left.tensor, right.tensor)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_residual_norm_metrics_share_the_fused_observation(device):
    layer = SimpleNamespace(config=SimpleNamespace(sequence_parallel=False))
    accumulator = torch.arange(24, device=device, dtype=torch.float32).reshape(3, 2, 4)
    output = accumulator + 2
    observed = {}

    with capture_tensor_observations(
        lambda owner, name, kind, tensor, dim: observed.update({kind: tensor}),
        frozenset({"residual_summary"}),
    ):
        observe_layer_residuals(layer, accumulator, output)

    assert set(observed) == {"residual_summary"}
    item = MetricTensor(
        observed["residual_summary"],
        (MetricSite("decoder.layers.0.residual_summary", "residual_summary"),),
    )
    executor = TensorMetricExecutor({})
    for name, expected in (
        ("layer-residual-accumulator-l2", accumulator.norm()),
        ("layer-residual-contribution-l2", (output - accumulator).norm()),
    ):
        metric = observer_mod.parse_tensor_metric_specs([f"{name}:1"])[0].metric
        results = executor.run(metric, (item,))
        assert {result.label for result in results} == {"decoder.layers.0", "global"}
        for result in results:
            torch.testing.assert_close(result.tensor, expected)


def test_parameter_and_gradient_metrics_share_batched_sinks_and_literal_family_names(monkeypatch):
    tensorboard = Mock()
    backend = Mock()
    wandb = _WandbSharedWriter(backend)
    monkeypatch.setattr(scalar_logging, "get_tensorboard_writer", lambda: tensorboard)
    monkeypatch.setattr(scalar_logging, "get_wandb_writer", lambda: wandb)
    monkeypatch.setattr(observer_mod, "get_nvtx_range", lambda: lambda *a, **k: nullcontext())
    model = _layer_model().cuda()
    for parameter in model.parameters():
        parameter.main_grad = torch.ones_like(parameter)
    observer = observer_mod.build_tensor_metric_observer(
        ["param-l2:1", "wgrad-summary-individual:1"]
    )

    observer(
        model=[model], optimizer=_fp32_optimizer(model), iteration=0, pg_collection=_pg_collection()
    )

    tensorboard.add_scalar.assert_not_called()
    assert tensorboard.file_writer.add_summary.call_count == 2  # One batch per metric.
    scalars = {}
    for call in tensorboard.file_writer.add_summary.call_args_list:
        summary, iteration = call.args
        assert iteration == 1
        scalars.update((value.tag, value.simple_value) for value in summary.value)
    assert scalars["tensor-metrics-global/param-l2/global"] == 5.0
    assert "tensor-metrics-breakdowns/param-l2/family/decoder.layers.ALL.weight" in scalars
    assert not any("*" in tag for tag in scalars)
    assert sum("/wgrad-summary-individual/" in tag for tag in scalars) == 6
    logged = {
        key: value for call in backend.log.call_args_list for key, value in call.args[0].items()
    }
    assert {tag: logged[tag] for tag in scalars} == scalars
