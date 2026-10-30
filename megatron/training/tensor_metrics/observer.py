# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training integration for configured tensor metrics."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TypeAlias

import torch

from megatron.core.optimizer import MegatronOptimizer
from megatron.core.parameter_names import CanonicalParameterNameMap
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_observation import capture_tensor_observations
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.contribution_diagnostics import validate_moe_contribution_config
from megatron.core.utils import unwrap_model
from megatron.training.global_vars import get_args, get_tokenizer
from megatron.training.utils import get_nvtx_range

from .activation_metrics import LoggingRankLocalExpert0ReluHealthMetric
from .core import (
    LogicalReductionMetric,
    MetricResult,
    MetricSite,
    MetricStep,
    MetricTensor,
    RankRelation,
    Replica,
    Shard,
    TensorMetric,
    TensorMetricExecutor,
)
from .data_source_loss import DataSourceLossMetric, capture_data_source_metrics
from .data_source_metrics import ModalityLossMetric
from .definitions import (
    AbsoluteMaximumMetric,
    GlobalL2NormMetric,
    L2NormMetric,
    LayerL2NormMetric,
    MaximumMetric,
    MeanAbsoluteRmsMetric,
    MeanColumnL2NormMetric,
    MeanRowL2NormMetric,
    ZeroFractionMetric,
)
from .embedding_metrics import EmbeddingWgradRowL2Metric
from .gdp_metrics import LayerGDPHeadStatisticMetric
from .moe_metrics import MoEContributionsMetric
from .optimizer_sources import (
    _build_optimizer_parameter_manifest,
    _optimizer_metric_tensor_batches,
    _OptimizerParameterManifestEntry,
)
from .residual_metrics import (
    LayerDResidualSummaryByTokenTypeMetric,
    LayerDResidualSummaryMetric,
    LayerResidualSummaryByTokenTypeMetric,
    LayerResidualSummaryMetric,
)
from .router_metrics import (
    LayerRouterDecisionEntropyMetric,
    LayerRouterExpertBiasMetric,
    LayerRouterExpertStateMetric,
    LayerRouterHealthMetric,
    LayerRouterLogitsL2NormMetric,
    LayerRouterLogitsMaxMetric,
    LayerRouterLogitsSampledMedianMetric,
    LayerRouterRoutingBalanceMetric,
    LayerRouterSeqAuxDecompositionMetric,
)
from .scalar_logging import write_partitioned_scalar_results
from .small_wgrad import SmallWgradPreparation, WgradSmallFractionMetric

__all__ = [
    "MetricResultSink",
    "ScheduledMetric",
    "TrainingTensorMetricObserver",
    "build_tensor_metric_observer",
    "parse_tensor_metric_specs",
]


MetricResultSink: TypeAlias = Callable[[TensorMetric, Sequence[MetricResult], int], None]


@dataclass(frozen=True)
class ScheduledMetric:
    """A metric paired with an externally configurable training interval.

    Args:
        metric: Metric to execute when the schedule is due.
        interval: Positive number of training iterations between observations.
    """

    metric: TensorMetric
    interval: int

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("ScheduledMetric.interval must be positive.")

    def is_due(self, iteration: int) -> bool:
        """Return whether the metric is due on a zero-based training iteration.

        Args:
            iteration: Zero-based training iteration.

        Returns:
            Whether this iteration completes an interval.
        """
        return (iteration + 1) % self.interval == 0


class _GlobalParameterL2NormMetric(GlobalL2NormMetric):
    name = "global-param-l2"


class _LayerParameterL2NormMetric(LayerL2NormMetric):
    name = "layer-param-l2"


class _GlobalParameterMeanRowL2NormMetric(MeanRowL2NormMetric):
    name = "global-param-mean-row-l2"
    source_kinds = frozenset({"parameter"})


class _GlobalParameterMeanColumnL2NormMetric(MeanColumnL2NormMetric):
    name = "global-param-mean-column-l2"
    source_kinds = frozenset({"parameter"})


class _GlobalWgradL2NormMetric(GlobalL2NormMetric):
    name = "global-wgrad-l2"
    source_kinds = frozenset({"wgrad"})


class _LayerWgradL2NormMetric(LayerL2NormMetric):
    name = "layer-wgrad-l2"
    source_kinds = frozenset({"wgrad"})
    include_global = True


class _LayerResidualL2NormMetric(LayerL2NormMetric):
    source_kinds = frozenset({"residual_summary"})
    summary_row = 0
    include_global = True

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[self.summary_row].square().sum()

    def contribution_batch(self, tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Select this population instead of applying fused L2 to all three summary rows."""
        return [self.contribution(tensor) for tensor in tensors]


class _LayerResidualAccumulatorL2NormMetric(_LayerResidualL2NormMetric):
    name = "layer-residual-accumulator-l2"


class _LayerResidualContributionL2NormMetric(_LayerResidualL2NormMetric):
    name = "layer-residual-contribution-l2"
    summary_row = 1


class _GlobalOutputLogitsL2NormMetric(GlobalL2NormMetric):
    name = "global-output-logits-l2"
    source_kinds = frozenset({"output_logits"})


class _GlobalMTPLogitsL2NormMetric(GlobalL2NormMetric):
    name = "global-mtp-logits-l2"
    source_kinds = frozenset({"mtp_logits"})


def _parameter_family_label(site: MetricSite) -> str:
    """Replace layer/expert instance indices with the literal family label ``ALL``."""
    parts = site.name.split(".")
    is_expert = "experts" in parts or "local_experts" in parts
    normalized = []
    for index, part in enumerate(parts):
        previous = parts[index - 1] if index else None
        if previous in {"layers", "experts", "local_experts"} and part.isdigit():
            normalized.append("ALL")
            continue
        if is_expert:
            if "[experts=" in part and part.endswith("]"):
                part = part.partition("[experts=")[0]
            for stem in ("weight", "bias"):
                if part.startswith(stem) and part[len(stem) :].isdigit():
                    part = stem
                    break
        normalized.append(part)
    return f"family/{'.'.join(normalized)}"


def _optimizer_reductions(
    values: Sequence[MetricTensor], *, individual: bool = False
) -> list[tuple[str, Sequence[MetricTensor]]]:
    """Group optimizer tensors individually, or by layer, family, and globally."""
    groups: dict[str, list[MetricTensor]] = {}
    for value in values:
        labels = tuple(
            site.name if individual else _parameter_family_label(site) for site in value.sites
        )
        if any(label != labels[0] for label in labels[1:]):
            raise ValueError("An optimizer metric tensor must identify one parameter or family.")
        groups.setdefault(labels[0], []).append(value)
    grouped = [(label, groups[label]) for label in sorted(groups)]
    if individual:
        return grouped
    layer_values = LayerL2NormMetric._selected_values(values)
    reductions = LayerL2NormMetric._reductions_by_layer(layer_values)
    reductions.extend(grouped)
    if values:
        reductions.append(("global", values))
    return reductions


class _OptimizerMetricBreakdowns(LogicalReductionMetric):
    """Share aggregation scopes across optimizer statistics without changing legacy metrics."""

    def __init__(self, *, individual: bool = False) -> None:
        self._individual = individual
        if individual:
            self.name = f"{self.name}-individual"

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Reduce prepared statistics at the configured aggregation scopes."""
        return self._start_logical_reductions(
            _optimizer_reductions(values, individual=self._individual)
        )


class _ParameterL2NormMetric(_OptimizerMetricBreakdowns, L2NormMetric):
    name = "param-l2"
    source_kinds = frozenset({"parameter"})


class _WgradL2NormMetric(_OptimizerMetricBreakdowns, L2NormMetric):
    name = "wgrad-l2"
    source_kinds = frozenset({"wgrad"})


class _WgradAbsoluteMaximumMetric(_OptimizerMetricBreakdowns, AbsoluteMaximumMetric):
    name = "wgrad-abs-max"
    source_kinds = frozenset({"wgrad"})


class _WgradZeroFractionMetric(_OptimizerMetricBreakdowns, ZeroFractionMetric):
    name = "wgrad-zero-fraction"
    source_kinds = frozenset({"wgrad"})


class _WgradSmallFractionMetric(_OptimizerMetricBreakdowns, WgradSmallFractionMetric):
    """Apply existing optimizer scopes to absolute or tensor-relative small counts."""

    def __init__(
        self, thresholds: Sequence[float], *, relative: bool = False, individual: bool = False
    ) -> None:
        WgradSmallFractionMetric.__init__(self, thresholds, relative=relative)
        _OptimizerMetricBreakdowns.__init__(self, individual=individual)


class _WgradSummaryMetric(_OptimizerMetricBreakdowns, MeanAbsoluteRmsMetric):
    """Summarize gradient elements by layer, parameter family, and globally."""

    name = "wgrad-summary"
    source_kinds = frozenset({"wgrad"})


class _IndividualWgradSummaryMetric(MeanAbsoluteRmsMetric):
    name = "wgrad-summary-individual"
    source_kinds = frozenset({"wgrad"})

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        return self._start_logical_reductions(_individual_optimizer_reductions(values))


class _OutputLogitsL2NormMetric(GlobalL2NormMetric):
    name = "output-logits-l2"
    source_kinds = frozenset({"output_logits"})


class _MTPLogitsL2NormMetric(GlobalL2NormMetric):
    name = "mtp-logits-l2"
    source_kinds = frozenset({"mtp_logits"})


class _GlobalOutputLogitsMaximumMetric(MaximumMetric):
    name = "output-logits-max"
    source_kinds = frozenset({"output_logits"})


def _mtp_logits_depth_label(site: MetricSite) -> str | None:
    """Return the stable MTP-depth label encoded by an ``mtp_logits.N`` site suffix."""
    name_parts = site.name.split(".")
    if len(name_parts) < 2 or name_parts[-2] != "mtp_logits" or not name_parts[-1].isdigit():
        return None
    return f"mtp_logits.{name_parts[-1]}"


def _mtp_logits_reductions(
    values: Sequence[MetricTensor],
) -> list[tuple[str, Sequence[MetricTensor]]]:
    """Group MTP-logit observations by prediction depth and over all depths."""
    values_by_depth: dict[str, list[MetricTensor]] = {}
    for value in values:
        labels = tuple(_mtp_logits_depth_label(site) for site in value.sites)
        depth_label = labels[0]
        if depth_label is None or any(label != depth_label for label in labels[1:]):
            raise ValueError("An MTP-logit metric tensor must identify one prediction depth.")
        values_by_depth.setdefault(depth_label, []).append(value)
    reductions: list[tuple[str, Sequence[MetricTensor]]] = list(values_by_depth.items())
    if values:
        reductions.append(("global", values))
    return reductions


class _LayerMTPLogitsMaximumMetric(MaximumMetric):
    name = "mtp-logits-max"
    source_kinds = frozenset({"mtp_logits"})

    def accepts(self, site: MetricSite) -> bool:
        return super().accepts(site) and _mtp_logits_depth_label(site) is not None

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        return self._start_logical_reductions(_mtp_logits_reductions(values))


def _metric_alias(metric_type: type[TensorMetric], name: str) -> TensorMetric:
    metric = metric_type()
    metric.name = name
    return metric


def _prepare_optimizer_metric_values(
    executor: TensorMetricExecutor,
    metrics: Sequence[TensorMetric],
    batches: Iterator[list[MetricTensor]],
    *,
    replay_batches: Callable[[], Iterator[list[MetricTensor]]] | None = None,
) -> dict[int, list[MetricTensor]]:
    """Compact each parameter's semantic copies before materializing the next parameter."""
    small_metrics = tuple(m for m in metrics if isinstance(m, WgradSmallFractionMetric))
    small = SmallWgradPreparation(small_metrics) if small_metrics else None
    metrics = tuple(m for m in metrics if not isinstance(m, WgradSmallFractionMetric))
    prepared = {id(metric): [] for metric in metrics}
    for batch in batches:
        if small is not None:
            small.add_batch(batch)
        for metric in metrics:
            prepared[id(metric)].extend(executor.prepare(metric, batch))
        # Release the current batch before advancing the generator to the next fused tensor.
        del batch
    if small is not None:
        prepared.update(small.finish(executor, replay_batches))
    return prepared


def _individual_optimizer_reductions(
    values: Sequence[MetricTensor],
) -> list[tuple[str, Sequence[MetricTensor]]]:
    """Group optimizer tensors by their complete canonical parameter name."""
    values_by_name: dict[str, list[MetricTensor]] = {}
    for value in values:
        names = tuple(site.name for site in value.sites)
        if any(name != names[0] for name in names[1:]):
            raise ValueError("An individual optimizer tensor must identify one parameter.")
        values_by_name.setdefault(names[0], []).append(value)
    return [(name, values_by_name[name]) for name in sorted(values_by_name)]


_TENSOR_METRIC_FACTORIES: dict[str, Callable[[], TensorMetric]] = {
    "moe-contributions": MoEContributionsMetric,
    "embedding-wgrad-row-l2": EmbeddingWgradRowL2Metric,
    "global-param-l2": _GlobalParameterL2NormMetric,
    "layer-param-l2": _LayerParameterL2NormMetric,
    "global-param-mean-row-l2": _GlobalParameterMeanRowL2NormMetric,
    "global-param-mean-column-l2": _GlobalParameterMeanColumnL2NormMetric,
    "global-wgrad-l2": _GlobalWgradL2NormMetric,
    "layer-wgrad-l2": _LayerWgradL2NormMetric,
    "layer-residual-accumulator-l2": _LayerResidualAccumulatorL2NormMetric,
    "layer-residual-contribution-l2": _LayerResidualContributionL2NormMetric,
    "global-output-logits-l2": _GlobalOutputLogitsL2NormMetric,
    "global-mtp-logits-l2": _GlobalMTPLogitsL2NormMetric,
    "layer-router-logits-l2": LayerRouterLogitsL2NormMetric,
    "layer-router-logits-max": LayerRouterLogitsMaxMetric,
    "layer-router-logits-sampled-median": LayerRouterLogitsSampledMedianMetric,
    "layer-router-decision-entropy": LayerRouterDecisionEntropyMetric,
    "layer-router-health": LayerRouterHealthMetric,
    "layer-router-seq-aux-decomposition": LayerRouterSeqAuxDecompositionMetric,
    "layer-router-routing-balance": LayerRouterRoutingBalanceMetric,
    "layer-router-expert-bias": LayerRouterExpertBiasMetric,
    "layer-router-expert-state": LayerRouterExpertStateMetric,
    "wgrad-summary": _WgradSummaryMetric,
    "wgrad-summary-individual": _IndividualWgradSummaryMetric,
    "residual-summary": LayerResidualSummaryMetric,
    "dresidual-summary": LayerDResidualSummaryMetric,
    "residual-summary-by-token-type": LayerResidualSummaryByTokenTypeMetric,
    "dresidual-summary-by-token-type": LayerDResidualSummaryByTokenTypeMetric,
    "output-logits-l2": _OutputLogitsL2NormMetric,
    "mtp-logits-l2": _MTPLogitsL2NormMetric,
    "output-logits-max": _GlobalOutputLogitsMaximumMetric,
    "mtp-logits-max": _LayerMTPLogitsMaximumMetric,
    "router-logits-l2": partial(_metric_alias, LayerRouterLogitsL2NormMetric, "router-logits-l2"),
    "router-logits-max": partial(_metric_alias, LayerRouterLogitsMaxMetric, "router-logits-max"),
    "router-decision-entropy": partial(
        _metric_alias, LayerRouterDecisionEntropyMetric, "router-decision-entropy"
    ),
    "router-health": partial(_metric_alias, LayerRouterHealthMetric, "router-health"),
    "router-seq-aux-decomposition": partial(
        _metric_alias, LayerRouterSeqAuxDecompositionMetric, "router-seq-aux-decomposition"
    ),
    "router-routing-balance": partial(
        _metric_alias, LayerRouterRoutingBalanceMetric, "router-routing-balance"
    ),
    "logging-rank-local-expert-0-relu-health": LoggingRankLocalExpert0ReluHealthMetric,
    "loss-by-modality": ModalityLossMetric,
}

for _metric_type in (
    _ParameterL2NormMetric,
    _WgradL2NormMetric,
    _WgradAbsoluteMaximumMetric,
    _WgradZeroFractionMetric,
):
    _TENSOR_METRIC_FACTORIES[_metric_type.name] = _metric_type
    _TENSOR_METRIC_FACTORIES[f"{_metric_type.name}-individual"] = partial(
        _metric_type, individual=True
    )

_TENSOR_METRIC_FACTORIES["gdp-g-stats"] = lambda: LayerGDPHeadStatisticMetric(
    name="gdp-g-stats", source_kind="gdp_g"
)
_TENSOR_METRIC_FACTORIES["gdp-beta-stats"] = lambda: LayerGDPHeadStatisticMetric(
    name="gdp-beta-stats", source_kind="gdp_beta"
)

_OPTIMIZER_SOURCE_KINDS = frozenset({"parameter", "wgrad", "embedding_wgrad_row_squared_l2"})
_EXPERT_RELU_SOURCE_KINDS = frozenset({"expert_pre_relu", "expert_post_relu"})
_GDP_SOURCE_KINDS = frozenset({"gdp_g", "gdp_beta"})
_FORWARD_BACKWARD_SOURCE_KINDS = frozenset(
    {
        "data_source_loss",
        "residual_summary",
        "residual_summary_boundary",
        "dresidual_summary",
        "residual_summary_by_token_type",
        "dresidual_summary_by_token_type",
        "output_logits",
        "mtp_logits",
        "router_logits",
        "router_scores",
        "router_diagnostics",
        "modality_loss",
        "expert_pre_relu",
        "expert_post_relu",
        "moe_contributions",
        *_GDP_SOURCE_KINDS,
    }
)
_SUPPORTED_SOURCE_KINDS = _OPTIMIZER_SOURCE_KINDS | _FORWARD_BACKWARD_SOURCE_KINDS


def parse_tensor_metric_specs(
    specifications: Sequence[str],
    data_source_names: Sequence[str] = (),
    *,
    wgrad_small_thresholds: Sequence[float] = (),
    wgrad_relative_small_thresholds: Sequence[float] = (0.01,),
) -> tuple[ScheduledMetric, ...]:
    """Parse configured tensor metrics in ``NAME:INTERVAL`` form.

    Args:
        specifications: Metric names paired with positive iteration intervals.
        data_source_names: Stable labels in configured source-ID order.
        wgrad_small_thresholds: Explicit absolute magnitude cutoffs (inclusive).
        wgrad_relative_small_thresholds: Fractions of each complete logical tensor's RMS.

    Returns:
        Scheduled metric instances in configuration order.

    Raises:
        ValueError: If a specification is malformed, duplicated, unknown, or has an invalid
            interval.
    """
    scheduled_metrics = []
    configured_names = set()
    factories = dict(_TENSOR_METRIC_FACTORIES)
    factories[DataSourceLossMetric.name] = partial(DataSourceLossMetric, data_source_names)
    for relative, thresholds in (
        (False, wgrad_small_thresholds),
        (True, wgrad_relative_small_thresholds),
    ):
        name = 'wgrad-relative-small-fraction' if relative else 'wgrad-small-fraction'
        for individual in (False, True):
            factories[name + ('-individual' if individual else '')] = partial(
                _WgradSmallFractionMetric, thresholds, relative=relative, individual=individual
            )
    available_names = ", ".join(sorted(factories))
    for specification in specifications:
        name, separator, interval_text = specification.strip().rpartition(":")
        if not separator or not name or not interval_text:
            raise ValueError(
                f"Invalid tensor metric specification {specification!r}; expected NAME:INTERVAL."
            )
        if name not in factories:
            raise ValueError(
                f"Unknown tensor metric {name!r}; available tensor metrics: {available_names}."
            )
        if name in configured_names:
            raise ValueError(f"Tensor metric {name!r} was configured more than once.")
        try:
            interval = int(interval_text)
        except ValueError as error:
            raise ValueError(
                f"Tensor metric {name!r} requires an integer interval, got {interval_text!r}."
            ) from error
        if interval <= 0:
            raise ValueError(f"Tensor metric {name!r} requires a positive interval.")
        configured_names.add(name)
        metric = factories[name]()
        metric.name = name
        scheduled_metrics.append(ScheduledMetric(metric, interval))
    return tuple(scheduled_metrics)


def build_tensor_metric_observer(
    specifications: Sequence[str],
    result_sink: MetricResultSink | None = None,
    iterations: Sequence[int] = (),
    data_source_names: Sequence[str] = (),
    *,
    wgrad_small_thresholds: Sequence[float] = (),
    wgrad_relative_small_thresholds: Sequence[float] = (0.01,),
) -> "TrainingTensorMetricObserver | None":
    """Build the training observer selected by tensor metric CLI specifications.

    Args:
        specifications: Metric names paired with intervals.
        result_sink: Optional result consumer receiving the one-based reporting iteration. The
            default writes existing scalar logging sinks.
        iterations: Optional one-based iterations added to the metric intervals.
        data_source_names: Stable labels in configured source-ID order.
        wgrad_small_thresholds: Explicit absolute magnitude cutoffs (inclusive).
        wgrad_relative_small_thresholds: Fractions of each complete logical tensor's RMS.

    Returns:
        Configured observer, or ``None`` when no metrics were requested.
    """
    scheduled_metrics = parse_tensor_metric_specs(
        specifications,
        data_source_names,
        wgrad_small_thresholds=wgrad_small_thresholds,
        wgrad_relative_small_thresholds=wgrad_relative_small_thresholds,
    )
    if not scheduled_metrics:
        return None
    return TrainingTensorMetricObserver(
        scheduled_metrics, result_sink=result_sink, iterations=iterations
    )


def _embedding_vocab_size() -> int | None:
    """Resolve an optional vocabulary limit; otherwise use the embedding's logical rows."""
    size = getattr(get_args(), "tensor_metrics_embedding_vocab_size", None)
    if size is None:
        try:
            size = get_tokenizer().vocab_size
        except AssertionError:
            # MIMO calls set_global_variables(..., build_tokenizer=False), so a missing
            # global tokenizer is expected. Infer the logical row count from the embedding
            # metadata instead; do not assume get_tokenizer() is available in this path.
            return None
    if size <= 0:
        raise ValueError("--tensor-metrics-embedding-vocab-size must be positive.")
    return size


class TrainingTensorMetricObserver:
    """Observe short-lived forward tensors and optimizer tensors during a training iteration.

    Forward tensors are prepared while available and only compact prepared states survive until the
    pre-step commit. A metric using a forward source must therefore make its prepared state
    independent of the observed tensor; partial CUDA graphs may reuse an output buffer after the
    callback returns. Parameter metrics observe optimizer/master values when locally available,
    and wgrad metrics observe finalized ``main_grad`` or ``grad`` values at that commit point.
    Ordinary, Distributed, LayerWise, and Chained optimizer storage is represented explicitly
    across tensor, expert, data, and expert-data parallel axes. GTP-rematerialized parameter and
    wgrad shards have physical padding removed by the optimizer source. Pipeline parallelism,
    full-iteration CUDA graphs, and FSDP parameter storage remain unsupported.

    Args:
        scheduled_metrics: Metrics paired with independent iteration intervals. Every metric must
            explicitly declare at least one source kind supported by this observer.
        result_sink: Optional result consumer receiving the one-based reporting iteration. The
            default writes TensorBoard or WandB and falls back to stdout on the last rank.
    """

    def __init__(
        self,
        scheduled_metrics: Sequence[ScheduledMetric],
        result_sink: MetricResultSink | None = None,
        iterations: Sequence[int] = (),
    ) -> None:
        self.scheduled_metrics = tuple(scheduled_metrics)
        if any(iteration <= 0 for iteration in iterations):
            raise ValueError("Tensor metric iterations must be positive and one-based.")
        self._iterations = frozenset(iterations)
        self._result_sink = write_partitioned_scalar_results if result_sink is None else result_sink
        self._validate_metric_sources(self.scheduled_metrics)
        self._embedding_vocab_size = (
            _embedding_vocab_size()
            if any(isinstance(s.metric, EmbeddingWgradRowL2Metric) for s in self.scheduled_metrics)
            else None
        )
        self._parameter_names: CanonicalParameterNameMap | None = None
        self._parameter_names_key: tuple[tuple[int, ...], int, int] | None = None
        self._optimizer_parameter_manifest: tuple[_OptimizerParameterManifestEntry, ...] | None = (
            None
        )
        self._optimizer_parameter_manifest_key: tuple[int, int] | None = None
        self._module_names: dict[int, str] | None = None
        self._module_names_key: tuple[int, ...] | None = None
        self._forward_observation_active = False
        self._prepared_forward_values: dict[int, list[MetricTensor]] | None = None
        self._prepared_forward_iteration: int | None = None

    @contextmanager
    def observe_forward_backward(
        self,
        *,
        model: Sequence[torch.nn.Module],
        iteration: int | None,
        pg_collection: ProcessGroupCollection,
        force: bool = False,
    ) -> Iterator[None]:
        """Observe forward tensors produced during one forward-backward execution.

        Args:
            model: Wrapped model chunks participating in the training step.
            iteration: Zero-based training iteration.
            pg_collection: Concrete model process groups.
            force: Collect all configured metrics on this attempt regardless of their schedules.
        """
        if iteration is None:
            raise ValueError("Configured tensor metrics require a training iteration.")
        due_metrics = self._due_metrics(iteration, force=force)
        forward_metrics = tuple(
            scheduled
            for scheduled in due_metrics
            if scheduled.metric.source_kinds & _FORWARD_BACKWARD_SOURCE_KINDS
        )
        if not forward_metrics:
            yield
            return
        _validate_supported_topology(pg_collection)
        if self._forward_observation_active:
            raise RuntimeError("Tensor metric forward observation scopes must not be nested.")

        source_kinds = frozenset().union(
            *(
                scheduled.metric.source_kinds & _FORWARD_BACKWARD_SOURCE_KINDS
                for scheduled in forward_metrics
            )
        )
        _validate_forward_observation_model(model, source_kinds)
        executor = TensorMetricExecutor(_tensor_metric_process_groups(pg_collection))
        module_names = self._get_module_names(model)
        prepared_values = {id(scheduled.metric): [] for scheduled in forward_metrics}
        self._prepared_forward_values = prepared_values
        self._prepared_forward_iteration = iteration
        self._forward_observation_active = True

        def observe_forward_tensor(
            owner: object,
            name: str,
            source_kind: str,
            tensor: torch.Tensor,
            tp_shard_dim: int | None,
        ) -> None:
            """Prepare one forward tensor for every active metric."""
            owner_name = module_names.get(id(owner))
            if owner_name is None:
                raise ValueError(
                    f"Cannot resolve a canonical tensor metric name for {type(owner).__name__}."
                )
            site_name = f"{owner_name}.{name}" if owner_name else name
            site = MetricSite(site_name, source_kind)
            value = MetricTensor(
                tensor=tensor,
                sites=(site,),
                rank_relations=(
                    _gdp_forward_rank_relations(
                        pg_collection,
                        chunkwise_cp=getattr(owner, "chunkwise_context_parallel", False),
                    )
                    if source_kind in _GDP_SOURCE_KINDS
                    else (
                        ()
                        if source_kind in _EXPERT_RELU_SOURCE_KINDS
                        else _forward_rank_relations(tp_shard_dim, pg_collection)
                    )
                ),
            )
            # Deliberately untimed: Megatron timers synchronize the device when they start and
            # stop, which would serialize the forward-backward pass once per observed tensor. The
            # commit in __call__ owns the "tensor-metrics" timer, where one synchronization is
            # affordable.
            with get_nvtx_range()("tensor-metrics"):
                for scheduled in forward_metrics:
                    prepared_values[id(scheduled.metric)].extend(
                        executor.prepare(scheduled.metric, (value,))
                    )

        source_names = next(
            (
                scheduled.metric.source_names
                for scheduled in forward_metrics
                if isinstance(scheduled.metric, DataSourceLossMetric)
            ),
            (),
        )
        try:
            with (
                capture_tensor_observations(observe_forward_tensor, source_kinds),
                capture_data_source_metrics(source_names, pg_collection),
            ):
                yield
        except BaseException:
            self._prepared_forward_values = None
            self._prepared_forward_iteration = None
            raise
        finally:
            self._forward_observation_active = False

    @torch.no_grad()
    def __call__(
        self,
        *,
        model: Sequence[torch.nn.Module],
        optimizer: MegatronOptimizer,
        iteration: int | None,
        pg_collection: ProcessGroupCollection,
        force: bool = False,
    ) -> None:
        """Run metrics due on this training iteration.

        Args:
            model: Wrapped model chunks participating in the training step.
            optimizer: Optimizer exposing locally valid parameter and finalized-wgrad storage.
            iteration: Zero-based training iteration.
            pg_collection: Concrete model process groups.
            force: Run all configured metrics. Forward metrics must also have been collected
                with ``force=True`` during the accepted forward-backward attempt.

        Raises:
            ValueError: If the iteration or process groups are unavailable.
            NotImplementedError: If a requested source or model topology is not yet supported.
        """
        if iteration is None:
            raise ValueError("Configured tensor metrics require a training iteration.")
        due_metrics = self._due_metrics(iteration, force=force)
        if not due_metrics:
            return
        forward_metrics = tuple(
            scheduled
            for scheduled in due_metrics
            if scheduled.metric.source_kinds & _FORWARD_BACKWARD_SOURCE_KINDS
        )
        _validate_supported_topology(pg_collection)
        if forward_metrics and self._prepared_forward_iteration != iteration:
            raise RuntimeError(
                "Forward tensor metrics require observe_forward_backward() around the "
                "forward-backward execution."
            )

        try:
            with get_nvtx_range()("tensor-metrics", time=True):
                optimizer_metrics = tuple(
                    scheduled
                    for scheduled in due_metrics
                    if scheduled.metric.source_kinds & _OPTIMIZER_SOURCE_KINDS
                )
                executor = TensorMetricExecutor(_tensor_metric_process_groups(pg_collection))
                prepared_optimizer_values = {}
                if optimizer_metrics:
                    parameter_names = self._get_parameter_names(model, pg_collection)
                    manifest = self._get_optimizer_parameter_manifest(
                        parameter_names, optimizer, pg_collection
                    )
                    metrics = tuple(scheduled.metric for scheduled in optimizer_metrics)
                    batches = partial(
                        _optimizer_metric_tensor_batches,
                        parameter_names,
                        optimizer,
                        pg_collection,
                        metrics,
                        manifest,
                        embedding_vocab_size=self._embedding_vocab_size,
                    )
                    prepared_optimizer_values = _prepare_optimizer_metric_values(
                        executor, metrics, batches(), replay_batches=batches
                    )
                prepared_forward_values = self._prepared_forward_values or {}
                for scheduled in due_metrics:
                    prepared_values = list(prepared_forward_values.get(id(scheduled.metric), ()))
                    prepared_values.extend(prepared_optimizer_values.get(id(scheduled.metric), ()))
                    results = executor.complete(
                        scheduled.metric, scheduled.metric.start(prepared_values)
                    )
                    self._result_sink(scheduled.metric, results, iteration + 1)
        finally:
            self._prepared_forward_values = None
            self._prepared_forward_iteration = None

    def has_due_metrics(self, iteration: int | None) -> bool:
        """Whether any configured metric runs on ``iteration``.

        Lets callers exempt the (potentially long, collective) metric commit from step-level
        hang-detection timeouts only on the iterations where it actually runs.
        """
        if iteration is None:
            return False
        return bool(self._due_metrics(iteration))

    def _due_metrics(self, iteration: int, *, force: bool = False) -> tuple[ScheduledMetric, ...]:
        """Return metrics due on ``iteration`` in configuration order."""
        scheduled_metrics = (
            self.scheduled_metrics
            if force or iteration + 1 in self._iterations
            else tuple(
                scheduled for scheduled in self.scheduled_metrics if scheduled.is_due(iteration)
            )
        )
        return tuple(
            scheduled for scheduled in scheduled_metrics if scheduled.metric.is_active_on_rank()
        )

    @staticmethod
    def _validate_metric_sources(scheduled_metrics: Sequence[ScheduledMetric]) -> None:
        undeclared_sources = {
            scheduled.metric.name or type(scheduled.metric).__name__
            for scheduled in scheduled_metrics
            if not scheduled.metric.source_kinds
        }
        if undeclared_sources:
            raise ValueError(
                "Metrics configured with the training tensor metric observer must explicitly "
                f"declare source_kinds; missing declarations: {sorted(undeclared_sources)}."
            )
        unsupported_sources = {
            source_kind
            for scheduled in scheduled_metrics
            for source_kind in scheduled.metric.source_kinds
            if source_kind not in _SUPPORTED_SOURCE_KINDS
        }
        if unsupported_sources:
            raise NotImplementedError(
                "The training tensor metric observer does not support sources "
                f"{sorted(unsupported_sources)}."
            )

    def _get_module_names(self, model: Sequence[torch.nn.Module]) -> dict[int, str]:
        """Return cached canonical module names used by forward observation sites."""
        key = tuple(id(model_chunk) for model_chunk in model)
        if self._module_names is None or self._module_names_key != key:
            names = {}
            use_chunk_prefix = len(model) > 1
            for chunk_index, model_chunk in enumerate(model):
                unwrapped = unwrap_model(model_chunk)
                chunk_prefix = f"model_chunk{chunk_index}." if use_chunk_prefix else ""
                for module_name, module in unwrapped.named_modules(remove_duplicate=False):
                    names.setdefault(id(module), f"{chunk_prefix}{module_name}")
            self._module_names = names
            self._module_names_key = key
        return self._module_names

    def _get_parameter_names(
        self, model: Sequence[torch.nn.Module], pg_collection: ProcessGroupCollection
    ) -> CanonicalParameterNameMap:
        """Return the cached canonical parameter names for this model topology."""
        ep_group = getattr(pg_collection, "ep", None)
        ep_size = ep_group.size() if ep_group is not None else 1
        ep_rank = ep_group.rank() if ep_group is not None and ep_size > 1 else 0
        key = (tuple(id(model_chunk) for model_chunk in model), ep_rank, ep_size)
        if self._parameter_names is None or self._parameter_names_key != key:
            self._parameter_names = CanonicalParameterNameMap(
                model, expert_parallel_rank=ep_rank, expert_parallel_size=ep_size
            )
            self._parameter_names_key = key
        return self._parameter_names

    def _get_optimizer_parameter_manifest(
        self,
        parameter_names: CanonicalParameterNameMap,
        optimizer: MegatronOptimizer,
        pg_collection: ProcessGroupCollection,
    ) -> tuple[_OptimizerParameterManifestEntry, ...]:
        """Return cached rank-symmetric optimizer parameter metadata."""
        key = (id(parameter_names), id(optimizer))
        if (
            self._optimizer_parameter_manifest is None
            or self._optimizer_parameter_manifest_key != key
        ):
            self._optimizer_parameter_manifest = _build_optimizer_parameter_manifest(
                parameter_names, optimizer, pg_collection
            )
            self._optimizer_parameter_manifest_key = key
        return self._optimizer_parameter_manifest


def _forward_rank_relations(
    tp_shard_dim: int | None, pg_collection: ProcessGroupCollection
) -> tuple[RankRelation, ...]:
    """Describe a forward tensor over tensor, GTP, and data/context parallel ranks."""
    tp_group = getattr(pg_collection, "tp", None)
    gtp_group = getattr(pg_collection, "gtp_remat", None)
    dp_cp_group = getattr(pg_collection, "dp_cp", None)
    if tp_group is None or gtp_group is None or dp_cp_group is None:
        raise ValueError("Forward tensor metrics require tp, gtp_remat, and dp_cp process groups.")
    tp_placement = (
        Shard(tp_shard_dim) if tp_shard_dim is not None and tp_group.size() > 1 else Replica()
    )
    # Like DP peers, GTP peers execute distinct microbatches even though their activations are not
    # tensor-dimension shards of one another.
    gtp_placement = Shard(None) if gtp_group.size() > 1 else Replica()
    dp_placement = Shard(None) if dp_cp_group.size() > 1 else Replica()
    return (
        RankRelation("tp", tp_placement),
        RankRelation("gtp", gtp_placement),
        RankRelation("dp", dp_placement),
    )


def _gdp_forward_rank_relations(
    pg_collection: ProcessGroupCollection, *, chunkwise_cp: bool = False
) -> tuple[RankRelation, ...]:
    """CP shards heads in headwise mode, but token populations in chunkwise mode."""
    cp_group = getattr(pg_collection, "cp", None)
    tp_group = getattr(pg_collection, "tp", None)
    gtp_group = getattr(pg_collection, "gtp_remat", None)
    dp_group = getattr(pg_collection, "dp", None)
    if cp_group is None or tp_group is None or gtp_group is None or dp_group is None:
        raise ValueError("GDP tensor metrics require cp, tp, gtp_remat, and dp process groups.")
    return (
        RankRelation(
            "gdp_cp", Shard(None if chunkwise_cp else -1) if cp_group.size() > 1 else Replica()
        ),
        RankRelation("tp", Shard(-1) if tp_group.size() > 1 else Replica()),
        RankRelation("gtp", Shard(None) if gtp_group.size() > 1 else Replica()),
        RankRelation("gdp_dp", Shard(None) if dp_group.size() > 1 else Replica()),
    )


def _validate_forward_observation_model(
    model: Sequence[torch.nn.Module], source_kinds: frozenset[str]
) -> None:
    """Reject graph modes that cannot produce reliable Python observation notifications."""
    for model_chunk in model:
        config = getattr(unwrap_model(model_chunk), "config", None)
        cuda_graph_impl = getattr(config, "cuda_graph_impl", "none")
        if cuda_graph_impl == "full_iteration":
            raise NotImplementedError(
                "Forward tensor metrics do not yet support full-iteration CUDA graphs."
            )
        if (
            "moe_contributions" in source_kinds
            and getattr(config, "num_moe_experts", None) is not None
        ):
            validate_moe_contribution_config(config)
            scopes = tuple(getattr(config, "cuda_graph_modules", ()))
            if cuda_graph_impl != "none" and (
                not scopes
                or any(
                    s
                    in {
                        CudaGraphModule.moe,
                        CudaGraphModule.moe_router,
                        CudaGraphModule.moe_preprocess,
                    }
                    for s in scopes
                )
            ):
                raise NotImplementedError(
                    "moe-contributions requires an eager MoE router, preprocess and combine."
                )
        has_expert_relu_sources = (
            not source_kinds.isdisjoint(_EXPERT_RELU_SOURCE_KINDS)
            and getattr(config, "num_moe_experts", None) is not None
        )
        if has_expert_relu_sources:
            if not getattr(config, "moe_grouped_gemm", False):
                raise NotImplementedError(
                    "Expert ReLU tensor metrics currently require --moe-grouped-gemm."
                )
            if getattr(config, "use_transformer_engine_op_fuser", False):
                raise NotImplementedError(
                    "Expert ReLU tensor metrics do not yet support the TE op-fuser."
                )
            if getattr(config, "moe_use_grouped_tensor", False):
                raise NotImplementedError(
                    "Expert ReLU tensor metrics do not yet support --moe-use-grouped-tensor."
                )
        if cuda_graph_impl not in {"local", "transformer_engine"}:
            continue
        graph_modules = tuple(getattr(config, "cuda_graph_modules", ()))
        router_sources = {"router_logits", "router_scores", "router_diagnostics"}
        router_is_graphed = not graph_modules or any(
            graph_module
            in {CudaGraphModule.moe, CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess}
            for graph_module in graph_modules
        )
        if not source_kinds.isdisjoint(router_sources) and router_is_graphed:
            raise NotImplementedError(
                "Router tensor metrics require an eager MoE router; disable moe, "
                "moe_router, and moe_preprocess CUDA graph modules for now."
            )
        gdp_is_graphed = not graph_modules or CudaGraphModule.mamba in graph_modules
        if not source_kinds.isdisjoint(_GDP_SOURCE_KINDS) and gdp_is_graphed:
            raise NotImplementedError(
                "GDP gating tensor metrics require eager GDP layers; exclude mamba from "
                "CUDA graph modules for now."
            )
        expert_activation_is_graphed = has_expert_relu_sources and (
            not graph_modules or CudaGraphModule.moe in graph_modules
        )
        if expert_activation_is_graphed:
            raise NotImplementedError(
                "Expert ReLU tensor metrics require eager grouped experts; exclude moe from "
                "CUDA graph modules for now."
            )


def _tensor_metric_process_groups(
    pg_collection: ProcessGroupCollection,
) -> dict[str, torch.distributed.ProcessGroup]:
    """Bind abstract metric axes to caller-supplied process groups."""
    groups = {}
    for axis, attribute in (
        ("tp", "tp"),
        ("gdp_cp", "cp"),
        ("expert_tp", "expt_tp"),
        ("ep", "ep"),
        ("gtp", "gtp_remat"),
        ("expert_gtp", "expt_gtp_remat"),
        ("dp", "dp_cp"),
        ("gdp_dp", "dp"),
        ("expert_dp", "expt_dp"),
    ):
        group = getattr(pg_collection, attribute, None)
        if group is not None:
            groups[axis] = group
    return groups


def _validate_supported_topology(pg_collection: ProcessGroupCollection) -> None:
    pp_group = getattr(pg_collection, "pp", None)
    if pp_group is not None and pp_group.size() > 1:
        raise NotImplementedError(
            "The training tensor metric observer does not yet support pipeline parallelism."
        )
