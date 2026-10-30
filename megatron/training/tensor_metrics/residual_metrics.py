# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor metrics comparing residual-stream accumulators and layer contributions."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from .core import (
    AllReduce,
    CollectiveRequest,
    CollectiveStage,
    MetricResult,
    MetricSite,
    MetricStep,
    MetricTensor,
    RankRelation,
    Replica,
    Shard,
    TensorMetric,
)
from .definitions import LayerL2NormMetric

__all__ = [
    "LayerDResidualSummaryMetric",
    "LayerDResidualSummaryByTokenTypeMetric",
    "LayerResidualSummaryMetric",
    "LayerResidualSummaryByTokenTypeMetric",
]


_SUMMARY_POPULATIONS = ("accumulator", "contribution", "correlation")
_BOUNDARY_POPULATIONS = ("activation",)
_BOUNDARY_SOURCE_KIND = "residual_summary_boundary"
_BOUNDARY_SITE_LABELS = {
    "last_layer": "last-layer",
    "last_layer_widereadout": "last-layer-wide-readout",
}
_SUMMARY_STATISTICS = ("rms", "p75", "p90", "p95", "max")
_SUMMARY_QUANTILES = (0.75, 0.9, 0.95)
_CORRELATION_HISTOGRAM_BINS = 20
_HistogramLayout = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _ResidualSummaryState:
    value: tuple[MetricTensor, ...] | MetricTensor
    minima: MetricTensor
    maxima: MetricTensor
    layout: _HistogramLayout | None
    remaining_axes: tuple[str, ...]
    label: str
    populations: tuple[str, ...]


def _population_axes(value: MetricTensor) -> tuple[str, ...]:
    return tuple(
        relation.axis
        for relation in value.rank_relations
        if isinstance(relation.placement, Shard)
    )


def _local_extrema(
    values: Sequence[MetricTensor], population_names: tuple[str, ...]
) -> tuple[MetricTensor, MetricTensor]:
    template = values[0]
    sites = tuple(site for value in values for site in value.sites)
    minima = torch.full(
        (len(population_names),), float("inf"), device=template.tensor.device
    )
    maxima = torch.full(
        (len(population_names),), float("-inf"), device=template.tensor.device
    )
    for value in values:
        tensor = value.tensor
        if not tensor.shape[1]:
            continue
        # All ranks must derive the same finite histogram layout, even when only one
        # rank observes NaN/Inf. The additive square sum propagates nonfinite values.
        finite = torch.isfinite(tensor)
        norm_population_count = len(population_names) - int(
            population_names[-1] == "correlation"
        )
        positive_norms = tensor[:norm_population_count].masked_fill(
            ~finite[:norm_population_count] | (tensor[:norm_population_count] <= 0),
            float("inf"),
        )
        minima[:norm_population_count] = torch.minimum(
            minima[:norm_population_count], positive_norms.amin(dim=1)
        )
        if norm_population_count != len(population_names):
            minima[-1] = torch.minimum(
                minima[-1], tensor[-1].masked_fill(~finite[-1], float("inf")).amin()
            )
        maxima = torch.maximum(
            maxima, tensor.masked_fill(~finite, float("-inf")).amax(dim=1)
        )
    return (
        MetricTensor(minima, sites, template.rank_relations),
        MetricTensor(maxima, sites, template.rank_relations),
    )


def _histogram_layout(
    minima: torch.Tensor, maxima: torch.Tensor, populations: tuple[str, ...]
) -> _HistogramLayout:
    layout = []
    for population, minimum, maximum in zip(populations, minima.tolist(), maxima.tolist()):
        if population == "correlation":
            layout.append((0, _CORRELATION_HISTOGRAM_BINS))
        elif not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= 0:
            layout.append((0, 2))
        else:
            minimum_exponent = math.floor(math.log2(minimum))
            maximum_exponent = math.ceil(math.log2(maximum))
            # Bin zero is an exact point mass at zero; the rest are logarithmic bins.
            layout.append((minimum_exponent, 1 + max(1, maximum_exponent - minimum_exponent)))
    return tuple(layout)


def _histogram_indices(
    population: torch.Tensor,
    population_name: str,
    minimum_exponent: int,
    bin_count: int,
) -> torch.Tensor:
    if population_name != "correlation":
        positive = population > 0
        indices = (
            torch.floor(
                torch.log2(torch.where(positive, population, torch.ones_like(population)))
            )
            .to(torch.int64)
            .sub(minimum_exponent - 1)
            .clamp_(1, bin_count - 1)
        )
        return indices.masked_fill(~positive, 0)
    return torch.floor(
        (population + 1.0) * (_CORRELATION_HISTOGRAM_BINS / 2.0)
    ).to(torch.int64).clamp_(0, _CORRELATION_HISTOGRAM_BINS - 1)


def _local_additive_state(
    values: Sequence[MetricTensor], layout: _HistogramLayout, populations: tuple[str, ...]
) -> MetricTensor:
    template = values[0]
    sites = tuple(site for value in values for site in value.sites)
    fields = []
    for population_index, (population_name, (minimum_exponent, bin_count)) in enumerate(
        zip(populations, layout)
    ):
        square_sum = torch.zeros((), dtype=torch.float64, device=template.tensor.device)
        count = torch.zeros((), dtype=torch.float64, device=template.tensor.device)
        histogram = torch.zeros(
            bin_count, dtype=torch.float64, device=template.tensor.device
        )
        for value in values:
            population = value.tensor[population_index]
            square_sum += population.double().square().sum()
            count += population.numel()
            if population.numel():
                indices = _histogram_indices(
                    population[torch.isfinite(population)],
                    population_name,
                    minimum_exponent,
                    bin_count,
                )
                histogram += torch.bincount(indices, minlength=bin_count).double()
        fields.extend((square_sum.unsqueeze(0), count.unsqueeze(0), histogram))
    return MetricTensor(torch.cat(fields), sites, template.rank_relations)


def _summarize(
    additive_state: torch.Tensor,
    minima: torch.Tensor,
    maxima: torch.Tensor,
    layout: _HistogramLayout,
    populations: tuple[str, ...],
) -> torch.Tensor:
    summaries = []
    offset = 0
    for population_index, (population_name, (minimum_exponent, bin_count)) in enumerate(
        zip(populations, layout)
    ):
        square_sum = additive_state[offset]
        count = additive_state[offset + 1]
        histogram = additive_state[offset + 2 : offset + 2 + bin_count]
        offset += 2 + bin_count
        if not torch.isfinite(square_sum).item():
            # A nonfinite observation invalidates this population, rather than being
            # silently dropped or crashing metric collection. SUM carries it to every rank.
            summaries.append(additive_state.new_full((len(_SUMMARY_STATISTICS),), float("nan")))
            continue
        nonempty = count > 0
        rms = torch.where(
            nonempty,
            (square_sum / count.clamp_min(1)).sqrt(),
            torch.zeros_like(square_sum),
        )
        if count.item():
            quantiles = histogram.new_tensor(_SUMMARY_QUANTILES)
            target_ranks = (quantiles * (count - 1)).ceil() + 1
            indices = (
                (histogram.cumsum(dim=0).unsqueeze(0) >= target_ranks.unsqueeze(1))
                .to(torch.int64)
                .argmax(dim=1)
            )
            if population_name != "correlation":
                quantile_values = torch.exp2(
                    histogram.new_tensor(minimum_exponent) + indices
                )
                quantile_values.masked_fill_(indices == 0, 0)
                quantile_values = torch.minimum(
                    quantile_values, maxima[population_index]
                )
            else:
                bin_width = 2.0 / _CORRELATION_HISTOGRAM_BINS
                quantile_values = -1.0 + (indices + 1) * bin_width
                quantile_values = quantile_values.clamp(
                    min=minima[population_index].item(),
                    max=maxima[population_index].item(),
                )
            maximum = maxima[population_index]
        else:
            quantile_values = histogram.new_zeros(len(_SUMMARY_QUANTILES))
            maximum = torch.zeros_like(count)
        summaries.append(
            torch.cat((rms.unsqueeze(0), quantile_values, maximum.unsqueeze(0)))
        )
    return torch.stack(summaries).float()


def _site_boundary_label(site: MetricSite) -> str | None:
    """Return the stable label for a decoder residual boundary observation."""
    if site.kind != _BOUNDARY_SOURCE_KIND:
        return None
    site_name = site.name.rsplit(".", 1)[-1]
    return _BOUNDARY_SITE_LABELS.get(site_name)


class _LayerResidualSummaryMetric(TensorMetric):
    """Summarize residual populations with compact distributed histograms."""

    def accepts(self, site: MetricSite) -> bool:
        """Select numbered-layer summaries and declared decoder boundaries."""
        return (
            super().accepts(site)
            and (
                LayerL2NormMetric._site_layer_label(site) is not None
                or _site_boundary_label(site) is not None
            )
        )

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Retain per-token populations until globally shared histogram ranges are known."""
        prepared = []
        for value in values:
            if value.is_placeholder:
                raise ValueError(
                    "Residual summaries do not support placeholder observations."
                )
            boundary_labels = tuple(_site_boundary_label(site) for site in value.sites)
            is_boundary = boundary_labels[0] is not None
            if any((label is not None) != is_boundary for label in boundary_labels[1:]):
                raise ValueError("A residual summary tensor cannot mix layer and boundary sites.")
            populations = _BOUNDARY_POPULATIONS if is_boundary else _SUMMARY_POPULATIONS
            if value.tensor.ndim != 2 or value.tensor.shape[0] != len(populations):
                raise ValueError(
                    f"Residual summaries require shape [{len(populations)}, tokens] for "
                    f"populations {populations}, got {tuple(value.tensor.shape)}."
                )
            relations = []
            for relation in value.rank_relations:
                if isinstance(relation.placement, Replica):
                    relations.append(relation)
                elif isinstance(relation.placement, Shard):
                    if relation.placement.dim not in {None, 1}:
                        raise ValueError(
                            "Residual-summary populations may only be sharded over tokens."
                        )
                    relations.append(RankRelation(relation.axis, Shard(1)))
                else:
                    raise ValueError(
                        "Residual summaries require Replica or Shard placements."
                    )
            prepared.append(value.with_tensor(value.tensor.float(), relations))
        return prepared

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Start approximate summaries for layer populations and decoder boundaries."""
        layer_values = [
            value
            for value in values
            if all(_site_boundary_label(site) is None for site in value.sites)
        ]
        steps = [
            self._start_summary(reduction_values, label)
            for label, reduction_values in LayerL2NormMetric._reductions_by_layer(
                layer_values, include_global=True
            )
        ]
        for boundary_label in _BOUNDARY_SITE_LABELS.values():
            boundary_values = [
                value
                for value in values
                if all(_site_boundary_label(site) == boundary_label for site in value.sites)
            ]
            if boundary_values:
                steps.append(
                    self._start_summary(
                        boundary_values, boundary_label, populations=_BOUNDARY_POPULATIONS
                    )
                )
        return steps

    def resume(
        self, values: Sequence[MetricTensor], continuation: object
    ) -> MetricStep:
        """Continue the extrema or histogram reduction encoded by the state."""
        if not isinstance(continuation, _ResidualSummaryState):
            raise ValueError("Invalid residual-summary continuation.")
        if continuation.layout is None:
            if len(values) != 2:
                raise ValueError(
                    "A residual-summary range stage requires min/max completions."
                )
            return self._advance(
                replace(continuation, minima=values[0], maxima=values[1])
            )
        if len(values) != 1:
            raise ValueError(
                "A residual-summary histogram stage requires one completion."
            )
        return self._advance(replace(continuation, value=values[0]))

    def finalize_results(self, results: Sequence[MetricResult]) -> list[MetricResult]:
        """Expand every fused result into population/statistic scalar subkeys."""
        finalized = []
        for result in results:
            if result.label in _BOUNDARY_SITE_LABELS.values():
                expected_shape = (len(_BOUNDARY_POPULATIONS), len(_SUMMARY_STATISTICS))
                if result.tensor.shape != expected_shape:
                    raise ValueError(
                        f"{type(self).__name__} requires boundary results of shape "
                        f"{expected_shape}, got {tuple(result.tensor.shape)}."
                    )
                finalized.extend(
                    MetricResult(result.tensor[0, index], f"{result.label}/{statistic}")
                    for index, statistic in enumerate(_SUMMARY_STATISTICS)
                )
                continue

            expected_shape = (len(_SUMMARY_POPULATIONS), len(_SUMMARY_STATISTICS))
            if result.tensor.shape != expected_shape:
                raise ValueError(
                    f"{type(self).__name__} requires results of shape {expected_shape}, "
                    f"got {tuple(result.tensor.shape)}."
                )
            for population_index, population in enumerate(_SUMMARY_POPULATIONS):
                for statistic_index, statistic in enumerate(_SUMMARY_STATISTICS):
                    finalized.append(
                        MetricResult(
                            result.tensor[population_index, statistic_index],
                            f"{result.label}/{population}/{statistic}",
                        )
                    )
        return finalized

    def _start_summary(
        self,
        values: Sequence[MetricTensor],
        label: str,
        *,
        populations: tuple[str, ...] = _SUMMARY_POPULATIONS,
    ) -> MetricStep:
        relations = values[0].rank_relations
        if any(value.rank_relations != relations for value in values[1:]):
            raise ValueError(
                "A residual-summary result requires identical rank relations."
            )
        source_values = tuple(values)
        minima, maxima = _local_extrema(source_values, populations)
        return self._advance(
            _ResidualSummaryState(
                source_values,
                minima,
                maxima,
                None,
                _population_axes(source_values[0]),
                label,
                populations,
            )
        )

    def _advance(self, state: _ResidualSummaryState) -> MetricStep:
        if state.layout is None:
            assert isinstance(state.value, tuple), (
                "An extrema-stage residual summary must retain its source values."
            )
            if state.remaining_axes:
                axis = state.remaining_axes[0]
                return CollectiveStage(
                    (
                        CollectiveRequest(
                            state.minima,
                            axis,
                            AllReduce(torch.distributed.ReduceOp.MIN),
                        ),
                        CollectiveRequest(
                            state.maxima,
                            axis,
                            AllReduce(torch.distributed.ReduceOp.MAX),
                        ),
                    ),
                    replace(state, remaining_axes=state.remaining_axes[1:]),
                )
            layout = _histogram_layout(
                state.minima.tensor, state.maxima.tensor, state.populations
            )
            return self._advance(
                _ResidualSummaryState(
                    _local_additive_state(state.value, layout, state.populations),
                    state.minima,
                    state.maxima,
                    layout,
                    _population_axes(state.value[0]),
                    state.label,
                    state.populations,
                )
            )

        assert isinstance(state.value, MetricTensor), (
            "A histogram-stage residual summary must contain additive state."
        )
        if state.remaining_axes:
            return CollectiveStage(
                (
                    CollectiveRequest(
                        state.value,
                        state.remaining_axes[0],
                        AllReduce(torch.distributed.ReduceOp.SUM),
                    ),
                ),
                replace(state, remaining_axes=state.remaining_axes[1:]),
            )
        return MetricResult(
            _summarize(
                state.value.tensor,
                state.minima.tensor,
                state.maxima.tensor,
                state.layout,
                state.populations,
            ),
            state.label,
        )


class LayerResidualSummaryMetric(_LayerResidualSummaryMetric):
    """Report fused token-distribution summaries for residual activations."""

    name = "residual-summary"
    source_kinds = frozenset({"residual_summary", _BOUNDARY_SOURCE_KIND})


class LayerDResidualSummaryMetric(_LayerResidualSummaryMetric):
    """Report fused token-distribution summaries for residual gradients."""

    name = "dresidual-summary"
    source_kinds = frozenset({"dresidual_summary"})


_TOKEN_TYPE_SITE_MARKER = ".token_type["


def _site_token_type_label(site: MetricSite) -> str | None:
    """Return the token-type label encoded in a residual-summary site."""
    _, separator, suffix = site.name.rpartition(_TOKEN_TYPE_SITE_MARKER)
    if not separator or not suffix.endswith("]") or len(suffix) == 1:
        return None
    return suffix[:-1]


class _LayerResidualSummaryByTokenTypeMetric(_LayerResidualSummaryMetric):
    """Summarize compact token-type populations with distributed histograms."""

    def accepts(self, site: MetricSite) -> bool:
        return super().accepts(site) and _site_token_type_label(site) is not None

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        values_by_token_type: dict[str, list[MetricTensor]] = {}
        for value in values:
            labels = tuple(_site_token_type_label(site) for site in value.sites)
            token_type_label = labels[0]
            if token_type_label is None or any(
                label != token_type_label for label in labels[1:]
            ):
                raise ValueError(
                    "A token-type summary tensor must identify one token type."
                )
            values_by_token_type.setdefault(token_type_label, []).append(value)

        steps = []
        for token_type_label, token_type_values in values_by_token_type.items():
            for layer_label, reduction_values in LayerL2NormMetric._reductions_by_layer(
                token_type_values, include_global=True
            ):
                steps.append(
                    self._start_summary(
                        reduction_values, f"{layer_label}/token-type/{token_type_label}"
                    )
                )
        return steps


class LayerResidualSummaryByTokenTypeMetric(_LayerResidualSummaryByTokenTypeMetric):
    """Report fused residual-activation summaries for each token type."""

    name = "residual-summary-by-token-type"
    source_kinds = frozenset({"residual_summary_by_token_type"})


class LayerDResidualSummaryByTokenTypeMetric(_LayerResidualSummaryByTokenTypeMetric):
    """Report fused residual-gradient summaries for each token type."""

    name = "dresidual-summary-by-token-type"
    source_kinds = frozenset({"dresidual_summary_by_token_type"})
