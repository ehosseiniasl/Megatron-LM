# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-head statistics for Gated Delta Product gating tensors."""

from collections.abc import Sequence
from typing import Literal

import torch

from .core import (
    AllGather,
    AllReduce,
    CollectiveRequest,
    CollectiveStage,
    MetricResult,
    MetricSite,
    MetricStep,
    MetricTensor,
    RankRelation,
    Shard,
    TensorMetric,
)
from .definitions import LayerL2NormMetric, _accumulation_dtype

__all__ = ["LayerGDPHeadStatisticMetric"]


_BETA_HISTOGRAM_BINS = 1024
_BETA_QUANTILES = (0.05, 0.50, 0.95)
_BETA_QUANTILE_NAMES = ("p05", "p50", "p95")
_HEAD_AXIS_ORDER = ("gdp_cp", "tp")
_Continuation = tuple[tuple[str, ...], int, str, MetricTensor | None]


def _stage(
    value: MetricTensor,
    axis: str,
    collective: AllGather | AllReduce,
    continuation: _Continuation,
) -> CollectiveStage:
    return CollectiveStage(
        (CollectiveRequest(value, axis, collective),), continuation
    )


class LayerGDPHeadStatisticMetric(TensorMetric):
    """Compute decay mean/std and beta quantiles per update/head.

    Token populations are reduced over DP, GTP, and chunkwise CP. Disjoint
    heads are gathered over headwise CP and then TP to preserve global order.
    """

    source_kinds: frozenset[Literal["gdp_g", "gdp_beta"]]

    def __init__(self, name: str, source_kind: Literal["gdp_g", "gdp_beta"]) -> None:
        if source_kind not in {"gdp_g", "gdp_beta"}:
            raise ValueError(f"Unsupported GDP gate summary source {source_kind!r}.")
        self.name = name
        self.source_kinds = frozenset({source_kind})
        self._is_decay = source_kind == "gdp_g"

    def accepts(self, site: MetricSite) -> bool:
        return (
            super().accepts(site)
            and LayerL2NormMetric._site_layer_label(site) is not None
        )

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        prepared = []
        expected_ndim = 3 if self._is_decay else 4
        for value in values:
            tensor = value.tensor.detach()
            if tensor.ndim != expected_ndim:
                if self._is_decay:
                    raise ValueError("GDP decay requires [batch, token, head].")
                raise ValueError("GDP beta requires [batch, token, update, head].")
            if 0 in tensor.shape[2:]:
                raise ValueError(
                    "GDP gate summaries require nonempty head/update dimensions."
                )
            tensor = tensor.to(dtype=_accumulation_dtype(tensor.dtype))
            contribution = (
                self._decay_contribution(tensor)
                if self._is_decay
                else self._beta_contribution(tensor)
            )
            relations = []
            for relation in value.rank_relations:
                placement = relation.placement
                if isinstance(placement, Shard) and placement.dim is not None:
                    if placement.dim % tensor.ndim != tensor.ndim - 1:
                        raise ValueError(
                            "GDP head metrics only support sharding the head dimension."
                        )
                    placement = Shard(-1)
                relations.append(RankRelation(relation.axis, placement))
            prepared.append(value.with_tensor(contribution, relations))
        return prepared

    @staticmethod
    def _decay_contribution(tensor: torch.Tensor) -> torch.Tensor:
        gate_shape = tensor.shape[2:]
        count = tensor.shape[0] * tensor.shape[1]
        if count:
            # Centering avoids losing small gate variation next to a large FP32 mean.
            anchor = tensor[0, 0]
            variance, offset = torch.var_mean(tensor - anchor, dim=(0, 1), correction=0)
            mean = anchor.double() + offset.double()
        else:
            variance = mean = tensor.new_zeros(gate_shape)
        return torch.stack(
            (
                mean.new_full(gate_shape, count, dtype=torch.float64),
                mean.double() * count,
                variance.double() * count,
            )
        )

    @staticmethod
    def _beta_contribution(tensor: torch.Tensor) -> torch.Tensor:
        updates, heads = tensor.shape[2:]
        valid = torch.isfinite(tensor) & (tensor >= 0.0) & (tensor <= 1.0)
        indices = torch.where(valid, tensor, 0).mul(_BETA_HISTOGRAM_BINS).floor().long()
        indices.clamp_(0, _BETA_HISTOGRAM_BINS - 1)
        offsets = _BETA_HISTOGRAM_BINS * torch.arange(
            updates * heads, device=tensor.device
        ).reshape(updates, heads)
        histogram = torch.bincount(
            (indices + offsets)[valid],
            minlength=updates * heads * _BETA_HISTOGRAM_BINS,
        ).reshape(updates, heads, _BETA_HISTOGRAM_BINS)
        return torch.cat(
            ((~valid).sum(dim=(0, 1)).unsqueeze(0), histogram.movedim(-1, 0)), dim=0
        ).double()

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        values_by_layer: dict[str, list[MetricTensor]] = {}
        for value in values:
            layer = LayerL2NormMetric._layer_label(value)
            if layer is None:
                raise ValueError(
                    "A GDP head metric tensor must identify one logical layer."
                )
            values_by_layer.setdefault(layer, []).append(value)

        steps = []
        for layer in sorted(values_by_layer):
            layer_values = values_by_layer[layer]
            first = layer_values[0]
            tensors = torch.stack(tuple(value.tensor for value in layer_values))
            combined = (
                self._combine_decay(tensors)
                if self._is_decay
                else tensors.sum(dim=0)
            )
            value = first.with_tensor(combined)
            shards = {
                relation.axis: relation.placement.dim
                for relation in value.rank_relations
                if isinstance(relation.placement, Shard)
            }
            population_axes = tuple(axis for axis, dim in shards.items() if dim is None)
            head_shards = {axis for axis, dim in shards.items() if dim == -1}
            if not head_shards.issubset(_HEAD_AXIS_ORDER):
                raise ValueError(
                    "GDP head shards must use the GDP CP or TP metric axes."
                )
            head_axes = tuple(axis for axis in _HEAD_AXIS_ORDER if axis in head_shards)
            axes = population_axes + head_axes
            steps.append(self._advance(value, axes, len(population_axes), layer))
        return steps

    @staticmethod
    def _combine_decay(tensors: torch.Tensor) -> torch.Tensor:
        counts, totals, m2 = tensors.unbind(dim=1)
        count, total = counts.sum(dim=0), totals.sum(dim=0)
        mean = total / count.clamp_min(1)
        local_means = totals / counts.clamp_min(1)
        centered = (m2 + counts * (local_means - mean).square()).sum(dim=0)
        return torch.stack((count, total, centered))

    def resume(
        self, values: Sequence[MetricTensor], continuation: object
    ) -> MetricStep:
        valid_continuation = isinstance(continuation, tuple) and len(continuation) == 4
        if not valid_continuation or len(values) != 1:
            raise ValueError("Invalid GDP gate summary continuation.")
        axes, population_axes, layer, local_moments = continuation
        value = values[0]
        if local_moments is not None:
            count, total = value.tensor.unbind()
            local_count, local_total, local_m2 = local_moments.tensor.unbind()
            corrected_m2 = local_m2 + local_count * (
                local_total / local_count.clamp_min(1) - total / count.clamp_min(1)
            ).square()
            value = local_moments.with_tensor(
                torch.stack((local_count, local_total, corrected_m2))
            )
            return _stage(
                value,
                axes[0],
                AllReduce(),
                (axes[1:], population_axes - 1, layer, None),
            )
        return self._advance(value, axes, population_axes, layer)

    def _advance(
        self,
        value: MetricTensor,
        axes: tuple[str, ...],
        population_axes: int,
        layer: str,
    ) -> MetricStep:
        if not axes:
            return MetricResult(value.tensor, layer)
        axis = axes[0]
        if not population_axes:
            return _stage(value, axis, AllGather(dim=-1), (axes[1:], 0, layer, None))
        if not self._is_decay:
            return _stage(
                value,
                axis,
                AllReduce(),
                (axes[1:], population_axes - 1, layer, None),
            )
        return _stage(
            value.with_tensor(value.tensor[:2]),
            axis,
            AllReduce(),
            (axes, population_axes, layer, value),
        )

    def finalize_results(self, results: Sequence[MetricResult]) -> list[MetricResult]:
        finalize = self._finalize_decay if self._is_decay else self._finalize_beta
        return [metric for result in results for metric in finalize(result)]

    @staticmethod
    def _finalize_decay(result: MetricResult) -> list[MetricResult]:
        count, total, m2 = result.tensor.unbind()
        denominator = count.clamp_min(1)
        summaries = (
            ("mean", total / denominator),
            ("std", (m2 / denominator).clamp_min(0).sqrt()),
        )
        return [
            MetricResult(values[head], f"{result.label}/head_{head}/{name}")
            for head in range(total.shape[0])
            for name, values in summaries
        ]

    @staticmethod
    def _finalize_beta(result: MetricResult) -> list[MetricResult]:
        invalid_count, histogram = result.tensor[0], result.tensor[1:]
        counts = histogram.sum(dim=0)
        positions = result.tensor.new_tensor(_BETA_QUANTILES)[:, None, None] * (
            counts - 1
        ).clamp_min(0)[None]
        targets = torch.cat((positions.floor(), positions.ceil()), dim=0) + 1
        indices = torch.searchsorted(
            histogram.cumsum(dim=0).permute(1, 2, 0).contiguous(),
            targets.permute(1, 2, 0).contiguous(),
        ).permute(2, 0, 1)
        indices.clamp_max_(_BETA_HISTOGRAM_BINS - 1)
        bin_values = (indices + 0.5) / _BETA_HISTOGRAM_BINS
        bin_values.masked_fill_(indices == 0, 0)
        bin_values.masked_fill_(indices == _BETA_HISTOGRAM_BINS - 1, 1)
        lower, upper = bin_values.chunk(2)
        values = lower + (positions - positions.floor()) * (upper - lower)
        values.masked_fill_(
            (counts[None] == 0) | (invalid_count[None] != 0), float("nan")
        )
        return [
            MetricResult(
                values[quantile, update, head],
                f"{result.label}/head_{head}/update_{update}/{name}",
            )
            for update in range(counts.shape[0])
            for head in range(counts.shape[1])
            for quantile, name in enumerate(_BETA_QUANTILE_NAMES)
        ]
