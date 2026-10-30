# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small-gradient counts, with a bounded-memory pass for logical-tensor RMS thresholds."""

import math
from collections.abc import Callable, Iterable, Sequence

import torch

from .core import (
    LogicalReductionMetric,
    MetricResult,
    MetricSite,
    MetricStep,
    MetricTensor,
    TensorMetricExecutor,
)

_CHUNK_SIZE = 1024 * 1024


def _threshold_label(value: float) -> str:
    # Shortest round-trip representation, with stable, compact exponent spelling.
    text = str(value)
    if 'e' in text:
        mantissa, exponent = text.split('e')
        return f'{mantissa}e{int(exponent)}'
    return text.removesuffix('.0')


def validate_thresholds(values: Sequence[float], option: str) -> tuple[float, ...]:
    """Validate, sort and deduplicate a nonempty list of nonnegative finite thresholds."""
    thresholds = tuple(sorted(set(float(value) for value in values)))
    if not thresholds or any(not math.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError(f'{option} requires a nonempty list of finite, nonnegative values.')
    return thresholds


def _local_statistics(
    tensor: torch.Tensor,
    thresholds: Sequence[float],
    rms: torch.Tensor | None = None,
    *,
    moments: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed int64 counts and FP64 moments without a tensor-by-threshold temporary.

    FP64 accumulation avoids squared-gradient underflow for the tiny values these metrics
    diagnose. Scratch space is bounded by a chunk (plus a flattening copy for strided inputs).
    The input, as in the existing wgrad metrics, is the optimizer's observed gradient snapshot.
    """
    counts = torch.zeros(len(thresholds) + 2, dtype=torch.int64, device=tensor.device)
    counts[-2] = tensor.numel()
    squared_sum = torch.zeros((), dtype=torch.float64, device=tensor.device)
    limits = torch.tensor(thresholds, dtype=torch.float64, device=tensor.device)
    if rms is not None:
        limits = limits * rms
        counts[-1] = (~torch.isfinite(rms)).to(torch.int64)
    for chunk in tensor.reshape(-1).split(_CHUNK_SIZE):
        values = chunk.to(torch.float64).abs()
        finite = torch.isfinite(values)
        if moments:
            squared_sum += values.square().sum()
        for index, limit in enumerate(limits.unbind()):
            counts[index] += torch.count_nonzero(finite & (values <= limit))
    moment_state = torch.stack((squared_sum, squared_sum.new_tensor(tensor.numel())))
    return counts, moment_state


class WgradSmallFractionMetric(LogicalReductionMetric):
    """Report small-gradient fractions from packed counts at one or more thresholds.

    Relative counts must be prepared by :class:`SmallWgradPreparation`, using a complete
    logical tensor's RMS, never the RMS of an individual optimizer or model-parallel shard.
    """

    source_kinds = frozenset({'wgrad'})

    def __init__(self, thresholds: Sequence[float], *, relative: bool = False) -> None:
        self.relative = relative
        self.name = 'wgrad-relative-small-fraction' if relative else 'wgrad-small-fraction'
        self.thresholds = validate_thresholds(thresholds, self.name)

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Count absolute thresholds; relative preparation requires a distributed RMS pass."""
        if self.relative:
            raise RuntimeError('Relative small-gradient metrics require SmallWgradPreparation.')
        return _local_statistics(tensor, self.thresholds)[0]

    def finalize(self, contribution: torch.Tensor) -> torch.Tensor:
        """Divide integer counts, propagating an invalid logical RMS to every containing scope."""
        counts, population, invalid = contribution[:-2], contribution[-2], contribution[-1]
        fractions = counts.to(torch.float64) / population.clamp_min(1)
        return torch.where(invalid > 0, float('nan'), fractions)

    def finalize_results(self, results: Sequence[MetricResult]) -> list[MetricResult]:
        """Emit a scalar subkey per configured threshold."""
        prefix = 'rms_fraction_' if self.relative else 'le_'
        return [
            MetricResult(value, f'{result.label}/{prefix}{_threshold_label(threshold)}')
            for result in results
            for threshold, value in zip(self.thresholds, result.tensor.unbind(), strict=True)
        ]


class _NamedStates(LogicalReductionMetric):
    """Reduce compact states once per logical tensor; output scopes can then share them."""

    source_kinds = frozenset({'wgrad'})

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pass through an already compacted additive state."""
        return tensor

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Reduce shards and owners independently for each logical tensor."""
        groups: dict[str, list[MetricTensor]] = {}
        for value in values:
            names = tuple(site.name for site in value.sites)
            if any(name != names[0] for name in names[1:]):
                raise ValueError('Small-gradient inputs must identify one logical tensor.')
            groups.setdefault(names[0], []).append(value)
        return self._start_logical_reductions(sorted(groups.items()))


def _complete_states(
    executor: TensorMetricExecutor, values: Sequence[MetricTensor]
) -> list[MetricResult]:
    metric = _NamedStates()
    return executor.complete(metric, metric.start(values))


class SmallWgradPreparation:
    """Share counts and RMS across summary/individual metrics, retaining only compact states.

    First visit: absolute counts and relative RMS moments. At commit, reduce the moments,
    revisit the optimizer-backed gradients to count relative thresholds, then reduce counts.
    No forward/backward collectives, gradient gathers, or model-sized gradient copies occur.
    """

    def __init__(self, metrics: Sequence[WgradSmallFractionMetric]) -> None:
        self.metrics = metrics
        self.thresholds = {
            relative: tuple(
                sorted({t for m in metrics if m.relative == relative for t in m.thresholds})
            )
            for relative in (False, True)
        }
        self.absolute: list[MetricTensor] = []
        self.moments: list[MetricTensor] = []

    def add_batch(self, batch: Sequence[MetricTensor]) -> None:
        """Compact one parameter's semantic views before the source yields another parameter."""
        for value in batch:
            if not all(site.kind == 'wgrad' for site in value.sites):
                continue
            counts, moments = _local_statistics(
                value.tensor, self.thresholds[False], moments=bool(self.thresholds[True])
            )
            if self.thresholds[False]:
                self.absolute.append(value.with_tensor(counts))
            if self.thresholds[True]:
                self.moments.append(value.with_tensor(moments))

    def finish(
        self,
        executor: TensorMetricExecutor,
        replay_batches: Callable[[], Iterable[list[MetricTensor]]] | None,
    ) -> dict[int, list[MetricTensor]]:
        """Resolve logical RMS, revisit gradient storage, and share fully reduced counts."""
        states = {False: self.absolute, True: []}
        if self.thresholds[True]:
            if replay_batches is None:
                raise ValueError(
                    'Relative small-gradient metrics require replayable optimizer sources.'
                )
            rms_by_name = {
                result.label: (result.tensor[0] / result.tensor[1].clamp_min(1)).sqrt()
                for result in _complete_states(executor, self.moments)
            }
            self.moments.clear()
            for batch in replay_batches():
                for value in batch:
                    if not all(site.kind == 'wgrad' for site in value.sites):
                        continue
                    counts, _ = _local_statistics(
                        value.tensor, self.thresholds[True], rms_by_name[value.sites[0].name]
                    )
                    states[True].append(value.with_tensor(counts))
                # Do not retain the last semantic copy across the next parameter's extraction.
                if batch:
                    del value
                del batch
        prepared = {}
        for relative in (False, True):
            if not self.thresholds[relative]:
                continue
            resolved = _complete_states(executor, states[relative])
            for metric in self.metrics:
                if metric.relative != relative:
                    continue
                indices = [self.thresholds[relative].index(t) for t in metric.thresholds] + [-2, -1]
                prepared[id(metric)] = [
                    MetricTensor(result.tensor[indices], (MetricSite(result.label, 'wgrad'),), ())
                    for result in resolved
                ]
        return prepared
