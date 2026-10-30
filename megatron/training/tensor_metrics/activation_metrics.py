# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Local activation-health metrics for grouped-expert MLP neurons."""

from collections.abc import Sequence

import torch

from .core import MetricResult, MetricSite, MetricStep, MetricTensor, TensorMetric
from .definitions import LayerL2NormMetric

__all__ = ["LoggingRankLocalExpert0ReluHealthMetric"]


_PRE_RELU_TOTAL = 0
_TOKEN_COUNT = 1
_ZERO_COUNT = 2
_STATISTIC_COUNT = 3
_PRE_RELU_SOURCE_KIND = "expert_pre_relu"
_POST_RELU_SOURCE_KIND = "expert_post_relu"


class LoggingRankLocalExpert0ReluHealthMetric(TensorMetric):
    """Summarize local expert 0 on the logging rank without communication."""

    name = "logging-rank-local-expert-0-relu-health"
    source_kinds = frozenset({_PRE_RELU_SOURCE_KIND, _POST_RELU_SOURCE_KIND})

    def is_active_on_rank(self) -> bool:
        """Run only where Megatron creates its TensorBoard and W&B writers."""
        return _is_logging_rank()

    def accepts(self, site: MetricSite) -> bool:
        """Select the first local expert in each numbered MoE layer."""
        return (
            super().accepts(site)
            and LayerL2NormMetric._site_layer_label(site) is not None
            and ".local_experts.0." in f".{site.name}."
        )

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Compact one activation observation to local per-neuron accumulators."""
        prepared = []
        for value in values:
            if len(value.sites) != 1:
                raise ValueError(
                    "A raw activation must identify exactly one source site."
                )
            site = value.sites[0]
            if not self.accepts(site):
                continue
            tensor = value.tensor
            if tensor.ndim < 3 or tensor.shape[0] != 1:
                raise ValueError(
                    "Expert activation observations require shape [1, tokens, ..., neurons]."
                )
            neurons = tensor.shape[-1]
            if neurons == 0:
                raise ValueError(
                    "Activation health metrics require at least one neuron."
                )
            tokens = tensor.reshape(1, -1, neurons)
            statistics = torch.zeros(
                (1, _STATISTIC_COUNT, neurons),
                dtype=torch.float32,
                device=tensor.device,
            )
            if site.kind == _PRE_RELU_SOURCE_KIND:
                statistics[:, _PRE_RELU_TOTAL] = tokens.float().sum(dim=1)
                statistics[:, _TOKEN_COUNT].fill_(tokens.shape[1])
            elif site.kind == _POST_RELU_SOURCE_KIND:
                statistics[:, _ZERO_COUNT] = (tokens == 0).sum(dim=1)
            else:
                raise ValueError(
                    f"Unexpected ReLU activation source kind {site.kind!r}."
                )
            prepared.append(value.with_tensor(statistics, ()))
        return prepared

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Aggregate the selected iteration and emit local layer/global summaries."""
        values_by_layer: dict[str, list[MetricTensor]] = {}
        for value in values:
            if len(value.sites) != 1:
                raise ValueError("A prepared activation must identify one source site.")
            layer = LayerL2NormMetric._layer_label(value)
            if layer is None:
                raise ValueError("An activation tensor must identify one logical layer.")
            values_by_layer.setdefault(layer, []).append(value)

        global_populations = []
        results = []
        for layer in sorted(values_by_layer):
            pre_relu_mean, relu_sparsity = self._statistics(values_by_layer[layer])
            if not pre_relu_mean.numel():
                # No tokens reached this expert in the iteration. It has no population
                # to summarize and must not dilute the other layers' global statistics.
                continue
            global_populations.append((pre_relu_mean, relu_sparsity))
            results.extend(self._summaries(layer, pre_relu_mean, relu_sparsity))

        if global_populations:
            results.extend(
                self._summaries(
                    "global",
                    torch.cat(tuple(population[0] for population in global_populations)),
                    torch.cat(tuple(population[1] for population in global_populations)),
                )
            )
        return results

    def resume(
        self, values: Sequence[MetricTensor], continuation: object
    ) -> MetricStep:
        """Reject collective continuations because this metric is strictly local."""
        raise RuntimeError("The logging-rank local ReLU metric must not communicate.")

    @staticmethod
    def _statistics(
        values: Sequence[MetricTensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = values[0].tensor.shape
        if any(value.tensor.shape != shape for value in values):
            raise ValueError(
                "Paired activation observations must have matching shapes."
            )
        source_counts = tuple(
            sum(value.sites[0].kind == source_kind for value in values)
            for source_kind in (_PRE_RELU_SOURCE_KIND, _POST_RELU_SOURCE_KIND)
        )
        if source_counts[0] == 0 or source_counts[0] != source_counts[1]:
            raise ValueError(
                "Activation health metrics require one observation of every source per sample."
            )
        statistics = torch.stack(tuple(value.tensor for value in values)).sum(dim=0)
        if statistics.ndim != 3 or statistics.shape[1] != _STATISTIC_COUNT:
            raise ValueError(
                "ReLU activation statistics must have shape [population, 3, neurons]."
            )
        counts = statistics[:, _TOKEN_COUNT].reshape(-1)
        valid = counts > 0
        denominators = counts[valid]
        pre_relu_mean = statistics[:, _PRE_RELU_TOTAL].reshape(-1)[valid] / denominators
        relu_sparsity = statistics[:, _ZERO_COUNT].reshape(-1)[valid] / denominators
        return pre_relu_mean, relu_sparsity

    @staticmethod
    def _summaries(
        label: str, pre_relu_mean: torch.Tensor, relu_sparsity: torch.Tensor
    ) -> list[MetricResult]:
        if not pre_relu_mean.numel():
            raise ValueError(
                "Activation distributions require at least one observed neuron."
            )
        return [
            MetricResult(pre_relu_mean.mean(), f"{label}/pre-relu-mean/mean"),
            MetricResult(
                pre_relu_mean.std(unbiased=False), f"{label}/pre-relu-mean/std"
            ),
            MetricResult(
                torch.quantile(pre_relu_mean, 0.01), f"{label}/pre-relu-mean/p01"
            ),
            MetricResult(
                torch.quantile(pre_relu_mean, 0.05), f"{label}/pre-relu-mean/p05"
            ),
            MetricResult(
                torch.quantile(pre_relu_mean, 0.50), f"{label}/pre-relu-mean/p50"
            ),
            MetricResult(
                torch.quantile(relu_sparsity, 0.50), f"{label}/relu-sparsity/p50"
            ),
            MetricResult(
                torch.quantile(relu_sparsity, 0.90), f"{label}/relu-sparsity/p90"
            ),
            MetricResult(
                torch.quantile(relu_sparsity, 0.99), f"{label}/relu-sparsity/p99"
            ),
        ]


def _is_logging_rank() -> bool:
    """Whether this process owns Megatron's TensorBoard and W&B writers."""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1
    )
