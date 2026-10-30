# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared/routed output magnitudes and routed mixture-rank profiles."""

from collections.abc import Sequence

import torch

from megatron.core.transformer.moe.contribution_diagnostics import HEADER_SIZE, SOURCE

from .core import (
    AllReduce,
    CollectiveRequest,
    CollectiveStage,
    MetricResult,
    MetricStep,
    MetricTensor,
    RankRelation,
    Shard,
    TensorMetric,
)
from .definitions import LayerL2NormMetric


class MoEContributionsMetric(TensorMetric):
    """Pool additive moments before taking ratios; keep MTP prediction depths distinct."""

    name = "moe-contributions"
    source_kinds = frozenset({SOURCE})

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Retain only detached summaries with token-population shard semantics."""
        result = []
        for value in values:
            if len(value.sites) != 1:
                raise ValueError(
                    "A MoE contribution summary must identify exactly one source site."
                )
            if not self.accepts(value.sites[0]):
                continue
            n = value.tensor.numel() - HEADER_SIZE
            if value.tensor.ndim != 1 or n <= 0 or n % 3:
                raise ValueError("Malformed MoE contribution summary.")
            relations = tuple(
                RankRelation(r.axis, Shard(None) if isinstance(r.placement, Shard) else r.placement)
                for r in value.rank_relations
            )
            result.append(value.with_tensor(value.tensor.detach().double().clone(), relations))
        return result

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Pool microbatches by layer and MTP depth, then sum across token shards."""
        groups: dict[str, list[MetricTensor]] = {}
        for value in values:
            layer = LayerL2NormMetric._layer_label(value)
            if layer is None:
                raise ValueError("MoE contributions require a numbered layer site.")
            suffix = value.sites[0].name.rsplit(SOURCE, 1)[-1]
            latent = suffix.endswith(".latent")
            suffix = suffix.removesuffix(".latent") if latent else suffix
            label = layer + ("/" + suffix[1:] if suffix else "") + (".latent" if latent else "")
            groups.setdefault(label, []).append(value)
        stages = []
        for label, items in sorted(groups.items()):
            first = items[0]
            if any(
                v.tensor.shape != first.tensor.shape or v.rank_relations != first.rank_relations
                for v in items
            ):
                raise ValueError("MoE contribution microbatches must have matching layouts.")
            total = first.with_tensor(torch.stack([v.tensor for v in items]).sum(0))
            axes = tuple(r.axis for r in total.rank_relations if isinstance(r.placement, Shard))
            stages.append(self._advance(total, axes, label))
        return stages

    def resume(self, values: Sequence[MetricTensor], continuation: object) -> MetricStep:
        """Continue a pooled population reduction."""
        if not isinstance(continuation, tuple) or len(continuation) != 2 or len(values) != 1:
            raise ValueError("Invalid MoE contribution continuation.")
        axes, label = continuation
        return self._advance(values[0], axes, label)

    def _advance(self, value: MetricTensor, axes: tuple[str, ...], label: str) -> MetricStep:
        if axes:
            return CollectiveStage(
                (CollectiveRequest(value, axes[0], AllReduce()),), (axes[1:], label)
            )
        return MetricResult(value.tensor, label)

    def finalize_results(self, results: Sequence[MetricResult]) -> list[MetricResult]:
        """Expand only after population reductions have finished."""
        return [scalar for result in results for scalar in self._scalar_results(result)]

    @staticmethod
    def _scalar_results(result: MetricResult) -> list[MetricResult]:
        """Expand one pooled vector to named ratios, profiles and denominator flags."""
        data = result.tensor
        tokens, shared, routed, dot, zero_shared, zero_routed, zero_weights = data[
            :HEADER_SIZE
        ].unbind()
        energy, weights, zeros = data[HEADER_SIZE:].reshape(3, -1).unbind()
        topk = energy.numel()

        def divide(a, b):
            return torch.where(b != 0, a / b, a.new_full((), float("nan")))

        scalars = {
            "tokens": tokens,
            "shared-rms": divide(shared, tokens).sqrt(),
            "routed-rms": divide(routed, tokens).sqrt(),
            "shared-over-routed-rms": divide(shared, routed).sqrt(),
            "shared-routed-alignment": divide(dot, shared.sqrt() * routed.sqrt()),
            "shared-zero-fraction": divide(zero_shared, tokens),
            "routed-zero-fraction": divide(zero_routed, tokens),
            "zero-weight-sum-fraction": divide(zero_weights, tokens),
            "top-over-last-rms": divide(energy[0], energy[-1]).sqrt(),
            "top-three-energy-share": divide(energy[: min(3, topk)].sum(), energy.sum()),
            "empty-population": (tokens == 0).double(),
            "zero-shared-energy": (shared == 0).double(),
            "zero-routed-energy": (routed == 0).double(),
            "zero-last-rank-energy": (energy[-1] == 0).double(),
            "zero-individual-energy": (energy.sum() == 0).double(),
        }
        for j in range(topk):
            scalars[f"rank_{j+1}/rms"] = divide(energy[j], tokens).sqrt()
            scalars[f"rank_{j+1}/energy-share"] = divide(energy[j], energy.sum())
            scalars[f"rank_{j+1}/mean-weight-share"] = divide(weights[j], tokens)
            scalars[f"rank_{j+1}/zero-fraction"] = divide(zeros[j], tokens)
        latent = result.label.endswith(".latent")
        label = result.label.removesuffix(".latent") if latent else result.label
        rank_keys = {
            "top-over-last-rms",
            "top-three-energy-share",
            "zero-last-rank-energy",
            "zero-individual-energy",
        }
        return [
            MetricResult(
                v,
                f"{label}/"
                + ("latent/" if latent and (key in rank_keys or key.startswith("rank_")) else "")
                + key,
            )
            for key, v in scalars.items()
        ]
