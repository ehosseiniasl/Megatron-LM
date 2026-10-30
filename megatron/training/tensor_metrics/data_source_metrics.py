# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Metrics that group language-model loss by modality."""

from collections.abc import Sequence

import torch

from megatron.core.tensor_observation import observe_tensor
from megatron.core.utils import unwrap_model

from .core import LogicalReductionMetric, MetricResult, MetricStep, MetricTensor

__all__ = [
    "ModalityLossMetric",
    "modality_metrics_requested",
    "observe_modality_loss",
]


_MODALITY_METRIC_NAMES = frozenset({"loss-by-modality"})


def modality_metrics_requested(specifications: Sequence[str]) -> bool:
    """Return whether any configured tensor metric needs per-token document modality."""
    return any(
        specification.strip().rpartition(":")[0] in _MODALITY_METRIC_NAMES
        for specification in specifications
    )


def observe_modality_loss(
    model: torch.nn.Module,
    losses: torch.Tensor,
    loss_mask: torch.Tensor,
    modality_ids: torch.Tensor,
) -> None:
    """Observe token-weighted loss sums and counts for text and image documents.

    ``modality_ids`` identifies the document that owns each predicted language token: zero is a
    text-only document and one is an image-conditioned document. The ordinary language-model loss
    mask remains authoritative, so image-token targets and any other masked targets are excluded.
    """
    if losses.shape != loss_mask.shape or losses.shape != modality_ids.shape:
        raise ValueError("Losses, loss mask, and modality IDs must have matching shapes.")
    if modality_ids.dtype not in {
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("Modality IDs must have a boolean or integer dtype.")
    if modality_ids.dtype != torch.bool and torch.any((modality_ids != 0) & (modality_ids != 1)):
        raise ValueError("Modality IDs must contain only text (0) or image (1).")

    is_image = modality_ids.to(device=losses.device, dtype=torch.bool)
    weighted_loss_mask = loss_mask.to(dtype=losses.dtype)
    modality_state = torch.stack(
        tuple(
            torch.stack(
                (
                    (losses * weighted_loss_mask * modality_mask).sum(),
                    (weighted_loss_mask * modality_mask).sum(),
                )
            )
            for modality_mask in (~is_image, is_image)
        )
    )
    observe_tensor(unwrap_model(model), "modality_loss", "modality_loss", modality_state)


class ModalityLossMetric(LogicalReductionMetric):
    """Report token-weighted language-model loss for text and image documents."""

    name = "loss-by-modality"
    source_kinds = frozenset({"modality_loss"})
    reduction_op = torch.distributed.ReduceOp.SUM
    _labels = ("text", "image")

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reject the generic contribution path; modality buckets are already compact."""
        raise NotImplementedError("Modality loss observations are already prepared by modality.")

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Validate and preserve compact ``[loss_sum, token_count]`` modality rows."""
        prepared = []
        for value in values:
            if value.tensor.shape != (len(self._labels), 2):
                raise ValueError(
                    "Modality loss observations require one [loss_sum, token_count] row for "
                    "text and image."
                )
            prepared.append(value.with_tensor(value.tensor.float()))
        return prepared

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Start one reduction branch for each document modality."""
        if not values:
            return []
        return [
            self._start_logical_reduction(
                tuple(value.with_tensor(value.tensor[index]) for value in values), label
            )
            for index, label in enumerate(self._labels)
        ]

    def finalize_results(self, results: Sequence[MetricResult]) -> list[MetricResult]:
        """Divide globally reduced loss sums by counts and omit empty modalities."""
        if not results:
            return []
        for result in results:
            if result.tensor.shape != (2,):
                raise ValueError(
                    "A reduced modality loss result must contain a loss sum and token count."
                )
        present = torch.stack(tuple(result.tensor[1] > 0 for result in results)).tolist()
        return [
            MetricResult(result.tensor[0] / result.tensor[1], result.label)
            for result, is_present in zip(results, present)
            if is_present
        ]
