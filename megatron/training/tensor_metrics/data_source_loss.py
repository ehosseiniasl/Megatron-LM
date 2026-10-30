# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Token-weighted training loss by configured MIMO data source."""

import re
from argparse import Namespace
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_observation import observe_tensor
from megatron.core.utils import unwrap_model

from .core import LogicalReductionMetric, MetricResult, MetricTensor

_SOURCE_SCOPE: ContextVar[tuple[str, ...]] = ContextVar("tensor_metric_data_sources", default=())


def data_source_metrics_requested(specifications: Sequence[str]) -> bool:
    """Return whether the configuration requests source-aligned token metadata."""
    return any(
        specification.strip().rpartition(":")[0] == DataSourceLossMetric.name
        for specification in specifications
    )


def configured_data_source_names(args: Namespace) -> tuple[str, ...]:
    """Return readable, unique labels in MIMO's pretokenized configuration order."""
    if getattr(args, "dataset_provider", None) == "mock":
        return ("000-mock",)
    config = getattr(args, "_mistral_pretokenized_config", None)
    if config is None:
        raise ValueError("Data-source metrics require MIMO Mistral or mock data.")
    names = []
    for index, source in enumerate(config.sources):
        path = Path(source.prefix_path)
        if path.name.startswith("chunks_"):
            path = path.parent
            if re.fullmatch(r"merged_[0-9]+b", path.name):
                path = path.parent
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name).strip("_") or "source"
        names.append(f"{index:03d}-{name}")
    return tuple(names)


@contextmanager
def capture_data_source_metrics(
    source_names: Sequence[str], pg_collection: ProcessGroupCollection
) -> Iterator[None]:
    """Scope source metadata to a sampled iteration using the observer's rank axes."""
    if source_names and getattr(pg_collection, "cp", None) is not None:
        if pg_collection.cp.size() > 1:
            raise NotImplementedError("Data-source tensor metrics currently require CP=1.")
    token = _SOURCE_SCOPE.set(tuple(source_names))
    try:
        yield
    finally:
        _SOURCE_SCOPE.reset(token)


def _validate_source_ids(source_ids: torch.Tensor, source_count: int) -> None:
    if source_ids.ndim != 2 or source_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Data source IDs must be integer tensors shaped [batch, sequence].")
    if torch.any((source_ids < 0) | (source_ids >= source_count)):
        raise ValueError("Data source IDs must identify configured sources.")


def _bucket_tokens(
    sums: torch.Tensor, counts: torch.Tensor, source_ids: torch.Tensor, source_count: int
) -> torch.Tensor:
    # FP64 represents integer element counts exactly well beyond a production batch's size.
    # Only token statistics and the small source table use FP64, never the full activations.
    statistics = torch.stack((sums.reshape(-1).double(), counts.reshape(-1).double()), dim=1)
    buckets = statistics.new_zeros((source_count, 2))
    buckets.index_add_(0, source_ids.reshape(-1).long(), statistics)
    return buckets


@torch.no_grad()
def observe_data_source_loss(
    model: torch.nn.Module, losses: torch.Tensor, loss_mask: torch.Tensor, source_ids: torch.Tensor
) -> None:
    """Observe source buckets using the unmodified language-model target loss mask."""
    scope = _SOURCE_SCOPE.get()
    if not scope:
        return
    _validate_source_ids(source_ids, len(scope))
    if losses.shape != loss_mask.shape or losses.shape != source_ids.shape:
        raise ValueError("Losses, loss mask, and source IDs must have matching token shapes.")
    masked_loss = (losses.detach().float() * loss_mask).masked_fill(loss_mask == 0, 0)
    buckets = _bucket_tokens(masked_loss, loss_mask, source_ids, len(scope))
    observe_tensor(unwrap_model(model), "data_source_loss", "data_source_loss", buckets)


class DataSourceLossMetric(LogicalReductionMetric):
    """Reduce source sums and counts before reporting means over the sampled iteration."""

    reduction_op = torch.distributed.ReduceOp.SUM
    name = "train-loss-by-data-source"
    source_kinds = frozenset({"data_source_loss"})

    def __init__(self, source_names: Sequence[str]) -> None:
        self.source_names = tuple(source_names)
        if not self.source_names or len(set(self.source_names)) != len(self.source_names):
            raise ValueError("Data-source metrics require nonempty, unique source names.")

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Source metrics prepare compact sum/count buckets directly."""
        raise NotImplementedError("Data-source metrics prepare source buckets directly.")

    def finalize_results(self, results: Sequence[MetricResult]) -> list[MetricResult]:
        """Omit globally absent sources and normalize after all ranks/microbatches merge."""
        if not results:
            return []
        if any(result.tensor.shape != (len(self.source_names), 2) for result in results):
            raise ValueError("Source results must contain one sum/count row per source.")
        present = torch.stack([result.tensor[:, 1] > 0 for result in results]).tolist()
        scalars = []
        for result, source_present in zip(results, present):
            prefix = "" if result.label == "global" else f"{result.label}/"
            rows = [
                (f"{prefix}source/{self.source_names[index]}", result.tensor[index])
                for index, exists in enumerate(source_present)
                if exists
            ]
            if any(source_present):
                rows.append((f"{prefix}global", result.tensor.sum(dim=0)))
            for label, (total, count) in rows:
                mean = total / count
                scalars.append(MetricResult(mean, label))
        return scalars

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Retain only the small source tables produced by the loss observation."""
        if any(value.tensor.shape != (len(self.source_names), 2) for value in values):
            raise ValueError("Source observations must contain one sum/count row per source.")
        return [value.with_tensor(value.tensor.detach().clone()) for value in values]
