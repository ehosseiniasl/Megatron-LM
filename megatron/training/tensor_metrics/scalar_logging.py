# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""High-volume scalar logging for production tensor metrics."""

from collections.abc import Sequence
from typing import Any

import torch

from megatron.training.global_vars import get_tensorboard_writer, get_wandb_writer

from .core import MetricResult, TensorMetric


def write_partitioned_scalar_results(
    metric: TensorMetric, results: Sequence[MetricResult], iteration: int
) -> None:
    """Write global and detailed scalar results into separate logger sections."""
    scalars = {}
    for result in results:
        if result.tensor.numel() != 1:
            raise ValueError(
                f"The default tensor metric sink requires scalar results, but {metric.name!r} "
                f"produced shape {tuple(result.tensor.shape)}."
            )
        section = (
            "tensor-metrics-global"
            if result.label == "global" or result.label.startswith("global/")
            else "tensor-metrics-breakdowns"
        )
        tag = f"{section}/{metric.name}/{result.label}"
        if tag in scalars:
            raise ValueError(
                f"Tensor metric {metric.name!r} produced duplicate result tag {tag!r}."
            )
        scalars[tag] = result.tensor.detach()

    tensorboard_writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    if scalars and (tensorboard_writer is not None or wandb_writer is not None):
        # Reuse the host values when both sinks are enabled. Passing CUDA tensors to W&B
        # otherwise lets its serializer copy and synchronize each scalar separately.
        host_scalars = _scalar_values_to_host(scalars)
        if tensorboard_writer is not None:
            _write_tensorboard_scalars(tensorboard_writer, host_scalars, iteration)
        if wandb_writer is not None:
            wandb_writer.log(host_scalars, iteration)
    if (
        scalars
        and tensorboard_writer is None
        and wandb_writer is None
        and _is_last_rank()
    ):
        printable_scalars = _scalar_values_to_host(scalars)
        print(
            f"tensor metrics at iteration {iteration}: {printable_scalars}", flush=True
        )


def _scalar_values_to_host(
    scalars: dict[str, torch.Tensor],
) -> dict[str, float | int | bool]:
    """Copy scalar values once per device/dtype, preserving their types and tag order."""
    buckets: dict[tuple[torch.device, torch.dtype], list[tuple[str, torch.Tensor]]] = {}
    for tag, value in scalars.items():
        buckets.setdefault((value.device, value.dtype), []).append((tag, value))

    host_scalars = {}
    for entries in buckets.values():
        values = (
            torch.stack(
                [
                    value if value.ndim == 0 else value.reshape(())
                    for _, value in entries
                ]
            )
            .cpu()
            .tolist()
        )
        host_scalars.update((tag, value) for (tag, _), value in zip(entries, values))
    return {tag: host_scalars[tag] for tag in scalars}


def _write_tensorboard_scalars(
    writer: Any, scalars: dict[str, float | int | bool], iteration: int
) -> None:
    """Enqueue one multi-value summary using already materialized host values."""
    from tensorboard.compat.proto.summary_pb2 import Summary

    summary = Summary(
        value=[
            Summary.Value(tag=tag, simple_value=float(value))
            for tag, value in scalars.items()
        ]
    )
    writer.file_writer.add_summary(summary, iteration)


def _is_last_rank() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1
    )
