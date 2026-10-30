# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compact input-embedding gradient diagnostics with global vocabulary row IDs."""

from collections.abc import Sequence

import torch

from .core import (
    FlatShard,
    LogicalReductionMetric,
    MetricResult,
    MetricSite,
    MetricStep,
    MetricTensor,
    RankRelation,
)


class EmbeddingWgradRowL2Metric(LogicalReductionMetric):
    """Mean and ten largest unscaled, pre-clipping input-embedding row norms.

    The optimizer adapter supplies squared-norm contributions indexed by global token ID,
    excluding storage padding and optionally vocabulary padding. Shards are summed before
    taking square roots; replicated gradients are counted once. The metric expands one vector
    result to scalar logging results after communication, avoiding a collective branch per scalar.
    """

    name = "embedding-wgrad-row-l2"
    source_kinds = frozenset({"embedding_wgrad_row_squared_l2"})

    def accepts(self, site: MetricSite) -> bool:
        """Select input word embeddings, including a module-qualified MIMO name."""
        return super().accepts(site) and (
            site.name == "embedding.word_embeddings.weight"
            or site.name.endswith(".embedding.word_embeddings.weight")
        )

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preserve the source's compact, global-row-indexed sum of squares."""
        return tensor

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Require one input embedding and reduce its row contributions once."""
        if len(values) != 1:
            raise ValueError("embedding-wgrad-row-l2 requires exactly one input word embedding.")
        return self._start_logical_reductions((("rows", values),))

    def finalize(self, contribution: torch.Tensor) -> torch.Tensor:
        """Return mean, top norms, and token IDs; ties prefer the smaller token ID."""
        norms = contribution.sqrt()
        count = min(10, norms.numel())
        indices = torch.argsort(norms, descending=True, stable=True)[:count]
        # Float64 preserves integer token IDs even for vocabularies larger than 2**24.
        return torch.cat(
            (norms.mean().reshape(1).double(), norms[indices].double(), indices.double())
        )

    def finalize_results(self, results: Sequence[MetricResult]) -> list[MetricResult]:
        """Expand the compact result after communication, before calling any logging sink."""
        scalars = []
        for result in results:
            count = (result.tensor.numel() - 1) // 2
            labels = ["mean"] + [f"top_{k}/l2" for k in range(1, count + 1)]
            labels += [f"top_{k}/token_id" for k in range(1, count + 1)]
            scalars.extend(
                MetricResult(value, label) for value, label in zip(result.tensor, labels)
            )
        return scalars


def _embedding_row_squared_l2(
    tensor: torch.Tensor,
    logical_shape: tuple[int, ...],
    storage_relations: tuple[RankRelation, ...],
    row_offset: int,
    vocab_size: int,
    loss_scale: torch.Tensor,
) -> torch.Tensor:
    """Map one optimizer gradient shard to a compact global-token sum-of-squares vector.

    Only row reductions are promoted to FP32, never a full embedding-sized temporary. Flat
    optimizer shards may split rows; their partial squared norms are added by the executor.
    Empty shards and remote-owner placeholders contribute the same-sized plain zero tensor.
    """
    if len(logical_shape) != 2 or logical_shape[1] <= 0:
        raise ValueError("Embedding row metrics require a matrix with a nonempty hidden dimension.")
    if vocab_size <= 0 or row_offset < 0:
        raise ValueError(
            "Embedding row metrics require a positive vocabulary and valid row offset."
        )
    result = torch.zeros(vocab_size, dtype=torch.float32, device=tensor.device)
    if not tensor.numel():
        return result
    flat_shards = [r.placement for r in storage_relations if isinstance(r.placement, FlatShard)]
    if len(flat_shards) > 1:
        raise ValueError("Embedding row metrics support at most one optimizer FlatShard.")
    columns = logical_shape[1]
    start, end = (flat_shards[0].start, flat_shards[0].end) if flat_shards else (0, tensor.numel())
    if end - start != tensor.numel() or end > logical_shape[0] * columns:
        raise ValueError("Embedding gradient storage does not match its logical shape or interval.")
    # Do not even reduce padded vocabulary rows, which may contain non-finite sentinels.
    end = min(end, max(0, vocab_size - row_offset) * columns)
    if end <= start:
        return result
    flat = tensor.detach().reshape(-1)[: end - start]
    scale = loss_scale.detach().to(device=tensor.device, dtype=torch.float32).reshape(())

    def squared_norm(values: torch.Tensor, dim: int) -> torch.Tensor:
        return (torch.linalg.vector_norm(values, dim=dim, dtype=torch.float32) / scale).square()

    row, column = divmod(start, columns)
    cursor = 0
    if column:
        cursor = min(columns - column, flat.numel())
        result[row_offset + row] = squared_norm(flat[:cursor], 0)
        row += 1
    full_rows = (flat.numel() - cursor) // columns
    if full_rows:
        stop = cursor + full_rows * columns
        result[row_offset + row : row_offset + row + full_rows] = squared_norm(
            flat[cursor:stop].view(full_rows, columns), 1
        )
        cursor = stop
        row += full_rows
    if cursor < flat.numel():
        result[row_offset + row] = squared_norm(flat[cursor:], 0)
    return result
