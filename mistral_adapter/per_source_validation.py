# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-datasource validation loss helpers."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast

import numpy as np
import torch

from mistral_adapter.pretokenized_config import (
    PretokenizedConfig,
    PretokenizedPerSourceValidationConfig,
    PretokenizedPerSourceValidationSourceConfig,
    SourceConfig,
)
from mistral_adapter.pretokenized_config_utils import apply_pretokenized_config_overrides
from mistral_adapter.pretokenized_iterator import MistralPretokenizedIterator
from mistral_fork.mistral.data.pretokenized.pretokenized_data_loader import (
    PretokenizedDataLoader,
)

PER_SOURCE_LOSS_KEY = "__per_source_lm_loss"
PER_SOURCE_NUM_SOURCES_KEY = "__per_source_num_sources"
PER_SOURCE_RESULT_PREFIX = f"{PER_SOURCE_LOSS_KEY}:"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerSourceValidationMetadata:
    """Ordered datasource names and their exact global input-token budgets."""

    source_names: tuple[str, ...]
    input_token_budgets_by_source: tuple[int, ...]

    @property
    def num_sources(self) -> int:
        return len(self.source_names)


def _metadata_from_object(value: Any) -> PerSourceValidationMetadata | None:
    has_input_token_budgets = hasattr(value, "input_token_budgets_by_source")
    if not has_input_token_budgets:
        return None
    assert hasattr(value, "source_names"), (
        "Per-source validation iterators that expose input_token_budgets_by_source "
        "must also expose source_names"
    )

    raw_source_names = value.source_names
    raw_input_token_budgets = value.input_token_budgets_by_source
    assert (raw_source_names is None) == (raw_input_token_budgets is None), (
        "source_names and input_token_budgets_by_source must either both be set " "or both be None"
    )
    if raw_source_names is None:
        return None

    source_names = tuple(raw_source_names)
    input_token_budgets = tuple(int(count) for count in raw_input_token_budgets)
    assert source_names, "Per-source validation requires at least one datasource"
    assert all(
        isinstance(name, str) and name for name in source_names
    ), f"Per-source validation names must be non-empty strings, got {source_names}"
    assert len(set(source_names)) == len(
        source_names
    ), f"Per-source validation names must be unique, got {source_names}"
    assert len(input_token_budgets) == len(source_names), (
        "Per-source validation metadata length mismatch: "
        f"{len(source_names)} names and {len(input_token_budgets)} input-token budgets"
    )
    assert all(count > 0 for count in input_token_budgets), (
        "Per-source validation input-token budgets must be positive, " f"got {input_token_budgets}"
    )
    return PerSourceValidationMetadata(source_names, input_token_budgets)


def get_per_source_validation_metadata(data_iterator: Any) -> PerSourceValidationMetadata | None:
    """Find and validate metadata through rerun/wrapper and virtual-pipeline layers."""

    metadata: list[PerSourceValidationMetadata] = []
    visited: set[int] = set()

    def visit(value: Any) -> None:
        if value is None or id(value) in visited:
            return
        visited.add(id(value))

        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return

        direct = _metadata_from_object(value)
        if direct is not None:
            metadata.append(direct)

        for attribute in ("iterable", "_inner"):
            wrapped = getattr(value, attribute, None)
            if wrapped is not None and wrapped is not value:
                visit(wrapped)

    visit(data_iterator)
    if not metadata:
        return None

    expected = metadata[0]
    assert all(candidate == expected for candidate in metadata[1:]), (
        "Per-source validation metadata disagrees across iterator/wrapper or virtual-pipeline "
        f"stages: {metadata}"
    )
    return expected


def expand_source_ids_to_token_shape(
    source_ids: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """Expand row-level datasource IDs before packed-sequence and CP transforms."""

    assert (
        source_ids.dtype == torch.int64
    ), f"Expected source_ids dtype torch.int64, got {source_ids.dtype}"
    assert (
        source_ids.dim() == 1
    ), f"Expected row-level source_ids with shape [micro_batch_size], got {source_ids.shape}"
    assert (
        loss_mask.dim() == 2
    ), f"Expected loss_mask with shape [batch, sequence], got {loss_mask.shape}"
    assert (
        source_ids.shape[0] == loss_mask.shape[0]
    ), f"source_ids rows ({source_ids.shape[0]}) do not match loss_mask rows ({loss_mask.shape[0]})"
    return source_ids.view(-1, 1).expand_as(loss_mask).contiguous()


def build_per_source_loss_report(
    *, losses: torch.Tensor, loss_mask: torch.Tensor, source_ids: torch.Tensor, num_sources: int
) -> torch.Tensor:
    """Return loss sums, scored-token counts, and input-token counts by source."""

    assert num_sources > 0, f"Expected a positive num_sources, got {num_sources}"
    losses = losses.view(-1)
    loss_mask = loss_mask.view(-1)
    source_ids = source_ids.view(-1)
    assert losses.shape == loss_mask.shape == source_ids.shape, (
        "Per-source validation tensors must have identical token shapes, got "
        f"losses={losses.shape}, loss_mask={loss_mask.shape}, source_ids={source_ids.shape}"
    )
    assert (
        source_ids.dtype == torch.int64
    ), f"Expected source_ids dtype torch.int64, got {source_ids.dtype}"

    invalid_ids = (source_ids < -1) | (source_ids >= num_sources)
    if torch.any(invalid_ids).item():
        invalid_values = source_ids[invalid_ids]
        raise AssertionError(
            "Per-source validation IDs must be -1 for padding or fall in "
            f"[0, {num_sources}), got {invalid_values[:16].tolist()} "
            f"({invalid_values.numel()} invalid tokens)"
        )

    scored_padding = (source_ids == -1) & (loss_mask != 0)
    if torch.any(scored_padding).item():
        raise AssertionError("Per-source validation padding IDs (-1) must have a zero loss mask")

    valid_ids = source_ids >= 0
    report = torch.zeros((num_sources, 3), dtype=torch.float64, device=losses.device)
    valid_source_ids = source_ids[valid_ids]
    report[:, 0].scatter_add_(
        0,
        valid_source_ids,
        losses[valid_ids].to(torch.float64) * loss_mask[valid_ids].to(torch.float64),
    )
    report[:, 1].scatter_add_(0, valid_source_ids, loss_mask[valid_ids].to(torch.float64))
    report[:, 2].scatter_add_(
        0, valid_source_ids, torch.ones_like(valid_source_ids, dtype=torch.float64)
    )
    return report


def per_source_result_key(source_name: str) -> str:
    return f"{PER_SOURCE_RESULT_PREFIX}{source_name}"


def source_name_from_result_key(key: str) -> str | None:
    if not key.startswith(PER_SOURCE_RESULT_PREFIX):
        return None
    return key[len(PER_SOURCE_RESULT_PREFIX) :]


class _ReplayableIterator(Protocol):
    def __next__(self) -> dict[str, torch.Tensor]: ...

    def ensure_started(self) -> None: ...

    def save_state(self) -> dict[str, Any] | None: ...

    def set_state(self, data: dict[str, Any] | None) -> None: ...

    def close(self) -> None: ...


class _TransformedIterator:
    def __init__(
        self,
        inner: _ReplayableIterator,
        transform: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._inner = inner
        self._transform = transform

    def __iter__(self) -> "_TransformedIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        return self._transform(next(self._inner))

    def ensure_started(self) -> None:
        self._inner.ensure_started()

    def save_state(self) -> dict[str, Any] | None:
        return self._inner.save_state()

    def set_state(self, data: dict[str, Any] | None) -> None:
        self._inner.set_state(data)

    def close(self) -> None:
        self._inner.close()


class _FixedIterator:
    """Rewind to the initial dataloader state before each validation event."""

    def __init__(self, inner: _ReplayableIterator, eval_iters: int, reset_period: int) -> None:
        self._inner = inner
        self._eval_iters = eval_iters
        self._reset_period = reset_period
        inner.ensure_started()
        self._initial_state = inner.save_state()
        assert self._initial_state is not None
        self._count = 0

    def __iter__(self) -> "_FixedIterator":
        return self

    def __len__(self) -> int:
        return self._eval_iters

    def __next__(self) -> dict[str, Any]:
        if self._count and self._count % self._reset_period == 0:
            self._inner.set_state(self._initial_state)
        self._count += 1
        return next(self._inner)

    def close(self) -> None:
        self._inner.close()


class _ReportingIterator:
    """Attach report metadata to a fixed validation iterator."""

    def __init__(
        self,
        inner: _FixedIterator,
        source_names: list[str],
        input_tokens_per_source: int,
    ) -> None:
        self._inner = inner
        self.source_names = tuple(source_names)
        self.input_token_budgets_by_source = (input_tokens_per_source,) * len(source_names)

    def __iter__(self) -> "_ReportingIterator":
        return self

    def __len__(self) -> int:
        return len(self._inner)

    def __next__(self) -> dict[str, Any]:
        batch = next(self._inner)
        assert "source_ids" in batch
        batch[PER_SOURCE_NUM_SOURCES_KEY] = len(self.source_names)
        return batch

    def close(self) -> None:
        self._inner.close()


def _sources_with_equal_index_budgets(
    sources: list[PretokenizedPerSourceValidationSourceConfig],
) -> list[SourceConfig]:
    """Convert named sources to loader configs sized for equal validation budgets.

    Source weights are a training-blend property. The explicit validation schedule
    selects every datasource directly, so uniform loader weights are used only to
    size each datasource's epoch-index coverage to ``input_tokens_per_source``.
    """

    weight = 1.0 / len(sources)
    return [
        SourceConfig.model_validate(source.model_dump(exclude={"name"})).model_copy(
            update={"weight": weight}
        )
        for source in sources
    ]


def select_top_weighted_validation_sources(
    config: PretokenizedPerSourceValidationConfig, top_n: int
) -> PretokenizedPerSourceValidationConfig:
    """Select the highest-weight validation sources, or all sources for zero."""

    if top_n < 0:
        raise ValueError("--per-source-validation-top-n must be non-negative")
    if top_n == 0:
        return config
    if top_n > len(config.sources):
        raise ValueError(
            "--per-source-validation-top-n cannot exceed the number of configured "
            f"validation sources ({len(config.sources)})"
        )
    sources = sorted(config.sources, key=lambda source: source.weight, reverse=True)[
        :top_n
    ]
    return config.model_copy(update={"sources": sources})


def resolve_per_source_validation_schedule(
    *,
    input_tokens_per_source: int,
    sequence_length: int,
    num_sources: int,
    eval_global_batch_size: int,
    requested_eval_iters: int,
) -> tuple[int, int, int]:
    """Resolve equal per-source rows and eval iterations from one budget mode."""

    if min(sequence_length, num_sources, eval_global_batch_size) <= 0:
        raise ValueError(
            "Sequence length, source count, and eval global batch size must be positive"
        )
    if input_tokens_per_source == 0:
        if requested_eval_iters <= 0:
            raise ValueError("input_tokens_per_source=0 requires a positive --eval-iters")
        rows_per_source = requested_eval_iters * eval_global_batch_size // num_sources
        if rows_per_source == 0:
            raise ValueError("The evaluation schedule must provide at least one row per source")
        return rows_per_source * sequence_length, rows_per_source, requested_eval_iters
    if input_tokens_per_source < 0:
        raise ValueError("input_tokens_per_source must be non-negative")
    if requested_eval_iters > 0:
        raise ValueError("Set input_tokens_per_source to 0 when supplying --eval-iters")
    if input_tokens_per_source % sequence_length:
        raise ValueError("input_tokens_per_source must be divisible by --seq-length")

    rows_per_source = input_tokens_per_source // sequence_length
    eval_iters = (
        rows_per_source * num_sources + eval_global_batch_size - 1
    ) // eval_global_batch_size
    return input_tokens_per_source, rows_per_source, eval_iters


def build_per_source_validation_iterator(
    *,
    args: Any,
    config: PretokenizedConfig,
    per_source_validation: PretokenizedPerSourceValidationConfig,
    dp_rank: int,
    dp_world_size: int,
    eval_global_batch_size: int,
    eval_micro_batch_size: int,
    eval_num_microbatches: int,
    load_image_pixels: bool = True,
    max_image_group_workers: int = 1,
    transform_batch: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    expose_source_metadata: bool = True,
) -> tuple[_FixedIterator | _ReportingIterator, int]:
    sources = per_source_validation.sources
    configured_budget = args.input_tokens_per_source
    input_tokens_per_source, rows_per_source, eval_iters = resolve_per_source_validation_schedule(
        input_tokens_per_source=configured_budget,
        sequence_length=args.seq_length,
        num_sources=len(sources),
        eval_global_batch_size=eval_global_batch_size,
        requested_eval_iters=args.eval_iters if configured_budget == 0 else 0,
    )
    source_ids = [
        source_id for _ in range(rows_per_source) for source_id in range(len(sources))
    ]
    num_source_rows = len(source_ids)
    num_scheduled_rows = eval_iters * eval_global_batch_size
    source_ids.extend([-1] * (num_scheduled_rows - num_source_rows))
    rows_per_microbatch = eval_micro_batch_size * dp_world_size
    assert num_scheduled_rows % rows_per_microbatch == 0
    reset_period = num_scheduled_rows // rows_per_microbatch
    assert reset_period == eval_iters * eval_num_microbatches

    validation_sources = _sources_with_equal_index_budgets(sources)
    valid_config = config.model_copy(
        update={
            "sources": validation_sources,
            "seed": per_source_validation.valid_seed,
            "epoch_index_cache_path": (
                per_source_validation.epoch_index_cache_path or config.epoch_index_cache_path
            ),
        }
    )
    valid_config = apply_pretokenized_config_overrides(
        valid_config,
        percentage_data_multiplier=getattr(args, "overwrite_percentage_data_multiplier", None),
    )
    raw_iterator = PerSourcePretokenizedIterator(
        config=valid_config,
        num_total_tokens=num_source_rows * args.seq_length,
        rank=dp_rank,
        world_size=dp_world_size,
        micro_batch_size=eval_micro_batch_size,
        num_workers=0,
        allow_state_resetting=True,
        allow_build_of_epoch_index=False,
        load_image_pixels=load_image_pixels,
        max_image_group_workers=max_image_group_workers,
        global_source_id_schedule=source_ids,
    )
    replayable: _ReplayableIterator = raw_iterator
    if transform_batch is not None:
        replayable = _TransformedIterator(raw_iterator, transform_batch)
    fixed = _FixedIterator(replayable, eval_iters, reset_period)
    iterator = (
        _ReportingIterator(
            fixed,
            [source.name for source in sources],
            input_tokens_per_source,
        )
        if expose_source_metadata
        else fixed
    )
    logger.info(
        "Built per-source validation iterator: sources=%d, "
        "input_tokens_per_source=%d, rows_per_source=%d, padding_rows=%d, "
        "eval_iters=%d, dp_rank=%d/%d",
        len(sources),
        input_tokens_per_source,
        rows_per_source,
        num_scheduled_rows - num_source_rows,
        eval_iters,
        dp_rank,
        dp_world_size,
    )
    return iterator, eval_iters


class PerSourcePretokenizedIterator(MistralPretokenizedIterator):
    """Pretokenized iterator driven by an exact physical-source row schedule."""

    def __init__(
        self,
        *,
        global_source_id_schedule: list[int],
        **iterator_kwargs: Any,
    ) -> None:
        config = iterator_kwargs["config"]
        rank = int(iterator_kwargs["rank"])
        world_size = int(iterator_kwargs["world_size"])
        micro_batch_size = int(iterator_kwargs["micro_batch_size"])
        assert int(iterator_kwargs.get("num_workers", 0)) == 0, (
            "Per-source validation requires num_workers=0"
        )
        assert global_source_id_schedule
        first_padding_row = next(
            (
                row
                for row, source_id in enumerate(global_source_id_schedule)
                if source_id == -1
            ),
            len(global_source_id_schedule),
        )
        assert all(source_id == -1 for source_id in global_source_id_schedule[first_padding_row:])

        self._rank = rank
        self._rows_per_global_microbatch = micro_batch_size * world_size
        assert len(global_source_id_schedule) % self._rows_per_global_microbatch == 0
        assert all(
            -1 <= source_id < len(config.sources)
            for source_id in global_source_id_schedule
        )
        self._global_source_id_schedule = global_source_id_schedule
        self._scheduled_row_index = 0
        super().__init__(**iterator_kwargs)

    def next_with_state(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if self._scheduled_row_index >= len(self._global_source_id_schedule):
            raise StopIteration

        loader = cast(PretokenizedDataLoader, self._loader)
        loader.start()
        state = loader.get_state()
        assert int(state["global_sequence_id"]) == self._scheduled_row_index

        schedule_end = self._scheduled_row_index + self._rows_per_global_microbatch
        global_source_ids = np.asarray(
            self._global_source_id_schedule[self._scheduled_row_index : schedule_end],
            dtype=np.int64,
        )
        batch = loader.get_batch_with_datasource_ids(
            local_size_for_this_batch=self._micro_batch_size,
            datasource_ids=global_source_ids.tolist(),
            skip=False,
        )
        assert batch is not None

        local_start = self._rank * self._micro_batch_size
        source_ids = torch.from_numpy(
            global_source_ids[local_start : local_start + self._micro_batch_size].copy()
        )
        self._scheduled_row_index = schedule_end
        result = (
            self._convert_batch_thd(batch)
            if self._interdocument_masking
            else self._convert_batch(batch)
        )
        self._pad_rows(result, source_ids)
        result["source_ids"] = source_ids
        return result, loader.get_state()

    def _pad_rows(self, batch: dict[str, torch.Tensor], source_ids: torch.Tensor) -> None:
        padding_rows = source_ids == -1
        if not torch.any(padding_rows):
            return
        batch["tokens"][padding_rows] = self._pad_id
        batch["labels"][padding_rows] = self._pad_id
        batch["loss_mask"][padding_rows] = 0
        if "cu_seqlens" in batch:
            batch["position_ids"][padding_rows] = torch.arange(self._seq_len)
            batch["cu_seqlens"][padding_rows] = self._seq_len
            batch["cu_seqlens"][padding_rows, 0] = 0
            batch["max_seqlen"][padding_rows] = self._seq_len

    def set_state(self, data: dict[str, Any] | None) -> None:
        super().set_state(data)
        assert data is not None
        global_sequence_id = int(data["global_sequence_id"])
        assert global_sequence_id % self._rows_per_global_microbatch == 0
        assert global_sequence_id <= len(self._global_source_id_schedule)
        self._scheduled_row_index = global_sequence_id
