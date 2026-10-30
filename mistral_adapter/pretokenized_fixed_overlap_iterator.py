"""Doc-local fixed-overlap validation over Mistral pretokenized sources."""

import math
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch

from mistral_adapter.pretokenized_config import PretokenizedConfig, SourceConfig
from mistral_fork.mistral.data.pretokenized.datasource import _reload_source_metadata_from_folder
from mistral_fork.mistral.data.pretokenized.megatron_index_reader import (
    DType,
    _FileBinReader,
    _IndexReader,
)
from mistral_fork.mistral.data.pretokenized.pretokenized_utils import DEFAULT_TOKEN_DTYPE
from mistral_fork.mistral.data.pretokenized.processing.paths_helpers import (
    PrefixPaths,
    check_chunk_is_complete,
)


@dataclass(frozen=True)
class _PretokenizedTextSource:
    prefix_path: str
    index: _IndexReader
    bin_reader: _FileBinReader


@dataclass(frozen=True)
class _FixedOverlapWindow:
    source_id: int
    doc_id: int
    start: int
    end: int
    skip_logprobs: int

    @property
    def num_logprobs(self) -> int:
        return self.end - self.start - 1


def _iter_fixed_overlap_windows(
    *, token_count: int, seq_len: int, overlap: int
) -> Iterator[tuple[int, int, int]]:
    assert token_count > 0, f"Expected a non-empty document, got {token_count=}"
    assert seq_len > 1, f"Expected {seq_len=} to leave room for labels"
    assert 0 <= overlap < seq_len, f"Expected overlap in [0, seq_len), got {overlap=}"

    context_tokens = max(overlap, 1)
    score_start = 1
    while score_start < token_count:
        start = max(0, score_start - context_tokens)
        end = min(start + seq_len, token_count)
        assert end > score_start, (
            f"Window cannot score any tokens with {token_count=}, {seq_len=}, "
            f"{overlap=}, {score_start=}"
        )
        skip_logprobs = score_start - start - 1
        yield start, end, skip_logprobs
        score_start = end


def _selected_doc_ids(*, source: SourceConfig, index: _IndexReader, seq_len: int) -> range:
    if source.percentage_data >= 1.0:
        return range(index.sequence_count)

    n_total_tokens = int(np.sum(index.sequence_lengths) - index.sequence_count)
    n_target_tokens = max(int(source.percentage_data * n_total_tokens), seq_len)
    n_tokens = 0
    doc_count = 0
    while doc_count < index.sequence_count and n_tokens < n_target_tokens:
        n_tokens += int(index.sequence_lengths[doc_count]) - 1
        doc_count += 1
    assert (
        doc_count > 0
    ), f"No documents selected for {source.prefix_path} with {source.percentage_data=}"
    return range(doc_count)


def _load_text_source(source: SourceConfig, *, skip_images: bool) -> _PretokenizedTextSource:
    prefix_paths = PrefixPaths.from_prefix(source.prefix_path)
    metadata = _reload_source_metadata_from_folder(prefix_paths)
    check_chunk_is_complete(prefix_paths, source_metadata=metadata, is_merged=True)
    assert not metadata.has_image_metadata or skip_images, (
        "fixed-overlap validation currently expects text-only pretokenized input "
        f"or skip_images=true, got images for {source.prefix_path}"
    )
    assert metadata.masking_strategy is None, (
        "fixed-overlap validation does not support pretokenized masks yet, got "
        f"{metadata.masking_strategy=} for {source.prefix_path}"
    )
    index = _IndexReader(idx_path=str(prefix_paths.index))
    bin_reader = _FileBinReader(str(prefix_paths.bin))
    return _PretokenizedTextSource(
        prefix_path=source.prefix_path, index=index, bin_reader=bin_reader
    )


def _pack_windows_into_rows(
    windows: list[_FixedOverlapWindow], *, row_seq_len: int
) -> list[list[_FixedOverlapWindow]]:
    assert row_seq_len > 0, f"Expected {row_seq_len=} to be positive"

    rows: list[list[_FixedOverlapWindow]] = []
    current: list[_FixedOverlapWindow] = []
    current_len = 0
    for window in windows:
        window_len = window.num_logprobs
        assert window_len > 0, (window, window_len)
        assert (
            window_len <= row_seq_len
        ), f"Window with {window_len=} does not fit in {row_seq_len=}: {window}"
        if current and current_len + window_len > row_seq_len:
            rows.append(current)
            current = []
            current_len = 0
        current.append(window)
        current_len += window_len

    if current:
        rows.append(current)

    return rows


class MistralPretokenizedFixedOverlapIterator:
    """Megatron batches matching fixed-overlap perplexity scoring.

    The normal Mistral pretokenized loader produces training-stream chunks:
    documents can be packed together and long documents are split into
    independent training sequences. This iterator is eval-only: it reads the
    same pretokenized documents, drops EOS, makes doc-local overlapping prompt
    windows, and masks overlap/padding so each content token is scored once.
    """

    def __init__(
        self,
        *,
        config: PretokenizedConfig,
        rank: int,
        world_size: int,
        micro_batch_size: int,
        window_seq_len: int,
        overlap: int,
        batch_seq_len: int,
        preserve_logical_seqlens: bool = False,
    ) -> None:
        assert micro_batch_size >= 1, f"Expected {micro_batch_size=} >= 1"
        assert 0 <= rank < world_size, f"Expected {rank=} in [0, {world_size=})"
        assert window_seq_len > 1, f"Expected {window_seq_len=} to leave room for labels"
        assert 0 <= overlap < window_seq_len, f"Expected overlap in [0, seq_len), got {overlap=}"
        assert batch_seq_len >= window_seq_len - 1, (
            "Batch sequence length must fit at least one validation window's "
            f"logprobs, got {batch_seq_len=} and {window_seq_len=}"
        )

        self._rank = rank
        self._world_size = world_size
        self._micro_batch_size = micro_batch_size
        self._window_seq_len = window_seq_len
        self._batch_seq_len = batch_seq_len
        self._preserve_logical_seqlens = preserve_logical_seqlens
        self._bos_id = config.bos_id
        self._eos_id = config.eos_id
        self._pad_id = config.pad_id
        self._global_microbatch_id = 0

        self._sources = [
            _load_text_source(source, skip_images=config.skip_images) for source in config.sources
        ]
        self._windows: list[_FixedOverlapWindow] = []
        self._num_scored_tokens = 0
        for source_id, source_config in enumerate(config.sources):
            source = self._sources[source_id]
            for doc_id in _selected_doc_ids(
                source=source_config, index=source.index, seq_len=window_seq_len
            ):
                doc_size_with_eos = int(source.index.sequence_lengths[doc_id])
                doc_token_count = doc_size_with_eos - 1
                assert doc_token_count >= 2, (
                    "Expected BOS plus at least one content token for "
                    f"{source.prefix_path} {doc_id=}, got {doc_token_count=}"
                )
                self._num_scored_tokens += doc_token_count - 1
                for start, end, skip_logprobs in _iter_fixed_overlap_windows(
                    token_count=doc_token_count, seq_len=window_seq_len, overlap=overlap
                ):
                    self._windows.append(
                        _FixedOverlapWindow(
                            source_id=source_id,
                            doc_id=doc_id,
                            start=start,
                            end=end,
                            skip_logprobs=skip_logprobs,
                        )
                    )

        assert self._windows, "No fixed-overlap validation windows were built"
        self._packed_rows = _pack_windows_into_rows(self._windows, row_seq_len=self._batch_seq_len)
        self.num_global_microbatches = math.ceil(
            len(self._packed_rows) / (micro_batch_size * world_size)
        )

    @property
    def num_windows(self) -> int:
        return len(self._windows)

    @property
    def num_scored_tokens(self) -> int:
        return self._num_scored_tokens

    @property
    def expected_eval_tokens(self) -> int:
        return self._num_scored_tokens

    def __iter__(self) -> "MistralPretokenizedFixedOverlapIterator":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        return self._next_rows()

    def _next_rows(self) -> dict[str, torch.Tensor]:
        base_row_id = (
            self._global_microbatch_id * self._world_size * self._micro_batch_size
            + self._rank * self._micro_batch_size
        )
        self._global_microbatch_id += 1

        tokens = torch.full(
            (self._micro_batch_size, self._batch_seq_len),
            fill_value=self._pad_id,
            dtype=torch.int64,
        )
        labels = torch.full_like(tokens, fill_value=self._pad_id)
        loss_mask = torch.zeros((self._micro_batch_size, self._batch_seq_len), dtype=torch.float32)

        row_segment_lengths: list[list[int]] = []
        row_real_token_counts: list[int] = []
        for row in range(self._micro_batch_size):
            packed_row_id = base_row_id + row
            if packed_row_id < len(self._packed_rows):
                packed_windows = self._packed_rows[packed_row_id]
            else:
                packed_windows = []

            segment_lengths: list[int] = []
            offset = 0
            for window in packed_windows:
                n_logprobs = self._fill_segment(
                    tokens=tokens,
                    labels=labels,
                    loss_mask=loss_mask,
                    row=row,
                    offset=offset,
                    window=window,
                )
                offset += n_logprobs
                segment_lengths.append(n_logprobs)

            row_real_token_counts.append(offset)
            max_padding_segment_len = self._window_seq_len - 1
            while offset < self._batch_seq_len:
                pad_len = min(max_padding_segment_len, self._batch_seq_len - offset)
                offset += pad_len
                segment_lengths.append(pad_len)
            row_segment_lengths.append(segment_lengths)

        position_ids = torch.zeros((self._micro_batch_size, self._batch_seq_len), dtype=torch.int64)
        cu_seqlens_rows: list[list[int]] = []
        max_seqlens: list[int] = []
        for row, segment_lengths in enumerate(row_segment_lengths):
            starts = [0]
            for segment_len in segment_lengths:
                starts.append(starts[-1] + int(segment_len))
            assert starts[-1] == self._batch_seq_len, (
                f"segments do not span row {row}: {starts=}, "
                f"batch_seq_len={self._batch_seq_len}"
            )
            cu_seqlens_rows.append(starts)
            max_seqlens.append(max(segment_lengths))
            for start, end in zip(starts[:-1], starts[1:]):
                position_ids[row, start:end] = torch.arange(end - start, dtype=torch.int64)

        max_row_len = max(len(row) for row in cu_seqlens_rows)
        cu_seqlens = torch.full(
            (self._micro_batch_size, max_row_len), self._batch_seq_len, dtype=torch.int32
        )
        for row, starts in enumerate(cu_seqlens_rows):
            cu_seqlens[row, : len(starts)] = torch.tensor(starts, dtype=torch.int32)

        result = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": torch.tensor(max_seqlens, dtype=torch.int32),
        }
        if self._preserve_logical_seqlens:
            result["cu_seqlens_padded"] = cu_seqlens
            result["cu_seqlens"] = torch.minimum(
                cu_seqlens, torch.tensor(row_real_token_counts, dtype=torch.int32).unsqueeze(1)
            )
        return result

    def _fill_segment(
        self,
        *,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        row: int,
        offset: int,
        window: _FixedOverlapWindow,
    ) -> int:
        token_values, label_values, mask_values = self._window_arrays(window)
        n_logprobs = len(token_values)
        end = offset + n_logprobs
        assert end <= tokens.shape[1], (offset, n_logprobs, tokens.shape)

        tokens[row, offset:end] = torch.from_numpy(token_values)
        labels[row, offset:end] = torch.from_numpy(label_values)
        loss_mask[row, offset:end] = torch.from_numpy(mask_values)
        return n_logprobs

    def _window_arrays(
        self, window: _FixedOverlapWindow
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = self._sources[window.source_id]
        doc_offset_bytes = int(source.index.sequence_pointers[window.doc_id])
        prompt = source.bin_reader.read(
            dtype=DEFAULT_TOKEN_DTYPE,
            offset=doc_offset_bytes + window.start * DType.size(DEFAULT_TOKEN_DTYPE),
            count=window.end - window.start,
        )
        assert len(prompt) >= 2, (window, len(prompt))
        if window.start == 0:
            assert int(prompt[0]) == self._bos_id, (window, int(prompt[0]))

        n_logprobs = len(prompt) - 1
        token_values = prompt[:-1].astype(np.int64)
        label_values = prompt[1:].astype(np.int64)
        mask_values = np.ones(n_logprobs, dtype=np.float32)
        if window.skip_logprobs:
            mask_values[: window.skip_logprobs] = 0.0
        return token_values, label_values, mask_values

    def ensure_started(self) -> None:
        return

    def save_state(self) -> dict[str, int]:
        return {"global_microbatch_id": self._global_microbatch_id}

    def set_state(self, data: dict[str, Any] | None) -> None:
        assert data is not None, "Cannot restore fixed-overlap iterator from None"
        self._global_microbatch_id = int(data["global_microbatch_id"])

    def close(self) -> None:
        return
