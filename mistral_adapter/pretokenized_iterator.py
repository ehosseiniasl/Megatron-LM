"""Mistral pretokenized adapter: wraps the in-repo PretokenizedDataLoader for Megatron-LM.

Uses the pretokenized data loading code at mistral_fork/mistral/data/pretokenized/.
"""

import logging
import typing
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from mistral_adapter.pretokenized_config import PretokenizedConfig
from mistral_adapter.pretokenized_config_utils import build_image_encoder_fn, config_to_source_items
from mistral_fork.mistral.data.data_loader import DataLoader
from mistral_fork.mistral.data.pretokenized.datasource import PretokenizedImageEncoder
from mistral_fork.mistral.data.pretokenized.distributed_prebuild_of_epoch_indexes import (
    nfs_barrier,
    prebuild_epoch_index,
)
from mistral_fork.mistral.data.pretokenized.pretokenized_args import PretokenizedArgs
from mistral_fork.mistral.data.pretokenized.pretokenized_data_loader import PretokenizedDataLoader
from mistral_fork.mistral.data.source import SampleType

logger = logging.getLogger(__name__)


def _configured_source_modalities(config: PretokenizedConfig) -> dict[str, bool]:
    """Map configured pretokenized source paths to document modality."""
    modalities = {}
    for source in config.sources:
        source_path = str(Path(source.prefix_path))
        is_multimodal = SampleType(source.sample_type).is_multimodal()
        configured_modality = modalities.setdefault(source_path, is_multimodal)
        if configured_modality != is_multimodal:
            raise ValueError(
                f"Pretokenized source path {source_path!r} has conflicting sample types."
            )
    return modalities


class MistralPretokenizedIterator:
    """Iterator that yields Megatron-compatible batch dicts from the in-repo PretokenizedDataLoader.

    When num_workers >= 1, data loading runs in separate subprocesses via
    MultiprocessedPretokenizedDataLoader, pre-fetching batches to hide I/O latency.

    When interdocument_masking=False, each call to __next__ returns:
        tokens:      int64  [micro_batch_size, seq_len]
        labels:      int64  [micro_batch_size, seq_len]
        loss_mask:   float32 [micro_batch_size, seq_len]
        position_ids: int64 [micro_batch_size, seq_len]

    When interdocument_masking=True (THD format), each call returns:
        tokens:      int64  [micro_batch_size, seq_len]
        labels:      int64  [micro_batch_size, seq_len]
        loss_mask:   float32 [micro_batch_size, seq_len]
        position_ids: int64 [micro_batch_size, seq_len]  (reset to 0 at each segment boundary)
        cu_seqlens:  int32  [micro_batch_size, padded_num_segments+1]
        max_seqlen:  int32  [micro_batch_size]

    When ``include_loss_modality=True``, either mode also returns ``loss_modality``: a boolean
    ``[micro_batch_size, seq_len]`` tensor that is true for tokens owned by image-conditioned
    documents. It describes the document, not the target-token type; consumers must still apply
    ``loss_mask`` to exclude image-token and other masked targets.

    Megatron's flatten_batch_for_packed_sequences() merges per-sample cu_seqlens
    into a single 1-D tensor and reshapes tokens/labels to (1, mbs*seq_len).
    """

    # PretokenizedDataLoader advances one global_sequence_id by local_batch * world_size and
    # slices [rank * local, (rank + 1) * local] out of it, so save_state() returns the same
    # cursor on every data-parallel rank and Megatron writes one state file for all of them.
    is_save_state_rank_independent = True

    def __init__(
        self,
        config: PretokenizedConfig,
        # Token budget used to size per-source epoch-index coverage.
        num_total_tokens: int,
        rank: int,
        world_size: int,
        micro_batch_size: int,
        num_workers: int = 0,
        prefetch_queue_depth: int = 2,
        allow_build_of_epoch_index: bool = False,
        allow_partial_epoch_index_cache: bool = False,
        allow_state_resetting: bool = False,
        nfs_barrier_dir: Path | None = None,
        epoch_index_build_timeout_s: float | None = None,
        image_encoder_build_fn: typing.Callable[[str], PretokenizedImageEncoder] | None = None,
        load_image_pixels: bool = True,
        max_image_group_workers: int = 1,
        include_loss_modality: bool = False,
        in_phase_transition: bool = False,
        include_data_source_ids: bool = False,
        preserve_logical_seqlens: bool = False,
        image_shard: tuple[int, int] | None = None,
    ) -> None:
        # Keep this strict while the two modes intentionally use different boundary
        # conventions. It can be relaxed once every consumer handles both conventions.
        assert config.drop_bos == (not config.interdocument_masking), (
            "Exactly one of interdocument_masking and drop_bos must be enabled: "
            "use drop_bos=True only when interdocument_masking=False"
        )
        seq_len = config.original_seq_len

        self._micro_batch_size = micro_batch_size
        self._seq_len = seq_len
        self._pad_id = config.pad_id
        self._img_id = config.img_id
        self._interdocument_masking = config.interdocument_masking
        self._preserve_logical_seqlens = preserve_logical_seqlens
        self._num_workers = num_workers
        self._include_loss_modality = include_loss_modality
        self._is_multimodal_by_source_path = _configured_source_modalities(config)
        self._data_source_indices = None
        if include_data_source_ids:
            paths = [str(Path(source.prefix_path)) for source in config.sources]
            if len(paths) != len(set(paths)):
                raise ValueError("Data-source metrics require unique configured source paths.")
            self._data_source_indices = {path: index for index, path in enumerate(paths)}

        # Auto-build image encoder from config if not provided
        if image_encoder_build_fn is None:
            image_encoder_build_fn = build_image_encoder_fn(config)

        source_items = config_to_source_items(config)
        num_toks_in_global_batch = micro_batch_size * seq_len * world_size

        if allow_build_of_epoch_index:
            assert (
                nfs_barrier_dir is not None
            ), "nfs_barrier_dir is required when allow_build_of_epoch_index=True"
            assert epoch_index_build_timeout_s is not None, (
                "epoch_index_build_timeout_s is required when " "allow_build_of_epoch_index=True"
            )
            assert epoch_index_build_timeout_s > 0, (
                "epoch_index_build_timeout_s must be > 0, got " f"{epoch_index_build_timeout_s}"
            )
            if torch.distributed.is_initialized():
                worker_rank = torch.distributed.get_rank()
                worker_world_size = torch.distributed.get_world_size()
            else:
                worker_rank = 0
                worker_world_size = 1

            # Ported from Mistral
            # `mistral/data/pretokenized/build_pretokenized.py::build_pretokenized_dataloader`
            # allow-build block, in the 2026-05-29 checkout.
            prebuild_epoch_index(
                source_items=source_items,
                seed=config.seed,
                num_total_tokens=num_total_tokens,
                seq_len=seq_len,
                bos_id=config.bos_id,
                eos_id=config.eos_id,
                pad_id=config.pad_id,
                epoch_index_cache_path=config.epoch_index_cache_path,
                image_encoder_build_fn=image_encoder_build_fn,
                skip_images=config.skip_images,
                worker_id=worker_rank,
                num_workers=worker_world_size,
                allow_epoch_index_building=True,
                selection_strategy=None,
                truncate_trailing_masked=config.truncate_trailing_masked,
            )
            if torch.distributed.is_initialized():
                nfs_barrier(
                    barrier_dir=nfs_barrier_dir,
                    name="building_of_indexes_pretokenized",
                    timeout_s=epoch_index_build_timeout_s,
                    rank=worker_rank,
                    world_size=worker_world_size,
                    error_msg=(
                        "Barrier timeout while building pretokenized epoch indexes. "
                        "Increase --pretokenized-build-of-epoch-index-timeout-s or "
                        "prebuild the indexes offline."
                    ),
                )

        pretokenized_args = PretokenizedArgs(
            per_source_filter=",".join(
                f"{item.source_path}:{item.filter_profile.value}" for item in source_items
            ),
            deprecated_source_set="megatron",
            epoch_index_cache_path=config.epoch_index_cache_path,
            allow_build_of_epoch_index=False,
            allow_partial_epoch_index_cache=allow_partial_epoch_index_cache,
            truncate_trailing_masked=config.truncate_trailing_masked,
            allow_non_sorted_sources_in_dataloader=(config.allow_non_sorted_sources_in_dataloader),
            skip_images=config.skip_images,
            drop_bos=config.drop_bos,
        )

        def _make_dataloader(
            multiprocessed_worker_info: tuple[int, int] | None
        ) -> PretokenizedDataLoader:
            return PretokenizedDataLoader(
                source_items=source_items,
                seed=config.seed,
                num_toks_in_global_batch=num_toks_in_global_batch,
                num_total_tokens=num_total_tokens,
                seq_len=seq_len,
                rank=rank,
                world_size=world_size,
                bos_id=config.bos_id,
                eos_id=config.eos_id,
                pad_id=config.pad_id,
                pretokenized_args=pretokenized_args,
                image_encoder_build_fn=image_encoder_build_fn,
                multiprocessed_worker_info=multiprocessed_worker_info,
                selection_strategy=None,
                optimized_datasource_id_sampling=config.optimized_datasource_id_sampling,
                allow_state_resetting=allow_state_resetting,
                in_phase_transition=in_phase_transition,
                load_image_pixels=load_image_pixels,
                max_image_group_workers=max_image_group_workers,
                image_shard=image_shard,
            )

        if num_workers >= 1:
            from mistral_fork.mistral.data.pretokenized.multiprocessed_pretokenized import (
                MultiprocessedPretokenizedDataLoader,
            )

            logger.info(
                f"Using MultiprocessedPretokenizedDataLoader with {num_workers} workers, "
                f"prefetch_queue_depth={prefetch_queue_depth}"
            )
            self._loader: DataLoader = MultiprocessedPretokenizedDataLoader(
                pretokenized_data_loader_fn=_make_dataloader,
                num_workers=num_workers,
                max_size=prefetch_queue_depth,
                timeout=600,
            )
            self._loader.set_wrapped_by_closing(True)
        else:
            self._loader = _make_dataloader(multiprocessed_worker_info=None)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        batch, _ = self.next_with_state()
        return batch

    def next_with_state(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Return the next batch and the native state immediately after it."""
        batch = self._loader.get_batch()
        state = self._loader.get_state()
        if self._interdocument_masking:
            result = self._convert_batch_thd(batch)
        else:
            result = self._convert_batch(batch)
        if getattr(self, "_data_source_indices", None) is not None:
            result["data_source_ids"] = self._build_data_source_ids(batch)
        return result, state

    def _build_data_source_ids(self, batch: Any) -> torch.Tensor:
        """Expand document provenance in token order using configured source indices."""
        paths = getattr(batch, "sample_pretokenized_paths", None)
        sizes = getattr(batch, "sizes", None)
        if paths is None or sizes is None or len(paths) != len(sizes):
            raise ValueError("Data-source metrics require document-aligned source paths and sizes.")
        try:
            indices = [self._data_source_indices[str(Path(path))] for path in paths]
        except KeyError as error:
            raise ValueError(f"Unconfigured document source path: {error.args[0]!r}") from error
        source_ids = np.repeat(np.asarray(indices, dtype=np.int64), sizes)
        if source_ids.size != self._micro_batch_size * self._seq_len:
            raise ValueError("Document source sizes must span the complete token batch.")
        return torch.from_numpy(source_ids.reshape(self._micro_batch_size, self._seq_len))

    def _convert_batch(self, batch: Any) -> dict[str, torch.Tensor]:
        B = self._micro_batch_size
        S = self._seq_len

        tokens = torch.from_numpy(batch.x.reshape(B, S))
        labels = torch.from_numpy(batch.y.reshape(B, S))

        if batch.y_mask is not None:
            loss_mask = torch.from_numpy(batch.y_mask.astype(np.float32).reshape(B, S))
        else:
            loss_mask = torch.ones(B, S, dtype=torch.float32)

        position_ids = torch.arange(S, dtype=torch.int64).unsqueeze(0).expand(B, S)

        result: dict[str, Any] = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        if self._include_loss_modality:
            result["loss_modality"] = self._build_loss_modality(batch, B, S)
        if hasattr(batch, "image_positions"):
            result["image_token_mask"] = self._image_token_mask(batch, B, S)
        if batch.images is not None and len(batch.images) > 0:
            result["images"] = batch.images

        return result

    def _image_token_mask(self, batch: Any, batch_size: int, seq_len: int) -> torch.Tensor:
        """Mark only image tokens backed by image metadata.

        Token ID equality alone is insufficient because ordinary text can contain
        the vocabulary ID reserved for image placeholders.
        """
        mask = torch.zeros(batch_size * seq_len, dtype=torch.bool)
        for position in batch.image_positions:
            assert 0 <= position.start < position.end <= mask.numel(), position
            mask[position.start : position.end] = True
        # An image span also contains row separators and the image-end marker.
        # Only img_id positions are routed through the vision encoder.
        mask &= torch.from_numpy(batch.x) == self._img_id
        return mask.reshape(batch_size, seq_len)

    def _convert_batch_thd(self, batch: Any) -> dict[str, torch.Tensor]:
        """Convert batch to THD format with per-sample cu_seqlens."""
        B = self._micro_batch_size
        S = self._seq_len
        total_len = B * S

        tokens = torch.from_numpy(batch.x.reshape(B, S))
        labels = torch.from_numpy(batch.y.reshape(B, S))

        if batch.y_mask is not None:
            loss_mask = torch.from_numpy(batch.y_mask.astype(np.float32).reshape(B, S))
        else:
            loss_mask = torch.ones(B, S, dtype=torch.float32)

        assert (
            hasattr(batch, "sizes") and batch.sizes is not None
        ), "batch.sizes must be available for interdocument masking"
        all_starts = np.concatenate([[0], np.cumsum(batch.sizes)]).tolist()
        assert all_starts[0] == 0 and all_starts[-1] == total_len, (
            f"batch.sizes doesn't span the full sequence. "
            f"cu_seqlens: {all_starts}, total_len: {total_len}"
        )

        # Input lengths come from the native producer before storage padding.
        # Image and prompt rows stay real even when their LM targets are masked.
        logical_starts = None
        if self._preserve_logical_seqlens and batch.num_pad_tokens:
            if batch.logical_sizes is None:
                raise ValueError("Padded Mistral batches must preserve logical document lengths")
            logical_starts = dict(zip(all_starts, np.cumsum([0, *batch.logical_sizes]).tolist()))

        cu_seqlens_rows = []
        logical_rows = []
        max_seqlens = []
        position_ids = torch.zeros(B, S, dtype=torch.int64)
        for b in range(B):
            sample_start = b * S
            sample_end = (b + 1) * S
            row = [0]
            for start in all_starts[1:]:
                if start <= sample_start:
                    continue
                if start > sample_end:
                    break
                row.append(start - sample_start)
            assert row[-1] == S, f"sample {b}: cu_seqlens doesn't end at seq_len {S}, got {row[-1]}"
            cu_seqlens_rows.append(row)
            if logical_starts is not None:
                logical_rows.append(
                    [
                        logical_starts[sample_start + offset] - logical_starts[sample_start]
                        for offset in row
                    ]
                )

            doc_lengths = [row[i + 1] - row[i] for i in range(len(row) - 1)]
            max_seqlens.append(max(doc_lengths))
            for i, doc_len in enumerate(doc_lengths):
                doc_start = row[i]
                doc_end = row[i + 1]
                position_ids[b, doc_start:doc_end] = torch.arange(doc_len, dtype=torch.int64)

        max_row_len = max(len(row) for row in cu_seqlens_rows)
        cu_seqlens = torch.full((B, max_row_len), S, dtype=torch.int32)
        for b, row in enumerate(cu_seqlens_rows):
            cu_seqlens[b, : len(row)] = torch.tensor(row, dtype=torch.int32)
        max_seqlen = torch.tensor(max_seqlens, dtype=torch.int32)

        result: dict[str, Any] = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }
        if logical_rows:
            result["cu_seqlens_padded"] = cu_seqlens
            logical_cu_seqlens = torch.empty_like(cu_seqlens)
            for b, row in enumerate(logical_rows):
                logical_cu_seqlens[b].fill_(row[-1])
                logical_cu_seqlens[b, : len(row)] = torch.tensor(row, dtype=torch.int32)
            result["cu_seqlens"] = logical_cu_seqlens
        if self._include_loss_modality:
            result["loss_modality"] = self._build_loss_modality(batch, B, S)
        if hasattr(batch, "image_positions"):
            result["image_token_mask"] = self._image_token_mask(batch, B, S)
        if batch.images is not None and len(batch.images) > 0:
            result["images"] = batch.images

        return result

    def _build_loss_modality(self, batch: Any, batch_size: int, seq_len: int) -> torch.Tensor:
        """Expand document source modalities into a token-aligned boolean indicator."""
        source_paths = getattr(batch, "sample_pretokenized_paths", None)
        if source_paths is None:
            raise ValueError("Loss-by-modality requires document-aligned source paths.")
        if len(batch.srcs) != len(batch.sizes) or len(source_paths) != len(batch.sizes):
            raise ValueError(
                "Loss-by-modality requires aligned document sources, source paths, and sizes."
            )

        document_is_multimodal = []
        for source_name, source_path in zip(batch.srcs, source_paths, strict=True):
            try:
                document_is_multimodal.append(
                    self._is_multimodal_by_source_path[str(Path(source_path))]
                )
            except KeyError as error:
                raise ValueError(
                    f"Document source {source_name!r} has unconfigured path {source_path!r}."
                ) from error

        token_is_multimodal = np.repeat(
            np.asarray(document_is_multimodal, dtype=np.bool_), batch.sizes
        )
        expected_tokens = batch_size * seq_len
        if token_is_multimodal.size != expected_tokens:
            raise ValueError(
                "Document sizes must span the complete batch when loss-by-modality is enabled "
                f"(got {token_is_multimodal.size} tokens, expected {expected_tokens})."
            )
        return torch.from_numpy(token_is_multimodal.reshape(batch_size, seq_len))

    def ensure_started(self) -> None:
        """Start the single-process loader so its initial state can be snapshotted.

        Only valid for num_workers=0 (the validation path). The single-process
        PretokenizedDataLoader.start() is idempotent, so calling this before the
        first batch lets the caller capture the global_sequence_id=0 state for
        later rewinding via set_state().
        """
        assert self._num_workers == 0, (
            f"ensure_started() is only supported for num_workers=0, " f"got {self._num_workers}"
        )
        self._loader.start()

    def set_state(self, data: dict[str, Any] | None) -> None:
        assert data is not None, "set_state called with None — not sure None is expected here"
        self._loader.set_state(data)

    def save_state(self) -> dict[str, Any] | None:
        """Required by Megatron's maybe_save_dataloader_state."""
        return self._loader.get_state()

    def close(self) -> None:
        """Clean up worker processes."""
        self._loader.close()
