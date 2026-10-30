import abc
import copy
import dataclasses
import hashlib
import json
import logging
import math
import multiprocessing
import os
import shutil
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Callable, Generator, Iterator

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

import torch
from PIL import Image
from pydantic import AliasChoices, Field

from mistral_fork.mistral.data.interleaved_pretokenized.selection_args import (
    SelectionStrategy,
)
from mistral_fork.mistral.data.iterator.utils import SequenceOfSamples, pad_sequence
from mistral_fork.mistral.data.pretokenized.image_transform_metadata import (
    TransformImageMetadata,
)
from mistral_fork.mistral.data.pretokenized.image_utils import load_contents_grouped
from mistral_fork.mistral.data.pretokenized.indexed_jsonl import (
    IndexedJSONL,
    _write_jsonl_and_index_batched,
    _write_jsonl_and_index_streaming,
)
from mistral_fork.mistral.data.pretokenized.indexed_mask import MaskReader
from mistral_fork.mistral.data.pretokenized.megatron_index_reader import (
    DType,
    _FileBinReader,
    _IndexReader,
    _ListTokensBinReader,
)
from mistral_fork.mistral.data.pretokenized.metadata import (
    DocumentIndexEntry,
    ImageIndexEntry,
    ScatteredMetadataReader,
    TextIndexEntry,
    _check_mask_correctness,
    _mask_size,
    _mask_to_array_mask,
    _MaskCursor,
    _MetadataReader,
)
from mistral_fork.mistral.data.pretokenized.pretokenized_utils import (
    DEFAULT_TOKEN_DTYPE,
)
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
    MaskingStrategy,
    PretokenizedDatasourceMetadataV2,
)
from mistral_fork.mistral.data.pretokenized.processing.paths_helpers import (
    PrefixPaths,
    check_chunk_is_complete,
    get_datasource_metadata_v2_path,
)
from mistral_fork.mistral.data.source import SampleType, get_src_name
from mistral_fork.mistral.monitoring.profiling import log_profile_if_slow
from mistral_fork.mistral_common.base import MistralBase
from mistral_fork.mistral_common.tokens.tokenizers.base import Span
from mistral_fork.shared.image_utils import TrainingImageEncoding
from mistral_fork.shared.s3_utils import is_object_path_dropped, validate_s3_path

logger = logging.getLogger(__name__)

EPOCH_INDEX_CODE_VERSION: int = 2

MetadataReaderType = ScatteredMetadataReader | _MetadataReader


@dataclasses.dataclass(frozen=True)
class DatasourceSpecs:
    """All what is needed to create a hash for pretokenized datasource"""

    prefix_path: str
    seed: int
    seq_len: int
    bos_id: int
    eos_id: int
    pad_id: int
    percentage_data: float  # between 0 and 1
    no_shuffling: bool  # only for testing !
    no_partial_sequence: bool  # for OCR data: don't break documents across sequences
    truncate_trailing_masked: bool = False

    def __post_init__(self) -> None:
        assert 1 >= self.percentage_data > 0, self


@dataclasses.dataclass
class DatasourceState:
    specs: DatasourceSpecs
    num_docs_per_epoch: int
    num_toks_per_epoch: int
    sequence_id: int
    num_toks_seen: int
    num_epochs: float  # num_toks_seen / n_toks_per_epoch

    def current_seq_id_and_increment_(self) -> int:
        seq_id = self.sequence_id
        self.sequence_id += 1
        self.num_toks_seen = self.sequence_id * self.specs.seq_len
        self.num_epochs = self.num_toks_seen / self.num_toks_per_epoch

        return seq_id


@dataclasses.dataclass
class _CurrentEpochIndex:
    epoch_id: int
    index: "IndexedJSONL"  # each line is a serialized DocumentIndexEntry


class PretokenizedImageEncoder(abc.ABC):
    @abc.abstractmethod
    def uuid(self) -> str:
        # unique id that depends on transforms, probabilities etc... for cache invalidation
        pass

    @abc.abstractmethod
    def to_infos_dict(self) -> dict[str, Any]:
        # to be able to have something human readable to inspect image encoder used
        pass

    @abc.abstractmethod
    def apply_transforms_and_encode_image(
        self,
        image: Image.Image,
        metadata: TransformImageMetadata,
        seed: tuple[int, ...],
    ) -> tuple[TrainingImageEncoding, TransformImageMetadata]:
        pass

    @abc.abstractmethod
    def precompute_transforms(
        self, metadata: TransformImageMetadata, seed: tuple[int, ...]
    ) -> tuple[int, TransformImageMetadata]:
        pass

    @abc.abstractmethod
    def get_transforms(self) -> Any:
        pass

    def encode_image_metadata(
        self, metadata: TransformImageMetadata, seed: tuple[int, ...]
    ) -> list[int]:
        """Build image tokens without loading pixels when the encoder supports it."""
        raise NotImplementedError(f"{type(self).__name__} has no metadata-only image path")


class MissingEpochIndex(Exception):
    def __init__(
        self, path: Path, specs: DatasourceSpecs, message: str | None = None
    ) -> None:
        self.path = path
        self.specs = specs
        if message is None:
            message = (
                f"Missing epoch index for path: {path} with specs: {specs}. "
                f"You can check for existing epoch_indexes for this datasource in {path.parent} "
                f"and/or build new indexes with allow_epoch_index_building"
            )
        assert isinstance(message, str)
        super().__init__(message)

    def __reduce__(self) -> tuple:
        return (type(self), (self.path, self.specs, self.args[0]))


class CorruptedEpochIndex(Exception):
    def __init__(self, path: Path, specs: DatasourceSpecs) -> None:
        self.path = path
        self.specs = specs
        super().__init__(
            f"Epoch index directory {path} exists but is incomplete. "
            f"This may indicate an ongoing sync or a previous crash. "
            f"If no sync is in progress, rebuild the index after removing it."
        )

    def __reduce__(self) -> tuple:
        return (type(self), (self.path, self.specs))


def build_sequence(
    docs: list[DocumentIndexEntry],
    image_encoder: PretokenizedImageEncoder | None,
    bin_reader: _FileBinReader | _ListTokensBinReader,
    bos_id: int,
    eos_id: int,
    is_masked_pretrain: bool,
    drop_bos: bool = False,
    load_image_pixels: bool = True,
    max_image_group_workers: int = 1,
    image_shard: tuple[int, int] | None = None,
) -> SequenceOfSamples:
    full_sequence: list[int] = []
    y_mask: list[bool] = []
    sizes: list[int] = []
    doc_ids: list[int] = []
    images: list[np.ndarray] = []
    images_positions: list[Span] = []
    doc_offset = 0
    timings: dict[str, float] = defaultdict(float)

    all_imgs = [img for doc in docs for img in doc.imgs]
    image_start, image_end = 0, len(all_imgs)
    if image_shard is not None:
        shard_rank, shard_size = image_shard
        # Contiguous partitions preserve image-token order when the bridge concatenates
        # encoder-rank outputs. Selection happens before storage reads and transforms.
        image_start = len(all_imgs) * shard_rank // shard_size
        image_end = len(all_imgs) * (shard_rank + 1) // shard_size
    loaded_images: list[Image.Image] = []
    if image_end > image_start and load_image_pixels:
        tick = time.time()
        loaded_images = load_contents_grouped(
            all_imgs[image_start:image_end], max_group_workers=max_image_group_workers
        )
        timings["image_load"] += time.time() - tick

    img_load_idx = 0
    global_image_idx = 0
    for doc in docs:
        doc_sequence = np.zeros(doc.size, dtype=DEFAULT_TOKEN_DTYPE)
        doc_y_mask: np.ndarray = np.ones(doc.size, dtype=np.bool_)
        img_doc_offset = 0
        for img in doc.imgs:
            assert img.metadata is not None
            assert img.seed is not None
            assert img.size is not None
            assert image_encoder is not None
            owns_image = image_start <= global_image_idx < image_end
            global_image_idx += 1
            if load_image_pixels and owns_image:
                image = loaded_images[img_load_idx]
                img_load_idx += 1
                tick = time.time()
                encoded_image, _ = image_encoder.apply_transforms_and_encode_image(
                    image,
                    metadata=img.metadata,
                    seed=img.seed,
                )
                timings["image_transform"] += time.time() - tick
                image_tokens = encoded_image.tokens
                images.append(encoded_image.image)
            else:
                tick = time.time()
                image_tokens = image_encoder.encode_image_metadata(
                    metadata=img.metadata,
                    seed=img.seed,
                )
                timings["image_metadata"] += time.time() - tick
            assert len(image_tokens) == img.size, (
                doc,
                len(image_tokens),
                img.size,
            )
            img_pos_in_doc = img.pos_in_text + img_doc_offset
            if owns_image:
                images_positions.append(
                    Span(
                        start=doc_offset + img_pos_in_doc,
                        end=doc_offset + img_pos_in_doc + img.size,
                    )
                )
            doc_sequence[img_pos_in_doc : img_pos_in_doc + img.size] = image_tokens
            doc_y_mask[img_pos_in_doc : img_pos_in_doc + img.size] = False
            img_doc_offset += img.size

        text_tokens = doc.get_text_tokens(bin_reader).tolist()
        doc_sequence[doc_y_mask] = text_tokens
        if doc.mask is not None:
            text_mask = _mask_to_array_mask(doc.mask)
            assert len(text_mask) == len(text_tokens), (
                len(text_mask),
                len(text_tokens),
            )
            assert np.all(doc_y_mask[doc_y_mask]), (doc_y_mask[doc_y_mask], doc.doc_id)
            doc_y_mask[doc_y_mask] = text_mask

        full_sequence.extend(doc_sequence.tolist())
        y_mask.extend(doc_y_mask.tolist())
        doc_offset = len(full_sequence)
        sizes.append(doc.size)
        doc_ids.append(doc.doc_id)

    assert sum(sizes) == len(full_sequence), (sum(sizes), len(full_sequence))
    x = full_sequence[:-1]
    y = [tok if tok != bos_id else eos_id for tok in full_sequence[1:]]
    y_mask = y_mask[1:]

    sizes[-1] -= 1
    if sizes[-1] == 0:
        sizes.pop(-1)
        doc_ids.pop(-1)

    if drop_bos:
        x = _replace_document_bos_with_eos(
            x=x,
            sizes=sizes,
            bos_id=bos_id,
            eos_id=eos_id,
        )

    return SequenceOfSamples(
        x=x,
        y=y,
        mask=y_mask,
        sizes=sizes,
        doc_ids=doc_ids,
        images=[torch.from_numpy(img).bfloat16() for img in images],
        image_positions=images_positions,
        image_pixels_loaded=load_image_pixels,
        num_pad_tokens=0,
        is_masked_pretrain=is_masked_pretrain,
        timings=timings,
    )


def _replace_document_bos_with_eos(
    *,
    x: list[int],
    sizes: list[int],
    bos_id: int,
    eos_id: int,
) -> list[int]:
    """Replace runtime document-boundary BOS inputs with EOS.

    Epoch indexes remain BOS-leading, and their sizes are preserved. Sizes are
    intentionally unchanged. When interdocument masking is disabled, they are
    logging metadata rather than attention boundaries, and the intended runtime
    streams are:

        x = [0,  1]   [EOS, 2, 3]    [EOS]
        y = [1, EOS]  [2,   3, EOS]  [4]
    """
    assert x, "Cannot replace document boundaries for an empty sequence"
    assert sizes, "Cannot replace document boundaries without document sizes"
    assert sum(sizes) == len(x), (sizes, len(x))

    boundary_offsets = set(np.cumsum(sizes)[:-1].tolist())
    assert all(x[offset] == bos_id for offset in boundary_offsets), (
        x,
        sizes,
        boundary_offsets,
        bos_id,
    )
    bos_offsets = {offset for offset, token_id in enumerate(x) if token_id == bos_id}
    expected_bos_offsets = boundary_offsets | ({0} if x[0] == bos_id else set())
    assert bos_offsets == expected_bos_offsets, (
        "BOS tokens must occur only at indexed document boundaries: "
        f"got {sorted(bos_offsets)}, expected {sorted(expected_bos_offsets)}"
    )

    replaced_x = np.asarray(x).copy()
    replaced_x[replaced_x == bos_id] = eos_id
    return replaced_x.tolist()


@dataclasses.dataclass(frozen=True)
class _DatasourceResources:
    prefix_paths: PrefixPaths
    source_metadata: PretokenizedDatasourceMetadataV2
    bin_reader: _FileBinReader
    index: _IndexReader
    image_encoder: PretokenizedImageEncoder | None


def _open_datasource_resources(
    *,
    specs: DatasourceSpecs,
    image_encoder_build_fn: Callable[[str], PretokenizedImageEncoder] | None,
    image_transform_type: str | None,
    source_metadata: PretokenizedDatasourceMetadataV2 | None,
    skip_images: bool,
) -> _DatasourceResources:
    prefix_paths = PrefixPaths.from_prefix(specs.prefix_path)
    assert prefix_paths.prefix.parent.exists(), (
        f"Pretokenized data directory not found: {prefix_paths.prefix.parent}. "
        f"The data for this source has likely not been pretokenized yet, "
        f"or the cache key / source_set configuration points to the wrong path."
    )
    if source_metadata is None:
        source_metadata = _reload_source_metadata_from_folder(prefix_paths)
    assert source_metadata is not None

    check_chunk_is_complete(
        prefix=prefix_paths, is_merged=True, source_metadata=source_metadata
    )

    bin_reader = _FileBinReader(str(prefix_paths.bin))
    has_images = source_metadata.has_image_metadata and not skip_images
    if source_metadata.has_image_metadata and skip_images:
        logger.info(f"Skipping images for source {specs.prefix_path}")

    image_encoder: PretokenizedImageEncoder | None = None
    if has_images:
        assert image_encoder_build_fn is not None
        assert image_transform_type is not None, (
            f"image_transform_type required for multimodal source: {specs.prefix_path}"
        )
        image_encoder = image_encoder_build_fn(image_transform_type)
        if source_metadata.masking_strategy not in (
            None,
            MaskingStrategy.INSTRUCT,
            MaskingStrategy.MASK_MMDOC_CONTENT,
        ):
            raise ValueError(
                f"Only no masking, MaskingStrategy.INSTRUCT, or "
                f"MaskingStrategy.MASK_MMDOC_CONTENT is supported for "
                f"multimodal datasources in the Megatron fork, got "
                f"{source_metadata.masking_strategy}"
            )

    index = _IndexReader(idx_path=str(prefix_paths.index))
    assert index.sequence_count == len(index.sequence_lengths), (
        f"{index.sequence_count} {len(index.sequence_lengths)}"
    )

    first_doc = bin_reader.read(
        dtype=DEFAULT_TOKEN_DTYPE,
        offset=index.sequence_pointers[0],
        count=index.sequence_lengths[0],
    )
    assert len(first_doc) > 1, (specs, len(first_doc))
    assert first_doc[-1] == specs.eos_id, specs
    assert first_doc[0] == specs.bos_id, specs
    assert all(t != specs.bos_id for t in first_doc[1:-1]), specs
    if source_metadata.sample_type != SampleType.INSTRUCT:
        assert all(t != specs.eos_id for t in first_doc[1:-1]), specs

    return _DatasourceResources(
        prefix_paths=prefix_paths,
        source_metadata=source_metadata,
        bin_reader=bin_reader,
        index=index,
        image_encoder=image_encoder,
    )


class Datasource:
    def __init__(
        self,
        specs: DatasourceSpecs,
        num_toks: int,
        cache_root_path: Path,
        image_encoder_build_fn: Callable[[str], PretokenizedImageEncoder] | None,
        image_transform_type: str | None,
        selection_strategy: SelectionStrategy | None,
        allow_epoch_index_building: bool,  # if False and need to build, then raise a `MissingEpochIndex`
        allow_partial_epoch_index_cache: bool = False,
        verbose: bool = False,
        # if not provided, it will look inside the folder (or parent folder for bwd compatibility)
        source_metadata: PretokenizedDatasourceMetadataV2 | None = None,
        skip_images: bool = False,
        drop_bos: bool = False,
        image_transform_workers: int = 1,
        load_image_pixels: bool = True,
        max_image_group_workers: int = 1,
        epoch_index_pds_filter: "PdsFilter | None" = None,
        image_shard: tuple[int, int] | None = None,
    ) -> None:
        assert image_transform_workers >= 1, image_transform_workers
        start = time.time()
        resources = _open_datasource_resources(
            specs=specs,
            image_encoder_build_fn=image_encoder_build_fn,
            image_transform_type=image_transform_type,
            source_metadata=source_metadata,
            skip_images=skip_images,
        )

        self._specs = specs
        self._source_metadata = resources.source_metadata
        self._drop_bos = drop_bos
        self._load_image_pixels = load_image_pixels
        self._max_image_group_workers = max_image_group_workers
        self._image_shard = image_shard
        self._prefix_path = specs.prefix_path
        self._bin_path = resources.prefix_paths.bin
        self._index_path = resources.prefix_paths.index
        self._image_metadata_path = resources.prefix_paths.image_metadata
        self._bin_reader = resources.bin_reader
        self.image_encoder = resources.image_encoder
        self._n_documents = resources.index.sequence_count

        self.cache_root_path = cache_root_path
        self.cache_root_path.mkdir(parents=True, exist_ok=True)

        def _log(msg: str) -> None:
            if verbose:
                logger.info(msg)

        _log(
            f"Datasource {specs.prefix_path} - preload token index in {time.time() - start}"
            f"\n - total number of sequences: {len(resources.index)}"
        )

        self._cum_sequence_count_per_epoch: list[int] = []
        self._sequence_count_per_epoch: list[int] = []
        self._epoch_stats: list[EpochStats] = []

        self._selection_strategy = selection_strategy

        num_sequences_to_validate = math.ceil(num_toks / specs.seq_len)
        self._num_sequences_to_validate = num_sequences_to_validate
        self._allow_partial_epoch_index_cache = allow_partial_epoch_index_cache

        epoch_id = 0
        current_num_sequences = 0
        while current_num_sequences <= num_sequences_to_validate:
            try:
                n_sequences, epoch_stats = _load_or_build_epoch_index(
                    resources=resources,
                    specs=specs,
                    epoch_id=epoch_id,
                    cache_root_path=self.cache_root_path,
                    selection_strategy=selection_strategy,
                    allow_epoch_index_building=allow_epoch_index_building,
                    skip_images=skip_images,
                    log_fn=_log,
                    image_transform_workers=image_transform_workers,
                    epoch_index_pds_filter=epoch_index_pds_filter,
                )
            except MissingEpochIndex:
                if allow_partial_epoch_index_cache and epoch_id > 0:
                    break
                raise

            current_num_sequences += n_sequences
            self._sequence_count_per_epoch.append(n_sequences)
            self._cum_sequence_count_per_epoch.append(current_num_sequences)
            self._epoch_stats.append(epoch_stats)
            epoch_id += 1

        self.current_epoch_index: _CurrentEpochIndex | None = None
        self.seq_len = specs.seq_len

    @staticmethod
    def prebuild_epoch_index(
        *,
        specs: DatasourceSpecs,
        epoch_id: int,
        cache_root_path: Path,
        image_encoder_build_fn: Callable[[str], PretokenizedImageEncoder] | None,
        image_transform_type: str | None,
        selection_strategy: SelectionStrategy | None,
        allow_epoch_index_building: bool,
        verbose: bool = False,
        source_metadata: PretokenizedDatasourceMetadataV2 | None = None,
        skip_images: bool = False,
        image_transform_workers: int = 1,
        epoch_index_pds_filter: "PdsFilter | None" = None,
    ) -> None:
        """Build or validate exactly one epoch without inspecting other epochs."""
        assert epoch_id >= 0, f"epoch_id must be non-negative, got {epoch_id}"
        resources = _open_datasource_resources(
            specs=specs,
            image_encoder_build_fn=image_encoder_build_fn,
            image_transform_type=image_transform_type,
            source_metadata=source_metadata,
            skip_images=skip_images,
        )
        cache_root_path.mkdir(parents=True, exist_ok=True)

        def _log(msg: str) -> None:
            if verbose:
                logger.info(msg)

        _load_or_build_epoch_index(
            resources=resources,
            specs=specs,
            epoch_id=epoch_id,
            cache_root_path=cache_root_path,
            selection_strategy=selection_strategy,
            allow_epoch_index_building=allow_epoch_index_building,
            skip_images=skip_images,
            log_fn=_log,
            image_transform_workers=image_transform_workers,
            epoch_index_pds_filter=epoch_index_pds_filter,
        )

    def _load_epoch_on_demand(self, seq_id: int) -> None:
        while seq_id >= self._cum_sequence_count_per_epoch[-1]:
            epoch_id = len(self._cum_sequence_count_per_epoch)
            epoch_index_dir = _get_epoch_index_path(
                epoch_indexes_root_dir=self.cache_root_path,
                specs=self._specs,
                epoch_id=epoch_id,
                image_encoder=self.image_encoder,
                selection_strategy=self._selection_strategy,
                has_loss_mask=self._source_metadata.has_loss_mask,
            )
            if epoch_index_dir.exists():
                if not _is_epoch_index_complete(epoch_index_dir):
                    raise CorruptedEpochIndex(path=epoch_index_dir, specs=self._specs)
            else:
                raise MissingEpochIndex(
                    path=epoch_index_dir,
                    specs=self._specs,
                    message=(
                        f"sequence_id={seq_id} for {self._specs.prefix_path} exceeds "
                        f"loaded epoch index coverage "
                        f"({self._cum_sequence_count_per_epoch[-1]} sequences across "
                        f"{len(self._cum_sequence_count_per_epoch)} epochs). "
                        f"Epoch index {epoch_id} not found on disk."
                    ),
                )

            epoch_stats = EpochStats.model_validate_json(
                _json_epoch_stats_path(epoch_index_dir).read_text()
            )
            n_sequences = epoch_stats.num_sequences
            current_num_sequences = self._cum_sequence_count_per_epoch[-1] + n_sequences
            self._sequence_count_per_epoch.append(n_sequences)
            self._cum_sequence_count_per_epoch.append(current_num_sequences)
            self._epoch_stats.append(epoch_stats)
            logger.info(
                "Loaded epoch %d on demand for %s (%d sequences, coverage now %d)",
                epoch_id,
                self._specs.prefix_path,
                n_sequences,
                current_num_sequences,
            )

    def _get_document_index_entry(self, seq_id: int) -> list[DocumentIndexEntry]:
        if (
            self._cum_sequence_count_per_epoch
            and seq_id >= self._cum_sequence_count_per_epoch[-1]
        ):
            self._load_epoch_on_demand(seq_id)

        if self.current_epoch_index is None or _epoch_id_changed(
            sequence_id=seq_id,
            cum_sequence_count_per_epoch=self._cum_sequence_count_per_epoch,
            prev_epoch_id=self.current_epoch_index.epoch_id,
        ):
            epoch_id = _get_epoch_id(
                sequence_id=seq_id,
                cum_sequence_count_per_epoch=self._cum_sequence_count_per_epoch,
            )
            logger.debug(f"Started iteration {epoch_id} over {self._prefix_path}")
            epoch_index_dir = _get_epoch_index_path(
                epoch_indexes_root_dir=self.cache_root_path,
                specs=self._specs,
                epoch_id=epoch_id,
                image_encoder=self.image_encoder,
                selection_strategy=self._selection_strategy,
                has_loss_mask=self._source_metadata.has_loss_mask,
            )
            jsonl_path = _jsonl_path(epoch_index_dir)
            jsonl_index_path = _jsonl_index_path(epoch_index_dir)
            if self.current_epoch_index is not None:
                self.current_epoch_index.index.close()
            self.current_epoch_index = _CurrentEpochIndex(
                epoch_id=epoch_id, index=IndexedJSONL(jsonl_path, jsonl_index_path)
            )

        assert self.current_epoch_index is not None
        local_seq_id = seq_id
        if self.current_epoch_index.epoch_id > 0:
            local_seq_id = (
                seq_id
                - self._cum_sequence_count_per_epoch[
                    self.current_epoch_index.epoch_id - 1
                ]
            )

        data = self.current_epoch_index.index.read(local_seq_id)
        return DocumentIndexEntry.list_from_bytes(data=data)

    def get_sequence(
        self,
        seq_id: int,
    ) -> SequenceOfSamples:
        docs = self._get_document_index_entry(seq_id)

        seq = build_sequence(
            docs=docs,
            image_encoder=self.image_encoder,
            bin_reader=self._bin_reader,
            bos_id=self._specs.bos_id,
            eos_id=self._specs.eos_id,
            is_masked_pretrain=self._source_metadata.has_loss_mask,
            drop_bos=self._drop_bos,
            load_image_pixels=self._load_image_pixels,
            max_image_group_workers=self._max_image_group_workers,
            image_shard=self._image_shard,
        )
        seq = pad_sequence(seq, mult_of=self.seq_len, pad_id=self._specs.pad_id)
        seq.check_pretrain_sample(
            seq_len=self.seq_len,
            bos_id=self._specs.bos_id,
            eos_id=self._specs.eos_id,
            drop_bos=self._drop_bos,
        )
        return seq

    def initial_state(self) -> "DatasourceState":
        assert len(self._cum_sequence_count_per_epoch) > 0
        return DatasourceState(
            specs=self._specs,
            sequence_id=0,
            num_docs_per_epoch=self._n_documents,
            num_toks_per_epoch=self._cum_sequence_count_per_epoch[0] * self.seq_len,
            num_epochs=0.0,
            num_toks_seen=0,
        )

    @property
    def src(self) -> str:
        return get_src_name(Path(self._source_metadata.original_source))

    def check_state(self, state: "DatasourceState") -> None:
        assert self._specs == state.specs, (
            f"Mismatch in specs between received state and datasource specs {self._specs} != {state}"
        )
        try:
            self._load_epoch_on_demand(
                state.sequence_id + self._num_sequences_to_validate - 1
            )
        except MissingEpochIndex:
            # Partial caches may end before the full future horizon, but must
            # contain the next unread sequence at the restored position.
            if (
                not self._allow_partial_epoch_index_cache
                or state.sequence_id >= self._cum_sequence_count_per_epoch[-1]
            ):
                raise

    @property
    def specs(self) -> DatasourceSpecs:
        return self._specs


def _get_source_epoch_index_folder(
    pretokenized_source: Path, epoch_index_folder: Path
) -> Path:
    assert str(pretokenized_source).startswith("/")
    source_hash = hashlib.sha256(str(pretokenized_source).encode()).hexdigest()
    return epoch_index_folder / source_hash


def _get_legacy_source_epoch_index_folder(
    pretokenized_source: Path, epoch_index_folder: Path
) -> Path:
    assert str(pretokenized_source).startswith("/")
    return epoch_index_folder / str(pretokenized_source)[1:]


def _epoch_index_infos_dict(
    epoch_id: int,
    specs: DatasourceSpecs,
    image_encoder: PretokenizedImageEncoder | None,
    epoch_index_pds_filter: "PdsFilter | None" = None,
) -> dict[str, Any]:
    # for human inspection of epoch_indexes
    # doing pydantic seemed overkill
    data = {
        "epoch_id": epoch_id,
        "source_specs": dataclasses.asdict(specs),
    }
    if image_encoder is not None:
        data["image_encoder_specs"] = image_encoder.to_infos_dict()
    if epoch_index_pds_filter is not None:
        from mistral_fork.mistral.data.pretokenized.components.filters import pds_filter_args

        data["epoch_index_pds_filter"] = {
            "name": epoch_index_pds_filter.value,
            **pds_filter_args(epoch_index_pds_filter),
        }
    return data


def _get_epoch_index_path(
    epoch_indexes_root_dir: Path,
    specs: DatasourceSpecs,
    epoch_id: int,
    selection_strategy: None | SelectionStrategy,
    image_encoder: PretokenizedImageEncoder | None,
    has_loss_mask: bool,
) -> Path:
    pretokenized_source = Path(specs.prefix_path).parent
    source_epoch_index_dir = _get_source_epoch_index_folder(
        pretokenized_source=pretokenized_source,
        epoch_index_folder=epoch_indexes_root_dir,
    )

    specs_dict = dataclasses.asdict(specs)
    # exclude no_partial_sequence when False to maintain backward compatibility with existing epoch index caches
    if not specs.no_partial_sequence:
        del specs_dict["no_partial_sequence"]
    truncate_active = specs.truncate_trailing_masked and has_loss_mask
    del specs_dict["truncate_trailing_masked"]

    hash = hashlib.md5(
        json.dumps(specs_dict, sort_keys=True, ensure_ascii=False).encode(
            encoding="utf-8"
        )
    ).hexdigest()
    img_hash = f"__img_{image_encoder.uuid()}" if image_encoder else ""
    truncate_suffix = "__truncate_trailing_masked" if truncate_active else ""
    version_suffix = (
        "" if EPOCH_INDEX_CODE_VERSION == 1 else f"__v{EPOCH_INDEX_CODE_VERSION}"
    )
    name = (
        str(Path(specs.prefix_path).stem)
        + f"__{hash}__epoch{epoch_id:03d}_seqlen{specs.seq_len}_seed{specs.seed}{version_suffix}{truncate_suffix}{img_hash}"
    )
    if selection_strategy is not None:
        name += f"_long_select{selection_strategy.get_hash()}"
    path = source_epoch_index_dir / name
    legacy_path = _get_legacy_source_epoch_index_folder(
        pretokenized_source=pretokenized_source,
        epoch_index_folder=epoch_indexes_root_dir,
    ) / name
    if not path.exists() and legacy_path.exists():
        return legacy_path
    return path


def _get_epoch_id(sequence_id: int, cum_sequence_count_per_epoch: list[int]) -> int:
    # ~5 epochs max, so binary search might be overkill
    epoch_id = 0
    while sequence_id >= cum_sequence_count_per_epoch[epoch_id]:
        epoch_id += 1
        assert epoch_id < len(cum_sequence_count_per_epoch), (
            f"sequence_id={sequence_id} exceeds loaded epoch index coverage "
            f"(max={cum_sequence_count_per_epoch[-1]} sequences across "
            f"{len(cum_sequence_count_per_epoch)} epochs)"
        )
    return epoch_id


def _epoch_id_changed(
    sequence_id: int, prev_epoch_id: int, cum_sequence_count_per_epoch: list[int]
) -> bool:
    return prev_epoch_id != _get_epoch_id(
        sequence_id=sequence_id,
        cum_sequence_count_per_epoch=cum_sequence_count_per_epoch,
    )


def _get_doc_size(doc_id: int, megatron_index_reader: _IndexReader) -> int:
    # for profiling
    return int(megatron_index_reader.sequence_lengths[doc_id])


def _get_doc_pointer(doc_id: int, megatron_index_reader: _IndexReader) -> int:
    # for profiling
    return int(megatron_index_reader.sequence_pointers[doc_id])


def _jsonl_path(epoch_index_dir: Path) -> Path:
    return epoch_index_dir / "epoch_index.jsonl"


def _jsonl_index_path(epoch_index_dir: Path) -> Path:
    return epoch_index_dir / "json_index.npy"


def _json_epoch_stats_path(epoch_index_dir: Path) -> Path:
    return epoch_index_dir / "epoch_stats.json"


def _json_epoch_index_infos_path(epoch_index_dir: Path) -> Path:
    return epoch_index_dir / "epoch_index_infos.json"


def _is_epoch_index_complete(epoch_index_dir: Path) -> bool:
    """Check if an epoch index directory is complete and valid.

    A directory is considered complete if all required files exist:
    - epoch_index.jsonl
    - json_index.npy
    - epoch_stats.json
    """
    if not epoch_index_dir.exists():
        return False

    required_files = [
        _jsonl_path(epoch_index_dir),
        _jsonl_index_path(epoch_index_dir),
        _json_epoch_stats_path(epoch_index_dir),
    ]
    return all(f.exists() for f in required_files)


def _cleanup_incomplete_epoch_index(
    epoch_index_dir: Path, log_fn: Callable[[str], None]
) -> None:
    """Remove an incomplete epoch index directory.

    This can happen if a previous process crashed while writing the index.

    Uses atomic rename to "claim" the directory before deletion, preventing
    race conditions when multiple workers try to clean up the same directory.
    """
    # Use atomic rename to claim ownership of the incomplete directory.
    # Only one worker will succeed in renaming it; others will get an error.
    cleanup_dir = epoch_index_dir.with_name(
        f".cleanup.{uuid.uuid4()!s}.{epoch_index_dir.name}"
    )
    try:
        epoch_index_dir.rename(cleanup_dir)
    except (FileNotFoundError, OSError):
        # Directory was already claimed/deleted by another worker, or doesn't exist
        log_fn(
            f"Incomplete epoch index at {epoch_index_dir} was already cleaned up by another worker"
        )
        return

    # We successfully claimed it, now delete
    log_fn(f"Found incomplete epoch index at {epoch_index_dir}, removing it to rebuild")
    shutil.rmtree(cleanup_dir)


class EpochStats(MistralBase):
    num_sequences: int = 0
    n_text_toks: int = 0
    n_imgs_toks: int = 0
    n_pad_toks: int = 0
    n_imgs: int = 0
    n_removed_toks_end_epoch: int = 0
    n_removed_imgs_end_epoch: int = 0
    # images that have n_toks > seq_len
    n_removed_imgs_too_big: int = Field(
        0,
        # backward compatibility
        validation_alias=AliasChoices(
            "n_removed_imgs_too_big", "n_removed_imgs_to_big"
        ),
    )
    # images that have n_toks < 1
    n_removed_imgs_too_small: int = 0
    # consecutive imgs with sum(toks) > seq_len
    n_removed_imgs_consecutive: int = 0
    # images whose object namespace is configured as unavailable
    n_removed_imgs_missing_bucket: int = 0

    # removed tokens and images when we have a full sequence masked
    n_removed_invalid_sequences: int = 0
    n_removed_toks_invalid_sequences: int = 0
    n_removed_imgs_invalid_sequences: int = 0

    n_removed_toks_trailing_masked: int = 0
    n_removed_imgs_trailing_masked: int = 0
    n_docs_truncated_trailing_masked: int = 0

    n_removed_docs_online_quarantine: int = 0
    n_removed_toks_online_quarantine: int = 0
    n_removed_imgs_online_quarantine: int = 0

    n_total_imgs: int = 0
    n_total_text_toks: int = 0
    n_docs_too_long: int = (
        0  # documents skipped because they exceed seq_len (no_partial_sequence)
    )

    def accumulate(self, other: "EpochStats") -> None:
        for field in EpochStats.model_fields:
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def to_dict(self) -> dict[str, int]:
        return self.model_dump(mode="python")

    @classmethod
    def from_dict(cls, dict: dict[str, int]) -> "EpochStats":
        return cls.model_validate(dict)


def _load_or_build_epoch_index(
    *,
    resources: _DatasourceResources,
    specs: DatasourceSpecs,
    epoch_id: int,
    cache_root_path: Path,
    selection_strategy: SelectionStrategy | None,
    allow_epoch_index_building: bool,
    skip_images: bool,
    log_fn: Callable[[str], None],
    image_transform_workers: int = 1,
    epoch_index_pds_filter: "PdsFilter | None" = None,
) -> tuple[int, EpochStats]:
    start = time.time()
    epoch_index_dir = _get_epoch_index_path(
        epoch_indexes_root_dir=cache_root_path,
        specs=specs,
        epoch_id=epoch_id,
        image_encoder=resources.image_encoder,
        selection_strategy=selection_strategy,
        has_loss_mask=resources.source_metadata.has_loss_mask,
    )
    if epoch_index_dir.exists():
        if not _is_epoch_index_complete(epoch_index_dir):
            raise CorruptedEpochIndex(path=epoch_index_dir, specs=specs)
        if epoch_index_pds_filter is not None:
            infos_path = _json_epoch_index_infos_path(epoch_index_dir)
            if not infos_path.is_file():
                raise RuntimeError(
                    f"Epoch index {epoch_index_dir} is missing filter metadata for "
                    f"{epoch_index_pds_filter.value}."
                )
            infos = json.loads(infos_path.read_text())
            configured_filter = infos.get("epoch_index_pds_filter", {}).get("name")
            if configured_filter != epoch_index_pds_filter.value:
                raise RuntimeError(
                    f"Epoch index {epoch_index_dir} was built with "
                    f"{configured_filter!r}, expected {epoch_index_pds_filter.value!r}."
                )
        with log_profile_if_slow(
            name=f"Datasource {specs.prefix_path} - load_epoch: {epoch_id}",
            threshold=1.0,
            add_log={},
        ):
            epoch_stats = EpochStats.model_validate_json(
                _json_epoch_stats_path(epoch_index_dir).read_text()
            )
            n_sequences = epoch_stats.num_sequences
        log_fn(
            f"Datasource {specs.prefix_path} - {epoch_id=} - {epoch_index_dir=} "
            f"preload cache {time.time() - start}"
        )
        return n_sequences, epoch_stats

    if not allow_epoch_index_building:
        raise MissingEpochIndex(path=epoch_index_dir, specs=specs)
    if not os.access(str(cache_root_path), os.W_OK):
        raise RuntimeError(
            f"You need to recompute some epoch_indexes, ie {epoch_index_dir}, "
            f"but {cache_root_path} is not writable"
        )

    infos_dict = _epoch_index_infos_dict(
        epoch_id=epoch_id,
        specs=specs,
        image_encoder=resources.image_encoder,
        epoch_index_pds_filter=epoch_index_pds_filter,
    )
    metadata_path = resources.prefix_paths.image_metadata
    if metadata_path.exists() and not skip_images:
        assert resources.image_encoder is not None, (
            "images metadata detected but no image encoder"
        )

    mask_reader: MaskReader | None = None
    if resources.source_metadata.has_loss_mask:
        mask_path = resources.prefix_paths.mask
        assert mask_path.exists(), mask_path
        mask_reader = MaskReader(mask_path)

    tmp_epoch_index_dir = epoch_index_dir.with_name(
        f"tmp.{uuid.uuid4()!s}.{epoch_index_dir.name}"
    )
    tmp_epoch_index_dir.mkdir(parents=True)
    tmp_jsonl_path = _jsonl_path(tmp_epoch_index_dir)
    tmp_json_index_path = _jsonl_index_path(tmp_epoch_index_dir)
    tmp_epoch_stats_path = _json_epoch_stats_path(tmp_epoch_index_dir)
    tmp_epoch_index_infos_path = _json_epoch_index_infos_path(tmp_epoch_index_dir)

    try:
        tmp_epoch_index_infos_path.write_text(
            json.dumps(infos_dict, indent=2, ensure_ascii=False)
        )

        start_build = time.time()
        log_fn(
            f"Datasource {specs.prefix_path}, starting to build epoch_index "
            f"for epoch {epoch_id}"
        )
        n_sequences, epoch_stats = _build_epoch_index(
            index=resources.index,
            n_documents=resources.index.sequence_count,
            specs=specs,
            epoch_id=epoch_id,
            metadata_path=(
                metadata_path
                if (
                    metadata_path.exists()
                    and not skip_images
                    and resources.image_encoder is not None
                )
                else None
            ),
            mask_reader=mask_reader,
            image_encoder=resources.image_encoder,
            selection_strategy=selection_strategy,
            output_jsonl_path=tmp_jsonl_path,
            output_index_path=tmp_json_index_path,
            image_transform_workers=image_transform_workers,
            epoch_index_pds_filter=epoch_index_pds_filter,
        )
        log_fn(
            f"Datasource {specs.prefix_path}, done building epoch_index for epoch "
            f"{epoch_id} in {time.time() - start_build}s"
        )

        tmp_epoch_stats_path.write_text(epoch_stats.model_dump_json(indent=2))
        try:
            tmp_epoch_index_dir.rename(epoch_index_dir)
        except (FileExistsError, OSError):
            log_fn(
                f"Datasource {specs.prefix_path} - {epoch_id=} "
                f"{epoch_index_dir=} already exists, another worker completed it "
                f"first. Cleaning up temp dir."
            )
            shutil.rmtree(tmp_epoch_index_dir)
            if not _is_epoch_index_complete(epoch_index_dir):
                raise RuntimeError(
                    f"Race condition: {epoch_index_dir} exists but is incomplete. "
                    f"This should not happen - another process may have crashed "
                    f"during rename."
                )
    except BaseException:
        logger.exception(
            f"Datasource {specs.prefix_path} - epoch {epoch_id} build failed, "
            f"cleaning up {tmp_epoch_index_dir}"
        )
        shutil.rmtree(tmp_epoch_index_dir, ignore_errors=True)
        raise

    log_fn(
        f"Datasource {specs.prefix_path} - {epoch_id=} {epoch_index_dir=} "
        f"all done in {time.time() - start}s"
    )
    return n_sequences, epoch_stats


class ConcatenateLogger:
    def __init__(
        self,
        n_documents: int,
        epoch_id: int,
        log_every_percent: float,
        source_path: str = "unknown",
    ) -> None:
        self.n_documents = n_documents
        self.epoch_id = epoch_id
        self.log_every_percent = log_every_percent
        self.source_path = source_path
        self._seen_docs = 0
        self._percent_threshold = log_every_percent
        self._proc_start = time.monotonic()
        assert psutil is not None
        self._process = psutil.Process()

    def on_document_processed(self, *, n_text_tokens: int, n_images: int) -> None:
        self._seen_docs += 1
        seen_percent = 100 * self._seen_docs / self.n_documents
        if seen_percent > self._percent_threshold:
            self._percent_threshold += self.log_every_percent
            elapsed = time.monotonic() - self._proc_start
            doc_per_s = self._seen_docs / elapsed
            tokens_per_s = n_text_tokens / elapsed
            images_per_s = n_images / elapsed
            assert psutil is not None
            host_mem = psutil.virtual_memory()
            used_by_process = self._process.memory_info().rss
            logger.info(
                f"[{self.source_path}] [Epoch {self.epoch_id}] "
                f"Processed {seen_percent:.1f}%, "
                f"{doc_per_s:.1f} doc/s, {tokens_per_s:.1f} tokens/s, "
                f"{images_per_s:.1f} img/s. "
                f"[used_py_process={int(used_by_process / 1024**3)}GB"
                f" - used_host={int(host_mem.used / 1024**3)}GB]"
            )


def _materialize_selected_sequence_index(
    index: _IndexReader,
    doc_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read selected index entries in document order, then restore selection order."""
    sorted_positions = np.argsort(doc_ids, kind="stable")
    sorted_doc_ids = doc_ids[sorted_positions]
    sorted_lengths = np.array(index.sequence_lengths[sorted_doc_ids], copy=True)
    sorted_pointers = np.array(index.sequence_pointers[sorted_doc_ids], copy=True)

    original_positions = np.empty_like(sorted_positions)
    original_positions[sorted_positions] = np.arange(
        len(sorted_positions), dtype=sorted_positions.dtype
    )
    return sorted_lengths[original_positions], sorted_pointers[original_positions]


def _shuffle_document_index(
    doc_ids: np.ndarray,
    sequence_lengths: np.ndarray,
    sequence_pointers: np.ndarray,
    seed: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle aligned document index arrays with the existing permutation."""
    assert len(doc_ids) == len(sequence_lengths) == len(sequence_pointers)
    positions = np.arange(len(doc_ids), dtype=np.int32)
    _rng(seed).shuffle(positions)
    return (
        doc_ids[positions],
        sequence_lengths[positions],
        sequence_pointers[positions],
    )


def _build_epoch_index(
    index: _IndexReader,
    n_documents: int,
    specs: DatasourceSpecs,
    epoch_id: int,
    selection_strategy: None | SelectionStrategy,
    metadata_path: Path | None,
    mask_reader: MaskReader | None,
    image_encoder: PretokenizedImageEncoder | None,
    output_jsonl_path: Path,
    output_index_path: Path,
    # One line per percent means 100 progress lines per source per epoch, on
    # every rank building an index. Ten is enough to watch a build progress.
    log_every_percent: float = 10.0,
    image_transform_workers: int = 1,
    epoch_index_pds_filter: "PdsFilter | None" = None,
) -> tuple[int, EpochStats]:
    """

    Main logic where we build the global mapping sequence_id -> DocumentIndexEntry that allows the concatenation of all documents at runtime to return the fixed sized sequence

    Note on epoch shuffling

        We drop the last sequence of each epoch if it is not full, making it easier to have a per epoch cache

        But we should revisit it (especially for long seq_len)

    """
    logger.info(
        f"Building epoch index for {specs.prefix_path}, epoch_id={epoch_id},specs={specs}"
    )
    item_size = DType.size(DEFAULT_TOKEN_DTYPE)

    if specs.percentage_data < 1:
        # this allow to make sure each datasource has at least seqlen tokens to avoid empty epoch indexes
        min_tokens = specs.seq_len
        doc_ids = _select_subset_of_documents(
            specs=specs, n_documents=n_documents, index=index, min_tokens=min_tokens
        )
    else:
        assert n_documents <= np.iinfo(np.int32).max, (
            f"n_documents={n_documents} exceeds int32 max ({np.iinfo(np.int32).max})"
        )
        doc_ids = np.arange(n_documents, dtype=np.int32)

    if selection_strategy is not None:
        documents_sizes = np.array(
            [index.sequence_lengths[doc_id] for doc_id in doc_ids]
        )
        doc_ids = selection_strategy.select(
            documents_sizes=documents_sizes,
            documents=doc_ids,
        )
    assert len(doc_ids) > 0, (
        f"No documents selected after selection strategies for {specs.prefix_path}, "
        f"epoch_id={epoch_id}. This will result in empty epoch index."
        "Consider increasing percentage_data or adjusting long_context_proportion."
    )

    if selection_strategy is None:
        # Without a selection strategy, doc_ids is the contiguous prefix produced
        # above. Copy the index sequentially before doc_ids is shuffled so later
        # batches do not issue random reads against the memory-mapped index file.
        assert int(doc_ids[0]) == 0 and int(doc_ids[-1]) == len(doc_ids) - 1
        sequence_lengths = np.array(index.sequence_lengths[: len(doc_ids)], copy=True)
        sequence_pointers = np.array(
            index.sequence_pointers[: len(doc_ids)], copy=True
        )
    else:
        sequence_lengths, sequence_pointers = _materialize_selected_sequence_index(
            index=index,
            doc_ids=doc_ids,
        )
    total_tokens = int(np.sum(sequence_lengths)) - len(doc_ids)

    logger.info(f"Selected {len(doc_ids)} documents.")
    if specs.no_shuffling:
        logger.warning(
            f"Not using shuffling for datasource {specs.prefix_path} -- epoch_id: {epoch_id}"
        )
        pass
    else:
        # 0 to be different with previous shuffling
        doc_ids, sequence_lengths, sequence_pointers = _shuffle_document_index(
            doc_ids=doc_ids,
            sequence_lengths=sequence_lengths,
            sequence_pointers=sequence_pointers,
            seed=(specs.seed, epoch_id, n_documents, 0),
        )

    scattered_metadata: ScatteredMetadataReader | None = None
    scatter_dir = output_jsonl_path.parent / "metadata_scatter"

    try:
        if metadata_path is not None:
            scatter_dir.mkdir()
            scattered_metadata = ScatteredMetadataReader(
                metadata_path=metadata_path,
                doc_ids=doc_ids,
                tmp_dir=scatter_dir,
            )

        stats_dict = EpochStats(n_removed_imgs_too_big=0).to_dict()
        image_transform_pool = None
        if image_encoder is not None and image_transform_workers > 1:
            logger.info(
                "[Epoch %s] Starting %s image-transform workers",
                epoch_id,
                image_transform_workers,
            )
            image_transform_pool = multiprocessing.Pool(
                processes=image_transform_workers,
                initializer=_initialize_image_transform_worker,
                initargs=(image_encoder,),
            )

        try:
            document_iterator = _make_document_entries(
                doc_ids=doc_ids,
                sequence_lengths=sequence_lengths,
                sequence_pointers=sequence_pointers,
                epoch_id=epoch_id,
                metadata_reader=scattered_metadata,
                image_encoder=image_encoder,
                item_size=item_size,
                n_documents=len(doc_ids),
                specs=specs,
                mask_reader=mask_reader,
                stats_dict=stats_dict,
                image_transform_pool=image_transform_pool,
                epoch_index_pds_filter=epoch_index_pds_filter,
            )
            concatenated_doc_iterator = concatenate_iterator(
                chunked_doc_iterator=document_iterator,
                seq_len=specs.seq_len,
                no_partial_sequence=specs.no_partial_sequence,
                stats_dict=stats_dict,
                concat_logger=ConcatenateLogger(
                    n_documents=len(doc_ids),
                    epoch_id=epoch_id,
                    log_every_percent=log_every_percent,
                    source_path=str(specs.prefix_path),
                ),
                truncate_trailing_masked=specs.truncate_trailing_masked,
            )

            has_images = scattered_metadata is not None
            estimated_n_sequences = total_tokens // specs.seq_len
            avg_docs_per_seq = max(1, len(doc_ids) // max(estimated_n_sequences, 1))
            text_bytes = estimated_n_sequences * avg_docs_per_seq * 80
            image_bytes = (
                max(len(doc_ids), total_tokens // 2048) * 700 if has_images else 0
            )
            estimated_total_bytes = text_bytes + image_bytes
            max_bucket_bytes = 256 * 1024**2
            n_buckets = max(1, math.ceil(estimated_total_bytes / max_bucket_bytes))
            logger.info(
                f"[Epoch {epoch_id}] Scatter config: {n_buckets} buckets "
                f"(estimated {estimated_total_bytes / 1024**3:.1f}GB, "
                f"text={text_bytes / 1024**3:.1f}GB, "
                f"images={image_bytes / 1024**3:.1f}GB)"
            )

            if specs.no_shuffling:
                write_start = time.time()
                n_sequences = _write_jsonl_and_index_streaming(
                    output_jsonl_path,
                    output_index_path,
                    (
                        _epoch_index_entry_to_bytes(seq)
                        for seq in concatenated_doc_iterator
                    ),
                )
                write_time = time.time() - write_start
                logger.info(
                    f"[Epoch {epoch_id}] Serialized and wrote {n_sequences} sequences "
                    f"in {write_time:.2f}s"
                )
            else:
                scatter_tmp_dir = output_jsonl_path.parent / "scatter"
                scatter_tmp_dir.mkdir()

                try:
                    scatter_start = time.time()
                    scatter_rng = np.random.default_rng(
                        (specs.seed, epoch_id, n_documents, 1)
                    )
                    _scatter_epoch_sequences(
                        concatenated_doc_iterator,
                        scatter_tmp_dir,
                        scatter_rng,
                        n_buckets,
                    )
                    scatter_time = time.time() - scatter_start

                    gather_start = time.time()
                    n_sequences = _write_jsonl_and_index_batched(
                        output_jsonl_path,
                        output_index_path,
                        _gather_epoch_sequences_batched(
                            scatter_tmp_dir, scatter_rng, n_buckets, epoch_id
                        ),
                    )
                    gather_time = time.time() - gather_start
                    logger.info(
                        f"[Epoch {epoch_id}] Shuffled and wrote {n_sequences} sequences: "
                        f"prepack_and_scatter={scatter_time:.2f}s, gather={gather_time:.2f}s"
                    )
                finally:
                    shutil.rmtree(scatter_tmp_dir, ignore_errors=True)
        finally:
            if image_transform_pool is not None:
                image_transform_pool.terminate()
                image_transform_pool.join()

        stats = EpochStats.from_dict(stats_dict)

        assert (
            stats.n_text_toks
            + stats.n_removed_toks_end_epoch
            + stats.n_removed_toks_invalid_sequences
            + stats.n_removed_toks_trailing_masked
            == stats.n_total_text_toks
        ), stats
        assert (
            stats.n_imgs
            + stats.n_removed_imgs_end_epoch
            + stats.n_removed_imgs_too_big
            + stats.n_removed_imgs_too_small
            + stats.n_removed_imgs_consecutive
            + stats.n_removed_imgs_invalid_sequences
            + stats.n_removed_imgs_missing_bucket
            + stats.n_removed_imgs_trailing_masked
            == stats.n_total_imgs
        ), stats

        assert n_sequences > 0, (
            f"Selected {len(doc_ids)} documents but produced no sequences."
        )

        return n_sequences, stats
    finally:
        if scattered_metadata is not None:
            scattered_metadata.cleanup()
        shutil.rmtree(scatter_dir, ignore_errors=True)


def _scatter_epoch_sequences(
    concatenated_doc_iterator: Iterator[list[DocumentIndexEntry]],
    tmp_dir: Path,
    scatter_rng: np.random.Generator,
    n_buckets: int,
) -> None:
    bucket_files = [open(tmp_dir / f"bucket_{k}.bin", "wb") for k in range(n_buckets)]
    batch_size = 10000
    bucket_ids = scatter_rng.integers(0, n_buckets, size=batch_size)
    idx = 0
    try:
        for sequence in concatenated_doc_iterator:
            if idx >= batch_size:
                bucket_ids = scatter_rng.integers(0, n_buckets, size=batch_size)
                idx = 0
            serialized = _epoch_index_entry_to_bytes(sequence)
            bucket_file = bucket_files[bucket_ids[idx]]
            bucket_file.write(len(serialized).to_bytes(4, "little"))
            bucket_file.write(serialized)
            idx += 1
    finally:
        for file in bucket_files:
            file.close()


def _load_bucket(bucket_path: Path) -> list[bytes]:
    records: list[bytes] = []
    with open(bucket_path, "rb") as file:
        while True:
            len_bytes = file.read(4)
            if not len_bytes:
                break
            length = int.from_bytes(len_bytes, "little")
            records.append(file.read(length))
    return records


def _gather_epoch_sequences_batched(
    tmp_dir: Path,
    rng: np.random.Generator,
    n_buckets: int,
    epoch_id: int,
) -> Iterator[list[bytes]]:
    assert psutil is not None
    process = psutil.Process()
    total_records = 0
    peak_rss_bytes = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_load_bucket, tmp_dir / "bucket_0.bin")

        for bucket_id in range(n_buckets):
            bucket_path = tmp_dir / f"bucket_{bucket_id}.bin"
            records = future.result()

            if bucket_id + 1 < n_buckets:
                next_bucket_path = tmp_dir / f"bucket_{bucket_id + 1}.bin"
                future = executor.submit(_load_bucket, next_bucket_path)

            current_rss = process.memory_info().rss
            peak_rss_bytes = max(peak_rss_bytes, current_rss)

            rng.shuffle(records)
            total_records += len(records)
            yield records
            bucket_path.unlink()

    peak_mb = peak_rss_bytes / (1024**2)
    logger.debug(
        f"[Epoch {epoch_id}] Gather stats: {total_records} records from "
        f"{n_buckets} buckets, gather_peak_memory={peak_mb:.0f}MB"
    )


def concatenate_iterator(
    chunked_doc_iterator: Iterator[DocumentIndexEntry | None],
    seq_len: int,
    no_partial_sequence: bool,
    stats_dict: dict[str, int] | None = None,
    concat_logger: ConcatenateLogger | None = None,
    yield_last_sequence: bool = False,
    truncate_trailing_masked: bool = False,
) -> Iterator[list[DocumentIndexEntry]]:
    if stats_dict is None:
        stats_dict = EpochStats(n_removed_imgs_too_big=0).to_dict()

    current_sequence: list[DocumentIndexEntry] = []
    current_sequence_size = 0
    for document_entry in chunked_doc_iterator:
        if document_entry is None:
            continue
        current_sequence, current_sequence_size, finished_sequences = (
            _concatenate_documents_into_fixed_size_sequences(
                document_entry=document_entry,
                current_sequence=current_sequence,
                current_sequence_size=current_sequence_size,
                seq_len=seq_len,
                no_partial_sequence=no_partial_sequence,
                stats_dict=stats_dict,
                truncate_trailing_masked=truncate_trailing_masked,
            )
        )
        stats_dict["num_sequences"] += len(finished_sequences)
        yield from finished_sequences

        if concat_logger is not None:
            concat_logger.on_document_processed(
                n_text_tokens=stats_dict["n_total_text_toks"],
                n_images=stats_dict["n_total_imgs"],
            )

    if len(current_sequence) > 0:
        if yield_last_sequence:
            yield current_sequence
        else:
            for doc in current_sequence:
                stats_dict["n_removed_toks_end_epoch"] += doc.text.size
                stats_dict["n_removed_imgs_end_epoch"] += len(doc.imgs)
                # # correct for dropped documents
                # stats_dict["n_text_toks"] -= doc.text.size
                # stats_dict["n_imgs"] -= len(doc.imgs)
                # stats_dict["n_imgs_toks"] -= sum(img.size for img in doc.imgs)


def _build_doc(
    doc_id: int,
    doc_size: int,
    doc_pointer: int,
    epoch_id: int,
    specs: DatasourceSpecs,
    images: list[ImageIndexEntry] | None,
    image_encoder: PretokenizedImageEncoder | None,
    transformed_images: "_TransformedDocumentImages | None",
    mask_reader: MaskReader | None,
    stats_dict: dict[str, int],
    n_documents: int,
    item_size: int,
) -> DocumentIndexEntry | None:
    text_doc_size_without_eos = doc_size - 1  # we remove EOS at the end
    stats_dict["n_total_text_toks"] += text_doc_size_without_eos
    assert text_doc_size_without_eos >= 2
    assert doc_pointer % item_size == 0, (doc_pointer, item_size)
    doc_offset = doc_pointer // item_size
    # Snapshot removal stats before _make_image_entries so we can restore them
    # if the document ends up being skipped (too long with no_partial_sequence)
    n_removed_too_small_before = stats_dict["n_removed_imgs_too_small"]
    n_removed_too_big_before = stats_dict["n_removed_imgs_too_big"]
    n_removed_consecutive_before = stats_dict["n_removed_imgs_consecutive"]
    n_removed_missing_bucket_before = stats_dict["n_removed_imgs_missing_bucket"]
    if images is not None:
        assert image_encoder is not None
        assert all(s.pos_in_text < text_doc_size_without_eos for s in images), (
            doc_id,
            "images can't be at the end, or just before EOS",
            text_doc_size_without_eos,
            [s.pos_in_text for s in images],
        )
        stats_dict["n_total_imgs"] += len(images)

        if transformed_images is None:
            image_entries, n_image_tokens = _make_image_entries(
                specs,
                images,
                image_encoder,
                (specs.seed, epoch_id, n_documents, doc_id),
                stats_dict,
            )
        else:
            image_entries = transformed_images.images
            n_image_tokens = transformed_images.n_tokens
            for key, count in transformed_images.removal_stats.items():
                stats_dict[key] += count
    else:
        image_entries, n_image_tokens = [], 0

    doc_size = text_doc_size_without_eos + n_image_tokens
    if specs.no_partial_sequence and doc_size > specs.seq_len:
        # document is too long, skipping it
        stats_dict["n_docs_too_long"] += 1
        stats_dict["n_total_text_toks"] -= text_doc_size_without_eos
        if images is not None:
            stats_dict["n_total_imgs"] -= len(images)
            # Restore image removal stats that were incremented by _make_image_entries
            stats_dict["n_removed_imgs_too_small"] = n_removed_too_small_before
            stats_dict["n_removed_imgs_too_big"] = n_removed_too_big_before
            stats_dict["n_removed_imgs_consecutive"] = n_removed_consecutive_before
            stats_dict["n_removed_imgs_missing_bucket"] = (
                n_removed_missing_bucket_before
            )

        logger.warning(
            f"Removed a too big document ({doc_size} > {specs.seq_len}) "
            f"in {specs.prefix_path} doc_id={doc_id}"
        )
        return None

    if mask_reader is not None:
        doc_mask = mask_reader.get_mask_segments_for_document(doc_id)
        mask = doc_mask.to_mask_index_entry()
        _check_mask_correctness(mask)
        # since we are trimming eos we are removing 1 token from last segment
        sizes, masks = mask
        last_segment_len = sizes[-1]
        if last_segment_len > 1:
            sizes[-1] = last_segment_len - 1
        else:
            sizes.pop()
            masks.pop()
        assert _mask_size(mask) == text_doc_size_without_eos, (
            _mask_size(mask),
            text_doc_size_without_eos,
        )
    else:
        mask = None

    return DocumentIndexEntry(
        doc_id,
        TextIndexEntry(doc_offset, text_doc_size_without_eos),
        image_entries,
        text_doc_size_without_eos + n_image_tokens,
        mask=mask,
    )


def _make_document_entries(
    doc_ids: np.ndarray,
    sequence_lengths: np.ndarray,
    sequence_pointers: np.ndarray,
    epoch_id: int,
    metadata_reader: MetadataReaderType | None,
    image_encoder: PretokenizedImageEncoder | None,
    item_size: int,
    n_documents: int,
    specs: DatasourceSpecs,
    mask_reader: MaskReader | None,
    stats_dict: dict[str, int],
    image_transform_pool: Pool | None,
    epoch_index_pds_filter: "PdsFilter | None" = None,
) -> Generator[DocumentIndexEntry | None, None, None]:
    image_matches_pds_filter = None
    if epoch_index_pds_filter is not None:
        from mistral_fork.mistral.data.pretokenized.components.filters import (
            image_index_entry_is_pds_quarantined as image_matches_pds_filter,
        )

    batch_size = 32768 if image_transform_pool is not None else 1_000_000
    total_docs = len(doc_ids)
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        batch_docs = doc_ids[batch_start:batch_end]
        batch_sizes = sequence_lengths[batch_start:batch_end]
        batch_pointers = sequence_pointers[batch_start:batch_end]
        images_by_document = (
            [metadata_reader[batch_start + i] for i in range(len(batch_docs))]
            if metadata_reader is not None
            else [None] * len(batch_docs)
        )
        quarantined_document_indices = {
            document_index
            for document_index, images in enumerate(images_by_document)
            if images
            and image_matches_pds_filter is not None
            and any(image_matches_pds_filter(image, epoch_index_pds_filter) for image in images)
        }
        stats_dict["n_removed_docs_online_quarantine"] += len(quarantined_document_indices)
        stats_dict["n_removed_toks_online_quarantine"] += sum(
            int(batch_sizes[document_index]) - 1
            for document_index in quarantined_document_indices
        )
        stats_dict["n_removed_imgs_online_quarantine"] += sum(
            len(images_by_document[document_index])
            for document_index in quarantined_document_indices
        )
        transformed_by_document: dict[int, _TransformedDocumentImages] = {}
        if image_transform_pool is not None:
            tasks: list[_ImageTransformTask] = []
            missing_buckets_by_document: dict[int, int] = defaultdict(int)
            sorted_images_by_document: list[list[ImageIndexEntry]] = []
            for document_index, images in enumerate(images_by_document):
                if document_index in quarantined_document_indices or not images:
                    sorted_images_by_document.append([])
                    continue
                sorted_images = _sorted_images(images)
                sorted_images_by_document.append(sorted_images)
                partial_seed = (
                    specs.seed,
                    epoch_id,
                    n_documents,
                    int(batch_docs[document_index]),
                )
                for image_index, image in enumerate(sorted_images):
                    if _image_is_in_missing_bucket(image):
                        missing_buckets_by_document[document_index] += 1
                        continue
                    assert image.metadata is not None
                    image.seed = (*partial_seed, image.img_id_in_doc)
                    tasks.append(
                        _ImageTransformTask(
                            document_index=document_index,
                            image_index=image_index,
                            metadata=image.metadata,
                            seed=image.seed,
                        )
                    )

            images_by_result_document: dict[int, list[ImageIndexEntry]] = defaultdict(
                list
            )
            for document_index, image_index, image_size in image_transform_pool.imap(
                _transform_image_worker, tasks, chunksize=256
            ):
                image = sorted_images_by_document[document_index][image_index]
                image.size = image_size
                images_by_result_document[document_index].append(image)

            for document_index, images in enumerate(images_by_document):
                if not images:
                    continue
                local_stats = EpochStats(n_removed_imgs_too_big=0).to_dict()
                local_stats["n_removed_imgs_missing_bucket"] = (
                    missing_buckets_by_document[document_index]
                )
                transformed_images, n_tokens = _finalize_image_entries(
                    specs=specs,
                    images=images_by_result_document[document_index],
                    stats_dict=local_stats,
                )
                transformed_by_document[document_index] = _TransformedDocumentImages(
                    images=transformed_images,
                    n_tokens=n_tokens,
                    removal_stats={
                        key: local_stats[key] for key in _IMAGE_REMOVAL_STAT_KEYS
                    },
                )

        for i, doc_id in enumerate(batch_docs):
            if i in quarantined_document_indices:
                yield None
                continue
            yield _build_doc(
                doc_id=int(doc_id),
                doc_size=int(batch_sizes[i]),
                doc_pointer=int(batch_pointers[i]),
                epoch_id=epoch_id,
                specs=specs,
                images=images_by_document[i],
                image_encoder=image_encoder,
                transformed_images=transformed_by_document.get(i),
                stats_dict=stats_dict,
                n_documents=n_documents,
                item_size=item_size,
                mask_reader=mask_reader,
            )


_IMAGE_REMOVAL_STAT_KEYS = (
    "n_removed_imgs_too_small",
    "n_removed_imgs_too_big",
    "n_removed_imgs_consecutive",
    "n_removed_imgs_missing_bucket",
)


@dataclasses.dataclass(frozen=True)
class _ImageTransformTask:
    document_index: int
    image_index: int
    metadata: TransformImageMetadata
    seed: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _TransformedDocumentImages:
    images: list[ImageIndexEntry]
    n_tokens: int
    removal_stats: dict[str, int]


_IMAGE_TRANSFORM_WORKER_ENCODER: PretokenizedImageEncoder | None = None


def _initialize_image_transform_worker(
    image_encoder: PretokenizedImageEncoder,
) -> None:
    global _IMAGE_TRANSFORM_WORKER_ENCODER
    _IMAGE_TRANSFORM_WORKER_ENCODER = image_encoder


def _transform_image_worker(task: _ImageTransformTask) -> tuple[int, int, int]:
    assert _IMAGE_TRANSFORM_WORKER_ENCODER is not None
    image_size, _ = _IMAGE_TRANSFORM_WORKER_ENCODER.precompute_transforms(
        metadata=task.metadata,
        seed=task.seed,
    )
    return task.document_index, task.image_index, image_size


def _sorted_images(images: list[ImageIndexEntry]) -> list[ImageIndexEntry]:
    return sorted(images, key=lambda image: (image.pos_in_text, image.img_id_in_doc))


def _image_is_in_missing_bucket(image: ImageIndexEntry) -> bool:
    return is_object_path_dropped(image.tar_path)


def _precompute_image_transform(
    *,
    image: ImageIndexEntry,
    image_encoder: PretokenizedImageEncoder,
    seed: tuple[int, ...],
) -> ImageIndexEntry:
    assert image.metadata is not None
    image.seed = seed
    image.size, _ = image_encoder.precompute_transforms(
        metadata=copy.deepcopy(image.metadata),
        seed=image.seed,
    )
    return image


def _finalize_image_entries(
    *,
    specs: DatasourceSpecs,
    images: list[ImageIndexEntry],
    stats_dict: dict[str, int],
) -> tuple[list[ImageIndexEntry], int]:
    new_images: list[ImageIndexEntry] = []
    n_image_tokens = 0

    pos_to_images: dict[int, list[ImageIndexEntry]] = defaultdict(list)
    for image in images:
        pos_to_images[image.pos_in_text].append(image)

    for _, image_group in sorted(pos_to_images.items()):
        new_images_group: list[ImageIndexEntry] = []
        for image in sorted(image_group, key=lambda item: item.img_id_in_doc):
            assert image.size is not None
            if image.size == 0:
                stats_dict["n_removed_imgs_too_small"] += 1
                continue
            if image.size >= specs.seq_len:
                stats_dict["n_removed_imgs_too_big"] += 1
                continue
            n_image_tokens += image.size
            new_images_group.append(image)

        while sum(image.size for image in new_images_group) >= specs.seq_len:
            removed_image = new_images_group.pop(0)
            assert removed_image.size is not None
            n_image_tokens -= removed_image.size
            stats_dict["n_removed_imgs_consecutive"] += 1

        new_images.extend(new_images_group)

    assert n_image_tokens == sum(image.size for image in new_images)
    return new_images, n_image_tokens


def _make_image_entries(
    specs: DatasourceSpecs,
    images: list[ImageIndexEntry],
    image_encoder: PretokenizedImageEncoder | None,
    partial_transform_seed: tuple[int, int, int, int],
    stats_dict: dict[str, int],
) -> tuple[list[ImageIndexEntry], int]:
    if len(images) == 0:
        return [], 0

    transformed_images = []
    assert image_encoder is not None
    for image in _sorted_images(images):
        if _image_is_in_missing_bucket(image):
            stats_dict["n_removed_imgs_missing_bucket"] += 1
            continue
        validate_s3_path(image.tar_path)
        transformed_images.append(
            _precompute_image_transform(
                image=image,
                image_encoder=image_encoder,
                seed=(*partial_transform_seed, image.img_id_in_doc),
            )
        )
    return _finalize_image_entries(
        specs=specs,
        images=transformed_images,
        stats_dict=stats_dict,
    )


def _concatenate_documents_into_fixed_size_sequences(
    document_entry: DocumentIndexEntry,
    current_sequence: list[DocumentIndexEntry],
    current_sequence_size: int,
    seq_len: int,
    no_partial_sequence: bool,
    stats_dict: dict[str, int],
    truncate_trailing_masked: bool = False,
) -> tuple[list[DocumentIndexEntry], int, list[list[DocumentIndexEntry]]]:
    finished_sequences: list[list[DocumentIndexEntry]] = []
    target_size = seq_len + 1
    missing_tokens = target_size - current_sequence_size

    if document_entry.size < missing_tokens:
        # we can append the whole document
        current_sequence.append(document_entry)
        current_sequence_size += document_entry.size
        return current_sequence, current_sequence_size, finished_sequences

    if no_partial_sequence:
        # the doc didn't fit in the current sequence and we cannot split it.
        # mark current sequence as finished (with padding), start a new one
        assert document_entry.size < target_size, (
            f"Document too big, should have been removed before! "
            f"{document_entry.doc_id=} {document_entry.size=} {seq_len=}"
        )

        finished_sequences.append(current_sequence)
        stats_dict["n_text_toks"] += sum(doc.text.size for doc in current_sequence)
        stats_dict["n_imgs"] += sum(len(doc.imgs) for doc in current_sequence)
        stats_dict["n_imgs_toks"] += sum(
            img.size for doc in current_sequence for img in doc.imgs
        )
        stats_dict["n_pad_toks"] += missing_tokens

        # Start new sequence with this document
        current_sequence = [document_entry]
        current_sequence_size = document_entry.size
        return current_sequence, current_sequence_size, finished_sequences

    mask_cursor = (
        _MaskCursor(document_entry.mask) if document_entry.mask is not None else None
    )

    while True:
        # we need to split the document, maybe multiple times
        assert missing_tokens > 0, (missing_tokens, current_sequence_size, seq_len)
        assert missing_tokens <= target_size, (missing_tokens, target_size)
        assert document_entry.size > 0, document_entry
        assert document_entry.size >= missing_tokens

        left, right = document_entry.split(missing_tokens, mask_cursor)
        # we can not end a sequence on a image
        split_on_text = left.size == missing_tokens
        split_overlap = 1 if split_on_text else 0

        left_to_append: DocumentIndexEntry | None = left
        left_truncated = False
        if truncate_trailing_masked and left.mask is not None:
            truncated_left, removed_toks, removed_imgs = left.truncate_trailing_masked(
                keep_first_token=len(current_sequence) > 0
            )
            if removed_toks > 0 or removed_imgs > 0:
                stats_dict["n_removed_toks_trailing_masked"] += (
                    removed_toks - split_overlap
                )
                stats_dict["n_removed_imgs_trailing_masked"] += removed_imgs
                stats_dict["n_docs_truncated_trailing_masked"] += 1
                split_overlap = 0
                left_to_append = truncated_left
                left_truncated = True

        if left_to_append is None:
            assert len(current_sequence) == 0, (
                "dropped a fully-truncated last piece while a preceding doc is in "
                "the sequence"
            )
        else:
            current_sequence.append(left_to_append)
            current_sequence_size += left_to_append.size

        padding_tokens = target_size - current_sequence_size
        if split_on_text and not left_truncated:
            assert padding_tokens == 0

        if _is_valid_sequence(current_sequence):
            finished_sequences.append(current_sequence)
            stats_dict["n_text_toks"] += (
                sum(doc.text.size for doc in current_sequence) - split_overlap
            )
            stats_dict["n_imgs"] += sum(len(doc.imgs) for doc in current_sequence)
            stats_dict["n_imgs_toks"] += sum(
                img.size for doc in current_sequence for img in doc.imgs
            )
            stats_dict["n_pad_toks"] += padding_tokens
        else:
            stats_dict["n_removed_invalid_sequences"] += 1
            stats_dict["n_removed_toks_invalid_sequences"] += (
                sum(doc.text.size for doc in current_sequence) - split_overlap
            )
            stats_dict["n_removed_imgs_invalid_sequences"] += sum(
                len(doc.imgs) for doc in current_sequence
            )

        document_entry = right
        current_sequence = []
        current_sequence_size = 0
        missing_tokens = target_size

        if document_entry.size < missing_tokens:
            # we can append the whole document and return
            if mask_cursor is not None:
                assert mask_cursor.remaining == document_entry.text.size, (
                    mask_cursor.remaining,
                    document_entry.text.size,
                )
                document_entry.mask = mask_cursor.take(mask_cursor.remaining)
            current_sequence.append(document_entry)
            current_sequence_size += document_entry.size
            return current_sequence, current_sequence_size, finished_sequences


def _rng(seed: tuple[int, ...]) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _select_subset_of_documents(
    specs: DatasourceSpecs, n_documents: int, index: _IndexReader, min_tokens: int
) -> np.ndarray:
    """We only select a subset of the documents"""
    start = time.time()
    assert specs.percentage_data < 1.0
    assert n_documents <= np.iinfo(np.int32).max, (
        f"n_documents={n_documents} exceeds int32 max ({np.iinfo(np.int32).max})"
    )
    doc_ids = np.arange(n_documents, dtype=np.int32)
    # IMPORTANT: seed dont depend on epoch, we keep the same documents everytime
    _rng(seed=(specs.seed, n_documents))
    documents_sizes = index.sequence_lengths
    assert len(documents_sizes) == n_documents, (
        n_documents,
        len(documents_sizes),
        specs.prefix_path,
    )
    n_total_tokens = int(
        np.sum(documents_sizes) - len(documents_sizes)
    )  # we remove EOS
    n_target_tokens = max(specs.percentage_data * n_total_tokens, min_tokens)
    logger.warning(
        f"For datasource: {specs.prefix_path}, going to keep only {n_target_tokens / 1e9}B toks out of {n_total_tokens / 1e9}B"
    )
    # if too slow, we can do np.cum_sum
    n_toks = 0
    i = 0
    while n_toks < n_target_tokens:
        doc_id = doc_ids[i]
        size = int(documents_sizes[doc_id]) - 1  # EOS
        assert size > 0, (specs.prefix_path, doc_id)
        n_toks += size
        i += 1
    assert i > 0, specs.prefix_path
    selected_doc_ids = np.sort(doc_ids[:i])

    logger.info(
        f"Selected {i} documents out of {n_documents} for {specs.prefix_path} ({100 * i / n_documents:.2f})% {specs.percentage_data=} in {time.time() - start}s"
    )

    return selected_doc_ids


def _epoch_index_entry_to_bytes(entries: list[DocumentIndexEntry]) -> bytes:
    return DocumentIndexEntry.to_bytes(entries) + b"\n"


def _reload_source_metadata_from_folder(
    prefix_paths: PrefixPaths,
) -> PretokenizedDatasourceMetadataV2:
    merged_chunks_folder = prefix_paths.prefix.parent
    metadata_in_merged_folder = get_datasource_metadata_v2_path(merged_chunks_folder)
    # Look in merged folder first, fall back to parent source folder
    source_folder = merged_chunks_folder.parent
    metadata_path = metadata_in_merged_folder
    if not metadata_path.exists():
        metadata_path = get_datasource_metadata_v2_path(source_folder)
    if not metadata_path.exists():
        raise RuntimeError(
            f"pretokenization_metadata_v2.json not found in {merged_chunks_folder} or {source_folder}"
        )
    source_metadata = PretokenizedDatasourceMetadataV2.model_validate_json(
        metadata_path.read_text()
    )
    return source_metadata


def _document_y_mask(doc: DocumentIndexEntry) -> np.ndarray:
    doc_y_mask = np.ones(doc.size, dtype=np.bool_)
    img_doc_offset = 0
    for img in doc.imgs:
        assert img.size is not None, (doc.doc_id, img)
        img_pos_in_doc = img.pos_in_text + img_doc_offset
        doc_y_mask[img_pos_in_doc : img_pos_in_doc + img.size] = False
        img_doc_offset += img.size

    if doc.mask is not None:
        text_mask = _mask_to_array_mask(doc.mask)
        text_positions = doc_y_mask.copy()
        assert len(text_mask) == doc.text.size, (
            len(text_mask),
            doc.text.size,
            doc.doc_id,
        )
        assert int(text_positions.sum()) == doc.text.size, (
            int(text_positions.sum()),
            doc.text.size,
            doc.doc_id,
        )
        doc_y_mask[text_positions] = text_mask

    return doc_y_mask


def _document_has_trainable_target(
    doc: DocumentIndexEntry, skip_first_token: bool
) -> bool:
    if doc.mask is None and not doc.imgs:
        return doc.size > int(skip_first_token)

    doc_y_mask = _document_y_mask(doc)
    if skip_first_token:
        doc_y_mask = doc_y_mask[1:]
    return bool(doc_y_mask.any())


def _is_valid_sequence(sequence: list[DocumentIndexEntry]) -> bool:
    skip_first_token = True
    for doc in sequence:
        if _document_has_trainable_target(doc=doc, skip_first_token=skip_first_token):
            return True
        skip_first_token = False
    return False
