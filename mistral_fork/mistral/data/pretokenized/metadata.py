import dataclasses
import logging
import mmap
import shutil
import struct
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Iterator, Type, Union

import msgspec
import numpy as np
from PIL import Image

from mistral_fork.mistral.data.pretokenized.image_transform_metadata import (
    NONE_BOUNDING_BOX,
    ImageBoundingBox,
    TransformImageMetadata,
)
from mistral_fork.mistral.data.pretokenized.image_utils import _load_image_or_fallback
from mistral_fork.mistral.data.pretokenized.megatron_index_reader import DType, _ListTokensBinReader
from mistral_fork.shared.cluster import clusterify_path
from mistral_fork.shared.common_utils import TarOffset

if TYPE_CHECKING:
    from io import BufferedReader

    from mistral_fork.mistral.data.pretokenized.megatron_index_reader import _FileBinReader

logger = logging.getLogger(__name__)

_JSON_ENCODER = msgspec.json.Encoder()
_DICT_DECODER = msgspec.json.Decoder(type=dict)
_LIST_OF_DICTS_DECODER = msgspec.json.Decoder(type=list[dict])

# Format: (sizes, masks) where sizes is list[int] and masks is list[int] (0 or 1)
# Example: ([1, 115, 1, 100], [1, 0, 1, 0]) means segments of size 1,115,1,100 with masks True,False,True,False
MaskIndexEntry = tuple[list[int], list[int]]


class _MaskCursor:
    """Consume consecutive pieces of a run-length encoded mask in linear time."""

    def __init__(self, mask: MaskIndexEntry) -> None:
        _check_mask_correctness(mask)
        self._sizes, self._masks = mask
        self._segment_idx = 0
        self._offset_in_segment = 0
        self._remaining = sum(self._sizes)

    @property
    def remaining(self) -> int:
        return self._remaining

    def take(self, size: int, overlap: int = 0) -> MaskIndexEntry:
        """Return ``size`` mask tokens and advance by ``size - overlap`` tokens."""
        assert overlap in (0, 1), overlap
        assert size >= overlap, (size, overlap)
        assert size <= self._remaining, (size, self._remaining)

        sizes: list[int] = []
        masks: list[int] = []
        remaining_to_take = size
        while remaining_to_take > 0:
            segment_size = self._sizes[self._segment_idx]
            take = min(
                segment_size - self._offset_in_segment, remaining_to_take
            )
            mask = self._masks[self._segment_idx]
            if masks and masks[-1] == mask:
                sizes[-1] += take
            else:
                sizes.append(take)
                masks.append(mask)
            self._offset_in_segment += take
            self._remaining -= take
            remaining_to_take -= take
            if self._offset_in_segment == segment_size:
                self._segment_idx += 1
                self._offset_in_segment = 0

        if overlap:
            # Text splits retain their boundary token in the next chunk.
            self._remaining += overlap
            if self._offset_in_segment == 0:
                self._segment_idx -= 1
                self._offset_in_segment = self._sizes[self._segment_idx]
            self._offset_in_segment -= overlap
        return sizes, masks


class TextIndexEntry(msgspec.Struct):
    offset: int  # in #elements, not in bytes!
    size: int  # in #elements, not in bytes!

    def split(
        self, size: int, overlap: int
    ) -> tuple["TextIndexEntry", "TextIndexEntry"]:
        assert self.size - (size - overlap) > 0
        assert self.size >= size, (self.size, size)

        left = TextIndexEntry(self.offset, size)
        right = TextIndexEntry(self.offset + size - overlap, self.size - size + overlap)
        return left, right


@dataclasses.dataclass(frozen=True)
class TransformOutput:
    # this is populated in build_epoch_index, since transform will be different at each epoch
    seed: tuple[int, ...]
    metadata_after_transform: TransformImageMetadata
    n_toks: int


class _TarOffset(msgspec.Struct, omit_defaults=True):
    # Same fields as the pydantic shared.common_utils.TarOffset
    offset: int
    size: int
    new_tar_format: bool = False
    tar_path: str | None = None

    @classmethod
    def from_pydantic(cls, model: TarOffset) -> "_TarOffset":
        """Convert from Pydantic model to msgspec struct."""
        return cls(**model.__dict__)

    def to_pydantic(self) -> TarOffset:
        """Convert from msgspec struct to Pydantic model."""
        return TarOffset(**msgspec.structs.asdict(self))


class ImageIndexEntry(msgspec.Struct):
    img_id_in_doc: int
    tar_offset: _TarOffset
    pos_in_text: int
    metadata: TransformImageMetadata
    size: int = -1  # number of tokens
    seed: tuple[int, ...] | None = None

    @property
    def tar_path(self) -> str:
        path = self.tar_offset.tar_path
        assert path is not None, f"tar_offset.tar_path is None: {self.tar_offset}"
        if not path.startswith("s3://"):
            return clusterify_path(path)
        return path

    @property
    def _tar_path(self) -> str:
        return self.tar_path

    def load_content(self) -> Image.Image:
        return _load_image_or_fallback(self)

    @classmethod
    def to_bytes(
        cls, entry: Union["ImageIndexEntry", list["ImageIndexEntry"]]
    ) -> bytes:
        return _JSON_ENCODER.encode(cls._to_mistral_wire_builtins(entry))

    @classmethod
    def _to_mistral_wire_builtins(
        cls, entry: Union["ImageIndexEntry", list["ImageIndexEntry"]]
    ) -> Any:
        return msgspec.to_builtins(entry, enc_hook=cls._enc_hook)

    @classmethod
    def _migrate_legacy_fields(cls, raw: dict) -> dict:
        """Transform old serialized format to current format."""
        if "tar_path" in raw:
            raw["tar_offset"]["tar_path"] = raw.pop("tar_path")

        # Handle depreciated field names
        if "metadata_before_transform" in raw:
            # depreciated format. We need to migrate
            assert "metadata" not in raw
            raw["metadata"] = raw.pop("metadata_before_transform")

        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("content_modified", None)
            metadata.pop("max_image_size", None)
            if (
                "_img_bbox_for_trim" in metadata
                and metadata["_img_bbox_for_trim"] == NONE_BOUNDING_BOX
            ):
                metadata["_img_bbox_for_trim"] = None

        transform_output: dict[str, int | tuple[int, ...]] = raw.pop(
            "maybe_transform_output", None
        )
        if transform_output is not None:
            # depreciated format. We need to migrate
            assert "seed" not in raw
            assert "size" not in raw
            # raise if seed or n_toks not in transform_output
            raw["seed"] = transform_output.get("seed")
            raw["size"] = transform_output.get("n_toks")

        return raw

    @classmethod
    def from_bytes(cls, data: bytes) -> "ImageIndexEntry":
        # Decode to dict first to handle backward compatibility
        raw = _DICT_DECODER.decode(data)
        raw = cls._migrate_legacy_fields(raw)
        return msgspec.convert(raw, ImageIndexEntry, dec_hook=cls._dec_hook)

    @classmethod
    def list_from_bytes(cls, data: bytes) -> list["ImageIndexEntry"]:
        # Decode to list of dicts first
        raw_list = _LIST_OF_DICTS_DECODER.decode(data)
        raw_list = [cls._migrate_legacy_fields(raw) for raw in raw_list]
        return msgspec.convert(raw_list, list[ImageIndexEntry], dec_hook=cls._dec_hook)

    @classmethod
    def _enc_hook(cls, obj: Any) -> Any:
        # Ported from: mistral/data/pretokenized/metadata.py (ImageIndexEntry._enc_hook)
        if isinstance(obj, ImageBoundingBox):
            return obj.to_mistral_json()
        raise NotImplementedError(f"Objects of type {type(obj)} are not supported")

    @classmethod
    def _dec_hook(cls, _type: Type, obj: Any) -> Any:
        # Ported from: mistral/data/pretokenized/metadata.py (ImageIndexEntry._dec_hook)
        if _type is ImageBoundingBox:
            if isinstance(obj, str):
                import json

                return ImageBoundingBox.from_mistral_dict(json.loads(obj))
            elif isinstance(obj, dict):
                return ImageBoundingBox.from_dict(obj)
            else:
                raise NotImplementedError(
                    f"ImageBoundingBox should be a string or a dict. Got {type(obj)}"
                )
        raise NotImplementedError(f"Objects of type {_type} are not supported")


class DocumentIndexEntry(msgspec.Struct):
    doc_id: int
    text: TextIndexEntry
    imgs: list[ImageIndexEntry]
    size: int = -1
    mask: MaskIndexEntry | None = None

    def __post_init__(self) -> None:
        # Here for BW compatibility. Size should be passed at build time now.
        # In post init rather than as a property for performance reasons
        if self.size == -1:
            self.size = self.text.size + sum(img.size for img in self.imgs)

    def get_text_tokens(
        self, bin_reader: "_FileBinReader | _ListTokensBinReader"
    ) -> np.ndarray:
        return bin_reader.read(
            dtype=np.int32,
            offset=self.text.offset * DType.size(np.int32),
            count=self.text.size,
        )

    def split(
        self, size: int, mask_cursor: _MaskCursor | None = None
    ) -> tuple["DocumentIndexEntry", "DocumentIndexEntry"]:
        overlap = 1  # split on text
        cumulated_img_size = 0
        for idx, img in enumerate(self.imgs):
            img_pos_in_sequence = img.pos_in_text + cumulated_img_size
            if img_pos_in_sequence >= size:
                # we are past the split point, split point is on text
                # total size = text_size + sum(img.size)
                left_text_size = size - cumulated_img_size
                break
            if img_pos_in_sequence < size <= (img_pos_in_sequence + img.size):
                # split point is on an image. left text stops where the (maybe) consecutive images start
                left_text_size = img.pos_in_text
                overlap = 0
                break
            cumulated_img_size += img.size
        else:
            # split on text
            # total size = text_size + sum(img.size)
            left_text_size = size - cumulated_img_size

        # assert self.size - (size - overlap) > 0
        # assert self.size >= size, (self.size, size)
        # assert self.size == self.text.size + sum(img.size for img in self.imgs)

        # Find break point in images. Should be < 10 most of the time. Images are sorted.
        idx = 0
        while idx < len(self.imgs) and self.imgs[idx].pos_in_text < left_text_size:
            idx += 1
        # the last left image finish before the split
        left_imgs = self.imgs[:idx]
        right_imgs = self.imgs[idx:]

        shift = left_text_size - overlap
        for img in right_imgs:
            img.pos_in_text -= shift

        left_text, right_text = self.text.split(left_text_size, overlap)
        if mask_cursor is not None:
            mask_left = mask_cursor.take(left_text_size, overlap)
            mask_right = None
        elif self.mask is not None:
            assert _mask_size(self.mask) == self.text.size, (
                _mask_size(self.mask),
                self.text.size,
            )
            mask_left, mask_right = _split_mask(self.mask, left_text_size, overlap)
            assert _mask_size(mask_right) == right_text.size, (
                _mask_size(mask_right),
                right_text.size,
            )
            assert _mask_size(mask_left) == left_text.size, (
                _mask_size(mask_left),
                left_text.size,
            )
        else:
            mask_right, mask_left = None, None

        left = DocumentIndexEntry(
            doc_id=self.doc_id,
            text=left_text,
            imgs=left_imgs,
            size=left_text_size + sum(img.size for img in left_imgs),
            mask=mask_left,
        )

        right = DocumentIndexEntry(
            doc_id=self.doc_id,
            text=right_text,
            imgs=right_imgs,
            size=self.size - left.size + overlap,
            mask=mask_right,
        )
        # assert left.size == left.text.size + sum(img.size for img in left.imgs)

        # assert right.size == right.text.size + sum(img.size for img in right.imgs)
        # assert left.size + right.size == self.size + overlap
        # assert left.text.size + right.text.size == self.text.size + overlap
        # assert len(left.imgs) + len(right.imgs) == len(self.imgs)
        # assert sum(img.size for img in right.imgs) + sum(
        #     img.size for img in left.imgs
        # ) == sum(img.size for img in self.imgs)

        return left, right

    def truncate_trailing_masked(
        self, keep_first_token: bool = False
    ) -> tuple["DocumentIndexEntry | None", int, int]:
        if self.mask is None:
            return self, 0, 0

        sizes, masks = self.mask
        if masks[-1] == 1:
            return self, 0, 0

        removed_text = sizes[-1]
        new_text_size = self.text.size - removed_text
        if (new_text_size <= 1 and not keep_first_token) or new_text_size == 0:
            return None, self.text.size, len(self.imgs)

        new_mask = (sizes[:-1], masks[:-1])
        kept_imgs = [img for img in self.imgs if img.pos_in_text < new_text_size]
        removed_imgs = len(self.imgs) - len(kept_imgs)
        truncated = DocumentIndexEntry(
            doc_id=self.doc_id,
            text=TextIndexEntry(self.text.offset, new_text_size),
            imgs=kept_imgs,
            size=new_text_size + sum(img.size for img in kept_imgs),
            mask=new_mask,
        )
        return truncated, removed_text, removed_imgs

    @classmethod
    def to_bytes(
        cls, entry: Union["DocumentIndexEntry", list["DocumentIndexEntry"]]
    ) -> bytes:
        return _JSON_ENCODER.encode(
            msgspec.to_builtins(entry, enc_hook=cls._enc_hook)
        )

    @classmethod
    def _migrate_legacy_fields(cls, raw: dict) -> dict:
        """Transform old serialized format to current format."""
        raw["imgs"] = [
            ImageIndexEntry._migrate_legacy_fields(raw_img) for raw_img in raw["imgs"]
        ]
        return raw

    @classmethod
    def from_bytes(cls, data: bytes) -> "DocumentIndexEntry":
        raw = _DICT_DECODER.decode(data)
        raw = cls._migrate_legacy_fields(raw)
        return msgspec.convert(raw, DocumentIndexEntry, dec_hook=cls._dec_hook)

    @classmethod
    def list_from_bytes(cls, data: bytes) -> list["DocumentIndexEntry"]:
        raw_list = _LIST_OF_DICTS_DECODER.decode(data)
        raw_list = [cls._migrate_legacy_fields(raw) for raw in raw_list]
        return msgspec.convert(
            raw_list, type=list[DocumentIndexEntry], dec_hook=cls._dec_hook
        )

    @classmethod
    def _enc_hook(cls, obj: Any) -> Any:
        # TODO: @quentin: get rid of this, perf killer
        return ImageIndexEntry._enc_hook(obj)

    @classmethod
    def _dec_hook(cls, type: Type, obj: Any) -> Any:
        return ImageIndexEntry._dec_hook(type, obj)

    def check(self) -> None:
        assert self.size == self.text.size + sum(img.size for img in self.imgs), (
            f"total size should match text + img sizes {self}"
        )
        if self.mask is not None:
            _check_mask_correctness(self.mask)
            assert self.text.size == _mask_size(self.mask), self


_INDEX_FOOTER = b"MTAIDX\x00\x00"
_INDEX_VERSION = 2
_INDEX_FOOTER_SIZE = 24


def _write_index_atomic(index: np.ndarray, index_path: Path) -> None:
    tmp_path = index_path.with_name(f"tmp.{uuid.uuid4()}.{index_path.name}")
    try:
        with tmp_path.open("wb") as file:
            file.write(index.tobytes())
            file.write(struct.pack("<Q", len(index)))
            file.write(struct.pack("<Q", _INDEX_VERSION))
            file.write(_INDEX_FOOTER)
        tmp_path.rename(index_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


class _MetadataWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: BinaryIO | None = None
        self._index_file: BinaryIO | None = None
        self._count = 0
        self._offset = 0

    def _index_path(self) -> Path:
        return Path(str(self._path) + ".idx")

    def _close_files(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._index_file is not None:
            self._index_file.close()
            self._index_file = None

    def __enter__(self) -> "_MetadataWriter":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self._close_files()
            self._path.unlink(missing_ok=True)
            self._index_path().unlink(missing_ok=True)

    def open(self) -> None:
        assert self._file is None, "Writer already opened"
        assert not self._path.exists(), f"File already exists: {self._path}"
        self._file = self._path.open("wb")
        self._index_file = self._index_path().open("wb")

    def add_document(self, metadata: list[ImageIndexEntry]) -> None:
        assert self._file is not None, "Writer not opened"
        assert self._index_file is not None, "Index writer not opened"
        self._index_file.write(struct.pack("<q", self._offset))
        self._count += 1
        data = ImageIndexEntry.to_bytes(metadata) + b"\n"
        self._file.write(data)
        self._offset += len(data)

    def merge_from(self, src_path: Path) -> None:
        assert self._file is not None, "Writer not opened"
        assert self._index_file is not None, "Index writer not opened"

        src_reader = _MetadataReader(src_path)
        try:
            adjusted = src_reader.line_positions + self._offset
            self._index_file.write(adjusted.tobytes())
            self._count += len(adjusted)
        finally:
            src_reader.close()

        with src_path.open("rb") as src_file:
            shutil.copyfileobj(src_file, self._file)
        self._offset = self._file.tell()

    def finalize(self) -> None:
        if self._file is None:
            return
        self._file.close()
        self._file = None

        assert self._index_file is not None, "Index writer not opened"
        self._index_file.write(struct.pack("<Q", self._count))
        self._index_file.write(struct.pack("<Q", _INDEX_VERSION))
        self._index_file.write(_INDEX_FOOTER)
        self._index_file.close()
        self._index_file = None


def dump_metadata(metadata: list[list[ImageIndexEntry]], metadata_path: Path) -> None:
    with _MetadataWriter(metadata_path) as writer:
        for doc_metadata in metadata:
            writer.add_document(doc_metadata)


class _MetadataReader:
    def __init__(self, metadata_path: str | Path) -> None:
        self.metadata_path = Path(metadata_path)
        self.file: None | BufferedReader = open(self.metadata_path, "rb")
        self.mmapped: None | mmap.mmap = (
            mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            if self.metadata_path.stat().st_size > 0
            else None
        )

        self.line_positions = self._load_or_build_index()

    def _get_index_path(self) -> Path:
        return Path(str(self.metadata_path) + ".idx")

    def _load_or_build_index(self) -> np.ndarray:
        index_path = self._get_index_path()
        if index_path.exists():
            try:
                index = self._load_index(index_path)
                if self._validate_index(index):
                    return index
                logger.warning(f"Index file {index_path} failed validation, rebuilding")
            except Exception as error:
                logger.warning(
                    f"Failed to load index file {index_path}: {error}, rebuilding"
                )

        index = self._build_index()
        try:
            _write_index_atomic(index, index_path)
        except Exception:
            logger.debug(f"Failed to save index {index_path}", exc_info=True)
        return index

    def _load_index(self, index_path: Path) -> np.ndarray:
        file_size = index_path.stat().st_size
        if file_size < _INDEX_FOOTER_SIZE:
            raise ValueError(f"Index file too small: {file_size} bytes")

        with index_path.open("rb") as file:
            file.seek(file_size - _INDEX_FOOTER_SIZE)
            count = struct.unpack("<Q", file.read(8))[0]
            version = struct.unpack("<Q", file.read(8))[0]
            magic = file.read(8)

        if magic != _INDEX_FOOTER:
            raise ValueError(f"Invalid index footer: {magic!r}")
        if version != _INDEX_VERSION:
            raise ValueError(f"Unsupported index version: {version}")

        if count == 0:
            return np.empty(0, dtype=np.int64)

        memmap = np.memmap(index_path, dtype=np.int64, mode="r", shape=(count,))
        index = np.array(memmap)
        del memmap
        return index

    def _validate_index(self, index: np.ndarray) -> bool:
        if len(index) == 0:
            return self.metadata_path.stat().st_size == 0
        if index[0] != 0:
            return False
        assert self.mmapped is not None
        return bool(index[-1] < len(self.mmapped))

    def _build_index(self) -> np.ndarray:
        if self.mmapped is None:
            return np.empty(0, dtype=np.int64)
        positions = [0]
        pos = 0
        while True:
            pos = self.mmapped.find(b"\n", pos)
            if pos == -1:
                break
            pos += 1
            if pos < len(self.mmapped):
                positions.append(pos)
        return np.array(positions, dtype=np.int64)

    def open(self) -> None:
        self.file = open(self.metadata_path, "rb")
        self.mmapped = (
            mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            if self.metadata_path.stat().st_size > 0
            else None
        )

    def __len__(self) -> int:
        return len(self.line_positions)

    def get_raw_bytes(self, idx: int) -> bytes:
        if idx < 0 or idx >= len(self.line_positions):
            raise IndexError("Line index out of range")
        assert self.mmapped is not None, "You may need to call open() first"

        start = self.line_positions[idx]
        if idx + 1 < len(self.line_positions):
            end = self.line_positions[idx + 1] - 1
        else:
            end = len(self.mmapped)
        return bytes(self.mmapped[start:end])

    def __getitem__(self, idx: int) -> list[ImageIndexEntry]:
        return ImageIndexEntry.list_from_bytes(self.get_raw_bytes(idx))

    def __iter__(self) -> Iterator[list[ImageIndexEntry]]:
        for i in range(len(self)):
            yield self[i]

    def close(self) -> None:
        if self.mmapped:
            self.mmapped.close()
            self.mmapped = None
        if self.file:
            self.file.close()
            self.file = None


class ScatteredMetadataReader:
    def __init__(
        self,
        metadata_path: Path,
        doc_ids: np.ndarray,
        tmp_dir: Path,
        max_bucket_bytes: int = 1024**3,
    ) -> None:
        self._tmp_dir = tmp_dir
        self._n_docs = len(doc_ids)
        assert self._n_docs > 0, "ScatteredMetadataReader requires non-empty doc_ids"

        max_docs = 2**32
        assert self._n_docs < max_docs, (
            f"n_docs={self._n_docs:_} exceeds max {max_docs:_} (uint32 limit for pos)"
        )

        doc_ids = np.asarray(doc_ids)
        order = np.argsort(doc_ids, kind="stable")
        self._sorted_doc_ids = doc_ids[order]
        self._pos_of_sorted = order.astype(np.uint32)
        assert self._sorted_doc_ids[0] >= 0, (
            f"doc_ids must be non-negative, got min={self._sorted_doc_ids[0]}"
        )

        metadata_reader = _MetadataReader(metadata_path)
        assert self._sorted_doc_ids[-1] < len(metadata_reader), (
            f"max doc_id={self._sorted_doc_ids[-1]} out of range for metadata with "
            f"{len(metadata_reader)} documents"
        )

        file_size = metadata_path.stat().st_size
        avg_entry_size = max(1, file_size // len(metadata_reader))
        self._entries_per_bucket = max(1, int(max_bucket_bytes // avg_entry_size))
        self._n_buckets = (
            self._n_docs + self._entries_per_bucket - 1
        ) // self._entries_per_bucket

        self._current_bucket_id: int | None = None
        self._current_bucket_data: dict[int, bytes] = {}

        start_time = time.time()
        try:
            self._scatter(metadata_reader)
        finally:
            metadata_reader.close()
        logger.info(
            f"Metadata scatter: {self._n_docs:_} docs into {self._n_buckets} "
            f"buckets in {time.time() - start_time:.1f}s"
        )

    def _bucket_id(self, pos: int) -> int:
        return pos // self._entries_per_bucket

    def _scatter(self, metadata_reader: _MetadataReader) -> None:
        bucket_files = [
            (self._tmp_dir / f"bucket_{bucket_id}.bin").open("wb")
            for bucket_id in range(self._n_buckets)
        ]
        try:
            for sorted_idx in range(self._n_docs):
                doc_id = int(self._sorted_doc_ids[sorted_idx])
                pos = int(self._pos_of_sorted[sorted_idx])
                data = metadata_reader.get_raw_bytes(doc_id)
                bucket_files[self._bucket_id(pos)].write(
                    struct.pack("<II", pos, len(data)) + data
                )
        finally:
            for file in bucket_files:
                file.close()

    def _load_bucket(self, bucket_id: int) -> dict[int, bytes]:
        bucket_path = self._tmp_dir / f"bucket_{bucket_id}.bin"
        data = bucket_path.read_bytes()
        records: dict[int, bytes] = {}
        offset = 0
        while offset < len(data):
            pos, length = struct.unpack_from("<II", data, offset)
            offset += 8
            end = offset + length
            records[pos] = data[offset:end]
            offset = end
        assert offset == len(data), (offset, len(data), bucket_path)
        return records

    def _ensure_bucket_loaded(self, bucket_id: int) -> None:
        if self._current_bucket_id != bucket_id:
            self._current_bucket_data.clear()
            self._current_bucket_data = self._load_bucket(bucket_id)
            self._current_bucket_id = bucket_id

    def __len__(self) -> int:
        return self._n_docs

    def __getitem__(self, pos: int) -> list[ImageIndexEntry]:
        assert 0 <= pos < self._n_docs, f"pos={pos} out of range [0, {self._n_docs})"
        self._ensure_bucket_loaded(self._bucket_id(pos))
        return ImageIndexEntry.list_from_bytes(self._current_bucket_data[pos])

    def cleanup(self) -> None:
        for bucket_id in range(self._n_buckets):
            (self._tmp_dir / f"bucket_{bucket_id}.bin").unlink(missing_ok=True)


def _split_mask(
    mask: MaskIndexEntry,
    left_text_size: int,
    overlap: int,
) -> tuple[MaskIndexEntry, MaskIndexEntry]:
    """
    We split the mask the same way the text is split. There is no special rule (ie we don't care if a sequence end
    with a masked part)
    """
    sizes, masks = mask
    left_sizes: list[int] = []
    left_masks: list[int] = []
    right_sizes: list[int] = []
    right_masks: list[int] = []

    # Where the right mask starts (due to overlap)
    right_start = left_text_size - overlap

    pos = 0  # cumulative position in text

    for length, value in zip(sizes, masks, strict=True):
        seg_start = pos
        seg_end = pos + length

        # Contribution to left
        if seg_start < left_text_size:
            left_len = min(seg_end, left_text_size) - seg_start
            if left_len > 0:
                left_sizes.append(left_len)
                left_masks.append(value)

        # Contribution to right
        if seg_end > right_start:
            right_len = seg_end - max(seg_start, right_start)
            if right_len > 0:
                right_sizes.append(right_len)
                right_masks.append(value)

        pos = seg_end

    return (left_sizes, left_masks), (right_sizes, right_masks)


def _mask_size(mask: MaskIndexEntry) -> int:
    sizes, _ = mask
    return sum(sizes)


def _check_mask_correctness(mask: MaskIndexEntry) -> None:
    sizes, masks = mask
    assert len(sizes) > 0, "mask must have at least one segment"
    assert all(s >= 1 for s in sizes), f"all segment sizes must be >= 1: {mask}"
    for i in range(1, len(masks)):
        assert masks[i] != masks[i - 1], (
            f"adjacent segments must have different mask values at index {i}: {mask}"
        )


def _mask_to_array_mask(mask: MaskIndexEntry) -> np.ndarray:
    sizes, masks = mask
    return np.array(
        [
            bool(value)
            for length, value in zip(sizes, masks, strict=True)
            for _ in range(length)
        ],
        dtype=bool,
    )
