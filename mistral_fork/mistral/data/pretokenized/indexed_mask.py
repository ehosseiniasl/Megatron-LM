import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

from mistral_fork.mistral.data.pretokenized.masking import DocumentMask

_MASK_INDEX_MAGIC = b"MASKID\x00\x01"
_MASK_INDEX_VERSION = 2

# Footer: magic, version, doc_count, total_segments, index_offset
_FOOTER_STRUCT = struct.Struct("<8sQQQQ")
_FOOTER_SIZE = _FOOTER_STRUCT.size

# 8-byte aligned row:
# - mask is uint8 (actual mask)
# - _reserved is 3 bytes so each row is exactly 8 bytes (alignment + future flags)
_SEGMENT_ROW_DTYPE = np.dtype(
    [
        ("size", "<i4"),
        ("mask", "<u1"),  # actual mask
        ("_reserved", "V3"),  # alignment + future flags
    ]
)
assert _SEGMENT_ROW_DTYPE.itemsize == 8


class IndexedMaskBuilder:
    """
    Builder for incrementally writing mask index files.

    Writes segment data (PAYLOAD) on the fly as documents are added,
    keeping only the index (doc_starts) in memory.
    Memory usage is ~8 bytes per document (int64).

    Usage:
        with IndexedMaskBuilder(path) as builder:
            for doc_mask in document_masks:
                builder.add_document(doc_mask)
        # File is finalized automatically on exit

    Or explicitly:
        builder = IndexedMaskBuilder(path)
        builder.open()
        for doc_mask in document_masks:
            builder.add_document(doc_mask)
        builder.finalize()
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: BinaryIO | None = None
        self._doc_starts: list[int] = []
        self._current_segment_offset: int = 0
        self._finalized: bool = False

    def __enter__(self) -> "IndexedMaskBuilder":
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
            # On exception, close file and clean up partial file
            if self._file is not None:
                self._file.close()
                self._file = None
            if self._path.exists():
                self._path.unlink()

    def open(self) -> None:
        """Open the file for writing."""
        assert self._file is None, "Builder already opened"
        assert not self._finalized, "Builder already finalized"
        self._file = self._path.open("wb")

    def add_document(self, doc_mask: DocumentMask) -> None:
        """Add a document mask to the file."""
        assert self._file is not None, "Builder not opened"
        assert not self._finalized, "Builder already finalized"

        sizes = doc_mask.sizes
        masks = doc_mask.mask
        assert len(sizes) == len(masks), "sizes/masks length mismatch"

        length = len(sizes)
        assert length < 2**31, f"Document has too many segments (>2^31): {length}"

        self._doc_starts.append(self._current_segment_offset)

        if length > 0:
            # Write segment rows directly
            payload = np.zeros(length, dtype=_SEGMENT_ROW_DTYPE)
            payload["size"] = np.asarray([int(s) for s in sizes], dtype=np.int32)
            payload["mask"] = np.asarray([int(m) for m in masks], dtype=np.uint8)
            self._file.write(payload.tobytes(order="C"))

        self._current_segment_offset += length

    def finalize(self) -> None:
        """Finalize the file by writing the index and footer."""
        assert self._file is not None, "Builder not opened"
        assert not self._finalized, "Builder already finalized"

        doc_count = len(self._doc_starts)
        total_segments = self._current_segment_offset

        assert total_segments < 2**63, (
            f"total_segments too large for int64: {total_segments}"
        )

        # Write INDEX (doc_starts with doc_count+1 elements, last is total_segments)
        index_offset = self._file.tell()
        doc_starts_with_end = self._doc_starts + [total_segments]
        doc_starts_arr = np.asarray(doc_starts_with_end, dtype=np.int64)

        self._file.write(doc_starts_arr.tobytes(order="C"))

        assert len(_MASK_INDEX_MAGIC) == 8

        # Write FOOTER
        self._file.write(
            _FOOTER_STRUCT.pack(
                _MASK_INDEX_MAGIC,
                int(_MASK_INDEX_VERSION),
                int(doc_count),
                int(total_segments),
                int(index_offset),
            )
        )

        self._file.close()
        self._file = None
        self._finalized = True


class MaskReader:
    """
    Layout: [PAYLOAD][INDEX][FOOTER]

    [PAYLOAD]
    rows: SegmentRow[total_segments] (8 bytes each)
        size: int32
        mask: uint8
        _reserved: 3 bytes (padding + future flags)

    [INDEX] (starts at index_offset)
    doc_starts: int64[doc_count + 1]
        - doc_starts[i] is the segment offset for document i
        - doc_starts[doc_count] == total_segments
        - length of document i = doc_starts[i+1] - doc_starts[i]

    [FOOTER]
    magic (8)
    version (u64)
    doc_count (u64)
    total_segments (u64)
    index_offset (u64)
    """

    def __init__(self, path: Path) -> None:
        file_size = path.stat().st_size
        assert file_size >= _FOOTER_SIZE, "File too small to contain footer"

        with path.open("rb") as f:
            f.seek(file_size - _FOOTER_SIZE)
            magic, version, doc_count, total_segments, index_offset = (
                _FOOTER_STRUCT.unpack(f.read(_FOOTER_SIZE))
            )

        assert magic == _MASK_INDEX_MAGIC, "Bad magic"
        assert version == _MASK_INDEX_VERSION, "Bad version"

        self.doc_count = int(doc_count)
        self.total_segments = int(total_segments)
        self.index_offset = int(index_offset)

        self._mmap = np.memmap(path, mode="r", order="C")

        # PAYLOAD rows start at 0
        self.rows = np.frombuffer(
            self._mmap,
            dtype=_SEGMENT_ROW_DTYPE,
            count=self.total_segments,
            offset=0,
        )

        # INDEX at index_offset (doc_count + 1 elements)
        self.doc_starts = np.frombuffer(
            self._mmap,
            dtype=np.int64,
            count=self.doc_count + 1,
            offset=self.index_offset,
        )

    def get_mask_segments_for_document(self, doc_id: int) -> DocumentMask:
        start = int(self.doc_starts[doc_id])
        end = int(self.doc_starts[doc_id + 1])

        doc_rows = self.rows[start:end]
        sizes = doc_rows["size"].tolist()
        mask = [bool(m) for m in doc_rows["mask"]]
        return DocumentMask(sizes=sizes, mask=mask)


def merge_indexed_mask_files(
    input_paths: list[Path],
    output_path: Path,
    *,
    chunk_bytes: int = 64 * 1024 * 1024,  # 64MB chunks
) -> None:
    """
    Merge multiple [PAYLOAD][INDEX][FOOTER] files by streaming payload and keeping
    only the index in memory.
    """
    assert input_paths, "No input files"

    readers = [MaskReader(p) for p in input_paths]
    total_doc_count = sum(r.doc_count for r in readers)
    total_segments = sum(r.total_segments for r in readers)

    out_doc_starts: list[int] = []

    seg_offset = 0  # running segment-row offset in merged payload

    with output_path.open("wb") as out:
        # 1) PAYLOAD: stream-copy rows for each file
        for r, p in zip(readers, input_paths, strict=True):
            # Only take first doc_count elements (not the trailing total_segments)
            doc_starts_for_file = r.doc_starts[: r.doc_count]
            out_doc_starts.extend(
                (doc_starts_for_file.astype(np.int64) + seg_offset).tolist()
            )

            payload_bytes = r.total_segments * _SEGMENT_ROW_DTYPE.itemsize
            _copy_range(
                p, out, src_offset=0, nbytes=payload_bytes, chunk_bytes=chunk_bytes
            )

            seg_offset += r.total_segments

        # 2) INDEX (doc_count + 1 elements, last is total_segments)
        index_offset = out.tell()
        out_doc_starts.append(total_segments)
        doc_starts_arr = np.asarray(out_doc_starts, dtype=np.int64)

        assert doc_starts_arr.shape[0] == total_doc_count + 1

        out.write(doc_starts_arr.tobytes(order="C"))

        # 3) FOOTER
        out.write(
            _FOOTER_STRUCT.pack(
                _MASK_INDEX_MAGIC,
                int(_MASK_INDEX_VERSION),
                int(total_doc_count),
                int(total_segments),
                int(index_offset),
            )
        )


def _copy_range(
    src_path: Path,
    dst_stream: BinaryIO,
    *,
    src_offset: int,
    nbytes: int,
    chunk_bytes: int,
) -> None:
    """Copy [src_offset, src_offset+nbytes) from src_path into dst_stream in chunks."""
    with src_path.open("rb") as src:
        src.seek(src_offset)
        remaining = nbytes
        while remaining:
            to_read = min(remaining, chunk_bytes)
            buf = src.read(to_read)
            if not buf:
                raise IOError("Unexpected EOF while copying")
            dst_stream.write(buf)
            remaining -= len(buf)
