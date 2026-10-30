from collections.abc import Iterator
from pathlib import Path

import numpy as np


class IndexedJSONL:
    def __init__(self, jsonl_path: Path, index_path: Path) -> None:
        # TODO: maybe remove the length and use readline() instead of self.file.read(length)
        #   but in this case we need to be careful not to have new lines when dumping jsonl
        self.jsonl_path = jsonl_path
        self.index = np.memmap(
            index_path, dtype=[("offset", "i8"), ("length", "i4")], mode="r"
        )
        self.file = open(jsonl_path, "rb", buffering=0)

    def read(self, idx: int) -> bytes:
        offset, length = self.index[idx]
        self.file.seek(offset)
        return self.file.read(length)

    def close(self) -> None:
        self.file.close()
        if hasattr(self, "index") and isinstance(self.index, np.memmap):
            # np.memmap keeps an fd open; deleting it closes the underlying mmap
            del self.index

    def __del__(self) -> None:
        self.close()


def _write_jsonl_and_index_streaming(
    jsonl_path: Path,
    index_path: Path,
    sequences: Iterator[bytes],
) -> int:
    count = 0
    with open(jsonl_path, "wb") as jsonl_file, open(index_path, "wb") as index_file:
        offset = 0
        for seq in sequences:
            jsonl_file.write(seq)
            index_file.write(
                np.array(
                    (offset, len(seq)), dtype=[("offset", "i8"), ("length", "i4")]
                ).tobytes()
            )
            offset += len(seq)
            count += 1
    return count


def _write_jsonl_and_index_batched(
    jsonl_path: Path,
    index_path: Path,
    batches: Iterator[list[bytes]],
) -> int:
    count = 0
    with open(jsonl_path, "wb") as jsonl_file, open(index_path, "wb") as index_file:
        offset = 0
        for batch in batches:
            n = len(batch)
            if n == 0:
                continue

            lengths = np.array([len(seq) for seq in batch], dtype=np.int32)
            offsets = np.empty(n, dtype=np.int64)
            offsets[0] = offset
            offsets[1:] = offset + np.cumsum(lengths[:-1], dtype=np.int64)
            offset += int(lengths.sum(dtype=np.int64))

            index_array = np.empty(n, dtype=[("offset", "i8"), ("length", "i4")])
            index_array["offset"] = offsets
            index_array["length"] = lengths

            jsonl_file.write(b"".join(batch))
            index_file.write(index_array.tobytes())
            count += n
    return count
