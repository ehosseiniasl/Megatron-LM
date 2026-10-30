#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(path):
    with path.open("rb") as stream:
        if stream.read(9) != b"MMIDIDX\x00\x00":
            raise ValueError(f"Bad indexed-dataset header: {path}")
        version = struct.unpack("<Q", stream.read(8))[0]
        if version != 1:
            raise ValueError(f"Unsupported indexed-dataset version {version}")
        dtype_code = struct.unpack("<B", stream.read(1))[0]
        sequence_count = struct.unpack("<Q", stream.read(8))[0]
        document_index_count = struct.unpack("<Q", stream.read(8))[0]
        arrays_offset = stream.tell()

    dtype = DTYPES[dtype_code]
    index = np.memmap(path, mode="r", dtype=np.uint8)
    lengths = np.frombuffer(index, dtype=np.int32, count=sequence_count, offset=arrays_offset)
    pointers = np.frombuffer(
        index, dtype=np.int64, count=sequence_count, offset=arrays_offset + lengths.nbytes
    )
    consumed = arrays_offset + lengths.nbytes + pointers.nbytes
    remaining = path.stat().st_size - consumed
    expected_document_bytes = document_index_count * np.dtype(np.int64).itemsize
    if remaining not in (0, expected_document_bytes):
        raise ValueError(
            f"Unexpected index size: {remaining} trailing bytes; expected 0 or "
            f"{expected_document_bytes}"
        )
    return dtype, lengths, pointers, remaining == 0, document_index_count


def without_leading_bos(ids, bos_token_id):
    if ids and bos_token_id is not None and ids[0] == bos_token_id:
        return ids[1:], True
    return ids, False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-prefix", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    idx_path = args.input_prefix.with_suffix(".idx")
    bin_path = args.input_prefix.with_suffix(".bin")
    manifest_path = args.output.with_name("manifest.json")
    temporary_path = args.output.with_suffix(f"{args.output.suffix}.tmp")
    for path in (args.output, manifest_path, temporary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    dtype, lengths, pointers, compact_index, document_index_count = load_index(idx_path)
    tokens = np.memmap(bin_path, mode="r", dtype=dtype)
    input_token_count = int(lengths.sum())
    if len(tokens) != input_token_count:
        raise ValueError(f"Expected {input_token_count} tokens, found {len(tokens)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_digest = hashlib.sha256()
    output_token_count = 0
    removed_leading_bos_count = 0
    with temporary_path.open("wb") as output:
        for sequence_id, (pointer, length) in enumerate(zip(pointers, lengths)):
            start = int(pointer) // np.dtype(dtype).itemsize
            ids = tokens[start : start + int(length)].tolist()
            ids, removed_bos = without_leading_bos(ids, tokenizer.bos_token_id)
            removed_leading_bos_count += int(removed_bos)
            output_token_count += len(ids)
            text = tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            record = {"id": sequence_id, "num_tokens": len(ids), "text": text}
            encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            output.write(encoded)
            output_digest.update(encoded)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, args.output)

    manifest = {
        "source_prefix": str(args.input_prefix),
        "source_uri": args.source_uri,
        "source_bin_bytes": bin_path.stat().st_size,
        "source_bin_sha256": sha256(bin_path),
        "source_idx_bytes": idx_path.stat().st_size,
        "source_idx_sha256": sha256(idx_path),
        "tokenizer_path": str(args.tokenizer),
        "output_path": str(args.output),
        "format": "JSON Lines; fields: id, num_tokens, text",
        "skip_special_tokens": False,
        "bos_token_id": tokenizer.bos_token_id,
        "removed_leading_bos_count": removed_leading_bos_count,
        "dtype": np.dtype(dtype).name,
        "compact_index_without_document_table": compact_index,
        "index_document_count_field": int(document_index_count),
        "sequence_count": len(lengths),
        "input_token_count": input_token_count,
        "output_token_count": output_token_count,
        "output_bytes": args.output.stat().st_size,
        "output_sha256": output_digest.hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
