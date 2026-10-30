#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("id", pa.string()),
        ("text", pa.large_string()),
        ("num_tokens", pa.int64()),
        ("split", pa.string()),
    ]
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arxiv_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            data = row.get("data")
            text = data.get("text") if isinstance(data, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number} has no string data.text")
            yield {
                "dataset": "arxiv_val_ppl",
                "id": str(row.get("uid", line_number - 1)),
                "text": text,
                "num_tokens": None,
                "split": str(row.get("split", "validation")),
            }


def catlm_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number} has no string text")
            yield {
                "dataset": "catlm_v2",
                "id": str(row.get("id", line_number - 1)),
                "text": text,
                "num_tokens": row.get("num_tokens"),
                "split": "validation",
            }


def write_rows(writer, rows, counts, batch_size=128):
    batch = []
    for row in rows:
        counts[row["dataset"]] += 1
        batch.append(row)
        if len(batch) == batch_size:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
            batch.clear()
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv-jsonl", type=Path, required=True)
    parser.add_argument("--catlm-jsonl", type=Path, required=True)
    parser.add_argument("--catlm-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output.resolve()
    manifest_path = output.with_suffix(".manifest.json")
    temporary_path = output.with_suffix(f"{output.suffix}.tmp")
    for path in (output, manifest_path, temporary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    writer = pq.ParquetWriter(
        temporary_path,
        SCHEMA,
        compression="zstd",
        compression_level=9,
        use_dictionary=["dataset", "split"],
        write_statistics=True,
    )
    try:
        write_rows(writer, arxiv_rows(args.arxiv_jsonl), counts)
        write_rows(writer, catlm_rows(args.catlm_jsonl), counts)
    finally:
        writer.close()
    os.replace(temporary_path, output)

    catlm_manifest = json.loads(args.catlm_manifest.read_text())
    manifest = {
        "format_version": 1,
        "parquet_file": output.name,
        "parquet_bytes": output.stat().st_size,
        "parquet_sha256": sha256(output),
        "schema": {field.name: str(field.type) for field in SCHEMA},
        "datasets": {
            "arxiv_val_ppl": {
                "rows": counts["arxiv_val_ppl"],
                "source_uri": "s3://nvidia-mistral-sharing/arxiv_val_ppl/data.jsonl",
                "source_bytes": args.arxiv_jsonl.stat().st_size,
                "source_sha256": sha256(args.arxiv_jsonl),
            },
            "catlm_v2": {
                "rows": counts["catlm_v2"],
                "source_uri": "s3://nvidia-mistral-sharing/share-bundle-muon-e32/eval_data/catlm_pretokenized_lobstenizer/merged_1099511627776b/",
                "source_bytes": args.catlm_jsonl.stat().st_size,
                "source_sha256": sha256(args.catlm_jsonl),
                "source_token_count": catlm_manifest["input_token_count"],
                "scored_token_count": catlm_manifest["output_token_count"],
                "source_sequence_count": catlm_manifest["sequence_count"],
                "detokenized_with": "lobstenizer-tekken-hf",
                "skip_special_tokens": catlm_manifest["skip_special_tokens"],
                "bos_token_id": catlm_manifest["bos_token_id"],
                "removed_leading_bos_count": catlm_manifest["removed_leading_bos_count"],
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
