#!/usr/bin/env python3
import argparse
import fnmatch
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem


REPO_ID = "nvidia/llm-compression-intelligence-2026"
SUBSET_PATTERNS = {
    "cc_interleaved": "cc_interleaved/test-*.parquet",
    "arxiv_math_interleaved": "data_interleaved/arxiv_math_interleaved/*.parquet",
    "arxiv_physics_interleaved": "data_interleaved/arxiv_physics_interleaved/*.parquet",
    "code_repo_packed": "data_interleaved/code_repo_packed/*.parquet",
}
SOURCE_COLUMNS = ("doc_id", "source", "content", "image_count")
SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("doc_id", pa.string()),
        ("source", pa.string()),
        ("content", pa.large_string()),
        ("image_count", pa.int32()),
    ]
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if not args.inventory_only and args.output is None:
        parser.error("--output is required unless --inventory-only is set")
    return args


def source_inventory(repo_id, revision):
    info = HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)
    siblings = {sibling.rfilename: sibling for sibling in info.siblings}
    inventory = {}
    for subset, pattern in SUBSET_PATTERNS.items():
        names = sorted(name for name in siblings if fnmatch.fnmatch(name, pattern))
        if not names:
            raise FileNotFoundError(f"No {repo_id} files match {pattern}")
        inventory[subset] = [
            {"path": name, "bytes": siblings[name].size, "sha256": siblings[name].lfs.sha256}
            for name in names
        ]
    return info.sha, inventory


def remote_path(repo_id, revision, filename):
    return f"datasets/{repo_id}@{revision}/{filename}"


def parquet_metadata(fs, path):
    with fs.open(path, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        missing = set(SOURCE_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        selected_bytes = 0
        for row_group_index in range(parquet.metadata.num_row_groups):
            row_group = parquet.metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                column = row_group.column(column_index)
                if column.path_in_schema in SOURCE_COLUMNS:
                    selected_bytes += column.total_compressed_size
        return parquet.metadata.num_rows, selected_bytes


def iter_batches(fs, path, batch_size=128):
    with fs.open(path, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        yield from parquet.iter_batches(columns=SOURCE_COLUMNS, batch_size=batch_size)


def canonical_row_digest(digest, dataset, doc_id, source, content, image_count):
    values = (dataset, doc_id, source, content, str(image_count))
    for value in values:
        digest.update(("" if value is None else str(value)).encode("utf-8"))
        digest.update(b"\0")
    digest.update(b"\n")


def main():
    args = parse_args()
    revision, inventory = source_inventory(args.repo_id, args.revision)
    fs = HfFileSystem()
    counts = Counter()
    selected_bytes = Counter()
    for subset, files in inventory.items():
        for source in files:
            path = remote_path(args.repo_id, revision, source["path"])
            rows, projected_bytes = parquet_metadata(fs, path)
            source["rows"] = rows
            source["selected_column_compressed_bytes"] = projected_bytes
            counts[subset] += rows
            selected_bytes[subset] += projected_bytes

    preflight = {
        "repo_id": args.repo_id,
        "revision": revision,
        "source_files": sum(len(files) for files in inventory.values()),
        "source_bytes": sum(source["bytes"] for files in inventory.values() for source in files),
        "selected_column_compressed_bytes": sum(selected_bytes.values()),
        "datasets": {
            subset: {
                "rows": counts[subset],
                "source_files": len(inventory[subset]),
                "source_bytes": sum(source["bytes"] for source in inventory[subset]),
                "selected_column_compressed_bytes": selected_bytes[subset],
            }
            for subset in SUBSET_PATTERNS
        },
    }
    if args.inventory_only:
        print(json.dumps(preflight, indent=2))
        return

    output = args.output.resolve()
    manifest_path = output.with_suffix(".manifest.json")
    temporary_path = output.with_suffix(f"{output.suffix}.tmp")
    for path in (output, manifest_path, temporary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    written = Counter()
    digests = {subset: hashlib.sha256() for subset in SUBSET_PATTERNS}
    writer = pq.ParquetWriter(
        temporary_path,
        SCHEMA,
        compression="zstd",
        compression_level=9,
        use_dictionary=["dataset", "source"],
        write_statistics=True,
    )
    try:
        for subset, files in inventory.items():
            for source in files:
                path = remote_path(args.repo_id, revision, source["path"])
                for batch in iter_batches(fs, path):
                    data = batch.to_pydict()
                    rows = []
                    for index in range(batch.num_rows):
                        row = {
                            "dataset": subset,
                            "doc_id": data["doc_id"][index],
                            "source": data["source"][index],
                            "content": data["content"][index],
                            "image_count": data["image_count"][index],
                        }
                        if not isinstance(row["content"], str):
                            raise ValueError(f"{path} row {index} has no string content")
                        canonical_row_digest(digests[subset], **row)
                        rows.append(row)
                    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                    written[subset] += len(rows)
    finally:
        writer.close()
    os.replace(temporary_path, output)

    if written != counts:
        raise ValueError(f"Row count changed during export: {written} != {counts}")
    manifest = {
        "format_version": 1,
        "parquet_file": output.name,
        "parquet_bytes": output.stat().st_size,
        "parquet_sha256": sha256(output),
        "schema": {field.name: str(field.type) for field in SCHEMA},
        "source_repo": args.repo_id,
        "source_revision": revision,
        "omitted_columns": ["images", "meta_json", "subset"],
        "datasets": {
            subset: {
                "rows": written[subset],
                "row_sha256": digests[subset].hexdigest(),
                "source_files": inventory[subset],
            }
            for subset in SUBSET_PATTERNS
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"preflight": preflight, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
