#!/usr/bin/env python3
"""Report JSONL file counts and sizes for sources in a pretokenization YAML."""

import argparse
import csv
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml


BYTES_PER_GB = 1_000_000_000


def _jsonl_files(source_path: Path) -> list[Path]:
    if source_path.suffix == ".jsonl":
        return [source_path] if source_path.is_file() else []
    return sorted(source_path.glob("*.jsonl"))


def _source_stats(source: dict[str, Any], source_index: int) -> dict[str, object]:
    source_path = Path(source["jsonl_dir"])
    sizes_gb = [path.stat().st_size / BYTES_PER_GB for path in _jsonl_files(source_path)]
    return {
        "source_index": source_index,
        "jsonl_dir": str(source_path),
        "num_jsonl_files": len(sizes_gb),
        "min_size_gb": min(sizes_gb) if sizes_gb else "",
        "mean_size_gb": fmean(sizes_gb) if sizes_gb else "",
        "max_size_gb": max(sizes_gb) if sizes_gb else "",
    }


def build_csv(config_path: Path, output_path: Path) -> None:
    with config_path.open() as file:
        config = yaml.safe_load(file)

    sources = config["sources"]
    rows = (_source_stats(source, source_index) for source_index, source in enumerate(sources))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_index",
                "jsonl_dir",
                "num_jsonl_files",
                "min_size_gb",
                "mean_size_gb",
                "max_size_gb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="JSONL pretokenization YAML")
    parser.add_argument("output", type=Path, help="Output CSV path")
    args = parser.parse_args()
    build_csv(args.config, args.output)


if __name__ == "__main__":
    main()
