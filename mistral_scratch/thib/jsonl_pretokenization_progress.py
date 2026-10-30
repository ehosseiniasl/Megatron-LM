#!/usr/bin/env python3
"""Report pretokenization chunk progress for sources in a YAML config."""

import argparse
import csv
from pathlib import Path
from typing import Any

from mistral_fork.mistral.data.pretokenized.processing.config import (
    _tokenizer_name_from_config,
    load_jsonl_pretokenization_config,
    resolve_config_source,
)


def _jsonl_files(source_path: Path) -> list[Path]:
    if source_path.suffix == ".jsonl":
        return [source_path] if source_path.is_file() else []
    return sorted(source_path.glob("*.jsonl"))


def _complete_chunk_count(chunks_dir: Path) -> int:
    if not chunks_dir.exists():
        return 0
    return sum(
        (chunk / "chunk.bin").is_file()
        and (chunk / "chunk.idx").is_file()
        and (chunk / "chunk.stats.json").is_file()
        for chunk in chunks_dir.iterdir()
        if chunk.is_dir() and not chunk.name.startswith("tmp.")
    )


def _source_progress(
    source: Any,
    source_index: int,
    *,
    jsonl_root: Path,
    tokenizer_name: Any,
    pretokenized_data_dir: Path,
) -> dict[str, object]:
    _, pretokenized_dir = resolve_config_source(
        source,
        jsonl_root=jsonl_root,
        tokenizer_name=tokenizer_name,
        pretokenized_data_dir=pretokenized_data_dir,
    )
    expected = len(_jsonl_files(Path(source.jsonl_dir)))
    merged_dirs = list(pretokenized_dir.glob("merged_*b"))
    if merged_dirs:
        completed = expected
    else:
        completed = min(
            _complete_chunk_count(pretokenized_dir / "chunks"),
            expected,
        )
    return {
        "source_index": source_index,
        "jsonl_dir": source.jsonl_dir,
        "pretokenized_dir": str(pretokenized_dir),
        "expected_chunks": expected,
        "completed_chunks": completed,
        "remaining_chunks": max(0, expected - completed),
        "progress_percent": (100.0 * completed / expected) if expected else "",
    }


def build_csv(config_path: Path, pretokenized_data_dir: Path, output_path: Path) -> None:
    config = load_jsonl_pretokenization_config(config_path)
    tokenizer_name = _tokenizer_name_from_config(config)
    rows = (
        _source_progress(
            source,
            source_index,
            jsonl_root=Path(config.jsonl_root_data_dir),
            tokenizer_name=tokenizer_name,
            pretokenized_data_dir=pretokenized_data_dir,
        )
        for source_index, source in enumerate(config.sources)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_index",
                "jsonl_dir",
                "pretokenized_dir",
                "expected_chunks",
                "completed_chunks",
                "remaining_chunks",
                "progress_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("pretokenized_data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_csv(args.config, args.pretokenized_data_dir, args.output)


if __name__ == "__main__":
    main()
