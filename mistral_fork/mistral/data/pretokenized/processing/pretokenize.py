from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import multiprocessing
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import yaml

from mistral_fork.mistral.data.pretokenized.cache_key import (
    TokenizerName,
    check_pds,
    check_rare_tokens,
    tokenizer_name_from_path,
)
from mistral_fork.mistral.data.pretokenized.document_filters import (
    PdsFilter,
    RareTokenFilter,
    preload_rare_token_filters,
    should_filter_bad_document,
    should_filter_rare_tokens,
)
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.source import SampleType

logger = logging.getLogger(__name__)
PRETOKENIZE_MODULE = "mistral_fork.mistral.data.pretokenized.processing.pretokenize"
SUPPORTED_SUBMIT_MASKING_STRATEGIES = {
    "MOCK",
    "MASK_FILE_SEPARATOR",
    "MASK_FILE_SEPARATOR_NO_LAZINESS",
    "MASK_REASONING_TRACES_QUESTIONS",
    "MASK_MIDTRAINING_SEGMENTS",
    "MASK_GITHUB_PR_FORMATTING",
    "MASK_GITHUB_PR_EXPANDED_CONTEXT",
    "MASK_GITHUB_PR_EXPANDED_CONTEXT_AND_CONTEXT",
    "MASK_LOCODIFF_MARKERS",
    "INSTRUCT",
}


def _simple_tokenizer_name_from_config(config: dict[str, Any]) -> str:
    return tokenizer_name_from_path(str(config["tokenizer_path"])).value


# Keep the check_missing path importable on login nodes without the pretok
# container. The canonical config loader imports pydantic/transformers through
# processing.config, which is not available in the login-node Python.
def _parse_simple_yaml_scalar(raw: str) -> Any:
    value = raw.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", maxsplit=1)[0].rstrip()
    return value


def _load_simple_jsonl_config(config_path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    in_sources = False
    current_source: dict[str, Any] | None = None

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "sources:":
            in_sources = True
            config["sources"] = sources
            continue

        if in_sources:
            if stripped.startswith("- "):
                current_source = {}
                sources.append(current_source)
                rest = stripped[2:].strip()
                if rest:
                    key, value = rest.split(":", maxsplit=1)
                    current_source[key.strip()] = _parse_simple_yaml_scalar(value)
                continue
            if current_source is not None and ":" in stripped:
                key, value = stripped.split(":", maxsplit=1)
                current_source[key.strip()] = _parse_simple_yaml_scalar(value)
                continue

        if ":" in stripped:
            key, value = stripped.split(":", maxsplit=1)
            config[key.strip()] = _parse_simple_yaml_scalar(value)

    config.setdefault("sources", sources)
    return config


def _source_jsonl_files(path: Path) -> list[Path]:
    if path.suffix == ".jsonl":
        return [path] if path.is_file() else []
    return sorted(path.glob("*.jsonl"))


def _source_relative_to_root(source_path: Path, jsonl_root: Path) -> Path:
    try:
        return source_path.relative_to(jsonl_root)
    except ValueError as exc:
        raise ValueError(
            f"Raw source path must be under {jsonl_root}: {source_path}"
        ) from exc


def _source_pretok_dir(
    source: dict[str, Any],
    *,
    jsonl_root: Path,
    pretokenized_data_dir: Path,
    tokenizer_name: str,
) -> Path:
    source_path = Path(str(source["jsonl_dir"]))
    profile = FilterArgsName(str(source["filter"]))
    suffix = profile.value
    masking_strategy = source.get("masking_strategy")
    if masking_strategy is not None:
        suffix = f"{suffix}_{str(masking_strategy).lower()}"
    configured_rare_token_filter = source["rare_token_filter"]
    rare_token_filter = (
        RareTokenFilter(str(configured_rare_token_filter))
        if configured_rare_token_filter is not None
        else None
    )
    check_rare_tokens(
        profile=profile,
        tokenizer_name=TokenizerName(tokenizer_name),
        rare_token_filter=rare_token_filter,
    )
    if rare_token_filter is not None:
        suffix = f"{suffix}_{rare_token_filter.value}"
    sample_type = SampleType(str(source["sample_type"]))
    configured_pds_filter = source["pds_filter"]
    pds_filter = (
        PdsFilter(str(configured_pds_filter))
        if configured_pds_filter is not None
        else None
    )
    check_pds(
        tokenizer_name=TokenizerName(tokenizer_name),
        sample_type=sample_type,
        pds_filter=pds_filter,
    )
    if pds_filter is not None:
        suffix = f"{suffix}_{pds_filter.value}"
    bad_document_filter = source.get("bad_document_filter")
    if bad_document_filter is not None:
        suffix = f"{suffix}_{bad_document_filter}"
    return (
        pretokenized_data_dir
        / tokenizer_name
        / suffix
        / _source_relative_to_root(source_path, jsonl_root)
    )


def _chunk_dir_for_jsonl(chunks_dir: Path, jsonl_path: Path) -> Path:
    return chunks_dir / jsonl_path.stem


def _present_chunk_names(chunks_dir: Path) -> set[str]:
    if not chunks_dir.exists():
        return set()
    return {
        child.name
        for child in chunks_dir.iterdir()
        if child.is_dir() and not child.name.startswith("tmp.")
    }


def _source_expected_n_chunks(chunks_dir: Path) -> int | None:
    metadata_path = chunks_dir / "pretokenization_metadata_v2.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata["n_chunks"])


def check_missing_from_args(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Check for missing JSONL pretokenization chunks."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pretokenized-data-dir", type=Path, required=True)
    parser.add_argument("--max-group-size", type=int, default=2**40)
    parser.add_argument(
        "--show-missing-chunks",
        action="store_true",
        help="Include exact missing chunk names in the JSON output.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    config = _load_simple_jsonl_config(args.config)
    jsonl_root = Path(str(config.get("jsonl_root_data_dir", "/mnt/vast/datasets")))
    tokenizer_name = _simple_tokenizer_name_from_config(config)
    tasks = []
    missing_jsonl_sources = []
    missing_counts: dict[str, int] = {}
    skipped_merged_sources: set[str] = set()
    completed_chunks = 0
    total_chunks = 0
    for source in config["sources"]:
        pretok_dir = _source_pretok_dir(
            source,
            jsonl_root=jsonl_root,
            pretokenized_data_dir=args.pretokenized_data_dir,
            tokenizer_name=tokenizer_name,
        )
        merged_dir = pretok_dir / f"merged_{args.max_group_size}b"
        if merged_dir.exists():
            skipped_merged_sources.add(str(pretok_dir))
            expected_n_chunks = _source_expected_n_chunks(merged_dir)
            if expected_n_chunks is not None:
                completed_chunks += expected_n_chunks
                total_chunks += expected_n_chunks
            continue

        chunks_dir = pretok_dir / "chunks"
        present_chunks = _present_chunk_names(chunks_dir)
        expected_n_chunks = _source_expected_n_chunks(chunks_dir)
        if expected_n_chunks is not None:
            completed_chunks += min(len(present_chunks), expected_n_chunks)
            total_chunks += expected_n_chunks
            n_missing = max(0, expected_n_chunks - len(present_chunks))
            if n_missing == 0:
                continue
            missing_counts[str(pretok_dir)] = n_missing
            if not args.show_missing_chunks:
                continue

        source_path = Path(str(source["jsonl_dir"]))
        jsonl_files = _source_jsonl_files(source_path)
        if not jsonl_files:
            missing_jsonl_sources.append(str(source_path))
            continue
        if expected_n_chunks is None:
            completed_chunks += min(len(present_chunks), len(jsonl_files))
            total_chunks += len(jsonl_files)
            n_missing = max(0, len(jsonl_files) - len(present_chunks))
            if n_missing == 0:
                continue
            missing_counts[str(pretok_dir)] = n_missing
            if not args.show_missing_chunks:
                continue
        tasks.extend((source, jsonl_path) for jsonl_path in jsonl_files)
    tasks.sort(key=lambda item: (str(item[0]["jsonl_dir"]), item[1].name))
    total_tasks = len(tasks)

    missing_chunks: dict[str, list[str]] = {}

    for source, jsonl_path in tasks:
        pretok_dir = _source_pretok_dir(
            source,
            jsonl_root=jsonl_root,
            pretokenized_data_dir=args.pretokenized_data_dir,
            tokenizer_name=tokenizer_name,
        )
        merged_dir = pretok_dir / f"merged_{args.max_group_size}b"
        if merged_dir.exists():
            skipped_merged_sources.add(str(pretok_dir))
            continue

        chunks_dir = pretok_dir / "chunks"
        chunk_dir = _chunk_dir_for_jsonl(chunks_dir, jsonl_path)
        if not chunk_dir.exists():
            missing_chunks.setdefault(str(pretok_dir), []).append(chunk_dir.name)

    n_missing_chunks = sum(missing_counts.values())
    n_missing_jsonl_sources = len(missing_jsonl_sources)
    logger.info(
        "Pretokenized chunks complete: %d/%d (%d missing)",
        completed_chunks,
        total_chunks,
        n_missing_chunks,
    )
    for pretok_dir, count in missing_counts.items():
        logger.info("Missing %d chunk(s) in %s", count, pretok_dir)
    if missing_jsonl_sources:
        logger.info(
            "No visible JSONL files for %d source(s): %s",
            len(missing_jsonl_sources),
            ", ".join(missing_jsonl_sources),
        )

    result = {
        "ok": not (n_missing_chunks or n_missing_jsonl_sources),
        "completed_chunks": completed_chunks,
        "total_chunks": total_chunks,
        "missing_chunk_count": n_missing_chunks,
        "checked_tasks": len(tasks),
        "total_tasks": total_tasks,
        "skipped_merged_sources": len(skipped_merged_sources),
        "missing_jsonl_sources": missing_jsonl_sources,
        "missing_counts": missing_counts,
    }
    if args.show_missing_chunks:
        result["missing_chunks"] = missing_chunks
    if result["ok"]:
        logger.info(
            "No missing chunks found (%d merged source(s) skipped)",
            len(skipped_merged_sources),
        )
    else:
        logger.info(
            "Found %d missing chunk(s), %d source(s) with no visible JSONLs",
            n_missing_chunks,
            n_missing_jsonl_sources,
        )
    return result


if (
    __name__ == "__main__"
    and len(sys.argv) > 1
    and sys.argv[1]
    in {
        "check_missing",
        "check-missing",
    }
):
    check_result = check_missing_from_args(sys.argv[2:])
    print(json.dumps(check_result, ensure_ascii=False, indent=2))
    if not check_result["ok"]:
        raise SystemExit(1)
    sys.exit(0)


def _jsonl_files_for_source_path(path: Path) -> list[Path]:
    if path.suffix == ".jsonl":
        return [path] if path.is_file() else []
    return sorted(path.glob("*.jsonl"))


def validate_submit_config(config_path: Path) -> None:
    try:
        from mistral_fork.mistral.data.pretokenized.processing.config import (
            load_jsonl_pretokenization_config,
        )
    except Exception as exc:
        logger.info(
            "Falling back to lightweight config validation because the full loader "
            "could not be imported: %s",
            exc,
        )
        config = _load_simple_jsonl_config(config_path)
        sources = config.get("sources") or []
        for source_index, source in enumerate(sources):
            masking_strategy = source.get("masking_strategy")
            if masking_strategy is None:
                continue
            if masking_strategy not in SUPPORTED_SUBMIT_MASKING_STRATEGIES:
                raise ValueError(
                    f"Unsupported masking_strategy for source {source_index}: "
                    f"{masking_strategy!r}. Supported values are: "
                    f"{sorted(SUPPORTED_SUBMIT_MASKING_STRATEGIES)}"
                )
        logger.info("Submit preflight validated %s config source(s)", len(sources))
        return

    load_jsonl_pretokenization_config(config_path)
    logger.info("Submit preflight validated config schema: %s", config_path)


def check_submit_source_visibility(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    sources = config.get("sources") or []
    missing = []
    for source_index, source in enumerate(sources):
        path = Path(source["jsonl_dir"])
        files = _jsonl_files_for_source_path(path)
        logger.info(
            "Submit preflight source %s: %s -> %s JSONL file(s)",
            source_index,
            path,
            len(files),
        )
        if not files:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Submit preflight could not see *.jsonl files for: "
            + ", ".join(missing)
            + ". If these paths require a Unix group, run newgrp before submitting."
        )


def submit_distributed_pretokenization(
    *,
    config_path: Path,
    pretokenized_data_dir: Path,
    num_shards: int,
    max_parallel_shards: int | None,
    cpus_per_shard: int,
    max_group_size: int,
    validation_fraction: float,
    guarantee_validation_per_chunk: bool,
    workdir: Path | None,
    logs_dir: Path | None,
    job_name: str | None,
    log_level: str,
    slurm_partition: str | None,
    slurm_qos: str | None,
    slurm_account: str | None,
    slurm_constraint: str | None,
    slurm_time: str | None,
    mem_gb: int | None,
    group: str | None,
    container_image: str | None,
    container_mounts: str | None,
    container_name: str | None,
    container_env: str | None,
    container_remap_root: bool,
    submit_source_check: bool,
    dry_run: bool,
) -> dict[str, str | None]:
    assert num_shards >= 1, num_shards
    max_parallel_shards = max_parallel_shards or num_shards
    mem_gb = mem_gb or 4 * cpus_per_shard
    workdir = Path.cwd().resolve() if workdir is None else workdir
    validate_submit_config(config_path)
    if submit_source_check:
        check_submit_source_visibility(config_path)

    timestamp = time.strftime("%y%m%d_%H%M%S")
    logs_dir = logs_dir or pretokenized_data_dir / "logs"
    job_name = job_name or f"jsonl_pretokenize_{timestamp}"
    job_dir = logs_dir / timestamp
    if job_dir.exists():
        job_dir = logs_dir / f"{timestamp}_{uuid.uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=False)
    submission_path = job_dir / "submission.sbatch"
    worker_processes_per_shard = max(1, cpus_per_shard // 2)

    worker_command_parts = [
        "python3",
        "-m",
        PRETOKENIZE_MODULE,
        "process",
        "--config",
        shlex.quote(str(config_path)),
        "--pretokenized-data-dir",
        shlex.quote(str(pretokenized_data_dir)),
        "--max-group-size",
        str(max_group_size),
        "--validation-fraction",
        str(validation_fraction),
        "--worker-id",
        '"${SLURM_ARRAY_TASK_ID}"',
        "--num-workers",
        '"${SLURM_ARRAY_TASK_COUNT}"',
        "--n-workers",
        str(worker_processes_per_shard),
        "--log-level",
        shlex.quote(log_level),
        *(
            ["--guarantee-validation-per-chunk"]
            if guarantee_validation_per_chunk
            else []
        ),
    ]
    worker_command = " ".join(worker_command_parts)

    srun_parts = [
        "srun",
        "--ntasks=1",
        '--cpus-per-task="${SLURM_CPUS_PER_TASK}"',
    ]
    if container_image is not None:
        srun_parts.append(f"--container-image={shlex.quote(container_image)}")
        srun_parts.append(f"--container-workdir={shlex.quote(str(workdir))}")
    if container_name is not None:
        srun_parts.append(f"--container-name={shlex.quote(container_name)}")
    if container_image is not None and not container_remap_root:
        srun_parts.append("--no-container-remap-root")
    if container_mounts is not None:
        srun_parts.append(f"--container-mounts={shlex.quote(container_mounts)}")
    if container_env is not None:
        srun_parts.append(f"--container-env={shlex.quote(container_env)}")

    launch_command = " ".join([*srun_parts, "bash", "-lc", shlex.quote(worker_command)])
    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{num_shards - 1}%{max_parallel_shards}",
        f"#SBATCH --cpus-per-task={cpus_per_shard}",
        "#SBATCH --tasks=1",
        f"#SBATCH --mem={mem_gb}GB",
        f"#SBATCH -o {job_dir}/slurm-%A_%4a.out",
    ]
    if slurm_account is not None:
        directives.append(f"#SBATCH --account={slurm_account}")
    if slurm_partition is not None:
        directives.append(f"#SBATCH --partition={slurm_partition}")
    if slurm_qos is not None:
        directives.append(f"#SBATCH --qos={slurm_qos}")
    if slurm_constraint is not None:
        directives.append(f"#SBATCH --constraint={slurm_constraint}")
    if slurm_time is not None:
        directives.append(f"#SBATCH --time={slurm_time}")

    body = [
        "set -x -euo pipefail",
        f"cd {shlex.quote(str(workdir))}",
        launch_command,
        'echo "Finished processing shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"',
    ]
    if group is not None:
        body = [
            f"newgrp {shlex.quote(group)} <<'NEWGRP_EOF'",
            *body,
            "NEWGRP_EOF",
        ]

    sbatch = "\n".join([*directives, "", "", *body, ""])
    submission_path.write_text(sbatch, encoding="utf-8")

    job_id = None
    if not dry_run:
        output = subprocess.check_output(
            ["sbatch", "--parsable", str(submission_path)], encoding="utf-8"
        )
        job_id = output.strip().split(";", maxsplit=1)[0]
        logger.info("Launched sbatch job with job_id %s", job_id)
        logger.info("Logs are in %s", job_dir)
    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "submission_path": str(submission_path),
    }


def build_distributed_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit JSONL pretokenization as a SLURM array job."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSONL pretokenization config YAML file",
    )
    parser.add_argument("--pretokenized-data-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--max-parallel-shards",
        type=int,
        default=None,
        help="Maximum number of array shards to run at the same time.",
    )
    parser.add_argument("--cpus-per-shard", type=int, default=64)
    parser.add_argument("--mem-gb", type=int, default=None)
    parser.add_argument("--max-group-size", type=int, default=2**40)
    parser.add_argument("--validation-fraction", type=float, default=0.001)
    parser.add_argument(
        "--guarantee-validation-per-chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--slurm-partition", default="cpu")
    parser.add_argument("--slurm-qos", default="cpu-normal")
    parser.add_argument("--slurm-account", default=None)
    parser.add_argument("--slurm-constraint", default=None)
    parser.add_argument("--slurm-time", default="48:00:00")
    parser.add_argument(
        "--group",
        default=None,
        help="Optional Unix group to enter with newgrp before processing.",
    )
    parser.add_argument("--container-image", default=None)
    parser.add_argument("--container-mounts", default=None)
    parser.add_argument("--container-name", default=None)
    parser.add_argument("--container-env", default=None)
    parser.add_argument(
        "--container-remap-root",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-submit-source-check",
        action="store_true",
        help="Skip the submit-side check that each config source has visible JSONLs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def distributed_pretokenization_main(argv: list[str] | None = None) -> None:
    parser = build_distributed_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    result = submit_distributed_pretokenization(
        config_path=Path(args.config),
        pretokenized_data_dir=Path(args.pretokenized_data_dir),
        num_shards=args.num_shards,
        max_parallel_shards=args.max_parallel_shards,
        cpus_per_shard=args.cpus_per_shard,
        max_group_size=args.max_group_size,
        validation_fraction=args.validation_fraction,
        guarantee_validation_per_chunk=args.guarantee_validation_per_chunk,
        workdir=args.workdir,
        logs_dir=args.logs_dir,
        job_name=args.job_name,
        log_level=args.log_level,
        slurm_partition=args.slurm_partition,
        slurm_qos=args.slurm_qos,
        slurm_account=args.slurm_account,
        slurm_constraint=args.slurm_constraint,
        slurm_time=args.slurm_time,
        mem_gb=args.mem_gb,
        group=args.group,
        container_image=args.container_image,
        container_mounts=args.container_mounts,
        container_name=args.container_name,
        container_env=args.container_env,
        container_remap_root=args.container_remap_root,
        submit_source_check=not args.skip_submit_source_check,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if (
    __name__ == "__main__"
    and len(sys.argv) > 1
    and sys.argv[1]
    in {
        "distributed_process",
        "distributed-process",
    }
):
    distributed_pretokenization_main(sys.argv[2:])
    sys.exit(0)

import numpy as np

from mistral_fork.mistral.data.pretokenized.components.filters import (
    extract_text_chunks,
    filter_args_json,
    filter_multimodal_content,
    is_geometry_item,
    is_masked_multimodal_text,
    is_valid_filtered_multimodal_content,
    is_valid_text_chunks,
    multimodal_document_has_pds_quarantined_image,
    multimodal_text,
    preload_pds_filters,
    resolve_image_tar_offset,
    text_chunks_from_multimodal_content,
)
from mistral_fork.mistral.data.pretokenized.components.masking import (
    PretokenizedDoc,
    TokenizerLike,
    tokenize_and_mask_file_separator,
    tokenize_and_mask_github_pr,
    tokenize_and_mask_locodiff,
    tokenize_and_mask_midtraining,
    tokenize_and_mask_reasoning,
)
from mistral_fork.mistral.data.pretokenized.image_transform_metadata import (
    TransformImageMetadata,
)
from mistral_fork.mistral.data.pretokenized.indexed_mask import IndexedMaskBuilder
from mistral_fork.mistral.data.pretokenized.masking import DocumentMask
from mistral_fork.mistral.data.pretokenized.megatron_index_reader import (
    IndexedDatasetBuilder,
)
from mistral_fork.mistral.data.pretokenized.metadata import (
    ImageIndexEntry,
    _MetadataWriter,
    _TarOffset,
)
from mistral_fork.mistral.data.pretokenized.processing.config import (
    JsonlSourceConfig,
    _tokenizer_name_from_config,
    config_processing_tasks,
    load_jsonl_pretokenization_config,
    resolve_config_source,
    shard_config_processing_tasks,
)
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
    ChunkStats,
    ImageAnnotationType,
    MaskingStrategy,
    PretokenizedDatasourceMetadataV2,
    required_image_annotations_for_transform,
)
from mistral_fork.mistral.data.pretokenized.processing.paths_helpers import (
    PrefixPaths,
    check_chunk_is_complete,
    get_chunks_dir,
    get_datasource_metadata_v2_path,
    get_merged_folder,
    get_single_chunk_dir,
    get_single_chunk_prefix,
)
from mistral_fork.mistral.data.pretokenized.processing.tokenizers import (
    build_pretokenization_tokenizer,
)

DEFAULT_TOKEN_DTYPE = np.int32
DEFAULT_PROGRESS_LOG_INTERVAL_SECONDS = 60.0
GEOMETRY_NORMALIZE_FACTOR = 999


def atomic_write_once(path: Path, content: str) -> None:
    """Write content to path atomically, first writer wins."""
    tmp_path = path.with_name(f"tmp.{uuid.uuid4()}.{path.name}")
    tmp_path.write_text(content, encoding="utf-8")
    try:
        os.link(tmp_path, path)
    except FileExistsError:
        pass
    finally:
        tmp_path.unlink(missing_ok=True)


def mkdir_concurrent_ok(path: Path) -> None:
    """Create a directory when multiple Slurm workers may race to create it."""
    for _ in range(5):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except FileExistsError:
            if path.is_dir():
                return
            time.sleep(0.1)
        except FileNotFoundError:
            time.sleep(0.1)
    path.mkdir(parents=True, exist_ok=True)


def _transform_metadata_from_image(image: dict[str, Any]) -> TransformImageMetadata:
    metadata = TransformImageMetadata(
        width=int(image["width"]), height=int(image["height"])
    )
    transform_args = image.get("transform_args") or {}
    metadata.l2_reconstruction_score = dict(
        transform_args.get("l2_reconstruction_score") or {}
    )
    img_bbox_for_trim = transform_args.get("img_bbox_for_trim")
    if img_bbox_for_trim is not None:
        metadata.img_bbox_for_trim = img_bbox_for_trim
    return metadata


def _resolved_image_tar_offset(
    raw: dict[str, Any], document: dict[str, Any], default_tar_path: str | None
) -> dict[str, Any] | None:
    return resolve_image_tar_offset(raw, document, default_tar_path)


def _validate_required_image_annotations(
    metadata: TransformImageMetadata, required_annotations: set[ImageAnnotationType]
) -> None:
    if ImageAnnotationType.IMG_BBOX_FOR_TRIM in required_annotations:
        assert metadata.img_bbox_for_trim_is_computed is True, (
            "Image is missing img_bbox_for_trim, required by the source image transform"
        )
    if ImageAnnotationType.L2_RECONSTRUCTION_SCORE in required_annotations:
        assert metadata.l2_reconstruction_score, (
            "Image is missing l2_reconstruction_score, required by the source image transform"
        )


def _image_entry_from_raw(
    raw: dict[str, Any],
    *,
    img_id: int,
    pos_in_text: int,
    document: dict[str, Any],
    required_annotations: set[ImageAnnotationType],
) -> Any | None:
    tar_offset = _resolved_image_tar_offset(raw, document, document.get("image_tar"))
    if tar_offset is None:
        return None
    if tar_offset["tar_path"] is None:
        raise ValueError(
            f"Image entry has no tar_path and document has no image_tar: {raw.get('id')}"
        )
    metadata = _transform_metadata_from_image(raw)
    _validate_required_image_annotations(metadata, required_annotations)
    return ImageIndexEntry(
        img_id_in_doc=img_id,
        tar_offset=_TarOffset(**tar_offset),
        pos_in_text=pos_in_text,
        metadata=metadata,
    )


def _add_mask_span(
    sizes: list[int], values: list[bool], length: int, value: bool
) -> None:
    if length == 0:
        return
    if values and values[-1] == value:
        sizes[-1] += length
    else:
        sizes.append(length)
        values.append(value)


def _scaled_geometry_value(value: Any) -> int:
    value = float(value)
    return int(round(value * GEOMETRY_NORMALIZE_FACTOR))


def _point_values(raw: Any) -> list[float]:
    if isinstance(raw, dict):
        return [float(raw["x"]), float(raw["y"])]
    assert isinstance(raw, (list, tuple)) and len(raw) == 2, raw
    return [float(raw[0]), float(raw[1])]


def _scaled_point(raw: Any) -> list[int]:
    return [_scaled_geometry_value(value) for value in _point_values(raw)]


def _selected_grounding_fields(
    item: dict[str, Any], fields: list[str] | None
) -> dict[str, Any]:
    selected = tuple(fields) if fields is not None else ("label", "caption")
    result: dict[str, Any] = {}
    for field in selected:
        value = item.get(field)
        if value is None:
            continue
        result[field] = (
            value.lower() if field == "label" and isinstance(value, str) else value
        )
    return result


def _geometry_to_tokenized_dict(
    item: dict[str, Any],
    *,
    fields: list[str] | None,
    cur_image_idx: int,
) -> dict[str, Any]:
    if any(key in item for key in ("box_2d", "point_2d", "obb_2d", "mask_2d")):
        result = {
            key: item[key]
            for key in ("box_2d", "point_2d", "obb_2d", "mask_2d")
            if key in item
        }
    elif "top_left" in item and "bottom_right" in item:
        result = {
            "box_2d": [
                *_scaled_point(item["top_left"]),
                *_scaled_point(item["bottom_right"]),
            ]
        }
    elif "point" in item:
        result = {"point_2d": _scaled_point(item["point"])}
    elif "center" in item and "w" in item and "h" in item:
        result = {
            "obb_2d": [
                *_scaled_point(item["center"]),
                _scaled_geometry_value(item["w"]),
                _scaled_geometry_value(item["h"]),
                int(round(math.degrees(float(item.get("angle", 0.0))) % 360)),
            ]
        }
    elif "polygons" in item:
        polygons = []
        for polygon in item["polygons"]:
            vertices = polygon.get("vertices") if isinstance(polygon, dict) else polygon
            coords = []
            for point in vertices:
                coords.extend(_scaled_point(point))
            polygons.append(coords)
        result = {"mask_2d": polygons}
    else:
        raise TypeError(f"Unsupported grounding geometry item: {item}")

    result.update(_selected_grounding_fields(item, fields))
    image_idx = item.get("image_idx")
    if isinstance(image_idx, int):
        rel_image_idx = 1 + image_idx - cur_image_idx
        if not (cur_image_idx == 0 and image_idx == 0) and rel_image_idx != 0:
            result["image_idx"] = rel_image_idx
    return result


def _geometry_text(
    geometries: list[dict[str, Any]],
    *,
    fields: list[str] | None,
    cur_image_idx: int,
) -> str:
    return json.dumps(
        [
            _geometry_to_tokenized_dict(
                item, fields=fields, cur_image_idx=cur_image_idx
            )
            for item in geometries
        ],
        ensure_ascii=False,
    )


def _tokenize_multimodal_document(
    sample: dict[str, Any],
    tokenizer: TokenizerLike,
    content: list[Any],
    source_metadata: PretokenizedDatasourceMetadataV2,
    masking_strategy: MaskingStrategy | None,
) -> PretokenizedDoc:
    document = sample["multimodal_document"]
    tokens = [tokenizer.bos_id]
    mask_sizes: list[int] = []
    mask_values: list[bool] = []
    if masking_strategy is MaskingStrategy.MASK_MMDOC_CONTENT:
        _add_mask_span(mask_sizes, mask_values, 1, True)
    images: list[Any] = []
    pending_geometries: list[dict[str, Any]] = []

    def flush_geometries() -> None:
        nonlocal pending_geometries
        if not pending_geometries:
            return
        geometry_tokens = tokenizer.tokenize(
            _geometry_text(
                pending_geometries,
                fields=source_metadata.grounding_fields_to_tokenize,
                cur_image_idx=len(images),
            )
        )
        tokens.extend(geometry_tokens)
        if masking_strategy is MaskingStrategy.MASK_MMDOC_CONTENT:
            should_train = not any(
                bool(item.get("is_masked", False)) for item in pending_geometries
            )
            _add_mask_span(mask_sizes, mask_values, len(geometry_tokens), should_train)
        pending_geometries = []

    for item in content:
        text = multimodal_text(item)
        if text is not None:
            flush_geometries()
            text_tokens = tokenizer.tokenize(text)
            tokens.extend(text_tokens)
            if masking_strategy is MaskingStrategy.MASK_MMDOC_CONTENT:
                _add_mask_span(
                    mask_sizes,
                    mask_values,
                    len(text_tokens),
                    not is_masked_multimodal_text(item),
                )
        elif isinstance(item, dict) and is_geometry_item(item):
            if "geometries" in item:
                geometries = item["geometries"]
                assert isinstance(geometries, list), item
                pending_geometries.extend(geometries)
            else:
                pending_geometries.append(item)
        elif isinstance(item, dict):
            flush_geometries()
            image = _image_entry_from_raw(
                item,
                img_id=len(images),
                pos_in_text=len(tokens),
                document=document,
                required_annotations=source_metadata.available_image_annotations,
            )
            if image is not None:
                images.append(image)
        else:
            raise TypeError(
                f"Unsupported multimodal content item type: {type(item).__name__}"
            )
    flush_geometries()
    text_length_without_eos = len(tokens)
    images = [image for image in images if image.pos_in_text < text_length_without_eos]
    tokens.append(tokenizer.eos_id)
    mask = None
    if masking_strategy is MaskingStrategy.MASK_MMDOC_CONTENT:
        _add_mask_span(mask_sizes, mask_values, 1, True)
        mask = DocumentMask(sizes=mask_sizes, mask=mask_values)
    return PretokenizedDoc(tokens=tokens, images=images, mask=mask)


def _tokenize_text_chunks(
    tokenizer: TokenizerLike, chunks: list[str]
) -> PretokenizedDoc:
    tokens = [tokenizer.bos_id]
    for chunk in chunks:
        tokens.extend(tokenizer.tokenize(chunk))
    tokens.append(tokenizer.eos_id)
    return PretokenizedDoc(tokens=tokens, images=[], mask=None)


def pretokenize_sample(
    sample: dict[str, Any],
    source_config: JsonlSourceConfig,
    tokenizer: TokenizerLike,
    source_metadata: PretokenizedDatasourceMetadataV2,
) -> PretokenizedDoc | None:
    masking_strategy = source_config.masking_strategy
    if sample.get("source") == "reasoning_traces":
        assert masking_strategy == MaskingStrategy.MASK_REASONING_TRACES_QUESTIONS, (
            "Always use MaskingStrategy.MASK_REASONING_TRACES_QUESTIONS for reasoning traces"
        )
    chunks = extract_text_chunks(sample, source_config)
    filtered_multimodal_content = None
    if source_config.sample_type.is_multimodal() and "multimodal_document" in sample:
        # Doing it before filter incase standard image filtering can remove the flagged image
        if multimodal_document_has_pds_quarantined_image(
            sample["multimodal_document"], source_config.pds_filter
        ):
            return None
        filtered_multimodal_content = filter_multimodal_content(
            sample["multimodal_document"], source_config
        )
        chunks = text_chunks_from_multimodal_content(filtered_multimodal_content)
        if not is_valid_filtered_multimodal_content(
            sample["multimodal_document"],
            filtered_multimodal_content,
            source_config,
        ):
            return None
    if not is_valid_text_chunks(
        chunks, source_config, multimodal_content=filtered_multimodal_content
    ):
        return None
    if source_config.sample_type.is_multimodal() and "multimodal_document" in sample:
        if masking_strategy not in (None, MaskingStrategy.MASK_MMDOC_CONTENT):
            raise NotImplementedError(
                f"Unsupported image masking strategy: {masking_strategy}"
            )
        assert filtered_multimodal_content is not None
        doc = _tokenize_multimodal_document(
            sample,
            tokenizer,
            filtered_multimodal_content,
            source_metadata,
            masking_strategy,
        )
    elif masking_strategy in {
        MaskingStrategy.MASK_FILE_SEPARATOR,
        MaskingStrategy.MASK_FILE_SEPARATOR_NO_LAZINESS,
    }:
        assert len(chunks) == 1, (
            f"{masking_strategy.value} expects one formatted text chunk"
        )
        doc = tokenize_and_mask_file_separator(tokenizer, chunks[0])
    elif masking_strategy is MaskingStrategy.MASK_REASONING_TRACES_QUESTIONS:
        doc = tokenize_and_mask_reasoning(tokenizer, sample)
    elif masking_strategy is MaskingStrategy.MASK_MIDTRAINING_SEGMENTS:
        doc = tokenize_and_mask_midtraining(tokenizer, sample)
    elif masking_strategy in {
        MaskingStrategy.MASK_GITHUB_PR_FORMATTING,
        MaskingStrategy.MASK_GITHUB_PR_EXPANDED_CONTEXT,
        MaskingStrategy.MASK_GITHUB_PR_EXPANDED_CONTEXT_AND_CONTEXT,
    }:
        doc = tokenize_and_mask_github_pr(tokenizer, sample, masking_strategy)
    elif masking_strategy is MaskingStrategy.MASK_LOCODIFF_MARKERS:
        assert len(chunks) == 1, (
            "MASK_LOCODIFF_MARKERS expects one formatted text chunk"
        )
        doc = tokenize_and_mask_locodiff(tokenizer, chunks[0])
    elif masking_strategy is None:
        doc = _tokenize_text_chunks(tokenizer, chunks)
    else:
        raise NotImplementedError(f"Unsupported masking strategy: {masking_strategy}")
    doc.check(tokenizer.bos_id, tokenizer.eos_id)
    return doc


def _line_preview(line: str, max_chars: int = 500) -> str:
    preview = line.rstrip("\n\r")
    if len(preview) > max_chars:
        preview = (
            preview[:max_chars] + f"... <truncated {len(preview) - max_chars} chars>"
        )
    return repr(preview)


def _sample_preview(sample: dict[str, Any], max_chars: int = 500) -> str:
    preview = json.dumps(sample, ensure_ascii=False, default=str)
    if len(preview) > max_chars:
        preview = (
            preview[:max_chars] + f"... <truncated {len(preview) - max_chars} chars>"
        )
    return preview


def _is_skippable_pretokenization_error(error: Exception) -> bool:
    if isinstance(
        error, AssertionError
    ) and "required by the source image transform" in str(error):
        return False
    return "Regex error while tokenizing" in str(error) or isinstance(
        error, (AssertionError, KeyError, TypeError, ValueError)
    )


@dataclasses.dataclass(frozen=True)
class JsonlRecord:
    sample: dict[str, Any]
    line_number: int
    byte_offset: int


@dataclasses.dataclass(frozen=True)
class JsonlDecodeFailure:
    path: Path
    line_number: int
    byte_offset: int
    error: json.JSONDecodeError
    line_preview: str


def _iter_jsonl(path: Path) -> Iterator[JsonlRecord | JsonlDecodeFailure]:
    byte_offset = 0
    with path.open("rb") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.decode("utf-8")
            try:
                yield JsonlRecord(
                    sample=json.loads(line),
                    line_number=line_number,
                    byte_offset=byte_offset,
                )
            except json.JSONDecodeError as error:
                failure = JsonlDecodeFailure(
                    path=path,
                    line_number=line_number,
                    byte_offset=byte_offset,
                    error=error,
                    line_preview=_line_preview(line),
                )
                logger.warning(
                    "Skipping malformed JSONL record in %s at line %d "
                    "(byte_offset=%d): %s; line=%s",
                    failure.path,
                    failure.line_number,
                    failure.byte_offset,
                    failure.error,
                    failure.line_preview,
                )
                yield failure
            byte_offset += len(raw_line)


def _log_chunk_progress(
    *,
    jsonl_path: Path,
    elapsed: float,
    n_docs_total: int,
    n_docs_train: int,
    n_docs_validation: int,
    n_docs_filtered: int,
    n_docs_nv_quarantine_filtered: int,
    n_docs_tokenization_errors: int,
    n_tokens_total: int,
) -> None:
    jsonl_label = f"{jsonl_path.parent.name}/{jsonl_path.name}"
    tok_per_s = n_tokens_total / elapsed if elapsed > 0 else 0.0
    btok_per_h = tok_per_s * 3600 / 1e9
    logger.info(
        "%s: progress after %.1fs: docs=%d train=%d validation=%d "
        "filtered=%d nv_quarantine_filtered=%d tokenization_errors=%d "
        "tokens=%d, %.0f tok/s "
        "(%.3f B tok / h)",
        jsonl_label,
        elapsed,
        n_docs_total,
        n_docs_train,
        n_docs_validation,
        n_docs_filtered,
        n_docs_nv_quarantine_filtered,
        n_docs_tokenization_errors,
        n_tokens_total,
        tok_per_s,
        btok_per_h,
    )


def _write_training_doc(
    *,
    doc: PretokenizedDoc,
    builder: IndexedDatasetBuilder,
    metadata_writer: Any | None,
    mask_builder: IndexedMaskBuilder | None,
    has_loss_mask: bool,
) -> int:
    builder.add_document([doc.tokens], [len(doc.tokens)])
    if metadata_writer is not None:
        metadata_writer.add_document(doc.images)
    else:
        assert not doc.images
    if has_loss_mask:
        assert mask_builder is not None
        assert doc.mask is not None
        mask_builder.add_document(doc.mask)
        return sum(
            size
            for size, value in zip(doc.mask.sizes, doc.mask.mask, strict=True)
            if not value
        )
    assert doc.mask is None
    return 0


def build_metadata(
    *,
    source_config: JsonlSourceConfig,
    input_path: Path,
    n_chunks: int,
    tokenizer_path: str,
    validation_fraction: float,
    guarantee_validation_per_chunk: bool,
    tokenizer_name: TokenizerName,
) -> PretokenizedDatasourceMetadataV2:
    return PretokenizedDatasourceMetadataV2(
        filter_name=source_config.filter,
        tokenizer_name=tokenizer_name,
        filter_args_json=filter_args_json(source_config),
        tokenizer_path=tokenizer_path,
        original_source=str(input_path),
        n_chunks=n_chunks,
        validation_fraction=validation_fraction,
        guarantee_validation_per_chunk=guarantee_validation_per_chunk,
        has_image_metadata=source_config.sample_type.is_multimodal(),
        sample_type=source_config.sample_type,
        masking_strategy=source_config.masking_strategy,
        rare_token_filter=source_config.rare_token_filter,
        pds_filter=source_config.pds_filter,
        bad_document_filter=source_config.bad_document_filter,
        available_image_annotations=required_image_annotations_for_transform(
            source_config.image_transform_type
        ),
    )


def _tokenize_file_to_chunk(
    *,
    jsonl_path: Path,
    dst_single_chunk_dir: Path,
    source_config: JsonlSourceConfig,
    tokenizer: TokenizerLike,
    source_metadata: PretokenizedDatasourceMetadataV2,
    validation_fraction: float,
    guarantee_validation_per_chunk: bool,
    debug_doc_limit: int | None = None,
) -> ChunkStats:
    assert 0 <= validation_fraction <= 1, validation_fraction
    assert not guarantee_validation_per_chunk or validation_fraction > 0
    assert debug_doc_limit is None or debug_doc_limit > 0, debug_doc_limit

    logger.info("Opening %s", jsonl_path)

    mkdir_concurrent_ok(dst_single_chunk_dir)
    chunk_paths = PrefixPaths.from_prefix(get_single_chunk_prefix(dst_single_chunk_dir))
    build_validation = validation_fraction > 0
    has_loss_mask = source_metadata.has_loss_mask
    has_image_metadata = source_config.sample_type.is_multimodal()

    with ExitStack() as stack:
        validation_fp = (
            stack.enter_context(chunk_paths.validation.open("w", encoding="utf-8"))
            if build_validation
            else None
        )
        builder = IndexedDatasetBuilder(
            str(chunk_paths.prefix), dtype=DEFAULT_TOKEN_DTYPE
        )
        mask_builder = (
            stack.enter_context(IndexedMaskBuilder(chunk_paths.mask))
            if has_loss_mask
            else None
        )
        metadata_writer = (
            stack.enter_context(_MetadataWriter(chunk_paths.image_metadata))
            if has_image_metadata
            else None
        )

        n_docs_total = 0
        n_docs_train = 0
        n_docs_validation = 0
        n_docs_filtered = 0
        n_docs_nv_quarantine_filtered = 0
        n_docs_tokenization_errors = 0
        n_tokens_train = 0
        n_tokens_validation = 0
        n_masked_tokens = 0
        counter = 1.0 if guarantee_validation_per_chunk else 0.0
        proc_start = time.time()
        next_progress_log = proc_start + DEFAULT_PROGRESS_LOG_INTERVAL_SECONDS

        raw_samples = _iter_jsonl(jsonl_path)
        if debug_doc_limit is not None:
            raw_samples = islice(raw_samples, debug_doc_limit)

        def maybe_log_progress() -> None:
            nonlocal next_progress_log
            now = time.time()
            if now >= next_progress_log:
                _log_chunk_progress(
                    jsonl_path=jsonl_path,
                    elapsed=now - proc_start,
                    n_docs_total=n_docs_total,
                    n_docs_train=n_docs_train,
                    n_docs_validation=n_docs_validation,
                    n_docs_filtered=n_docs_filtered,
                    n_docs_nv_quarantine_filtered=n_docs_nv_quarantine_filtered,
                    n_docs_tokenization_errors=n_docs_tokenization_errors,
                    n_tokens_total=n_tokens_train + n_tokens_validation,
                )
                next_progress_log = now + DEFAULT_PROGRESS_LOG_INTERVAL_SECONDS

        for record in raw_samples:
            n_docs_total += 1
            if isinstance(record, JsonlDecodeFailure):
                n_docs_tokenization_errors += 1
                maybe_log_progress()
                continue
            raw_sample = record.sample
            line_number = record.line_number
            if should_filter_bad_document(
                source_config.bad_document_filter,
                jsonl_path=jsonl_path,
                byte_offset=record.byte_offset,
            ):
                n_docs_filtered += 1
                maybe_log_progress()
                continue
            will_validate = build_validation and (
                counter + validation_fraction >= 1
                if guarantee_validation_per_chunk
                else counter + validation_fraction > 1
            )
            validation_json = (
                json.dumps(raw_sample, ensure_ascii=False) if will_validate else None
            )
            try:
                doc = pretokenize_sample(
                    raw_sample, source_config, tokenizer, source_metadata
                )
            except Exception as error:
                if not _is_skippable_pretokenization_error(error):
                    raise
                logger.warning(
                    "Skipping untokenizable JSONL record in %s at line %d: %r; "
                    "keys=%s; sample=%s",
                    jsonl_path,
                    line_number,
                    error,
                    sorted(raw_sample.keys()),
                    _sample_preview(raw_sample),
                )
                n_docs_tokenization_errors += 1
                maybe_log_progress()
                continue

            if doc is None:
                n_docs_filtered += 1
                if (
                    source_config.sample_type.is_multimodal()
                    and "multimodal_document" in raw_sample
                    and multimodal_document_has_pds_quarantined_image(
                        raw_sample["multimodal_document"], source_config.pds_filter
                    )
                ):
                    n_docs_nv_quarantine_filtered += 1
                maybe_log_progress()
                continue
            if should_filter_rare_tokens(source_config.rare_token_filter, doc.tokens):
                n_docs_filtered += 1
                maybe_log_progress()
                continue

            counter += validation_fraction
            if will_validate:
                assert validation_fp is not None
                assert validation_json is not None
                validation_fp.write(validation_json + "\n")
                counter -= 1
                n_docs_validation += 1
                n_tokens_validation += len(doc.tokens)
            else:
                n_masked_tokens += _write_training_doc(
                    doc=doc,
                    builder=builder,
                    metadata_writer=metadata_writer,
                    mask_builder=mask_builder,
                    has_loss_mask=has_loss_mask,
                )
                n_docs_train += 1
                n_tokens_train += len(doc.tokens)
            maybe_log_progress()

        if n_docs_tokenization_errors > 5:
            raise RuntimeError(
                f"Too many tokenization errors: {n_docs_tokenization_errors} for {jsonl_path}"
            )
        builder.finalize()

    stats = ChunkStats(
        n_tokens_train=n_tokens_train,
        n_docs_train=n_docs_train,
        n_docs_filtered=n_docs_filtered,
        n_docs_total=n_docs_total,
        n_docs_validation=n_docs_validation,
        n_tokens_validation=n_tokens_validation,
        n_docs_tokenization_errors=n_docs_tokenization_errors,
        n_masked_tokens=n_masked_tokens,
        n_docs_nv_quarantine_filtered=n_docs_nv_quarantine_filtered,
    )
    chunk_paths.stats.write_text(
        json.dumps(stats.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )
    elapsed = time.time() - proc_start
    n_tokens_total = n_tokens_train + n_tokens_validation
    tok_per_s = n_tokens_total / elapsed if elapsed > 0 else 0.0
    btok_per_h = tok_per_s * 3600 / 1e9
    mbs = jsonl_path.stat().st_size / elapsed / 1024 / 1024 if elapsed > 0 else 0.0
    jsonl_label = f"{jsonl_path.parent.name}/{jsonl_path.name}"
    logger.info(
        "%s: done in %.1fs, %.3f MB/s, %.0f tok/s (%.3f B tok / h). %s",
        jsonl_label,
        elapsed,
        mbs,
        tok_per_s,
        btok_per_h,
        stats,
    )
    check_chunk_is_complete(
        chunk_paths.prefix, source_metadata=source_metadata, is_merged=False
    )
    return stats


def tokenize_file_to_chunk(
    *,
    jsonl_path: Path,
    dst_single_chunk_dir: Path,
    source_config: JsonlSourceConfig,
    tokenizer: TokenizerLike,
    source_metadata: PretokenizedDatasourceMetadataV2,
    validation_fraction: float,
    guarantee_validation_per_chunk: bool,
    debug_doc_limit: int | None = None,
) -> ChunkStats:
    tmp_dir = dst_single_chunk_dir.with_name(
        f"tmp.{uuid.uuid4()}.{dst_single_chunk_dir.name}"
    )
    try:
        stats = _tokenize_file_to_chunk(
            jsonl_path=jsonl_path,
            dst_single_chunk_dir=tmp_dir,
            source_config=source_config,
            tokenizer=tokenizer,
            source_metadata=source_metadata,
            validation_fraction=validation_fraction,
            guarantee_validation_per_chunk=guarantee_validation_per_chunk,
            debug_doc_limit=debug_doc_limit,
        )
        try:
            tmp_dir.rename(dst_single_chunk_dir)
        except (FileExistsError, OSError):
            logger.warning(
                "%s already exists, did you launch two pretokenizations in parallel?",
                dst_single_chunk_dir,
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return stats
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def build_tokenizer(tokenizer_path: str | Path) -> TokenizerLike:
    return build_pretokenization_tokenizer(tokenizer_path)


def _empty_stats() -> ChunkStats:
    return ChunkStats(
        n_tokens_train=0,
        n_docs_train=0,
        n_docs_filtered=0,
        n_docs_total=0,
        n_docs_validation=0,
        n_tokens_validation=0,
        n_docs_tokenization_errors=0,
        n_masked_tokens=0,
    )


def _jsonl_path_for_task(task: ConfigProcessingTask) -> Path:
    source_path = Path(task.source_config.jsonl_dir)
    return (
        source_path
        if source_path.suffix == ".jsonl"
        else source_path / task.jsonl_file_name
    )


@dataclasses.dataclass(frozen=True)
class _PoolTask:
    jsonl_path: Path
    pretok_dir: Path
    dst_single_chunk_dir: Path
    source_config: JsonlSourceConfig
    source_metadata: PretokenizedDatasourceMetadataV2
    tokenizer_path: str
    validation_fraction: float
    guarantee_validation_per_chunk: bool
    debug_first_chunk_docs: int | None


@dataclasses.dataclass(frozen=True)
class _PoolResult:
    pretok_dir: Path
    jsonl_path: Path
    filter_name: str
    masking_strategy: str | None
    ok: bool
    stats: ChunkStats | None = None
    error: str | None = None


_TOKENIZER_CACHE: dict[str, TokenizerLike] = {}


def _pool_worker_init(log_level: int) -> None:
    logging.basicConfig(level=log_level)


def _get_cached_tokenizer(tokenizer_path: str) -> TokenizerLike:
    tokenizer = _TOKENIZER_CACHE.get(tokenizer_path)
    if tokenizer is None:
        tokenizer = build_tokenizer(tokenizer_path)
        _TOKENIZER_CACHE[tokenizer_path] = tokenizer
    return tokenizer


def _process_pool_task(task: _PoolTask) -> _PoolResult:
    try:
        tokenizer = _get_cached_tokenizer(task.tokenizer_path)
        stats = tokenize_file_to_chunk(
            jsonl_path=task.jsonl_path,
            dst_single_chunk_dir=task.dst_single_chunk_dir,
            source_config=task.source_config,
            tokenizer=tokenizer,
            source_metadata=task.source_metadata,
            validation_fraction=task.validation_fraction,
            guarantee_validation_per_chunk=task.guarantee_validation_per_chunk,
            debug_doc_limit=task.debug_first_chunk_docs,
        )
        return _PoolResult(
            pretok_dir=task.pretok_dir,
            jsonl_path=task.jsonl_path,
            filter_name=task.source_config.filter.value,
            masking_strategy=(
                task.source_config.masking_strategy.value
                if task.source_config.masking_strategy is not None
                else None
            ),
            ok=True,
            stats=stats,
        )
    except Exception as exc:
        logger.exception("Failed to pretokenize %s", task.jsonl_path)
        return _PoolResult(
            pretok_dir=task.pretok_dir,
            jsonl_path=task.jsonl_path,
            filter_name=task.source_config.filter.value,
            masking_strategy=(
                task.source_config.masking_strategy.value
                if task.source_config.masking_strategy is not None
                else None
            ),
            ok=False,
            error=repr(exc),
        )


def process_config(
    *,
    config_path: Path,
    pretokenized_data_dir: Path,
    max_group_size: int,
    debug_first_chunk_docs: int | None,
    validation_fraction: float,
    guarantee_validation_per_chunk: bool,
    worker_id: int,
    num_workers: int,
    n_workers: int,
) -> dict[str, Any]:
    assert n_workers >= 1, n_workers
    config = load_jsonl_pretokenization_config(config_path)
    tokenizer_path = config.tokenizer_path
    tasks = config_processing_tasks(
        config,
        debug_doc_limit=debug_first_chunk_docs,
    )
    total_tasks = len(tasks)
    tasks = shard_config_processing_tasks(
        tasks, worker_id=worker_id, num_workers=num_workers
    )
    logger.info(
        "Worker %s/%s processing %s/%s JSONL chunk(s)",
        worker_id,
        num_workers,
        len(tasks),
        total_tasks,
    )

    tokenizer_name = _tokenizer_name_from_config(config)
    jsonl_root = Path(config.jsonl_root_data_dir)
    runnable_tasks = []
    skipped_merged_sources: set[Path] = set()
    for task in tasks:
        _, pretok_dir = resolve_config_source(
            task.source_config,
            jsonl_root=jsonl_root,
            tokenizer_name=tokenizer_name,
            pretokenized_data_dir=pretokenized_data_dir,
        )
        merged_dir = get_merged_folder(pretok_dir, max_group_size)
        if merged_dir.exists():
            if pretok_dir not in skipped_merged_sources:
                logger.info(
                    "%s is already merged; skipping pretokenization", pretok_dir
                )
                skipped_merged_sources.add(pretok_dir)
            continue
        runnable_tasks.append(task)

    stats = []
    task_results = []
    pool_tasks = []
    for task in runnable_tasks:
        _, pretok_dir = resolve_config_source(
            task.source_config,
            jsonl_root=jsonl_root,
            tokenizer_name=tokenizer_name,
            pretokenized_data_dir=pretokenized_data_dir,
        )
        jsonl_path = _jsonl_path_for_task(task)
        chunks_dir = get_chunks_dir(pretok_dir)
        mkdir_concurrent_ok(chunks_dir)
        metadata = build_metadata(
            source_config=task.source_config,
            input_path=Path(task.source_config.jsonl_dir),
            n_chunks=task.source_n_chunks,
            tokenizer_path=tokenizer_path,
            validation_fraction=validation_fraction,
            tokenizer_name=tokenizer_name,
            guarantee_validation_per_chunk=guarantee_validation_per_chunk,
        )
        metadata_path = get_datasource_metadata_v2_path(chunks_dir)
        atomic_write_once(metadata_path, metadata.model_dump_json(indent=2))
        reloaded_metadata = PretokenizedDatasourceMetadataV2.model_validate_json(
            metadata_path.read_text()
        )
        assert reloaded_metadata == metadata, (reloaded_metadata, metadata)

        dst_single_chunk_dir = get_single_chunk_dir(chunks_dir, jsonl_path)
        if dst_single_chunk_dir.exists():
            logger.info("%s already exists; skipping", dst_single_chunk_dir)
            task_stats = _empty_stats()
            task_results.append(
                {
                    "pretok_dir": str(pretok_dir),
                    "jsonl_path": str(jsonl_path),
                    "filter": task.source_config.filter.value,
                    "masking_strategy": (
                        task.source_config.masking_strategy.value
                        if task.source_config.masking_strategy is not None
                        else None
                    ),
                    "stats": task_stats.model_dump(mode="json"),
                }
            )
        else:
            pool_tasks.append(
                _PoolTask(
                    jsonl_path=jsonl_path,
                    pretok_dir=pretok_dir,
                    dst_single_chunk_dir=dst_single_chunk_dir,
                    source_config=task.source_config,
                    source_metadata=metadata,
                    tokenizer_path=tokenizer_path,
                    validation_fraction=validation_fraction,
                    guarantee_validation_per_chunk=guarantee_validation_per_chunk,
                    debug_first_chunk_docs=debug_first_chunk_docs,
                )
            )

    if pool_tasks:
        preload_pds_filters(
            [
                task.source_config.pds_filter
                for task in pool_tasks
                if task.source_config.pds_filter is not None
            ]
        )
        preload_rare_token_filters(
            [
                task.source_config.rare_token_filter
                for task in pool_tasks
                if task.source_config.rare_token_filter is not None
            ]
        )
        pool_size = min(n_workers, len(pool_tasks))
        logger.info(
            "Worker %s/%s pretokenizing %s file(s) with %s local process(es)",
            worker_id,
            num_workers,
            len(pool_tasks),
            pool_size,
        )
        failures = []
        n_done = 0
        with multiprocessing.Pool(
            pool_size, initializer=_pool_worker_init, initargs=(logger.level,)
        ) as pool:
            for result in pool.imap_unordered(_process_pool_task, pool_tasks):
                n_done += 1
                if result.ok:
                    assert result.stats is not None
                    stats.append(result.stats)
                    logger.info(
                        "[%s/%s] Done %s", n_done, len(pool_tasks), result.jsonl_path
                    )
                else:
                    failures.append(result)
                    logger.error(
                        "[%s/%s] FAILED %s: %s",
                        n_done,
                        len(pool_tasks),
                        result.jsonl_path,
                        result.error,
                    )
                task_results.append(
                    {
                        "pretok_dir": str(result.pretok_dir),
                        "jsonl_path": str(result.jsonl_path),
                        "filter": result.filter_name,
                        "masking_strategy": result.masking_strategy,
                        "stats": (
                            result.stats.model_dump(mode="json")
                            if result.stats is not None
                            else None
                        ),
                        "error": result.error,
                    }
                )
        if failures:
            details = "\n".join(
                f"  {result.jsonl_path}: {result.error}" for result in failures
            )
            raise RuntimeError(
                f"{len(failures)}/{len(pool_tasks)} files failed to pretokenize:\n"
                f"{details}"
            )

    return {
        "tasks": task_results,
        "stats": (ChunkStats.merge(stats) if stats else _empty_stats()).model_dump(
            mode="json"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSONL pretokenization config YAML file",
    )
    parser.add_argument("--pretokenized-data-dir", type=Path, required=True)
    parser.add_argument("--max-group-size", type=int, default=2**40)
    parser.add_argument("--validation-fraction", type=float, default=0.001)
    parser.add_argument(
        "--guarantee-validation-per-chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--worker_id", "--worker-id", dest="worker_id", type=int, default=0
    )
    parser.add_argument(
        "--num_workers", "--num-workers", dest="num_workers", type=int, default=1
    )
    parser.add_argument(
        "--n_workers",
        "--n-workers",
        dest="n_workers",
        type=int,
        default=8,
        help="Number of local processes used by this worker.",
    )
    parser.add_argument(
        "--debug-first-chunk-docs",
        type=int,
        default=None,
        help="Temporary debug mode: only pretokenize the first N raw docs from chunk 0",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"distributed_process", "distributed-process"}:
        distributed_pretokenization_main(argv[1:])
        return
    if argv and argv[0] in {"check_missing", "check-missing"}:
        result = check_missing_from_args(argv[1:])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["ok"]:
            raise SystemExit(1)
        return
    if argv and argv[0] == "process":
        argv = argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    config_path = Path(args.config)
    pretokenized_data_dir = Path(args.pretokenized_data_dir)
    max_group_size = args.max_group_size
    debug_first_chunk_docs = args.debug_first_chunk_docs
    validation_fraction = args.validation_fraction
    guarantee_validation_per_chunk = args.guarantee_validation_per_chunk
    worker_id = args.worker_id
    num_workers = args.num_workers
    n_workers = args.n_workers

    result = process_config(
        config_path=config_path,
        pretokenized_data_dir=pretokenized_data_dir,
        max_group_size=max_group_size,
        debug_first_chunk_docs=debug_first_chunk_docs,
        validation_fraction=validation_fraction,
        guarantee_validation_per_chunk=guarantee_validation_per_chunk,
        worker_id=worker_id,
        num_workers=num_workers,
        n_workers=n_workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
