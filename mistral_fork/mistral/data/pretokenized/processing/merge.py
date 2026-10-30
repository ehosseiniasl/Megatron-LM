from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import multiprocessing
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)
MERGE_MODULE = "mistral_fork.mistral.data.pretokenized.processing.merge"


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _chunks_bin_size(chunks: list[PrefixPaths]) -> int:
    return sum(chunk.bin.stat().st_size for chunk in chunks)


def submit_distributed_merge(
    *,
    config_path: Path,
    pretokenized_data_dir: Path,
    num_shards: int,
    max_parallel_shards: int | None,
    cpus_per_shard: int,
    max_group_size: int,
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
    dry_run: bool,
) -> dict[str, str | None]:
    assert num_shards >= 1, num_shards
    max_parallel_shards = max_parallel_shards or num_shards
    mem_gb = mem_gb or 4 * cpus_per_shard
    workdir = Path.cwd().resolve() if workdir is None else workdir
    timestamp = time.strftime("%y%m%d_%H%M%S")
    logs_dir = logs_dir or pretokenized_data_dir / "logs"
    job_name = job_name or f"jsonl_merge_{timestamp}"
    job_dir = logs_dir / timestamp
    if job_dir.exists():
        job_dir = logs_dir / f"{timestamp}_{uuid.uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=False)
    submission_path = job_dir / "submission.sbatch"

    worker_command = " ".join(
        [
            "python3",
            "-m",
            MERGE_MODULE,
            "--config",
            shlex.quote(str(config_path)),
            "--pretokenized-data-dir",
            shlex.quote(str(pretokenized_data_dir)),
            "--max-group-size",
            str(max_group_size),
            "--worker-id",
            '"${SLURM_ARRAY_TASK_ID}"',
            "--num-workers",
            '"${SLURM_ARRAY_TASK_COUNT}"',
            "--n-workers",
            '"${SLURM_CPUS_PER_TASK}"',
            "--log-level",
            shlex.quote(log_level),
        ]
    )
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

    launch_command = " ".join(
        [*srun_parts, "bash", "-lc", shlex.quote(worker_command)]
    )
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
        'echo "Finished merging shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"',
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
        description="Submit JSONL pretokenized merge as a SLURM array job."
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
    parser.add_argument("--cpus-per-shard", type=int, default=8)
    parser.add_argument("--mem-gb", type=int, default=None)
    parser.add_argument("--max-group-size", type=int, default=2**40)
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
        help="Optional Unix group to enter with newgrp before merging.",
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def distributed_merge_main(argv: list[str] | None = None) -> None:
    parser = build_distributed_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    result = submit_distributed_merge(
        config_path=Path(args.config),
        pretokenized_data_dir=Path(args.pretokenized_data_dir),
        num_shards=args.num_shards,
        max_parallel_shards=args.max_parallel_shards,
        cpus_per_shard=args.cpus_per_shard,
        max_group_size=args.max_group_size,
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
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in {
    "distributed_merge",
    "distributed-merge",
}:
    distributed_merge_main(sys.argv[2:])
    sys.exit(0)

from mistral_fork.mistral.data.pretokenized.indexed_mask import merge_indexed_mask_files
from mistral_fork.mistral.data.pretokenized.megatron_index_reader import IndexedDatasetBuilder
from mistral_fork.mistral.data.pretokenized.metadata import _MetadataWriter
from mistral_fork.mistral.data.pretokenized.processing.config import (
    _tokenizer_name_from_config,
    load_jsonl_pretokenization_config,
    resolve_config_source,
)
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
    ChunkStats,
    PretokenizedDatasourceMetadataV2,
)
from mistral_fork.mistral.data.pretokenized.processing.paths_helpers import (
    PrefixPaths,
    check_chunk_is_complete,
    check_no_conflicting_merged_folder_for_source,
    get_chunks_dir,
    get_datasource_metadata_v2_path,
    get_merged_folder,
    get_single_chunk_prefix,
)


def _check_bin_files_present(chunks: list[PrefixPaths]) -> None:
    missing = [chunk.bin for chunk in chunks if not chunk.bin.exists()]
    assert not missing, f"Missing chunk .bin files: {missing}"


def _get_group_slices_by_max_mem(
    bin_files: list[Path], max_group_size: int
) -> list[int]:
    total_size = sum(bin_file.stat().st_size for bin_file in bin_files)
    n_chunks = max(1, math.ceil(total_size / max_group_size))
    target_size = total_size / n_chunks
    distance = target_size
    result = [0]
    for i, bin_file in enumerate(bin_files):
        if distance < 0:
            result.append(i)
            distance = target_size
        distance -= bin_file.stat().st_size
    result.append(len(bin_files))
    return result


def merge_stats(input_stats_paths: list[Path], dst_dir: Path) -> ChunkStats:
    stats = [
        ChunkStats.model_validate_json(path.read_text()) for path in input_stats_paths
    ]
    merged = ChunkStats.merge(stats)
    (dst_dir / "merged_stats.json").write_text(
        merged.model_dump_json(indent=2), encoding="utf-8"
    )
    return merged


def merge_validation_jsonl(
    input_validation_paths: list[Path], pretokenized_folder: Path
) -> None:
    merged_jsonl_path = pretokenized_folder / "validation.jsonl"
    if merged_jsonl_path.exists():
        return
    tmp_path = merged_jsonl_path.with_name(
        f"tmp.{uuid.uuid4()}.{merged_jsonl_path.name}"
    )
    with tmp_path.open("w", encoding="utf-8") as output:
        for path in input_validation_paths:
            with path.open("r", encoding="utf-8") as input_file:
                shutil.copyfileobj(input_file, output)
    tmp_path.rename(merged_jsonl_path)


def _merge_chunks(
    src_chunks: list[PrefixPaths],
    dst_merged_prefix: Path,
    metadata: PretokenizedDatasourceMetadataV2,
) -> None:
    start_time = time.monotonic()
    total_size = _chunks_bin_size(src_chunks)
    logger.info(
        "Merging %d chunks into %s (%s)",
        len(src_chunks),
        dst_merged_prefix,
        _format_bytes(total_size),
    )
    dst = PrefixPaths.from_prefix(dst_merged_prefix)
    first_chunk = src_chunks[0]
    logger.info(
        "Copying initial chunk %s -> %s (%s)",
        first_chunk.prefix,
        dst.prefix,
        _format_bytes(first_chunk.bin.stat().st_size),
    )
    chunk_start_time = time.monotonic()
    builder = IndexedDatasetBuilder.from_chunk(
        str(first_chunk.prefix), str(dst.prefix)
    )
    logger.info(
        "Copied initial chunk %s in %.1fs",
        first_chunk.prefix,
        time.monotonic() - chunk_start_time,
    )
    for chunk_index, src_chunk in enumerate(src_chunks[1:], start=2):
        chunk_start_time = time.monotonic()
        logger.info(
            "Merging chunk %d/%d: %s (%s)",
            chunk_index,
            len(src_chunks),
            src_chunk.prefix,
            _format_bytes(src_chunk.bin.stat().st_size),
        )
        builder.merge_from(str(src_chunk.prefix))
        logger.info(
            "Merged chunk %d/%d in %.1fs",
            chunk_index,
            len(src_chunks),
            time.monotonic() - chunk_start_time,
        )
    logger.info("Finalizing merged chunk %s", dst.prefix)
    finalize_start_time = time.monotonic()
    builder.finalize()
    logger.info(
        "Finalized merged chunk %s in %.1fs",
        dst.prefix,
        time.monotonic() - finalize_start_time,
    )

    if metadata.has_loss_mask:
        mask_start_time = time.monotonic()
        logger.info("Merging loss masks into %s", dst.mask)
        merge_indexed_mask_files([chunk.mask for chunk in src_chunks], dst.mask)
        logger.info("Merged loss masks in %.1fs", time.monotonic() - mask_start_time)
    if metadata.has_image_metadata:
        image_metadata_start_time = time.monotonic()
        logger.info("Merging image metadata into %s", dst.image_metadata)
        with _MetadataWriter(dst.image_metadata) as writer:
            for src_chunk in src_chunks:
                writer.merge_from(src_chunk.image_metadata)
        logger.info(
            "Merged image metadata in %.1fs",
            time.monotonic() - image_metadata_start_time,
        )
    check_chunk_is_complete(dst.prefix, source_metadata=metadata, is_merged=True)
    logger.info(
        "Finished merging %d chunks into %s in %.1fs",
        len(src_chunks),
        dst_merged_prefix,
        time.monotonic() - start_time,
    )


def _delete_chunks_dir(source_dir: Path) -> None:
    chunks_dir = get_chunks_dir(source_dir)
    if not chunks_dir.exists():
        return
    to_delete = chunks_dir.with_name("chunks_to_delete")
    logger.info("Deleting source chunks directory %s", chunks_dir)
    chunks_dir.rename(to_delete)
    shutil.rmtree(to_delete)
    logger.info("Deleted source chunks directory %s", chunks_dir)


def merge_source(*, source_dir: Path, max_group_size: int) -> Path:
    source_start_time = time.monotonic()
    logger.info("Starting merge for source_dir=%s", source_dir)
    check_no_conflicting_merged_folder_for_source(source_dir, max_group_size)
    dst_merged_dir = get_merged_folder(source_dir, max_group_size)
    if dst_merged_dir.exists():
        logger.info("%s already exists; assuming it is complete", dst_merged_dir)
        _delete_chunks_dir(source_dir)
        return dst_merged_dir

    chunks_dir = get_chunks_dir(source_dir)
    metadata_path = get_datasource_metadata_v2_path(chunks_dir)
    metadata = PretokenizedDatasourceMetadataV2.model_validate_json(
        metadata_path.read_text()
    )
    logger.info(
        "Loaded metadata from %s: n_chunks=%d validation_fraction=%s "
        "has_loss_mask=%s has_image_metadata=%s",
        metadata_path,
        metadata.n_chunks,
        metadata.validation_fraction,
        metadata.has_loss_mask,
        metadata.has_image_metadata,
    )
    present_chunks = [
        path
        for path in sorted(chunks_dir.iterdir())
        if path.is_dir() and not path.name.startswith("tmp")
    ]
    logger.info(
        "Found %d present chunk directories in %s",
        len(present_chunks),
        chunks_dir,
    )
    assert present_chunks, f"No chunk directories found in {chunks_dir}"
    assert len(present_chunks) == metadata.n_chunks, (metadata.n_chunks, present_chunks)

    chunks_to_merge = [
        PrefixPaths.from_prefix(get_single_chunk_prefix(chunk))
        for chunk in present_chunks
    ]
    _check_bin_files_present(chunks_to_merge)
    logger.info(
        "Validated presence of %d bin files totaling %s",
        len(chunks_to_merge),
        _format_bytes(_chunks_bin_size(chunks_to_merge)),
    )
    chunk_check_start_time = time.monotonic()
    for chunk in chunks_to_merge:
        check_chunk_is_complete(chunk, source_metadata=metadata, is_merged=False)
    logger.info(
        "Checked %d source chunks for completeness in %.1fs",
        len(chunks_to_merge),
        time.monotonic() - chunk_check_start_time,
    )
    if metadata.validation_fraction > 0:
        validation_start_time = time.monotonic()
        logger.info("Merging validation jsonl files for %s", source_dir)
        merge_validation_jsonl(
            [chunk.validation for chunk in chunks_to_merge], source_dir
        )
        logger.info(
            "Merged validation jsonl files in %.1fs",
            time.monotonic() - validation_start_time,
        )

    group_slices = _get_group_slices_by_max_mem(
        [chunk.bin for chunk in chunks_to_merge], max_group_size
    )
    n_groups = len(group_slices) - 1
    logger.info(
        "Split %d chunks into %d merge groups with max_group_size=%s",
        len(chunks_to_merge),
        n_groups,
        _format_bytes(max_group_size),
    )
    tmp_dst = dst_merged_dir.with_name(f"tmp.{uuid.uuid4()}.{dst_merged_dir.name}")
    tmp_dst.mkdir(parents=True, exist_ok=False)
    stats_start_time = time.monotonic()
    logger.info("Merging stats into %s", tmp_dst)
    merge_stats([chunk.stats for chunk in chunks_to_merge], tmp_dst)
    logger.info("Merged stats in %.1fs", time.monotonic() - stats_start_time)
    for group_index, (start, end) in enumerate(
        zip(group_slices[:-1], group_slices[1:], strict=True), start=1
    ):
        logger.info(
            "Starting merge group %d/%d for chunk slice [%d:%d]",
            group_index,
            n_groups,
            start,
            end,
        )
        _merge_chunks(
            chunks_to_merge[start:end],
            tmp_dst / f"chunks_{start:05}-{end:05}",
            metadata,
        )
        logger.info(
            "Finished merge group %d/%d for chunk slice [%d:%d]",
            group_index,
            n_groups,
            start,
            end,
        )
    get_datasource_metadata_v2_path(tmp_dst).write_text(
        metadata.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("Renaming temporary merge directory %s -> %s", tmp_dst, dst_merged_dir)
    tmp_dst.rename(dst_merged_dir)

    _delete_chunks_dir(source_dir)
    logger.info(
        "Finished merge for source_dir=%s -> %s in %.1fs",
        source_dir,
        dst_merged_dir,
        time.monotonic() - source_start_time,
    )
    return dst_merged_dir


@dataclasses.dataclass(frozen=True)
class _MergePoolTask:
    source_index: int
    assigned_index: int
    n_assigned_sources: int
    raw_path: Path
    pretok_dir: Path
    max_group_size: int


@dataclasses.dataclass(frozen=True)
class _MergePoolResult:
    source_index: int
    assigned_index: int
    n_assigned_sources: int
    raw_path: Path
    pretok_dir: Path
    ok: bool
    merged_dir: Path | None = None
    error: str | None = None


def _pool_worker_init(log_level: int) -> None:
    logging.basicConfig(level=log_level)


def _process_merge_pool_task(task: _MergePoolTask) -> _MergePoolResult:
    try:
        logger.info(
            "Starting assigned source %d/%d (config source index %d): "
            "raw=%s pretok=%s",
            task.assigned_index,
            task.n_assigned_sources,
            task.source_index,
            task.raw_path,
            task.pretok_dir,
        )
        merged_dir = merge_source(
            source_dir=task.pretok_dir,
            max_group_size=task.max_group_size,
        )
        logger.info(
            "Finished assigned source %d/%d (config source index %d): merged=%s",
            task.assigned_index,
            task.n_assigned_sources,
            task.source_index,
            merged_dir,
        )
        return _MergePoolResult(
            source_index=task.source_index,
            assigned_index=task.assigned_index,
            n_assigned_sources=task.n_assigned_sources,
            raw_path=task.raw_path,
            pretok_dir=task.pretok_dir,
            ok=True,
            merged_dir=merged_dir,
        )
    except Exception as exc:
        logger.exception("Failed to merge %s", task.pretok_dir)
        return _MergePoolResult(
            source_index=task.source_index,
            assigned_index=task.assigned_index,
            n_assigned_sources=task.n_assigned_sources,
            raw_path=task.raw_path,
            pretok_dir=task.pretok_dir,
            ok=False,
            error=repr(exc),
        )


def merge_config(
    *,
    config_path: Path,
    pretokenized_data_dir: Path,
    max_group_size: int,
    worker_id: int = 0,
    num_workers: int = 1,
    n_workers: int = 1,
) -> dict[str, object]:
    assert num_workers >= 1, num_workers
    assert 0 <= worker_id < num_workers, (worker_id, num_workers)
    assert n_workers >= 1, n_workers
    config = load_jsonl_pretokenization_config(config_path)
    tokenizer_name = _tokenizer_name_from_config(config)
    jsonl_root = Path(config.jsonl_root_data_dir)
    indexed_sources = list(enumerate(config.sources))
    assigned_sources = indexed_sources[worker_id::num_workers]
    logger.info(
        "Merge worker %d/%d loaded %d total sources from %s; assigned %d sources",
        worker_id,
        num_workers,
        len(indexed_sources),
        config_path,
        len(assigned_sources),
    )
    pool_tasks = []
    for assigned_index, (source_index, source_config) in enumerate(
        assigned_sources, start=1
    ):
        raw_path, pretok_dir = resolve_config_source(
            source_config,
            jsonl_root=jsonl_root,
            tokenizer_name=tokenizer_name,
            pretokenized_data_dir=pretokenized_data_dir,
        )
        pool_tasks.append(
            _MergePoolTask(
                source_index=source_index,
                assigned_index=assigned_index,
                n_assigned_sources=len(assigned_sources),
                raw_path=raw_path,
                pretok_dir=pretok_dir,
                max_group_size=max_group_size,
            )
        )

    pool_size = min(n_workers, len(pool_tasks))
    if pool_size == 0:
        return {}
    logger.info(
        "Worker %d/%d merging %d source(s) with %d local process(es)",
        worker_id,
        num_workers,
        len(pool_tasks),
        pool_size,
    )

    results = []
    if pool_size == 1:
        for task in pool_tasks:
            results.append(_process_merge_pool_task(task))
    else:
        with multiprocessing.Pool(
            pool_size,
            initializer=_pool_worker_init,
            initargs=(logger.getEffectiveLevel(),),
        ) as pool:
            for pool_result in pool.imap_unordered(
                _process_merge_pool_task, pool_tasks
            ):
                results.append(pool_result)
                if pool_result.ok:
                    logger.info(
                        "[%d/%d] Done %s",
                        len(results),
                        len(pool_tasks),
                        pool_result.raw_path,
                    )
                else:
                    logger.error(
                        "[%d/%d] FAILED %s: %s",
                        len(results),
                        len(pool_tasks),
                        pool_result.raw_path,
                        pool_result.error,
                    )

    failures = [pool_result for pool_result in results if not pool_result.ok]
    if failures:
        details = "\n".join(
            f"  {pool_result.raw_path}: {pool_result.error}"
            for pool_result in failures
        )
        raise RuntimeError(
            f"{len(failures)}/{len(pool_tasks)} sources failed to merge:\n{details}"
        )

    result = {}
    for pool_result in sorted(results, key=lambda item: item.assigned_index):
        assert pool_result.merged_dir is not None
        logger.info(
            "Worker %d/%d finished assigned source %d/%d "
            "(config source index %d): merged=%s",
            worker_id,
            num_workers,
            pool_result.assigned_index,
            pool_result.n_assigned_sources,
            pool_result.source_index,
            pool_result.merged_dir,
        )
        result[str(pool_result.raw_path)] = {
            "pretok_dir": str(pool_result.pretok_dir),
            "merged_dir": str(pool_result.merged_dir),
        }
    return result


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
    parser.add_argument(
        "--worker_id", "--worker-id", dest="worker_id", type=int, default=0
    )
    parser.add_argument(
        "--num_workers", "--num-workers", dest="num_workers", type=int, default=1
    )
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"distributed_merge", "distributed-merge"}:
        distributed_merge_main(argv[1:])
        return
    if argv and argv[0] == "merge":
        argv = argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    config_path = Path(args.config)
    pretokenized_data_dir = Path(args.pretokenized_data_dir)
    max_group_size = args.max_group_size
    worker_id = args.worker_id
    num_workers = args.num_workers
    n_workers = args.n_workers

    result = merge_config(
        config_path=config_path,
        pretokenized_data_dir=pretokenized_data_dir,
        max_group_size=max_group_size,
        worker_id=worker_id,
        num_workers=num_workers,
        n_workers=n_workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
