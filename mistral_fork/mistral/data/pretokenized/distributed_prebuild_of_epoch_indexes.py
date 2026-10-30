import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mistral_fork.mistral.data.interleaved_pretokenized.selection_args import (
    SelectionStrategy,
)
from mistral_fork.mistral.data.pretokenized.datasource import (
    Datasource,
    DatasourceSpecs,
    MissingEpochIndex,
    PretokenizedImageEncoder,
)
from mistral_fork.mistral.data.pretokenized.document_filters import PdsFilter
from mistral_fork.mistral.data.pretokenized.pretokenized_utils import SourceItem
from mistral_fork.mistral.data.source import SampleType

logger = logging.getLogger(__name__)


# Ported from Mistral `mistral/checkpointing/checkpointing_utils.py::_wait_for_paths_to_exist`
# around line 1019 in the 2026-05-29 checkout.
def _wait_for_paths_to_exist(
    paths: list[Path], timeout_s: float, error_msg: str
) -> None:
    missing = set(paths)
    start = time.time()
    while missing:
        elapsed = time.time() - start
        if elapsed > timeout_s:
            missing_str = "\n - ".join(str(path) for path in sorted(missing)[:8])
            if len(missing) > 8:
                missing_str += f"\n and {len(missing) - 8} more"
            raise TimeoutError(
                f"{error_msg}\nMissing: {len(missing)} ranks after {elapsed}s\n"
                f"{missing_str}"
            )
        found = {path for path in missing if path.exists()}
        missing -= found
        time.sleep(0.1)


def _job_id() -> str:
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID") or "local"


def _restart_count() -> str:
    return os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")


# Ported in spirit from Mistral `mistral/checkpointing/checkpointing_utils.py::_make_unique`
# around line 1088 in the 2026-05-29 checkout.
def _make_unique(name: str) -> str:
    return f"{name}.{_job_id()}.{_restart_count()}"


# Ported from Mistral `mistral/checkpointing/checkpointing_utils.py::nfs_barrier`
# around line 1139 in the 2026-05-29 checkout.
def nfs_barrier(
    barrier_dir: Path,
    name: str,
    timeout_s: float,
    rank: int,
    world_size: int,
    error_msg: str,
) -> None:
    barrier_dir.mkdir(exist_ok=True, parents=True)
    this_barrier_dir = barrier_dir / _make_unique(f"dones.{name}")
    final_file = barrier_dir / _make_unique(f"all_done.{name}")

    if final_file.exists():
        raise RuntimeError(
            f"nfs_barrier collision detected: {name!r} was already used in this job. "
            f"Each barrier call must use a unique name. Found existing file: "
            f"{final_file}"
        )

    logger.info(f"This barrier dir: {this_barrier_dir}")
    this_barrier_dir.mkdir(exist_ok=True)
    (this_barrier_dir / f"done.{rank:05d}").touch()
    if rank == 0:
        paths = [this_barrier_dir / f"done.{r:05d}" for r in range(world_size)]
        _wait_for_paths_to_exist(paths, timeout_s=timeout_s, error_msg=error_msg)
        final_file.touch()
        shutil.rmtree(this_barrier_dir)
    else:
        _wait_for_paths_to_exist([final_file], timeout_s=timeout_s, error_msg=error_msg)


# Mirrors the `Datasource(...)` construction inside Mistral
# `mistral/data/pretokenized/build_pretokenized.py::prebuild_epoch_index`.
def _build_datasource(
    *,
    source: SourceItem,
    seed: int,
    seq_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    num_tokens: int,
    epoch_index_cache_path: str,
    image_encoder_build_fn: Callable[[str], PretokenizedImageEncoder] | None,
    image_transform_type: str | None,
    skip_images: bool,
    allow_epoch_index_building: bool,
    selection_strategy: SelectionStrategy | None,
    truncate_trailing_masked: bool,
    epoch_id: int | None,
    image_transform_workers: int = 1,
    epoch_index_pds_filter: PdsFilter | None = None,
) -> None:
    assert source.pretokenized_path is not None, "pretokenized_path must be set"
    specs = DatasourceSpecs(
        prefix_path=str(source.pretokenized_path),
        seed=seed,
        seq_len=seq_len,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        no_shuffling=False,
        percentage_data=source.percentage_data,
        no_partial_sequence=source.sample_type == SampleType.PRETRAIN_OCR,
        truncate_trailing_masked=truncate_trailing_masked,
    )
    if epoch_id is None:
        Datasource(
            specs=specs,
            num_toks=num_tokens,
            cache_root_path=Path(epoch_index_cache_path),
            image_encoder_build_fn=image_encoder_build_fn,
            image_transform_type=image_transform_type,
            allow_epoch_index_building=allow_epoch_index_building,
            skip_images=skip_images,
            selection_strategy=selection_strategy,
            image_transform_workers=image_transform_workers,
            epoch_index_pds_filter=epoch_index_pds_filter,
        )
    else:
        Datasource.prebuild_epoch_index(
            specs=specs,
            epoch_id=epoch_id,
            cache_root_path=Path(epoch_index_cache_path),
            image_encoder_build_fn=image_encoder_build_fn,
            image_transform_type=image_transform_type,
            allow_epoch_index_building=allow_epoch_index_building,
            skip_images=skip_images,
            selection_strategy=selection_strategy,
            image_transform_workers=image_transform_workers,
            epoch_index_pds_filter=epoch_index_pds_filter,
        )


@dataclass(frozen=True)
class _PrebuildTask:
    source: SourceItem
    seed: int
    seq_len: int
    bos_id: int
    eos_id: int
    pad_id: int
    num_tokens: int
    epoch_index_cache_path: str
    image_encoder_build_fn: Callable[[str], PretokenizedImageEncoder] | None
    skip_images: bool
    allow_epoch_index_building: bool
    selection_strategy: SelectionStrategy | None
    truncate_trailing_masked: bool
    epoch_id: int | None
    image_transform_workers: int
    epoch_index_pds_filter: PdsFilter | None


def _task_label(task: _PrebuildTask) -> str:
    assert task.source.pretokenized_path is not None, "pretokenized_path must be set"
    prefix_path = str(task.source.pretokenized_path)
    if task.epoch_id is None:
        return prefix_path
    return f"{prefix_path} epoch {task.epoch_id}"


def _prebuild_source(task: _PrebuildTask) -> str:
    source = task.source
    task_label = _task_label(task)
    logger.info(f"Launching prebuild for {task_label}")
    _build_datasource(
        source=source,
        seed=task.seed,
        seq_len=task.seq_len,
        bos_id=task.bos_id,
        eos_id=task.eos_id,
        pad_id=task.pad_id,
        num_tokens=task.num_tokens,
        epoch_index_cache_path=task.epoch_index_cache_path,
        image_encoder_build_fn=task.image_encoder_build_fn,
        image_transform_type=source.image_transform_type,
        skip_images=task.skip_images,
        allow_epoch_index_building=task.allow_epoch_index_building,
        selection_strategy=task.selection_strategy,
        truncate_trailing_masked=task.truncate_trailing_masked,
        epoch_id=task.epoch_id,
        image_transform_workers=task.image_transform_workers,
        epoch_index_pds_filter=task.epoch_index_pds_filter,
    )
    return task_label


# Ported from Mistral `mistral/data/pretokenized/build_pretokenized.py::prebuild_epoch_index`
# in the 2026-05-29 checkout.
def prebuild_epoch_index(
    *,
    source_items: list[SourceItem],
    seed: int,
    num_total_tokens: int,
    seq_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    epoch_index_cache_path: str,
    image_encoder_build_fn: Callable[[str], PretokenizedImageEncoder] | None,
    skip_images: bool,
    worker_id: int,
    num_workers: int,
    processes_per_worker: int = 1,
    allow_epoch_index_building: bool = True,
    selection_strategy: SelectionStrategy | None = None,
    truncate_trailing_masked: bool = False,
    image_transform_workers: int = 1,
    epoch_index_pds_filter: PdsFilter | None = None,
) -> None:
    assert 0 <= worker_id < num_workers, (worker_id, num_workers)
    assert processes_per_worker >= 1, (
        f"processes_per_worker must be positive, got {processes_per_worker}"
    )
    assert image_transform_workers >= 1, image_transform_workers
    if epoch_index_pds_filter is not None and skip_images:
        raise ValueError("--epoch-index-pds-filter cannot be used with --skip-images")

    source_items = sorted(
        source_items,
        key=lambda source: -source.compute_safe_num_tokens(
            num_total_tokens, sequence_length=seq_len
        ),
    )
    # Prioritize legacy horizon-based builds, then order explicit builds by epoch.
    # This controls round-robin rank assignment and per-rank submission order; it
    # does not impose a completion barrier between groups because ranks and local
    # process pools execute concurrently.
    source_epoch_pairs: list[tuple[SourceItem, int | None]] = [
        (source, None)
        for source in source_items
        if source.n_epoches_to_prebuild is None
    ]
    source_epoch_pairs.extend(
        (source, epoch_id)
        for epoch_id in range(
            max(
                (source.n_epoches_to_prebuild or 0 for source in source_items),
                default=0,
            )
        )
        for source in source_items
        if source.n_epoches_to_prebuild is not None
        and epoch_id < source.n_epoches_to_prebuild
    )
    tasks = [
        _PrebuildTask(
            source=source,
            seed=seed,
            seq_len=seq_len,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            num_tokens=source.compute_safe_num_tokens(
                num_total_tokens, sequence_length=seq_len
            ),
            epoch_index_cache_path=epoch_index_cache_path,
            image_encoder_build_fn=image_encoder_build_fn,
            skip_images=skip_images,
            allow_epoch_index_building=allow_epoch_index_building,
            selection_strategy=selection_strategy,
            truncate_trailing_masked=truncate_trailing_masked,
            epoch_id=epoch_id,
            image_transform_workers=image_transform_workers,
            epoch_index_pds_filter=epoch_index_pds_filter,
        )
        for source, epoch_id in source_epoch_pairs
    ]
    tasks = tasks[worker_id::num_workers]

    local_processes = min(processes_per_worker, len(tasks))
    logger.info(
        f"Going to process {len(tasks)} epoch-index tasks with {local_processes} local "
        f"processes (rank {worker_id}/{num_workers})"
    )
    if not tasks:
        return

    if local_processes == 1:
        for task in tasks:
            task_label = _prebuild_source(task)
            logger.info(f"Done prebuilding for {task_label}")
        return

    with ProcessPoolExecutor(max_workers=local_processes) as executor:
        future_to_task = {
            executor.submit(_prebuild_source, task): task for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            task_label = _task_label(task)
            try:
                future.result()
            except Exception as error:
                for pending in future_to_task:
                    pending.cancel()
                raise RuntimeError(
                    f"Epoch-index prebuild failed for {task_label} on rank "
                    f"{worker_id}/{num_workers}"
                ) from error
            logger.info(f"Done prebuilding for {task_label}")


# Inspired by Mistral
# `mistral/data/pretokenized/build_pretokenized.py::_check_all_epochs_indexes_are_present`.
def check_prebuild(
    *,
    source_items: list[SourceItem],
    seed: int,
    num_total_tokens: int,
    seq_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    epoch_index_cache_path: str,
    image_encoder_build_fn: Callable[[str], PretokenizedImageEncoder] | None,
    skip_images: bool,
    selection_strategy: SelectionStrategy | None = None,
    truncate_trailing_masked: bool = False,
    epoch_index_pds_filter: PdsFilter | None = None,
) -> None:
    if epoch_index_pds_filter is not None and skip_images:
        raise ValueError("--epoch-index-pds-filter cannot be used with --skip-images")
    logger.info("Checking if all epoch indexes are present...")
    total = len(source_items)
    missing_paths: list[Path] = []

    for source in source_items:
        assert source.pretokenized_path is not None, "pretokenized_path must be set"
        try:
            _build_datasource(
                source=source,
                seed=seed,
                seq_len=seq_len,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                num_tokens=source.compute_safe_num_tokens(
                    num_total_tokens, sequence_length=seq_len
                ),
                epoch_index_cache_path=epoch_index_cache_path,
                image_encoder_build_fn=image_encoder_build_fn,
                image_transform_type=source.image_transform_type,
                skip_images=skip_images,
                allow_epoch_index_building=False,
                selection_strategy=selection_strategy,
                truncate_trailing_masked=truncate_trailing_masked,
                epoch_id=None,
                image_transform_workers=1,
                epoch_index_pds_filter=epoch_index_pds_filter,
            )
        except MissingEpochIndex as e:
            logger.error(f"Missing epoch index for {source.pretokenized_path}: {e}")
            missing_paths.append(e.path)

    n_missing = len(missing_paths)
    n_present = total - n_missing
    logger.info(f"Epoch indexes present: {n_present}/{total}")
    if n_missing > 0:
        logger.error(f"Epoch indexes missing: {n_missing}/{total}")
        first_missing = "\n\n".join(str(path) for path in missing_paths[:5])
        raise RuntimeError(
            f"{n_missing}/{total} epoch indexes are missing. "
            f"First missing paths:\n\n{first_missing}"
        )
    logger.info(f"All epoch indexes are present: {n_present}/{total}")
