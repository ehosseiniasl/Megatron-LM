import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mistral_fork.mistral.data.pretokenized import (
    distributed_prebuild_of_epoch_indexes as distributed_prebuild,
)
from mistral_fork.mistral.data.pretokenized.document_filters import PdsFilter
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.pretokenized_utils import (
    ConstantWeightSchedule,
    SourceItem,
)
from mistral_fork.mistral.data.pretokenized.processing import (
    prebuild_epoch_indexes_megatron as prebuild_cli,
)
from mistral_fork.mistral.data.source import SampleType


def _source(
    name: str, weight: float, n_epoches_to_prebuild: int | None = None
) -> SourceItem:
    path = Path("/tmp") / name
    return SourceItem(
        source_path=path,
        sample_type=SampleType.PRETRAIN,
        weight_schedule=ConstantWeightSchedule(weight),
        percentage_data=1.0,
        filter_profile=FilterArgsName.NO_FILTER,
        pretokenized_path=path,
        n_epoches_to_prebuild=n_epoches_to_prebuild,
    )


def _prebuild_kwargs(source_items: list[SourceItem]) -> dict[str, Any]:
    return {
        "source_items": source_items,
        "seed": 0,
        "num_total_tokens": 10_000,
        "seq_len": 16,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 0,
        "epoch_index_cache_path": "/tmp/epoch-index",
        "image_encoder_build_fn": None,
        "skip_images": True,
        "worker_id": 1,
        "num_workers": 2,
        "processes_per_worker": 2,
    }


def test_prebuild_epoch_index_schedules_rank_sources_on_local_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[str] = []
    executor_sizes: list[int] = []

    class ImmediateFuture:
        def __init__(self, task: distributed_prebuild._PrebuildTask) -> None:
            self.task = task

        def result(self) -> str:
            assert self.task.source.pretokenized_path is not None
            path = str(self.task.source.pretokenized_path)
            submitted.append(path)
            return path

        def cancel(self) -> bool:
            return False

    class ImmediateExecutor:
        def __init__(self, max_workers: int) -> None:
            executor_sizes.append(max_workers)

        def __enter__(self) -> "ImmediateExecutor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(
            self, function: object, task: distributed_prebuild._PrebuildTask
        ) -> ImmediateFuture:
            assert function is distributed_prebuild._prebuild_source
            return ImmediateFuture(task)

    monkeypatch.setattr(distributed_prebuild, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(distributed_prebuild, "as_completed", lambda futures: futures)

    sources = [
        _source("smallest", 0.1),
        _source("largest", 0.4),
        _source("second", 0.3),
        _source("third", 0.2),
    ]
    distributed_prebuild.prebuild_epoch_index(**_prebuild_kwargs(sources))

    assert executor_sizes == [2]
    assert submitted == ["/tmp/second", "/tmp/smallest"]


def test_prebuild_epoch_index_rejects_zero_local_processes() -> None:
    kwargs = _prebuild_kwargs([_source("source", 1.0)])
    kwargs["processes_per_worker"] = 0

    with pytest.raises(AssertionError, match="processes_per_worker must be positive"):
        distributed_prebuild.prebuild_epoch_index(**kwargs)


def test_prebuild_epoch_index_orders_legacy_sources_before_epoch_major_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[tuple[str, int | None]] = []

    def record_task(task: distributed_prebuild._PrebuildTask) -> str:
        assert task.source.pretokenized_path is not None
        prefix_path = str(task.source.pretokenized_path)
        submitted.append((prefix_path, task.epoch_id))
        return distributed_prebuild._task_label(task)

    monkeypatch.setattr(distributed_prebuild, "_prebuild_source", record_task)
    sources = [
        _source("three_epochs", 0.4, n_epoches_to_prebuild=3),
        _source("two_epochs", 0.3, n_epoches_to_prebuild=2),
        _source("legacy", 0.2),
    ]
    kwargs = _prebuild_kwargs(sources)
    kwargs["processes_per_worker"] = 1
    kwargs["worker_id"] = 0
    kwargs["num_workers"] = 1

    distributed_prebuild.prebuild_epoch_index(**kwargs)

    assert submitted == [
        ("/tmp/legacy", None),
        ("/tmp/three_epochs", 0),
        ("/tmp/two_epochs", 0),
        ("/tmp/three_epochs", 1),
        ("/tmp/two_epochs", 1),
        ("/tmp/three_epochs", 2),
    ]


def test_prebuild_epoch_index_skips_pool_when_rank_has_no_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_executor(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"Unexpected executor construction: {args=}, {kwargs=}")

    monkeypatch.setattr(distributed_prebuild, "ProcessPoolExecutor", fail_executor)
    kwargs = _prebuild_kwargs([_source("source", 1.0)])
    kwargs["worker_id"] = 3
    kwargs["num_workers"] = 4

    distributed_prebuild.prebuild_epoch_index(**kwargs)


def test_prebuild_epoch_index_rejects_quarantine_filter_with_skip_images() -> None:
    kwargs = _prebuild_kwargs([_source("source", 1.0)])
    kwargs["epoch_index_pds_filter"] = PdsFilter.NVIDIA_PDS_QUARANTINE_V3

    with pytest.raises(ValueError, match="cannot be used with --skip-images"):
        distributed_prebuild.prebuild_epoch_index(**kwargs)


def test_prebuild_cli_passes_quarantine_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        sources=[],
        seed=0,
        original_seq_len=16,
        epoch_index_cache_path="/tmp/cache",
        bos_id=1,
        eos_id=2,
        pad_id=0,
        skip_images=False,
        truncate_trailing_masked=False,
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(prebuild_cli, "_load_config", lambda *args, **kwargs: (config, 16))
    monkeypatch.setattr(prebuild_cli, "config_to_source_items", lambda config: [])
    monkeypatch.setattr(prebuild_cli, "build_image_encoder_fn", lambda config: None)
    monkeypatch.setattr(
        prebuild_cli,
        "prebuild_epoch_index",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prebuild_epoch_indexes_megatron.py",
            "prebuild",
            "--config",
            "/tmp/config.yaml",
            "--epoch-index-pds-filter",
            PdsFilter.NVIDIA_PDS_QUARANTINE_V3.value,
        ],
    )

    prebuild_cli.main()

    assert captured["epoch_index_pds_filter"] is PdsFilter.NVIDIA_PDS_QUARANTINE_V3
