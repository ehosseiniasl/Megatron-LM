"""Validate and atomically record ordered VLMEvalKit suite progress."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from .configured_eval_run import BenchmarkSuiteConfig
from .utils import (
    StrictJsonError,
    is_safe_relative_path,
    load_json_bytes,
    require_shared_file,
    resolve_directory,
    write_json,
)
from .vlmeval_result_contract import (
    DATASET_STATUS_RECORD_FIELDS,
    METRICS_FILENAME,
    STATUS_FILENAME,
    normalized_metrics_from_payload,
)


class VLMEvalSuiteStatusError(RuntimeError):
    """Raised when suite progress cannot be recorded or trusted."""


@dataclass(frozen=True)
class SuiteDatasetStatus:
    """One immutable dataset ledger state and optional terminal artifact paths.

    Paths are normalized relative to the suite run directory. Callers do not
    mutate this record; invalid membership, artifact, or payload contracts
    raise :class:`VLMEvalSuiteStatusError` during validation or persistence.
    """

    dataset: str
    status: str
    dataset_directory: str
    status_path: str | None
    metrics_path: str | None


@dataclass(frozen=True)
class VLMEvalSuiteStatusArtifact:
    """The read or written ledger path and its immutable ordered records.

    ``artifact`` is a shared regular file under the suite run directory, and
    ``dataset_statuses`` has the configured suite order. No record or artifact
    is mutated by construction; invalid ledger state raises
    :class:`VLMEvalSuiteStatusError` in the owning operations.
    """

    artifact: Path
    dataset_statuses: tuple[SuiteDatasetStatus, ...]


SUITE_STATUS_FILENAME = "vlmeval_suite_status.json"
_SAFE_COMPONENT = re.compile(r"[a-z0-9][a-z0-9-]*")
_STATUSES = frozenset({"completed", "failed", "not_started"})


def suite_dataset_component(dataset: str) -> str:
    """Return ``dataset``'s deterministic safe directory component.

    The input is not mutated. A name with no usable lowercase alphanumeric
    component raises :class:`VLMEvalSuiteStatusError`.
    """
    component = re.sub(r"[^a-z0-9]+", "-", dataset.lower()).strip("-")
    if not _SAFE_COMPONENT.fullmatch(component):
        raise VLMEvalSuiteStatusError(f"dataset has no safe directory name: {dataset!r}")
    return component


def load_strict_suite_json(path: Path) -> object:
    """Read at most 2 MiB of strict UTF-8 JSON from ``path`` without mutation.

    Duplicate keys, nonfinite constants, malformed input, and filesystem errors
    raise :class:`VLMEvalSuiteStatusError`.
    """
    try:
        return load_json_bytes(path.read_bytes(), maximum_bytes=2_097_152)
    except (OSError, StrictJsonError) as error:
        raise VLMEvalSuiteStatusError(
            f"cannot read strict JSON artifact: {path.name}"
        ) from error


def _validate_native_completed(
    root: Path,
    record: SuiteDatasetStatus,
    native: dict[str, object],
) -> None:
    if not is_safe_relative_path(record.metrics_path):
        raise VLMEvalSuiteStatusError("completed dataset has no metrics provenance")
    metrics = require_shared_file(root / record.metrics_path, root, "dataset metrics")
    normalized_metrics_from_payload(load_strict_suite_json(metrics))
    dataset_root = (root / record.dataset_directory).resolve(strict=True)
    native_metrics = native.get("metrics_path")
    if not is_safe_relative_path(native_metrics):
        raise VLMEvalSuiteStatusError("completed status has no metrics path")
    if (dataset_root / native_metrics).resolve(strict=True) != metrics:
        raise VLMEvalSuiteStatusError("completed status metrics path differs")
    for path_field in (
        "demonstration_and_scored_row_partitions_path",
        "prediction_path",
        "score_path",
    ):
        relative = native.get(path_field)
        if not is_safe_relative_path(relative):
            raise VLMEvalSuiteStatusError(f"completed status lacks {path_field}")
        require_shared_file(dataset_root / relative, dataset_root, path_field)
    expected = native.get("expected_scored_indices")
    observed = native.get("observed_prediction_count")
    if type(expected) is not list or type(observed) is not int or observed != len(expected):
        raise VLMEvalSuiteStatusError("completed status has inconsistent prediction counts")
    if native.get("failure") is not None:
        raise VLMEvalSuiteStatusError("completed status contains a failure")


def _validate_dataset_status_payload(root: Path, record: SuiteDatasetStatus) -> None:
    if record.status == "not_started":
        if record.status_path is not None or record.metrics_path is not None:
            raise VLMEvalSuiteStatusError("not_started dataset has terminal artifacts")
        return
    if not is_safe_relative_path(record.status_path):
        raise VLMEvalSuiteStatusError("terminal dataset status reference is invalid")
    if record.status_path != f"{record.dataset_directory}/{STATUS_FILENAME}":
        raise VLMEvalSuiteStatusError("dataset status is outside its configured dataset directory")
    status = require_shared_file(root / record.status_path, root, "dataset status")
    payload = load_strict_suite_json(status)
    valid = (
        type(payload) is dict
        and payload.get("schema_version") == 3
        and type(payload.get("datasets")) is list
        and len(payload["datasets"]) == 1
        and type(payload["datasets"][0]) is dict
        and set(payload["datasets"][0]) == DATASET_STATUS_RECORD_FIELDS
    )
    if not valid:
        raise VLMEvalSuiteStatusError("dataset status payload has the wrong schema")
    native = payload["datasets"][0]
    if native.get("dataset") != record.dataset or native.get("status") != record.status:
        raise VLMEvalSuiteStatusError("dataset status identity differs from the suite ledger")
    if record.status == "completed":
        if record.metrics_path != f"{record.dataset_directory}/{METRICS_FILENAME}":
            raise VLMEvalSuiteStatusError("dataset metrics are outside the configured directory")
        _validate_native_completed(root, record, native)
    elif record.metrics_path is not None:
        raise VLMEvalSuiteStatusError("failed dataset must not claim completed metrics")


def validate_suite_dataset_statuses(
    suite: BenchmarkSuiteConfig,
    run_directory: Path,
    records: tuple[SuiteDatasetStatus, ...],
) -> tuple[SuiteDatasetStatus, ...]:
    """Validate ordered records and referenced artifacts without mutation.

    ``suite`` defines required membership and order; ``run_directory`` contains
    every terminal artifact. The returned tuple is the original immutable tuple.
    Invalid records, paths, metrics schemas, or artifact access raise
    :class:`VLMEvalSuiteStatusError`.
    """
    root = resolve_directory(run_directory)
    expected = tuple(
        (item.dataset, f"datasets/{suite_dataset_component(item.dataset)}")
        for item in suite.benchmarks
    )
    if type(records) is not tuple or len(records) != len(expected):
        raise VLMEvalSuiteStatusError("suite status count differs from the configured suite")
    for record, planned in zip(records, expected, strict=True):
        if type(record) is not SuiteDatasetStatus:
            raise VLMEvalSuiteStatusError("suite status contains the wrong record type")
        if (record.dataset, record.dataset_directory) != planned:
            raise VLMEvalSuiteStatusError("suite status membership or order differs")
        if record.status not in _STATUSES or not is_safe_relative_path(
            record.dataset_directory
        ):
            raise VLMEvalSuiteStatusError("suite dataset status is invalid")
        try:
            _validate_dataset_status_payload(root, record)
        except VLMEvalSuiteStatusError:
            raise
        except Exception as error:
            raise VLMEvalSuiteStatusError("suite dataset artifact is invalid") from error
    return records


def planned_suite_dataset_statuses(
    suite: BenchmarkSuiteConfig,
) -> tuple[SuiteDatasetStatus, ...]:
    """Return ordered ``not_started`` records for ``suite`` without mutation.

    ``suite`` must be an exact benchmark-suite record. Invalid suite inputs
    raise :class:`VLMEvalSuiteStatusError`.
    """
    if type(suite) is not BenchmarkSuiteConfig:
        raise VLMEvalSuiteStatusError("suite has the wrong record type")
    return tuple(
        SuiteDatasetStatus(
            benchmark.dataset,
            "not_started",
            f"datasets/{suite_dataset_component(benchmark.dataset)}",
            None,
            None,
        )
        for benchmark in suite.benchmarks
    )


def prepare_suite_dataset_directory(run_directory: Path, dataset: str) -> Path:
    """Create or validate a shared dataset directory below ``run_directory``.

    Only the ``datasets`` parent and named dataset directory may be created.
    Existing paths are not otherwise mutated. Invalid paths, links, or access
    modes raise :class:`VLMEvalSuiteStatusError`.
    """
    root = resolve_directory(run_directory)
    datasets_root = root / "datasets"
    try:
        if not datasets_root.exists():
            os.mkdir(datasets_root, 0o750)
            os.chmod(datasets_root, 0o750)
        if datasets_root.resolve(strict=True) != datasets_root:
            raise ValueError("suite datasets root must not be a symlink")
        datasets_mode = datasets_root.stat().st_mode
        path = datasets_root / suite_dataset_component(dataset)
        if not path.exists():
            os.mkdir(path, 0o750)
            os.chmod(path, 0o750)
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        if resolved != path:
            raise ValueError("suite dataset directory must not be a symlink")
        mode = resolved.stat().st_mode
    except (OSError, ValueError) as error:
        raise VLMEvalSuiteStatusError("cannot prepare suite dataset directory") from error
    required = stat.S_IRGRP | stat.S_IXGRP
    accessible = os.access(resolved, os.R_OK | os.W_OK | os.X_OK)
    parents_shared = stat.S_ISDIR(datasets_mode) and datasets_mode & required == required
    valid_directory = stat.S_ISDIR(mode) and mode & required == required and accessible
    if not parents_shared or not valid_directory:
        raise VLMEvalSuiteStatusError("suite dataset directory is not shared and writable")
    return resolved


def _single_replacement(
    suite: BenchmarkSuiteConfig, record: SuiteDatasetStatus
) -> tuple[SuiteDatasetStatus, ...]:
    return tuple(
        record if planned.dataset == record.dataset else planned
        for planned in planned_suite_dataset_statuses(suite)
    )


def suite_dataset_status_from_artifacts(
    suite: BenchmarkSuiteConfig,
    run_directory: Path,
    dataset: str,
    status: str,
    status_artifact: Path,
    metrics_artifact: Path | None = None,
) -> SuiteDatasetStatus:
    """Build a validated terminal record from shared result artifacts.

    The artifacts are read and never changed; their returned paths are recorded
    relative to ``run_directory``. Unknown dataset/status values, inaccessible
    artifacts, or invalid terminal payloads raise :class:`VLMEvalSuiteStatusError`.
    """
    root = resolve_directory(run_directory)
    planned = {item.dataset: item for item in planned_suite_dataset_statuses(suite)}
    if dataset not in planned or status not in {"completed", "failed"}:
        raise VLMEvalSuiteStatusError("terminal dataset identity is not in the suite")
    try:
        status_path = require_shared_file(status_artifact, root, "dataset status")
        metrics_path = (
            None
            if metrics_artifact is None
            else require_shared_file(metrics_artifact, root, "dataset metrics")
        )
    except ValueError as error:
        raise VLMEvalSuiteStatusError("terminal dataset artifact is invalid") from error
    record = SuiteDatasetStatus(
        dataset,
        status,
        planned[dataset].dataset_directory,
        status_path.relative_to(root).as_posix(),
        None if metrics_path is None else metrics_path.relative_to(root).as_posix(),
    )
    validate_suite_dataset_statuses(suite, root, _single_replacement(suite, record))
    return record


def write_vlmeval_suite_status(
    suite: BenchmarkSuiteConfig,
    run_directory: Path,
    records: tuple[SuiteDatasetStatus, ...],
) -> VLMEvalSuiteStatusArtifact:
    """Validate and atomically replace the complete suite ledger.

    ``records`` and the referenced artifacts are validated without mutation;
    only the canonical ledger file is replaced. Invalid inputs or write failures
    raise :class:`VLMEvalSuiteStatusError` or the underlying artifact error.
    """
    root = resolve_directory(run_directory)
    validated = validate_suite_dataset_statuses(suite, root, records)
    destination = root / SUITE_STATUS_FILENAME
    payload = {
        "schema_version": 2,
        "suite": suite.name,
        "datasets": [asdict(record) for record in validated],
    }
    write_json(destination, payload)
    return VLMEvalSuiteStatusArtifact(destination, validated)


def load_vlmeval_suite_status(
    suite: BenchmarkSuiteConfig,
    run_directory: Path,
) -> VLMEvalSuiteStatusArtifact:
    """Read and validate the canonical ledger for ``suite`` without mutation.

    The returned record has the configured suite order and validated completed
    metrics. Missing, private, malformed, or mismatched artifacts raise
    :class:`VLMEvalSuiteStatusError`.
    """
    root = resolve_directory(run_directory)
    try:
        artifact = require_shared_file(
            root / SUITE_STATUS_FILENAME, root, "suite status"
        )
    except ValueError as error:
        raise VLMEvalSuiteStatusError("suite status artifact is invalid") from error
    payload = load_strict_suite_json(artifact)
    valid = (
        type(payload) is dict
        and set(payload) == {
            "schema_version",
            "suite",
            "datasets",
        }
        and payload.get("schema_version") == 2
        and payload.get("suite") == suite.name
        and type(payload.get("datasets")) is list
    )
    if not valid:
        raise VLMEvalSuiteStatusError("suite status payload differs from the configured suite")
    try:
        records = tuple(SuiteDatasetStatus(**item) for item in payload["datasets"])
    except (TypeError, AttributeError) as error:
        raise VLMEvalSuiteStatusError("suite status records are malformed") from error
    validated = validate_suite_dataset_statuses(suite, root, records)
    return VLMEvalSuiteStatusArtifact(artifact, validated)


__all__ = (
    "SUITE_STATUS_FILENAME",
    "SuiteDatasetStatus",
    "VLMEvalSuiteStatusArtifact",
    "VLMEvalSuiteStatusError",
    "load_strict_suite_json",
    "load_vlmeval_suite_status",
    "planned_suite_dataset_statuses",
    "prepare_suite_dataset_directory",
    "suite_dataset_component",
    "suite_dataset_status_from_artifacts",
    "validate_suite_dataset_statuses",
    "write_vlmeval_suite_status",
)
