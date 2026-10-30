"""Deterministic run manifests, terminal dataset records, and safe resumption."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .configured_eval_run import EvalConfigError, ResolvedEvalRun
from .utils import (
    JsonArtifactError,
    StrictJsonError,
    is_safe_relative_path,
    load_json_bytes,
    relative_path,
    require_shared_file,
    write_json,
)


MANIFEST_FILENAME = "megatron_dynamic_server_vlm_eval_manifest.json"
_TERMINAL_STATUSES = frozenset({"completed", "failed"})
_RESUMABLE_STATUSES = frozenset({"prepared", "partial", "failed"})
_ALL_STATUSES = _RESUMABLE_STATUSES | {"completed"}
_STABLE_FIELDS = (
    "schema_version",
    "configured_eval_run",
    "resolved_paths",
    "resolved_cli_overrides",
)


@dataclass(frozen=True)
class DatasetTerminalStatus:
    """Immutable dataset terminal state and manifest-relative status artifact path."""

    dataset: str
    status: str
    status_path: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvalConfigError(message)


def validate_dataset_terminal_statuses(
    run_status: str,
    records: tuple[DatasetTerminalStatus, ...],
) -> tuple[DatasetTerminalStatus, ...]:
    """Validate and dataset-sort terminal records without mutating ``records``.

    ``run_status`` and every terminal record must agree on completion or
    failure. Invalid types, paths, duplicates, and inconsistent states raise
    :class:`EvalConfigError`; the returned tuple contains the same records.
    """
    _require(
        type(run_status) is str and run_status in _ALL_STATUSES,
        "invalid run status for terminal datasets",
    )
    _require(type(records) is tuple, "dataset terminal statuses must be a tuple")
    for record in records:
        _require(type(record) is DatasetTerminalStatus, "invalid dataset terminal status record")
        valid_dataset = (
            type(record.dataset) is str
            and bool(record.dataset)
            and record.dataset == record.dataset.strip()
        )
        _require(valid_dataset, "terminal dataset name must be nonempty and stripped")
        _require(
            record.status in _TERMINAL_STATUSES,
            "terminal dataset status must be completed or failed",
        )
        _require(
            is_safe_relative_path(record.status_path),
            "terminal dataset status path must be safe and relative",
        )
    datasets = [record.dataset for record in records]
    paths = [record.status_path for record in records]
    _require(
        len(datasets) == len(set(datasets)) and len(paths) == len(set(paths)),
        "dataset terminal statuses contain duplicates",
    )
    _require(not (run_status == "prepared" and records), "prepared run has statuses")
    all_completed = bool(records) and all(record.status == "completed" for record in records)
    any_failed = any(record.status == "failed" for record in records)
    _require(run_status != "completed" or all_completed, "invalid completed statuses")
    _require(run_status != "failed" or any_failed, "invalid failed statuses")
    _require(run_status != "partial" or not all_completed, "invalid partial statuses")
    return tuple(sorted(records, key=lambda record: record.dataset))


def decode_dataset_terminal_statuses(value: object) -> tuple[DatasetTerminalStatus, ...]:
    """Decode manifest terminal records without mutating ``value``.

    A JSON list of exact ``DatasetTerminalStatus`` fields returns an immutable
    tuple. Malformed values raise :class:`EvalConfigError`.
    """
    if not isinstance(value, list):
        raise EvalConfigError("manifest dataset terminal statuses must be a list")
    try:
        return tuple(DatasetTerminalStatus(**record) for record in value)
    except (TypeError, AttributeError) as error:
        raise EvalConfigError("malformed dataset terminal status record") from error


def named_dataset_terminal_status_for_artifact(
    resolved: ResolvedEvalRun,
    dataset: str,
    status: str,
    artifact: Path,
) -> DatasetTerminalStatus:
    """Build one validated terminal reference for a contained existing artifact.

    Inputs are not mutated. The returned path is normalized relative to the
    resolved run directory; invalid resolved records, identities, or artifact
    paths raise :class:`EvalConfigError`.
    """
    _require(type(resolved) is ResolvedEvalRun, "resolved run has the wrong record type")
    _require(type(dataset) is str and bool(dataset.strip()), "dataset name is invalid")
    _require(isinstance(artifact, Path), "terminal status artifact must be a Path")
    try:
        relative = relative_path(artifact, Path(resolved.paths.run_directory))
    except ValueError as error:
        raise EvalConfigError(
            "terminal status artifact is unavailable or outside the run"
        ) from error
    assert relative is not None
    return validate_dataset_terminal_statuses(
        status,
        (DatasetTerminalStatus(dataset, status, relative),),
    )[0]


def dataset_terminal_status_for_artifact(
    resolved: ResolvedEvalRun,
    status: str,
    artifact: Path,
) -> DatasetTerminalStatus:
    """Build one validated terminal reference for ``resolved``'s single benchmark.

    Inputs are not mutated. A suite-only run, an unavailable artifact, or an
    invalid terminal state raises :class:`EvalConfigError`.
    """
    if resolved.configured.benchmark is None:
        raise EvalConfigError("single-dataset terminal status requires a benchmark")
    return named_dataset_terminal_status_for_artifact(
        resolved,
        resolved.configured.benchmark.dataset,
        status,
        artifact,
    )


def stable_manifest_payload(resolved: ResolvedEvalRun) -> dict[str, Any]:
    """Return fresh stable resolved input fields without execution metadata.

    ``resolved`` is not mutated. The payload omits inactive configuration arms
    and reader-defined unit details; malformed input record access propagates.
    """
    configured_payload = asdict(resolved.configured)
    benchmark_payload = configured_payload.get("benchmark")
    if isinstance(benchmark_payload, dict) and benchmark_payload.get("prompt_adapter") is None:
        benchmark_payload.pop("prompt_adapter", None)
    suite_payload = configured_payload.get("benchmark_suite")
    if isinstance(suite_payload, dict):
        for entry in suite_payload.get("benchmarks", []):
            if isinstance(entry, dict) and entry.get("prompt_adapter") is None:
                entry.pop("prompt_adapter", None)
    if resolved.configured.benchmark_array_selection is None:
        configured_payload.pop("benchmark_array_selection")
    if resolved.configured.benchmark_suite is None:
        configured_payload.pop("benchmark_suite")
        if (
            isinstance(benchmark_payload, dict)
            and benchmark_payload.get("result_metric_unit") == "reader_defined"
        ):
            benchmark_payload.pop("result_metric_unit")
    return {
        "schema_version": 2,
        "configured_eval_run": configured_payload,
        "resolved_paths": asdict(resolved.paths),
        "resolved_cli_overrides": [
            {"name": name, "value": value} for name, value in resolved.resolved_cli_overrides
        ],
    }


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    payload = copy.deepcopy(payload)
    del payload["configured_eval_run"]["output"]["resume"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_config_fingerprint(resolved: ResolvedEvalRun) -> str:
    """Hash stable resolved inputs excluding per-invocation resume authorization.

    ``resolved`` is not mutated. Serialization failures propagate from the
    configuration payload construction.
    """
    return _payload_fingerprint(stable_manifest_payload(resolved))


def build_resolved_run_manifest(
    resolved: ResolvedEvalRun,
    *,
    run_status: str = "prepared",
    dataset_terminal_statuses: tuple[DatasetTerminalStatus, ...] = (),
    generated_at: str | None = None,
    slurm_job_id: str | None = None,
) -> dict[str, Any]:
    """Build a fresh validated manifest without mutating inputs or the filesystem.

    Terminal records are validated and dataset-sorted. Unsupported states raise
    :class:`EvalConfigError`; the returned dictionary combines stable identity
    with the supplied volatile execution metadata.
    """
    if run_status not in _ALL_STATUSES:
        raise EvalConfigError(f"unsupported run manifest status: {run_status}")
    terminals = validate_dataset_terminal_statuses(run_status, dataset_terminal_statuses)
    manifest = stable_manifest_payload(resolved)
    manifest["config_fingerprint"] = canonical_config_fingerprint(resolved)
    manifest["run_status"] = run_status
    manifest["dataset_terminal_statuses"] = [asdict(record) for record in terminals]
    manifest["generated_at"] = generated_at or datetime.now(timezone.utc).isoformat()
    if slurm_job_id is not None:
        manifest["slurm_job_id"] = str(slurm_job_id)
    return manifest


def _read_existing_manifest(path: Path) -> dict[str, Any]:
    try:
        artifact = require_shared_file(path, path.parent, "existing run manifest")
        value = load_json_bytes(artifact.read_bytes(), maximum_bytes=2_097_152)
    except StrictJsonError as error:
        raise EvalConfigError(f"cannot read existing run manifest: {path}") from error
    except ValueError as error:
        if "group-readable" in str(error):
            raise EvalConfigError(f"existing run manifest is not group-readable: {path}") from error
        raise EvalConfigError(f"existing run has no readable {MANIFEST_FILENAME}") from error
    except OSError as error:
        raise EvalConfigError(f"cannot read existing run manifest: {path}") from error
    if type(value) is not dict:
        raise EvalConfigError("existing run manifest must be a JSON object")
    return value


def _require_accessible_run_directory(path: Path) -> None:
    mode = path.stat().st_mode
    if (
        not os.access(path, os.R_OK | os.X_OK)
        or not mode & stat.S_IRGRP
        or not mode & stat.S_IXGRP
    ):
        raise EvalConfigError(f"run directory is not group-readable and traversable: {path}")


def _require_matching_fingerprint(resolved: ResolvedEvalRun, existing: dict[str, Any]) -> None:
    expected_fingerprint = canonical_config_fingerprint(resolved)
    try:
        existing_payload = {field: existing[field] for field in _STABLE_FIELDS}
        recorded_fingerprint = existing["config_fingerprint"]
        actual_fingerprint = _payload_fingerprint(existing_payload)
    except (KeyError, TypeError, ValueError) as error:
        raise EvalConfigError(
            "existing run manifest is missing stable fingerprint inputs"
        ) from error
    if recorded_fingerprint != actual_fingerprint:
        raise EvalConfigError("existing run manifest fingerprint does not match its payload")
    if recorded_fingerprint != expected_fingerprint:
        raise EvalConfigError("existing run manifest has a different config fingerprint")


def validate_prepared_run_manifest(resolved: ResolvedEvalRun, slurm_job_id: str) -> Path:
    """Return the matching prepared manifest without mutating filesystem state.

    ``slurm_job_id`` must match the prepared server manifest. Missing, private,
    malformed, fingerprint-mismatched, or non-prepared manifests raise
    :class:`EvalConfigError`.
    """
    if type(slurm_job_id) is not str or not slurm_job_id.isdecimal():
        raise EvalConfigError("Slurm job ID must be a decimal string")
    run_directory = Path(resolved.paths.run_directory)
    if not run_directory.is_dir():
        raise EvalConfigError("prepared run directory is unavailable")
    _require_accessible_run_directory(run_directory)
    manifest_path = run_directory / MANIFEST_FILENAME
    existing = _read_existing_manifest(manifest_path)
    _require_matching_fingerprint(resolved, existing)
    if existing.get("run_status") != "prepared":
        raise EvalConfigError("server job step did not leave a prepared manifest")
    if existing.get("slurm_job_id") != slurm_job_id:
        raise EvalConfigError("prepared manifest has a different Slurm job ID")
    terminal = decode_dataset_terminal_statuses(existing.get("dataset_terminal_statuses"))
    validate_dataset_terminal_statuses("prepared", terminal)
    return manifest_path


def validate_output_destination(resolved: ResolvedEvalRun) -> None:
    """Allow only a matching resumable output run without mutating filesystem state.

    A missing destination is accepted. Existing non-directories, inaccessible
    runs, mismatched fingerprints, completed runs, or invalid terminal state
    raise :class:`EvalConfigError`.
    """
    run_directory = Path(resolved.paths.run_directory)
    if not run_directory.exists():
        return
    if not run_directory.is_dir():
        raise EvalConfigError(f"run destination exists and is not a directory: {run_directory}")
    _require_accessible_run_directory(run_directory)
    if not resolved.configured.output.resume:
        raise EvalConfigError(
            f"run destination already exists and resume is false: {run_directory}"
        )
    existing = _read_existing_manifest(run_directory / MANIFEST_FILENAME)
    _require_matching_fingerprint(resolved, existing)
    status = existing.get("run_status")
    if type(status) is not str or status not in _ALL_STATUSES:
        raise EvalConfigError(f"existing run has a non-resumable status: {status!r}")
    terminals = decode_dataset_terminal_statuses(existing.get("dataset_terminal_statuses"))
    validate_dataset_terminal_statuses(status, terminals)
    if status == "completed":
        raise EvalConfigError("completed evaluation runs cannot be resumed")
    if status not in _RESUMABLE_STATUSES:
        raise EvalConfigError(f"existing run has a non-resumable status: {status!r}")


def write_resolved_run_manifest(
    resolved: ResolvedEvalRun,
    *,
    run_status: str = "prepared",
    dataset_terminal_statuses: tuple[DatasetTerminalStatus, ...] = (),
    generated_at: str | None = None,
    slurm_job_id: str | None = None,
) -> Path:
    """Durably replace this run's mode-0640 manifest and return its path.

    The existing run directory is required and inputs are not mutated. Manifest
    validation or shared JSON writer failures are translated to
    :class:`EvalConfigError`.
    """
    run_directory = Path(resolved.paths.run_directory)
    if not run_directory.is_dir():
        raise EvalConfigError(f"run directory must exist before manifest creation: {run_directory}")
    _require_accessible_run_directory(run_directory)
    manifest = build_resolved_run_manifest(
        resolved,
        run_status=run_status,
        dataset_terminal_statuses=dataset_terminal_statuses,
        generated_at=generated_at,
        slurm_job_id=slurm_job_id,
    )
    try:
        return write_json(run_directory / MANIFEST_FILENAME, manifest)
    except JsonArtifactError as error:
        raise EvalConfigError(f"cannot write run manifest: {run_directory}") from error
