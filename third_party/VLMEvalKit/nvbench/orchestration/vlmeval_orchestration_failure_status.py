"""Truthful dataset status for failures outside native result finalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .configured_eval_run import BenchmarkConfig
from .fewshot_dataset_adaptation.demonstration_selection import (
    DemonstrationAndScoredRowPartitionsArtifact,
)
from .utils import (
    load_json_bytes,
    relative_path,
    require_shared_file,
    resolve_directory,
    sanitized_error_text,
    write_json,
)
from .vlmeval_generation_workflow import VLMEvalGenerationResult
from .vlmeval_result_contract import DATASET_STATUS_RECORD_FIELDS, STATUS_FILENAME


_STAGE = re.compile(r"[a-z][a-z0-9_]*")


class VLMEvalOrchestrationFailureStatusError(RuntimeError):
    """Raised when an orchestration failure status cannot be recorded safely."""


@dataclass(frozen=True)
class VLMEvalFailureStatusReference:
    """Path to a validated or newly written failed dataset status."""

    status_artifact: Path


def _failure_record(
    error: Exception, stage: str, root: Path
) -> dict[str, str | None]:
    if not isinstance(error, Exception):
        raise VLMEvalOrchestrationFailureStatusError("failure must be an Exception")
    if type(stage) is not str or _STAGE.fullmatch(stage) is None:
        raise VLMEvalOrchestrationFailureStatusError("failure stage is invalid")
    cause = error.__cause__
    if cause is not None and not isinstance(cause, Exception):
        cause = None
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": sanitized_error_text(error, root),
        "cause_type": None if cause is None else type(cause).__name__,
        "cause_message": None if cause is None else sanitized_error_text(cause, root),
    }


def _row_partition_fields(
    value: DemonstrationAndScoredRowPartitionsArtifact | None,
    root: Path,
) -> dict[str, str | None]:
    if value is None:
        return {
            "demonstration_and_scored_row_partitions_path": None,
        }
    if type(value) is not DemonstrationAndScoredRowPartitionsArtifact:
        raise VLMEvalOrchestrationFailureStatusError(
            "row partitions have the wrong record type"
        )
    artifact = require_shared_file(
        value.artifact, root, "demonstration and scored row partitions"
    )
    return {
        "demonstration_and_scored_row_partitions_path": relative_path(artifact, root),
    }


def _completed_artifacts(
    result: VLMEvalGenerationResult,
    root: Path,
) -> tuple[dict[str, object], tuple[object, ...]]:
    if type(result) is not VLMEvalGenerationResult:
        raise VLMEvalOrchestrationFailureStatusError("completed result has the wrong type")
    paths = {
        "prediction": require_shared_file(
            result.prediction_artifact, root, "prediction artifact"
        ),
        "score": require_shared_file(result.score_artifact, root, "score artifact"),
        "metrics": require_shared_file(
            result.metrics_artifact, root, "metrics artifact"
        ),
    }
    row_partitions = DemonstrationAndScoredRowPartitionsArtifact(
        result.row_partitions_artifact
    )
    record = {
        "observed_prediction_count": len(result.selection.scored_indices),
        **_row_partition_fields(row_partitions, root),
        "prediction_path": relative_path(paths["prediction"], root),
        "score_path": relative_path(paths["score"], root),
        "metrics_path": relative_path(paths["metrics"], root),
    }
    return record, result.selection.scored_indices


def write_vlmeval_orchestration_failure_status(
    benchmark: BenchmarkConfig,
    run_directory: Path,
    error: Exception,
    *,
    stage: str,
    row_partitions: DemonstrationAndScoredRowPartitionsArtifact | None = None,
    completed_result: VLMEvalGenerationResult | None = None,
) -> VLMEvalFailureStatusReference:
    """Write one failed status while preserving available completed artifacts.

    ``benchmark`` and ``run_directory`` identify the single dataset attempt;
    ``error`` and ``stage`` describe its first failure. Callers may supply
    either a partition artifact or a completed result, never both. The function
    atomically replaces the owned dataset-status JSON and returns its path.
    Inputs and raw artifacts are not modified. Invalid records, unsafe paths,
    or write failures raise :class:`VLMEvalOrchestrationFailureStatusError`.
    """
    if type(benchmark) is not BenchmarkConfig:
        raise VLMEvalOrchestrationFailureStatusError("benchmark has the wrong record type")
    try:
        root = resolve_directory(run_directory)
        artifact_record: dict[str, object] = {
            "observed_prediction_count": None,
            **_row_partition_fields(row_partitions, root),
            "prediction_path": None,
            "score_path": None,
            "metrics_path": None,
        }
        expected_indices: tuple[object, ...] = ()
        if completed_result is not None:
            if row_partitions is not None:
                raise VLMEvalOrchestrationFailureStatusError(
                    "row partitions must not accompany a completed result"
                )
            artifact_record, expected_indices = _completed_artifacts(completed_result, root)
        record = {
            "dataset": benchmark.dataset,
            "status": "failed",
            "expected_scored_indices": list(expected_indices),
            **artifact_record,
            "failure": _failure_record(error, stage, root),
        }
        destination = root / STATUS_FILENAME
        write_json(destination, {"schema_version": 3, "datasets": [record]})
        return VLMEvalFailureStatusReference(destination)
    except VLMEvalOrchestrationFailureStatusError:
        raise
    except Exception as status_error:
        raise VLMEvalOrchestrationFailureStatusError(
            f"cannot write orchestration failure status: {status_error}"
        ) from status_error


def reference_existing_vlmeval_failure_status(
    benchmark: BenchmarkConfig,
    run_directory: Path,
) -> VLMEvalFailureStatusReference:
    """Return a validated native-finalization failure-status reference.

    ``benchmark`` selects the expected dataset and ``run_directory`` contains
    the existing status plus its row-partition artifact. No file is modified.
    Missing, malformed, mismatched, or unsafe artifacts raise
    :class:`VLMEvalOrchestrationFailureStatusError`.
    """
    try:
        root = resolve_directory(run_directory)
        status = require_shared_file(root / STATUS_FILENAME, root, "dataset status")
        payload = load_json_bytes(status.read_bytes(), maximum_bytes=2_097_152)
        if type(payload) is not dict:
            raise ValueError("existing dataset status is not an object")
        datasets = payload["datasets"]
        valid = (
            type(payload.get("schema_version")) is int
            and payload.get("schema_version") == 3
            and type(datasets) is list
            and len(datasets) == 1
            and type(datasets[0]) is dict
            and set(datasets[0]) == DATASET_STATUS_RECORD_FIELDS
            and datasets[0].get("dataset") == benchmark.dataset
            and datasets[0].get("status") == "failed"
            and type(datasets[0].get("expected_scored_indices")) is list
            and type(datasets[0].get("failure")) is dict
        )
        if not valid:
            raise ValueError("existing dataset status is not the expected failed record")
        record = datasets[0]
        partition_path = record["demonstration_and_scored_row_partitions_path"]
        if type(partition_path) is not str:
            raise ValueError("existing failed status has no row-partition provenance")
        fields = _row_partition_fields(
            DemonstrationAndScoredRowPartitionsArtifact(root / partition_path),
            root,
        )
        if fields["demonstration_and_scored_row_partitions_path"] != partition_path:
            raise ValueError("existing row-partition path is not normalized")
        return VLMEvalFailureStatusReference(status)
    except Exception as error:
        raise VLMEvalOrchestrationFailureStatusError(
            "cannot retain native failed dataset status"
        ) from error
