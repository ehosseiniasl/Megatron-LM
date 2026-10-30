"""Durable artifact contract between GPU prediction and CPU scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .configured_eval_run import BenchmarkConfig
from .fewshot_dataset_adaptation.demonstration_selection import FewshotRowSelection
from .utils import (
    is_safe_relative_path,
    load_json_bytes,
    relative_path,
    require_shared_file,
    write_json,
)
from .vlmeval_result_contract import validate_prediction_completeness


PREDICTIONS_READY_FILENAME = "vlmeval_predictions_ready.v1.json"
_FIELDS = frozenset(
    {
        "schema_version",
        "dataset",
        "expected_scored_indices",
        "observed_prediction_count",
        "row_partitions_path",
        "prediction_path",
    }
)


class VLMEvalPredictionHandoffError(RuntimeError):
    """Raised when a prediction handoff is incomplete or inconsistent."""


@dataclass(frozen=True)
class VLMEvalPredictionResult:
    """Validated raw predictions ready for scoring in another process."""

    selection: FewshotRowSelection
    row_partitions_artifact: Path
    prediction_artifact: Path
    readiness_artifact: Path


def load_upstream_prediction(path: str) -> object:
    """Load a native prediction artifact without interpreting answer strings."""
    from vlmeval.smp import file

    if Path(path).suffix.lower() == ".xlsx":
        return file.pd.read_excel(path, keep_default_na=False)
    return file.load(path)


def _relative_artifact(root: Path, value: object, label: str) -> Path:
    if type(value) is not str or not is_safe_relative_path(value):
        raise VLMEvalPredictionHandoffError(f"{label} path is unsafe")
    return require_shared_file(root / value, root, label)


def _validated_predictions(
    prediction: Path,
    selection: FewshotRowSelection,
    loader: Callable[[str], object] | None,
) -> int:
    load = load_upstream_prediction if loader is None else loader
    try:
        completeness = validate_prediction_completeness(
            load(str(prediction)), selection.scored_indices
        )
    except Exception as error:
        raise VLMEvalPredictionHandoffError(
            "prediction artifact is incomplete or mismatched"
        ) from error
    return completeness.observed_row_count


def publish_prediction_handoff(
    benchmark: BenchmarkConfig,
    selection: FewshotRowSelection,
    row_partitions: Path,
    prediction: Path,
    run_directory: Path,
    *,
    prediction_loader: Callable[[str], object] | None = None,
) -> VLMEvalPredictionResult:
    """Validate raw predictions and atomically publish their readiness record."""
    root = run_directory.resolve(strict=True)
    partitions = require_shared_file(row_partitions, root, "row partitions")
    prediction = require_shared_file(prediction, root, "prediction artifact")
    count = _validated_predictions(prediction, selection, prediction_loader)
    readiness = root / PREDICTIONS_READY_FILENAME
    result = VLMEvalPredictionResult(selection, partitions, prediction, readiness)
    write_json(
        readiness,
        {
            "schema_version": 1,
            "dataset": benchmark.dataset,
            "expected_scored_indices": list(selection.scored_indices),
            "observed_prediction_count": count,
            "row_partitions_path": relative_path(partitions, root),
            "prediction_path": relative_path(prediction, root),
        },
    )
    return result


def load_prediction_handoff(
    benchmark: BenchmarkConfig,
    selection: FewshotRowSelection,
    run_directory: Path,
    *,
    prediction_loader: Callable[[str], object] | None = None,
) -> VLMEvalPredictionResult:
    """Load and revalidate a prediction handoff before CPU scoring."""
    root = run_directory.resolve(strict=True)
    readiness = require_shared_file(
        root / PREDICTIONS_READY_FILENAME, root, "prediction readiness artifact"
    )
    try:
        payload = load_json_bytes(readiness.read_bytes(), maximum_bytes=4_194_304)
    except Exception as error:
        raise VLMEvalPredictionHandoffError(
            "prediction readiness artifact is invalid"
        ) from error
    if (
        type(payload) is not dict
        or set(payload) != _FIELDS
        or payload.get("schema_version") != 1
        or payload.get("dataset") != benchmark.dataset
        or payload.get("expected_scored_indices") != list(selection.scored_indices)
    ):
        raise VLMEvalPredictionHandoffError(
            "prediction readiness artifact does not match the benchmark selection"
        )
    partitions = _relative_artifact(root, payload["row_partitions_path"], "row partitions")
    prediction = _relative_artifact(root, payload["prediction_path"], "prediction artifact")
    count = _validated_predictions(prediction, selection, prediction_loader)
    if payload.get("observed_prediction_count") != count:
        raise VLMEvalPredictionHandoffError(
            "prediction readiness count does not match the prediction artifact"
        )
    return VLMEvalPredictionResult(selection, partitions, prediction, readiness)


__all__ = (
    "PREDICTIONS_READY_FILENAME",
    "VLMEvalPredictionHandoffError",
    "VLMEvalPredictionResult",
    "load_prediction_handoff",
    "load_upstream_prediction",
    "publish_prediction_handoff",
)
