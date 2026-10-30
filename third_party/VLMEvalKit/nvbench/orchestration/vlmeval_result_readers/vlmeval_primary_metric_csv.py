"""Strict primary-metric readers for labeled VLMEvalKit CSV tables."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import shared_file_with_relative_path

if TYPE_CHECKING:
    from ..vlmeval_result_contract import NormalizedMetric


class VLMEvalPrimaryMetricCSVError(ValueError):
    """Raised when a labeled native score table violates its profile."""


@dataclass(frozen=True)
class _Profile:
    label_column: str
    label_value: str
    metrics: tuple[tuple[str, str], ...]


_PROFILES = {
    "chartqapro_primary_csv": _Profile("0", "Overall", (("1", "overall_accuracy"),)),
    "mathvista_primary_csv": _Profile(
        "Task&Skill", "Overall", (("acc", "overall_accuracy"),)
    ),
    "worldvqa_primary_csv": _Profile(
        "Metric",
        "Overall",
        (
            ("Accuracy", "accuracy"),
            ("CGA", "conditional_generation_accuracy"),
            ("AttemptRate", "attempt_rate"),
            ("F-score", "f_score"),
        ),
    ),
}
_MAXIMUM_BYTES = 1_048_576
_MAXIMUM_ROWS = 256
_MAXIMUM_COLUMNS = 256


def _table(path: Path) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise VLMEvalPrimaryMetricCSVError("score artifact cannot be read") from error
    if not raw or len(raw) > _MAXIMUM_BYTES or b"\x00" in raw:
        raise VLMEvalPrimaryMetricCSVError("score artifact has invalid content")
    try:
        rows = list(
            csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
        )
    except (UnicodeDecodeError, csv.Error) as error:
        raise VLMEvalPrimaryMetricCSVError(
            "score artifact is not strict UTF-8 CSV"
        ) from error
    if len(rows) < 2 or len(rows) - 1 > _MAXIMUM_ROWS:
        raise VLMEvalPrimaryMetricCSVError("score artifact has an invalid row count")
    header = tuple(rows[0])
    body = tuple(tuple(row) for row in rows[1:])
    if (
        not header
        or len(header) > _MAXIMUM_COLUMNS
        or any(not field or field != field.strip() for field in header)
        or len(header) != len(set(header))
        or any(len(row) != len(header) for row in body)
    ):
        raise VLMEvalPrimaryMetricCSVError("score artifact has an invalid table shape")
    return header, body


def _number(value: str, unit: str, label: str) -> float:
    if not value or value != value.strip():
        raise VLMEvalPrimaryMetricCSVError(
            f"{label} must be a nonempty stripped number"
        )
    try:
        number = float(value)
    except ValueError as error:
        raise VLMEvalPrimaryMetricCSVError(f"{label} is not numeric") from error
    maximum = 1 if unit == "fraction" else 100
    if not math.isfinite(number) or number < 0 or number > maximum:
        raise VLMEvalPrimaryMetricCSVError(f"{label} is outside the {unit} range")
    return number


def read_vlmeval_primary_metric_csv(
    score_artifact: Path,
    run_directory: Path,
    *,
    dataset: str,
    unit: str,
    profile: str,
) -> tuple[NormalizedMetric, ...]:
    """Normalize the configured primary row from one native labeled CSV table."""
    from ..vlmeval_result_contract import NormalizedMetric

    if type(dataset) is not str or not dataset or dataset != dataset.strip():
        raise VLMEvalPrimaryMetricCSVError("dataset must be a nonempty stripped string")
    if unit not in {"fraction", "percent"}:
        raise VLMEvalPrimaryMetricCSVError("metric unit must be fraction or percent")
    if profile not in _PROFILES:
        raise VLMEvalPrimaryMetricCSVError("primary-metric CSV profile is unsupported")
    try:
        score, source_file = shared_file_with_relative_path(
            score_artifact, run_directory, "score artifact"
        )
    except ValueError as error:
        raise VLMEvalPrimaryMetricCSVError(str(error)) from error
    header, rows = _table(score)
    configured = _PROFILES[profile]
    required = (configured.label_column,) + tuple(
        field for field, _ in configured.metrics
    )
    if any(field not in header for field in required):
        raise VLMEvalPrimaryMetricCSVError("score artifact is missing required columns")
    label_position = header.index(configured.label_column)
    selected = tuple(
        row for row in rows if row[label_position] == configured.label_value
    )
    if len(selected) != 1:
        raise VLMEvalPrimaryMetricCSVError(
            "score artifact must contain one primary row"
        )
    row = selected[0]
    return tuple(
        NormalizedMetric(
            dataset=dataset,
            metric=metric,
            value=_number(row[header.index(field)], unit, field),
            unit=unit,
            source_file=source_file,
            source_field=f"{configured.label_value}:{field}",
        )
        for field, metric in configured.metrics
    )


__all__ = (
    "VLMEvalPrimaryMetricCSVError",
    "read_vlmeval_primary_metric_csv",
)
