"""Strict reader for standard ScreenSpot point-grounding score JSON."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import StrictJsonError, load_json_bytes, shared_file_with_relative_path

if TYPE_CHECKING:
    from ..vlmeval_result_contract import NormalizedMetric


_FIELDS = (
    ("text:cnt", "text_count", "count"),
    ("icon:cnt", "icon_count", "count"),
    ("Overall_Accuracy", "overall_accuracy", "percent"),
    ("Format_Err_Rate", "format_error_rate", "percent"),
    ("Text_Accuracy", "text_accuracy", "percent"),
    ("Icon_Accuracy", "icon_accuracy", "percent"),
)
_MAXIMUM_BYTES = 65_536


class ScreenSpotScoreJSONError(ValueError):
    """Raised when a ScreenSpot score JSON violates the pinned schema."""


def _payload(score: Path) -> dict[str, object]:
    try:
        raw = score.read_bytes()
    except OSError as error:
        raise ScreenSpotScoreJSONError("score artifact cannot be read") from error
    if not raw or len(raw) > _MAXIMUM_BYTES:
        raise ScreenSpotScoreJSONError("score artifact has an invalid byte size")
    try:
        payload = load_json_bytes(raw, maximum_bytes=_MAXIMUM_BYTES)
    except StrictJsonError as error:
        raise ScreenSpotScoreJSONError("score artifact is not strict UTF-8 JSON") from error
    if type(payload) is not dict or set(payload) != {field for field, _, _ in _FIELDS}:
        raise ScreenSpotScoreJSONError("score fields do not match the pinned schema")
    return payload


def _values(payload: dict[str, object]) -> tuple[int | float, ...]:
    values: list[int | float] = []
    for field, _, unit in _FIELDS:
        value = payload[field]
        if unit == "count":
            if type(value) is not int or value < 0:
                raise ScreenSpotScoreJSONError(f"{field} must be a nonnegative integer")
        elif type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ScreenSpotScoreJSONError(f"{field} must be a percentage")
        values.append(value)

    text_count, icon_count = values[:2]
    if text_count + icon_count <= 0:
        raise ScreenSpotScoreJSONError("score artifact contains no evaluated samples")
    overall, text_accuracy, icon_accuracy = values[2], values[4], values[5]
    weighted = (text_count * text_accuracy + icon_count * icon_accuracy) / (text_count + icon_count)
    if not math.isclose(overall, weighted, rel_tol=0, abs_tol=1e-9):
        raise ScreenSpotScoreJSONError("overall accuracy does not match subgroup scores")
    return tuple(values)


def read_screenspot_score_json(
    score_artifact: Path, run_directory: Path, *, dataset: str
) -> tuple[NormalizedMetric, ...]:
    """Normalize standard ScreenSpot counts and point-grounding percentages."""
    from ..vlmeval_result_contract import NormalizedMetric

    if type(dataset) is not str or not dataset.startswith("ScreenSpot_"):
        raise ScreenSpotScoreJSONError("dataset must be a standard ScreenSpot subset")
    try:
        score, source_file = shared_file_with_relative_path(
            score_artifact, run_directory, "score artifact"
        )
    except ValueError as error:
        raise ScreenSpotScoreJSONError(str(error)) from error
    values = _values(_payload(score))
    return tuple(
        NormalizedMetric(
            dataset=dataset,
            metric=metric,
            value=value,
            unit=unit,
            source_file=source_file,
            source_field=field,
        )
        for (field, metric, unit), value in zip(_FIELDS, values, strict=True)
    )
