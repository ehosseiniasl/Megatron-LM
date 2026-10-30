"""Strict reader for the pinned upstream OCRBench ``_score.json`` schema."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import StrictJsonError, load_json_bytes, shared_file_with_relative_path

if TYPE_CHECKING:
    from ..vlmeval_result_contract import NormalizedMetric

_FIELDS = (
    ("Text Recognition", "text_recognition", 300, "count"),
    ("Scene Text-centric VQA", "scene_text_centric_vqa", 200, "count"),
    ("Doc-oriented VQA", "doc_oriented_vqa", 200, "count"),
    ("Key Information Extraction", "key_information_extraction", 200, "count"),
    (
        "Handwritten Mathematical Expression Recognition",
        "handwritten_mathematical_expression_recognition",
        100,
        "count",
    ),
    ("Final Score", "final_score", 1000, "count"),
    ("Final Score Norm", "final_score_norm", 100, "percent"),
)
_MAXIMUM_BYTES = 65_536


class OCRBenchScoreJSONError(ValueError):
    """Raised when an OCRBench score JSON file violates its pinned schema."""


def _load_exact_json(score: Path) -> dict[str, object]:
    try:
        raw = score.read_bytes()
    except OSError as error:
        raise OCRBenchScoreJSONError("score artifact cannot be read") from error
    if not raw or len(raw) > _MAXIMUM_BYTES:
        raise OCRBenchScoreJSONError("score artifact has an invalid byte size")
    try:
        payload = load_json_bytes(raw, maximum_bytes=_MAXIMUM_BYTES)
    except StrictJsonError as error:
        raise OCRBenchScoreJSONError("score artifact is not strict UTF-8 JSON") from error
    if type(payload) is not dict:
        raise OCRBenchScoreJSONError("OCRBench score JSON must contain one object")
    expected = {field for field, _, _, _ in _FIELDS}
    if set(payload) != expected:
        raise OCRBenchScoreJSONError("OCRBench score fields do not match the pinned schema")
    return payload


def _validated_values(payload: dict[str, object]) -> tuple[int | float, ...]:
    values: list[int | float] = []
    for position, (field, _, bound, _) in enumerate(_FIELDS):
        value = payload[field]
        if position < 6:
            if type(value) is not int:
                raise OCRBenchScoreJSONError(f"{field} must be an integer count")
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise OCRBenchScoreJSONError(f"{field} must be a finite number")
        if value < 0 or value > bound:
            raise OCRBenchScoreJSONError(f"{field} is outside its supported bounds")
        values.append(value)
    category_sum = sum(values[:5])
    if values[5] != category_sum:
        raise OCRBenchScoreJSONError("Final Score does not equal the category sum")
    if values[6] != values[5] / 10:
        raise OCRBenchScoreJSONError("Final Score Norm does not equal Final Score / 10")
    return tuple(values)


def read_ocrbench_score_json(
    score_artifact: Path, run_directory: Path
) -> tuple[NormalizedMetric, ...]:
    """Return seven source-ordered metrics from a contained OCRBench score file.

    Both paths must be absolute and the artifact must be group-readable beneath
    the run directory. The file is bounded and read without mutation. Schema,
    consistency, path, or syntax failures raise :class:`OCRBenchScoreJSONError`.
    """
    from ..vlmeval_result_contract import NormalizedMetric

    try:
        score, source_file = shared_file_with_relative_path(
            score_artifact, run_directory, "score artifact"
        )
    except ValueError as error:
        raise OCRBenchScoreJSONError(str(error)) from error
    values = _validated_values(_load_exact_json(score))
    return tuple(
        NormalizedMetric(
            dataset="OCRBench",
            metric=metric,
            value=value,
            unit=unit,
            source_file=source_file,
            source_field=field,
        )
        for (field, metric, _, unit), value in zip(_FIELDS, values)
    )
