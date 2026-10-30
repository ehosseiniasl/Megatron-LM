"""Strict normalization for benchmark-native VLMEvalKit ``_acc.csv`` files."""

from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import shared_file_with_relative_path

if TYPE_CHECKING:
    from ..vlmeval_result_contract import NormalizedMetric


_MAXIMUM_BYTES = 1_048_576
_MAXIMUM_ROWS = 128
_MAXIMUM_COLUMNS = 256
_INTEGER = re.compile(r"[+-]?[0-9]+")
_NON_IDENTIFIER = re.compile(r"[^a-z0-9]+")


class VLMEvalAccuracyCSVError(ValueError):
    """Raised when a native accuracy CSV violates the bounded contract."""


def _load_rows(score: Path) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    try:
        raw = score.read_bytes()
    except OSError as error:
        raise VLMEvalAccuracyCSVError("score artifact cannot be read") from error
    if not raw or len(raw) > _MAXIMUM_BYTES or b"\x00" in raw:
        raise VLMEvalAccuracyCSVError("score artifact has an invalid byte size or content")
    try:
        text = raw.decode("utf-8")
        parsed = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise VLMEvalAccuracyCSVError("score artifact is not strict UTF-8 CSV") from error
    if len(parsed) < 2 or len(parsed) - 1 > _MAXIMUM_ROWS:
        raise VLMEvalAccuracyCSVError("score artifact has an invalid row count")
    header = tuple(parsed[0])
    if not header or len(header) > _MAXIMUM_COLUMNS:
        raise VLMEvalAccuracyCSVError("score artifact has an invalid column count")
    if any(not field or field != field.strip() for field in header):
        raise VLMEvalAccuracyCSVError("score header fields must be nonempty and stripped")
    if len(header) != len(set(header)):
        raise VLMEvalAccuracyCSVError("score header fields must be unique")
    rows = tuple(tuple(row) for row in parsed[1:])
    if any(len(row) != len(header) for row in rows):
        raise VLMEvalAccuracyCSVError("score rows do not match the header")
    return header, rows


def _identifier(value: str) -> str:
    identifier = _NON_IDENTIFIER.sub("_", value.lower()).strip("_")
    if not identifier:
        raise VLMEvalAccuracyCSVError("metric field has no usable identifier")
    if identifier[0].isdigit():
        identifier = f"metric_{identifier}"
    return identifier


def _number(value: str, unit: str, label: str) -> int | float:
    if not value or value != value.strip():
        raise VLMEvalAccuracyCSVError(f"{label} must be a nonempty stripped number")
    try:
        number: int | float = int(value) if _INTEGER.fullmatch(value) else float(value)
    except ValueError as error:
        raise VLMEvalAccuracyCSVError(f"{label} is not numeric") from error
    if type(number) is float and not math.isfinite(number):
        raise VLMEvalAccuracyCSVError(f"{label} must be finite")
    maximum = 1 if unit == "fraction" else 100
    if number < 0 or number > maximum:
        raise VLMEvalAccuracyCSVError(f"{label} is outside the {unit} range")
    return number


def read_vlmeval_accuracy_csv(
    score_artifact: Path,
    run_directory: Path,
    *,
    dataset: str,
    unit: str,
) -> tuple[NormalizedMetric, ...]:
    """Normalize a contained native accuracy CSV into ordered metric records.

    ``score_artifact`` must be group-readable beneath ``run_directory``.
    ``dataset`` labels every returned metric and ``unit`` is ``fraction`` or
    ``percent``. The source file is bounded and never modified. Invalid paths,
    CSV shapes, split labels, identifiers, numeric values, or ranges raise
    :class:`VLMEvalAccuracyCSVError`.
    """
    from ..vlmeval_result_contract import NormalizedMetric

    if type(dataset) is not str or not dataset or dataset != dataset.strip():
        raise VLMEvalAccuracyCSVError("dataset must be a nonempty stripped string")
    if unit not in {"fraction", "percent"}:
        raise VLMEvalAccuracyCSVError("accuracy CSV unit must be fraction or percent")
    try:
        score, source_file = shared_file_with_relative_path(
            score_artifact, run_directory, "score artifact"
        )
    except ValueError as error:
        raise VLMEvalAccuracyCSVError(str(error)) from error
    header, rows = _load_rows(score)
    split_column = header[0] == "split"
    if "split" in header[1:]:
        raise VLMEvalAccuracyCSVError("split may appear only as the first column")
    fields = header[1:] if split_column else header
    if not fields or (not split_column and len(rows) != 1):
        raise VLMEvalAccuracyCSVError("accuracy table has an unsupported row shape")
    split_values = tuple(row[0] for row in rows) if split_column else ()
    if split_column and (
        any(not value or value != value.strip() for value in split_values)
        or len(split_values) != len(set(split_values))
    ):
        raise VLMEvalAccuracyCSVError("split labels must be nonempty, stripped, and unique")
    multi_row = len(rows) > 1
    metrics: list[NormalizedMetric] = []
    for row in rows:
        split = row[0] if split_column else None
        values = row[1:] if split_column else row
        for field, value in zip(fields, values, strict=True):
            metric = _identifier(field)
            source_field = field
            if multi_row:
                metric = f"{_identifier(split or '')}_{metric}"
                source_field = f"{split}:{field}"
            metrics.append(
                NormalizedMetric(
                    dataset=dataset,
                    metric=metric,
                    value=_number(value, unit, source_field),
                    unit=unit,
                    source_file=source_file,
                    source_field=source_field,
                )
            )
    return tuple(metrics)
