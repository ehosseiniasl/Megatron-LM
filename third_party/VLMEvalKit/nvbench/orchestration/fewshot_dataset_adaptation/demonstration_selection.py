"""Select, load, and serialize deterministic few-shot demonstration partitions."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..configured_eval_run import BenchmarkConfig, EvalConfigError
from ..utils import (
    VLMEvalRowIdentityError,
    VLMEvalRowIndex,
    require_shared_file,
    require_unique_vlmeval_row_indices,
    require_vlmeval_row_index,
    resolve_directory,
    write_json,
)
from .fewshot_conversation_builder import FewshotConversationError, format_demonstration_answer
from .predefined_demonstration_registry import (
    PredefinedDemonstrationError,
    load_predefined_demonstrations,
)


class DemonstrationSelectionError(EvalConfigError):
    """Raised when target or packaged rows cannot satisfy the selection contract."""


class VLMEvalDemonstrationScoredRowPartitionArtifactError(RuntimeError):
    """Raised when an already-validated row partition cannot be serialized."""


@dataclass(frozen=True)
class DemonstrationExample:
    """One isolated demonstration record and its stable, namespaced source ID.

    ``record`` is a deep copy owned by the selection result. Its nested values
    may be changed by a caller without mutating the upstream dataset.
    """

    source_index: str
    record: dict[str, object]


@dataclass(frozen=True)
class FewshotRowSelection:
    """Disjoint demonstration and scored-row partitions over upstream data.

    Positions refer to the original positional dataframe ordering. Scored
    indices preserve the canonical integer-or-string type stored by upstream.
    """

    demonstrations: tuple[DemonstrationExample, ...]
    demonstration_positions: tuple[int, ...]
    scored_positions: tuple[int, ...]
    scored_indices: tuple[VLMEvalRowIndex, ...]


@dataclass(frozen=True)
class DemonstrationAndScoredRowPartitionsArtifact:
    """Absolute path to one immutable serialized demonstration and scored-row partition."""

    artifact: Path


PARTITIONS_FILENAME = "demonstration_and_scored_row_partitions.json"
_REASONING_CUE = "Let's reason step by step. "


def _prepend_reasoning_cue(
    demonstrations: tuple[tuple[str, dict[str, object]], ...],
) -> tuple[tuple[str, dict[str, object]], ...]:
    """Prepend the opt-in cue to isolated assistant demonstration answers."""
    for _, record in demonstrations:
        answer = record.get("answer")
        if type(answer) is not str or not answer:
            raise DemonstrationSelectionError("reasoning demonstration must have a nonempty answer")
        record["answer"] = _REASONING_CUE + answer
    return demonstrations


def _read_row(data: object, position: int) -> dict[str, object]:
    try:
        row = data.iloc[position]  # type: ignore[attr-defined]
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise DemonstrationSelectionError(f"cannot read target row {position}") from error
    if isinstance(row, Mapping):
        mapping = row
    elif callable(getattr(row, "to_dict", None)):
        mapping = row.to_dict()
    else:
        raise DemonstrationSelectionError(f"target row {position} is not record-like")
    if not isinstance(mapping, Mapping) or not all(type(key) is str for key in mapping):
        raise DemonstrationSelectionError(f"target row {position} is not string-keyed")
    return dict(mapping)


def _target_index(record: Mapping[str, object], position: int) -> VLMEvalRowIndex:
    try:
        return require_vlmeval_row_index(record.get("index"), label=f"target row {position} index")
    except VLMEvalRowIdentityError as error:
        raise DemonstrationSelectionError(str(error)) from error


def _record_has_demonstration_answer(
    record: Mapping[str, object], benchmark: BenchmarkConfig
) -> bool:
    try:
        # Selection validates the bbox without opening images. Original image
        # dimensions are resolved from native prompts when building demo turns.
        answer_format = benchmark.demonstration_answer_format
        if answer_format == "screenspot_click_normalized":
            answer_format = "screenspot_click"
        format_demonstration_answer(record, answer_format)
    except FewshotConversationError:
        return False
    return True


def _bounded_selection_complete(
    benchmark: BenchmarkConfig,
    materialized_count: int,
    demonstration_count: int,
    excluded_count: int,
) -> bool:
    if demonstration_count < benchmark.num_shot:
        return False
    limit = benchmark.scored_limit
    return limit is not None and materialized_count - excluded_count >= limit


def select_fewshot_rows(data: object, benchmark: BenchmarkConfig) -> FewshotRowSelection:
    """Partition a pandas-like dataset into demonstrations and scored positions.

    ``data`` must be sized and support positional ``iloc`` row access.
    Demonstrations use the first eligible benchmark rows. The function never
    mutates ``data`` and deep-copies only returned demonstration records. It may
    stop after a bounded scored prefix is complete.
    """
    if type(benchmark) is not BenchmarkConfig:
        raise DemonstrationSelectionError("benchmark must be an exact BenchmarkConfig")
    limit = benchmark.scored_limit
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise DemonstrationSelectionError("scored limit must be a positive integer")
    try:
        configured_demo_indices = require_unique_vlmeval_row_indices(
            benchmark.demonstration_indices, label="demonstration indices"
        )
    except VLMEvalRowIdentityError as error:
        raise DemonstrationSelectionError(str(error)) from error
    if configured_demo_indices:
        if benchmark.demonstration_type != "benchmark":
            raise DemonstrationSelectionError(
                "explicit demonstration indices require benchmark demonstrations"
            )
        if len(configured_demo_indices) != benchmark.num_shot:
            raise DemonstrationSelectionError(
                "explicit demonstration index count must equal num_shot"
            )
    try:
        row_count = len(data)  # type: ignore[arg-type]
    except TypeError as error:
        raise DemonstrationSelectionError("target data must be sized") from error

    if benchmark.demonstration_type != "benchmark":
        try:
            predefined = load_predefined_demonstrations(
                benchmark.dataset, benchmark.demonstration_type, benchmark.num_shot
            )
            if benchmark.prepend_reasoning_cue:
                predefined = _prepend_reasoning_cue(predefined)
        except PredefinedDemonstrationError as error:
            raise DemonstrationSelectionError(str(error)) from error
        materialized_count = row_count if limit is None else min(row_count, limit)
        scored_indices: list[VLMEvalRowIndex] = []
        seen_indices: set[VLMEvalRowIndex] = set()
        seen_text_indices: set[str] = set()
        for position in range(materialized_count):
            index = _target_index(_read_row(data, position), position)
            text_index = str(index)
            if index in seen_indices or text_index in seen_text_indices:
                raise DemonstrationSelectionError("materialized target row indices must be unique")
            scored_indices.append(index)
            seen_indices.add(index)
            seen_text_indices.add(text_index)
        return FewshotRowSelection(
            demonstrations=tuple(
                DemonstrationExample(source_index, record) for source_index, record in predefined
            ),
            demonstration_positions=(),
            scored_positions=tuple(range(materialized_count)),
            scored_indices=tuple(scored_indices),
        )

    demonstration_positions: list[int] = []
    demonstrations: list[DemonstrationExample] = []
    explicit_demo_indices = tuple(str(index) for index in configured_demo_indices)
    explicit_demo_index_set = set(explicit_demo_indices)

    indices: list[VLMEvalRowIndex] = []
    seen_indices: set[VLMEvalRowIndex] = set()
    seen_text_indices: set[str] = set()
    for position in range(row_count):
        record = _read_row(data, position)
        index = _target_index(record, position)
        text_index = str(index)
        if index in seen_indices or text_index in seen_text_indices:
            raise DemonstrationSelectionError("materialized target row indices must be unique")
        indices.append(index)
        seen_indices.add(index)
        seen_text_indices.add(text_index)

        should_consider_demonstration = (
            text_index in explicit_demo_index_set
            if explicit_demo_indices
            else len(demonstrations) < benchmark.num_shot
        )
        if should_consider_demonstration:
            if _record_has_demonstration_answer(record, benchmark):
                demonstration = copy.deepcopy(record)
                demonstrations.append(DemonstrationExample(text_index, demonstration))
                demonstration_positions.append(position)

        if _bounded_selection_complete(
            benchmark, len(indices), len(demonstrations), len(demonstration_positions)
        ):
            break

    if explicit_demo_indices:
        selected_by_index = {
            demonstration.source_index: (position, demonstration)
            for position, demonstration in zip(demonstration_positions, demonstrations, strict=True)
        }
        missing = tuple(
            source_index
            for source_index in explicit_demo_indices
            if source_index not in selected_by_index
        )
        if missing:
            raise DemonstrationSelectionError(
                f"explicit benchmark demonstrations are missing or ineligible: {missing}"
            )
        demonstration_positions = [
            selected_by_index[source_index][0] for source_index in explicit_demo_indices
        ]
        demonstrations = [
            selected_by_index[source_index][1] for source_index in explicit_demo_indices
        ]
    elif len(demonstrations) < benchmark.num_shot:
        raise DemonstrationSelectionError("benchmark has insufficient eligible demonstrations")

    removed = set(demonstration_positions)
    scored_positions = tuple(
        position for position in range(len(indices)) if position not in removed
    )
    if limit is not None:
        scored_positions = scored_positions[:limit]
    return FewshotRowSelection(
        demonstrations=tuple(demonstrations),
        demonstration_positions=tuple(demonstration_positions),
        scored_positions=scored_positions,
        scored_indices=tuple(indices[position] for position in scored_positions),
    )


def demonstration_and_scored_row_partitions_payload(
    selection: FewshotRowSelection, benchmark: BenchmarkConfig
) -> dict[str, object]:
    """Serialize the selection already validated by selection and proxy construction."""
    if type(selection) is not FewshotRowSelection:
        raise VLMEvalDemonstrationScoredRowPartitionArtifactError(
            "selection must be an exact FewshotRowSelection"
        )
    if type(benchmark) is not BenchmarkConfig:
        raise VLMEvalDemonstrationScoredRowPartitionArtifactError(
            "benchmark must be an exact BenchmarkConfig"
        )
    demonstration_positions: tuple[int | None, ...]
    if benchmark.demonstration_source == "benchmark":
        demonstration_positions = selection.demonstration_positions
    else:
        demonstration_positions = (None,) * len(selection.demonstrations)
    return {
        "schema_version": 2,
        "dataset": benchmark.dataset,
        "demonstration_type": benchmark.demonstration_type,
        "demonstration_source": benchmark.demonstration_source,
        "include_demonstration_images": benchmark.include_demonstration_images,
        "prepend_reasoning_cue": benchmark.prepend_reasoning_cue,
        "demonstration_count": len(selection.demonstrations),
        "scored_row_count": len(selection.scored_indices),
        "demonstrations": [
            {"source_index": item.source_index, "target_position": target_position}
            for item, target_position in zip(
                selection.demonstrations, demonstration_positions, strict=True
            )
        ],
        "scored_rows": [
            {"target_index": target_index, "target_position": target_position}
            for target_index, target_position in zip(
                selection.scored_indices, selection.scored_positions, strict=True
            )
        ],
    }


def write_demonstration_and_scored_row_partitions(
    selection: FewshotRowSelection, benchmark: BenchmarkConfig, run_directory: Path
) -> DemonstrationAndScoredRowPartitionsArtifact:
    """Write a deterministic partition record and return its contained absolute path.

    ``selection`` is already validated by selection and proxy construction and is not mutated.
    ``run_directory`` must be an existing writable root; serialization creates or replaces the one
    owned JSON artifact. Invalid inputs or I/O failures raise
    :class:`VLMEvalDemonstrationScoredRowPartitionArtifactError`.
    """
    payload = demonstration_and_scored_row_partitions_payload(selection, benchmark)
    try:
        root = resolve_directory(run_directory)
        written = write_json(root / PARTITIONS_FILENAME, payload)
        artifact = require_shared_file(written, root, "row-partition artifact")
    except Exception as error:
        raise VLMEvalDemonstrationScoredRowPartitionArtifactError(
            "cannot write demonstration and scored-row partitions"
        ) from error
    return DemonstrationAndScoredRowPartitionsArtifact(artifact)
