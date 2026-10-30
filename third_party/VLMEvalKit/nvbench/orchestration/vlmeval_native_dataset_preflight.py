"""Model-free preflight for configured native VLMEvalKit datasets and prompts."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .configured_eval_run import BenchmarkConfig, BenchmarkSuiteConfig
from .fewshot_dataset_adaptation.demonstration_selection import select_fewshot_rows
from .fewshot_dataset_adaptation.fewshot_conversation_builder import (
    VLMEvalFewshotDatasetProxy,
)
from .utils import temporary_environment


class VLMEvalNativeDatasetContractError(ValueError):
    """Raised when a native dataset does not match its configured contract."""


class VLMEvalNativeDatasetPreflightError(RuntimeError):
    """Raised when any configured native dataset or prompt fails preflight."""


@dataclass(frozen=True)
class VLMEvalNativeDatasetPreflightRecord:
    """One read-only observed dataset/prompt preflight result.

    Fields report configured and actual class identities, selected-row counts,
    and prompt shape. The preflight constructs no prediction or score artifact;
    invalid native datasets or prompts raise
    :class:`VLMEvalNativeDatasetPreflightError` before a record is returned.
    """

    dataset: str
    configured_class: str
    actual_class: str
    source_row_count: int
    demonstration_count: int
    scored_row_count: int
    prompt_turn_count: int
    prompt_image_count: int


@dataclass(frozen=True)
class VLMEvalNativeDatasetPreflightResult:
    """Ordered immutable records returned by a successful native preflight.

    ``records`` follows the configured suite order and owns no mutable dataset
    data. Incomplete or invalid preflight work raises
    :class:`VLMEvalNativeDatasetPreflightError` instead of returning a result.
    """

    records: tuple[VLMEvalNativeDatasetPreflightRecord, ...]


DatasetFactory = Callable[[str], object]


def build_upstream_dataset(dataset_name: str) -> object:
    """Construct one named dataset through the native VLMEvalKit factory.

    The function mutates no nvbench state. Unknown datasets and unavailable
    upstream dependencies raise their native VLMEvalKit/import errors.
    """
    from vlmeval.dataset import build_dataset

    return build_dataset(dataset_name)


def validate_vlmeval_native_dataset(
    dataset: object,
    benchmark: BenchmarkConfig,
) -> None:
    """Validate one native dataset against ``benchmark`` without mutation.

    The dataset must implement the configured class lineage, exact dataset name,
    non-``None`` data, and callable prompt/evaluation seams. Contract failures
    raise :class:`VLMEvalNativeDatasetContractError`.
    """
    if dataset is None:
        raise VLMEvalNativeDatasetContractError("dataset factory returned no dataset")
    actual_class = type(dataset).__name__
    compatible_classes = {candidate.__name__ for candidate in type(dataset).__mro__}
    if benchmark.upstream_class not in compatible_classes:
        raise VLMEvalNativeDatasetContractError(
            f"dataset class mismatch: expected {benchmark.upstream_class}, got {actual_class}"
        )
    if getattr(dataset, "dataset_name", None) != benchmark.dataset:
        raise VLMEvalNativeDatasetContractError(
            f"dataset name mismatch: expected {benchmark.dataset}"
        )
    if getattr(dataset, "data", None) is None:
        raise VLMEvalNativeDatasetContractError("dataset has no data")
    for method_name in ("build_prompt", "evaluate"):
        if not callable(getattr(dataset, method_name, None)):
            raise VLMEvalNativeDatasetContractError(
                f"dataset has no callable {method_name} method"
            )


def _cache_root(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise VLMEvalNativeDatasetPreflightError("cache root must be an absolute Path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise VLMEvalNativeDatasetPreflightError("cache root is unavailable") from error
    if resolved != path or not resolved.is_dir() or not os.access(resolved, os.R_OK | os.X_OK):
        raise VLMEvalNativeDatasetPreflightError(
            "cache root must be a readable, traversable, nonsymlink directory"
        )
    return resolved


def _message_content(message: object, position: int) -> Sequence[object]:
    if not isinstance(message, Mapping):
        raise VLMEvalNativeDatasetPreflightError(
            f"prompt message {position} is not a mapping"
        )
    content = message.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
        raise VLMEvalNativeDatasetPreflightError(
            f"prompt message {position} has invalid content"
        )
    return content


def _image_count(content: Sequence[object], position: int) -> int:
    count = 0
    for item in content:
        if not isinstance(item, Mapping) or item.get("type") not in {"text", "image"}:
            raise VLMEvalNativeDatasetPreflightError(
                f"prompt message {position} has an invalid item"
            )
        count += int(item["type"] == "image")
    return count


def _validate_prompt(prompt: object, num_shot: int, include_demo_images: bool) -> tuple[int, int]:
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes, bytearray)):
        raise VLMEvalNativeDatasetPreflightError("few-shot prompt is not a message sequence")
    expected_turns = num_shot * 2 + 1
    if len(prompt) != expected_turns:
        raise VLMEvalNativeDatasetPreflightError("few-shot prompt turn count is incorrect")
    contents = tuple(_message_content(message, position) for position, message in enumerate(prompt))
    image_counts = tuple(
        _image_count(content, position) for position, content in enumerate(contents)
    )
    demonstration_images = image_counts[0 : num_shot * 2 : 2]
    if include_demo_images and any(count < 1 for count in demonstration_images):
        raise VLMEvalNativeDatasetPreflightError("an image-bearing demonstration has no image")
    if not include_demo_images and any(demonstration_images):
        raise VLMEvalNativeDatasetPreflightError("a text-only demonstration retained an image")
    if image_counts[-1] < 1:
        raise VLMEvalNativeDatasetPreflightError("the scored query has no image")
    return len(prompt), sum(image_counts)


def preflight_datasets(
    suite: BenchmarkSuiteConfig,
    cache_root: Path,
    *,
    dataset_factory: DatasetFactory | None = None,
) -> VLMEvalNativeDatasetPreflightResult:
    """Return model-free dataset and first-prompt records for ``suite``.

    ``suite`` supplies the ordered benchmarks and ``cache_root`` must be an
    absolute readable directory. The factory runs while ``LMUData`` is
    temporarily bound to that root; no request, prediction, score, or source
    dataset mutation occurs. Invalid inputs, native datasets, selections, or
    prompts raise :class:`VLMEvalNativeDatasetPreflightError`.
    """
    if type(suite) is not BenchmarkSuiteConfig:
        raise VLMEvalNativeDatasetPreflightError("suite has the wrong record type")
    root = _cache_root(cache_root)
    factory = build_upstream_dataset if dataset_factory is None else dataset_factory
    records: list[VLMEvalNativeDatasetPreflightRecord] = []
    with temporary_environment("LMUData", str(root)):
        for benchmark in suite.benchmarks:
            try:
                dataset = factory(benchmark.dataset)
                validate_vlmeval_native_dataset(dataset, benchmark)
                source_rows = len(dataset.data)  # type: ignore[attr-defined]
                selection = select_fewshot_rows(  # type: ignore[attr-defined]
                    dataset.data, benchmark
                )
                proxy = VLMEvalFewshotDatasetProxy(dataset, selection, benchmark)
                if len(proxy) == 0:
                    raise VLMEvalNativeDatasetPreflightError("selection has no scored row")
                prompt = proxy.build_prompt(0)
                turns, images = _validate_prompt(
                    prompt,
                    benchmark.num_shot,
                    benchmark.include_demonstration_images,
                )
            except Exception as error:
                if isinstance(error, VLMEvalNativeDatasetPreflightError):
                    raise
                raise VLMEvalNativeDatasetPreflightError(
                    f"native dataset preflight failed for {benchmark.dataset}"
                ) from error
            records.append(
                VLMEvalNativeDatasetPreflightRecord(
                    dataset=benchmark.dataset,
                    configured_class=benchmark.upstream_class,
                    actual_class=type(dataset).__name__,
                    source_row_count=source_rows,
                    demonstration_count=len(selection.demonstrations),
                    scored_row_count=len(selection.scored_positions),
                    prompt_turn_count=turns,
                    prompt_image_count=images,
                )
            )
    if len(records) != len(suite.benchmarks):
        raise VLMEvalNativeDatasetPreflightError("preflight record count is incomplete")
    return VLMEvalNativeDatasetPreflightResult(tuple(records))
