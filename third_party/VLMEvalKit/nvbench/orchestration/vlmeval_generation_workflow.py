"""Compose native VLMEvalKit dataset construction and generation inference."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .configured_eval_run import BenchmarkConfig
from .fewshot_dataset_adaptation.demonstration_selection import (
    PARTITIONS_FILENAME,
    DemonstrationAndScoredRowPartitionsArtifact,
    FewshotRowSelection,
    demonstration_and_scored_row_partitions_payload,
    select_fewshot_rows,
    write_demonstration_and_scored_row_partitions,
)
from .fewshot_dataset_adaptation.fewshot_conversation_builder import (
    VLMEvalFewshotDatasetProxy,
)
from .vlmeval_native_dataset_preflight import (
    VLMEvalNativeDatasetContractError,
    build_upstream_dataset,
    validate_vlmeval_native_dataset,
)
from .vlmeval_result_contract import (
    NormalizedMetric,
    finalize_vlmeval_generated_answer_result,
)
from .utils import load_json_bytes, require_shared_file
from .vlmeval_prediction_handoff import (
    PREDICTIONS_READY_FILENAME,
    VLMEvalPredictionHandoffError,
    VLMEvalPredictionResult,
    load_prediction_handoff,
    publish_prediction_handoff,
)


class _InferenceRunner(Protocol):
    def __call__(
        self,
        *,
        model: object,
        work_dir: str,
        model_name: str,
        dataset: object,
        api_nproc: int,
        retry_failed: bool,
    ) -> object: ...


class _PredictionPathResolver(Protocol):
    def __call__(
        self,
        work_dir: str,
        model_name: str,
        dataset_name: str,
        *,
        use_env_format: bool,
    ) -> str: ...


def _run_upstream_inference_job(
    *,
    model: object,
    work_dir: str,
    model_name: str,
    dataset: object,
    api_nproc: int,
    retry_failed: bool,
) -> object:
    """Run the native inference job with composed client and dataset objects."""
    from vlmeval.inference import infer_data_job

    return infer_data_job(
        model=model,
        work_dir=work_dir,
        model_name=model_name,
        dataset=dataset,
        api_nproc=api_nproc,
        retry_failed=retry_failed,
    )


def _resolve_upstream_prediction_path(
    work_dir: str,
    model_name: str,
    dataset_name: str,
    *,
    use_env_format: bool,
) -> str:
    """Resolve the native environment-selected prediction artifact path."""
    from vlmeval.smp import get_pred_file_path

    if use_env_format is not True:
        raise ValueError("environment-aware output format is required")
    return get_pred_file_path(
        work_dir,
        model_name,
        dataset_name,
        use_env_format=True,
    )


class VLMEvalGenerationWorkflowError(RuntimeError):
    """Raised when generation orchestration cannot satisfy its local contract."""

    def __init__(
        self,
        message: str,
        *,
        row_partitions: DemonstrationAndScoredRowPartitionsArtifact | None = None,
    ) -> None:
        if (
            row_partitions is not None
            and type(row_partitions) is not DemonstrationAndScoredRowPartitionsArtifact
        ):
            raise TypeError("row_partitions must be an exact artifact record or None")
        super().__init__(message)
        self.row_partitions = row_partitions


@dataclass(frozen=True)
class VLMEvalGenerationResult:
    """Validated selection plus native and normalized artifacts for one dataset.

    Every path is under the caller-owned work directory. The workflow returns
    this immutable record only after inference, native scoring, completeness
    validation, and terminal artifact writes all succeed.
    """

    selection: FewshotRowSelection
    row_partitions_artifact: Path
    prediction_artifact: Path
    score_artifact: Path
    metrics_artifact: Path
    status_artifact: Path
    normalized_metrics: tuple[NormalizedMetric, ...]


@dataclass(frozen=True)
class _PreparedBenchmark:
    dataset_proxy: object
    selection: FewshotRowSelection


_SAFE_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _validate_evaluation_judge(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise VLMEvalGenerationWorkflowError(
            "evaluation judge must be a nonempty stripped string"
        )
    return value


def _writable_work_directory(work_dir: str | os.PathLike[str]) -> Path:
    try:
        path = Path(work_dir).resolve(strict=True)
    except Exception as error:
        raise VLMEvalGenerationWorkflowError(
            f"work directory does not exist: {work_dir!r}"
        ) from error
    mode = path.stat().st_mode
    writable_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    if (
        not path.is_dir()
        or not mode & writable_bits
        or not os.access(path, os.W_OK | os.X_OK)
    ):
        raise VLMEvalGenerationWorkflowError(
            f"work directory is not a writable directory: {path}"
        )
    return path


def _resolved_artifact_path(
    resolver: _PredictionPathResolver,
    work_dir: Path,
    model_name: str,
    dataset_name: str,
    row_partitions: DemonstrationAndScoredRowPartitionsArtifact,
) -> Path:
    try:
        configured = resolver(
            str(work_dir),
            model_name,
            dataset_name,
            use_env_format=True,
        )
        if (
            type(configured) is not str
            or not configured
            or not Path(configured).is_absolute()
        ):
            raise ValueError(
                "prediction artifact path must be a nonempty absolute string"
            )
        artifact = Path(configured).resolve(strict=False)
        artifact.relative_to(work_dir)
    except Exception as error:
        raise VLMEvalGenerationWorkflowError(
            "upstream prediction artifact path is invalid or outside the work directory",
            row_partitions=row_partitions,
        ) from error
    return artifact


def _require_readable_artifact(
    artifact: Path,
    work_dir: Path,
    row_partitions: DemonstrationAndScoredRowPartitionsArtifact,
) -> Path:
    try:
        resolved = artifact.resolve(strict=True)
        resolved.relative_to(work_dir)
    except (OSError, ValueError) as error:
        raise VLMEvalGenerationWorkflowError(
            "prediction artifact is missing or outside the work directory",
            row_partitions=row_partitions,
        ) from error
    mode = resolved.stat().st_mode
    if (
        not stat.S_ISREG(mode)
        or not mode & stat.S_IRGRP
        or not os.access(resolved, os.R_OK)
    ):
        raise VLMEvalGenerationWorkflowError(
            f"prediction artifact is not a readable regular file: {resolved}",
            row_partitions=row_partitions,
        )
    return resolved


def _prepare_benchmark(
    benchmark: BenchmarkConfig,
    dataset_factory: Callable[[str], object] | None,
) -> _PreparedBenchmark:
    factory = build_upstream_dataset if dataset_factory is None else dataset_factory
    try:
        dataset = factory(benchmark.dataset)
    except Exception as error:
        raise VLMEvalGenerationWorkflowError("dataset factory failed") from error
    try:
        validate_vlmeval_native_dataset(dataset, benchmark)
    except VLMEvalNativeDatasetContractError as error:
        raise VLMEvalGenerationWorkflowError(str(error)) from error
    selection = select_fewshot_rows(
        dataset.data,  # type: ignore[attr-defined]
        benchmark,
    )
    return _PreparedBenchmark(
        VLMEvalFewshotDatasetProxy(dataset, selection, benchmark), selection
    )


def _publish_prediction_result(
    benchmark: BenchmarkConfig,
    prepared: _PreparedBenchmark,
    directory: Path,
    row_partitions: DemonstrationAndScoredRowPartitionsArtifact,
    prediction: Path,
    prediction_loader: Callable[[str], object] | None,
) -> VLMEvalPredictionResult:
    try:
        return publish_prediction_handoff(
            benchmark,
            prepared.selection,
            row_partitions.artifact,
            prediction,
            directory,
            prediction_loader=prediction_loader,
        )
    except VLMEvalPredictionHandoffError as error:
        raise VLMEvalGenerationWorkflowError(
            str(error),
            row_partitions=row_partitions,
        ) from error


def _require_matching_row_partitions(
    benchmark: BenchmarkConfig,
    prepared: _PreparedBenchmark,
    directory: Path,
) -> DemonstrationAndScoredRowPartitionsArtifact:
    artifact = require_shared_file(
        directory / PARTITIONS_FILENAME, directory, "row partitions"
    )
    try:
        payload = load_json_bytes(artifact.read_bytes(), maximum_bytes=4_194_304)
    except Exception as error:
        raise VLMEvalGenerationWorkflowError("row partitions are invalid") from error
    selection = prepared.selection
    expected = demonstration_and_scored_row_partitions_payload(selection, benchmark)
    if payload != expected:
        raise VLMEvalGenerationWorkflowError(
            "row partitions do not match the benchmark selection"
        )
    return DemonstrationAndScoredRowPartitionsArtifact(artifact)


def _run_vlmeval_predictions(
    client: object,
    benchmark: BenchmarkConfig,
    work_dir: str | os.PathLike[str],
    model_name: str,
    *,
    api_nproc: int,
    retry_failed: bool = True,
    dataset_factory: Callable[[str], object] | None = None,
    inference_runner: _InferenceRunner | None = None,
    prediction_path_resolver: _PredictionPathResolver | None = None,
    prediction_loader: Callable[[str], object] | None = None,
) -> tuple[VLMEvalPredictionResult, _PreparedBenchmark]:
    """Generate and validate one benchmark's raw prediction artifact."""
    directory = _writable_work_directory(work_dir)
    if type(model_name) is not str or not _SAFE_MODEL_NAME.fullmatch(model_name):
        raise VLMEvalGenerationWorkflowError("model name is empty or unsafe")
    if type(api_nproc) is not int or api_nproc <= 0:
        raise VLMEvalGenerationWorkflowError("api_nproc must be a positive integer")
    if type(retry_failed) is not bool:
        raise VLMEvalGenerationWorkflowError("retry_failed must be a boolean")
    if getattr(client, "is_api", None) is not True:
        raise VLMEvalGenerationWorkflowError("client must declare is_api = True")
    if not callable(getattr(client, "generate", None)):
        raise VLMEvalGenerationWorkflowError("client has no callable generate method")

    runner = (
        _run_upstream_inference_job if inference_runner is None else inference_runner
    )
    resolver = (
        _resolve_upstream_prediction_path
        if prediction_path_resolver is None
        else prediction_path_resolver
    )
    prepared = _prepare_benchmark(benchmark, dataset_factory)
    selection = prepared.selection
    proxy = prepared.dataset_proxy
    try:
        row_partitions = write_demonstration_and_scored_row_partitions(
            selection, benchmark, directory
        )
    except Exception as error:
        raise VLMEvalGenerationWorkflowError(
            "row-partition artifact creation failed"
        ) from error
    artifact = _resolved_artifact_path(
        resolver,
        directory,
        model_name,
        benchmark.dataset,
        row_partitions,
    )
    try:
        runner(
            model=client,
            work_dir=str(directory),
            model_name=model_name,
            dataset=proxy,
            api_nproc=api_nproc,
            retry_failed=retry_failed,
        )
    except Exception as error:
        raise VLMEvalGenerationWorkflowError(
            "upstream inference job failed", row_partitions=row_partitions
        ) from error

    prediction = _require_readable_artifact(artifact, directory, row_partitions)
    prediction_result = _publish_prediction_result(
        benchmark,
        prepared,
        directory,
        row_partitions,
        prediction,
        prediction_loader,
    )
    return prediction_result, prepared


def adopt_existing_vlmeval_predictions(
    benchmark: BenchmarkConfig,
    work_dir: str | os.PathLike[str],
    model_name: str,
    *,
    dataset_factory: Callable[[str], object] | None = None,
    prediction_path_resolver: _PredictionPathResolver | None = None,
    prediction_loader: Callable[[str], object] | None = None,
) -> VLMEvalPredictionResult:
    """Validate a pre-split run's complete predictions and publish readiness."""
    directory = _writable_work_directory(work_dir)
    if type(model_name) is not str or not _SAFE_MODEL_NAME.fullmatch(model_name):
        raise VLMEvalGenerationWorkflowError("model name is empty or unsafe")
    prepared = _prepare_benchmark(benchmark, dataset_factory)
    row_partitions = _require_matching_row_partitions(
        benchmark, prepared, directory
    )
    resolver = (
        _resolve_upstream_prediction_path
        if prediction_path_resolver is None
        else prediction_path_resolver
    )
    artifact = _resolved_artifact_path(
        resolver,
        directory,
        model_name,
        benchmark.dataset,
        row_partitions,
    )
    prediction = _require_readable_artifact(artifact, directory, row_partitions)
    return _publish_prediction_result(
        benchmark,
        prepared,
        directory,
        row_partitions,
        prediction,
        prediction_loader,
    )


def run_vlmeval_predictions(
    client: object,
    benchmark: BenchmarkConfig,
    work_dir: str | os.PathLike[str],
    model_name: str,
    *,
    api_nproc: int,
    retry_failed: bool = True,
    dataset_factory: Callable[[str], object] | None = None,
    inference_runner: _InferenceRunner | None = None,
    prediction_path_resolver: _PredictionPathResolver | None = None,
    prediction_loader: Callable[[str], object] | None = None,
) -> VLMEvalPredictionResult:
    """Generate complete predictions and publish the CPU-scoring handoff."""
    result, _ = _run_vlmeval_predictions(
        client,
        benchmark,
        work_dir,
        model_name,
        api_nproc=api_nproc,
        retry_failed=retry_failed,
        dataset_factory=dataset_factory,
        inference_runner=inference_runner,
        prediction_path_resolver=prediction_path_resolver,
        prediction_loader=prediction_loader,
    )
    return result


def score_vlmeval_predictions(
    benchmark: BenchmarkConfig,
    work_dir: str | os.PathLike[str],
    *,
    evaluation_judge: str,
    dataset_factory: Callable[[str], object] | None = None,
    prediction_loader: Callable[[str], object] | None = None,
) -> VLMEvalGenerationResult:
    """Score a validated prediction handoff without starting a model server."""
    evaluation_judge = _validate_evaluation_judge(evaluation_judge)
    directory = _writable_work_directory(work_dir)
    prepared = _prepare_benchmark(benchmark, dataset_factory)
    try:
        prediction_result = load_prediction_handoff(
            benchmark,
            prepared.selection,
            directory,
            prediction_loader=prediction_loader,
        )
    except VLMEvalPredictionHandoffError as error:
        raise VLMEvalGenerationWorkflowError(str(error)) from error
    finalized = finalize_vlmeval_generated_answer_result(
        prepared.dataset_proxy,
        benchmark,
        prepared.selection,
        prediction_result.prediction_artifact,
        directory,
        row_partitions=DemonstrationAndScoredRowPartitionsArtifact(
            prediction_result.row_partitions_artifact
        ),
        evaluation_judge=evaluation_judge,
        prediction_loader=prediction_loader,
    )
    return VLMEvalGenerationResult(
        selection=prepared.selection,
        row_partitions_artifact=finalized.row_partitions_artifact,
        prediction_artifact=finalized.prediction_artifact,
        score_artifact=finalized.score_artifact,
        metrics_artifact=finalized.metrics_artifact,
        status_artifact=finalized.status_artifact,
        normalized_metrics=finalized.normalized_metrics,
    )


def run_vlmeval_generation(
    client: object,
    benchmark: BenchmarkConfig,
    work_dir: str | os.PathLike[str],
    model_name: str,
    *,
    api_nproc: int,
    evaluation_judge: str = "exact_matching",
    retry_failed: bool = True,
    dataset_factory: Callable[[str], object] | None = None,
    inference_runner: _InferenceRunner | None = None,
    prediction_path_resolver: _PredictionPathResolver | None = None,
    prediction_loader: Callable[[str], object] | None = None,
) -> VLMEvalGenerationResult:
    """Run prediction and scoring together for the ordinary one-stage path."""
    evaluation_judge = _validate_evaluation_judge(evaluation_judge)
    prediction_result, prepared = _run_vlmeval_predictions(
        client,
        benchmark,
        work_dir,
        model_name,
        api_nproc=api_nproc,
        retry_failed=retry_failed,
        dataset_factory=dataset_factory,
        inference_runner=inference_runner,
        prediction_path_resolver=prediction_path_resolver,
        prediction_loader=prediction_loader,
    )
    finalized = finalize_vlmeval_generated_answer_result(
        prepared.dataset_proxy,
        benchmark,
        prepared.selection,
        prediction_result.prediction_artifact,
        Path(work_dir).resolve(strict=True),
        row_partitions=DemonstrationAndScoredRowPartitionsArtifact(
            prediction_result.row_partitions_artifact
        ),
        evaluation_judge=evaluation_judge,
        prediction_loader=prediction_loader,
    )
    return VLMEvalGenerationResult(
        selection=prepared.selection,
        row_partitions_artifact=finalized.row_partitions_artifact,
        prediction_artifact=finalized.prediction_artifact,
        score_artifact=finalized.score_artifact,
        metrics_artifact=finalized.metrics_artifact,
        status_artifact=finalized.status_artifact,
        normalized_metrics=finalized.normalized_metrics,
    )
