"""Sequential suite execution, aggregation, and terminal manifest recording."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .configured_eval_run import BenchmarkConfig, BenchmarkSuiteConfig, ResolvedEvalRun
from .resolved_run_manifest import (
    DatasetTerminalStatus,
    named_dataset_terminal_status_for_artifact,
    write_resolved_run_manifest,
)
from .utils import load_json_bytes, require_shared_file, sanitized_error_text, write_json
from .vlmeval_generation_workflow import (
    VLMEvalGenerationResult,
    VLMEvalGenerationWorkflowError,
    run_vlmeval_generation,
)
from .vlmeval_orchestration_failure_status import (
    reference_existing_vlmeval_failure_status,
    write_vlmeval_orchestration_failure_status,
)
from .vlmeval_suite_status_artifact import (
    SUITE_STATUS_FILENAME,
    SuiteDatasetStatus,
    VLMEvalSuiteStatusArtifact,
    load_vlmeval_suite_status,
    planned_suite_dataset_statuses,
    prepare_suite_dataset_directory,
    suite_dataset_status_from_artifacts,
    write_vlmeval_suite_status,
)
from .vlmeval_result_contract import (
    GeneratedAnswerResultFinalizationError,
    NormalizedMetric,
    normalized_metrics_from_payload,
    normalized_metrics_payload,
)


class VLMEvalGenerationSuiteWorkflowError(RuntimeError):
    """Raised after persisting the first terminal dataset failure."""

    def __init__(
        self, message: str, status: VLMEvalSuiteStatusArtifact | None
    ) -> None:
        super().__init__(message)
        self.suite_status = status


@dataclass(frozen=True)
class VLMEvalGenerationSuiteResult:
    """Immutable completed suite ledger path and ordered dataset statuses.

    The workflow constructs this only after every configured dataset completed.
    It does not transfer ownership or mutate the referenced artifacts; failures
    during execution instead raise :class:`VLMEvalGenerationSuiteWorkflowError`.
    """

    suite_status_artifact: Path
    dataset_statuses: tuple[SuiteDatasetStatus, ...]


GenerationRunner = Callable[..., VLMEvalGenerationResult]

SUITE_METRICS_FILENAME = "vlmeval_suite_metrics.v1.json"
SUITE_FAILURE_FILENAME = "vlmeval_suite_orchestration_failure.json"


class VLMEvalSuiteResultRecordingError(RuntimeError):
    """Raised when suite terminal artifacts cannot be trusted or written."""


def _replace_status(
    records: tuple[SuiteDatasetStatus, ...],
    replacement: SuiteDatasetStatus,
) -> tuple[SuiteDatasetStatus, ...]:
    return tuple(
        replacement if record.dataset == replacement.dataset else record
        for record in records
    )


def _failed_dataset_status(
    suite: BenchmarkSuiteConfig,
    benchmark: BenchmarkConfig,
    directory: Path,
    root: Path,
    error: Exception,
) -> SuiteDatasetStatus:
    if isinstance(error, GeneratedAnswerResultFinalizationError):
        reference = reference_existing_vlmeval_failure_status(benchmark, directory)
    else:
        row_partitions = (
            error.row_partitions
            if isinstance(error, VLMEvalGenerationWorkflowError)
            else None
        )
        reference = write_vlmeval_orchestration_failure_status(
            benchmark,
            directory,
            error,
            stage="vlmeval_generation",
            row_partitions=row_partitions,
        )
    return suite_dataset_status_from_artifacts(
        suite,
        root,
        benchmark.dataset,
        "failed",
        reference.status_artifact,
    )


def run_vlmeval_generation_suite(
    client: object,
    suite: BenchmarkSuiteConfig,
    work_dir: str | Path,
    model_name: str,
    *,
    api_nproc: int,
    resume: bool,
    generation_runner: GenerationRunner | None = None,
) -> VLMEvalGenerationSuiteResult:
    """Run suite entries in order and return their completed immutable ledger result.

    The runner receives one prepared directory per dataset and writes native
    generation artifacts; this workflow persists ledger transitions after each
    dataset. It does not mutate ``suite``. Invalid inputs, unsafe resume state,
    and the first terminal dataset failure raise
    :class:`VLMEvalGenerationSuiteWorkflowError` after recording progress.
    """
    if type(suite) is not BenchmarkSuiteConfig:
        raise TypeError("suite must be a BenchmarkSuiteConfig")
    if type(resume) is not bool:
        raise TypeError("resume must be a boolean")
    root = Path(work_dir).resolve(strict=True)
    runner = run_vlmeval_generation if generation_runner is None else generation_runner
    if resume and (root / SUITE_STATUS_FILENAME).is_file():
        status = load_vlmeval_suite_status(suite, root)
    else:
        if resume and (root / "datasets").exists():
            raise VLMEvalGenerationSuiteWorkflowError(
                "suite resume has dataset outputs but no progress ledger",
                None,
            )
        status = write_vlmeval_suite_status(
            suite, root, planned_suite_dataset_statuses(suite)
        )
    records = status.dataset_statuses
    for benchmark, existing in zip(suite.benchmarks, records, strict=True):
        if existing.status == "completed":
            continue
        directory = prepare_suite_dataset_directory(root, benchmark.dataset)
        try:
            result = runner(
                client,
                benchmark,
                directory,
                model_name,
                api_nproc=api_nproc,
                evaluation_judge=benchmark.judge,
            )
            if type(result) is not VLMEvalGenerationResult:
                raise TypeError("dataset generation returned the wrong result type")
            replacement = suite_dataset_status_from_artifacts(
                suite,
                root,
                benchmark.dataset,
                "completed",
                result.status_artifact,
                result.metrics_artifact,
            )
        except Exception as error:
            try:
                replacement = _failed_dataset_status(
                    suite, benchmark, directory, root, error
                )
                records = _replace_status(records, replacement)
                status = write_vlmeval_suite_status(suite, root, records)
            except Exception as status_error:
                raise VLMEvalGenerationSuiteWorkflowError(
                    f"suite failed for {benchmark.dataset}; status recording failed",
                    status,
                ) from status_error
            raise VLMEvalGenerationSuiteWorkflowError(
                f"suite failed for {benchmark.dataset}", status
            ) from error
        records = _replace_status(records, replacement)
        status = write_vlmeval_suite_status(suite, root, records)
    if any(record.status != "completed" for record in records):
        raise VLMEvalGenerationSuiteWorkflowError("suite did not complete", status)
    return VLMEvalGenerationSuiteResult(status.artifact, records)


def _suite(resolved: ResolvedEvalRun) -> BenchmarkSuiteConfig:
    suite = resolved.configured.benchmark_suite
    if type(suite) is not BenchmarkSuiteConfig or resolved.configured.benchmark is not None:
        raise VLMEvalSuiteResultRecordingError("resolved run is not a benchmark suite")
    return suite


def _loaded_status(
    resolved: ResolvedEvalRun,
    expected: VLMEvalGenerationSuiteResult | VLMEvalSuiteStatusArtifact | None,
) -> VLMEvalSuiteStatusArtifact | None:
    root = Path(resolved.paths.run_directory)
    status_path = root / SUITE_STATUS_FILENAME
    if not status_path.exists() and expected is None:
        return None
    try:
        loaded = load_vlmeval_suite_status(_suite(resolved), root)
    except Exception as error:
        raise VLMEvalSuiteResultRecordingError("suite status cannot be revalidated") from error
    if expected is not None:
        expected_path = (
            expected.suite_status_artifact
            if type(expected) is VLMEvalGenerationSuiteResult
            else expected.artifact
        )
        if loaded.artifact != expected_path:
            raise VLMEvalSuiteResultRecordingError("suite status changed before recording")
    return loaded


def _terminal_statuses(
    resolved: ResolvedEvalRun,
    status: VLMEvalSuiteStatusArtifact | None,
) -> tuple[DatasetTerminalStatus, ...]:
    if status is None:
        return ()
    root = Path(resolved.paths.run_directory)
    terminal = []
    for record in status.dataset_statuses:
        if record.status == "not_started":
            continue
        if record.status_path is None:
            raise VLMEvalSuiteResultRecordingError("terminal suite entry has no status")
        terminal.append(
            named_dataset_terminal_status_for_artifact(
                resolved,
                record.dataset,
                record.status,
                root / record.status_path,
            )
        )
    return tuple(terminal)


def _metric_records(record: SuiteDatasetStatus, root: Path) -> tuple[NormalizedMetric, ...]:
    if record.status != "completed" or record.metrics_path is None:
        raise VLMEvalSuiteResultRecordingError("suite metrics require completed datasets")
    artifact = require_shared_file(root / record.metrics_path, root, "dataset metrics")
    try:
        payload = load_json_bytes(artifact.read_bytes(), maximum_bytes=2_097_152)
        metrics = normalized_metrics_from_payload(payload)
    except Exception as error:
        raise VLMEvalSuiteResultRecordingError("dataset metrics payload is invalid") from error
    prefixed = []
    for metric in metrics:
        source = PurePosixPath(metric.source_file)
        if source.is_absolute() or ".." in source.parts:
            raise VLMEvalSuiteResultRecordingError("dataset metric source path is unsafe")
        prefixed.append(
            NormalizedMetric(
                metric.dataset,
                metric.metric,
                metric.value,
                metric.unit,
                f"{record.dataset_directory}/{source.as_posix()}",
                metric.source_field,
            )
        )
    return tuple(prefixed)


def _write_suite_metrics(status: VLMEvalSuiteStatusArtifact, root: Path) -> Path:
    metrics = tuple(
        metric
        for record in status.dataset_statuses
        for metric in _metric_records(record, root)
    )
    destination = root / SUITE_METRICS_FILENAME
    write_json(destination, normalized_metrics_payload(metrics))
    return destination


def complete_vlmeval_suite_run(
    resolved: ResolvedEvalRun,
    job_id: str,
    result: VLMEvalGenerationSuiteResult,
) -> tuple[Path, Path, Path]:
    """Aggregate completed suite metrics and write its terminal manifest.

    ``result`` and ``resolved`` are read without mutation. The function writes
    the normalized aggregate metrics and resolved manifest, returning their paths
    plus the validated ledger path. Invalid state or artifact contracts raise
    :class:`VLMEvalSuiteResultRecordingError`.
    """
    if type(result) is not VLMEvalGenerationSuiteResult:
        raise VLMEvalSuiteResultRecordingError("suite result has the wrong record type")
    status = _loaded_status(resolved, result)
    if status is None or any(item.status != "completed" for item in status.dataset_statuses):
        raise VLMEvalSuiteResultRecordingError("suite result is not complete")
    root = Path(resolved.paths.run_directory)
    metrics = _write_suite_metrics(status, root)
    manifest = write_resolved_run_manifest(
        resolved,
        run_status="completed",
        dataset_terminal_statuses=_terminal_statuses(resolved, status),
        slurm_job_id=job_id,
    )
    return manifest, status.artifact, metrics


def _write_suite_failure(
    resolved: ResolvedEvalRun,
    error: Exception,
    stage: str,
    status: VLMEvalSuiteStatusArtifact | None,
) -> DatasetTerminalStatus:
    root = Path(resolved.paths.run_directory)
    payload = {
        "schema_version": 2,
        "suite": _suite(resolved).name,
        "status": "failed",
        "stage": stage,
        "failure_type": type(error).__name__,
        "message": sanitized_error_text(error, root),
        "suite_status_path": (
            None if status is None else status.artifact.relative_to(root).as_posix()
        ),
    }
    artifact = root / SUITE_FAILURE_FILENAME
    write_json(artifact, payload)
    return named_dataset_terminal_status_for_artifact(
        resolved, f"suite:{_suite(resolved).name}", "failed", artifact
    )


def record_failed_vlmeval_suite_run(
    resolved: ResolvedEvalRun,
    job_id: str,
    error: Exception,
    stage: str,
    completed_result: VLMEvalGenerationSuiteResult | None,
) -> Path:
    """Write a failed suite manifest and return its path.

    Existing terminal dataset entries are preserved. When no dataset failure was
    recorded, this writes a suite-level failure artifact. Inputs are not mutated;
    invalid ledger, artifact, or manifest state raises the relevant contract error.
    """
    expected = (
        error.suite_status
        if isinstance(error, VLMEvalGenerationSuiteWorkflowError)
        else completed_result
    )
    status = _loaded_status(resolved, expected)
    terminals = _terminal_statuses(resolved, status)
    if not any(item.status == "failed" for item in terminals):
        terminals += (_write_suite_failure(resolved, error, stage, status),)
    return write_resolved_run_manifest(
        resolved,
        run_status="failed",
        dataset_terminal_statuses=terminals,
        slurm_job_id=job_id,
    )
