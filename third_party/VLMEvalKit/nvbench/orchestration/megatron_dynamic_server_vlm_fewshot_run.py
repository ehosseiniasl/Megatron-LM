"""Run VLMEvalKit against the separately supervised Megatron server step."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from . import vlmeval_generation_suite_workflow as suite_generation
from . import vlmeval_generation_workflow as generation
from .benchmark_array_selection import common_generation_config
from .configured_eval_run import GenerationConfig, ResolvedEvalRun
from .dynamic_server_http_client.megatron_dynamic_server_chat_client import (
    MegatronDynamicServerChatClient,
)
from .dynamic_server_http_client.server_preflight import (
    run_server_generation_preflight,
    wait_for_server_health,
)
from .megatron_server_process_management.megatron_dynamic_server_launch import (
    raise_on_termination_signal,
)
from .megatron_server_process_management.split_megatron_server_step import (
    PreparedServerStep,
    SplitServerStepControlPaths,
    TerminalServerStep,
    external_server_process_poll,
    wait_for_prepared_server_step,
    wait_for_terminal_server_step,
    write_server_step_shutdown_request,
)
from .resolved_run_manifest import (
    canonical_config_fingerprint,
    dataset_terminal_status_for_artifact,
    validate_prepared_run_manifest,
    write_resolved_run_manifest,
)
from .utils import require_shared_file, temporary_environment
from .vlmeval_result_contract import (
    GeneratedAnswerResultFinalizationError,
)
from .vlmeval_orchestration_failure_status import (
    reference_existing_vlmeval_failure_status,
    write_vlmeval_orchestration_failure_status,
)
from .vlmeval_generation_suite_workflow import (
    complete_vlmeval_suite_run,
    record_failed_vlmeval_suite_run,
)


_PREFLIGHT_IMAGE = Path("assets/server_preflight.png")


# Lifecycle records and production dependency bindings.


class MegatronDynamicServerVlmFewshotRunError(RuntimeError):
    """Raised after truthful terminal state is recorded whenever recording is possible."""


@dataclass(frozen=True)
class MegatronDynamicServerVlmFewshotRunResult:
    """Immutable paths returned after a completed external-server client run.

    ``manifest_artifact``, ``dataset_status_artifact``, and ``metrics_artifact``
    are existing terminal artifacts. This record does not mutate inputs and is
    returned only after clean server termination.
    """

    manifest_artifact: Path
    dataset_status_artifact: Path
    metrics_artifact: Path


@dataclass(frozen=True)
class MegatronDynamicServerVlmPredictionRunResult:
    """Durable prediction handoff returned after clean server termination."""

    prediction_artifact: Path
    readiness_artifact: Path


@dataclass(frozen=True)
class MegatronDynamicServerVlmFewshotDependencies:
    """Explicit external boundaries for one client lifecycle invocation.

    Callers may supply deterministic implementations for preparation, health,
    generation, and cleanup. The record mutates nothing; callback failures
    propagate through the lifecycle's truthful recording and cleanup policy.
    """

    prepared_waiter: Callable[
        ..., tuple[SplitServerStepControlPaths, PreparedServerStep]
    ]
    manifest_validator: Callable[[ResolvedEvalRun, str], Path]
    health_waiter: Callable[..., object]
    client_factory: Callable[..., object]
    generation_preflight: Callable[[object, Path], object]
    generation_runner: Callable[..., generation.VLMEvalGenerationResult]
    prediction_runner: Callable[..., generation.VLMEvalPredictionResult]
    suite_generation_runner: Callable[
        ..., suite_generation.VLMEvalGenerationSuiteResult
    ]
    shutdown_writer: Callable[[SplitServerStepControlPaths, str], Path]
    terminal_waiter: Callable[..., TerminalServerStep]


GenerationResult = (
    generation.VLMEvalGenerationResult
    | generation.VLMEvalPredictionResult
    | suite_generation.VLMEvalGenerationSuiteResult
)


def production_fewshot_run_dependencies() -> MegatronDynamicServerVlmFewshotDependencies:
    """Return fresh production bindings without mutating process-global state.

    The returned record binds only supported split-step, HTTP, and generation
    implementations. Import or construction errors from those implementations
    propagate to the caller.
    """
    return MegatronDynamicServerVlmFewshotDependencies(
        wait_for_prepared_server_step,
        validate_prepared_run_manifest,
        wait_for_server_health,
        MegatronDynamicServerChatClient,
        run_server_generation_preflight,
        generation.run_vlmeval_generation,
        generation.run_vlmeval_predictions,
        suite_generation.run_vlmeval_generation_suite,
        write_server_step_shutdown_request,
        wait_for_terminal_server_step,
    )


# Single-benchmark versus suite dispatch and terminal result recording.


def run_configured_generation(
    resolved: ResolvedEvalRun,
    dependencies: MegatronDynamicServerVlmFewshotDependencies,
    client: object,
    generation_config: GenerationConfig,
) -> GenerationResult:
    """Run the configured single benchmark or suite and return its typed result.

    ``resolved`` selects one mode; ``dependencies`` owns generation side
    effects. ``LMUData`` is temporarily bound to the cache root and restored
    before returning or propagating a generation exception.
    """
    with temporary_environment("LMUData", resolved.paths.cache_root):
        suite = resolved.configured.benchmark_suite
        if suite is not None:
            return dependencies.suite_generation_runner(
                client,
                suite,
                resolved.paths.run_directory,
                resolved.configured.model.name,
                api_nproc=resolved.configured.run.api_nproc,
                resume=resolved.configured.output.resume,
            )
        benchmark = resolved.configured.benchmark
        if benchmark is None:
            raise MegatronDynamicServerVlmFewshotRunError(
                "configured run has neither a benchmark nor a benchmark suite"
            )
        return dependencies.generation_runner(
            client,
            benchmark,
            resolved.paths.run_directory,
            resolved.configured.model.name,
            api_nproc=resolved.configured.run.api_nproc,
            evaluation_judge=benchmark.judge,
        )


def run_configured_predictions(
    resolved: ResolvedEvalRun,
    dependencies: MegatronDynamicServerVlmFewshotDependencies,
    client: object,
) -> generation.VLMEvalPredictionResult:
    """Generate one array-selected benchmark without scoring its answers."""
    benchmark = resolved.configured.benchmark
    if benchmark is None or resolved.configured.benchmark_suite is not None:
        raise MegatronDynamicServerVlmFewshotRunError(
            "prediction-only mode requires one array-selected benchmark"
        )
    with temporary_environment("LMUData", resolved.paths.cache_root):
        return dependencies.prediction_runner(
            client,
            benchmark,
            resolved.paths.run_directory,
            resolved.configured.model.name,
            api_nproc=resolved.configured.run.api_nproc,
        )


def record_failed_vlmeval_run(
    resolved: ResolvedEvalRun,
    job_id: str,
    error: Exception,
    stage: str,
    completed_result: generation.VLMEvalGenerationResult | None,
) -> Path:
    """Write failed single-dataset status and return its failed-manifest path.

    Inputs are read without mutation. Existing finalization failure state is
    reused when valid; otherwise native failure state is written. Invalid mode,
    status, or manifest operations raise their underlying contract errors.
    """
    root = Path(resolved.paths.run_directory)
    benchmark = resolved.configured.benchmark
    if benchmark is None or resolved.configured.benchmark_suite is not None:
        raise ValueError("single-dataset result recording requires one benchmark")
    if isinstance(error, GeneratedAnswerResultFinalizationError):
        try:
            reference = reference_existing_vlmeval_failure_status(benchmark, root)
        except Exception:
            reference = write_vlmeval_orchestration_failure_status(
                benchmark, root, error, stage=stage
            )
    else:
        row_partitions = None
        if isinstance(error, generation.VLMEvalGenerationWorkflowError):
            row_partitions = error.row_partitions
        reference = write_vlmeval_orchestration_failure_status(
            benchmark,
            root,
            error,
            stage=stage,
            row_partitions=row_partitions,
            completed_result=completed_result,
        )
    terminal = dataset_terminal_status_for_artifact(
        resolved, "failed", reference.status_artifact
    )
    return write_resolved_run_manifest(
        resolved,
        run_status="failed",
        dataset_terminal_statuses=(terminal,),
        slurm_job_id=job_id,
    )


def complete_vlmeval_run(
    resolved: ResolvedEvalRun,
    job_id: str,
    result: generation.VLMEvalGenerationResult,
) -> tuple[Path, Path, Path]:
    """Write the completed single-dataset manifest and return terminal paths.

    ``result`` is read without mutation. Its status must be a contained shared
    file for a single-benchmark run; invalid mode or artifact state raises the
    relevant validation error.
    """
    root = Path(resolved.paths.run_directory)
    if resolved.configured.benchmark is None or resolved.configured.benchmark_suite is not None:
        raise ValueError("single-dataset result recording requires one benchmark")
    status = require_shared_file(result.status_artifact, root, "dataset status")
    terminal = dataset_terminal_status_for_artifact(resolved, "completed", status)
    manifest = write_resolved_run_manifest(
        resolved,
        run_status="completed",
        dataset_terminal_statuses=(terminal,),
        slurm_job_id=job_id,
    )
    return manifest, status, result.metrics_artifact


def record_failed_configured_run(
    resolved: ResolvedEvalRun,
    slurm_job_id: str,
    error: Exception,
    stage: str,
    result: GenerationResult | None,
) -> None:
    """Persist failure state for the selected mode without mutating ``result``.

    A cross-mode partial result is ignored rather than mis-recorded. Artifact
    and manifest failures propagate from the selected recorder.
    """
    if resolved.configured.benchmark_suite is not None:
        suite_result = (
            result if type(result) is suite_generation.VLMEvalGenerationSuiteResult else None
        )
        record_failed_vlmeval_suite_run(
            resolved, slurm_job_id, error, stage, suite_result
        )
        return
    single_result = result if type(result) is generation.VLMEvalGenerationResult else None
    record_failed_vlmeval_run(resolved, slurm_job_id, error, stage, single_result)


def complete_configured_run(
    resolved: ResolvedEvalRun,
    slurm_job_id: str,
    result: GenerationResult | None,
) -> MegatronDynamicServerVlmFewshotRunResult:
    """Validate the selected result mode and return completed terminal paths.

    Inputs are not mutated. Missing or cross-mode results raise
    :class:`MegatronDynamicServerVlmFewshotRunError`; recorder failures retain
    their original contract type.
    """
    if resolved.configured.benchmark_suite is not None:
        if type(result) is not suite_generation.VLMEvalGenerationSuiteResult:
            raise MegatronDynamicServerVlmFewshotRunError(
                "suite generation returned no result"
            )
        completed = complete_vlmeval_suite_run(resolved, slurm_job_id, result)
    else:
        if type(result) is not generation.VLMEvalGenerationResult:
            raise MegatronDynamicServerVlmFewshotRunError(
                "generation returned no result"
            )
        completed = complete_vlmeval_run(resolved, slurm_job_id, result)
    manifest, dataset_status, metrics = completed
    return MegatronDynamicServerVlmFewshotRunResult(
        manifest_artifact=manifest,
        dataset_status_artifact=dataset_status,
        metrics_artifact=metrics,
    )


# External-server client orchestration.


def _request_clean_server_shutdown(
    dependencies: MegatronDynamicServerVlmFewshotDependencies,
    paths: SplitServerStepControlPaths,
    fingerprint: str,
) -> None:
    dependencies.shutdown_writer(paths, fingerprint)
    terminal = dependencies.terminal_waiter(paths, fingerprint, 60.0)
    if terminal.status != "clean":
        raise MegatronDynamicServerVlmFewshotRunError(
            "server step reported failed cleanup"
        )


def execute_megatron_dynamic_server_vlm_fewshot_client(
    resolved: ResolvedEvalRun,
    slurm_job_id: str,
    *,
    expected_endpoint: str,
    expected_health_endpoint: str,
    prediction_only: bool = False,
    dependencies: MegatronDynamicServerVlmFewshotDependencies | None = None,
) -> MegatronDynamicServerVlmFewshotRunResult | MegatronDynamicServerVlmPredictionRunResult:
    """Evaluate through one prepared external server and record terminal state.

    ``resolved`` supplies immutable paths, model policy, and benchmark
    selection. The expected loopback endpoints must equal the server's prepared
    control artifact. Once that artifact is observed, shutdown is always
    requested, including after generation failure. The function temporarily
    binds ``LMUData`` and restores its prior value. It returns only after clean
    server termination and durable terminal artifacts; expected lifecycle
    failures raise :class:`MegatronDynamicServerVlmFewshotRunError` with the
    original cause preserved.
    """
    if type(resolved) is not ResolvedEvalRun:
        raise MegatronDynamicServerVlmFewshotRunError("resolved run has the wrong type")
    if type(prediction_only) is not bool:
        raise MegatronDynamicServerVlmFewshotRunError(
            "prediction_only must be a boolean"
        )
    deps = production_fewshot_run_dependencies() if dependencies is None else dependencies
    if type(deps) is not MegatronDynamicServerVlmFewshotDependencies:
        raise MegatronDynamicServerVlmFewshotRunError("run dependencies have the wrong type")
    fingerprint = canonical_config_fingerprint(resolved)
    timeout = resolved.configured.model.server.health_timeout_seconds
    paths: SplitServerStepControlPaths | None = None
    result: GenerationResult | None = None
    error: Exception | None = None
    stage = "server_prepared"
    try:
        paths, prepared = deps.prepared_waiter(
            Path(resolved.paths.run_directory), slurm_job_id, fingerprint, timeout
        )
        if prepared.endpoint != expected_endpoint:
            raise MegatronDynamicServerVlmFewshotRunError(
                "server prepared an unexpected generation endpoint"
            )
        if prepared.health_endpoint != expected_health_endpoint:
            raise MegatronDynamicServerVlmFewshotRunError(
                "server prepared an unexpected health endpoint"
            )
        deps.manifest_validator(resolved, slurm_job_id)
        process_poll = lambda: external_server_process_poll(paths, fingerprint)
        allowed_host = urlsplit(expected_endpoint).hostname
        with raise_on_termination_signal():
            stage = "server_health"
            deps.health_waiter(
                prepared.health_endpoint,
                timeout,
                process_poll,
                allowed_host=allowed_host,
            )
            stage = "server_generation_preflight"
            generation_config = common_generation_config(resolved.configured)
            client = deps.client_factory(
                prepared.endpoint,
                resolved.configured.model.name,
                generation_config,
                resolved.configured.model.server.request_timeout_seconds,
                allowed_host=allowed_host,
            )
            image = Path(__file__).resolve().parent.parent / _PREFLIGHT_IMAGE
            deps.generation_preflight(client, image)
            stage = "vlmeval_generation"
            if prediction_only:
                result = run_configured_predictions(resolved, deps, client)
            else:
                result = run_configured_generation(
                    resolved, deps, client, generation_config
                )
    except Exception as run_error:
        error = run_error
    finally:
        if paths is not None:
            try:
                _request_clean_server_shutdown(deps, paths, fingerprint)
            except Exception as cleanup_error:
                if error is None:
                    error = cleanup_error
                    stage = "server_cleanup"
                else:
                    stage = f"{stage}_with_server_cleanup_failure"

    if error is not None:
        if Path(resolved.paths.run_directory).is_dir():
            try:
                record_failed_configured_run(
                    resolved, slurm_job_id, error, stage, result
                )
            except Exception as status_error:
                raise MegatronDynamicServerVlmFewshotRunError(
                    f"run failed during {stage}; terminal status could not be written"
                ) from status_error
        raise MegatronDynamicServerVlmFewshotRunError(
            f"run failed during {stage}"
        ) from error
    if prediction_only:
        if type(result) is not generation.VLMEvalPredictionResult:
            raise MegatronDynamicServerVlmFewshotRunError(
                "prediction-only generation returned no result"
            )
        return MegatronDynamicServerVlmPredictionRunResult(
            result.prediction_artifact, result.readiness_artifact
        )
    return complete_configured_run(resolved, slurm_job_id, result)
