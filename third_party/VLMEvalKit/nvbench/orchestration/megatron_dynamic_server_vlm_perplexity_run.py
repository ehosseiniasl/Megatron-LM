"""Run conditional perplexity against a separately supervised Megatron server."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .configured_eval_run import ConditionalPerplexityBenchmarkConfig, ResolvedEvalRun
from .dynamic_server_http_client.json_http_transport import same_http_origin
from .dynamic_server_http_client.megatron_prompt_logprob_client import MegatronPromptLogprobClient
from .dynamic_server_http_client.server_preflight import wait_for_server_health
from .huggingface_tokenizer import load_huggingface_tokenizer
from .megatron_server_process_management.megatron_dynamic_server_launch import (
    raise_on_termination_signal,
)
from .megatron_server_process_management.split_megatron_server_step import (
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
from .vlm_perplexity_result_contract import write_failed_perplexity_status
from .vlm_perplexity_workflow import (
    STATUS_FILENAME,
    ConditionalPerplexityWorkflowResult,
    run_conditional_perplexity,
)


class MegatronDynamicServerVlmPerplexityRunError(RuntimeError):
    """Raised after server cleanup and truthful terminal-state recording."""


@dataclass(frozen=True)
class MegatronDynamicServerVlmPerplexityRunResult:
    """Terminal artifacts returned by a completed conditional-PPL run."""

    manifest_artifact: Path
    dataset_status_artifact: Path
    metrics_artifact: Path


def execute_megatron_dynamic_server_vlm_perplexity_client(
    resolved: ResolvedEvalRun,
    slurm_job_id: str,
    *,
    expected_endpoint: str,
    expected_health_endpoint: str,
) -> MegatronDynamicServerVlmPerplexityRunResult:
    """Run PPL through a prepared server, publish artifacts, and stop it cleanly."""
    benchmark = resolved.configured.benchmark
    if type(benchmark) is not ConditionalPerplexityBenchmarkConfig:
        raise MegatronDynamicServerVlmPerplexityRunError(
            "resolved run does not select conditional perplexity"
        )
    fingerprint = canonical_config_fingerprint(resolved)
    timeout = resolved.configured.model.server.health_timeout_seconds
    controls = None
    result: ConditionalPerplexityWorkflowResult | None = None
    failure: Exception | None = None
    stage = "server_prepared"
    try:
        controls, prepared = wait_for_prepared_server_step(
            Path(resolved.paths.run_directory), slurm_job_id, fingerprint, timeout
        )
        if (
            urlsplit(prepared.endpoint).path != "/v1/chat/completions"
            or urlsplit(expected_endpoint).path != "/v1/completions"
            or not same_http_origin(prepared.endpoint, expected_endpoint)
        ):
            raise MegatronDynamicServerVlmPerplexityRunError(
                "server prepared an unexpected endpoint"
            )
        if prepared.health_endpoint != expected_health_endpoint:
            raise MegatronDynamicServerVlmPerplexityRunError(
                "server prepared an unexpected health endpoint"
            )
        validate_prepared_run_manifest(resolved, slurm_job_id)
        process_poll = lambda: external_server_process_poll(controls, fingerprint)
        allowed_host = urlsplit(expected_endpoint).hostname
        with raise_on_termination_signal():
            stage = "server_health"
            wait_for_server_health(
                expected_health_endpoint, timeout, process_poll, allowed_host=allowed_host
            )
            stage = "conditional_perplexity"
            client = MegatronPromptLogprobClient(
                expected_endpoint,
                resolved.configured.model.name,
                resolved.configured.model.server.request_timeout_seconds,
                benchmark.request.retry,
                allowed_host=allowed_host,
            )
            tokenizer_config = resolved.configured.model.tokenizer
            tokenizer = load_huggingface_tokenizer(
                tokenizer_config.repository, tokenizer_config.revision
            )
            result = run_conditional_perplexity(resolved, client, tokenizer)
    except Exception as error:
        failure = error
    finally:
        if controls is not None:
            try:
                write_server_step_shutdown_request(controls, fingerprint)
                terminal = wait_for_terminal_server_step(controls, fingerprint, 60.0)
                if terminal.status != "clean":
                    raise MegatronDynamicServerVlmPerplexityRunError(
                        "server step reported failed cleanup"
                    )
            except Exception as cleanup_error:
                if failure is None:
                    failure = cleanup_error
                    stage = "server_cleanup"

    if failure is not None:
        if Path(resolved.paths.run_directory).is_dir():
            status = Path(resolved.paths.run_directory) / STATUS_FILENAME
            if not status.exists():
                write_failed_perplexity_status(
                    status,
                    dataset=benchmark.dataset,
                    expected=None,
                    completed=None,
                    failed=None,
                    stage=stage,
                    error=failure,
                )
            terminal = dataset_terminal_status_for_artifact(resolved, "failed", status)
            write_resolved_run_manifest(
                resolved,
                run_status="failed",
                dataset_terminal_statuses=(terminal,),
                slurm_job_id=slurm_job_id,
            )
        raise MegatronDynamicServerVlmPerplexityRunError(
            f"conditional-perplexity run failed during {stage}"
        ) from failure
    if result is None:
        raise MegatronDynamicServerVlmPerplexityRunError("PPL workflow returned no result")
    terminal = dataset_terminal_status_for_artifact(resolved, "completed", result.status_artifact)
    manifest = write_resolved_run_manifest(
        resolved,
        run_status="completed",
        dataset_terminal_statuses=(terminal,),
        slurm_job_id=slurm_job_id,
    )
    return MegatronDynamicServerVlmPerplexityRunResult(
        manifest, result.status_artifact, result.metrics_artifact
    )


__all__ = (
    "MegatronDynamicServerVlmPerplexityRunError",
    "MegatronDynamicServerVlmPerplexityRunResult",
    "execute_megatron_dynamic_server_vlm_perplexity_client",
)
