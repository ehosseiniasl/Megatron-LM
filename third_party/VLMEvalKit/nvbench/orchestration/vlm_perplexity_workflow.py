"""Human-readable conditional-perplexity evaluation workflow."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .configured_eval_run import ConditionalPerplexityBenchmarkConfig, ResolvedEvalRun
from .dynamic_server_http_client.megatron_prompt_logprob_client import (
    MegatronPromptLogprobClient,
    PromptTokenizer,
    build_scored_prompt_token_ids,
)
from .perplexity_datasets.surge_mm import (
    SurgeMMImageReader,
    SurgeMMSample,
    load_packed_jsonl_tar,
)
from .utils import append_jsonl_record, write_json, write_jsonl
from .vlm_perplexity_result_contract import (
    ConditionalPerplexityScore,
    aggregate_conditional_perplexity,
    load_completed_perplexity_scores,
    score_description,
    write_failed_perplexity_status,
)

SAMPLES_FILENAME = "surge_mm_ppl.samples.jsonl"
FAILURES_FILENAME = "surge_mm_ppl.failures.jsonl"
METRICS_FILENAME = "surge_mm_ppl.metrics.json"
STATUS_FILENAME = "vlmeval_dataset_status.json"


class ConditionalPerplexityWorkflowError(RuntimeError):
    """Raised after durable diagnostic state is written for an incomplete run."""


@dataclass(frozen=True)
class ConditionalPerplexityWorkflowResult:
    """Terminal artifacts produced only by a complete PPL run."""

    status_artifact: Path
    metrics_artifact: Path
    samples_artifact: Path


def _score_one(
    sample: SurgeMMSample,
    benchmark: ConditionalPerplexityBenchmarkConfig,
    tokenizer: PromptTokenizer,
    client: MegatronPromptLogprobClient,
    image_reader: SurgeMMImageReader,
) -> ConditionalPerplexityScore:
    prompt_tokens = build_scored_prompt_token_ids(
        tokenizer, benchmark.request.instruction, sample.description
    )
    response = client.score_prompt(prompt_tokens, image_reader.read(sample))
    return score_description(
        uid=sample.uid,
        category=sample.category,
        description=sample.description,
        prompt_tokens=response.tokens,
        prompt_logprobs=response.logprobs,
    )


def run_conditional_perplexity(
    resolved: ResolvedEvalRun, client: MegatronPromptLogprobClient, tokenizer: PromptTokenizer
) -> ConditionalPerplexityWorkflowResult:
    """Score pending samples, require exact coverage, and publish pooled metrics."""
    benchmark = resolved.configured.benchmark
    if type(benchmark) is not ConditionalPerplexityBenchmarkConfig:
        raise ConditionalPerplexityWorkflowError(
            "resolved run does not select conditional perplexity"
        )
    if benchmark.data_format != "packed_jsonl_tar":
        raise ConditionalPerplexityWorkflowError("resolved PPL dataset is unavailable")
    run_directory = Path(resolved.paths.run_directory)
    samples_path = run_directory / SAMPLES_FILENAME
    failures_path = run_directory / FAILURES_FILENAME
    metrics_path = run_directory / METRICS_FILENAME
    status_path = run_directory / STATUS_FILENAME
    samples = load_packed_jsonl_tar(
        Path(benchmark.configured_data_path), limit=benchmark.scored_limit
    )
    if samples_path.exists() and not resolved.configured.output.resume:
        raise ConditionalPerplexityWorkflowError(
            "existing PPL sample ledger requires output.resume=true"
        )
    completed = list(load_completed_perplexity_scores(samples_path))
    completed_ids = {score.uid for score in completed}
    expected_ids = {sample.uid for sample in samples}
    if not completed_ids.issubset(expected_ids):
        raise ConditionalPerplexityWorkflowError(
            "completed sample ledger does not belong to the configured dataset"
        )
    pending = tuple(sample for sample in samples if sample.uid not in completed_ids)
    failures: list[dict[str, object]] = []

    with SurgeMMImageReader() as image_reader:
        def evaluate(
            sample: SurgeMMSample,
        ) -> tuple[SurgeMMSample, ConditionalPerplexityScore | None, Exception | None]:
            try:
                return sample, _score_one(sample, benchmark, tokenizer, client, image_reader), None
            except Exception as error:
                return sample, None, error

        with ThreadPoolExecutor(max_workers=resolved.configured.run.api_nproc) as executor:
            for sample, score, error in executor.map(evaluate, pending):
                if error is not None:
                    failures.append(
                        {
                            "uid": sample.uid,
                            "error_type": type(error).__name__,
                            "error": str(error)[:500],
                        }
                    )
                    continue
                assert score is not None
                append_jsonl_record(samples_path, asdict(score))
                completed.append(score)

    write_jsonl(failures_path, tuple(failures))
    observed_ids = {score.uid for score in completed}
    complete = not failures and observed_ids == expected_ids and len(completed) == len(samples)
    status_payload = {
        "schema_version": 1,
        "evaluation_kind": "conditional_perplexity",
        "dataset": benchmark.dataset,
        "status": "completed" if complete else "failed",
        "expected_sample_count": len(samples),
        "completed_sample_count": len(completed),
        "failed_sample_count": len(failures),
        "samples_path": SAMPLES_FILENAME,
        "failures_path": FAILURES_FILENAME,
        "metrics_path": METRICS_FILENAME if complete else None,
    }
    if not complete:
        error = ConditionalPerplexityWorkflowError(
            f"conditional perplexity is incomplete: {len(completed)}/{len(samples)} samples"
        )
        write_failed_perplexity_status(
            status_path,
            dataset=benchmark.dataset,
            expected=len(samples),
            completed=len(completed),
            failed=len(failures),
            stage="conditional_perplexity",
            error=error,
        )
        raise error

    metrics = {
        "schema_version": 1,
        "evaluation_kind": "conditional_perplexity",
        "dataset": benchmark.dataset,
        **aggregate_conditional_perplexity(tuple(completed)),
    }
    write_json(metrics_path, metrics)
    write_json(status_path, status_payload)
    return ConditionalPerplexityWorkflowResult(status_path, metrics_path, samples_path)


__all__ = (
    "ConditionalPerplexityWorkflowError",
    "ConditionalPerplexityWorkflowResult",
    "run_conditional_perplexity",
)
