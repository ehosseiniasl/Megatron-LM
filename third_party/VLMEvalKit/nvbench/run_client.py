"""Run the VLMEvalKit client against its sibling Megatron server job step."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import MutableMapping

from nvbench.orchestration import (
    eval_run_path_resolution as preparation_module,
    megatron_dynamic_server_vlm_fewshot_run as run_module,
    megatron_dynamic_server_vlm_perplexity_run as perplexity_run_module,
)
from nvbench.orchestration.configured_eval_run import ConditionalPerplexityBenchmarkConfig
from nvbench.orchestration.judge_runtime import prepare_judge_runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--coordinator-port", type=int, default=8001)
    parser.add_argument("--benchmark-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    environment: MutableMapping[str, str] | None = None,
) -> int:
    """Run one array-selected client against its sibling server and print results."""
    args = _parser().parse_args(argv)
    execution_environment = os.environ if environment is None else environment

    resolved = preparation_module.resolve_megatron_dynamic_server_vlm_fewshot_run(
        args.run_config,
        args.workspace,
        benchmark_index=args.benchmark_index,
        scored_limit=args.limit,
        validate_destination=False,
    )
    benchmark = resolved.configured.benchmark
    if benchmark is None:
        raise RuntimeError("array-selected client did not resolve one benchmark")

    health_endpoint = f"http://{args.server_host}:{args.server_port}/v1/health"
    if type(benchmark) is ConditionalPerplexityBenchmarkConfig:
        endpoint = f"http://{args.server_host}:{args.server_port}/v1/completions"
        result = perplexity_run_module.execute_megatron_dynamic_server_vlm_perplexity_client(
            resolved,
            execution_environment.get("SLURM_JOB_ID", ""),
            expected_endpoint=endpoint,
            expected_health_endpoint=health_endpoint,
        )
    else:
        prediction_only = benchmark.judge != "exact_matching"
        if not prediction_only:
            prepare_judge_runtime(benchmark.judge, execution_environment)
        endpoint = f"http://{args.server_host}:{args.server_port}/v1/chat/completions"
        result = run_module.execute_megatron_dynamic_server_vlm_fewshot_client(
            resolved,
            execution_environment.get("SLURM_JOB_ID", ""),
            expected_endpoint=endpoint,
            expected_health_endpoint=health_endpoint,
            prediction_only=prediction_only,
        )

    if type(result) is run_module.MegatronDynamicServerVlmPredictionRunResult:
        payload = {
            "run_status": "predictions_ready",
            "prediction": str(result.prediction_artifact),
            "readiness": str(result.readiness_artifact),
        }
    else:
        payload = {
            "run_status": "completed",
            "manifest": str(result.manifest_artifact),
            "dataset_status": str(result.dataset_status_artifact),
            "metrics": str(result.metrics_artifact),
        }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"nvbench client step failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
