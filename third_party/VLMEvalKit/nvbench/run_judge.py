"""Score one completed nvbench prediction artifact without allocating GPUs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import MutableMapping
from pathlib import Path

from nvbench.orchestration import (
    eval_run_path_resolution as preparation_module,
    megatron_dynamic_server_vlm_fewshot_run as run_module,
)
from nvbench.orchestration.judge_runtime import prepare_judge_runtime
from nvbench.orchestration.utils import temporary_environment
from nvbench.orchestration.vlmeval_generation_workflow import (
    adopt_existing_vlmeval_predictions,
    score_vlmeval_predictions,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--benchmark-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--adopt-existing-predictions",
        action="store_true",
        help="Validate and adopt complete predictions from a pre-split run",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    environment: MutableMapping[str, str] | None = None,
) -> int:
    """Validate the GPU handoff, run native judge scoring, and record results."""
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
    if benchmark is None or benchmark.judge == "exact_matching":
        raise RuntimeError("CPU scoring requires one LLM-judged benchmark")

    result = None
    try:
        prepare_judge_runtime(benchmark.judge, execution_environment)
        with temporary_environment("LMUData", resolved.paths.cache_root):
            if args.adopt_existing_predictions:
                adopt_existing_vlmeval_predictions(
                    benchmark,
                    resolved.paths.run_directory,
                    resolved.configured.model.name,
                )
            result = score_vlmeval_predictions(
                benchmark,
                resolved.paths.run_directory,
                evaluation_judge=benchmark.judge,
            )
        manifest, status, metrics = run_module.complete_vlmeval_run(
            resolved, execution_environment.get("SLURM_JOB_ID", ""), result
        )
    except Exception as error:
        if Path(resolved.paths.run_directory).is_dir():
            run_module.record_failed_vlmeval_run(
                resolved,
                execution_environment.get("SLURM_JOB_ID", ""),
                error,
                "vlmeval_judge_scoring",
                result,
            )
        raise

    print(
        json.dumps(
            {
                "run_status": "completed",
                "manifest": str(manifest),
                "dataset_status": str(status),
                "metrics": str(metrics),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"nvbench judge step failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
