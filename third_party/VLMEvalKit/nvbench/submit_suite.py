"""Plan or submit one resource-aware VLMEvalKit benchmark suite."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nvbench.orchestration.suite_submission import (
    build_judge_scoring_submission_plan,
    build_suite_submission_plan,
    submit_judge_scoring_plan,
    submit_suite_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--benchmark-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print the exact plan and submit it unless ``--dry-run`` was requested."""
    args = _parser().parse_args(argv)
    launcher = Path(__file__).resolve().with_name("run_vlm_eval.sbatch")
    judge_launcher = Path(__file__).resolve().with_name("run_judge_eval.sbatch")
    plan = build_suite_submission_plan(
        args.run_config,
        args.workspace,
        launcher,
        benchmark_index=args.benchmark_index,
        limit=args.limit,
    )
    judge_plan = build_judge_scoring_submission_plan(
        args.run_config,
        args.workspace,
        judge_launcher,
        benchmark_index=args.benchmark_index,
        limit=args.limit,
    )
    payload = {
        "submissions": [asdict(item) for item in plan],
        "judge_submissions": [asdict(item) for item in judge_plan],
    }
    if not args.dry_run:
        generation_job_ids = submit_suite_plan(plan)
        payload["job_ids"] = generation_job_ids
        payload["judge_job_ids"] = submit_judge_scoring_plan(
            judge_plan, plan, generation_job_ids
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
