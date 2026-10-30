"""Essential tests for resource-aware suite submission."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from orchestration.suite_submission import (
    SuiteSubmissionError,
    build_judge_scoring_submission_plan,
    build_suite_submission_plan,
    submit_judge_scoring_plan,
    submit_suite_plan,
)


NVBENCH = Path(__file__).resolve().parents[1]
RUN_CONFIG = NVBENCH / "configs/run_configs/n4_v6_example.yaml"
PPL_RUN_CONFIG = NVBENCH / "configs/run_configs/n4_v6_surge_mm_ppl_example.yaml"
LAUNCHER = NVBENCH / "run_vlm_eval.sbatch"
JUDGE_LAUNCHER = NVBENCH / "run_judge_eval.sbatch"


class SuiteSubmissionTest(unittest.TestCase):
    def test_conditional_perplexity_does_not_plan_judge_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            plan = build_judge_scoring_submission_plan(
                PPL_RUN_CONFIG, workspace, JUDGE_LAUNCHER
            )
        self.assertEqual(plan, ())

    def test_judge_plan_is_cpu_only_and_groups_by_generation_resource(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            plan = build_judge_scoring_submission_plan(
                RUN_CONFIG, workspace, JUDGE_LAUNCHER
            )
        self.assertEqual(tuple(item.source_resource for item in plan), ("n1", "n4"))
        self.assertEqual(plan[0].benchmark_indices, (13,))
        self.assertEqual(plan[1].benchmark_indices, (5,))
        for item in plan:
            self.assertIn("--partition=cpu", item.argv)
            self.assertIn("--nodes=1", item.argv)
            self.assertIn("--cpus-per-task=32", item.argv)
            self.assertFalse(any("gpu" in argument for argument in item.argv))

    def test_judge_submission_uses_afterany_for_matching_gpu_group(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            generation = build_suite_submission_plan(
                RUN_CONFIG, workspace, LAUNCHER
            )
            scoring = build_judge_scoring_submission_plan(
                RUN_CONFIG, workspace, JUDGE_LAUNCHER
            )
        with patch(
            "orchestration.suite_submission.subprocess.run",
            side_effect=(
                SimpleNamespace(stdout="20001\n"),
                SimpleNamespace(stdout="20002\n"),
            ),
        ) as run:
            job_ids = submit_judge_scoring_plan(
                scoring,
                generation,
                ("10001;cmh", "10002", "10003"),
                {"NVBENCH_CLIENT_IMAGE": "client.sqsh"},
            )
        self.assertEqual(job_ids, ("20001", "20002"))
        first_argv = run.call_args_list[0].args[0]
        second_argv = run.call_args_list[1].args[0]
        self.assertIn("--dependency=afterany:10001", first_argv)
        self.assertIn("--dependency=afterany:10002", second_argv)

    def test_plan_groups_sparse_indices_by_configured_resource(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            plan = build_suite_submission_plan(RUN_CONFIG, workspace, LAUNCHER)
        self.assertEqual(tuple(item.resource for item in plan), ("n1", "n4", "n8"))
        self.assertEqual(plan[0].benchmark_indices, (0, 4, 9, 10, 11, 12, 13, 15, 16, 17))
        self.assertEqual(plan[1].benchmark_indices, (1, 2, 3, 5, 6, 14))
        self.assertEqual(plan[2].benchmark_indices, (7, 8))
        self.assertIn("--nodes=1", plan[0].argv)
        self.assertIn("--nodes=4", plan[1].argv)
        self.assertIn("--nodes=8", plan[2].argv)
        self.assertIn("--array=7,8%1", plan[2].argv)

    def test_debug_plan_selects_one_benchmark_and_forwards_limit(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            plan = build_suite_submission_plan(
                RUN_CONFIG,
                workspace,
                LAUNCHER,
                benchmark_index=0,
                limit=10,
            )
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].benchmark_indices, (0,))
        self.assertEqual(plan[0].datasets, ("OCRBench",))
        self.assertIn("--array=0%2", plan[0].argv)
        self.assertEqual(plan[0].argv[-2:], ("--limit", "10"))

    def test_debug_plan_rejects_invalid_selection_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            for keyword in (
                {"benchmark_index": 18},
                {"benchmark_index": True},
                {"limit": 0},
                {"limit": True},
            ):
                with self.subTest(keyword=keyword):
                    with self.assertRaises(SuiteSubmissionError):
                        build_suite_submission_plan(
                            RUN_CONFIG,
                            workspace,
                            LAUNCHER,
                            **keyword,
                        )

    def test_submit_uses_argv_without_shell_and_resource_environment(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            plan = build_suite_submission_plan(RUN_CONFIG, workspace, LAUNCHER)[:1]
        with patch(
            "orchestration.suite_submission.subprocess.run",
            return_value=SimpleNamespace(stdout="12345\n"),
        ) as run:
            job_ids = submit_suite_plan(
                plan,
                {"NVBENCH_SERVER_IMAGE": "server.sqsh", "NVBENCH_CLIENT_IMAGE": "client.sqsh"},
            )
        self.assertEqual(job_ids, ("12345",))
        _, keywords = run.call_args
        self.assertNotIn("shell", keywords)
        self.assertEqual(keywords["env"]["NVBENCH_SERVER_CPUS"], "56")

    def test_submit_requires_both_container_images(self) -> None:
        with self.assertRaisesRegex(SuiteSubmissionError, "NVBENCH_SERVER_IMAGE"):
            submit_suite_plan((), {})

    def test_submit_rejects_noncanonical_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            plan = build_suite_submission_plan(RUN_CONFIG, workspace, LAUNCHER)[:1]
        with patch(
            "orchestration.suite_submission.subprocess.run",
            return_value=SimpleNamespace(stdout="12345 unexpected\n"),
        ):
            with self.assertRaisesRegex(SuiteSubmissionError, "invalid job ID"):
                submit_suite_plan(
                    plan,
                    {
                        "NVBENCH_SERVER_IMAGE": "server.sqsh",
                        "NVBENCH_CLIENT_IMAGE": "client.sqsh",
                    },
                )


if __name__ == "__main__":
    unittest.main()
