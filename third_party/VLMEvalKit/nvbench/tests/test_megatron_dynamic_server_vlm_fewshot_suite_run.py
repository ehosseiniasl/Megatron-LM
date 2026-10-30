from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from orchestration.megatron_dynamic_server_vlm_fewshot_run import (
    execute_megatron_dynamic_server_vlm_fewshot_client,
)
from orchestration.vlmeval_generation_suite_workflow import (
    run_vlmeval_generation_suite,
)
from phase5_orchestration_test_support import (
    client_dependencies,
    prepare_client_side_state,
    prepared_run_fixture,
)
from phase8_suite_test_support import completed_generation_result, three_dataset_suite


class MegatronDynamicServerVlmFewshotSuiteRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture, prepared = prepared_run_fixture()
        configured = replace(
            prepared.resolved.configured,
            benchmark=None,
            benchmark_suite=three_dataset_suite(),
        )
        self.prepared = replace(
            prepared,
            resolved=replace(prepared.resolved, configured=configured),
        )

    def tearDown(self) -> None:
        self.fixture.close()

    def test_client_reuses_preflight_then_runs_and_finalizes_suite(self) -> None:
        prepare_client_side_state(self.prepared)
        events: list[str] = []

        def suite_runner(
            client,
            suite,
            work_dir,
            model_name,
            *,
            api_nproc,
            resume,
        ):
            self.assertFalse(resume)

            def dataset_runner(
                _client,
                benchmark,
                dataset_dir,
                _model_name,
                *,
                api_nproc,
                evaluation_judge,
            ):
                self.assertEqual(api_nproc, 8)
                self.assertEqual(evaluation_judge, "exact_matching")
                return completed_generation_result(Path(dataset_dir), benchmark.dataset)

            return run_vlmeval_generation_suite(
                client,
                suite,
                work_dir,
                model_name,
                api_nproc=api_nproc,
                resume=resume,
                generation_runner=dataset_runner,
            )

        result = execute_megatron_dynamic_server_vlm_fewshot_client(
            self.prepared.resolved,
            "12345",
            expected_endpoint=self.prepared.launch.endpoint,
            expected_health_endpoint=self.prepared.launch.health_endpoint,
            dependencies=client_dependencies(
                events,
                lambda *_args, **_kwargs: self.fail("single generation was called"),
                suite_generation_runner=suite_runner,
            ),
        )
        self.assertEqual(events.count("suite_generation"), 1)
        self.assertNotIn("generation", events)
        manifest = json.loads(result.manifest_artifact.read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_status"], "completed")
        self.assertEqual(len(manifest["dataset_terminal_statuses"]), 3)
        self.assertEqual(result.dataset_status_artifact.name, "vlmeval_suite_status.json")
        self.assertEqual(result.metrics_artifact.name, "vlmeval_suite_metrics.v1.json")


if __name__ == "__main__":
    unittest.main()
