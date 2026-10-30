from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from orchestration.megatron_dynamic_server_vlm_fewshot_run import (
    MegatronDynamicServerVlmFewshotRunError,
    execute_megatron_dynamic_server_vlm_fewshot_client,
)
from orchestration.resolved_run_manifest import MANIFEST_FILENAME
from phase5_orchestration_test_support import (
    client_dependencies,
    completed_result,
    prepare_client_side_state,
    prepared_run_fixture,
)


class MegatronDynamicServerVlmFewshotRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture, self.prepared = prepared_run_fixture()

    def tearDown(self) -> None:
        self.fixture.close()

    @property
    def run_directory(self) -> Path:
        return Path(self.prepared.resolved.paths.run_directory)

    def execute(self, dependencies, *, endpoint: str | None = None):
        return execute_megatron_dynamic_server_vlm_fewshot_client(
            self.prepared.resolved,
            "12345",
            expected_endpoint=endpoint or self.prepared.launch.endpoint,
            expected_health_endpoint=self.prepared.launch.health_endpoint,
            dependencies=dependencies,
        )

    def test_success_waits_for_cleanup_before_completing_manifest(self) -> None:
        prepare_client_side_state(self.prepared)
        events: list[str] = []
        previous = os.environ.get("LMUData")
        os.environ["LMUData"] = "sentinel-before-run"

        def generation(
            _client,
            benchmark,
            work_dir,
            model_name,
            *,
            api_nproc,
            evaluation_judge,
        ):
            self.assertEqual(benchmark, self.prepared.resolved.configured.benchmark)
            self.assertEqual(Path(work_dir), self.run_directory)
            self.assertEqual(model_name, self.prepared.resolved.configured.model.name)
            self.assertEqual(api_nproc, 8)
            self.assertEqual(evaluation_judge, "exact_matching")
            self.assertEqual(os.environ["LMUData"], self.prepared.resolved.paths.cache_root)
            return completed_result(self.run_directory)

        try:
            result = self.execute(client_dependencies(events, generation))
            self.assertEqual(os.environ.get("LMUData"), "sentinel-before-run")
        finally:
            if previous is None:
                os.environ.pop("LMUData", None)
            else:
                os.environ["LMUData"] = previous

        self.assertEqual(
            events,
            [
                "prepared",
                "manifest",
                "health",
                "client",
                "preflight",
                "generation",
                "shutdown",
                "terminal",
            ],
        )
        manifest = json.loads(result.manifest_artifact.read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_status"], "completed")
        self.assertEqual(manifest["slurm_job_id"], "12345")
        self.assertEqual(result.manifest_artifact, self.run_directory / MANIFEST_FILENAME)

    def test_failed_server_cleanup_prevents_completed_manifest(self) -> None:
        prepare_client_side_state(self.prepared)
        events: list[str] = []
        with self.assertRaisesRegex(
            MegatronDynamicServerVlmFewshotRunError, "server_cleanup"
        ):
            self.execute(
                client_dependencies(
                    events,
                    lambda *_args, **_kwargs: completed_result(self.run_directory),
                    terminal_status="failed",
                )
            )
        manifest = json.loads(
            (self.run_directory / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["run_status"], "failed")
        self.assertEqual(events[-2:], ["shutdown", "terminal"])

    def test_endpoint_mismatch_still_requests_server_shutdown(self) -> None:
        prepare_client_side_state(self.prepared)
        events: list[str] = []
        with self.assertRaisesRegex(
            MegatronDynamicServerVlmFewshotRunError, "server_prepared"
        ):
            self.execute(
                client_dependencies(
                    events,
                    lambda *_args, **_kwargs: self.fail("generation was called"),
                ),
                endpoint="http://127.0.0.1:9000/v1/chat/completions",
            )
        self.assertEqual(events, ["prepared", "shutdown", "terminal"])

    def test_wrong_resolved_record_fails_before_dependency_calls(self) -> None:
        events: list[str] = []
        with self.assertRaisesRegex(
            MegatronDynamicServerVlmFewshotRunError, "wrong type"
        ):
            execute_megatron_dynamic_server_vlm_fewshot_client(
                object(),  # type: ignore[arg-type]
                "12345",
                expected_endpoint=self.prepared.launch.endpoint,
                expected_health_endpoint=self.prepared.launch.health_endpoint,
                dependencies=client_dependencies(
                    events,
                    lambda *_args, **_kwargs: self.fail("generation was called"),
                ),
            )
        self.assertEqual(events, [])
        self.assertFalse(self.run_directory.exists())


if __name__ == "__main__":
    unittest.main()
