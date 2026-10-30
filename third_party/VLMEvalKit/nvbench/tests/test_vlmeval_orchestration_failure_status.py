from __future__ import annotations

import json
import unittest
from pathlib import Path

from orchestration.megatron_dynamic_server_vlm_fewshot_run import (
    MegatronDynamicServerVlmFewshotRunError,
    execute_megatron_dynamic_server_vlm_fewshot_client,
)
from orchestration.vlmeval_result_contract import (
    GeneratedAnswerResultFinalizationError,
)
from orchestration.vlmeval_generation_workflow import (
    VLMEvalGenerationWorkflowError,
)
from phase5_orchestration_test_support import (
    client_dependencies,
    completed_result,
    prepare_client_side_state,
    prepared_run_fixture,
    row_partitions,
    shared_file,
)


_STATUS_KEYS = {
    "dataset",
    "status",
    "expected_scored_indices",
    "observed_prediction_count",
    "demonstration_and_scored_row_partitions_path",
    "prediction_path",
    "score_path",
    "metrics_path",
    "failure",
}


class VLMEvalOrchestrationFailureStatusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture, self.prepared = prepared_run_fixture()
        prepare_client_side_state(self.prepared)

    def tearDown(self) -> None:
        self.fixture.close()

    @property
    def run_directory(self) -> Path:
        return Path(self.prepared.resolved.paths.run_directory)

    def _execute(
        self,
        events: list[str],
        generation,
        *,
        terminal_status: str = "clean",
        health_waiter=None,
    ):
        return execute_megatron_dynamic_server_vlm_fewshot_client(
            self.prepared.resolved,
            "12345",
            expected_endpoint=self.prepared.launch.endpoint,
            expected_health_endpoint=self.prepared.launch.health_endpoint,
            dependencies=client_dependencies(
                events,
                generation,
                terminal_status=terminal_status,
                health_waiter=health_waiter,
            ),
        )

    def _status(self) -> tuple[dict[str, object], dict[str, object]]:
        manifest = json.loads(
            (self.run_directory / "megatron_dynamic_server_vlm_eval_manifest.json")
            .read_text(encoding="utf-8")
        )
        terminal = manifest["dataset_terminal_statuses"][0]
        status_path = self.run_directory / terminal["status_path"]
        status = json.loads(status_path.read_text(encoding="utf-8"))
        return manifest, status

    def _write_native_status(self, partition_path: str) -> None:
        record = {
            "dataset": "OCRBench",
            "status": "failed",
            "expected_scored_indices": [17],
            "observed_prediction_count": 1,
            "demonstration_and_scored_row_partitions_path": partition_path,
            "prediction_path": "prediction.xlsx",
            "score_path": None,
            "metrics_path": None,
            "failure": {"type": "ValueError", "message": "bad score"},
        }
        shared_file(
            self.run_directory / "vlmeval_dataset_status.json",
            json.dumps({"schema_version": 3, "datasets": [record]}).encode("utf-8"),
        )

    def test_health_failure_has_null_preselection_partition_fields(self) -> None:
        events: list[str] = []

        def fail_health(*_args: object) -> None:
            raise RuntimeError("health unavailable")

        with self.assertRaisesRegex(MegatronDynamicServerVlmFewshotRunError, "server_health"):
            self._execute(
                events,
                lambda *_args, **_kwargs: self.fail("generation was called"),
                health_waiter=fail_health,
            )
        manifest, payload = self._status()
        record = payload["datasets"][0]
        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(set(record), _STATUS_KEYS)
        self.assertEqual(manifest["run_status"], "failed")
        self.assertEqual(record["failure"]["stage"], "server_health")
        self.assertIsNone(record["demonstration_and_scored_row_partitions_path"])
        self.assertEqual(events[-2:], ["shutdown", "terminal"])

    def test_workflow_failure_carries_exact_partition_reference(self) -> None:
        events: list[str] = []

        def generation(*_args, **_kwargs):
            partitions = row_partitions(self.run_directory)
            raise VLMEvalGenerationWorkflowError(
                "prediction path failed",
                row_partitions=partitions,
            )

        with self.assertRaisesRegex(MegatronDynamicServerVlmFewshotRunError, "vlmeval"):
            self._execute(events, generation)
        _, payload = self._status()
        record = payload["datasets"][0]
        self.assertEqual(
            record["demonstration_and_scored_row_partitions_path"],
            "demonstration_and_scored_row_partitions.json",
        )

    def test_cleanup_failure_derives_partition_from_completed_result(self) -> None:
        events: list[str] = []

        def generation(*_args, **_kwargs):
            return completed_result(self.run_directory)

        with self.assertRaisesRegex(MegatronDynamicServerVlmFewshotRunError, "server_cleanup"):
            self._execute(events, generation, terminal_status="failed")
        _, payload = self._status()
        record = payload["datasets"][0]
        self.assertEqual(record["expected_scored_indices"], [17])
        self.assertEqual(record["prediction_path"], "prediction.xlsx")
        self.assertEqual(
            record["demonstration_and_scored_row_partitions_path"],
            "demonstration_and_scored_row_partitions.json",
        )

    def test_native_finalization_failure_status_is_retained_and_bound(self) -> None:
        events: list[str] = []

        def generation(*_args, **_kwargs):
            partitions = row_partitions(self.run_directory)
            self._write_native_status(partitions.artifact.name)
            raise GeneratedAnswerResultFinalizationError("native finalization failed")

        with self.assertRaises(MegatronDynamicServerVlmFewshotRunError):
            self._execute(events, generation)
        _, payload = self._status()
        self.assertEqual(payload["datasets"][0]["prediction_path"], "prediction.xlsx")
        self.assertNotIn("stage", payload["datasets"][0]["failure"])

    def test_invalid_native_status_falls_back_to_null_partition_provenance(self) -> None:
        events: list[str] = []

        def generation(*_args, **_kwargs):
            partitions = row_partitions(self.run_directory)
            partitions.artifact.chmod(0o600)
            self._write_native_status(partitions.artifact.name)
            raise GeneratedAnswerResultFinalizationError("invalid partition")

        with self.assertRaises(MegatronDynamicServerVlmFewshotRunError):
            self._execute(events, generation)
        _, payload = self._status()
        record = payload["datasets"][0]
        self.assertEqual(record["failure"]["stage"], "vlmeval_generation")
        self.assertIsNone(record["demonstration_and_scored_row_partitions_path"])


if __name__ == "__main__":
    unittest.main()
