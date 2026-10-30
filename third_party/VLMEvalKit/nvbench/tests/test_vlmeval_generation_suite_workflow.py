from __future__ import annotations

import json
import unittest
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from orchestration.vlmeval_generation_suite_workflow import (
    VLMEvalGenerationSuiteWorkflowError,
    complete_vlmeval_suite_run,
    record_failed_vlmeval_suite_run,
    run_vlmeval_generation_suite,
)
from orchestration.vlmeval_generation_workflow import (
    VLMEvalGenerationWorkflowError,
)
from orchestration.resolved_run_manifest import (
    write_resolved_run_manifest,
)
from orchestration.vlmeval_suite_status_artifact import (
    VLMEvalSuiteStatusError,
    load_vlmeval_suite_status,
)
from phase8_suite_test_support import (
    completed_generation_result,
    load_json,
    resolved_suite_fixture,
    three_dataset_suite,
)

class VLMEvalGenerationSuiteWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.root.chmod(0o750)
        self.suite = three_dataset_suite()
        self.calls: list[tuple[str, Path]] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @property
    def suite_dataset_names(self) -> tuple[str, ...]:
        return tuple(item.dataset for item in self.suite.benchmarks)

    def _runner(
        self,
        _client: object,
        benchmark: object,
        work_dir: str | Path,
        _model_name: str,
        *,
        api_nproc: int,
        evaluation_judge: str,
    ):
        self.assertEqual(api_nproc, 8)
        self.assertEqual(evaluation_judge, "exact_matching")
        directory = Path(work_dir)
        dataset = benchmark.dataset
        self.calls.append((dataset, directory))
        return completed_generation_result(directory, dataset)

    def _execute(
        self,
        runner: Callable[..., object] | None = None,
        *,
        resume: bool = False,
    ):
        return run_vlmeval_generation_suite(
            object(),
            self.suite,
            self.root,
            "unit-model",
            api_nproc=8,
            resume=resume,
            generation_runner=self._runner if runner is None else runner,
        )

    def test_success_records_ordered_paths_and_counts(self) -> None:
        result = self._execute()
        expected = self.suite_dataset_names
        self.assertEqual(tuple(dataset for dataset, _ in self.calls), expected)
        self.assertEqual(tuple(item.dataset for item in result.dataset_statuses), expected)
        self.assertEqual(len({directory for _, directory in self.calls}), len(expected))
        self.assertTrue(
            all(directory.parent == self.root / "datasets" for _, directory in self.calls)
        )
        self.assertTrue(all(item.status == "completed" for item in result.dataset_statuses))
        self.assertEqual(result.suite_status_artifact, self.root / "vlmeval_suite_status.json")

        ledger = json.loads(result.suite_status_artifact.read_text(encoding="utf-8"))
        self.assertEqual(ledger["schema_version"], 2)
        self.assertEqual([item["dataset"] for item in ledger["datasets"]], list(expected))
        first = result.dataset_statuses[0]
        assert first.status_path is not None
        native = json.loads((self.root / first.status_path).read_text(encoding="utf-8"))
        record = native["datasets"][0]
        self.assertEqual(native["schema_version"], 3)
        self.assertEqual(record["observed_prediction_count"], 1)
        self.assertEqual(
            record["observed_prediction_count"],
            len(record["expected_scored_indices"]),
        )
        self.assertIsNone(record["failure"])
        self.assertFalse(any(key.endswith("_sha256") for key in record))

    def test_each_benchmark_selects_its_own_judge(self) -> None:
        benchmarks = list(self.suite.benchmarks)
        benchmarks[1] = replace(benchmarks[1], judge="unit_llm_judge")
        self.suite = replace(self.suite, benchmarks=tuple(benchmarks))
        observed: list[str] = []

        def runner(
            _client,
            benchmark,
            work_dir,
            _model_name,
            *,
            api_nproc,
            evaluation_judge,
        ):
            self.assertEqual(api_nproc, 8)
            observed.append(evaluation_judge)
            return completed_generation_result(Path(work_dir), benchmark.dataset)

        self._execute(runner)
        self.assertEqual(observed, ["exact_matching", "unit_llm_judge", "exact_matching"])

    def test_failure_stops_before_later_dataset_and_persists_progress(self) -> None:
        failed_dataset = self.suite.benchmarks[1].dataset

        def runner(
            client: object,
            benchmark: object,
            work_dir: str | Path,
            model_name: str,
            *,
            api_nproc: int,
            evaluation_judge: str,
        ):
            if benchmark.dataset == failed_dataset:
                self.calls.append((benchmark.dataset, Path(work_dir)))
                raise RuntimeError("synthetic dataset failure")
            return self._runner(
                client,
                benchmark,
                work_dir,
                model_name,
                api_nproc=api_nproc,
                evaluation_judge=evaluation_judge,
            )

        with self.assertRaisesRegex(
            VLMEvalGenerationSuiteWorkflowError, failed_dataset
        ) as raised:
            self._execute(runner)
        self.assertEqual([dataset for dataset, _ in self.calls], list(self.suite_dataset_names[:2]))
        statuses = tuple(item.status for item in raised.exception.suite_status.dataset_statuses)
        self.assertEqual(statuses, ("completed", "failed", "not_started"))

    def test_resume_skips_only_completed_dataset(self) -> None:
        failed_dataset = self.suite_dataset_names[1]

        def first_runner(
            client: object,
            benchmark: object,
            work_dir: str | Path,
            model_name: str,
            *,
            api_nproc: int,
            evaluation_judge: str,
        ):
            if benchmark.dataset == failed_dataset:
                raise RuntimeError("first-attempt failure")
            return self._runner(
                client,
                benchmark,
                work_dir,
                model_name,
                api_nproc=api_nproc,
                evaluation_judge=evaluation_judge,
            )

        with self.assertRaises(VLMEvalGenerationSuiteWorkflowError):
            self._execute(first_runner)
        self.calls.clear()
        result = self._execute(resume=True)
        self.assertEqual(
            [dataset for dataset, _ in self.calls],
            list(self.suite_dataset_names[1:]),
        )
        self.assertTrue(all(item.status == "completed" for item in result.dataset_statuses))

    def test_resume_rejects_untrusted_completed_artifacts(self) -> None:
        changes = {
            "missing": lambda entry: entry.update(
                metrics_path=f"{entry['dataset_directory']}/missing_metrics.json"
            ),
            "outside": lambda entry: entry.update(
                metrics_path="outside/vlmeval_metrics.v1.json"
            ),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                result = self._execute()
                path = result.suite_status_artifact
                payload = json.loads(path.read_text(encoding="utf-8"))
                change(payload["datasets"][0])
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.calls.clear()
                with self.assertRaises(VLMEvalSuiteStatusError):
                    self._execute(resume=True)
                self.assertEqual(self.calls, [])

        changes = (
            ("malformed", self._malform_metrics),
            ("private", self._privatize_metrics),
        )
        for name, mutate in changes:
            with self.subTest(name=name):
                self._execute()
                mutate()
                self.calls.clear()
                with self.assertRaises(VLMEvalSuiteStatusError):
                    self._execute(resume=True)
                self.assertEqual(self.calls, [])

    def _malform_metrics(self) -> None:
        status = load_vlmeval_suite_status(self.suite, self.root).dataset_statuses[0]
        assert status.metrics_path is not None
        (self.root / status.metrics_path).write_text("{}\n", encoding="utf-8")

    def _privatize_metrics(self) -> None:
        status = load_vlmeval_suite_status(self.suite, self.root).dataset_statuses[0]
        assert status.metrics_path is not None
        (self.root / status.metrics_path).chmod(0o600)

    def test_failure_status_preserves_one_bounded_sanitized_direct_cause(self) -> None:
        failed = self.suite.benchmarks[0]

        def runner(
            _client: object,
            _benchmark: object,
            work_dir: str | Path,
            _model_name: str,
            *,
            api_nproc: int,
            evaluation_judge: str,
        ):
            self.assertEqual(api_nproc, 8)
            self.assertEqual(evaluation_judge, "exact_matching")
            detail = f"request failed\ninside {work_dir}/private " + ("x" * 600)
            try:
                raise RuntimeError(detail)
            except RuntimeError as cause:
                raise VLMEvalGenerationWorkflowError(
                    "upstream inference job failed"
                ) from cause

        with self.assertRaises(VLMEvalGenerationSuiteWorkflowError) as raised:
            self._execute(runner)
        status = raised.exception.suite_status
        assert status is not None
        failed_record = status.dataset_statuses[0]
        assert failed_record.status_path is not None
        payload = json.loads((self.root / failed_record.status_path).read_text())
        failure = payload["datasets"][0]["failure"]
        self.assertEqual(failure["type"], "VLMEvalGenerationWorkflowError")
        self.assertEqual(failure["message"], "upstream inference job failed")
        self.assertEqual(failure["cause_type"], "RuntimeError")
        self.assertIn("<run_directory>/private", failure["cause_message"])
        self.assertNotIn(str(self.root), failure["cause_message"])
        self.assertLessEqual(len(failure["cause_message"]), 500)
        self.assertEqual(failed_record.dataset, failed.dataset)


class VLMEvalSuiteResultRecordingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture, self.resolved = resolved_suite_fixture()
        self.root = Path(self.resolved.paths.run_directory)
        self.suite = self.resolved.configured.benchmark_suite
        assert self.suite is not None
        write_resolved_run_manifest(self.resolved, slurm_job_id="12345")

    def tearDown(self) -> None:
        self.fixture.close()

    @staticmethod
    def _runner(
        _client: object,
        benchmark: object,
        work_dir: str | Path,
        _model_name: str,
        *,
        api_nproc: int,
        evaluation_judge: str,
    ):
        assert api_nproc == 8
        assert evaluation_judge == "exact_matching"
        return completed_generation_result(Path(work_dir), benchmark.dataset)

    def _suite_result(self):
        return run_vlmeval_generation_suite(
            object(),
            self.suite,
            self.root,
            "unit-model",
            api_nproc=8,
            resume=False,
            generation_runner=self._runner,
        )

    def test_completion_aggregates_metrics_and_terminal_paths(self) -> None:
        result = self._suite_result()
        manifest_path, suite_status_path, metrics_path = complete_vlmeval_suite_run(
            self.resolved, "12345", result
        )
        manifest = load_json(manifest_path)
        metrics = load_json(metrics_path)
        terminal = manifest["dataset_terminal_statuses"]

        self.assertEqual(manifest["run_status"], "completed")
        self.assertEqual(suite_status_path, result.suite_status_artifact)
        self.assertEqual(len(terminal), 3)
        self.assertEqual(len(metrics["metrics"]), 3)
        self.assertTrue(
            all(item["source_file"].startswith("datasets/") for item in metrics["metrics"])
        )
        self.assertEqual(
            [item["dataset"] for item in terminal],
            sorted(item.dataset for item in result.dataset_statuses),
        )
        self.assertTrue(
            all(set(item) == {"dataset", "status", "status_path"} for item in terminal)
        )
        self.assertTrue(all(item["status"] == "completed" for item in terminal))

    def test_dataset_failure_manifest_preserves_completed_and_failed_entries(self) -> None:
        failed = self.suite.benchmarks[1].dataset

        def runner(
            client: object,
            benchmark: object,
            work_dir: str | Path,
            model_name: str,
            *,
            api_nproc: int,
            evaluation_judge: str,
        ):
            if benchmark.dataset == failed:
                raise RuntimeError("synthetic failure")
            return self._runner(
                client,
                benchmark,
                work_dir,
                model_name,
                api_nproc=api_nproc,
                evaluation_judge=evaluation_judge,
            )

        with self.assertRaises(VLMEvalGenerationSuiteWorkflowError) as raised:
            run_vlmeval_generation_suite(
                object(),
                self.suite,
                self.root,
                "unit-model",
                api_nproc=8,
                resume=False,
                generation_runner=runner,
            )
        manifest_path = record_failed_vlmeval_suite_run(
            self.resolved,
            "12345",
            raised.exception,
            "vlmeval_generation",
            None,
        )
        terminal = load_json(manifest_path)["dataset_terminal_statuses"]
        self.assertEqual(
            {item["dataset"]: item["status"] for item in terminal},
            {self.suite.benchmarks[0].dataset: "completed", failed: "failed"},
        )
        self.assertTrue(
            all(set(item) == {"dataset", "status", "status_path"} for item in terminal)
        )

    def test_cleanup_failure_adds_suite_terminal_without_relabeling_datasets(self) -> None:
        result = self._suite_result()
        manifest_path = record_failed_vlmeval_suite_run(
            self.resolved,
            "12345",
            RuntimeError("cleanup failed"),
            "server_cleanup",
            result,
        )
        terminal = load_json(manifest_path)["dataset_terminal_statuses"]
        self.assertEqual(sum(item["status"] == "completed" for item in terminal), 3)
        failed = [item for item in terminal if item["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertTrue(failed[0]["dataset"].startswith("suite:"))
        self.assertEqual(set(failed[0]), {"dataset", "status", "status_path"})


if __name__ == "__main__":
    unittest.main()
