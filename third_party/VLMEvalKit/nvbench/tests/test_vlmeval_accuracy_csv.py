from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from orchestration.eval_run_config_loading import (
    load_configured_eval_run,
)
from orchestration.vlmeval_result_contract import (
    VLMEvalResultContractError,
    read_normalized_metrics,
)
from orchestration.vlmeval_result_readers import (
    VLMEvalAccuracyCSVError,
    read_vlmeval_accuracy_csv,
)


NVBENCH = Path(__file__).resolve().parents[1]
FIXTURES = NVBENCH / "tests" / "fixtures" / "vlmeval_accuracy_csv"
RUN_CONFIG = (
    NVBENCH
    / "configs"
    / "run_configs"
    / "n4_v6_example.yaml"
)
EXPECTED_COUNTS = {
    "AI2D_TEST": 16,
    "CV-Bench-2D": 3,
    "ChartQA_TEST": 3,
    "DocVQA_VAL": 2,
    "GQA_TestDev_Balanced": 6,
    "InfoVQA_VAL": 2,
    "MMMU_DEV_VAL": 74,
    "MMMU_Pro_10c": 31,
    "MMStar": 7,
    "RealWorldQA": 1,
    "SEEDBench_IMG": 10,
    "TextVQA_VAL": 1,
}


class VLMEvalAccuracyCSVTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configured = load_configured_eval_run(RUN_CONFIG)
        assert configured.benchmark_suite is not None
        cls.benchmarks = {
            item.dataset: item for item in configured.benchmark_suite.benchmarks
        }

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.root.chmod(0o750)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _install(self, dataset: str) -> Path:
        name = f"megatron_vlm_{dataset}_acc.csv"
        destination = self.root / name
        shutil.copyfile(FIXTURES / name, destination)
        destination.chmod(0o640)
        return destination

    def _write(self, content: bytes) -> Path:
        path = self.root / "model_Dataset_acc.csv"
        path.write_bytes(content)
        path.chmod(0o640)
        return path

    def test_all_twelve_historical_native_artifacts_normalize_without_mutation(self) -> None:
        self.assertTrue(set(EXPECTED_COUNTS).issubset(self.benchmarks))
        for dataset, expected_count in EXPECTED_COUNTS.items():
            with self.subTest(dataset=dataset):
                benchmark = self.benchmarks[dataset]
                score = self._install(dataset)
                before = score.read_bytes()
                before_digest = hashlib.sha256(before).hexdigest()
                metrics = read_normalized_metrics(benchmark, score, self.root)
                self.assertEqual(len(metrics), expected_count)
                self.assertTrue(all(item.dataset == dataset for item in metrics))
                self.assertTrue(
                    all(item.unit == benchmark.result_metric_unit for item in metrics)
                )
                self.assertTrue(all(item.source_file == score.name for item in metrics))
                self.assertEqual(score.read_bytes(), before)
                self.assertEqual(hashlib.sha256(score.read_bytes()).hexdigest(), before_digest)

    def test_mmmu_split_rows_are_qualified_and_traceable(self) -> None:
        benchmark = self.benchmarks["MMMU_DEV_VAL"]
        metrics = read_normalized_metrics(
            benchmark, self._install(benchmark.dataset), self.root
        )
        self.assertEqual(metrics[0].metric, "validation_overall")
        self.assertEqual(metrics[0].source_field, "validation:Overall")
        dev = next(item for item in metrics if item.metric == "dev_overall")
        self.assertEqual(dev.source_field, "dev:Overall")
        self.assertEqual(dev.unit, "fraction")

    def test_single_split_row_does_not_leak_the_placeholder_label(self) -> None:
        benchmark = self.benchmarks["AI2D_TEST"]
        metrics = read_normalized_metrics(
            benchmark, self._install(benchmark.dataset), self.root
        )
        self.assertEqual(metrics[0].metric, "overall")
        self.assertEqual(metrics[0].source_field, "Overall")
        self.assertNotIn("none", {item.metric for item in metrics})

    def test_reader_rejects_malformed_or_out_of_range_tables(self) -> None:
        cases = (
            b"Overall,Overall\n0.5,0.5\n",
            b"Overall\n",
            b"Overall,Other\n0.5\n",
            b"Overall\nnot-a-number\n",
            b"Overall\nnan\n",
            b"Overall\n1.1\n",
            b"split,Overall\nnone,0.5\nnone,0.4\n",
            b"split,Overall\n,0.5\n",
            b"\xff\n",
        )
        for content in cases:
            with self.subTest(content=content), self.assertRaises(VLMEvalAccuracyCSVError):
                read_vlmeval_accuracy_csv(
                    self._write(content),
                    self.root,
                    dataset="Dataset",
                    unit="fraction",
                )
        with self.assertRaises(VLMEvalAccuracyCSVError):
            read_vlmeval_accuracy_csv(
                self._write(b"Overall\n1\n"),
                self.root,
                dataset="Dataset",
                unit="reader_defined",
            )

    def test_dispatch_rejects_identifier_collisions(self) -> None:
        benchmark = self.benchmarks["ChartQA_TEST"]
        score = self._write(b"A-B,A B\n1,2\n")
        with self.assertRaises(VLMEvalResultContractError):
            read_normalized_metrics(benchmark, score, self.root)


if __name__ == "__main__":
    unittest.main()
