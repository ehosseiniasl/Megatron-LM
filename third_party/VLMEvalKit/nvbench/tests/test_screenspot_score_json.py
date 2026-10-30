from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from orchestration.configured_eval_run import BenchmarkConfig, EvalConfigError
from orchestration.vlmeval_result_contract import (
    VLMEvalResultContractError,
    read_normalized_metrics,
    validate_benchmark_result_contract,
)
from phase3_generation_test_support import benchmark


def screenspot_benchmark() -> BenchmarkConfig:
    return replace(
        benchmark(upstream_class="ScreenSpot", dataset="ScreenSpot_Mobile"),
        result_reader="screenspot_score_json",
        expected_result_suffix="_score.json",
        result_metric_unit="reader_defined",
    )


class ScreenSpotScoreJSONTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.root.chmod(0o750)
        self.score = self.root / "model_ScreenSpot_Mobile_score.json"
        self.benchmark = screenspot_benchmark()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, payload: object) -> None:
        self.score.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        self.score.chmod(0o640)

    def valid_payload(self) -> dict[str, object]:
        return {
            "text:cnt": 60,
            "icon:cnt": 40,
            "Overall_Accuracy": 35.0,
            "Format_Err_Rate": 4.0,
            "Text_Accuracy": 45.0,
            "Icon_Accuracy": 20.0,
        }

    def test_contract_and_normalized_metrics(self) -> None:
        validate_benchmark_result_contract(self.benchmark)
        self.write(self.valid_payload())
        metrics = read_normalized_metrics(self.benchmark, self.score, self.root)
        self.assertEqual(
            tuple(metric.metric for metric in metrics),
            (
                "text_count",
                "icon_count",
                "overall_accuracy",
                "format_error_rate",
                "text_accuracy",
                "icon_accuracy",
            ),
        )
        self.assertEqual(tuple(metric.value for metric in metrics), (60, 40, 35.0, 4.0, 45.0, 20.0))
        self.assertEqual(
            tuple(metric.unit for metric in metrics), ("count", "count") + ("percent",) * 4
        )

    def test_wrong_dataset_class_or_profile_fails(self) -> None:
        for changes in (
            {"dataset": "ScreenSpot_Pro_Development"},
            {"upstream_class": "ScreenSpot_Pro"},
            {"result_reader": "future_reader"},
            {"expected_result_suffix": "_acc.csv"},
        ):
            with self.subTest(changes=changes), self.assertRaises(EvalConfigError):
                validate_benchmark_result_contract(replace(self.benchmark, **changes))

    def test_malformed_or_inconsistent_scores_fail_closed(self) -> None:
        cases = []
        missing = self.valid_payload()
        missing.pop("Icon_Accuracy")
        cases.append(missing)
        bad_count = self.valid_payload()
        bad_count["text:cnt"] = -1
        cases.append(bad_count)
        bad_percent = self.valid_payload()
        bad_percent["Format_Err_Rate"] = 101
        cases.append(bad_percent)
        inconsistent = self.valid_payload()
        inconsistent["Overall_Accuracy"] = 36
        cases.append(inconsistent)
        for payload in cases:
            with self.subTest(payload=payload):
                self.write(payload)
                with self.assertRaises(VLMEvalResultContractError):
                    read_normalized_metrics(self.benchmark, self.score, self.root)


if __name__ == "__main__":
    unittest.main()
