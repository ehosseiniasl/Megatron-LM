"""Tests for labeled VLMEvalKit primary-metric CSV readers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orchestration.vlmeval_result_readers.vlmeval_primary_metric_csv import (
    VLMEvalPrimaryMetricCSVError,
    read_vlmeval_primary_metric_csv,
)


def score_file(root: Path, content: str) -> Path:
    path = root / "score.csv"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o640)
    return path


class VLMEvalPrimaryMetricCSVTest(unittest.TestCase):
    def test_reads_chartqapro_mathvista_and_worldvqa_primary_rows(self) -> None:
        cases = (
            (
                "chartqapro_primary_csv",
                "fraction",
                "0,1\nConversational,0.5\nOverall,0.625\n",
                {"overall_accuracy": 0.625},
            ),
            (
                "mathvista_primary_csv",
                "percent",
                "Task&Skill,tot,prefetch,hit,prefetch_rate,acc\n"
                "Overall,1000,900,420,90,42\ngeometry,200,180,70,90,35\n",
                {"overall_accuracy": 42.0},
            ),
            (
                "worldvqa_primary_csv",
                "fraction",
                "Metric,Total,Attempted,Correct,Accuracy,CGA,AttemptRate,F-score\n"
                "hard,100,80,40,0.4,0.5,0.8,0.444444\n"
                "Overall,200,150,90,0.45,0.6,0.75,0.514286\n",
                {
                    "accuracy": 0.45,
                    "conditional_generation_accuracy": 0.6,
                    "attempt_rate": 0.75,
                    "f_score": 0.514286,
                },
            ),
        )
        for profile, unit, content, expected in cases:
            with (
                self.subTest(profile=profile),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                metrics = read_vlmeval_primary_metric_csv(
                    score_file(root, content),
                    root,
                    dataset="unit_dataset",
                    unit=unit,
                    profile=profile,
                )
                self.assertEqual(
                    {item.metric: item.value for item in metrics}, expected
                )
                self.assertTrue(all(item.unit == unit for item in metrics))

    def test_rejects_missing_duplicate_and_out_of_range_primary_metrics(self) -> None:
        cases = (
            "0,1\ncategory,0.5\n",
            "0,1\nOverall,0.5\nOverall,0.6\n",
            "0,1\nOverall,1.1\n",
        )
        for content in cases:
            with (
                self.subTest(content=content),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                with self.assertRaises(VLMEvalPrimaryMetricCSVError):
                    read_vlmeval_primary_metric_csv(
                        score_file(root, content),
                        root,
                        dataset="unit_dataset",
                        unit="fraction",
                        profile="chartqapro_primary_csv",
                    )


if __name__ == "__main__":
    unittest.main()
