"""Static contract for the bundled ScreenSpot four-shot profile."""

from __future__ import annotations

import unittest
from pathlib import Path

from orchestration.eval_component_config_loading import load_benchmark_suite
from orchestration.vlmeval_result_contract import validate_benchmark_result_contract

NVBENCH = Path(__file__).resolve().parents[1]
SUITES = NVBENCH / "configs/suites"


class ScreenSpotProfilesTest(unittest.TestCase):
    def test_mobile_fourshot_suite_pins_diverse_native_demonstrations(self) -> None:
        suite = load_benchmark_suite(SUITES / "n4_v6.yaml")
        benchmark = suite.benchmarks[-1]
        validate_benchmark_result_contract(benchmark)
        self.assertEqual(benchmark.dataset, "ScreenSpot_Mobile")
        self.assertEqual(benchmark.upstream_class, "ScreenSpot")
        self.assertEqual(benchmark.num_shot, 4)
        self.assertEqual(benchmark.demonstration_type, "benchmark")
        self.assertEqual(benchmark.demonstration_indices, (94, 9, 24, 66))
        self.assertFalse(benchmark.prepend_reasoning_cue)
        self.assertEqual(benchmark.demonstration_answer_format, "screenspot_click")
        self.assertTrue(benchmark.include_demonstration_images)
        self.assertEqual(benchmark.server_context_length, 16384)
        self.assertEqual(benchmark.result_reader, "screenspot_score_json")
        self.assertEqual(benchmark.expected_result_suffix, "_score.json")
        self.assertEqual(benchmark.generation.chat_template, "minimal")
        self.assertEqual(benchmark.generation.max_new_tokens, 128)
        self.assertEqual(benchmark.resource, "n1")


if __name__ == "__main__":
    unittest.main()
