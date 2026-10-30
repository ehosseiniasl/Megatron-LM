"""Static contracts for the compact N4 VLM evaluation configs."""

from __future__ import annotations

import unittest
from pathlib import Path

from orchestration.eval_run_config_loading import load_configured_eval_run

NVBENCH = Path(__file__).resolve().parents[1]
CONFIGS = NVBENCH / "configs"
RUN = CONFIGS / "run_configs/n4_v6_example.yaml"
EXPECTED_DATASETS = (
    "OCRBench",
    "DocVQA_VAL",
    "ChartQA_TEST",
    "ChartQAPro",
    "RealWorldQA",
    "WorldVQA",
    "AI2D_TEST",
    "SEEDBench_IMG",
    "GQA_TestDev_Balanced",
    "MMStar",
    "InfoVQA_VAL",
    "MMMU_Pro_10c",
    "MMMU_Pro_10c_COT",
    "MathVista_MINI",
    "TextVQA_VAL",
    "CV-Bench-2D",
    "MMMU_DEV_VAL",
    "ScreenSpot_Mobile",
)


class N4PR431ProfilesTest(unittest.TestCase):
    def test_model_config_contains_only_serving_overrides(self) -> None:
        config = load_configured_eval_run(RUN)
        self.assertEqual(config.run.gpus_per_node, 4)
        self.assertEqual(config.run.api_nproc, 16)
        self.assertEqual(config.run.expert_model_parallel_size, 4)
        cli = dict(config.model.server.cli_overrides)
        self.assertEqual(cli["tokenizer-prompt-format"], "mistral")
        self.assertEqual(cli["vlm-language-transformer-impl"], "inference_optimized")
        self.assertEqual(cli["vlm-image-preprocess-preset"], "mistral-crawl-eval")
        self.assertEqual(cli["moe-router-dtype"], "fp32")
        self.assertEqual(cli["inference-moe-token-dispatcher-type"], "nccl")
        self.assertEqual(len(cli), 5)
        self.assertIsNone(config.benchmark)
        self.assertIsNotNone(config.benchmark_suite)

    def test_suite_matches_the_n4_task_contract(self) -> None:
        config = load_configured_eval_run(RUN)
        suite = config.benchmark_suite
        self.assertIsNotNone(suite)
        self.assertEqual(
            tuple(benchmark.dataset for benchmark in suite.benchmarks), EXPECTED_DATASETS
        )
        cot = next(
            benchmark for benchmark in suite.benchmarks if benchmark.dataset == "MMMU_Pro_10c_COT"
        )
        direct = next(
            benchmark for benchmark in suite.benchmarks if benchmark.dataset == "MMMU_Pro_10c"
        )
        screenspot = next(
            benchmark for benchmark in suite.benchmarks if benchmark.dataset == "ScreenSpot_Mobile"
        )
        ordinary = tuple(
            benchmark
            for benchmark in suite.benchmarks
            if benchmark is not cot and benchmark is not screenspot
        )
        self.assertTrue(
            all(
                benchmark.num_shot == 2
                and benchmark.demonstration_type == "benchmark"
                and benchmark.demonstration_answer_format in {"label", "raw"}
                and benchmark.include_demonstration_images
                and not benchmark.prepend_reasoning_cue
                and benchmark.scored_limit is None
                and benchmark.generation.image_detail == "low"
                and benchmark.generation.max_new_tokens == 64
                for benchmark in ordinary
            )
        )
        self.assertEqual(cot.demonstration_type, "reasoning")
        self.assertEqual(cot.demonstration_answer_format, "raw")
        self.assertTrue(cot.include_demonstration_images)
        self.assertTrue(cot.prepend_reasoning_cue)
        self.assertIsNone(cot.scored_limit)
        self.assertEqual(cot.generation.max_new_tokens, 512)
        self.assertEqual(cot.generation.stop, direct.generation.stop)
        self.assertEqual(screenspot.num_shot, 4)
        self.assertEqual(screenspot.demonstration_type, "benchmark")
        self.assertEqual(screenspot.demonstration_answer_format, "screenspot_click_normalized")
        self.assertEqual(
            screenspot.generation.chat_template,
            "assets/screenspot_shared_gui_end_marker_template.jinja",
        )
        self.assertEqual(screenspot.generation.stop, ("__________",))
        self.assertEqual(screenspot.demonstration_indices, (94, 9, 24, 66))
        self.assertTrue(screenspot.include_demonstration_images)
        self.assertFalse(screenspot.prepend_reasoning_cue)
        self.assertEqual(screenspot.generation.max_new_tokens, 128)
        self.assertEqual(screenspot.server_context_length, 16384)


if __name__ == "__main__":
    unittest.main()
