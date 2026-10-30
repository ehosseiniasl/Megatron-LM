"""Essential tests for the compact evaluation configuration."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from orchestration.benchmark_array_selection import bind_benchmark_array_selection
from orchestration.configured_eval_run import ConditionalPerplexityBenchmarkConfig, EvalConfigError
from orchestration.eval_run_config_loading import load_configured_eval_run

NVBENCH = Path(__file__).resolve().parents[1]
CONFIGS = NVBENCH / "configs"
RUN_RELATIVE = Path("run_configs/n4_v6_example.yaml")


class EvalRunConfigLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "configs"
        shutil.copytree(CONFIGS, self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @property
    def run_path(self) -> Path:
        return self.root / RUN_RELATIVE

    def replace(self, relative: str | Path, old: str, new: str) -> None:
        path = self.root / relative
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def assert_load_error(self) -> None:
        with self.assertRaises(EvalConfigError):
            load_configured_eval_run(self.run_path)

    def test_example_loads_complete_benchmark_suite(self) -> None:
        config = load_configured_eval_run(self.run_path)
        self.assertEqual(config.run.expert_model_parallel_size, 4)
        self.assertEqual(len(config.benchmark_suite.benchmarks), 18)
        self.assertEqual(config.benchmark_suite.benchmarks[0].dataset, "OCRBench")
        self.assertTrue(
            all(
                item.num_shot == 2
                for item in config.benchmark_suite.benchmarks
                if item.dataset != "ScreenSpot_Mobile"
            )
        )
        self.assertTrue(
            all(
                item.generation.chat_template
                == (
                    "assets/screenspot_shared_gui_end_marker_template.jinja"
                    if item.dataset == "ScreenSpot_Mobile"
                    else "minimal"
                )
                for item in config.benchmark_suite.benchmarks
            )
        )
        self.assertFalse(config.checkpoint.trusted_standalone_common_state)
        self.assertEqual(config.model.tokenizer.repository, "nvidia/n4-v6-262k")
        self.assertEqual(
            config.model.tokenizer.revision, "8f5dd326777cbdd5c018d92ba35f473d971e121d"
        )
        self.assertIsNone(config.model.server.context_length)
        mmmu_pro = next(
            item for item in config.benchmark_suite.benchmarks if item.dataset == "MMMU_Pro_10c"
        )
        self.assertEqual(mmmu_pro.server_context_length, 16384)
        cvbench = next(
            item for item in config.benchmark_suite.benchmarks if item.dataset == "CV-Bench-2D"
        )
        self.assertEqual(cvbench.prompt_adapter, "cvbench_explicit_options")
        self.assertTrue(
            all(
                item.prompt_adapter is None
                for item in config.benchmark_suite.benchmarks
                if item.dataset != "CV-Bench-2D"
            )
        )
        screenspot = config.benchmark_suite.benchmarks[-1]
        self.assertEqual(screenspot.dataset, "ScreenSpot_Mobile")
        self.assertEqual(screenspot.num_shot, 4)
        self.assertEqual(screenspot.demonstration_indices, (94, 9, 24, 66))

    def test_dataset_request_override_inherits_unspecified_suite_defaults(self) -> None:
        config = load_configured_eval_run(self.run_path)
        by_dataset = {item.dataset: item for item in config.benchmark_suite.benchmarks}
        direct = by_dataset["MMMU_Pro_10c"]
        cot = by_dataset["MMMU_Pro_10c_COT"]
        self.assertEqual(direct.generation.max_new_tokens, 64)
        self.assertEqual(cot.generation.max_new_tokens, 512)
        self.assertEqual(cot.generation.stop, direct.generation.stop)
        self.assertEqual(cot.generation.temperature, direct.generation.temperature)
        self.assertEqual(direct.server_context_length, 16384)
        self.assertEqual(cot.server_context_length, 16384)
        self.assertTrue(cot.prepend_reasoning_cue)
        self.assertFalse(direct.prepend_reasoning_cue)

    def test_trusted_standalone_common_state_is_an_explicit_boolean_opt_in(self) -> None:
        self.replace(
            RUN_RELATIVE,
            "  iteration: 1",
            "  iteration: 1\n  trusted_standalone_common_state: true",
        )
        config = load_configured_eval_run(self.run_path)
        self.assertTrue(config.checkpoint.trusted_standalone_common_state)
        self.replace(
            RUN_RELATIVE,
            "  trusted_standalone_common_state: true",
            "  trusted_standalone_common_state: legacy",
        )
        self.assert_load_error()

    def test_shot_count_and_demonstration_are_dataset_owned(self) -> None:
        suite = Path("suites/n4_v6.yaml")
        self.replace(suite, "      num_shot: 2", "      num_shot: 4")
        self.replace(
            suite,
            "      demonstration: {type: benchmark, include_images: true}",
            "      demonstration: {type: trivial, include_images: false}",
        )
        config = load_configured_eval_run(self.run_path)
        first, second = config.benchmark_suite.benchmarks[:2]
        self.assertEqual(first.num_shot, 4)
        self.assertEqual(first.demonstration_type, "trivial")
        self.assertFalse(first.include_demonstration_images)
        self.assertFalse(first.prepend_reasoning_cue)
        self.assertEqual(second.num_shot, 2)
        self.assertEqual(second.demonstration_type, "benchmark")
        self.assertTrue(second.include_demonstration_images)

    def test_explicit_demonstration_indices_are_dataset_owned_and_strict(self) -> None:
        suite = Path("suites/n4_v6.yaml")
        self.replace(
            suite,
            "      demonstration: {type: benchmark, include_images: true}",
            "      demonstration: {type: benchmark, include_images: true, indices: [4, row-9]}",
        )
        config = load_configured_eval_run(self.run_path)
        first, second = config.benchmark_suite.benchmarks[:2]
        self.assertEqual(first.demonstration_indices, (4, "row-9"))
        self.assertEqual(second.demonstration_indices, ())

        shutil.rmtree(self.root)
        shutil.copytree(CONFIGS, self.root)
        self.replace(
            suite,
            "      demonstration: {type: benchmark, include_images: true}",
            "      demonstration: {type: benchmark, include_images: true, indices: [1, '1']}",
        )
        self.assert_load_error()

    def test_array_binding_selects_each_benchmarks_resource_class(self) -> None:
        config = load_configured_eval_run(self.run_path)
        four_node = bind_benchmark_array_selection(config, 1)
        eight_node = bind_benchmark_array_selection(config, 7)
        self.assertEqual((four_node.run.resource, four_node.run.nodes), ("n4", 4))
        self.assertEqual((eight_node.run.resource, eight_node.run.nodes), ("n8", 8))
        self.assertEqual(four_node.run.expert_model_parallel_size, 16)
        self.assertEqual(eight_node.run.expert_model_parallel_size, 32)

    def test_array_binding_applies_only_the_selected_server_context_override(self) -> None:
        config = load_configured_eval_run(self.run_path)
        ocrbench = bind_benchmark_array_selection(config, 0)
        mmmu_pro = bind_benchmark_array_selection(config, 11)
        self.assertIsNone(config.model.server.context_length)
        self.assertIsNone(ocrbench.model.server.context_length)
        self.assertEqual(mmmu_pro.benchmark.server_context_length, 16384)
        self.assertEqual(mmmu_pro.model.server.context_length, 16384)

    def test_benchmark_server_context_override_is_strict(self) -> None:
        suite = Path("suites/n4_v6.yaml")
        self.replace(
            suite, "      server: {context_length: 16384}", "      server: {context_length: 0}"
        )
        self.assert_load_error()
        shutil.rmtree(self.root)
        shutil.copytree(CONFIGS, self.root)
        self.replace(
            suite, "      server: {context_length: 16384}", "      server: {context_length: long}"
        )
        self.assert_load_error()

    def test_conditional_perplexity_rejects_generation_chat_template(self) -> None:
        path = self.root / "run_configs/n4_v6_surge_mm_ppl_example.yaml"
        config = load_configured_eval_run(path)
        self.assertEqual(len(config.benchmark_suite.benchmarks), 1)
        self.assertIs(
            type(config.benchmark_suite.benchmarks[0]), ConditionalPerplexityBenchmarkConfig
        )
        selected = bind_benchmark_array_selection(config, 0, scored_limit=2)
        self.assertIs(type(selected.benchmark), ConditionalPerplexityBenchmarkConfig)
        self.assertEqual(selected.benchmark.data_format, "packed_jsonl_tar")
        self.assertEqual(selected.benchmark.scored_limit, 2)
        self.assertEqual(selected.run.resource, "n1")

        self.replace(
            Path("suites/surge_mm_ppl.yaml"),
            "    retry: 10",
            "    retry: 10\n    chat_template: minimal",
        )
        with self.assertRaises(EvalConfigError):
            load_configured_eval_run(path)

    def test_unknown_or_missing_fields_fail(self) -> None:
        self.replace(RUN_RELATIVE, "schema_version: 1", "schema_version: 1\nsurprise: true")
        self.assert_load_error()
        shutil.rmtree(self.root)
        shutil.copytree(CONFIGS, self.root)
        self.replace(RUN_RELATIVE, "  iteration: 1\n", "")
        self.assert_load_error()

    def test_unknown_prompt_adapter_fails(self) -> None:
        suite = Path("suites/n4_v6.yaml")
        self.replace(
            suite, "      prompt_adapter: cvbench_explicit_options", "      prompt_adapter: unknown"
        )
        self.assert_load_error()

    def test_generation_chat_template_policy_is_explicit_and_strict(self) -> None:
        suite = Path("suites/n4_v6.yaml")
        self.replace(suite, "    chat_template: minimal\n", "")
        self.assert_load_error()
        shutil.rmtree(self.root)
        shutil.copytree(CONFIGS, self.root)
        self.replace(suite, "    chat_template: minimal", "    chat_template: tokenizer")
        config = load_configured_eval_run(self.run_path)
        self.assertTrue(
            all(
                item.generation.chat_template
                == (
                    "assets/screenspot_shared_gui_end_marker_template.jinja"
                    if item.dataset == "ScreenSpot_Mobile"
                    else "tokenizer"
                )
                for item in config.benchmark_suite.benchmarks
            )
        )
        shutil.rmtree(self.root)
        shutil.copytree(CONFIGS, self.root)
        self.replace(suite, "    chat_template: minimal", "    chat_template: ../escape.jinja")
        self.assert_load_error()

    def test_model_cannot_override_evaluator_owned_chat_template(self) -> None:
        model = Path("models/n4_v6.yaml")
        self.replace(
            model,
            "      tokenizer-prompt-format: mistral",
            "      tokenizer-prompt-format: mistral\n      chat-template: assets/custom.jinja",
        )
        self.assert_load_error()

    def test_former_run_level_shot_count_and_incomplete_demonstrations_fail(self) -> None:
        self.replace(RUN_RELATIVE, "  name: n4_v6_example", "  name: n4_v6_example\n  num_shot: 2")
        self.assert_load_error()
        shutil.rmtree(self.root)
        shutil.copytree(CONFIGS, self.root)
        self.replace(
            Path("suites/n4_v6.yaml"),
            "      demonstration: {type: benchmark, include_images: true}",
            "      demonstration: {type: benchmark}",
        )
        self.assert_load_error()

    def test_explicit_demonstration_indices_must_match_benchmark_shot_count(self) -> None:
        suite = Path("suites/n4_v6.yaml")
        self.replace(
            suite,
            "      demonstration: {type: benchmark, include_images: true}",
            "      demonstration: {type: benchmark, include_images: true, indices: [4]}",
        )
        self.assert_load_error()

    def test_component_reference_cannot_escape_config_root(self) -> None:
        outside = self.root.parent / "outside.yaml"
        shutil.copyfile(self.root / "models/n4_v6.yaml", outside)
        self.replace(RUN_RELATIVE, "../models/n4_v6.yaml", "../../outside.yaml")
        with self.assertRaisesRegex(EvalConfigError, "escapes the configuration root"):
            load_configured_eval_run(self.run_path)

    def test_duplicate_yaml_keys_and_boolean_resource_fail(self) -> None:
        cluster = Path("clusters/cmh.yaml")
        self.replace(cluster, "      nodes: 1", "      nodes: 1\n      nodes: 1")
        self.assert_load_error()
        shutil.rmtree(self.root)
        shutil.copytree(CONFIGS, self.root)
        self.replace(cluster, "      nodes: 1", "      nodes: true")
        self.assert_load_error()


if __name__ == "__main__":
    unittest.main()
