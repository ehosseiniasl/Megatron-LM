from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from orchestration.benchmark_array_selection import (
    benchmark_array_run_component,
    bind_benchmark_array_selection,
    common_generation_config,
)
from orchestration.configured_eval_run import (
    BenchmarkSuiteConfig,
    EvalConfigError,
)
from orchestration.eval_run_config_loading import (
    load_configured_eval_run,
)
from orchestration.eval_run_path_resolution import resolve_eval_run
from orchestration.resolved_run_manifest import (
    canonical_config_fingerprint,
    stable_manifest_payload,
)

from phase1_resolution_test_support import ResolutionFixture


NVBENCH_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RUN = (
    NVBENCH_ROOT
    / "configs"
    / "run_configs"
    / "n4_v6_example.yaml"
)


class BenchmarkArraySelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ResolutionFixture()
        base = self.fixture.config()
        second = replace(
            base.benchmark,
            name="docvqa-smoke",
            dataset="DocVQA_VAL",
            upstream_class="DocVQADataset",
            result_reader="vlmeval_accuracy_csv",
            expected_result_suffix="_acc.csv",
            result_metric_unit="fraction",
            num_shot=4,
            include_demonstration_images=False,
        )
        self.suite_config = replace(
            base,
            benchmark=None,
            benchmark_suite=BenchmarkSuiteConfig(
                "fixture-suite",
                str(self.fixture.metadata),
                (base.benchmark, second),
            ),
        )

    def tearDown(self) -> None:
        self.fixture.close()

    def test_binds_first_and_last_entries_with_complete_suite_identity(self) -> None:
        first = bind_benchmark_array_selection(self.suite_config, 0)
        last = bind_benchmark_array_selection(self.suite_config, 1)
        self.assertEqual(first.benchmark.dataset, "OCRBench")
        self.assertEqual(last.benchmark.dataset, "DocVQA_VAL")
        self.assertEqual((first.benchmark.num_shot, last.benchmark.num_shot), (2, 4))
        self.assertTrue(first.benchmark.include_demonstration_images)
        self.assertFalse(last.benchmark.include_demonstration_images)
        self.assertIsNone(first.benchmark_suite)
        self.assertEqual(
            last.benchmark_array_selection.ordered_datasets,
            ("OCRBench", "DocVQA_VAL"),
        )
        self.assertEqual(last.benchmark_array_selection.source_suite_name, "fixture-suite")
        self.assertEqual(benchmark_array_run_component(first), "fixture-run__array-00-of-02")
        self.assertEqual(benchmark_array_run_component(last), "fixture-run__array-01-of-02")

    def test_selected_benchmark_context_override_is_bound_to_its_server(self) -> None:
        suite = self.suite_config.benchmark_suite
        second_benchmark = replace(suite.benchmarks[1], server_context_length=16384)
        configured = replace(
            self.suite_config,
            benchmark_suite=replace(
                suite,
                benchmarks=(suite.benchmarks[0], second_benchmark),
            ),
        )
        first = bind_benchmark_array_selection(configured, 0)
        second = bind_benchmark_array_selection(configured, 1)
        self.assertEqual(first.model.server.context_length, 8192)
        self.assertEqual(second.model.server.context_length, 16384)

    def test_rejects_invalid_or_repeated_binding_before_mutation(self) -> None:
        for value in (-1, 2, True, 0.0):
            with self.subTest(value=value):
                with self.assertRaises(EvalConfigError):
                    bind_benchmark_array_selection(self.suite_config, value)
        selected = bind_benchmark_array_selection(self.suite_config, 0)
        with self.assertRaisesRegex(EvalConfigError, "already bound"):
            bind_benchmark_array_selection(selected, 0)
        self.assertIsNotNone(self.suite_config.benchmark_suite)
        self.assertIsNone(self.suite_config.benchmark_array_selection)

    def test_debug_limit_is_positive_and_bound_to_the_selected_benchmark(self) -> None:
        selected = bind_benchmark_array_selection(
            self.suite_config,
            0,
            scored_limit=7,
        )
        self.assertEqual(selected.benchmark.scored_limit, 7)
        self.assertEqual(self.suite_config.benchmark_suite.benchmarks[0].scored_limit, 10)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(EvalConfigError, "positive integer"):
                    bind_benchmark_array_selection(
                        self.suite_config,
                        0,
                        scored_limit=value,
                    )

    def test_validates_the_complete_suite_before_selecting_one_entry(self) -> None:
        suite = self.suite_config.benchmark_suite
        invalid_last = replace(
            suite.benchmarks[-1],
            generation=replace(suite.benchmarks[-1].generation, max_new_tokens=9000),
        )
        invalid = replace(
            self.suite_config,
            benchmark_suite=replace(
                suite,
                benchmarks=(suite.benchmarks[0], invalid_last),
            ),
        )
        with self.assertRaisesRegex(EvalConfigError, "exceeds the server context"):
            bind_benchmark_array_selection(invalid, 0)

    def test_single_benchmark_accepts_only_index_zero(self) -> None:
        selected = bind_benchmark_array_selection(self.fixture.config(), 0)
        self.assertEqual(selected.benchmark.dataset, "OCRBench")
        self.assertEqual(selected.benchmark_array_selection.benchmark_count, 1)
        self.assertIsNone(selected.benchmark_array_selection.source_suite_name)
        self.assertEqual(benchmark_array_run_component(selected), "fixture-run__array-00-of-01")
        with self.assertRaisesRegex(EvalConfigError, "outside"):
            bind_benchmark_array_selection(self.fixture.config(), 1)

    def test_reasoning_cue_is_boolean_and_reasoning_only(self) -> None:
        base = self.fixture.config()
        with self.assertRaisesRegex(EvalConfigError, "must be boolean"):
            bind_benchmark_array_selection(
                replace(
                    base,
                    benchmark=replace(base.benchmark, prepend_reasoning_cue="yes"),
                ),
                0,
            )
        with self.assertRaisesRegex(EvalConfigError, "requires reasoning"):
            bind_benchmark_array_selection(
                replace(
                    base,
                    benchmark=replace(base.benchmark, prepend_reasoning_cue=True),
                ),
                0,
            )

    def test_selection_changes_resolved_path_and_manifest_fingerprint(self) -> None:
        resolved = []
        for index in (0, 1):
            selected = bind_benchmark_array_selection(self.suite_config, index)
            resolved.append(
                resolve_eval_run(
                    selected,
                    self.fixture.workspace,
                    validate_destination=False,
                )
            )
        self.assertNotEqual(resolved[0].paths.run_directory, resolved[1].paths.run_directory)
        self.assertNotEqual(
            canonical_config_fingerprint(resolved[0]),
            canonical_config_fingerprint(resolved[1]),
        )
        selection = stable_manifest_payload(resolved[1])["configured_eval_run"][
            "benchmark_array_selection"
        ]
        self.assertEqual(selection["benchmark_index"], 1)
        self.assertEqual(selection["source_suite_path"], str(self.fixture.metadata))

    def test_unbound_manifest_shape_remains_compatible(self) -> None:
        resolved = resolve_eval_run(
            self.fixture.config(),
            self.fixture.workspace,
            validate_destination=False,
        )
        self.assertNotIn(
            "benchmark_array_selection",
            stable_manifest_payload(resolved)["configured_eval_run"],
        )

    def test_real_canonical_first_and_last_indices_are_stable(self) -> None:
        config = load_configured_eval_run(CANONICAL_RUN)
        first = bind_benchmark_array_selection(config, 0)
        cot = bind_benchmark_array_selection(config, 12)
        previous_last = bind_benchmark_array_selection(config, 16)
        last = bind_benchmark_array_selection(config, 17)
        self.assertEqual(first.benchmark.dataset, "OCRBench")
        self.assertEqual(cot.benchmark.dataset, "MMMU_Pro_10c_COT")
        self.assertTrue(cot.benchmark.prepend_reasoning_cue)
        self.assertEqual(cot.benchmark.generation.max_new_tokens, 512)
        self.assertEqual(common_generation_config(cot).max_new_tokens, 512)
        self.assertEqual(previous_last.benchmark.dataset, "MMMU_DEV_VAL")
        self.assertEqual(last.benchmark.dataset, "ScreenSpot_Mobile")
        self.assertEqual(last.benchmark.num_shot, 4)
        self.assertEqual(last.benchmark_array_selection.benchmark_count, 18)
        with self.assertRaisesRegex(EvalConfigError, "outside"):
            bind_benchmark_array_selection(config, 18)
        with self.assertRaisesRegex(EvalConfigError, "one selected benchmark"):
            common_generation_config(config)


if __name__ == "__main__":
    unittest.main()
