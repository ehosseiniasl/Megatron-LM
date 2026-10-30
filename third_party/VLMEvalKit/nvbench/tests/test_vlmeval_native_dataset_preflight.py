from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from orchestration.eval_run_config_loading import (
    load_configured_eval_run,
)
from orchestration.vlmeval_native_dataset_preflight import (
    preflight_datasets,
)


NVBENCH_ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG = (
    NVBENCH_ROOT
    / "configs"
    / "run_configs"
    / "n4_v6_example.yaml"
)


class FakeRows:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.iloc = FakeILoc(self)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: str) -> list[object]:
        return [row[key] for row in self.rows]

    def copy(self, deep: bool = True) -> FakeRows:
        del deep
        return FakeRows([row.copy() for row in self.rows])


class FakeILoc:
    def __init__(self, data: FakeRows) -> None:
        self.data = data

    def __getitem__(self, position: int | list[int]) -> object:
        if type(position) is int:
            return self.data.rows[position].copy()
        return FakeRows([self.data.rows[index].copy() for index in position])


class NativeDataset:
    def __init__(self, dataset_name: str, rows: list[dict[str, object]]) -> None:
        self.dataset_name = dataset_name
        self.data = FakeRows(rows)

    def build_prompt(self, record: object) -> list[dict[str, str]]:
        if not isinstance(record, dict):
            raise TypeError("record must be a dictionary")
        return [
            {"type": "image", "value": str(record["image_path"])},
            {"type": "text", "value": str(record["question"])},
        ]

    def evaluate(self, eval_file: object, **kwargs: object) -> dict[str, object]:
        return {"eval_file": eval_file, "kwargs": kwargs}


class VLMEvalNativeDatasetPreflightTest(unittest.TestCase):
    def test_full_suite_builds_from_one_bound_cache(self) -> None:
        configured = load_configured_eval_run(RUN_CONFIG)
        suite = configured.benchmark_suite
        assert suite is not None
        by_name = {benchmark.dataset: benchmark for benchmark in suite.benchmarks}
        observed_lmudata: list[str | None] = []

        def factory(dataset_name: str) -> object:
            benchmark = by_name[dataset_name]
            observed_lmudata.append(os.environ.get("LMUData"))
            count = 14 if benchmark.demonstration_source == "benchmark" else 10
            if benchmark.demonstration_indices:
                count = max(int(index) for index in benchmark.demonstration_indices) + 11
            rows = [
                {
                    "index": index,
                    "question": f"question {index}",
                    "answer": "A",
                    "A": "first",
                    "B": "second",
                    "bbox": "[10, 20, 11, 21]",
                    "image_path": f"/tmp/image-{index}.png",
                }
                for index in range(count)
            ]
            dataset_type = type(benchmark.upstream_class, (NativeDataset,), {})
            return dataset_type(dataset_name, rows)

        previous = os.environ.get("LMUData")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory).resolve(strict=True)
            result = preflight_datasets(
                suite,
                cache,
                dataset_factory=factory,
            )
        self.assertEqual(os.environ.get("LMUData"), previous)
        self.assertEqual(len(result.records), 18)
        self.assertEqual(
            tuple(record.dataset for record in result.records),
            tuple(benchmark.dataset for benchmark in suite.benchmarks),
        )
        self.assertEqual(observed_lmudata, [str(cache)] * 18)
        for record, benchmark in zip(result.records, suite.benchmarks, strict=True):
            with self.subTest(dataset=record.dataset):
                self.assertEqual(record.configured_class, benchmark.upstream_class)
                self.assertEqual(record.actual_class, benchmark.upstream_class)
                self.assertEqual(record.demonstration_count, benchmark.num_shot)
                source_rows = 14 if benchmark.demonstration_source == "benchmark" else 10
                if benchmark.demonstration_indices:
                    source_rows = max(int(index) for index in benchmark.demonstration_indices) + 11
                expected_rows = source_rows
                if benchmark.demonstration_source == "benchmark":
                    expected_rows -= benchmark.num_shot
                self.assertEqual(record.scored_row_count, expected_rows)
                self.assertEqual(record.prompt_turn_count, benchmark.num_shot * 2 + 1)
                expected_images = benchmark.num_shot + 1
                if not benchmark.include_demonstration_images:
                    expected_images = 1
                self.assertEqual(record.prompt_image_count, expected_images)

    def test_packaged_text_only_demonstrations_keep_all_query_rows(self) -> None:
        configured = load_configured_eval_run(RUN_CONFIG)
        source_suite = configured.benchmark_suite
        assert source_suite is not None
        benchmark = replace(
            source_suite.benchmarks[0],
            demonstration_type="trivial",
            include_demonstration_images=False,
        )
        suite = replace(source_suite, benchmarks=(benchmark,))
        rows = [
            {
                "index": index,
                "question": f"question {index}",
                "answer": "answer",
                "image_path": f"/tmp/image-{index}.png",
            }
            for index in range(5)
        ]
        dataset_type = type(benchmark.upstream_class, (NativeDataset,), {})
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory).resolve(strict=True)
            result = preflight_datasets(
                suite,
                cache,
                dataset_factory=lambda name: dataset_type(name, rows),
            )
        record = result.records[0]
        self.assertEqual(record.demonstration_count, 2)
        self.assertEqual(record.scored_row_count, 5)
        self.assertEqual(record.prompt_turn_count, 5)
        self.assertEqual(record.prompt_image_count, 1)


if __name__ == "__main__":
    unittest.main()
