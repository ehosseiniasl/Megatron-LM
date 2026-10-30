"""Tests for benchmark-owned few-shot selection and its artifact."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from orchestration.fewshot_dataset_adaptation.demonstration_selection import (
    DemonstrationSelectionError,
    PARTITIONS_FILENAME,
    select_fewshot_rows,
    write_demonstration_and_scored_row_partitions,
)
from orchestration.fewshot_dataset_adaptation.predefined_demonstration_registry import (
    PredefinedDemonstrationError,
    load_predefined_demonstrations,
)
from phase3_generation_test_support import benchmark


class _ILoc:
    def __init__(self, frame: "Frame") -> None:
        self.frame = frame

    def __getitem__(self, position: int):
        self.frame.accessed.append(position)
        return self.frame.rows[position]


class Frame:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.accessed: list[int] = []
        self.iloc = _ILoc(self)

    def __len__(self) -> int:
        return len(self.rows)


def rows() -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "question": f"question {index}",
            "answer": "" if index == 0 else f"answer {index}",
            "image_path": f"image-{index}.png",
            "nested": {"position": index},
        }
        for index in range(6)
    ]


class DemonstrationSelectionTest(unittest.TestCase):
    def test_first_eligible_rows_are_removed_from_scoring(self) -> None:
        data = Frame(rows())
        config = replace(benchmark(), num_shot=2, scored_limit=3)
        selection = select_fewshot_rows(data, config)
        self.assertEqual(selection.demonstration_positions, (1, 2))
        self.assertEqual(selection.scored_positions, (0, 3, 4))
        self.assertEqual(selection.scored_indices, (0, 3, 4))
        self.assertEqual(tuple(item.source_index for item in selection.demonstrations), ("1", "2"))

    def test_selection_is_deep_copied_and_frozen(self) -> None:
        source = rows()
        original = copy.deepcopy(source)
        selection = select_fewshot_rows(Frame(source), replace(benchmark(), num_shot=1))
        selection.demonstrations[0].record["question"] = "changed"
        self.assertEqual(source, original)
        with self.assertRaises(FrozenInstanceError):
            selection.scored_positions = ()

    def test_selection_preserves_native_demonstration_fields(self) -> None:
        source = rows()
        source[1]["answer"] = "['300/9']"
        selection = select_fewshot_rows(
            Frame(source), replace(benchmark(), num_shot=1)
        )
        self.assertEqual(selection.demonstrations[0].record["answer"], "['300/9']")

    def test_unsupported_style_and_insufficient_rows_fail(self) -> None:
        with self.assertRaisesRegex(DemonstrationSelectionError, "does not support"):
            select_fewshot_rows(
                Frame(rows()),
                replace(benchmark(), dataset="RealWorldQA", demonstration_type="rich"),
            )
        with self.assertRaisesRegex(DemonstrationSelectionError, "insufficient"):
            select_fewshot_rows(Frame(rows()[:2]), replace(benchmark(), num_shot=2))

    def test_packaged_demonstrations_leave_all_target_rows_scoreable(self) -> None:
        config = replace(
            benchmark(),
            dataset="OCRBench",
            demonstration_type="trivial",
            num_shot=2,
            scored_limit=3,
        )
        selection = select_fewshot_rows(Frame(rows()), config)
        self.assertEqual(selection.demonstration_positions, ())
        self.assertEqual(selection.scored_positions, (0, 1, 2))
        self.assertEqual(selection.scored_indices, (0, 1, 2))
        self.assertEqual(len(selection.demonstrations), 2)
        self.assertTrue(
            all(
                item.source_index.startswith("nvbench:OCRBench:trivial:")
                for item in selection.demonstrations
            )
        )
        self.assertTrue(
            all(Path(str(item.record["image_path"])).is_file() for item in selection.demonstrations)
        )

    def test_predefined_registry_records_are_isolated(self) -> None:
        rich = load_predefined_demonstrations("MMMU_DEV_VAL", "rich", 2)
        trivial = load_predefined_demonstrations("MMMU_DEV_VAL", "trivial", 2)
        self.assertNotEqual(rich[0][0], trivial[0][0])
        rich[0][1]["question"] = "changed"
        reloaded = load_predefined_demonstrations("MMMU_DEV_VAL", "rich", 2)
        self.assertNotEqual(reloaded[0][1]["question"], "changed")
        with self.assertRaisesRegex(PredefinedDemonstrationError, "provides only"):
            load_predefined_demonstrations("OCRBench", "trivial", 9)
        with self.assertRaisesRegex(PredefinedDemonstrationError, "does not support"):
            load_predefined_demonstrations("OCRBench", "rich", 0)

    def test_reasoning_cue_is_an_opt_in_answer_prefix(self) -> None:
        config = replace(
            benchmark(),
            dataset="MMMU_Pro_10c_COT",
            demonstration_type="reasoning",
            demonstration_answer_format="raw",
            prepend_reasoning_cue=True,
            num_shot=2,
        )
        selection = select_fewshot_rows(Frame(rows()), config)
        canonical = load_predefined_demonstrations(
            "MMMU_Pro_10c_COT", "reasoning", 2
        )
        for selected, (_, plain) in zip(selection.demonstrations, canonical, strict=True):
            self.assertEqual(
                selected.record["answer"],
                "Let's reason step by step. " + plain["answer"],
            )
        reloaded = load_predefined_demonstrations(
            "MMMU_Pro_10c_COT", "reasoning", 2
        )
        self.assertEqual(reloaded, canonical)

    def test_screenspot_boxes_are_eligible_benchmark_demonstrations(self) -> None:
        source = [
            {
                "index": index,
                "question": f"click target {index}",
                "bbox": f"[{10 * index}, {20 * index}, 11, 21]",
                "image_path": f"screen-{index}.png",
            }
            for index in range(5)
        ]
        config = replace(
            benchmark(),
            dataset="ScreenSpot_Mobile",
            demonstration_answer_format="screenspot_click",
            num_shot=2,
            scored_limit=3,
        )
        selection = select_fewshot_rows(Frame(source), config)
        self.assertEqual(selection.demonstration_positions, (0, 1))
        self.assertEqual(selection.scored_positions, (2, 3, 4))
        self.assertEqual(selection.scored_indices, (2, 3, 4))

    def test_explicit_benchmark_indices_preserve_order_and_are_not_scored(self) -> None:
        config = replace(
            benchmark(),
            num_shot=2,
            scored_limit=3,
            demonstration_indices=(5, 2),
        )
        selection = select_fewshot_rows(Frame(rows()), config)
        self.assertEqual(
            tuple(item.source_index for item in selection.demonstrations),
            ("5", "2"),
        )
        self.assertEqual(selection.demonstration_positions, (5, 2))
        self.assertEqual(selection.scored_positions, (0, 1, 3))
        self.assertEqual(selection.scored_indices, (0, 1, 3))

    def test_missing_explicit_benchmark_index_fails(self) -> None:
        config = replace(
            benchmark(),
            num_shot=2,
            demonstration_indices=(2, "missing"),
        )
        with self.assertRaisesRegex(
            DemonstrationSelectionError,
            "explicit benchmark demonstrations are missing or ineligible",
        ):
            select_fewshot_rows(Frame(rows()), config)

    def test_partition_artifact_records_exact_selection(self) -> None:
        selection = select_fewshot_rows(Frame(rows()), replace(benchmark(), num_shot=1))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o750)
            artifact = write_demonstration_and_scored_row_partitions(selection, benchmark(), root)
            payload = json.loads((root / PARTITIONS_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(artifact.artifact, root / PARTITIONS_FILENAME)
            self.assertEqual(
                [row["target_index"] for row in payload["scored_rows"]],
                list(selection.scored_indices),
            )
            self.assertEqual(payload["demonstration_type"], "benchmark")
            self.assertTrue(payload["include_demonstration_images"])
            self.assertFalse(payload["prepend_reasoning_cue"])

    def test_packaged_partition_records_null_native_positions(self) -> None:
        config = replace(
            benchmark(), dataset="DocVQA_VAL", demonstration_type="trivial", num_shot=2
        )
        selection = select_fewshot_rows(Frame(rows()), config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o750)
            write_demonstration_and_scored_row_partitions(selection, config, root)
            payload = json.loads((root / PARTITIONS_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["demonstration_source"], "packaged")
        self.assertEqual(
            [item["target_position"] for item in payload["demonstrations"]],
            [None, None],
        )


if __name__ == "__main__":
    unittest.main()
