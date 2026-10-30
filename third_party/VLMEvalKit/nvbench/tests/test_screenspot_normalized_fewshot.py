"""Regression checks for the experimentally verified ScreenSpot format."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from jinja2 import Environment, StrictUndefined
from orchestration.benchmark_array_selection import bind_benchmark_array_selection
from orchestration.configured_eval_run import EvalConfigError
from orchestration.eval_run_config_loading import load_configured_eval_run
from orchestration.fewshot_dataset_adaptation.demonstration_selection import (
    DemonstrationExample,
    select_fewshot_rows,
)
from orchestration.fewshot_dataset_adaptation.fewshot_conversation_builder import (
    FewshotConversationError,
    build_fewshot_conversation,
    format_demonstration_answer,
)
from PIL import Image

NVBENCH = Path(__file__).resolve().parents[1]
GUI = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform pyautogui click/moveTo action to complete the task. "
    "The answer format is `pyautogui.click(x=?, y=?), x and y is necessary`"
)
TASK = "Please complete the following tasks by clicking using `pyautogui.click`:\n"


class NativePromptDataset:
    @staticmethod
    def build_prompt(record):
        return [
            {"role": "system", "type": "text", "value": GUI},
            *({"type": "image", "value": path} for path in record["images"]),
            {"type": "text", "value": TASK + record["question"]},
        ]


class ScreenSpotNormalizedFewshotTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def image(self, name, size):
        path = self.root / name
        Image.new("RGB", size, "navy").save(path)
        return str(path)

    def conversation(self, record, *, include_images=True):
        return build_fewshot_conversation(
            NativePromptDataset(),
            (DemonstrationExample("arbitrary-id", record),),
            {"images": ["query.png"], "question": "query"},
            include_demonstration_images=include_images,
            demonstration_answer_format="screenspot_click_normalized",
        )

    def test_arbitrary_demo_uses_original_dimensions_even_without_demo_images(self):
        record = {
            "bbox": [100, 20, 101, 21],
            "question": "demo",
            "images": [self.image("arbitrary.png", (600, 200))],
        }
        before = copy.deepcopy(record)
        for include_images in (True, False):
            turns = self.conversation(record, include_images=include_images)
            self.assertEqual(
                turns[1]["content"][0]["value"], "pyautogui.click(x=0.250000, y=0.150000)"
            )
            self.assertEqual(
                any(item["type"] == "image" for item in turns[0]["content"]), include_images
            )
            self.assertTrue(any(item["type"] == "image" for item in turns[-1]["content"]))
        self.assertEqual(record, before)

    def test_single_pixel_box_and_legacy_pixel_format(self):
        self.assertEqual(
            format_demonstration_answer(
                {"bbox": [599, 199, 1, 1]}, "screenspot_click_normalized", image_size=(600, 200)
            ),
            "pyautogui.click(x=0.998333, y=0.995000)",
        )
        self.assertEqual(
            format_demonstration_answer({"bbox": [100, 20, 101, 21]}, "screenspot_click"),
            "pyautogui.click(x=150, y=30)",
        )

    def test_dimensions_and_out_of_image_centers_fail_explicitly(self):
        for size in (None, (0, 200), (600, False), (600.0, 200), [600, 200], (600,)):
            with self.subTest(size=size), self.assertRaises(FewshotConversationError):
                format_demonstration_answer(
                    {"bbox": [0, 0, 1, 1]}, "screenspot_click_normalized", image_size=size
                )
        for bbox in (
            [600, 0, 1, 1],
            [0, 200, 1, 1],
            [0, 0, float("nan"), 1],
            [True, 0, 1, 1],
            [0, 0, 0, 1],
        ):
            with self.subTest(bbox=bbox), self.assertRaises(FewshotConversationError):
                format_demonstration_answer(
                    {"bbox": bbox}, "screenspot_click_normalized", image_size=(600, 200)
                )

    def test_missing_corrupt_and_ambiguous_images_fail(self):
        good = self.image("valid.png", (100, 100))
        corrupt = self.root / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        for images in ([], [good, good], [str(self.root / "missing.png")], [str(corrupt)]):
            with self.subTest(images=images), self.assertRaises(FewshotConversationError):
                self.conversation({"bbox": [0, 0, 1, 1], "question": "demo", "images": images})

    def test_canonical_selection_and_render_match_the_successful_experiment(self):
        import pandas as pd

        config = load_configured_eval_run(NVBENCH / "configs/run_configs/n4_v6_example.yaml")
        selected = bind_benchmark_array_selection(config, 17)
        benchmark = selected.benchmark
        cases = (
            (94, "back", [20, 162, 90, 88], (1170, 2532)),
            (9, "search bar", [825, 72, 848, 77], (2360, 1640)),
            (24, "downloads", [6, 1003, 624, 86], (2360, 1640)),
            (66, "invert the lens", [965, 2105, 145, 153], (1170, 2532)),
        )
        records = [
            {
                "index": index,
                "question": question,
                "bbox": bbox,
                "images": [self.image(f"{index}.png", size)],
            }
            for index, question, bbox, size in cases
        ]
        query = {"index": 0, "question": "check the weather", "images": ["query.png"]}
        frame = pd.DataFrame([query, *reversed(records)])
        partition = select_fewshot_rows(frame, benchmark)
        self.assertEqual(
            [demo.source_index for demo in partition.demonstrations], ["94", "9", "24", "66"]
        )
        self.assertEqual(partition.scored_indices, (0,))
        turns = build_fewshot_conversation(
            NativePromptDataset(),
            partition.demonstrations,
            query,
            include_demonstration_images=True,
            demonstration_answer_format=benchmark.demonstration_answer_format,
        )
        flattened = []
        image_index = 0
        for turn in turns:
            values = []
            for item in turn["content"]:
                if item["type"] == "image":
                    values.append(f"<image_{image_index}>")
                    image_index += 1
                else:
                    values.append(item["value"])
            flattened.append({"role": turn["role"], "content": "\n".join(values)})
        template = Environment(undefined=StrictUndefined).from_string(
            (NVBENCH / benchmark.generation.chat_template).read_text()
        )
        actual = template.render(messages=flattened)
        expected = (
            NVBENCH / "tests/fixtures/screenspot/normalized_fourshot_prompt.txt"
        ).read_text()
        self.assertEqual(actual, expected)
        self.assertEqual(actual.count(GUI), 1)
        self.assertEqual(actual.count(TASK), 1)
        self.assertEqual(image_index, 5)
        self.assertEqual(actual.count("\n__________\n"), 4)
        self.assertEqual(benchmark.generation.stop, ("__________",))

    def test_normalized_formatter_rejects_other_bbox_contracts_and_zero_shot_marker(self):
        config = load_configured_eval_run(NVBENCH / "configs/run_configs/n4_v6_example.yaml")
        selected = bind_benchmark_array_selection(config, 17)
        for dataset, upstream in (
            ("ScreenSpot_Pro_CAD", "ScreenSpot_Pro"),
            ("OCRBench", "OCRBench"),
        ):
            with (
                self.subTest(dataset=dataset),
                self.assertRaisesRegex(
                    EvalConfigError,
                    "standard ScreenSpot|unsupported VLMEvalKit benchmark result contract",
                ),
            ):
                bind_benchmark_array_selection(
                    replace(
                        selected,
                        benchmark_array_selection=None,
                        benchmark=replace(
                            selected.benchmark, dataset=dataset, upstream_class=upstream
                        ),
                    ),
                    0,
                )
        with self.assertRaisesRegex(EvalConfigError, "requires few-shot"):
            bind_benchmark_array_selection(
                replace(
                    selected,
                    benchmark_array_selection=None,
                    benchmark=replace(selected.benchmark, num_shot=0, demonstration_indices=()),
                ),
                0,
            )

    def test_template_keeps_non_screenspot_messages_unchanged(self):
        template = Environment(undefined=StrictUndefined).from_string(
            (NVBENCH / "assets/screenspot_shared_gui_end_marker_template.jinja").read_text()
        )
        turns = [
            {"role": "user", "content": "demo"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "query"},
        ]
        minimal = Environment(undefined=StrictUndefined).from_string(
            (NVBENCH / "assets/minimal_chat_template.jinja").read_text()
        )
        self.assertEqual(template.render(messages=turns), minimal.render(messages=turns))


if __name__ == "__main__":
    unittest.main()
