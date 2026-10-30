"""Compatibility tests for nvbench-owned predefined demonstrations."""

from __future__ import annotations

import string
import unittest

from orchestration.fewshot_dataset_adaptation.demonstration_selection import (
    DemonstrationExample,
)
from orchestration.fewshot_dataset_adaptation.fewshot_conversation_builder import (
    build_fewshot_conversation,
)
from orchestration.fewshot_dataset_adaptation.predefined_demonstration_registry import (
    load_predefined_demonstrations,
)


class NativePromptDataset:
    """Exercise the image/question/options fields consumed by native builders."""

    @staticmethod
    def build_prompt(record: dict[str, object]) -> list[dict[str, str]]:
        question = str(record["question"])
        options = [
            f"{label}. {record[label]}"
            for label in string.ascii_uppercase
            if label in record
        ]
        text = question if not options else f"{question}\nOptions:\n" + "\n".join(options)
        return [
            {"type": "image", "value": str(record["image_path"])},
            {"type": "text", "value": text},
        ]


class PredefinedDemonstrationsTest(unittest.TestCase):
    CASES = (
        ("DocVQA_VAL", "trivial", "raw"),
        ("MMMU_DEV_VAL", "rich", "label"),
        ("MMMU_DEV_VAL", "trivial", "label"),
        ("MMMU_Pro_10c_COT", "reasoning", "raw"),
        ("MMMU_Pro_10c", "trivial", "label"),
        ("OCRBench", "trivial", "raw"),
        ("RealWorldQA", "trivial", "label"),
    )

    def test_every_supported_dataset_style_builds_with_native_prompt_logic(self) -> None:
        for name, style, answer_format in self.CASES:
            with self.subTest(dataset=name, style=style):
                source_id, record = load_predefined_demonstrations(name, style, 1)[0]
                dataset = NativePromptDataset()
                prompt = dataset.build_prompt(record)
                self.assertTrue(any(item["type"] == "image" for item in prompt))
                self.assertTrue(any(item["type"] == "text" for item in prompt))

                conversation = build_fewshot_conversation(
                    dataset,
                    (DemonstrationExample(source_id, record),),
                    record,
                    include_demonstration_images=True,
                    demonstration_answer_format=answer_format,
                )
                self.assertEqual([turn["role"] for turn in conversation], ["user", "assistant", "user"])
                self.assertTrue(
                    any(item["type"] == "image" for item in conversation[0]["content"])
                )

    def test_reasoning_records_support_a_direct_label_only_control(self) -> None:
        source_id, record = load_predefined_demonstrations(
            "MMMU_Pro_10c", "reasoning", 1
        )[0]
        conversation = build_fewshot_conversation(
            NativePromptDataset(),
            (DemonstrationExample(source_id, record),),
            record,
            include_demonstration_images=True,
            demonstration_answer_format="label",
        )
        self.assertEqual(conversation[1]["content"], [{"type": "text", "value": "B"}])

    def test_text_only_removes_only_the_demonstration_image(self) -> None:
        source_id, record = load_predefined_demonstrations("MMMU_DEV_VAL", "trivial", 1)[0]
        dataset = NativePromptDataset()
        conversation = build_fewshot_conversation(
            dataset,
            (DemonstrationExample(source_id, record),),
            record,
            include_demonstration_images=False,
            demonstration_answer_format="label",
        )
        self.assertFalse(any(item["type"] == "image" for item in conversation[0]["content"]))
        self.assertTrue(any(item["type"] == "image" for item in conversation[2]["content"]))


if __name__ == "__main__":
    unittest.main()
