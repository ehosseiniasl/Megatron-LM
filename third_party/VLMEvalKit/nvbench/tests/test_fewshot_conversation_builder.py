from __future__ import annotations

import copy
import unittest
from pathlib import Path

from orchestration.configured_eval_run import (
    BenchmarkConfig,
    GenerationConfig,
)
from orchestration.fewshot_dataset_adaptation.demonstration_selection import (
    DemonstrationExample,
    FewshotRowSelection,
)
from orchestration.fewshot_dataset_adaptation import (
    fewshot_conversation_builder as conversation_builder,
)

FewshotConversationError = conversation_builder.FewshotConversationError
VLMEvalMessageError = conversation_builder.VLMEvalMessageError
VLMEvalFewshotDatasetProxy = conversation_builder.VLMEvalFewshotDatasetProxy
VLMEvalFewshotDatasetProxyError = conversation_builder.VLMEvalFewshotDatasetProxyError
build_fewshot_conversation = conversation_builder.build_fewshot_conversation
copy_vlmeval_prompt_items = conversation_builder.copy_vlmeval_prompt_items
format_demonstration_answer = conversation_builder.format_demonstration_answer


class RecordingDataset:
    def __init__(self, *, mutate_inputs: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.mutate_inputs = mutate_inputs

    def build_prompt(self, record: object) -> object:
        if not isinstance(record, dict):
            raise TypeError("test records must be dictionaries")
        self.calls.append(copy.deepcopy(record))
        prompt = record["prompt"]
        if self.mutate_inputs:
            record["question"] = "mutated"
            if isinstance(prompt, list):
                prompt.append({"type": "text", "value": "dataset-added"})
        return prompt


def prompt(image_names: tuple[str, ...], question: str) -> list[dict[str, str]]:
    return [
        *({"type": "image", "value": image} for image in image_names),
        {"type": "text", "value": question},
    ]


class FewshotConversationBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.demonstrations = (
            DemonstrationExample(
                "packaged:0",
                {
                    "question": "demo zero",
                    "answer": "alpha",
                    "prompt": prompt(("demo-0.png",), "demo zero"),
                },
            ),
            DemonstrationExample(
                "packaged:1",
                {
                    "question": "demo one",
                    "answer": "beta",
                    "prompt": prompt(("demo-1a.png", "demo-1b.png"), "demo one"),
                },
            ),
        )
        self.query = {
            "index": "target-7",
            "question": "target",
            "prompt": prompt(("query-a.png", "query-b.png"), "target"),
        }

    def test_snapshot_has_two_alternating_demo_pairs_then_query(self) -> None:
        conversation = build_fewshot_conversation(
            RecordingDataset(),
            self.demonstrations,
            self.query,
            include_demonstration_images=True,
            demonstration_answer_format="raw",
        )
        self.assertEqual(
            conversation,
            [
                {
                    "role": "user",
                    "content": prompt(("demo-0.png",), "demo zero"),
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": "alpha"}],
                },
                {
                    "role": "user",
                    "content": prompt(
                        ("demo-1a.png", "demo-1b.png"), "demo one"
                    ),
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": "beta"}],
                },
                {
                    "role": "user",
                    "content": prompt(("query-a.png", "query-b.png"), "target"),
                },
            ],
        )

    def test_demo_images_can_be_removed_without_changing_query_images(self) -> None:
        conversation = build_fewshot_conversation(
            RecordingDataset(),
            self.demonstrations,
            self.query,
            include_demonstration_images=False,
            demonstration_answer_format="raw",
        )
        self.assertEqual(conversation[0]["content"], [
            {"type": "text", "value": "demo zero"}
        ])
        self.assertEqual(conversation[2]["content"], [
            {"type": "text", "value": "demo one"}
        ])
        self.assertEqual(
            conversation[-1]["content"],
            prompt(("query-a.png", "query-b.png"), "target"),
        )

    def test_fresh_prompt_calls_and_input_mutation_isolation(self) -> None:
        dataset = RecordingDataset(mutate_inputs=True)
        original_demos = copy.deepcopy(self.demonstrations)
        original_query = copy.deepcopy(self.query)
        first = build_fewshot_conversation(
            dataset,
            self.demonstrations,
            self.query,
            include_demonstration_images=True,
            demonstration_answer_format="raw",
        )
        second = build_fewshot_conversation(
            dataset,
            self.demonstrations,
            self.query,
            include_demonstration_images=True,
            demonstration_answer_format="raw",
        )
        self.assertEqual(len(dataset.calls), 6)
        self.assertEqual(dataset.calls[:3], dataset.calls[3:])
        self.assertEqual(self.demonstrations, original_demos)
        self.assertEqual(self.query, original_query)
        self.assertEqual(first, second)
        self.assertTrue(all(message is not self.query for message in dataset.calls))

    def test_zero_demonstrations_uses_the_same_query_path(self) -> None:
        conversation = build_fewshot_conversation(
            RecordingDataset(),
            (),
            self.query,
            include_demonstration_images=False,
            demonstration_answer_format="raw",
        )
        self.assertEqual(conversation, [{
            "role": "user",
            "content": prompt(("query-a.png", "query-b.png"), "target"),
        }])

    def test_prompt_adapter_applies_to_demonstrations_and_query(self) -> None:
        demonstration = DemonstrationExample(
            "cv-demo",
            {
                "question": "demo question",
                "A": "left",
                "B": "right",
                "answer": "A",
                "prompt": prompt(("demo.png",), "raw demo prompt"),
            },
        )
        query = {
            "question": "query question",
            "A": "near",
            "B": "far",
            "prompt": prompt(("query.png",), "raw query prompt"),
        }
        conversation = build_fewshot_conversation(
            RecordingDataset(),
            (demonstration,),
            query,
            include_demonstration_images=True,
            demonstration_answer_format="label",
            prompt_adapter="cvbench_explicit_options",
        )
        self.assertEqual(
            conversation[0]["content"][-1]["value"],
            "Question: demo question\nOptions:\nA. left\nB. right\n"
            "Please select the correct answer from the options above. \n",
        )
        self.assertEqual(conversation[1]["content"][0]["value"], "A")
        self.assertEqual(
            conversation[2]["content"][-1]["value"],
            "Question: query question\nOptions:\nA. near\nB. far\n"
            "Please select the correct answer from the options above. \n",
        )

    def test_raw_label_and_value_answer_formats(self) -> None:
        self.assertEqual(format_demonstration_answer({"answer": ["x", "y"]}, "raw"), "x")
        self.assertEqual(
            format_demonstration_answer({"answer": "['x', 'y']"}, "raw"),
            "x",
        )
        options = {"A": "red", "B": "blue", "answer": "blue"}
        self.assertEqual(format_demonstration_answer(options, "label"), "B")
        self.assertEqual(
            format_demonstration_answer({**options, "answer_label": "A"}, "label"),
            "A",
        )
        self.assertEqual(
            format_demonstration_answer({**options, "answer": "B"}, "value"),
            "blue",
        )
        self.assertEqual(
            format_demonstration_answer({**options, "answer_value": "azure"}, "value"),
            "azure",
        )
        self.assertEqual(
            format_demonstration_answer(
                {"bbox": "[223, 78, 601, 593]"}, "screenspot_click"
            ),
            "pyautogui.click(x=523, y=374)",
        )

    def test_answer_failures_are_explicit(self) -> None:
        failures = (
            ({}, "raw"),
            ({"answer": " "}, "raw"),
            ({"answer": "x"}, "unknown"),
            ({"A": "same", "B": "same", "answer": "same"}, "label"),
            ({"A": "x", "answer_label": "B"}, "label"),
            ({"A": "x", "B": float("nan"), "answer_label": "B"}, "label"),
            ({"A": "x", "B": "y", "answer": "z"}, "value"),
            ({"answer": []}, "raw"),
            ({"answer": "[]"}, "raw"),
            ({"answer": [[]]}, "raw"),
            ({"answer": True}, "raw"),
            ({"answer": float("inf")}, "raw"),
            ({"bbox": "[0, 0, 0, 4]"}, "screenspot_click"),
            ({"bbox": "[0, 0, nan, 4]"}, "screenspot_click"),
            ({"bbox": "[0, 0, 4]"}, "screenspot_click"),
        )
        for record, answer_format in failures:
            with self.subTest(record=record, answer_format=answer_format):
                with self.assertRaises(FewshotConversationError):
                    format_demonstration_answer(record, answer_format)

    def test_prompt_item_conversion_copies_and_preserves_order(self) -> None:
        source = prompt(("first.png", "second.png"), "question")
        copied = copy_vlmeval_prompt_items(source)
        self.assertEqual(copied, source)
        self.assertIsNot(copied, source)
        self.assertTrue(all(left is not right for left, right in zip(copied, source)))
        copied[0]["value"] = "changed.png"
        self.assertEqual(source[0]["value"], "first.png")

    def test_prompt_item_conversion_omits_native_empty_text_separators(self) -> None:
        source = [
            {"type": "text", "value": ""},
            {"type": "image", "value": "first.png"},
            {"type": "text", "value": "  "},
            {"type": "image", "value": "second.png"},
            {"type": "text", "value": "question"},
        ]
        self.assertEqual(
            copy_vlmeval_prompt_items(source),
            [
                {"type": "image", "value": "first.png"},
                {"type": "image", "value": "second.png"},
                {"type": "text", "value": "question"},
            ],
        )

    def test_prompt_item_conversion_flattens_native_system_text(self) -> None:
        source = [
            {"role": "system", "type": "text", "value": "click precisely"},
            {"type": "image", "value": "screen.png"},
            {"type": "text", "value": "open settings"},
        ]
        self.assertEqual(
            copy_vlmeval_prompt_items(source),
            [
                {"type": "text", "value": "click precisely"},
                {"type": "image", "value": "screen.png"},
                {"type": "text", "value": "open settings"},
            ],
        )

    def test_prompt_item_conversion_rejects_invalid_contracts(self) -> None:
        invalid = (
            "text",
            [],
            ["text"],
            [{"type": "text"}],
            [{"type": "text", "value": "x", "extra": True}],
            [{"role": "assistant", "type": "text", "value": "x"}],
            [{"role": "system", "type": "image", "value": "x.png"}],
            [{"type": "audio", "value": "x"}],
            [{"type": [], "value": "x"}],
            [{"type": "image", "value": " "}],
            [{"type": "image", "value": 3}],
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(VLMEvalMessageError):
                    copy_vlmeval_prompt_items(value)
        with self.assertRaises(VLMEvalMessageError):
            copy_vlmeval_prompt_items(
                [{"type": "image", "value": "only.png"}], include_images=False
            )


class FakeRow(dict[str, object]):
    def to_dict(self) -> dict[str, object]:
        return dict(self)


class FakeILoc:
    def __init__(self, frame: "FakeFrame") -> None:
        self.frame = frame

    def __getitem__(self, key: object) -> object:
        if type(key) is int:
            return FakeRow(copy.deepcopy(self.frame.rows[key]))
        if isinstance(key, list) and all(type(value) is int for value in key):
            return FakeFrame([self.frame.rows[value] for value in key])
        raise TypeError("unsupported fake iloc key")


class FakeFrame:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = copy.deepcopy(rows)
        self.iloc = FakeILoc(self)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: str) -> list[object]:
        return [row[key] for row in self.rows]

    def __setitem__(self, key: str, values: list[object]) -> None:
        if len(values) != len(self.rows):
            raise ValueError("column length mismatch")
        for row, value in zip(self.rows, values):
            row[key] = value

    def pop(self, key: str) -> list[object]:
        return [row.pop(key) for row in self.rows]

    def copy(self, deep: bool = True) -> "FakeFrame":
        if not deep:
            raise AssertionError("proxy must request a deep data copy")
        return FakeFrame(self.rows)


class FakeDataset:
    def __init__(self) -> None:
        self.data = FakeFrame(
            [
                {
                    "index": f"target-{position}",
                    "question": f"question {position}",
                    "answer": f"answer {position}",
                    "image": f"encoded-{position}",
                    "image_path": [f"{position}-a.png", f"{position}-b.png"],
                }
                for position in range(4)
            ]
        )
        self.dataset_name = "FakeVQA"
        self.extra_attribute = "delegated"
        self.prompt_calls: list[dict[str, object]] = []
        self.evaluate_calls: list[tuple[object, dict[str, object]]] = []

    def build_prompt(self, record: object) -> list[dict[str, str]]:
        if not isinstance(record, dict):
            raise TypeError("record must be a dictionary")
        self.prompt_calls.append(copy.deepcopy(record))
        raw_paths = record["image_path"]
        paths = [raw_paths] if isinstance(raw_paths, str) else list(raw_paths)
        return [
            *({"type": "image", "value": str(path)} for path in paths),
            {"type": "text", "value": str(record["question"])},
        ]

    def evaluate(self, eval_file: object, **judge_kwargs: object) -> object:
        self.evaluate_calls.append((eval_file, judge_kwargs))
        return {"score": 0.75}


def benchmark() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="fake_1shot",
        dataset="FakeVQA",
        upstream_class="FakeDataset",
        modality="multimodal",
        num_shot=1,
        demonstration_type="benchmark",
        demonstration_answer_format="raw",
        include_demonstration_images=True,
        prepend_reasoning_cue=False,
        scored_limit=2,
        result_reader="fake",
        expected_result_suffix="_score.json",
        result_metric_unit="reader_defined",
        judge="exact_matching",
        resource="n1",
        generation=GenerationConfig(
            temperature=0.0,
            retry=1,
            image_size=-1,
            image_detail="low",
            max_new_tokens=8,
            stop=("\n\n",),
            enable_thinking=False,
            chat_template="minimal",
        ),
    )


def selection(
    *,
    positions: tuple[int, ...] = (1, 3),
    indices: tuple[object, ...] = ("target-1", "target-3"),
) -> FewshotRowSelection:
    demonstration = DemonstrationExample(
        "target-0",
        {
            "index": "target-0",
            "question": "demo question",
            "answer": "demo answer",
            "image_path": "demo.png",
        },
    )
    return FewshotRowSelection((demonstration,), (0,), positions, indices)


class VLMEvalFewshotDatasetProxyTest(unittest.TestCase):
    def test_scored_view_len_getitem_and_native_indices(self) -> None:
        wrapped = FakeDataset()
        wrapped_rows = copy.deepcopy(wrapped.data.rows)
        wrapped_class = type(wrapped)
        proxy = VLMEvalFewshotDatasetProxy(wrapped, selection(), benchmark())
        self.assertEqual(len(proxy), 2)
        self.assertEqual(proxy.data["index"], ["target-1", "target-3"])
        self.assertEqual(proxy[0]["question"], "question 1")
        returned = proxy[0]
        returned["question"] = "changed"
        self.assertEqual(proxy[0]["question"], "question 1")
        self.assertEqual(wrapped.data.rows, wrapped_rows)
        self.assertIs(type(wrapped), wrapped_class)
        self.assertIs(proxy.wrapped_dataset, wrapped)

    def test_upstream_style_mutation_changes_only_proxy_data(self) -> None:
        wrapped = FakeDataset()
        original = copy.deepcopy(wrapped.data.rows)
        proxy = VLMEvalFewshotDatasetProxy(wrapped, selection(), benchmark())
        proxy.data["prediction"] = ["one", "two"]
        proxy.data.pop("image")
        self.assertEqual(wrapped.data.rows, original)
        self.assertNotIn("image", proxy.data.rows[0])
        self.assertEqual(proxy.data["prediction"], ["one", "two"])

    def test_build_prompt_uses_demo_and_selected_query_through_wrapped_dataset(self) -> None:
        wrapped = FakeDataset()
        proxy = VLMEvalFewshotDatasetProxy(wrapped, selection(), benchmark())
        conversation = proxy.build_prompt(0)
        self.assertEqual(len(wrapped.prompt_calls), 2)
        self.assertEqual(wrapped.prompt_calls[0]["index"], "target-0")
        self.assertEqual(wrapped.prompt_calls[1]["index"], "target-1")
        self.assertEqual(
            [message["role"] for message in conversation],
            ["user", "assistant", "user"],
        )
        self.assertEqual(
            conversation[-1]["content"][:2],
            [
                {"type": "image", "value": "1-a.png"},
                {"type": "image", "value": "1-b.png"},
            ],
        )
        self.assertTrue(proxy.force_use_dataset_prompt)

    def test_evaluate_and_unknown_attributes_delegate_without_rewriting_artifact(self) -> None:
        wrapped = FakeDataset()
        proxy = VLMEvalFewshotDatasetProxy(wrapped, selection(), benchmark())
        artifact = object()
        result = proxy.evaluate(artifact, judge="exact_matching")
        self.assertEqual(result, {"score": 0.75})
        self.assertEqual(wrapped.evaluate_calls, [(artifact, {"judge": "exact_matching"})])
        self.assertIs(wrapped.evaluate_calls[0][0], artifact)
        self.assertEqual(proxy.dataset_name, "FakeVQA")
        self.assertEqual(proxy.extra_attribute, "delegated")

    def test_invalid_selection_positions_and_indices_fail(self) -> None:
        invalid = (
            selection(positions=(-1,), indices=("target-3",)),
            selection(positions=(1, 1), indices=("target-1", "target-1")),
            selection(positions=(4,), indices=("missing",)),
            selection(positions=(1,), indices=("wrong",)),
        )
        for value in invalid:
            with self.subTest(selection=value):
                with self.assertRaises(VLMEvalFewshotDatasetProxyError):
                    VLMEvalFewshotDatasetProxy(FakeDataset(), value, benchmark())

    def test_internally_inconsistent_selection_fails(self) -> None:
        base = selection()
        invalid = (
            FewshotRowSelection((), (), base.scored_positions, base.scored_indices),
            FewshotRowSelection(
                base.demonstrations,
                (),
                base.scored_positions,
                base.scored_indices,
            ),
            FewshotRowSelection(
                base.demonstrations,
                (0,),
                (0, 1),
                ("target-0", "target-1"),
            ),
            FewshotRowSelection(
                (
                    DemonstrationExample(
                        "wrong",
                        base.demonstrations[0].record,
                    ),
                ),
                (0,),
                base.scored_positions,
                base.scored_indices,
            ),
        )
        for value in invalid:
            with self.subTest(selection=value):
                with self.assertRaises(VLMEvalFewshotDatasetProxyError):
                    VLMEvalFewshotDatasetProxy(FakeDataset(), value, benchmark())

    def test_source_uses_composition_without_registry_or_monkeypatch_patterns(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "orchestration"
            / "fewshot_dataset_adaptation"
            / "fewshot_conversation_builder.py"
        )
        source = source_path.read_text(encoding="utf-8")
        for forbidden in ("supported_VLM", "sys.modules", "monkeypatch", ".__class__ ="):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
