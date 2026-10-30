"""Focused contract tests for nvbench-owned chat-template selection."""

from __future__ import annotations
import unittest
from pathlib import Path

from jinja2 import Environment
from orchestration.chat_template_policy import (
    MINIMAL_CHAT_TEMPLATE,
    configured_chat_template_path,
    parse_chat_template_policy,
)
from orchestration.configured_eval_run import EvalConfigError

NVBENCH = Path(__file__).resolve().parents[1]


class ChatTemplatePolicyTest(unittest.TestCase):
    def test_builtin_and_custom_policies_normalize_in_one_place(self) -> None:
        self.assertEqual(
            configured_chat_template_path(parse_chat_template_policy("minimal", "policy")),
            MINIMAL_CHAT_TEMPLATE,
        )
        self.assertIsNone(
            configured_chat_template_path(parse_chat_template_policy("tokenizer", "policy"))
        )
        self.assertEqual(
            configured_chat_template_path(
                parse_chat_template_policy("assets/custom.jinja", "policy")
            ),
            "assets/custom.jinja",
        )

    def test_unsafe_or_ambiguous_custom_paths_fail(self) -> None:
        for value in ("", " minimal", "/tmp/template.jinja", "../template.jinja", "asset.txt"):
            with self.subTest(value=value), self.assertRaises(EvalConfigError):
                parse_chat_template_policy(value, "policy")

    def test_minimal_template_contains_only_content_delimiters(self) -> None:
        template = (NVBENCH / MINIMAL_CHAT_TEMPLATE).read_text(encoding="utf-8")
        self.assertIn("message['content'] + '\\n'", template)
        self.assertIn("message['content'] + '\\n\\n'", template)
        for forbidden in ("question:", "answer:", "</s>", "bos_token", "eos_token"):
            self.assertNotIn(forbidden, template)

    def test_minimal_template_renders_only_contents_and_newlines(self) -> None:
        source = (NVBENCH / MINIMAL_CHAT_TEMPLATE).read_text(encoding="utf-8")
        rendered = (
            Environment()
            .from_string(source)
            .render(
                messages=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "demo question"},
                    {"role": "assistant", "content": "demo answer"},
                    {"role": "user", "content": "target question"},
                ],
                add_generation_prompt=True,
            )
        )
        self.assertEqual(rendered, "system\ndemo question\ndemo answer\n\ntarget question\n")


if __name__ == "__main__":
    unittest.main()
