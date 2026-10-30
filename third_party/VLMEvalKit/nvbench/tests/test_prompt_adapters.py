from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from orchestration.fewshot_dataset_adaptation.prompt_adapters import (
    PromptAdapterError,
    apply_prompt_adapter,
    supported_prompt_adapters,
)


class PromptAdapterTest(unittest.TestCase):
    def test_cvbench_explicit_options_matches_reference_prompt(self) -> None:
        native = [
            {"type": "image", "value": "image.png"},
            {"type": "text", "value": "raw prompt\nA zero\nB two"},
        ]
        record = {"question": "How many objects are visible?", "A": 0, "B": 2, "C": float("nan")}

        adapted = apply_prompt_adapter("cvbench_explicit_options", native, record)

        self.assertEqual(
            adapted,
            [
                {"type": "image", "value": "image.png"},
                {
                    "type": "text",
                    "value": (
                        "Question: How many objects are visible?\n"
                        "Options:\n"
                        "A. 0\n"
                        "B. 2\n"
                        "Please select the correct answer from the options above. \n"
                    ),
                },
            ],
        )
        self.assertEqual(native[1]["value"], "raw prompt\nA zero\nB two")

    def test_no_adapter_copies_without_changing_prompt(self) -> None:
        native = [{"type": "text", "value": "unchanged"}]
        adapted = apply_prompt_adapter(None, native, {})
        self.assertEqual(adapted, native)
        self.assertIsNot(adapted, native)
        self.assertIsNot(adapted[0], native[0])

    def test_unknown_adapter_and_invalid_record_fail(self) -> None:
        with self.assertRaisesRegex(PromptAdapterError, "unsupported"):
            apply_prompt_adapter("unknown", [], {})
        with self.assertRaisesRegex(PromptAdapterError, "mapping-like"):
            apply_prompt_adapter(
                "cvbench_explicit_options", [{"type": "text", "value": "raw"}], object()
            )

    def test_supported_adapter_names_are_explicit(self) -> None:
        self.assertEqual(supported_prompt_adapters(), frozenset({"cvbench_explicit_options"}))

    def test_submit_suite_help_does_not_require_pandas(self) -> None:
        package_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "pandas.py").write_text(
                "raise RuntimeError('pandas imported on the login node')\n", encoding="utf-8"
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join((directory, str(package_root)))
            environment["PYTHONNOUSERSITE"] = "1"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [sys.executable, "-m", "nvbench.submit_suite", "--help"],
                cwd=package_root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-config", result.stdout)


if __name__ == "__main__":
    unittest.main()
