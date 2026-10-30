#!/usr/bin/env python3
"""Lightweight integration test for BigCodeBench task utilities."""

import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from datasets import load_dataset

from lm_eval.tasks.bigcodebench.utils import (
    build_references,
    build_text_complete,
    build_text_instruct,
    score_predictions,
    score_predictions_instruct,
)


def _validate_doc_fields(doc: dict) -> bool:
    required = [
        "task_id",
        "complete_prompt",
        "instruct_prompt",
        "entry_point",
        "test",
    ]
    missing = [k for k in required if k not in doc]
    if missing:
        print(f"  FAIL: Missing required fields: {missing}")
        return False
    return True


def main() -> bool:
    print("=" * 60)
    print("BigCodeBench Task Test")
    print("=" * 60)

    # Load a single sample from the same split used by our task configs.
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    doc = ds[0]

    if not _validate_doc_fields(doc):
        return False

    # Basic prompt/reference checks.
    complete_prompt = build_text_complete(doc)
    instruct_prompt = build_text_instruct(doc)
    references = build_references(doc)
    if not (isinstance(complete_prompt, str) and isinstance(instruct_prompt, str) and isinstance(references, str)):
        print("  FAIL: Prompt/reference builders did not return strings")
        return False
    print("  PASS: Prompt/reference builders")

    # Intentionally wrong responses should almost always fail on benchmark tests.
    wrong_complete_resp = "\n    return None\n"
    wrong_instruct_resp = "def noop(*args, **kwargs):\n    return None\n"

    complete_results = score_predictions([[wrong_complete_resp]], [doc])
    instruct_results = score_predictions_instruct([[wrong_instruct_resp]], [doc])

    # Validate result structure and expected failed status.
    try:
        complete_result = complete_results[0][0]
        instruct_result = instruct_results[0][0]
        complete_passed = bool(complete_result["passed"])
        instruct_passed = bool(instruct_result["passed"])
    except Exception as e:
        print(f"  FAIL: Unexpected result structure: {e}")
        return False

    print(f"  complete wrong-code passed: {complete_passed}")
    print(f"  instruct wrong-code passed: {instruct_passed}")

    if complete_passed or instruct_passed:
        print("  FAIL: Wrong-code sample unexpectedly passed")
        return False

    print("  PASS: Wrong-code checks failed as expected")
    print("=" * 60)
    print("SUMMARY: PASS")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
