from __future__ import annotations

import pytest

from mistral_fork.mistral.data.pretokenized.components.masking import (
    locodiff_segments,
    tokenize_and_mask_github_pr,
)
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import MaskingStrategy


class RecordingTokenizer:
    bos_id = 1
    eos_id = 2

    def __init__(self) -> None:
        self.calls: list[str] = []

    def tokenize(self, text: str) -> list[int]:
        self.calls.append(text)
        return [len(self.calls) + 2]

LOCODIFF_REGULAR = (
    "<git_diff>\n"
    "commit 1: initial commit\n"
    "\n"
    "diff --git a/src/main.py b/src/main.py\n"
    "--- /dev/null\n"
    "+++ b/src/main.py\n"
    "+print('hello')\n"
    "</git_diff>\n"
    "<OUTPUT>\n"
    "print('hello world')\n"
    "</OUTPUT>"
)

LOCODIFF_CONTENT_BEFORE = (
    "<content_before>\n"
    "print('old code')\n"
    "</content_before>\n"
    "\n"
    "<git_diff>\n"
    "commit 1: update code\n"
    "\n"
    "diff --git a/src/main.py b/src/main.py\n"
    "-print('old code')\n"
    "+print('new code')\n"
    "</git_diff>\n"
    "<OUTPUT>\n"
    "print('new code')\n"
    "</OUTPUT>"
)

LOCODIFF_CONTENT_BEFORE_ADJACENT = (
    "<content_before>\n"
    "print('old code')\n"
    "</content_before>\n"
    "<git_diff>\n"
    "commit 1: update code\n"
    "\n"
    "diff --git a/src/main.py b/src/main.py\n"
    "-print('old code')\n"
    "+print('new code')\n"
    "</git_diff>\n"
    "<OUTPUT>\n"
    "print('new code')\n"
    "</OUTPUT>"
)


def test_github_pr_merges_adjacent_segments_before_tokenization() -> None:
    tokenizer = RecordingTokenizer()
    sample = {
        "named_contents": [
            ["pr_description", "description"],
            ["discussion", "\n<<<<<<<< Discussions >>>>>>>>\n\nbody"],
        ]
    }

    tokenize_and_mask_github_pr(
        tokenizer,
        sample,
        MaskingStrategy.MASK_GITHUB_PR_EXPANDED_CONTEXT,
    )

    assert tokenizer.calls == [
        "description\n\n\n",
        "<<<<<<<< Discussions >>>>>>>>\n\n",
        "body",
    ]


@pytest.mark.parametrize(
    "text",
    [
        LOCODIFF_REGULAR,
        LOCODIFF_CONTENT_BEFORE,
        LOCODIFF_CONTENT_BEFORE_ADJACENT,
    ],
)
def test_locodiff_segments_roundtrip_and_mask_tags(text: str) -> None:
    segments = locodiff_segments(text)

    assert "".join(segment for segment, _ in segments) == text
    assert segments[0][1] is False
    assert segments[-1][1] is False
    for left, right in zip(segments, segments[1:], strict=True):
        assert left[1] is not right[1]


def test_locodiff_regular_segments() -> None:
    assert locodiff_segments(LOCODIFF_REGULAR) == [
        ("<git_diff>\n", False),
        (
            "commit 1: initial commit\n"
            "\n"
            "diff --git a/src/main.py b/src/main.py\n"
            "--- /dev/null\n"
            "+++ b/src/main.py\n"
            "+print('hello')\n",
            True,
        ),
        ("</git_diff>\n<OUTPUT>\n", False),
        ("print('hello world')", True),
        ("\n</OUTPUT>", False),
    ]
