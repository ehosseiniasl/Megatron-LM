from __future__ import annotations

import pytest

from mistral_fork.mistral.data.pretokenized.cache_key import (
    TokenizerName,
    compute_cache_key,
)
from mistral_fork.mistral.data.pretokenized.components.masking import (
    midtraining_segments,
    tokenize_and_mask_midtraining,
)
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
    MaskingStrategy,
)
from mistral_fork.mistral.data.source import SampleType


class _CharacterTokenizer:
    bos_id = 1
    eos_id = 2

    def tokenize(self, text: str) -> list[int]:
        return [ord(character) + 3 for character in text]


def _sample(text: str, mask_segments: list[list[int | bool]]) -> dict[str, object]:
    return {"source": "midtraining", "text": text, "mask_segments": mask_segments}


def test_midtraining_segments_and_mask_roundtrip() -> None:
    text = "<system>masked</system>trained"
    boundary = text.index("trained")
    sample = _sample(text, [[0, boundary, False], [boundary, len(text), True]])

    assert midtraining_segments(sample) == [("<system>masked</system>", False), ("trained", True)]

    document = tokenize_and_mask_midtraining(_CharacterTokenizer(), sample)

    assert document.tokens == [1, *[ord(character) + 3 for character in text], 2]
    assert document.mask is not None
    assert document.mask.sizes == [1, boundary, len(text) - boundary + 1]
    assert document.mask.mask == [True, False, True]
    document.check(_CharacterTokenizer.bos_id, _CharacterTokenizer.eos_id)


@pytest.mark.parametrize(
    ("mask_segments", "expected_sizes", "expected_mask"),
    [
        ([[0, 4, True]], [6], [True]),
        ([[0, 4, False]], [1, 4, 1], [True, False, True]),
        ([[0, 1, False], [1, 3, False], [3, 4, True]], [1, 3, 2], [True, False, True]),
    ],
)
def test_midtraining_mask_edge_cases(
    mask_segments: list[list[int | bool]], expected_sizes: list[int], expected_mask: list[bool]
) -> None:
    document = tokenize_and_mask_midtraining(_CharacterTokenizer(), _sample("text", mask_segments))

    assert document.mask is not None
    assert document.mask.sizes == expected_sizes
    assert document.mask.mask == expected_mask


@pytest.mark.parametrize(
    ("sample", "match"),
    [
        ({"source": "other", "text": "hello", "mask_segments": [[0, 5, True]]}, "expects source"),
        ({"source": "midtraining", "text": "hello"}, "should always be present"),
        (_sample("hello", []), "must not be empty"),
        (_sample("hello", [[0, 5]]), r"must be \[start, end, train\]"),
        (_sample("hello", [[0, 5, 1]]), "must have types"),
        (_sample("hello", [[1, 5, True]]), "First segment must start at 0"),
        (_sample("hello", [[0, 4, True]]), "Last segment must end"),
        (_sample("hello", [[0, 2, True], [3, 5, False]]), "Segments must be contiguous"),
        (_sample("hello", [[0, 3, True], [2, 5, False]]), "Segments must be contiguous"),
        (_sample("hello", [[0, 0, True], [0, 5, False]]), "must satisfy"),
    ],
)
def test_midtraining_segments_reject_invalid_samples(sample: dict[str, object], match: str) -> None:
    with pytest.raises(AssertionError, match=match):
        midtraining_segments(sample)


def test_midtraining_strategy_is_serializable_and_cache_keyed() -> None:
    strategy = MaskingStrategy("MASK_MIDTRAINING_SEGMENTS")

    assert strategy is MaskingStrategy.MASK_MIDTRAINING_SEGMENTS
    assert (
        compute_cache_key(
            profile=FilterArgsName.PRETRAIN_V4__LOBSTRAL,
            tokenizer_name=TokenizerName.PRETRAIN_V3_LOBSTENIZER,
            sample_type=SampleType.PRETRAIN,
            rare_token_filter=None,
            pds_filter=None,
            masking_strategy=strategy,
        )
        == "pretrain_v3_lobstenizer/pretrain_v4__lobstral_mask_midtraining_segments"
    )
