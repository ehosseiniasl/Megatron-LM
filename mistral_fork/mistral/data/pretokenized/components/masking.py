from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable
from typing import Any, Protocol

from mistral_fork.mistral.data.pretokenized.masking import DocumentMask
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import MaskingStrategy

FILES_SEPARATOR = "\n\n\n"
FILE_PATH_MARKER = "+++++ "
FILE_SEPARATOR_RE = re.compile(r"(?:^|\n)(\+{5})(?= )")
GITHUB_PR_RENDERED_SECTION_HEADER_RE = re.compile(r"<<<<<<<< [^\n]* >>>>>>>>\n\n")
LOCODIFF_CONTENT_BEFORE_OPEN = "<content_before>\n"
LOCODIFF_CONTENT_BEFORE_CLOSE = "</content_before>\n"
LOCODIFF_GIT_DIFF_OPEN = "<git_diff>\n"
LOCODIFF_GIT_DIFF_CLOSE = "</git_diff>\n"
LOCODIFF_OUTPUT_OPEN = "<OUTPUT>\n"
LOCODIFF_OUTPUT_CLOSE = "\n</OUTPUT>"


class TokenizerLike(Protocol):
    bos_id: int
    eos_id: int

    def tokenize(self, text: str) -> list[int]: ...


@dataclasses.dataclass(frozen=True)
class PretokenizedDoc:
    tokens: list[int]
    images: list[Any]
    mask: DocumentMask | None

    def check(self, bos_id: int, eos_id: int) -> None:
        assert len(self.tokens) > 1, "Document must have more than one token"
        assert self.tokens[0] == bos_id, "Document must start with BOS"
        assert self.tokens[-1] == eos_id, "Document must end with EOS"
        assert all(img.pos_in_text < len(self.tokens) - 1 for img in self.images), (
            [img.pos_in_text for img in self.images],
            len(self.tokens),
        )
        if self.mask is not None:
            self.mask.check()
            assert self.mask.mask[0] is True, "First token must be trainable"
            assert self.mask.mask[-1] is True, "Final token must be trainable"


def format_repo_level(repo_texts: list[list[str]]) -> str:
    assert all(len(item) == 2 for item in repo_texts), (
        "repo_texts must contain [path, text] pairs"
    )
    files_with_path = [
        f"{FILE_PATH_MARKER}{file_path}\n{text}" for file_path, text in repo_texts
    ]
    return FILES_SEPARATOR.join(files_with_path)


def _add_mask_span(
    sizes: list[int], values: list[bool], length: int, value: bool
) -> None:
    if length == 0:
        return
    if values and values[-1] == value:
        sizes[-1] += length
    else:
        sizes.append(length)
        values.append(value)


def _merge_adjacent_segments(
    segments: Iterable[tuple[str, bool]],
) -> list[tuple[str, bool]]:
    merged_segments: list[tuple[str, bool]] = []
    for text, should_train in segments:
        if not text:
            continue
        if merged_segments and merged_segments[-1][1] == should_train:
            previous_text, _ = merged_segments[-1]
            merged_segments[-1] = (previous_text + text, should_train)
        else:
            merged_segments.append((text, should_train))
    return merged_segments


def encode_segments_with_mask(
    tokenizer: TokenizerLike,
    segments: Iterable[tuple[str, bool]],
    *,
    bos: bool,
    eos: bool,
) -> PretokenizedDoc:
    all_tokens: list[int] = []
    mask_sizes: list[int] = []
    mask_values: list[bool] = []

    if bos:
        all_tokens.append(tokenizer.bos_id)
        _add_mask_span(mask_sizes, mask_values, 1, True)

    for text, should_train in segments:
        tokens = tokenizer.tokenize(text)
        all_tokens.extend(tokens)
        _add_mask_span(mask_sizes, mask_values, len(tokens), should_train)

    if eos:
        all_tokens.append(tokenizer.eos_id)
        _add_mask_span(mask_sizes, mask_values, 1, True)

    return PretokenizedDoc(
        tokens=all_tokens,
        images=[],
        mask=DocumentMask(sizes=mask_sizes, mask=mask_values),
    )


def file_separator_segments(text: str) -> list[tuple[str, bool]]:
    chunks = FILE_SEPARATOR_RE.split(text)
    segments: list[tuple[str, bool]] = []
    for i, chunk in enumerate(chunks):
        should_keep = i % 2 == 0
        if not should_keep:
            assert chunk == "+++++", chunk
        segments.append((chunk, should_keep))
    return segments


def tokenize_and_mask_file_separator(
    tokenizer: TokenizerLike, text: str
) -> PretokenizedDoc:
    return encode_segments_with_mask(
        tokenizer, file_separator_segments(text), bos=True, eos=True
    )


def reasoning_segments(sample: dict[str, Any]) -> list[tuple[str, bool]]:
    text = sample["text"]
    segments_end = sample.get("segments_end")
    if segments_end is None:
        segments_end = sample.get("meta", {}).get("formatting", {}).get("segments_end")
    if segments_end is None:
        raise ValueError("reasoning_traces sample is missing segments_end")
    assert len(segments_end) == 5, segments_end
    starts = [0, *segments_end[:-1]]
    question, pre_formatting, reasoning, post_formatting, answer = [
        text[start:end] for start, end in zip(starts, segments_end, strict=True)
    ]
    assert pre_formatting == post_formatting == "\n\n", (
        pre_formatting,
        post_formatting,
    )
    return [
        (question + pre_formatting, False),
        (reasoning + post_formatting + answer, True),
    ]


def tokenize_and_mask_reasoning(
    tokenizer: TokenizerLike, sample: dict[str, Any]
) -> PretokenizedDoc:
    return encode_segments_with_mask(
        tokenizer, reasoning_segments(sample), bos=True, eos=True
    )


def midtraining_segments(sample: dict[str, Any]) -> list[tuple[str, bool]]:
    """Return validated text and train-mask segments for a midtraining sample."""
    assert sample.get("source") == "midtraining", (
        f"MASK_MIDTRAINING_SEGMENTS expects source='midtraining', got {sample.get('source')!r}"
    )
    assert "mask_segments" in sample, (
        "mask_segments should always be present on midtraining samples"
    )
    text = sample["text"]
    assert isinstance(text, str), type(text)
    mask_segments = sample["mask_segments"]
    assert isinstance(mask_segments, list), type(mask_segments)
    assert mask_segments, "mask_segments must not be empty"

    validated_segments: list[tuple[int, int, bool]] = []
    for i, segment in enumerate(mask_segments):
        assert isinstance(segment, list) and len(segment) == 3, (
            f"Mask segment {i} must be [start, end, train], got {segment!r}"
        )
        start, end, train = segment
        assert type(start) is int and type(end) is int and type(train) is bool, (
            f"Mask segment {i} must have types [int, int, bool], got {segment!r}"
        )
        assert 0 <= start < end <= len(text), (
            f"Mask segment {i} must satisfy 0 <= start < end <= len(text)={len(text)}, "
            f"got {segment!r}"
        )
        validated_segments.append((start, end, train))

    assert validated_segments[0][0] == 0, (
        f"First segment must start at 0, got {validated_segments[0][0]}"
    )
    assert validated_segments[-1][1] == len(text), (
        f"Last segment must end at len(text)={len(text)}, got {validated_segments[-1][1]}"
    )
    for i in range(1, len(validated_segments)):
        previous_end = validated_segments[i - 1][1]
        start = validated_segments[i][0]
        assert start == previous_end, (
            f"Segments must be contiguous: segment {i} starts at {start} "
            f"but segment {i - 1} ends at {previous_end}"
        )

    return [(text[start:end], train) for start, end, train in validated_segments]


def tokenize_and_mask_midtraining(
    tokenizer: TokenizerLike, sample: dict[str, Any]
) -> PretokenizedDoc:
    """Tokenize a midtraining sample using its character-level mask segments."""
    return encode_segments_with_mask(
        tokenizer, midtraining_segments(sample), bos=True, eos=True
    )


def _github_pr_masked_sections(masking_strategy: MaskingStrategy) -> set[str]:
    if masking_strategy is MaskingStrategy.MASK_GITHUB_PR_FORMATTING:
        return {"header"}
    if masking_strategy is MaskingStrategy.MASK_GITHUB_PR_EXPANDED_CONTEXT:
        return {"header", "expanded_context"}
    if masking_strategy is MaskingStrategy.MASK_GITHUB_PR_EXPANDED_CONTEXT_AND_CONTEXT:
        return {"header", "expanded_context", "context"}
    raise NotImplementedError(f"Unsupported GitHub PR masking strategy: {masking_strategy}")


def _split_github_pr_rendered_section_headers(
    text: str, should_train: bool
) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    cursor = 0
    for match in GITHUB_PR_RENDERED_SECTION_HEADER_RE.finditer(text):
        if match.start() > cursor:
            segments.append((text[cursor : match.start()], should_train))
        segments.append((match.group(0), False))
        cursor = match.end()
    if cursor < len(text):
        segments.append((text[cursor:], should_train))
    return segments


def github_pr_segments(
    sample: dict[str, Any], masking_strategy: MaskingStrategy
) -> list[tuple[str, bool]]:
    masked_sections = _github_pr_masked_sections(masking_strategy)
    segments = []
    named_contents = sample["named_contents"]
    for i, item in enumerate(named_contents):
        assert isinstance(item, list) and len(item) == 2, item
        section_name, content = item
        assert isinstance(section_name, str), item
        assert isinstance(content, str), item
        separator = "\n\n" if i < len(named_contents) - 1 else ""
        should_train = section_name not in masked_sections
        segments.extend(
            _split_github_pr_rendered_section_headers(content + separator, should_train)
        )
    return segments


def tokenize_and_mask_github_pr(
    tokenizer: TokenizerLike,
    sample: dict[str, Any],
    masking_strategy: MaskingStrategy,
) -> PretokenizedDoc:
    segments = _merge_adjacent_segments(
        github_pr_segments(sample, masking_strategy)
    )
    return encode_segments_with_mask(
        tokenizer, segments, bos=True, eos=True
    )


def locodiff_segments(text: str) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    pos = 0

    if text.startswith(LOCODIFF_CONTENT_BEFORE_OPEN):
        segments.append((LOCODIFF_CONTENT_BEFORE_OPEN, False))
        pos = len(LOCODIFF_CONTENT_BEFORE_OPEN)
        close_index = text.find(LOCODIFF_CONTENT_BEFORE_CLOSE, pos)
        assert close_index >= 0, "Missing </content_before>"
        segments.append((text[pos:close_index], True))
        segments.append((LOCODIFF_CONTENT_BEFORE_CLOSE, False))
        pos = close_index + len(LOCODIFF_CONTENT_BEFORE_CLOSE)

    diff_open_index = text.find(LOCODIFF_GIT_DIFF_OPEN, pos)
    assert diff_open_index >= 0, "Missing <git_diff>"
    if diff_open_index > pos:
        segments.append((text[pos:diff_open_index], True))
        segments.append((LOCODIFF_GIT_DIFF_OPEN, False))
    elif segments and not segments[-1][1]:
        previous_text, _ = segments.pop()
        segments.append((previous_text + LOCODIFF_GIT_DIFF_OPEN, False))
    else:
        assert text.startswith(LOCODIFF_GIT_DIFF_OPEN), (
            "Expected text to start with <content_before> or <git_diff>, "
            f"got: {text[:80]!r}"
        )
        segments.append((LOCODIFF_GIT_DIFF_OPEN, False))
    pos = diff_open_index + len(LOCODIFF_GIT_DIFF_OPEN)

    diff_close_index = text.find(LOCODIFF_GIT_DIFF_CLOSE, pos)
    assert diff_close_index >= 0, "Missing </git_diff>"
    segments.append((text[pos:diff_close_index], True))

    after_diff_close = diff_close_index + len(LOCODIFF_GIT_DIFF_CLOSE)
    output_open_index = text.find(LOCODIFF_OUTPUT_OPEN, after_diff_close)
    assert output_open_index >= 0, "Missing <OUTPUT>"
    if output_open_index > after_diff_close:
        segments.append((LOCODIFF_GIT_DIFF_CLOSE, False))
        segments.append((text[after_diff_close:output_open_index], True))
        segments.append((LOCODIFF_OUTPUT_OPEN, False))
    else:
        segments.append(
            (
                text[
                    diff_close_index : output_open_index
                    + len(LOCODIFF_OUTPUT_OPEN)
                ],
                False,
            )
        )
    pos = output_open_index + len(LOCODIFF_OUTPUT_OPEN)

    output_close_index = text.rfind(LOCODIFF_OUTPUT_CLOSE)
    assert output_close_index >= pos, "Missing or empty </OUTPUT> block"
    segments.append((text[pos:output_close_index], True))
    segments.append((text[output_close_index:], False))

    reconstructed = "".join(segment for segment, _ in segments)
    assert reconstructed == text, (
        f"Segment reconstruction mismatch: got {len(reconstructed)} chars, "
        f"expected {len(text)}"
    )
    return segments


def tokenize_and_mask_locodiff(
    tokenizer: TokenizerLike,
    text: str,
) -> PretokenizedDoc:
    return encode_segments_with_mask(
        tokenizer, locodiff_segments(text), bos=True, eos=True
    )
