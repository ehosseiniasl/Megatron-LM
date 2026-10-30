"""Reference-suffix alignment and pooled conditional-perplexity metrics."""

from __future__ import annotations
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .utils import StrictJsonError, load_json_bytes, write_json


class ConditionalPerplexityResultError(RuntimeError):
    """Raised when token alignment or metric inputs violate the contract."""


@dataclass(frozen=True)
class ConditionalPerplexityScore:
    """Sufficient statistics for one completely scored sample."""

    uid: str
    category: str
    loglikelihood: float
    token_count: int
    byte_count: int
    word_count: int


def load_completed_perplexity_scores(path: Path) -> tuple[ConditionalPerplexityScore, ...]:
    """Read and validate an append-only completed-sample ledger for resume."""
    if not path.exists():
        return ()
    scores: list[ConditionalPerplexityScore] = []
    seen: set[str] = set()
    try:
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                value = load_json_bytes(raw, maximum_bytes=64 * 1024)
                if type(value) is not dict or set(value) != {
                    "uid",
                    "category",
                    "loglikelihood",
                    "token_count",
                    "byte_count",
                    "word_count",
                }:
                    raise ConditionalPerplexityResultError(
                        f"invalid completed sample at line {line_number}"
                    )
                score = ConditionalPerplexityScore(**value)
                if score.uid in seen:
                    raise ConditionalPerplexityResultError(
                        f"duplicate completed sample uid: {score.uid}"
                    )
                _pooled((score,))
                scores.append(score)
                seen.add(score.uid)
    except (OSError, StrictJsonError, TypeError) as error:
        raise ConditionalPerplexityResultError("cannot resume completed samples") from error
    return tuple(scores)


def description_token_suffix(prompt_tokens: list[str], description: str) -> slice:
    """Locate the smallest returned token-piece suffix ending in the reference text."""
    if not prompt_tokens or any(type(token) is not str for token in prompt_tokens):
        raise ConditionalPerplexityResultError("prompt token pieces must be nonempty strings")
    if type(description) is not str or not description.strip():
        raise ConditionalPerplexityResultError("description must be nonempty")
    target = unicodedata.normalize("NFC", description.strip())
    suffix = ""
    for start in range(len(prompt_tokens) - 1, -1, -1):
        suffix = prompt_tokens[start] + suffix
        if unicodedata.normalize("NFC", suffix).endswith(target):
            return slice(start, len(prompt_tokens))
    raise ConditionalPerplexityResultError(
        "prompt tokenization does not end with the reference description"
    )


def score_description(
    *,
    uid: str,
    category: str,
    description: str,
    prompt_tokens: list[str],
    prompt_logprobs: list[float | None],
) -> ConditionalPerplexityScore:
    """Select and sum the reference-description log-probability suffix."""
    if len(prompt_tokens) != len(prompt_logprobs):
        raise ConditionalPerplexityResultError("prompt tokens and logprobs have different lengths")
    selected = description_token_suffix(prompt_tokens, description)
    values = prompt_logprobs[selected]
    if not values or any(type(value) not in (int, float) for value in values):
        raise ConditionalPerplexityResultError("description logprobs are missing or nonnumeric")
    numeric = tuple(float(value) for value in values)  # type: ignore[arg-type]
    if not all(math.isfinite(value) and value <= 0 for value in numeric):
        raise ConditionalPerplexityResultError(
            "description logprobs must be finite and nonpositive"
        )
    byte_count = len(description.encode("utf-8"))
    word_count = len(description.split())
    if byte_count <= 0 or word_count <= 0:
        raise ConditionalPerplexityResultError("description has no metric denominator")
    return ConditionalPerplexityScore(
        uid=uid,
        category=category,
        loglikelihood=sum(numeric),
        token_count=len(numeric),
        byte_count=byte_count,
        word_count=word_count,
    )


def _pooled(scores: tuple[ConditionalPerplexityScore, ...]) -> dict[str, float | int]:
    if not scores:
        raise ConditionalPerplexityResultError("cannot aggregate an empty score set")
    loglikelihood = sum(item.loglikelihood for item in scores)
    token_count = sum(item.token_count for item in scores)
    byte_count = sum(item.byte_count for item in scores)
    word_count = sum(item.word_count for item in scores)
    if min(token_count, byte_count, word_count) <= 0 or not math.isfinite(loglikelihood):
        raise ConditionalPerplexityResultError("aggregate sufficient statistics are invalid")
    nll = -loglikelihood
    try:
        ppl_word = math.exp(nll / word_count)
        ppl_byte = math.exp(nll / byte_count)
    except OverflowError as error:
        raise ConditionalPerplexityResultError("perplexity is not finite") from error
    return {
        "num_samples": len(scores),
        "loglikelihood": loglikelihood,
        "token_count": token_count,
        "byte_count": byte_count,
        "word_count": word_count,
        "perplexity_per_word": ppl_word,
        "perplexity_per_byte": ppl_byte,
        "bits_per_byte": nll / byte_count / math.log(2),
        "nats_per_token": nll / token_count,
    }


def aggregate_conditional_perplexity(
    scores: tuple[ConditionalPerplexityScore, ...],
) -> dict[str, object]:
    """Pool sufficient statistics overall and independently by category."""
    if len({item.uid for item in scores}) != len(scores):
        raise ConditionalPerplexityResultError("sample scores contain duplicate uids")
    categories = sorted({item.category for item in scores})
    return {
        "overall": _pooled(scores),
        "by_image_type": {
            category: _pooled(tuple(item for item in scores if item.category == category))
            for category in categories
        },
    }


def write_failed_perplexity_status(
    path: Path,
    *,
    dataset: str,
    expected: int | None,
    completed: int | None,
    failed: int | None,
    stage: str,
    error: Exception,
) -> Path:
    """Write a truthful terminal failure status for any PPL lifecycle stage."""
    return write_json(
        path,
        {
            "schema_version": 1,
            "evaluation_kind": "conditional_perplexity",
            "dataset": dataset,
            "status": "failed",
            "expected_sample_count": expected,
            "completed_sample_count": completed,
            "failed_sample_count": failed,
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        },
    )


__all__ = (
    "ConditionalPerplexityResultError",
    "ConditionalPerplexityScore",
    "aggregate_conditional_perplexity",
    "description_token_suffix",
    "load_completed_perplexity_scores",
    "score_description",
    "write_failed_perplexity_status",
)
