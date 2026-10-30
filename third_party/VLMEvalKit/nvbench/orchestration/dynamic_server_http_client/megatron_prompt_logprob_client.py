"""Prompt-logprob client for the Megatron multimodal completions endpoint."""

from __future__ import annotations
import base64
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ..utils import StrictJsonError, load_json_bytes
from .json_http_transport import (
    JsonHttpResponseTooLargeError,
    RetryableJsonHttpError,
    TerminalJsonHttpError,
    post_json_once,
    validate_local_http_endpoint,
)


class PromptLogprobClientError(RuntimeError):
    """Raised when the request or response violates the prompt-logprob contract."""


@dataclass(frozen=True)
class PromptLogprobs:
    """Prompt token IDs and their aligned conditional log-probabilities."""

    token_ids: list[int]
    tokens: list[str]
    logprobs: list[float | None]


class PromptTokenizer(Protocol):
    """Hugging Face tokenizer operations needed by the PPL protocol."""

    unk_token_id: int | None

    def encode(self, text: str, **kwargs: object) -> list[int]:
        """Encode one complete scored prompt."""
        ...

    def convert_tokens_to_ids(self, token: str) -> int:
        """Resolve one tokenizer-native special token."""
        ...


def resolve_image_token(tokenizer: PromptTokenizer) -> str:
    """Resolve the tokenizer-native image marker without config duplication."""
    matches: list[str] = []
    for candidate in ("<img>", "<image>"):
        token_id = tokenizer.convert_tokens_to_ids(candidate)
        if type(token_id) is not int or token_id == tokenizer.unk_token_id:
            continue
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if encoded == [token_id]:
            matches.append(candidate)
    if len(matches) != 1:
        raise PromptLogprobClientError(
            "tokenizer must expose exactly one atomic <img> or <image> token"
        )
    return matches[0]


def build_scored_prompt_token_ids(
    tokenizer: PromptTokenizer, instruction: str, description: str
) -> list[int]:
    """Encode instruction, image marker, and reference together exactly once."""
    if type(instruction) is not str or not instruction:
        raise PromptLogprobClientError("instruction must be a nonempty string")
    if type(description) is not str or not description.strip():
        raise PromptLogprobClientError("description must be a nonempty string")
    image_token = resolve_image_token(tokenizer)
    tokens = tokenizer.encode(
        f"{instruction}{image_token}{description.strip()}", add_special_tokens=False
    )
    if not tokens or any(type(value) is not int for value in tokens):
        raise PromptLogprobClientError("tokenizer returned invalid prompt token IDs")
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    if tokens.count(image_token_id) != 1:
        raise PromptLogprobClientError("scored prompt must contain exactly one image token")
    return tokens


def build_prompt_logprob_request(
    *, prompt_token_ids: list[int], image_bytes: bytes, model: str
) -> bytes:
    """Build the native Megatron completion request for a scored prompt."""
    if not prompt_token_ids or any(type(value) is not int for value in prompt_token_ids):
        raise PromptLogprobClientError("prompt token IDs must be nonempty integers")
    if type(image_bytes) is not bytes or not image_bytes:
        raise PromptLogprobClientError("image bytes must be nonempty")
    if type(model) is not str or not model.strip() or model != model.strip():
        raise PromptLogprobClientError("model must be a nonempty stripped string")
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "multi_modal_data": {"image": [base64.b64encode(image_bytes).decode("ascii")]},
        "echo": True,
        "logprobs": 0,
        "max_tokens": 0,
        "temperature": 0.0,
        "top_p": 1.0,
        "stop": [],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def parse_prompt_logprob_response(raw: bytes) -> PromptLogprobs:
    """Parse the Megatron legacy completion logprob arrays with exact alignment."""
    try:
        payload = load_json_bytes(raw, maximum_bytes=64 * 1024 * 1024)
    except StrictJsonError as error:
        raise PromptLogprobClientError("response is not strict UTF-8 JSON") from error
    if type(payload) is not dict:
        raise PromptLogprobClientError("response JSON must be an object")
    choices = payload.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise PromptLogprobClientError("response must contain exactly one choice")
    choice = choices[0]
    token_ids = choice.get("prompt_token_ids")
    logprobs = choice.get("logprobs")
    if type(token_ids) is not list or any(type(value) is not int for value in token_ids):
        raise PromptLogprobClientError("response prompt_token_ids are invalid")
    if type(logprobs) is not dict or type(logprobs.get("token_logprobs")) is not list:
        raise PromptLogprobClientError("response prompt logprobs are missing")
    tokens = logprobs.get("tokens")
    values = logprobs["token_logprobs"]
    if (
        type(tokens) is not list
        or any(type(value) is not str for value in tokens)
        or len(tokens) != len(token_ids)
        or len(values) != len(token_ids)
    ):
        raise PromptLogprobClientError("response prompt tokens and logprobs are misaligned")
    for value in values:
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value > 0
        ):
            raise PromptLogprobClientError("response contains an invalid prompt logprob")
    return PromptLogprobs(list(token_ids), list(tokens), list(values))


class MegatronPromptLogprobClient:
    """Bounded retrying client for multimodal prompt likelihood requests."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout_seconds: float,
        retries: int,
        *,
        allowed_host: str | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout_seconds <= 0 or type(retries) is not int or retries <= 0:
            raise PromptLogprobClientError("timeout and retries must be positive")
        try:
            self.endpoint = validate_local_http_endpoint(
                endpoint, path="/v1/completions", allowed_host=allowed_host
            )
        except ValueError as error:
            raise PromptLogprobClientError(str(error)) from error
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.sleeper = sleeper

    def score_prompt(self, prompt_token_ids: list[int], image_bytes: bytes) -> PromptLogprobs:
        """Request prompt logprobs, retrying transient HTTP and transport failures."""
        body = build_prompt_logprob_request(
            prompt_token_ids=prompt_token_ids, image_bytes=image_bytes, model=self.model
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                raw = post_json_once(
                    self.endpoint,
                    body,
                    timeout_seconds=self.timeout_seconds,
                    maximum_response_bytes=64 * 1024 * 1024,
                )
                return parse_prompt_logprob_response(raw)
            except RetryableJsonHttpError as error:
                last_error = error
            except (TerminalJsonHttpError, JsonHttpResponseTooLargeError) as error:
                raise PromptLogprobClientError(str(error)) from error
            if attempt + 1 < self.retries:
                self.sleeper(min(2**attempt, 10))
        raise PromptLogprobClientError("prompt-logprob request retries exhausted") from last_error


__all__ = (
    "MegatronPromptLogprobClient",
    "PromptLogprobClientError",
    "PromptLogprobs",
    "PromptTokenizer",
    "build_scored_prompt_token_ids",
    "build_prompt_logprob_request",
    "parse_prompt_logprob_response",
    "resolve_image_token",
)
