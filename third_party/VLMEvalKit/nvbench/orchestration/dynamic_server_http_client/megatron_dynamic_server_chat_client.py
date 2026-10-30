"""Validated chat-completions protocol and transport for a local Megatron server."""

from __future__ import annotations
import base64
import binascii
import json
import math
import os
import stat
import time
from collections.abc import Callable
from pathlib import Path

from ..configured_eval_run import GenerationConfig
from ..utils import VLMEVAL_FAILURE_SENTINEL, StrictJsonError, load_json_bytes
from .json_http_transport import (
    JsonHttpResponseTooLargeError,
    RetryableJsonHttpError,
    TerminalJsonHttpError,
    post_json_once,
    validate_local_http_endpoint,
)

ImageEncoder = Callable[[Path, int], str]
_MAXIMUM_RESPONSE_BYTES = 16_777_216
_SUPPORTED_IMAGE_DATA_URL_PREFIXES = frozenset(
    {
        "data:image/gif;base64,",
        "data:image/jpeg;base64,",
        "data:image/png;base64,",
        "data:image/webp;base64,",
    }
)


class MegatronDynamicServerChatClientError(RuntimeError):
    """Base error for the owned dynamic-server client boundary."""


class ChatRequestValidationError(ValueError):
    """Raised when an outbound chat request violates the local contract."""


class ChatResponseContractError(MegatronDynamicServerChatClientError):
    """Raised when a server response violates the expected wire contract."""


class _EmptyAssistantResponseError(ChatResponseContractError):
    """Raised when an otherwise valid response contains no assistant text."""


class TransientRequestExhaustedError(MegatronDynamicServerChatClientError):
    """Raised after every configured transient request retry fails."""


class TerminalHTTPError(MegatronDynamicServerChatClientError):
    """Raised for a non-retryable or unexpected HTTP status."""


class _RetryableRequestError(Exception):
    pass


def _request(condition: bool, message: str) -> None:
    if not condition:
        raise ChatRequestValidationError(message)


def _response(condition: bool, message: str) -> None:
    if not condition:
        raise ChatResponseContractError(message)


def validate_model_identifier(value: object) -> str:
    """Return a nonempty stripped model identifier without mutating ``value``.

    Invalid values raise :class:`ChatRequestValidationError`.
    """
    valid = type(value) is str and bool(value) and value == value.strip()
    _request(valid, "model identifier must be a nonempty stripped string")
    return value


def validate_generation(value: object) -> GenerationConfig:
    """Return a valid ``GenerationConfig`` without mutating its fields.

    Invalid type or generation settings raise :class:`ChatRequestValidationError`.
    """
    _request(type(value) is GenerationConfig, "generation must be a GenerationConfig")
    generation = value
    temperature_ok = type(generation.temperature) in (int, float)
    stop_ok = type(generation.stop) is tuple and bool(generation.stop)
    checks = (
        (temperature_ok and math.isfinite(generation.temperature), "finite temperature"),
        (temperature_ok and generation.temperature >= 0, "nonnegative temperature"),
        (type(generation.retry) is int and generation.retry > 0, "positive retry"),
        (
            type(generation.image_size) is int
            and (generation.image_size == -1 or generation.image_size > 0),
            "image_size",
        ),
        (
            type(generation.image_detail) is str and generation.image_detail in {"high", "low"},
            "image_detail",
        ),
        (type(generation.max_new_tokens) is int and generation.max_new_tokens > 0, "max tokens"),
        (stop_ok, "nonempty stop tuple"),
        (stop_ok and all(type(item) is str and bool(item) for item in generation.stop), "stops"),
        (type(generation.enable_thinking) is bool, "boolean enable_thinking"),
    )
    for valid, label in checks:
        _request(valid, f"invalid generation setting: {label}")
    return generation


def validate_image_encoder(value: object) -> ImageEncoder:
    """Return ``value`` or the native image encoder without mutating either.

    A non-callable non-``None`` value raises :class:`ChatRequestValidationError`.
    """
    _request(value is None or callable(value), "invalid image encoder")
    return _default_image_encoder if value is None else value


def _default_image_encoder(image_path: Path, target_size: int) -> str:
    from PIL import Image

    from vlmeval.smp import encode_image_to_base64

    try:
        with Image.open(image_path) as image:
            image.load()
            return encode_image_to_base64(image, target_size=target_size)
    except OSError as error:
        raise ChatRequestValidationError(f"cannot decode image: {image_path}") from error


def _canonical_base64(encoded: object) -> str:
    _request(type(encoded) is str and bool(encoded), "image encoder returned no base64 data")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ChatRequestValidationError("image encoder returned invalid base64 data") from error
    canonical = base64.b64encode(decoded).decode("ascii")
    _request(bool(decoded) and canonical == encoded, "image encoder returned noncanonical base64")
    return encoded


def _image_url(value: str, size: int, encoder: ImageEncoder) -> str:
    for prefix in _SUPPORTED_IMAGE_DATA_URL_PREFIXES:
        if value.startswith(prefix):
            encoded = value.removeprefix(prefix)
            return f"{prefix}{_canonical_base64(encoded)}"

    candidate = Path(value)
    _request(candidate.is_absolute(), "image path must be absolute")
    try:
        path = candidate.resolve(strict=True)
        mode = path.stat().st_mode
    except OSError as error:
        raise ChatRequestValidationError(f"image path is unavailable: {candidate}") from error
    _request(stat.S_ISREG(mode) and os.access(path, os.R_OK), "image is not a readable file")
    try:
        encoded = encoder(path, size)
    except ChatRequestValidationError:
        raise
    except Exception as error:
        raise ChatRequestValidationError(f"cannot encode image: {path}") from error
    return f"data:image/jpeg;base64,{_canonical_base64(encoded)}"


def _converted_messages(
    message: object, image_size: int, image_detail: str, encoder: ImageEncoder
) -> list[dict[str, object]]:
    _request(type(message) is list and bool(message), "message must be a nonempty role list")
    converted: list[dict[str, object]] = []
    for position, turn in enumerate(message):
        _request(type(turn) is dict and set(turn) == {"role", "content"}, "invalid turn")
        role = "user" if position % 2 == 0 else "assistant"
        _request(turn["role"] == role, "conversation roles must alternate from user")
        content = turn["content"]
        _request(type(content) is list and bool(content), "turn content must be nonempty")
        converted_content: list[dict[str, object]] = []
        for item in content:
            _request(type(item) is dict and set(item) == {"type", "value"}, "invalid item")
            kind, item_value = item["type"], item["value"]
            _request(type(item_value) is str and bool(item_value.strip()), "empty item value")
            if kind == "text":
                converted_content.append({"type": "text", "text": item_value})
            elif kind == "image" and role == "user":
                image_url = _image_url(item_value, image_size, encoder)
                converted_content.append(
                    {"type": "image_url", "image_url": {"url": image_url, "detail": image_detail}}
                )
            else:
                raise ChatRequestValidationError("invalid content type for role")
        if role == "assistant":
            valid = len(converted_content) == 1 and converted_content[0]["type"] == "text"
            _request(valid, "assistant turn must contain exactly one text item")
        else:
            _request(any(item["type"] == "text" for item in converted_content), "user needs text")
        converted.append({"role": role, "content": converted_content})
    _request(converted[-1]["role"] == "user", "conversation must end with user")
    return converted


def build_chat_request_body(
    message: object,
    model_identifier: str,
    generation: GenerationConfig,
    image_encoder: ImageEncoder,
) -> bytes:
    """Build deterministic chat JSON without mutating ``message``.

    Image paths are read and encoded with ``image_encoder``; canonical raster
    data URLs emitted by native datasets are passed through. Invalid roles,
    items, images, encodings, identity, or generation settings raise
    :class:`ChatRequestValidationError`.
    """
    payload = {
        "model": model_identifier,
        "messages": _converted_messages(
            message, generation.image_size, generation.image_detail, image_encoder
        ),
        "n": 1,
        "temperature": generation.temperature,
        "max_tokens": generation.max_new_tokens,
        "stop": list(generation.stop),
        "enable_thinking": generation.enable_thinking,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def parse_chat_response(raw: bytes) -> str:
    """Return assistant text from one response payload without mutation.

    Malformed UTF-8/JSON, invalid choices, empty text, and the standard failure
    sentinel raise :class:`ChatResponseContractError`; callers classify retries.
    """
    try:
        payload = load_json_bytes(raw, maximum_bytes=_MAXIMUM_RESPONSE_BYTES)
    except StrictJsonError as error:
        raise ChatResponseContractError("response is not valid UTF-8 JSON") from error
    _response(type(payload) is dict, "response JSON must be an object")
    choices = payload.get("choices")
    valid = type(choices) is list and len(choices) == 1 and type(choices[0]) is dict
    _response(valid, "response must contain exactly one object choice")
    choice = choices[0]
    _response(type(choice.get("index")) is int and choice["index"] == 0, "invalid choice index")
    _response(choice.get("finish_reason") in {"stop", "length"}, "invalid finish reason")
    assistant = choice.get("message")
    _response(type(assistant) is dict and assistant.get("role") == "assistant", "invalid message")
    content = assistant.get("content")
    _response(type(content) is str, "assistant content must be text")
    if not content.strip():
        raise _EmptyAssistantResponseError("assistant content is empty")
    _response(VLMEVAL_FAILURE_SENTINEL not in content, "VLMEvalKit failure sentinel returned")
    return content


def _validated_endpoint(value: object, allowed_host: str | None = None) -> str:
    try:
        return validate_local_http_endpoint(
            value, path="/v1/chat/completions", allowed_host=allowed_host
        )
    except ValueError as error:
        raise ChatRequestValidationError(str(error)) from error


class MegatronDynamicServerChatClient:
    """Bounded HTTP adapter implementing VLMEvalKit's API model seam.

    Construction validates endpoint, optional scheduler-owned remote host,
    immutable generation policy, retry bounds, and encoder. ``generate`` sends
    one deterministic request, retries only transient failures, never mutates
    its input, and returns text or typed errors.
    """

    is_api = True

    def __init__(
        self,
        endpoint: str,
        model_identifier: str,
        generation: GenerationConfig,
        request_timeout_seconds: float,
        *,
        retry_backoff_seconds: float = 0.05,
        maximum_response_bytes: int = 1_048_576,
        sleeper: Callable[[float], None] = time.sleep,
        image_encoder: Callable[[Path, int], str] | None = None,
        allowed_host: str | None = None,
    ) -> None:
        self.endpoint = _validated_endpoint(endpoint, allowed_host)
        self.model_identifier = validate_model_identifier(model_identifier)
        self.generation = validate_generation(generation)
        timeout_ok = type(request_timeout_seconds) in (int, float) and math.isfinite(
            request_timeout_seconds
        )
        backoff_ok = type(retry_backoff_seconds) in (int, float) and math.isfinite(
            retry_backoff_seconds
        )
        checks = (
            (timeout_ok and request_timeout_seconds > 0, "request timeout"),
            (backoff_ok and 0 <= retry_backoff_seconds <= 10, "retry backoff"),
            (
                type(maximum_response_bytes) is int
                and 1 <= maximum_response_bytes <= _MAXIMUM_RESPONSE_BYTES,
                "response byte limit",
            ),
            (callable(sleeper), "sleeper"),
        )
        for valid, label in checks:
            _request(valid, f"invalid {label}")
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.maximum_response_bytes = maximum_response_bytes
        self._sleeper = sleeper
        self._image_encoder: ImageEncoder = validate_image_encoder(image_encoder)

    def _send(self, body: bytes) -> bytes:
        try:
            return post_json_once(
                self.endpoint,
                body,
                timeout_seconds=self.request_timeout_seconds,
                maximum_response_bytes=self.maximum_response_bytes,
            )
        except RetryableJsonHttpError as error:
            raise _RetryableRequestError(str(error)) from error
        except TerminalJsonHttpError as error:
            raise TerminalHTTPError(str(error)) from error
        except JsonHttpResponseTooLargeError as error:
            raise ChatResponseContractError(str(error)) from error

    def generate(self, message: object, dataset: str | None = None) -> str:
        """Send ``message`` and return one assistant response without mutation.

        ``dataset`` serves only the upstream API seam. Invalid requests, terminal
        HTTP failures, retry exhaustion, and response-contract failures raise the
        corresponding typed exception.
        """
        valid_dataset = dataset is None or (type(dataset) is str and bool(dataset.strip()))
        _request(valid_dataset, "dataset must be a nonempty string or null")
        body = build_chat_request_body(
            message, self.model_identifier, self.generation, self._image_encoder
        )
        for attempt in range(self.generation.retry):
            try:
                return parse_chat_response(self._send(body))
            except (_RetryableRequestError, _EmptyAssistantResponseError) as error:
                if attempt + 1 == self.generation.retry:
                    if isinstance(error, _EmptyAssistantResponseError):
                        return VLMEVAL_FAILURE_SENTINEL
                    raise TransientRequestExhaustedError(
                        f"transient request failed after {self.generation.retry} attempts: {error}"
                    ) from error
                delay = min(self.retry_backoff_seconds * (attempt + 1), 10.0)
                self._sleeper(delay)
