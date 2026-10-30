"""Bounded health and generation readiness evidence for a local server."""

from __future__ import annotations

import copy
import math
import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from ..utils import StrictJsonError, VLMEVAL_FAILURE_SENTINEL, load_json_bytes

_MAXIMUM_BODY_BYTES = 65_536
_RETRYABLE_HTTP_STATUS = 503
_DATASET_LABEL = "nvbench_preflight"
_TEXT_PROMPT = "Reply briefly to confirm text generation is ready."
_IMAGE_PROMPT = "Describe this image in one short phrase."


class ServerHealthPreflightError(RuntimeError):
    """Base error for bounded server health polling."""


class ServerHealthConfigurationError(ValueError):
    """Raised when health polling configuration is unsafe or invalid."""


class ServerHealthEarlyExitError(ServerHealthPreflightError):
    """Raised when the observed server launcher exits before readiness."""


class ServerHealthTimeoutError(ServerHealthPreflightError):
    """Raised when the server does not become ready before the deadline."""


class ServerHealthResponseError(ServerHealthPreflightError):
    """Raised for a terminal HTTP status or malformed health response."""


class ServerGenerationPreflightError(RuntimeError):
    """Base error for the bounded generation readiness request pair."""


class ServerGenerationConfigurationError(ValueError):
    """Raised when the client or preflight image contract is invalid."""


class ServerGenerationResponseError(ServerGenerationPreflightError):
    """Raised when a preflight response is empty, invalid, or failed."""


@dataclass(frozen=True)
class ServerHealthEvidence:
    """Strict response values and timing evidence from successful health polling."""

    endpoint: str
    status: str
    ready: bool
    service: str | None
    attempt_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class ServerGenerationEvidence:
    """Validated text and image responses returned by generation readiness checks."""

    text_response: str
    image_response: str


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        """Reject redirects so health probes remain on the validated endpoint."""
        return None


def _default_opener(request: Request, *, timeout: float):
    opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
    return opener.open(request, timeout=timeout)


def _positive_finite(value: object, label: str) -> float:
    valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    if not valid:
        raise ServerHealthConfigurationError(f"{label} must be finite and positive")
    return float(value)


def _validated_health_endpoint(value: object, allowed_host: str | None = None) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ServerHealthConfigurationError("health endpoint must be a nonempty string")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ServerHealthConfigurationError("health endpoint port is invalid") from error
    if allowed_host is not None and (
        type(allowed_host) is not str
        or not allowed_host
        or allowed_host != allowed_host.strip()
        or any(character in allowed_host for character in "/?#@\0")
    ):
        raise ServerHealthConfigurationError("allowed health host is invalid")
    host = "127.0.0.1" if allowed_host is None else allowed_host
    valid = (
        parsed.scheme == "http"
        and parsed.hostname == host
        and type(port) is int
        and 1024 <= port <= 65_535
        and parsed.netloc == f"{host}:{port}"
        and parsed.path == "/v1/health"
        and not parsed.query
        and not parsed.fragment
        and parsed.username is None
        and parsed.password is None
    )
    if not valid:
        raise ServerHealthConfigurationError(
            "health endpoint must be credential-free allowed HTTP /v1/health"
        )
    return value


def _parse_ready_body(raw: object) -> tuple[bool, str, str | None]:
    if type(raw) is not bytes or not raw or len(raw) > _MAXIMUM_BODY_BYTES:
        raise ServerHealthResponseError("health response has an invalid byte body")
    try:
        payload = load_json_bytes(raw, maximum_bytes=_MAXIMUM_BODY_BYTES)
    except StrictJsonError as error:
        raise ServerHealthResponseError("health response is not strict UTF-8 JSON") from error
    if type(payload) is not dict:
        raise ServerHealthResponseError("health response JSON must be an object")
    status, ready = payload.get("status"), payload.get("ready")
    if type(status) is not str or not status or type(ready) is not bool:
        raise ServerHealthResponseError("health status and ready fields have wrong types")
    service = payload.get("service")
    if service is not None and (
        type(service) is not str or not service or service != service.strip()
    ):
        raise ServerHealthResponseError("health service field has the wrong type")
    return status == "ok" and ready is True, status, service


def _health_attempt(
    endpoint: str,
    request_timeout: float,
    opener: Callable[..., object],
) -> tuple[bool, str, str | None] | None:
    request = Request(endpoint, method="GET")
    try:
        with opener(request, timeout=request_timeout) as response:  # type: ignore[attr-defined]
            status_code = getattr(response, "status", None)
            if status_code == _RETRYABLE_HTTP_STATUS:
                return None
            if type(status_code) is not int or status_code != 200:
                raise ServerHealthResponseError("health endpoint returned a terminal HTTP status")
            raw = response.read(_MAXIMUM_BODY_BYTES + 1)
    except HTTPError as error:
        status_code = error.code
        error.close()
        if status_code == _RETRYABLE_HTTP_STATUS:
            return None
        raise ServerHealthResponseError(
            "health endpoint returned a terminal HTTP status"
        ) from error
    except (URLError, TimeoutError, OSError):
        return None
    return _parse_ready_body(raw)


def wait_for_server_health(
    health_endpoint: str,
    readiness_timeout_seconds: float,
    process_poll: Callable[[], int | None],
    *,
    request_timeout_seconds: float = 2.0,
    poll_interval_seconds: float = 0.25,
    opener: Callable[..., object] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    allowed_host: str | None = None,
) -> ServerHealthEvidence:
    """Wait for one ready health response and return immutable polling evidence.

    The endpoint must be the credential-free allowed route and ``process_poll``
    observes the owned launcher. This does not start or stop a server. Invalid
    configuration, early exit, timeout, and terminal responses raise typed errors.
    """
    endpoint = _validated_health_endpoint(health_endpoint, allowed_host)
    readiness_timeout = _positive_finite(readiness_timeout_seconds, "readiness timeout")
    request_timeout = _positive_finite(request_timeout_seconds, "request timeout")
    poll_interval = _positive_finite(poll_interval_seconds, "poll interval")
    if not callable(process_poll) or not callable(monotonic) or not callable(sleeper):
        raise ServerHealthConfigurationError("health polling seams must be callable")
    open_request = _default_opener if opener is None else opener
    if not callable(open_request):
        raise ServerHealthConfigurationError("health opener must be callable")
    started = monotonic()
    if type(started) not in (int, float) or not math.isfinite(started):
        raise ServerHealthConfigurationError("monotonic clock returned an invalid value")
    attempts = 0
    while True:
        return_code = process_poll()
        if return_code is not None:
            if type(return_code) is not int:
                raise ServerHealthPreflightError("process poll returned an invalid value")
            raise ServerHealthEarlyExitError(
                f"server process exited before readiness with return code {return_code}"
            )
        now = monotonic()
        if type(now) not in (int, float) or not math.isfinite(now) or now < started:
            raise ServerHealthPreflightError("monotonic clock became invalid")
        remaining = readiness_timeout - (now - started)
        if remaining <= 0:
            raise ServerHealthTimeoutError("server health readiness deadline expired")
        attempts += 1
        result = _health_attempt(endpoint, min(request_timeout, remaining), open_request)
        finished = monotonic()
        if (
            type(finished) not in (int, float)
            or not math.isfinite(finished)
            or finished < started
        ):
            raise ServerHealthPreflightError("monotonic clock became invalid")
        elapsed = finished - started
        if result is not None and result[0] and elapsed <= readiness_timeout:
            return ServerHealthEvidence(endpoint, result[1], True, result[2], attempts, elapsed)
        if elapsed >= readiness_timeout:
            raise ServerHealthTimeoutError("server health readiness deadline expired")
        sleeper(min(poll_interval, readiness_timeout - elapsed))


def _shared_image(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ServerGenerationConfigurationError("preflight image must be an absolute Path")
    try:
        resolved = path.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as error:
        raise ServerGenerationConfigurationError("preflight image is unavailable") from error
    if not stat.S_ISREG(mode) or not os.access(resolved, os.R_OK):
        raise ServerGenerationConfigurationError("preflight image is not a readable file")
    if not mode & stat.S_IRGRP:
        raise ServerGenerationConfigurationError("preflight image is not group-readable")
    return resolved


def _response(value: object, label: str) -> str:
    valid = type(value) is str and bool(value.strip())
    if not valid:
        raise ServerGenerationResponseError(f"{label} preflight response is empty or invalid")
    if VLMEVAL_FAILURE_SENTINEL in value:
        raise ServerGenerationResponseError(f"{label} preflight returned failure sentinel")
    return value


def _generate_preserving_message(client: object, message: list[dict[str, object]]) -> object:
    before = copy.deepcopy(message)
    result = client.generate(message, dataset=_DATASET_LABEL)  # type: ignore[attr-defined]
    if message != before:
        raise ServerGenerationPreflightError("generation client mutated the preflight message")
    return result


def run_server_generation_preflight(
    client: object,
    image_path: Path,
) -> ServerGenerationEvidence:
    """Require text and image responses, returning immutable readiness evidence.

    ``client`` must implement the API generation seam and ``image_path`` must be
    an absolute readable group-readable regular file. Neither input is mutated;
    invalid inputs, mutated messages, empty responses, and sentinels raise typed
    configuration, response, or preflight errors. This does not manage a server.
    """
    if getattr(client, "is_api", None) is not True or not callable(
        getattr(client, "generate", None)
    ):
        raise ServerGenerationConfigurationError("client must satisfy the API generation contract")
    image = _shared_image(image_path)
    text_message: list[dict[str, object]] = [
        {"role": "user", "content": [{"type": "text", "value": _TEXT_PROMPT}]}
    ]
    text_response = _response(_generate_preserving_message(client, text_message), "text")
    image_message: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "value": str(image)},
                {"type": "text", "value": _IMAGE_PROMPT},
            ],
        }
    ]
    image_response = _response(_generate_preserving_message(client, image_message), "image")
    return ServerGenerationEvidence(text_response, image_response)
