"""Small shared HTTP mechanics for local dynamic-server JSON protocols."""

from __future__ import annotations
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class JsonHttpTransportError(RuntimeError):
    """Base error for one local JSON HTTP request."""


class RetryableJsonHttpError(JsonHttpTransportError):
    """Raised for a retryable status or connection failure."""


class TerminalJsonHttpError(JsonHttpTransportError):
    """Raised for a non-retryable HTTP status."""


class JsonHttpResponseTooLargeError(JsonHttpTransportError):
    """Raised when a response exceeds its caller-owned byte limit."""


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def validate_local_http_endpoint(
    value: object, *, path: str, allowed_host: str | None = None
) -> str:
    """Validate one credential-free endpoint on the configured local server host."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("invalid endpoint")
    if type(path) is not str or not path.startswith("/"):
        raise ValueError("invalid endpoint path contract")
    if allowed_host is not None and (
        type(allowed_host) is not str
        or not allowed_host
        or allowed_host != allowed_host.strip()
        or any(character in allowed_host for character in "/?#@\0")
    ):
        raise ValueError("allowed endpoint host is invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValueError("invalid endpoint port") from error
    hosts = {"127.0.0.1", "localhost", "::1"} if allowed_host is None else {allowed_host}
    valid = (
        parsed.scheme == "http"
        and parsed.hostname in hosts
        and port is not None
        and parsed.path == path
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )
    if not valid:
        raise ValueError(f"endpoint must be credential-free allowed HTTP {path}")
    return value


def same_http_origin(left: str, right: str) -> bool:
    """Return whether two already-validated HTTP endpoints share one server origin."""
    first = urlsplit(left)
    second = urlsplit(right)
    return (first.scheme, first.hostname, first.port) == (
        second.scheme,
        second.hostname,
        second.port,
    )


def post_json_once(
    endpoint: str, body: bytes, *, timeout_seconds: float, maximum_response_bytes: int
) -> bytes:
    """POST one JSON body with redirects/proxies disabled and a bounded response."""
    request = Request(endpoint, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise TerminalJsonHttpError(f"unexpected HTTP status {response.status}")
            raw = response.read(maximum_response_bytes + 1)
    except HTTPError as error:
        status = error.code
        error.close()
        if status in RETRYABLE_HTTP_STATUSES:
            raise RetryableJsonHttpError(f"retryable HTTP status {status}") from error
        raise TerminalJsonHttpError(f"terminal HTTP status {status}") from error
    except OSError as error:
        raise RetryableJsonHttpError(
            f"transient connection failure: {type(error).__name__}"
        ) from error
    if len(raw) > maximum_response_bytes:
        raise JsonHttpResponseTooLargeError("response exceeds byte limit")
    return raw


__all__ = (
    "JsonHttpResponseTooLargeError",
    "RetryableJsonHttpError",
    "TerminalJsonHttpError",
    "post_json_once",
    "same_http_origin",
    "validate_local_http_endpoint",
)
