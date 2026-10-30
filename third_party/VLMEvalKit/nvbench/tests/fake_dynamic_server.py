"""Context-managed loopback chat-completions server for client tests."""

from __future__ import annotations

import copy
import json
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def chat_completion_response(
    text: object = "answer", finish_reason: object = "stop", index: object = 0
) -> dict[str, object]:
    return {
        "choices": [
            {
                "index": index,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ]
    }


@dataclass(frozen=True)
class QueuedResponse:
    status: int = 200
    body: object = field(default_factory=chat_completion_response)
    delay_seconds: float = 0.0
    disconnect: bool = False


@dataclass(frozen=True)
class RecordedRequest:
    body: object
    headers: dict[str, str]


class _FixtureState:
    def __init__(self, responses: tuple[QueuedResponse, ...]) -> None:
        self.responses = deque(responses)
        self.requests: list[RecordedRequest] = []
        self.lock = threading.Lock()

    def record_and_pop(self, request: RecordedRequest) -> QueuedResponse:
        with self.lock:
            self.requests.append(request)
            if self.responses:
                return self.responses.popleft()
        return QueuedResponse(status=500, body={"error": "response queue exhausted"})


class _LoopbackServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body: object = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = raw
        headers = {key.lower(): value for key, value in self.headers.items()}
        state = self.server.fixture_state  # type: ignore[attr-defined]
        response = state.record_and_pop(RecordedRequest(copy.deepcopy(body), headers))
        if response.delay_seconds:
            time.sleep(response.delay_seconds)
        if response.disconnect:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            return
        if isinstance(response.body, bytes):
            payload = response.body
        elif isinstance(response.body, str):
            payload = response.body.encode("utf-8")
        else:
            payload = json.dumps(response.body).encode("utf-8")
        self.send_response(response.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, _format: str, *_args: object) -> None:
        pass


class FakeDynamicServer:
    def __init__(self, responses: list[QueuedResponse] | tuple[QueuedResponse, ...]) -> None:
        self._state = _FixtureState(tuple(responses))
        self._server: _LoopbackServer | None = None
        self._thread: threading.Thread | None = None
        self._endpoint: str | None = None

    @property
    def endpoint(self) -> str:
        if self._endpoint is None:
            raise RuntimeError("fake server has not started")
        return self._endpoint

    @property
    def requests(self) -> tuple[RecordedRequest, ...]:
        with self._state.lock:
            return tuple(copy.deepcopy(self._state.requests))

    @property
    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "FakeDynamicServer":
        if self._server is not None:
            raise RuntimeError("fake server cannot be entered twice")
        server = _LoopbackServer(("127.0.0.1", 0), _Handler)
        server.fixture_state = self._state  # type: ignore[attr-defined]
        port = server.server_address[1]
        self._endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name=f"fake-dynamic-server-{port}",
            kwargs={"poll_interval": 0.01},
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._server is None or self._thread is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("fake server thread did not stop")
