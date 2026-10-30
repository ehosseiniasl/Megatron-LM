from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from urllib.error import URLError

from orchestration.dynamic_server_http_client.server_preflight import (
    ServerHealthConfigurationError,
    ServerHealthEarlyExitError,
    ServerHealthEvidence,
    ServerHealthPreflightError,
    ServerHealthResponseError,
    ServerHealthTimeoutError,
    wait_for_server_health,
)
from orchestration.dynamic_server_http_client import (
    server_preflight as generation_preflight,
)


ENDPOINT = "http://127.0.0.1:8123/v1/health"

ServerGenerationConfigurationError = generation_preflight.ServerGenerationConfigurationError
ServerGenerationEvidence = generation_preflight.ServerGenerationEvidence
ServerGenerationPreflightError = generation_preflight.ServerGenerationPreflightError
ServerGenerationResponseError = generation_preflight.ServerGenerationResponseError
run_server_generation_preflight = generation_preflight.run_server_generation_preflight


class MemoryResponse:
    def __init__(self, status: object, body: bytes) -> None:
        self.status = status
        self.body = body

    def __enter__(self) -> "MemoryResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self, maximum: int) -> bytes:
        return self.body[:maximum]


class QueueOpener:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str, float]] = []

    def __call__(self, request, *, timeout: float) -> object:
        self.calls.append((request.full_url, request.get_method(), timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClock:
    def __init__(self) -> None:
        self.value = 10.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.value += duration


def response(payload: object, status: object = 200) -> MemoryResponse:
    return MemoryResponse(status, json.dumps(payload).encode("utf-8"))


def wait(
    request_opener: QueueOpener,
    clock: FakeClock,
    **changes: object,
) -> ServerHealthEvidence:
    arguments: dict[str, object] = {
        "health_endpoint": ENDPOINT,
        "readiness_timeout_seconds": 2.0,
        "request_timeout_seconds": 0.5,
        "poll_interval_seconds": 0.1,
        "process_poll": lambda: None,
        "opener": request_opener,
        "monotonic": clock.monotonic,
        "sleeper": clock.sleep,
    }
    arguments.update(changes)
    return wait_for_server_health(**arguments)  # type: ignore[arg-type]


class ServerHealthPreflightTest(unittest.TestCase):
    def test_ready_response_returns_frozen_evidence_and_exact_request(self) -> None:
        opener = QueueOpener([response({"status": "ok", "ready": True, "service": "mimo"})])
        clock = FakeClock()
        evidence = wait(opener, clock)
        self.assertEqual(
            evidence,
            ServerHealthEvidence(ENDPOINT, "ok", True, "mimo", 1, 0.0),
        )
        self.assertEqual(opener.calls, [(ENDPOINT, "GET", 0.5)])
        self.assertEqual(clock.sleeps, [])
        with self.assertRaises(FrozenInstanceError):
            evidence.ready = False  # type: ignore[misc]

    def test_connection_503_and_not_ready_responses_retry_in_order(self) -> None:
        opener = QueueOpener(
            [
                URLError("starting"),
                response({"ignored": True}, status=503),
                response({"status": "starting", "ready": False}),
                response({"status": "ok", "ready": True}),
            ]
        )
        clock = FakeClock()
        polls: list[None] = []
        evidence = wait(opener, clock, process_poll=lambda: polls.append(None))
        self.assertEqual(evidence.attempt_count, 4)
        self.assertAlmostEqual(evidence.elapsed_seconds, 0.3)
        self.assertEqual(len(polls), 4)
        self.assertEqual(clock.sleeps, [0.1, 0.1, 0.1])

    def test_process_exit_and_deadline_are_terminal(self) -> None:
        opener = QueueOpener([])
        with self.assertRaisesRegex(ServerHealthEarlyExitError, "return code 17"):
            wait(opener, FakeClock(), process_poll=lambda: 17)
        self.assertEqual(opener.calls, [])

        opener = QueueOpener(
            [response({"status": "starting", "ready": False}) for _ in range(3)]
        )
        clock = FakeClock()
        with self.assertRaises(ServerHealthTimeoutError):
            wait(
                opener,
                clock,
                readiness_timeout_seconds=0.25,
                poll_interval_seconds=0.1,
            )
        self.assertEqual(len(opener.calls), 3)
        self.assertAlmostEqual(sum(clock.sleeps), 0.25)

    def test_malformed_or_oversized_200_response_fails_immediately(self) -> None:
        malformed = (
            b'{"status":"ok","status":"ok","ready":true}',
            b'{"status":"ok","ready":true,"value":NaN}',
            b"[]",
            b'{"status":1,"ready":true}',
            b'{"status":"ok","ready":1}',
            b'{"status":"ok","ready":true,"service":1}',
            b"\xff",
            b"x" * 65_537,
        )
        for body in malformed:
            with self.subTest(body=body[:40]):
                opener = QueueOpener([MemoryResponse(200, body)])
                clock = FakeClock()
                with self.assertRaises(ServerHealthResponseError):
                    wait(opener, clock)
                self.assertEqual(len(opener.calls), 1)
                self.assertEqual(clock.sleeps, [])

    def test_terminal_http_status_and_invalid_poll_value_fail(self) -> None:
        with self.assertRaises(ServerHealthResponseError):
            wait(QueueOpener([response({}, 404)]), FakeClock())
        with self.assertRaisesRegex(ServerHealthEarlyExitError, "return code 0"):
            wait(QueueOpener([]), FakeClock(), process_poll=lambda: 0)
        with self.assertRaisesRegex(ServerHealthPreflightError, "process poll"):
            wait(QueueOpener([]), FakeClock(), process_poll=lambda: True)

    def test_nonexact_loopback_endpoints_are_rejected(self) -> None:
        invalid = (
            "https://127.0.0.1:8123/v1/health",
            "http://localhost:8123/v1/health",
            "http://user@127.0.0.1:8123/v1/health",
            "http://127.0.0.1:80/v1/health",
            "http://127.0.0.1:8123/v1/health?x=1",
            "http://127.0.0.1:8123/v1/health#fragment",
            "http://127.0.0.1:8123/v1/chat/completions",
            "http://127.0.0.1/v1/health",
        )
        for endpoint in invalid:
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ServerHealthConfigurationError):
                    wait(QueueOpener([]), FakeClock(), health_endpoint=endpoint)

    def test_explicit_remote_host_is_accepted_but_must_match(self) -> None:
        opener = QueueOpener([response({"status": "ok", "ready": True}, 200)])
        evidence = wait_for_server_health(
            "http://server-node-0:8123/v1/health",
            10,
            lambda: None,
            allowed_host="server-node-0",
            opener=opener,
            monotonic=FakeClock().monotonic,
            sleeper=lambda _seconds: None,
        )
        self.assertTrue(evidence.ready)
        with self.assertRaises(ServerHealthConfigurationError):
            wait_for_server_health(
                "http://server-node-1:8123/v1/health",
                10,
                lambda: None,
                allowed_host="server-node-0",
            )

    def test_timing_and_callable_configuration_fails_closed(self) -> None:
        cases = (
            {"readiness_timeout_seconds": 0},
            {"readiness_timeout_seconds": True},
            {"readiness_timeout_seconds": float("nan")},
            {"request_timeout_seconds": 0},
            {"poll_interval_seconds": -1},
            {"process_poll": None},
            {"opener": object()},
            {"monotonic": None},
            {"sleeper": None},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ServerHealthConfigurationError):
                    wait(QueueOpener([]), FakeClock(), **changes)


class RecordingClient:
    is_api = True

    def __init__(self, responses: tuple[object, ...] = ("text ready", "image ready")) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[object, object]] = []

    def generate(self, message: object, dataset: str | None = None) -> object:
        self.calls.append((copy.deepcopy(message), dataset))
        return self.responses.pop(0)


def shared_image(root: Path) -> Path:
    path = root / "preflight.png"
    path.write_bytes(b"synthetic-image")
    path.chmod(0o640)
    return path


class ServerGenerationPreflightTest(unittest.TestCase):
    def test_sends_exact_text_then_image_messages_and_returns_frozen_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = shared_image(root)
            before = image.read_bytes()
            client = RecordingClient()
            evidence = run_server_generation_preflight(client, image)

            self.assertEqual(evidence, ServerGenerationEvidence("text ready", "image ready"))
            self.assertEqual(image.read_bytes(), before)
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(client.calls[0][1], "nvbench_preflight")
            self.assertEqual(client.calls[1][1], "nvbench_preflight")
            self.assertEqual(
                client.calls[0][0],
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "value": "Reply briefly to confirm text generation is ready.",
                            }
                        ],
                    }
                ],
            )
            self.assertEqual(
                client.calls[1][0],
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "value": str(image.resolve())},
                            {
                                "type": "text",
                                "value": "Describe this image in one short phrase.",
                            },
                        ],
                    }
                ],
            )
            with self.assertRaises(FrozenInstanceError):
                evidence.text_response = "changed"  # type: ignore[misc]

    def test_invalid_client_or_image_fails_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = shared_image(root)
            invalid_clients = (object(), type("NotAPI", (), {"generate": lambda *_: "x"})())
            for client in invalid_clients:
                with self.subTest(client=type(client).__name__):
                    with self.assertRaises(ServerGenerationConfigurationError):
                        run_server_generation_preflight(client, image)

            client = RecordingClient()
            directory = root / "directory"
            directory.mkdir()
            missing = root / "missing.png"
            private = root / "private.png"
            private.write_bytes(b"private")
            private.chmod(0o600)
            for candidate in (Path("relative.png"), directory, missing, private):
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ServerGenerationConfigurationError):
                        run_server_generation_preflight(client, candidate)
            self.assertEqual(client.calls, [])

    def test_empty_nonstring_and_failure_sentinel_responses_fail(self) -> None:
        cases = (
            ((None,), "text"),
            (("",), "text"),
            (("   ",), "text"),
            (("Failed to obtain answer",), "text"),
            (("text ready", None), "image"),
            (("text ready", "Failed to obtain answer"), "image"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            image = shared_image(Path(temporary))
            for responses, label in cases:
                with self.subTest(responses=responses):
                    client = RecordingClient(responses)
                    with self.assertRaisesRegex(ServerGenerationResponseError, label):
                        run_server_generation_preflight(client, image)

    def test_client_message_mutation_is_detected(self) -> None:
        class MutatingClient(RecordingClient):
            def generate(self, message: object, dataset: str | None = None) -> object:
                message.append({"role": "assistant", "content": []})
                return "mutated"

        with tempfile.TemporaryDirectory() as temporary:
            image = shared_image(Path(temporary))
            with self.assertRaisesRegex(ServerGenerationPreflightError, "mutated"):
                run_server_generation_preflight(MutatingClient(), image)


if __name__ == "__main__":
    unittest.main()
