"""Loopback tests for the strict Megatron dynamic chat-completions client."""

from __future__ import annotations

import base64
import copy
import inspect
import io
import socket
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from fake_dynamic_server import (
    FakeDynamicServer,
    QueuedResponse,
    chat_completion_response,
)
from orchestration.configured_eval_run import GenerationConfig
from orchestration.dynamic_server_http_client import (
    megatron_dynamic_server_chat_client as chat_client,
)

ChatRequestValidationError = chat_client.ChatRequestValidationError
ChatResponseContractError = chat_client.ChatResponseContractError
MegatronDynamicServerChatClient = chat_client.MegatronDynamicServerChatClient
TerminalHTTPError = chat_client.TerminalHTTPError
TransientRequestExhaustedError = chat_client.TransientRequestExhaustedError


class DynamicServerChatClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.red = self.root / "red.png"
        self.blue = self.root / "blue.png"
        Image.new("RGB", (20, 10), "red").save(self.red)
        Image.new("RGB", (12, 18), "blue").save(self.blue)
        self.generation = GenerationConfig(
            temperature=0.0,
            retry=3,
            image_size=-1,
            image_detail="low",
            max_new_tokens=64,
            stop=("\n\n",),
            enable_thinking=False,
            chat_template="minimal",
        )
        self.encoder_calls: list[tuple[Path, int]] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def encode_jpeg(self, path: Path, target_size: int) -> str:
        self.encoder_calls.append((path, target_size))
        with Image.open(path) as image:
            image.load()
            output = io.BytesIO()
            image.save(output, format="JPEG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    def client(
        self,
        endpoint: str,
        *,
        generation: GenerationConfig | None = None,
        timeout: float = 0.2,
        sleeper=None,
        maximum_response_bytes: int = 1_048_576,
        image_encoder=None,
        model_identifier: str = "local-mimo",
        allowed_host: str | None = None,
    ) -> MegatronDynamicServerChatClient:
        return MegatronDynamicServerChatClient(
            endpoint=endpoint,
            model_identifier=model_identifier,
            generation=self.generation if generation is None else generation,
            request_timeout_seconds=timeout,
            retry_backoff_seconds=0.001,
            maximum_response_bytes=maximum_response_bytes,
            sleeper=(lambda _delay: None) if sleeper is None else sleeper,
            image_encoder=self.encode_jpeg if image_encoder is None else image_encoder,
            allowed_host=allowed_host,
        )

    @staticmethod
    def text_conversation() -> list[dict[str, object]]:
        return [{"role": "user", "content": [{"type": "text", "value": "question"}]}]

    def image_conversation(self) -> list[dict[str, object]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "value": str(self.red)},
                    {"type": "text", "value": "demo question"},
                    {"type": "image", "value": str(self.blue)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "value": "demo answer"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "value": str(self.blue)},
                    {"type": "text", "value": "target question"},
                ],
            },
        ]

    def test_exact_multiturn_multiimage_request_and_input_isolation(self) -> None:
        conversation = self.image_conversation()
        original = copy.deepcopy(conversation)
        with FakeDynamicServer([QueuedResponse(body=chat_completion_response("APPLE"))]) as server:
            client = self.client(server.endpoint)
            self.assertTrue(client.is_api)
            self.assertEqual(client.generate(conversation, dataset="OCRBench"), "APPLE")
            recorded = server.requests
        self.assertEqual(conversation, original)
        self.assertEqual(len(recorded), 1)
        body, headers = recorded[0].body, recorded[0].headers
        self.assertEqual(headers["content-type"], "application/json")
        self.assertNotIn("authorization", headers)
        self.assertEqual(
            {key: body[key] for key in body if key != "messages"},
            {
                "model": "local-mimo",
                "n": 1,
                "temperature": 0.0,
                "max_tokens": 64,
                "stop": ["\n\n"],
                "enable_thinking": False,
            },
        )
        messages = body["messages"]
        self.assertEqual([turn["role"] for turn in messages], ["user", "assistant", "user"])
        self.assertEqual(
            [item["type"] for item in messages[0]["content"]],
            ["image_url", "text", "image_url"],
        )
        self.assertEqual(
            [call[0] for call in self.encoder_calls],
            [self.red.resolve(), self.blue.resolve(), self.blue.resolve()],
        )
        self.assertTrue(all(call[1] == -1 for call in self.encoder_calls))
        for turn in (messages[0], messages[2]):
            for item in turn["content"]:
                if item["type"] != "image_url":
                    continue
                url = item["image_url"]["url"]
                self.assertEqual(item["image_url"]["detail"], "low")
                self.assertTrue(url.startswith("data:image/jpeg;base64,"))
                encoded = url.removeprefix("data:image/jpeg;base64,")
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
                    self.assertEqual(decoded.format, "JPEG")

    def test_native_image_data_url_is_preserved_without_reencoding(self) -> None:
        encoded = base64.b64encode(b"native-image-bytes").decode("ascii")
        image_url = f"data:image/jpeg;base64,{encoded}"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "value": image_url},
                    {"type": "text", "value": "question"},
                ],
            }
        ]
        responses = [QueuedResponse(body=chat_completion_response("answer"))]
        with FakeDynamicServer(responses) as server:
            result = self.client(server.endpoint).generate(conversation)
            recorded = server.requests

        self.assertEqual(result, "answer")
        sent_url = recorded[0].body["messages"][0]["content"][0]["image_url"]["url"]
        self.assertEqual(sent_url, image_url)
        self.assertEqual(self.encoder_calls, [])

        for invalid in (
            "data:image/jpeg;base64,%%%",
            "data:image/svg+xml;base64,PHN2Zy8+",
        ):
            bad = copy.deepcopy(conversation)
            bad[0]["content"][0]["value"] = invalid
            with FakeDynamicServer([]) as server:
                with self.assertRaises(ChatRequestValidationError):
                    self.client(server.endpoint).generate(bad)
                self.assertEqual(server.requests, ())

    def test_stop_and_length_finish_reasons_are_accepted(self) -> None:
        for reason in ("stop", "length"):
            with self.subTest(reason=reason):
                response = chat_completion_response("answer", finish_reason=reason)
                with FakeDynamicServer([QueuedResponse(body=response)]) as server:
                    result = self.client(server.endpoint).generate(self.text_conversation())
                self.assertEqual(result, "answer")

    def test_terminal_http_status_is_not_retried(self) -> None:
        for status in (400, 401, 418, 501, 201):
            sleeps: list[float] = []
            with self.subTest(status=status):
                with FakeDynamicServer([QueuedResponse(status=status)]) as server:
                    with self.assertRaises(TerminalHTTPError):
                        self.client(server.endpoint, sleeper=sleeps.append).generate(
                            self.text_conversation()
                        )
                    self.assertEqual(len(server.requests), 1)
                self.assertEqual(sleeps, [])

    def test_retryable_status_and_disconnect_recover(self) -> None:
        cases = tuple(QueuedResponse(status=status) for status in (408, 429, 500, 502, 503, 504))
        cases += (QueuedResponse(disconnect=True),)
        for first_response in cases:
            sleeps: list[float] = []
            with self.subTest(first_response=first_response):
                responses = [first_response, QueuedResponse(body=chat_completion_response("ok"))]
                with FakeDynamicServer(responses) as server:
                    result = self.client(server.endpoint, sleeper=sleeps.append).generate(
                        self.text_conversation()
                    )
                    self.assertEqual(len(server.requests), 2)
                self.assertEqual(result, "ok")
                self.assertEqual(sleeps, [0.001])

    def test_empty_assistant_response_retries_and_recovers(self) -> None:
        sleeps: list[float] = []
        responses = [
            QueuedResponse(body=chat_completion_response("")),
            QueuedResponse(body=chat_completion_response("answer")),
        ]
        with FakeDynamicServer(responses) as server:
            result = self.client(server.endpoint, sleeper=sleeps.append).generate(
                self.text_conversation()
            )
            self.assertEqual(len(server.requests), 2)
        self.assertEqual(result, "answer")
        self.assertEqual(sleeps, [0.001])

    def test_empty_assistant_response_becomes_auditable_failure_sentinel(self) -> None:
        sleeps: list[float] = []
        responses = [
            QueuedResponse(body=chat_completion_response("")) for _ in range(3)
        ]
        with FakeDynamicServer(responses) as server:
            result = self.client(server.endpoint, sleeper=sleeps.append).generate(
                self.text_conversation()
            )
            self.assertEqual(len(server.requests), 3)
        self.assertEqual(result, chat_client.VLMEVAL_FAILURE_SENTINEL)
        self.assertEqual(sleeps, [0.001, 0.002])

    def test_timeout_is_retryable_and_can_recover(self) -> None:
        responses = (
            QueuedResponse(delay_seconds=0.05),
            QueuedResponse(body=chat_completion_response("after timeout")),
        )
        with FakeDynamicServer(responses) as server:
            result = self.client(server.endpoint, timeout=0.01).generate(
                self.text_conversation()
            )
            self.assertEqual(len(server.requests), 2)
        self.assertEqual(result, "after timeout")

    def test_transient_exhaustion_uses_total_attempt_count_and_bounded_sleeps(self) -> None:
        sleeps: list[float] = []
        responses = [QueuedResponse(status=429) for _ in range(3)]
        with FakeDynamicServer(responses) as server:
            with self.assertRaisesRegex(
                TransientRequestExhaustedError,
                "after 3 attempts: retryable HTTP status 429",
            ):
                self.client(server.endpoint, sleeper=sleeps.append).generate(
                    self.text_conversation()
                )
            self.assertEqual(len(server.requests), 3)
        self.assertEqual(sleeps, [0.001, 0.002])

    def test_response_contract_failures_are_terminal(self) -> None:
        two_choices = chat_completion_response()
        two_choices["choices"].append(copy.deepcopy(two_choices["choices"][0]))
        bad_message = chat_completion_response()
        bad_message["choices"][0]["message"] = []
        wrong_role = chat_completion_response()
        wrong_role["choices"][0]["message"]["role"] = "user"
        cases = (
            b"{",
            b"\xff",
            b'{"choices":[],"choices":[]}',
            b'{"choices":NaN}',
            [],
            {"choices": []},
            two_choices,
            chat_completion_response(index=1),
            bad_message,
            wrong_role,
            chat_completion_response(text=7),
            chat_completion_response(finish_reason="tool_calls"),
            chat_completion_response(text="Failed to obtain answer via API."),
        )
        for body in cases:
            with self.subTest(body=body):
                with FakeDynamicServer([QueuedResponse(body=body)]) as server:
                    with self.assertRaises(ChatResponseContractError):
                        self.client(server.endpoint).generate(self.text_conversation())
                    self.assertEqual(len(server.requests), 1)

    def test_oversized_response_is_terminal(self) -> None:
        with FakeDynamicServer([QueuedResponse(body=b"x" * 129)]) as server:
            with self.assertRaisesRegex(ChatResponseContractError, "byte limit"):
                self.client(server.endpoint, maximum_response_bytes=128).generate(
                    self.text_conversation()
                )
            self.assertEqual(len(server.requests), 1)

    def test_endpoint_and_generation_settings_are_validated(self) -> None:
        invalid_endpoints = (
            "https://127.0.0.1:8000/v1/chat/completions",
            "http://example.com:8000/v1/chat/completions",
            "http://127.0.0.1/v1/chat/completions",
            "http://127.0.0.1:8000/other",
            "http://user@127.0.0.1:8000/v1/chat/completions",
            "http://127.0.0.1:8000/v1/chat/completions?key=x",
        )
        for endpoint in invalid_endpoints:
            with self.subTest(endpoint=endpoint), self.assertRaises(ChatRequestValidationError):
                self.client(endpoint)
        remote = self.client(
            "http://server-node-0:8000/v1/chat/completions",
            allowed_host="server-node-0",
        )
        self.assertEqual(remote.endpoint, "http://server-node-0:8000/v1/chat/completions")
        with self.assertRaises(ChatRequestValidationError):
            self.client(
                "http://server-node-1:8000/v1/chat/completions",
                allowed_host="server-node-0",
            )
        with FakeDynamicServer([]) as server:
            for model_identifier in ("", " local-mimo", "local-mimo "):
                with self.subTest(model_identifier=model_identifier):
                    with self.assertRaises(ChatRequestValidationError):
                        self.client(server.endpoint, model_identifier=model_identifier)
            invalid_generations = (
                replace(self.generation, retry=0),
                replace(self.generation, temperature=float("nan")),
                replace(self.generation, image_size=0),
                replace(self.generation, image_detail="auto"),
                replace(self.generation, image_detail=[]),
                replace(self.generation, max_new_tokens=0),
                replace(self.generation, stop=()),
            )
            for generation in invalid_generations:
                with self.subTest(generation=generation):
                    with self.assertRaises(ChatRequestValidationError):
                        self.client(server.endpoint, generation=generation)

    def test_invalid_conversations_and_images_fail_before_network(self) -> None:
        invalid = (
            [],
            [{"role": "assistant", "content": [{"type": "text", "value": "x"}]}],
            [{"role": "user", "content": []}],
            [{"role": "user", "content": [{"type": "text", "value": ""}]}],
            [{"role": "user", "content": [{"type": "audio", "value": "x"}]}],
            [{"role": "user", "content": [{"type": "image", "value": "https://x"}]}],
            [{"role": "user", "content": [{"type": "image", "value": str(self.root)}]}],
            [
                {"role": "user", "content": [{"type": "text", "value": "x"}]},
                {"role": "assistant", "content": [{"type": "image", "value": str(self.red)}]},
            ],
        )
        with FakeDynamicServer([]) as server:
            client = self.client(server.endpoint)
            for message in invalid:
                with self.subTest(message=message), self.assertRaises(ChatRequestValidationError):
                    client.generate(message)
            self.assertEqual(server.requests, ())

    def test_image_encoder_output_must_be_nonempty_canonical_base64(self) -> None:
        conversation = self.image_conversation()
        with FakeDynamicServer([]) as server:
            for encoded in ("", "%%%", "Zh=="):
                with self.subTest(encoded=encoded):
                    encoder = lambda _path, _size, value=encoded: value
                    client = self.client(server.endpoint, image_encoder=encoder)
                    with self.assertRaises(ChatRequestValidationError):
                        client.generate(conversation)
            self.assertEqual(server.requests, ())

    def test_no_generation_override_kwargs_and_default_encoder_contract(self) -> None:
        with FakeDynamicServer([]) as server:
            client = self.client(server.endpoint)
            with self.assertRaises(TypeError):
                client.generate(self.text_conversation(), "OCRBench", temperature=1.0)
        source = inspect.getsource(
            __import__(
                "orchestration.dynamic_server_http_client."
                "megatron_dynamic_server_chat_client",
                fromlist=["_default_image_encoder"],
            )._default_image_encoder
        )
        self.assertIn("from vlmeval.smp import encode_image_to_base64", source)
        self.assertIn("encode_image_to_base64(image, target_size=target_size)", source)

    def test_fake_server_context_closes_thread_and_socket(self) -> None:
        server = FakeDynamicServer([QueuedResponse()])
        with server:
            endpoint = server.endpoint
            self.assertTrue(server.thread_alive)
            self.client(endpoint).generate(self.text_conversation())
        self.assertFalse(server.thread_alive)
        parsed = urlsplit(endpoint)
        with self.assertRaises(OSError):
            socket.create_connection((parsed.hostname, parsed.port), timeout=0.02)


if __name__ == "__main__":
    unittest.main()
