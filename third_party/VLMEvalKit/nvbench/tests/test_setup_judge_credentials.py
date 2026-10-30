"""Installer test against a local OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from nvbench import setup_judge_credentials


class _CompatibleJudgeHandler(BaseHTTPRequestHandler):
    observed_authorization = ""
    observed_payload: dict[str, object] = {}

    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(size))
        type(self).observed_authorization = self.headers["Authorization"]
        type(self).observed_payload = payload
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class _DeniedJudgeHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = json.dumps(
            {"error": {"message": "key unit-secret is not allowed for this model"}}
        ).encode("utf-8")
        self.send_response(403)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class SetupJudgeCredentialsTest(unittest.TestCase):
    def test_interactive_setup_requires_an_explicit_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / ".config" / "nvbench" / "judge.json"
            with (
                mock.patch.dict(os.environ, {"HOME": temporary}),
                mock.patch("builtins.input", side_effect=["", ""]),
                mock.patch("getpass.getpass") as getpass,
                self.assertRaisesRegex(
                    setup_judge_credentials.JudgeCredentialSetupError,
                    "single-line value",
                ),
            ):
                setup_judge_credentials.main(["--output", str(destination)])
            getpass.assert_not_called()
            self.assertFalse(destination.exists())

    def test_interactive_setup_probes_then_installs_user_only_file(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CompatibleJudgeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / ".config" / "nvbench" / "judge.json"
                base_url = f"http://127.0.0.1:{server.server_address[1]}/v1/"
                endpoint = f"{base_url}chat/completions"
                with (
                    mock.patch.dict(os.environ, {"HOME": temporary}),
                    mock.patch("builtins.input", side_effect=[base_url, "unit-judge"]),
                    mock.patch("getpass.getpass", return_value="unit-secret"),
                ):
                    self.assertEqual(
                        setup_judge_credentials.main(
                            ["--output", str(destination), "--timeout", "2"]
                        ),
                        0,
                    )
                self.assertEqual(destination.stat().st_mode & 0o777, 0o600)
                self.assertEqual(destination.parent.stat().st_mode & 0o777, 0o700)
                payload = json.loads(destination.read_text(encoding="utf-8"))
                self.assertEqual(payload["OPENAI_API_BASE"], endpoint)
                self.assertEqual(payload["OPENAI_API_KEY"], "unit-secret")
                self.assertEqual(
                    _CompatibleJudgeHandler.observed_authorization,
                    "Bearer unit-secret",
                )
                self.assertEqual(
                    _CompatibleJudgeHandler.observed_payload,
                    {
                        "model": "unit-judge",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant.",
                            },
                            {"role": "user", "content": "Hello! Can you help me?"},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1024,
                        "stream": False,
                    },
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_provider_error_is_actionable_and_redacts_the_key(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _DeniedJudgeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = (
                f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
            )
            with self.assertRaises(
                setup_judge_credentials.JudgeCredentialSetupError
            ) as caught:
                setup_judge_credentials._probe(endpoint, "unit-secret", "unit-judge", 2)
            message = str(caught.exception)
            self.assertIn("HTTP 403", message)
            self.assertIn("<redacted>", message)
            self.assertNotIn("unit-secret", message)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
