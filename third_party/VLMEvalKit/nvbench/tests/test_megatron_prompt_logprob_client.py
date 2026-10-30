"""Tests for the Megatron multimodal prompt-logprob protocol."""

from __future__ import annotations
import base64
import unittest

from fake_dynamic_server import FakeDynamicServer, QueuedResponse
from orchestration.dynamic_server_http_client.megatron_prompt_logprob_client import (
    MegatronPromptLogprobClient,
    PromptLogprobClientError,
    build_scored_prompt_token_ids,
)


class CharacterTokenizer:
    unk_token_id = 0
    image_token_id = 10_000

    def convert_tokens_to_ids(self, token):
        return self.image_token_id if token == "<img>" else 0

    def encode(self, text, **_kwargs):
        values = []
        while text:
            if text.startswith("<img>"):
                values.append(self.image_token_id)
                text = text[5:]
            else:
                values.append(ord(text[0]))
                text = text[1:]
        return values

    def decode(self, token_ids, **_kwargs):
        return "".join(
            "<img>" if value == self.image_token_id else chr(value) for value in token_ids
        )


def completion_response(token_ids, logprobs):
    return {
        "choices": [
            {
                "prompt_token_ids": token_ids,
                "logprobs": {
                    "tokens": [str(token) for token in token_ids],
                    "token_logprobs": logprobs,
                },
            }
        ]
    }


class MegatronPromptLogprobClientTest(unittest.TestCase):
    def test_exact_native_request_and_response(self) -> None:
        tokenizer = CharacterTokenizer()
        tokens = build_scored_prompt_token_ids(tokenizer, "instruction\n", "answer")
        response = completion_response(tokens, [None] + [-0.1] * (len(tokens) - 1))
        with FakeDynamicServer([QueuedResponse(body=response)]) as server:
            endpoint = server.endpoint.replace("/v1/chat/completions", "/v1/completions")
            client = MegatronPromptLogprobClient(endpoint, "model", 1.0, 2)
            result = client.score_prompt(tokens, b"jpeg")
            request = server.requests[0].body
        self.assertEqual(result.token_ids, tokens)
        self.assertEqual(request["prompt"], tokens)
        self.assertEqual(request["max_tokens"], 0)
        self.assertEqual(request["echo"], True)
        self.assertEqual(request["logprobs"], 0)
        self.assertEqual(base64.b64decode(request["multi_modal_data"]["image"][0]), b"jpeg")

    def test_retries_transient_status_and_rejects_misalignment(self) -> None:
        tokens = [1, 2]
        with FakeDynamicServer(
            [
                QueuedResponse(status=503, body={"error": "busy"}),
                QueuedResponse(body=completion_response(tokens, [None, -1.0])),
            ]
        ) as server:
            endpoint = server.endpoint.replace("/v1/chat/completions", "/v1/completions")
            result = MegatronPromptLogprobClient(
                endpoint, "model", 1.0, 2, sleeper=lambda _delay: None
            ).score_prompt(tokens, b"jpeg")
            self.assertEqual(result.logprobs, [None, -1.0])
            self.assertEqual(len(server.requests), 2)
        with FakeDynamicServer(
            [QueuedResponse(body=completion_response(tokens, [None]))]
        ) as server:
            endpoint = server.endpoint.replace("/v1/chat/completions", "/v1/completions")
            with self.assertRaisesRegex(PromptLogprobClientError, "misaligned"):
                MegatronPromptLogprobClient(endpoint, "model", 1.0, 1).score_prompt(tokens, b"jpeg")


if __name__ == "__main__":
    unittest.main()
