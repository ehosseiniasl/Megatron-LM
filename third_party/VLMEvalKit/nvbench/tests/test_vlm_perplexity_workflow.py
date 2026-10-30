"""Offline end-to-end tests for the conditional-perplexity workflow."""

from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from orchestration.configured_eval_run import (
    ConditionalPerplexityBenchmarkConfig,
    ConditionalPerplexityRequestConfig,
)
from orchestration.dynamic_server_http_client.megatron_prompt_logprob_client import PromptLogprobs
from orchestration.vlm_perplexity_workflow import (
    ConditionalPerplexityWorkflowError,
    run_conditional_perplexity,
)
from test_surge_mm_dataset import record, write_packed_fixture


class CharacterTokenizer:
    unk_token_id = 0
    image_token_id = 10_000

    def convert_tokens_to_ids(self, token):
        return self.image_token_id if token == "<img>" else self.unk_token_id

    def encode(self, text, **_kwargs):
        result = []
        while text:
            if text.startswith("<img>"):
                result.append(self.image_token_id)
                text = text[5:]
            else:
                result.append(ord(text[0]))
                text = text[1:]
        return result

    def decode(self, token_ids, **_kwargs):
        return "".join(
            "<img>" if value == self.image_token_id else chr(value) for value in token_ids
        )


class RecordingClient:
    def __init__(self, failing_token: int | None = None, returned_prefix: list[int] | None = None):
        self.calls = []
        self.failing_token = failing_token
        self.returned_prefix = returned_prefix or []

    def score_prompt(self, prompt_token_ids, image_bytes):
        self.calls.append((list(prompt_token_ids), image_bytes))
        if self.failing_token in prompt_token_ids:
            raise RuntimeError("injected request failure")
        returned = self.returned_prefix + list(prompt_token_ids)
        pieces = ["<img>" if value == 10_000 else chr(value) for value in returned]
        return PromptLogprobs(returned, pieces, [None] + [-0.1] * (len(returned) - 1))


def resolved_run(data: Path, output: Path, *, resume: bool = False):
    benchmark = ConditionalPerplexityBenchmarkConfig(
        name="surge_mm_ppl",
        dataset="surge_mm_ppl",
        data_format="packed_jsonl_tar",
        configured_data_path=str(data),
        category_field="image_type",
        resource="n1",
        scored_limit=None,
        request=ConditionalPerplexityRequestConfig(
            instruction="Describe the image in detail.\n", temperature=0, top_p=1, retry=2
        ),
    )
    return SimpleNamespace(
        configured=SimpleNamespace(
            benchmark=benchmark,
            run=SimpleNamespace(api_nproc=2),
            output=SimpleNamespace(resume=resume),
        ),
        paths=SimpleNamespace(run_directory=str(output)),
    )


class ConditionalPerplexityWorkflowTest(unittest.TestCase):
    def test_complete_run_and_resume_do_not_duplicate_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = write_packed_fixture(root, [record("a", "alpha"), record("b", "beta")])
            output = root / "run"
            output.mkdir()
            client = RecordingClient()
            result = run_conditional_perplexity(
                resolved_run(data, output), client, CharacterTokenizer()
            )
            self.assertEqual(len(client.calls), 2)
            metrics = json.loads(result.metrics_artifact.read_text())
            self.assertEqual(metrics["overall"]["num_samples"], 2)
            self.assertEqual(set(metrics["by_image_type"]), {"chart"})
            resumed_client = RecordingClient()
            run_conditional_perplexity(
                resolved_run(data, output, resume=True), resumed_client, CharacterTokenizer()
            )
            self.assertEqual(resumed_client.calls, [])
            self.assertEqual(len(result.samples_artifact.read_text().splitlines()), 2)

    def test_scores_returned_suffix_after_server_prompt_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = write_packed_fixture(root, [record("a", "alpha")])
            output = root / "run"
            output.mkdir()
            client = RecordingClient(returned_prefix=[ord("x")])

            result = run_conditional_perplexity(
                resolved_run(data, output), client, CharacterTokenizer()
            )

            metrics = json.loads(result.metrics_artifact.read_text())
            self.assertEqual(metrics["overall"]["num_samples"], 1)
            self.assertEqual(metrics["overall"]["token_count"], len("alpha"))

    def test_incomplete_run_has_no_completed_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = write_packed_fixture(root, [record("a", "alpha"), record("b", "zeta")])
            output = root / "run"
            output.mkdir()
            with self.assertRaisesRegex(ConditionalPerplexityWorkflowError, "incomplete"):
                run_conditional_perplexity(
                    resolved_run(data, output),
                    RecordingClient(failing_token=ord("z")),
                    CharacterTokenizer(),
                )
            status = json.loads((output / "vlmeval_dataset_status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertFalse((output / "surge_mm_ppl.metrics.json").exists())

            resumed_client = RecordingClient()
            result = run_conditional_perplexity(
                resolved_run(data, output, resume=True), resumed_client, CharacterTokenizer()
            )
            self.assertEqual(len(resumed_client.calls), 1)
            self.assertEqual(len(result.samples_artifact.read_text().splitlines()), 2)
            resumed_status = json.loads(result.status_artifact.read_text())
            self.assertEqual(resumed_status["status"], "completed")


if __name__ == "__main__":
    unittest.main()
