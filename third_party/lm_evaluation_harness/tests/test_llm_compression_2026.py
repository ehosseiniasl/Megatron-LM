from datasets import Dataset, DatasetDict
import pytest

from lm_eval.api.instance import Instance
from lm_eval.models.text_completions import TokenizerFreeLocalCompletionsLM
from lm_eval.tasks.llm_compression_2026.utils import (
    LLMCompression2026Task,
    doc_to_text_only_target,
)


def test_doc_to_text_only_target_strips_hf_image_refs():
    doc = {
        "content": "Alpha ![diagram](image://doc-1/0000) beta ![photo](image://doc-1/0001 \"title\") gamma"
    }

    assert doc_to_text_only_target(doc) == "Alpha  beta  gamma"


def test_doc_to_text_only_target_preserves_regular_markdown_images():
    doc = {"content": "Keep ![badge](https://example.com/badge.svg) and ![local](assets/pic.png)."}

    assert doc_to_text_only_target(doc) == doc["content"]


def _tiny_dataset():
    return DatasetDict({"test": Dataset.from_list([{"content": "alpha beta gamma"}])})


def test_task_attaches_configured_token_window_without_changing_rolling_args():
    task = LLMCompression2026Task(
        config={
            "task": "tiny_llm_compression_2026",
            "custom_dataset": _tiny_dataset,
            "dataset_path": "tiny_llm_compression_2026",
            "test_split": "test",
            "doc_to_text": "",
            "doc_to_target": doc_to_text_only_target,
            "output_type": "loglikelihood_rolling",
            "metadata": {"version": 1.0, "token_window": 65536},
        }
    )

    instance = task.construct_requests(
        {"content": "alpha beta gamma"},
        "",
        metadata=("tiny_llm_compression_2026", 0, 1),
    )

    assert instance.args == ("alpha beta gamma",)
    assert instance.token_window == 65536


def test_notok_completions_uses_request_token_window_when_present():
    model = object.__new__(TokenizerFreeLocalCompletionsLM)
    model._max_length = 4096
    request = Instance(
        request_type="loglikelihood_rolling",
        doc={"content": "alpha"},
        arguments=("alpha",),
        idx=0,
    )
    request.token_window = 8192

    assert model._rolling_token_window(request) == 8192


def test_notok_completions_rolling_requires_tokenizer():
    model = object.__new__(TokenizerFreeLocalCompletionsLM)
    model._hf_tokenizer = None
    request = Instance(
        request_type="loglikelihood_rolling",
        doc={"content": "alpha"},
        arguments=("alpha",),
        idx=0,
    )

    with pytest.raises(ValueError, match="requires tokenizer"):
        model.loglikelihood_rolling([request], disable_tqdm=True)


class _WhitespaceTokenizer:
    def encode(self, string, add_special_tokens=False):
        return string.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def test_notok_completions_uses_tokenizer_windows_for_rolling():
    model = object.__new__(TokenizerFreeLocalCompletionsLM)
    model._hf_tokenizer = _WhitespaceTokenizer()
    model._max_length = 6
    captured = {}

    def capture_submit(all_windows, all_context_chars, window_counts, disable_tqdm):
        captured["windows"] = all_windows
        captured["context_chars"] = all_context_chars
        captured["window_counts"] = window_counts
        return [123.0]

    model._submit_and_score = capture_submit
    request = Instance(
        request_type="loglikelihood_rolling",
        doc={"content": "t0 t1 t2 t3 t4"},
        arguments=("t0 t1 t2 t3 t4",),
        idx=0,
    )

    assert model.loglikelihood_rolling([request], disable_tqdm=True) == [123.0]
    assert captured["windows"] == ["t0 t1", "t0 t1 t2 t3", "t2 t3 t4"]
    assert captured["context_chars"] == [0, len("t0 t1"), len("t2 t3")]
    assert captured["window_counts"] == [3]
