from pathlib import Path

import pytest
import yaml
from datasets import Dataset

from lm_eval.tasks import TaskManager
from lm_eval.tasks.llm_compression_2026 import utils as llmc_utils
from lm_eval.tasks.mistral_ppl.utils import load_mistral_ppl_dataset


def _llm_compression_task_config(dataset_name):
    return {
        "task": f"test_{dataset_name}",
        "dataset_path": "local_llm_compression_2026",
        "dataset_name": dataset_name,
        "dataset_kwargs": {},
        "test_split": "test",
        "doc_to_text": "",
        "doc_to_target": llmc_utils.doc_to_text_only_target,
        "output_type": "loglikelihood_rolling",
        "metadata": {"version": 1.0},
    }


def test_llm_compression_loads_local_jsonl_without_hub_access(tmp_path, monkeypatch):
    (tmp_path / "cc_interleaved.jsonl").write_text('{"content": "local row"}\n')
    monkeypatch.setenv("LM_EVAL_LLM_COMPRESSION_2026_DATA_DIR", str(tmp_path))

    task = llmc_utils.LLMCompression2026Task(config=_llm_compression_task_config("cc_interleaved"))

    assert task.test_docs()[0]["content"] == "local row"


@pytest.mark.parametrize(
    ("subset", "expected_rows"),
    [
        ("cc_interleaved", 9000),
        ("arxiv_math_interleaved", 5000),
        ("arxiv_physics_interleaved", 5000),
        ("code_repo_packed", 74),
    ],
)
def test_llm_compression_packaged_data_is_available_without_overrides(
    subset, expected_rows, monkeypatch
):
    monkeypatch.delenv("LM_EVAL_LLM_COMPRESSION_2026_PARQUET_PATH", raising=False)
    monkeypatch.delenv("LM_EVAL_LLM_COMPRESSION_2026_DATA_DIR", raising=False)
    monkeypatch.delenv(f"LM_EVAL_LLM_COMPRESSION_2026_{subset.upper()}_PATH", raising=False)

    task = llmc_utils.LLMCompression2026Task(config=_llm_compression_task_config(subset))

    assert len(task.test_docs()) == expected_rows
    assert set(task.test_docs()["dataset"]) == {subset}


def test_mistral_arxiv_alias_loads_raw_jsonl(tmp_path, monkeypatch):
    (tmp_path / "arxiv_val_ppl.jsonl").write_text('{"text": "math row"}\n')
    monkeypatch.setenv("LM_EVAL_MISTRAL_PPL_DATA_DIR", str(tmp_path))

    dataset = load_mistral_ppl_dataset("arxiv_val_ppl")

    assert dataset["test"][0]["text"] == "math row"


def test_mistral_packaged_parquet_filters_subsets(tmp_path, monkeypatch):
    parquet_path = tmp_path / "mistral_ppl_eval.parquet"
    Dataset.from_list(
        [
            {"dataset": "arxiv_val_ppl", "text": "math row"},
            {"dataset": "catlm_v2", "text": "cat row"},
        ]
    ).to_parquet(parquet_path)
    monkeypatch.setenv("LM_EVAL_MISTRAL_PPL_PARQUET_PATH", str(parquet_path))

    arxiv = load_mistral_ppl_dataset("arxiv_val_ppl")
    catlm = load_mistral_ppl_dataset("catlm_v2")

    assert arxiv["test"][0]["text"] == "math row"
    assert catlm["test"][0]["text"] == "cat row"


def test_mistral_packaged_parquet_is_available_without_overrides(monkeypatch):
    for name in (
        "LM_EVAL_MISTRAL_PPL_PARQUET_PATH",
        "LM_EVAL_MISTRAL_PPL_DATA_DIR",
        "LM_EVAL_MISTRAL_PPL_ARXIV_VAL_PPL_PATH",
        "LM_EVAL_MISTRAL_PPL_CATLM_V2_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    arxiv = load_mistral_ppl_dataset("arxiv_val_ppl")["test"]
    catlm = load_mistral_ppl_dataset("catlm_v2")["test"]

    assert len(arxiv) == 1095
    assert len(catlm) == 1890
    assert set(arxiv["dataset"]) == {"arxiv_val_ppl"}
    assert set(catlm["dataset"]) == {"catlm_v2"}
    assert all(not text.startswith("<s>") for text in catlm["text"])


def test_mistral_ppl_rejects_indexed_data(tmp_path, monkeypatch):
    catlm_dir = tmp_path / "catlm_v2"
    catlm_dir.mkdir()
    (catlm_dir / "chunk.bin").write_bytes(b"")
    (catlm_dir / "chunk.idx").write_bytes(b"")
    monkeypatch.setenv("LM_EVAL_MISTRAL_PPL_DATA_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="indexed data, not raw text"):
        load_mistral_ppl_dataset("catlm_v2")


def test_nemo4_fast_does_not_resolve_private_nvidia_datasets():
    harness_root = Path(__file__).parents[1]
    with (harness_root / "nvbench" / "nemo4-fast.yml").open() as config_file:
        suite = yaml.safe_load(config_file)

    task_manager = TaskManager()
    private_paths = []
    for task in suite["tasks"]:
        for resolved in task_manager._resolve_spec(task["task_name"]):
            dataset_path = resolved.get("dataset_path")
            if isinstance(dataset_path, str) and dataset_path.lower().startswith("nvidia/"):
                private_paths.append((task["task_name"], dataset_path))

    assert private_paths == []
