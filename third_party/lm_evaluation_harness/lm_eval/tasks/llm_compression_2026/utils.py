import os
import re
from pathlib import Path

import datasets

from lm_eval.tasks.local_dataset import LocalRollingTask, load_local_dataset


IMAGE_URI_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\(image://[^)\s]+(?:\s+\"[^\"]*\")?\)")
PACKAGED_DATASET = Path(__file__).with_name("llm_compression_2026_eval.parquet")


def _is_lfs_pointer(path):
    with path.open("rb") as stream:
        return stream.read(42).startswith(b"version https://git-lfs.github.com/spec/")


def load_packaged_llm_compression_dataset(subset):
    path = Path(os.environ.get("LM_EVAL_LLM_COMPRESSION_2026_PARQUET_PATH", PACKAGED_DATASET))
    if not path.is_file():
        raise FileNotFoundError(f"Packaged LLM compression data is missing: {path}")
    if _is_lfs_pointer(path):
        raise FileNotFoundError(
            f"{path} is a Git LFS pointer, not the Parquet payload. Run `git lfs pull`."
        )

    dataset = datasets.load_dataset("parquet", data_files={"test": str(path)})["test"]
    dataset = dataset.filter(lambda row: row["dataset"] == subset)
    if not dataset:
        raise ValueError(f"Packaged LLM compression data has no {subset} rows: {path}")
    return datasets.DatasetDict({"test": dataset})


def doc_to_text_only_target(doc):
    content = doc.get("content") or ""
    if not isinstance(content, str):
        return ""
    return IMAGE_URI_MARKDOWN_RE.sub("", content)


class LLMCompression2026Task(LocalRollingTask):
    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config=None):
        if config is not None:
            config = dict(config)
            config.pop("class", None)
        super().__init__(
            data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode, config=config
        )

    def download(self, dataset_kwargs=None):
        if self.config.custom_dataset is not None:
            return super().download(dataset_kwargs)

        dataset_kwargs = dataset_kwargs or {}
        subset_key = re.sub(r"[^A-Z0-9]+", "_", self.DATASET_NAME.upper()).strip("_")
        has_local_override = any(
            (
                dataset_kwargs.get("data_dir"),
                os.environ.get("LM_EVAL_LLM_COMPRESSION_2026_DATA_DIR"),
                os.environ.get(f"LM_EVAL_LLM_COMPRESSION_2026_{subset_key}_PATH"),
            )
        )
        if has_local_override:
            self.dataset = load_local_dataset(
                env_prefix="LM_EVAL_LLM_COMPRESSION_2026",
                subset=self.DATASET_NAME,
                data_dir=dataset_kwargs.get("data_dir"),
            )
        else:
            self.dataset = load_packaged_llm_compression_dataset(self.DATASET_NAME)
