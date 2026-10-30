import os
import re
from pathlib import Path

import datasets

from lm_eval.api.task import ConfigurableTask


SUPPORTED_RAW_SUFFIXES = (".jsonl", ".json", ".parquet")


def _configured_token_window(metadata):
    if not metadata:
        return None

    token_window = metadata.get("token_window")
    if token_window in (None, ""):
        return None

    try:
        token_window = int(token_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata.token_window must be an integer") from exc

    if token_window < 4:
        raise ValueError("metadata.token_window must be at least 4")

    return token_window


class LocalRollingTask(ConfigurableTask):
    def construct_requests(self, doc, ctx, **kwargs):
        instance = super().construct_requests(doc, ctx, **kwargs)
        token_window = _configured_token_window(self.config.metadata)
        if token_window is None:
            return instance

        instances = instance if isinstance(instance, list) else [instance]
        for request in instances:
            request.token_window = token_window
        return instance


def _env_key(value):
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _as_test_dataset(dataset):
    if isinstance(dataset, datasets.Dataset):
        return datasets.DatasetDict({"test": dataset})
    if "test" in dataset:
        return dataset
    if "validation" in dataset:
        return datasets.DatasetDict({"test": dataset["validation"]})
    first_split = next(iter(dataset))
    return datasets.DatasetDict({"test": dataset[first_split]})


def _load_path(path):
    if path.is_dir() and (path / "dataset_dict.json").is_file():
        return _as_test_dataset(datasets.load_from_disk(str(path)))
    if path.is_dir() and (path / "state.json").is_file():
        return _as_test_dataset(datasets.load_from_disk(str(path)))

    files = []
    if path.is_file() and path.suffix in SUPPORTED_RAW_SUFFIXES:
        files = [path]
    elif path.is_dir():
        files = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix in SUPPORTED_RAW_SUFFIXES
        )

    if not files:
        indexed_files = (
            list(path.rglob("*.bin")) + list(path.rglob("*.idx")) if path.exists() else []
        )
        if indexed_files:
            raise ValueError(
                f"{path} contains Megatron indexed data, not raw text. "
                "Export JSONL or Parquet before running lm-eval."
            )
        raise FileNotFoundError(f"No JSONL, JSON, Parquet, or saved Dataset found at {path}")

    formats = {".jsonl": "json", ".json": "json", ".parquet": "parquet"}
    file_formats = {formats[file.suffix] for file in files}
    if len(file_formats) != 1:
        raise ValueError(f"Mixed local dataset formats are not supported at {path}")
    return datasets.load_dataset(
        file_formats.pop(), data_files={"test": [str(file) for file in files]}
    )


def load_local_dataset(*, env_prefix, subset, data_dir=None, aliases=()):
    subset_key = _env_key(subset)
    explicit_path = os.environ.get(f"{env_prefix}_{subset_key}_PATH")
    if explicit_path:
        return _load_path(Path(explicit_path))

    root_value = data_dir or os.environ.get(f"{env_prefix}_DATA_DIR")
    if not root_value:
        raise FileNotFoundError(
            f"Set {env_prefix}_{subset_key}_PATH or {env_prefix}_DATA_DIR "
            f"to load the {subset} evaluation data without Hugging Face Hub access."
        )

    root = Path(root_value)
    names = (subset, *aliases)
    for name in names:
        candidates = [root / name]
        candidates.extend(root / f"{name}{suffix}" for suffix in SUPPORTED_RAW_SUFFIXES)
        for candidate in candidates:
            if candidate.exists():
                return _load_path(candidate)

    matches = sorted(
        {
            candidate
            for name in names
            for candidate in root.glob(f"{name}*")
            if candidate.is_file() and candidate.suffix in SUPPORTED_RAW_SUFFIXES
        }
    )
    if matches:
        formats = {".jsonl": "json", ".json": "json", ".parquet": "parquet"}
        file_formats = {formats[file.suffix] for file in matches}
        if len(file_formats) != 1:
            raise ValueError(f"Mixed local dataset formats are not supported under {root}")
        return datasets.load_dataset(
            file_formats.pop(), data_files={"test": [str(file) for file in matches]}
        )

    raise FileNotFoundError(
        f"Could not find {', '.join(names)} under {root}. "
        f"Set {env_prefix}_{subset_key}_PATH to the exact dataset path."
    )
