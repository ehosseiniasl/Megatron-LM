import os
from pathlib import Path

import datasets

from lm_eval.tasks.local_dataset import LocalRollingTask, load_local_dataset


PACKAGED_DATASET = Path(__file__).with_name("mistral_ppl_eval.parquet")
ALIASES = {
    "arxiv_val_ppl": ("arxiv_math", "arxiv"),
    "catlm_v2": ("catlm", "cat_lm_v2", "cat-lm-v2"),
}


class MistralPPLTask(LocalRollingTask):
    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config=None):
        if config is not None:
            config = dict(config)
            config.pop("class", None)
        super().__init__(
            data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode, config=config
        )


def load_mistral_ppl_dataset(subset, data_dir=None):
    if subset not in ALIASES:
        raise ValueError(f"Unsupported Mistral PPL subset: {subset}")

    packaged_path = os.environ.get("LM_EVAL_MISTRAL_PPL_PARQUET_PATH")
    has_local_override = data_dir or os.environ.get("LM_EVAL_MISTRAL_PPL_DATA_DIR")
    has_local_override = has_local_override or os.environ.get(
        f"LM_EVAL_MISTRAL_PPL_{subset.upper()}_PATH"
    )
    if packaged_path or not has_local_override:
        path = Path(packaged_path) if packaged_path else PACKAGED_DATASET
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing packaged Mistral PPL data: {path}. Run `git lfs pull` or set "
                "LM_EVAL_MISTRAL_PPL_PARQUET_PATH."
            )
        with path.open("rb") as stream:
            header = stream.read(42)
        if header.startswith(b"version https://git-lfs.github.com/spec"):
            raise RuntimeError(f"{path} is a Git LFS pointer. Run `git lfs pull`.")
        dataset = datasets.load_dataset("parquet", data_files={"test": str(path)})
        filtered = dataset["test"].filter(lambda row: row["dataset"] == subset)
        if len(filtered) == 0:
            raise ValueError(f"Packaged Mistral PPL data contains no {subset} rows")
        return datasets.DatasetDict({"test": filtered})

    return load_local_dataset(
        env_prefix="LM_EVAL_MISTRAL_PPL", subset=subset, data_dir=data_dir, aliases=ALIASES[subset]
    )


def doc_to_target(doc):
    for field in ("content", "text"):
        value = doc.get(field)
        if isinstance(value, str):
            return value
    raise ValueError("Mistral PPL rows must contain a string 'content' or 'text' field")
