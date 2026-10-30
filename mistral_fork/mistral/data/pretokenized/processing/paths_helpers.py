import dataclasses
from pathlib import Path

from mistral_fork.mistral.data.pretokenized.cache_key import (
    TokenizerName,
    compute_cache_key,
)
from mistral_fork.mistral.data.pretokenized.document_filters import (
    BadDocumentFilter,
    PdsFilter,
    RareTokenFilter,
)
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
    MaskingStrategy,
    PretokenizedDatasourceMetadataV2,
)
from mistral_fork.mistral.data.source import SampleType


@dataclasses.dataclass(frozen=True)
class PrefixPaths:
    prefix: Path

    def _with(self, suffix: str) -> Path:
        return self.prefix.with_name(self.prefix.name + suffix)

    @property
    def bin(self) -> Path:
        return self._with(".bin")

    @property
    def index(self) -> Path:
        return self._with(".idx")

    @property
    def mask(self) -> Path:
        return self._with(".mask")

    @property
    def stats(self) -> Path:
        return self._with(".stats.json")

    @property
    def image_metadata(self) -> Path:
        return self._with(".metadata")

    @property
    def validation(self) -> Path:
        return self._with(".validation.jsonl")

    @classmethod
    def from_prefix(cls, prefix: str | Path) -> "PrefixPaths":
        return cls(Path(prefix).expanduser())


def check_chunk_is_complete(
    prefix: str | Path | PrefixPaths,
    source_metadata: PretokenizedDatasourceMetadataV2,
    is_merged: bool,
) -> None:
    if isinstance(prefix, PrefixPaths):
        paths = prefix
    else:
        paths = PrefixPaths.from_prefix(prefix)

    assert paths.prefix.parent.exists(), (
        f"Pretokenized data directory not found: {paths.prefix.parent}. "
        f"The data has likely not been pretokenized yet, "
        f"or the cache key / source_set configuration points to the wrong path."
    )

    expected = {paths.bin, paths.index}
    if source_metadata.has_image_metadata:
        expected.add(paths.image_metadata)
    if not is_merged:
        expected.add(paths.stats)
        if source_metadata.validation_fraction > 0.0:
            expected.add(paths.validation)

    if source_metadata.masking_strategy is not None:
        expected.add(paths.mask)

    missing = {p for p in expected if not p.exists()}
    assert not missing, missing


def get_datasource_metadata_path(path: Path) -> Path:
    return path / "pretokenization_metadata.json"


def get_datasource_metadata_v2_path(path: Path) -> Path:
    return path / "pretokenization_metadata_v2.json"


def get_chunks_dir(source_dir: Path) -> Path:
    return source_dir / "chunks"


def get_single_chunk_dir(chunks_dir: Path, original_jsonl_file: Path) -> Path:
    return chunks_dir / original_jsonl_file.stem


def get_single_chunk_prefix(single_chunk_dir: Path) -> str:
    return str(single_chunk_dir / "chunk")


def get_merged_folder(meg_folder: Path, max_group_size_in_byte: int) -> Path:
    return meg_folder / f"merged_{max_group_size_in_byte}b"


def get_pretokenized_path(
    source_path: str,
    tokenizer_name: TokenizerName,
    profile: FilterArgsName,
    masking_strategy: MaskingStrategy | None,
    sample_type: SampleType,
    rare_token_filter: RareTokenFilter | None,
    pds_filter: PdsFilter | None,
    pretokenization_root: str,
    bad_document_filter: BadDocumentFilter | None = None,
) -> Path:
    cache_key = compute_cache_key(
        profile=profile,
        tokenizer_name=tokenizer_name,
        sample_type=sample_type,
        masking_strategy=masking_strategy,
        rare_token_filter=rare_token_filter,
        pds_filter=pds_filter,
        bad_document_filter=bad_document_filter,
    )
    return Path(pretokenization_root) / cache_key / source_path


def get_present_merged_folders(source_dir: Path) -> set[str]:
    if not source_dir.exists():
        return set()
    return {
        child.name
        for child in source_dir.iterdir()
        if child.is_dir()
        and child.name.startswith("merged_")
        and child.name.endswith("b")
    }


def check_no_conflicting_merged_folder_for_source(
    source_dir: Path, merged_size: int
) -> None:
    present = get_present_merged_folders(source_dir)
    if len(present) > 1:
        raise ValueError(
            f"{source_dir} has multiple merged folders: {sorted(present)}. "
            "Delete all but one before merging."
        )

    expected = get_merged_folder(source_dir, merged_size).name
    if present and present != {expected}:
        raise ValueError(
            f"{source_dir} was already merged as {next(iter(present))}, "
            f"cannot re-process with {expected}."
        )


def get_prefix_from_bin(bin_file: Path) -> PrefixPaths:
    assert bin_file.name.endswith(".bin"), bin_file
    return PrefixPaths(prefix=bin_file.with_name(bin_file.name[:-4]))
