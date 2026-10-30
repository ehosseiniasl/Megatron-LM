from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationInfo, field_validator

from mistral_adapter.pretokenized_config import CONFIG_VERSION, PretokenizedConfig
from mistral_adapter.pretokenized_config import SourceConfig as MegatronSourceConfig
from mistral_fork.mistral.data.pretokenized.cache_key import (
    TokenizerName,
    check_pds,
    check_rare_tokens,
    tokenizer_name_from_path,
)
from mistral_fork.mistral.data.pretokenized.document_filters import (
    BadDocumentFilter,
    PdsFilter,
    RareTokenFilter,
)
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
    ImageTransformType,
    MaskingStrategy,
)
from mistral_fork.mistral.data.pretokenized.processing.paths_helpers import (
    get_merged_folder,
    get_pretokenized_path,
)
from mistral_fork.mistral.data.pretokenized.processing.tokenizers import (
    detect_pretokenization_tokenizer_type,
    special_token_ids_from_tokenizer,
)
from mistral_fork.mistral.data.source import SampleType

DATASETS_ROOT = Path("/mnt/vast/datasets")
JSONL_PRETOKENIZATION_CONFIG_VERSION = 1
DEFAULT_JSONL_ROOT_DATA_DIR = DATASETS_ROOT
DEFAULT_EPOCH_INDEX_CACHE_PATH = Path("/mnt/pretokenized/epoch_indexes")
DEFAULT_ORIGINAL_SEQ_LEN = 8192


class JsonlSourceConfig(BaseModel):
    """Raw JSONL source config used by the fork-local pretokenization script."""

    jsonl_dir: str
    jsonl_root_data_dir: str | None = None
    weight: float
    percentage_data: float
    sample_type: SampleType
    image_transform_type: ImageTransformType | None = None
    filter: FilterArgsName
    masking_strategy: MaskingStrategy | None = None
    rare_token_filter: RareTokenFilter | None
    pds_filter: PdsFilter | None
    bad_document_filter: BadDocumentFilter | None = None


class JsonlPretokenizationConfig(BaseModel):
    """Config for turning raw JSONL sources into pretokenized Megatron data."""

    version: int = JSONL_PRETOKENIZATION_CONFIG_VERSION
    tokenizer_path: str
    jsonl_root_data_dir: str = str(DEFAULT_JSONL_ROOT_DATA_DIR)
    sources: list[JsonlSourceConfig]

    @field_validator("sources", mode="after")
    @classmethod
    def reject_duplicate_jsonl_sources(
        cls, sources: list[JsonlSourceConfig]
    ) -> list[JsonlSourceConfig]:
        """Reject duplicate raw paths before processing can be scheduled."""

        seen_paths: dict[Path, int] = {}
        for source_index, source in enumerate(sources):
            jsonl_path = Path(str(source.jsonl_dir))
            if jsonl_path in seen_paths:
                previous_index = seen_paths[jsonl_path]
                raise ValueError(
                    f"Duplicate JSONL source path {jsonl_path} in sources["
                    f"{previous_index}] and sources[{source_index}]"
                )
            seen_paths[jsonl_path] = source_index
        return sources

    @field_validator("sources", mode="after")
    @classmethod
    def check_source_filter_policies(
        cls, sources: list[JsonlSourceConfig], info: ValidationInfo
    ) -> list[JsonlSourceConfig]:
        """Check explicit source filters without changing their configured values."""

        tokenizer_name = tokenizer_name_from_path(info.data["tokenizer_path"])
        for source in sources:
            check_rare_tokens(
                profile=source.filter,
                tokenizer_name=tokenizer_name,
                rare_token_filter=source.rare_token_filter,
            )
            check_pds(
                tokenizer_name=tokenizer_name,
                sample_type=source.sample_type,
                pds_filter=source.pds_filter,
            )
        return sources


@dataclasses.dataclass(frozen=True)
class ConfigProcessingTask:
    source_config: JsonlSourceConfig
    jsonl_file_name: str
    source_n_chunks: int


def load_jsonl_pretokenization_config(path: Path) -> JsonlPretokenizationConfig:
    with path.open() as file:
        raw = yaml.safe_load(file)
    config = JsonlPretokenizationConfig.model_validate(raw)
    assert config.version == JSONL_PRETOKENIZATION_CONFIG_VERSION, (
        f"JSONL pretokenization config version mismatch: file has v{config.version}, "
        f"but this script expects v{JSONL_PRETOKENIZATION_CONFIG_VERSION}."
    )
    return config


def _tokenizer_name_from_config(config: JsonlPretokenizationConfig) -> TokenizerName:
    detect_pretokenization_tokenizer_type(config.tokenizer_path)
    return tokenizer_name_from_path(config.tokenizer_path)


def _relative_source_path(raw_path: Path, jsonl_root: Path) -> Path:
    try:
        return raw_path.relative_to(jsonl_root)
    except ValueError as exc:
        raise ValueError(
            f"Raw source path must be under {jsonl_root}: {raw_path}"
        ) from exc


def _input_jsonl_files(input_path: Path) -> list[Path]:
    if input_path.suffix == ".jsonl":
        assert input_path.is_file(), f"Input JSONL file does not exist: {input_path}"
        return [input_path]
    files = sorted(input_path.glob("*.jsonl"))
    assert files, f"No *.jsonl files found in {input_path}"
    return files


def resolve_config_source(
    config_source: JsonlSourceConfig,
    *,
    jsonl_root: Path,
    tokenizer_name: TokenizerName,
    pretokenized_data_dir: Path,
) -> tuple[Path, Path]:
    raw_path = Path(config_source.jsonl_dir)
    source_jsonl_root = (
        Path(config_source.jsonl_root_data_dir)
        if config_source.jsonl_root_data_dir is not None
        else jsonl_root
    )
    relative_source_path = str(_relative_source_path(raw_path, source_jsonl_root))
    pretok_dir = get_pretokenized_path(
        source_path=relative_source_path,
        tokenizer_name=tokenizer_name,
        profile=config_source.filter,
        masking_strategy=config_source.masking_strategy,
        sample_type=config_source.sample_type,
        pretokenization_root=str(pretokenized_data_dir),
        rare_token_filter=config_source.rare_token_filter,
        pds_filter=config_source.pds_filter,
        bad_document_filter=config_source.bad_document_filter,
    )
    return raw_path, pretok_dir


def config_processing_tasks(
    config: JsonlPretokenizationConfig,
    *,
    debug_doc_limit: int | None,
) -> list[ConfigProcessingTask]:
    tasks: list[ConfigProcessingTask] = []
    for source_config in config.sources:
        jsonl_files = _input_jsonl_files(Path(source_config.jsonl_dir))
        if debug_doc_limit is not None:
            jsonl_files = jsonl_files[:1]
        tasks.extend(
            ConfigProcessingTask(
                source_config=source_config,
                jsonl_file_name=jsonl_path.name,
                source_n_chunks=len(jsonl_files),
            )
            for jsonl_path in jsonl_files
        )
    return sorted(
        tasks, key=lambda task: (task.source_config.jsonl_dir, task.jsonl_file_name)
    )


def shard_config_processing_tasks(
    tasks: list[ConfigProcessingTask], *, worker_id: int, num_workers: int
) -> list[ConfigProcessingTask]:
    assert num_workers >= 1, num_workers
    assert 0 <= worker_id < num_workers, (worker_id, num_workers)
    return tasks[worker_id::num_workers]


def _merged_prefixes_with_fractions(merged_dir: Path) -> list[tuple[Path, float]]:
    bin_files = sorted(merged_dir.glob("chunks_*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No chunks_*.bin files found in {merged_dir}")
    sizes = [path.stat().st_size for path in bin_files]
    total_size = sum(sizes)
    if total_size <= 0:
        raise ValueError(f"Merged bin files are empty in {merged_dir}")
    return [
        (bin_file.with_name(bin_file.name[:-4]), size / total_size)
        for bin_file, size in zip(bin_files, sizes, strict=True)
    ]


def _sorted_source_configs(
    sources: list[JsonlSourceConfig],
) -> list[JsonlSourceConfig]:
    """Match Mistral's deterministic sample-type and path ordering."""

    sample_type_order = {
        sample_type: index for index, sample_type in enumerate(SampleType)
    }
    return sorted(
        sources,
        key=lambda source: (
            sample_type_order[source.sample_type],
            Path(source.jsonl_dir),
        ),
    )


def build_megatron_pretokenized_config_from_jsonl_config(
    config: JsonlPretokenizationConfig,
    *,
    pretokenized_data_dir: Path,
    epoch_index_cache_path: Path,
    original_seq_len: int,
    default_max_group_size: int,
    interdocument_masking: bool = True,
    truncate_trailing_masked: bool = True,
    optimized_datasource_id_sampling: bool = True,
    allow_non_sorted_sources_in_dataloader: bool = False,
    seed: int = 0,
    skip_images: bool = True,
) -> PretokenizedConfig:
    assert pretokenized_data_dir.is_absolute(), pretokenized_data_dir
    assert epoch_index_cache_path.is_absolute(), epoch_index_cache_path
    tokenizer_name = _tokenizer_name_from_config(config)
    jsonl_root = Path(config.jsonl_root_data_dir)
    special_ids = special_token_ids_from_tokenizer(config.tokenizer_path)
    sources: list[MegatronSourceConfig] = []
    for source_config in _sorted_source_configs(config.sources):
        _, pretok_dir = resolve_config_source(
            source_config,
            jsonl_root=jsonl_root,
            tokenizer_name=tokenizer_name,
            pretokenized_data_dir=pretokenized_data_dir,
        )
        merged_dir = get_merged_folder(pretok_dir, default_max_group_size)
        for prefix_path, fraction in _merged_prefixes_with_fractions(merged_dir):
            sources.append(
                MegatronSourceConfig(
                    prefix_path=str(prefix_path),
                    weight=source_config.weight * fraction,
                    percentage_data=source_config.percentage_data,
                    sample_type=source_config.sample_type.value,
                    image_transform_type=source_config.image_transform_type,
                )
            )

    return PretokenizedConfig(
        version=CONFIG_VERSION,
        sources=sources,
        bos_id=special_ids["bos_id"],
        eos_id=special_ids["eos_id"],
        pad_id=special_ids["pad_id"],
        img_id=special_ids["img_id"],
        img_break_id=special_ids["img_break_id"],
        img_end_id=special_ids["img_end_id"],
        seed=seed,
        epoch_index_cache_path=str(epoch_index_cache_path),
        interdocument_masking=interdocument_masking,
        drop_bos=not interdocument_masking,
        original_seq_len=original_seq_len,
        truncate_trailing_masked=truncate_trailing_masked,
        tokenizer_name=tokenizer_name.value,
        optimized_datasource_id_sampling=optimized_datasource_id_sampling,
        allow_non_sorted_sources_in_dataloader=allow_non_sorted_sources_in_dataloader,
        skip_images=skip_images,
    )


def write_megatron_pretokenized_config_from_jsonl_config(
    *,
    config_path: Path,
    output_path: Path,
    pretokenized_data_dir: Path,
    epoch_index_cache_path: Path,
    original_seq_len: int,
    max_group_size: int,
    interdocument_masking: bool = True,
    truncate_trailing_masked: bool = True,
    optimized_datasource_id_sampling: bool = True,
    allow_non_sorted_sources_in_dataloader: bool = False,
    seed: int = 0,
    skip_images: bool = True,
) -> PretokenizedConfig:
    config = load_jsonl_pretokenization_config(config_path)
    megatron_config = build_megatron_pretokenized_config_from_jsonl_config(
        config,
        pretokenized_data_dir=pretokenized_data_dir,
        epoch_index_cache_path=epoch_index_cache_path,
        original_seq_len=original_seq_len,
        default_max_group_size=max_group_size,
        interdocument_masking=interdocument_masking,
        truncate_trailing_masked=truncate_trailing_masked,
        optimized_datasource_id_sampling=optimized_datasource_id_sampling,
        allow_non_sorted_sources_in_dataloader=allow_non_sorted_sources_in_dataloader,
        seed=seed,
        skip_images=skip_images,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            megatron_config.model_dump(mode="json", exclude_none=False),
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return megatron_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Megatron runtime PretokenizedConfig from a raw JSONL config."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretokenized-data-dir", type=Path, required=True)
    parser.add_argument(
        "--epoch-index-cache-path",
        type=Path,
        default=DEFAULT_EPOCH_INDEX_CACHE_PATH,
    )
    parser.add_argument(
        "--original-seq-len", type=int, default=DEFAULT_ORIGINAL_SEQ_LEN
    )
    parser.add_argument("--max-group-size", type=int, default=2**40)
    parser.add_argument(
        "--interdocument-masking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--truncate-trailing-masked",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--optimized-datasource-id-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-non-sorted-sources-in-dataloader",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-images", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = write_megatron_pretokenized_config_from_jsonl_config(
        config_path=args.config,
        output_path=args.output,
        pretokenized_data_dir=args.pretokenized_data_dir,
        epoch_index_cache_path=args.epoch_index_cache_path,
        original_seq_len=args.original_seq_len,
        max_group_size=args.max_group_size,
        interdocument_masking=args.interdocument_masking,
        truncate_trailing_masked=args.truncate_trailing_masked,
        optimized_datasource_id_sampling=args.optimized_datasource_id_sampling,
        allow_non_sorted_sources_in_dataloader=args.allow_non_sorted_sources_in_dataloader,
        seed=args.seed,
        skip_images=args.skip_images,
    )
    print(f"Wrote {len(config.sources)} sources to {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
