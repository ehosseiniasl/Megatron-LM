"""Pydantic config for the pretokenized dataloader.

This is the contract between the Mistral launcher and Megatron's pretrain_gpt.py.
The Mistral launcher builds a PretokenizedConfig, serializes it to YAML, and
Megatron reads it back. Both sides validate against the same schema.

Bump CONFIG_VERSION when changing the schema in a backward-incompatible way.
"""

from pydantic import BaseModel, Field, field_validator

CONFIG_VERSION = 9


class SourceConfig(BaseModel):
    prefix_path: str
    weight: float
    percentage_data: float
    sample_type: str = "pretrain"
    image_transform_type: str | None = None
    n_epoches_to_prebuild: int | None = Field(default=None, ge=1)


class PretokenizedValidationConfig(BaseModel):
    """Validation mix for the fixed validation iterator.

    When present, Megatron builds a fixed validation iterator that replays the
    same batches every eval round. valid_num_tokens is the global eval horizon
    (across all DP ranks) per round; it is converted to Megatron's eval_iters
    internally. valid_seed must be set explicitly (typically distinct from the
    train seed) so validation sampling is independent of training.
    """

    valid_sources: list[SourceConfig]
    valid_num_tokens: int
    valid_seed: int


class PretokenizedPerplexityEvalConfig(BaseModel):
    """Document-level fixed-overlap perplexity eval over pretokenized sources."""

    name: str
    sources: list[SourceConfig]
    fixed_overlap: int
    fixed_overlap_eval_seq_len: int | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        assert value, "perplexity eval set name must be non-empty"
        return value


class PretokenizedPerSourceValidationSourceConfig(SourceConfig):
    """One named physical datasource whose loss is reported independently."""

    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        assert value.strip(), "per-source validation name must be non-empty"
        assert value == value.strip() and not any(
            character in value for character in "\r\n\t|"
        ), f"per-source validation name contains unsupported characters: {value!r}"
        return value

class PretokenizedPerSourceValidationConfig(BaseModel):
    """Physical datasources whose validation losses are reported independently."""

    sources: list[PretokenizedPerSourceValidationSourceConfig]
    valid_seed: int
    epoch_index_cache_path: str | None = None

    @field_validator("sources")
    @classmethod
    def validate_sources(
        cls, value: list[PretokenizedPerSourceValidationSourceConfig]
    ) -> list[PretokenizedPerSourceValidationSourceConfig]:
        assert value, "per-source validation requires at least one source"
        names = [source.name for source in value]
        assert len(names) == len(set(names)), (
            f"per-source validation names must be unique, got {names}"
        )
        prefix_paths = [source.prefix_path for source in value]
        assert len(prefix_paths) == len(set(prefix_paths)), (
            "per-source validation prefix paths must be unique, "
            f"got {prefix_paths}"
        )
        return value


class PretokenizedConfig(BaseModel):
    version: int = CONFIG_VERSION
    sources: list[SourceConfig]
    bos_id: int
    eos_id: int
    pad_id: int
    img_id: int
    img_break_id: int
    img_end_id: int
    seed: int
    epoch_index_cache_path: str
    interdocument_masking: bool = True
    # Required so every config explicitly selects its document-boundary convention.
    drop_bos: bool
    original_seq_len: int
    # Kept explicit for parity checks with Mistral configs. The Megatron fork
    # currently only supports the Mistral default behavior.
    truncate_trailing_masked: bool
    tokenizer_name: str | None = None
    optimized_datasource_id_sampling: bool = True
    allow_non_sorted_sources_in_dataloader: bool = False
    skip_images: bool = False
    validation_set: PretokenizedValidationConfig | None = None
    per_source_validation: PretokenizedPerSourceValidationConfig | None = None
    perplexity_eval_sets: list[PretokenizedPerplexityEvalConfig] = Field(
        default_factory=list
    )

    @field_validator("truncate_trailing_masked")
    @classmethod
    def validate_truncate_trailing_masked(cls, value: bool) -> bool:
        assert value is True, (
            "Megatron requires truncate_trailing_masked=true. The field is kept "
            "explicit so configs can be compared with the Mistral codebase."
        )
        return value
