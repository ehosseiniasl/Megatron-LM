"""Tokenizer name registry and cache-key helpers for pretokenized data."""

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from mistral_fork.mistral.data.pretokenized.document_filters import (
    BadDocumentFilter,
    PdsFilter,
    RareTokenFilter,
)
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.source import SampleType

if TYPE_CHECKING:
    from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
        MaskingStrategy,
    )


class TokenizerName(StrEnum):
    """Registered tokenizer names.

    String values are stable and are used in pretokenized cache keys.
    """

    PRETRAIN_V3_LOBSTENIZER = "pretrain_v3_lobstenizer"
    ML5_HF_TOKENIZER = "ml5_hf_tokenizer"
    N4_V6_262K_ML4_REGEX = "n4_v6_262k_ml4_regex"


TIKTOKEN_JSON_TOKENIZER_NAMES_BY_FILENAME = {
    "n4-v6-262k.json": TokenizerName.N4_V6_262K_ML4_REGEX,
}
HUGGINGFACE_TOKENIZER_NAMES_BY_DIRECTORY = {
    "lobstenizer_hf": TokenizerName.PRETRAIN_V3_LOBSTENIZER,
    "ml5_tokenizer_hf": TokenizerName.ML5_HF_TOKENIZER,
}
TOKENIZER_NAMES_BY_PATH_NAME = {
    **TIKTOKEN_JSON_TOKENIZER_NAMES_BY_FILENAME,
    **HUGGINGFACE_TOKENIZER_NAMES_BY_DIRECTORY,
}


def tokenizer_name_from_path(tokenizer_path: str | Path) -> TokenizerName:
    return TOKENIZER_NAMES_BY_PATH_NAME[Path(tokenizer_path).name]


PRETRAIN_V3_RARE_TOKEN_FILTER_POLICY: dict[FilterArgsName, RareTokenFilter | None] = {
    FilterArgsName.NO_FILTER: None,
    FilterArgsName.PRETRAIN_V3__SINKDEEP: None,
    FilterArgsName.PRETRAIN_V4__LOBSTRAL: None,
    FilterArgsName.PRETRAIN_V4__LOBSTRAL_NO_MAX_LENGTH: None,
    FilterArgsName.MULTIMODAL_V4__LOBSTRAL: None,
    FilterArgsName.MULTIMODAL_V1__GROUNDING: None,
}
ML4_REGEX_RARE_TOKEN_FILTER_POLICY: dict[FilterArgsName, RareTokenFilter | None] = {
    FilterArgsName.PRETRAIN_V4__LOBSTRAL_NO_MAX_LENGTH: (
        RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1
    ),
    FilterArgsName.MULTIMODAL_V4__LOBSTRAL: (
        RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1
    ),
    FilterArgsName.MULTIMODAL_V1__GROUNDING: (
        RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1
    ),
}
RARE_TOKEN_FILTER_POLICY_BY_TOKENIZER: dict[
    TokenizerName, dict[FilterArgsName, RareTokenFilter | None]
] = {
    TokenizerName.PRETRAIN_V3_LOBSTENIZER: PRETRAIN_V3_RARE_TOKEN_FILTER_POLICY,
    TokenizerName.ML5_HF_TOKENIZER: ML4_REGEX_RARE_TOKEN_FILTER_POLICY,
    TokenizerName.N4_V6_262K_ML4_REGEX: ML4_REGEX_RARE_TOKEN_FILTER_POLICY,
}
PDS_FILTER_POLICY_BY_TOKENIZER: dict[TokenizerName, PdsFilter | None] = {
    TokenizerName.PRETRAIN_V3_LOBSTENIZER: None,
    TokenizerName.ML5_HF_TOKENIZER: PdsFilter.NVIDIA_PDS_QUARANTINE_V2,
    TokenizerName.N4_V6_262K_ML4_REGEX: PdsFilter.NVIDIA_PDS_QUARANTINE_V2,
}


def check_rare_tokens(
    *,
    profile: FilterArgsName,
    tokenizer_name: TokenizerName,
    rare_token_filter: RareTokenFilter | None,
) -> None:
    """Check that the explicit rare-token filter satisfies tokenizer policy."""

    tokenizer_policy = RARE_TOKEN_FILTER_POLICY_BY_TOKENIZER[tokenizer_name]
    if profile not in tokenizer_policy:
        profile_names = ", ".join(
            sorted(allowed_profile.value for allowed_profile in tokenizer_policy)
        )
        raise ValueError(
            f"{tokenizer_name.value} requires one of filter profiles {profile_names}, "
            f"but {profile.value} was configured"
        )
    required_filter = tokenizer_policy[profile]
    if required_filter is None:
        return
    if rare_token_filter is not required_filter:
        configured_value = (
            "null" if rare_token_filter is None else rare_token_filter.value
        )
        raise ValueError(
            f"{profile.value} with {tokenizer_name.value} requires "
            f"rare_token_filter={required_filter.value}, got {configured_value}"
        )


def check_pds(
    *,
    tokenizer_name: TokenizerName,
    sample_type: SampleType,
    pds_filter: PdsFilter | None,
) -> None:
    """Check that the explicit PDS filter satisfies tokenizer/sample policy."""

    if not sample_type.is_multimodal():
        if pds_filter is not None:
            raise ValueError(
                f"{sample_type.value} is not multimodal and requires "
                f"pds_filter=null, got {pds_filter.value}"
            )
        return

    required_filter = PDS_FILTER_POLICY_BY_TOKENIZER[tokenizer_name]
    if required_filter is None:
        return
    if pds_filter is not required_filter:
        configured_value = "null" if pds_filter is None else pds_filter.value
        raise ValueError(
            f"{sample_type.value} with {tokenizer_name.value} requires "
            f"pds_filter={required_filter.value}, got {configured_value}"
        )


def compute_cache_key(
    profile: FilterArgsName,
    tokenizer_name: TokenizerName,
    sample_type: SampleType,
    rare_token_filter: RareTokenFilter | None,
    pds_filter: PdsFilter | None,
    masking_strategy: "MaskingStrategy | None" = None,
    bad_document_filter: BadDocumentFilter | None = None,
) -> str:
    """Compute a stable cache key from all pretokenization transformations."""

    check_rare_tokens(
        profile=profile,
        tokenizer_name=tokenizer_name,
        rare_token_filter=rare_token_filter,
    )
    check_pds(
        tokenizer_name=tokenizer_name,
        sample_type=sample_type,
        pds_filter=pds_filter,
    )
    suffix = profile.value
    if masking_strategy is not None:
        suffix = f"{suffix}_{masking_strategy.value.lower()}"
    if rare_token_filter is not None:
        suffix = f"{suffix}_{rare_token_filter.value}"
    if pds_filter is not None:
        suffix = f"{suffix}_{pds_filter.value}"
    if bad_document_filter is not None:
        suffix = f"{suffix}_{bad_document_filter.value}"
    return f"{tokenizer_name.value}/{suffix}"
