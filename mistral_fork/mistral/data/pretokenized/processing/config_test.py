import json
from pathlib import Path

import pytest

from mistral_adapter.pretokenized_config_utils import build_image_encoder_fn
from mistral_fork.mistral.data.pretokenized import cache_key
from mistral_fork.mistral.data.pretokenized.document_filters import (
    BadDocumentFilter,
    PdsFilter,
    RareTokenFilter,
)
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.processing import config as config_module


def _jsonl_pretokenization_config() -> config_module.JsonlPretokenizationConfig:
    return config_module.JsonlPretokenizationConfig.model_validate(
        {
            "version": config_module.JSONL_PRETOKENIZATION_CONFIG_VERSION,
            "tokenizer_path": "/tokenizers/lobstenizer_hf",
            "jsonl_root_data_dir": "/data",
            "sources": [
                {
                    "jsonl_dir": "/data/source",
                    "weight": 1.0,
                    "percentage_data": 0.5,
                    "sample_type": "pretrain",
                    "filter": "pretrain_v4__lobstral",
                    "rare_token_filter": None,
                    "pds_filter": None,
                }
            ],
        }
    )


def _write_tiktoken_vocab(path: Path) -> None:
    path.write_text(
        json.dumps([{"rank": 0, "token_bytes": "AA==", "token_str": ""}]),
        encoding="utf-8",
    )


def test_tokenizer_name_from_tiktoken_json_config(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "n4-v6-262k.json"
    _write_tiktoken_vocab(tokenizer_path)
    config = _jsonl_pretokenization_config().model_copy(
        update={"tokenizer_path": str(tokenizer_path)}
    )

    assert (
        config_module._tokenizer_name_from_config(config)
        == config_module.TokenizerName.N4_V6_262K_ML4_REGEX
    )


@pytest.mark.parametrize(
    ("directory", "expected_name"),
    [
        ("lobstenizer_hf", config_module.TokenizerName.PRETRAIN_V3_LOBSTENIZER),
        ("ml5_tokenizer_hf", config_module.TokenizerName.ML5_HF_TOKENIZER),
    ],
)
def test_tokenizer_name_from_huggingface_config(
    directory: str, expected_name: config_module.TokenizerName
) -> None:
    config = _jsonl_pretokenization_config().model_copy(
        update={"tokenizer_path": f"/tokenizers/{directory}"}
    )

    assert config_module._tokenizer_name_from_config(config) == expected_name


def test_unregistered_huggingface_config_has_no_cache_key() -> None:
    config = _jsonl_pretokenization_config().model_copy(
        update={"tokenizer_path": "/tokenizers/unregistered_hf"}
    )

    with pytest.raises(KeyError, match="unregistered_hf"):
        config_module._tokenizer_name_from_config(config)


def test_tiktoken_json_config_requires_registered_cache_key(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "custom-tiktoken.json"
    _write_tiktoken_vocab(tokenizer_path)
    config = _jsonl_pretokenization_config().model_copy(
        update={"tokenizer_path": str(tokenizer_path)}
    )

    with pytest.raises(KeyError, match="custom-tiktoken.json"):
        config_module._tokenizer_name_from_config(config)


def test_tiktoken_json_config_supports_multimodal_epoch_index_building(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "n4-v6-262k.json"
    _write_tiktoken_vocab(tokenizer_path)
    jsonl_root = tmp_path / "data"
    source_dir = jsonl_root / "source"
    base_config = _jsonl_pretokenization_config()
    source = base_config.sources[0].model_copy(
        update={
            "jsonl_dir": str(source_dir),
            "sample_type": config_module.SampleType.PRETRAIN_IMAGE,
            "image_transform_type": "crawl_default_no_high_res",
            "filter": FilterArgsName.MULTIMODAL_V4__LOBSTRAL,
            "rare_token_filter": RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
        }
    )
    config = base_config.model_copy(
        update={
            "tokenizer_path": str(tokenizer_path),
            "jsonl_root_data_dir": str(jsonl_root),
            "sources": [source],
        }
    )
    pretokenized_data_dir = tmp_path / "pretok"
    _, source_pretokenized_dir = config_module.resolve_config_source(
        config.sources[0],
        jsonl_root=jsonl_root,
        tokenizer_name=config_module.TokenizerName.N4_V6_262K,
        pretokenized_data_dir=pretokenized_data_dir,
    )
    merged_dir = config_module.get_merged_folder(
        source_pretokenized_dir, max_group_size_in_byte=2**40
    )
    merged_dir.mkdir(parents=True)
    (merged_dir / "chunks_0.bin").write_bytes(b"pretend pretokenized tokens")

    megatron_config = (
        config_module.build_megatron_pretokenized_config_from_jsonl_config(
            config,
            pretokenized_data_dir=pretokenized_data_dir,
            epoch_index_cache_path=tmp_path / "epoch_index",
            original_seq_len=4096,
            default_max_group_size=2**40,
            interdocument_masking=False,
            skip_images=False,
        )
    )

    assert megatron_config.version == config_module.CONFIG_VERSION == 9
    assert megatron_config.interdocument_masking is False
    assert megatron_config.drop_bos is True
    assert megatron_config.bos_id == 1
    assert megatron_config.eos_id == 2
    assert megatron_config.pad_id == 3
    assert megatron_config.img_id == 18
    assert megatron_config.img_break_id == 20
    assert megatron_config.img_end_id == 19
    assert megatron_config.skip_images is False
    image_encoder_builder = build_image_encoder_fn(megatron_config)
    assert image_encoder_builder is not None
    image_encoder = image_encoder_builder("crawl_default_no_high_res")
    assert image_encoder.to_infos_dict()["special_ids"] == {
        "img": 18,
        "img_break": 20,
        "img_end": 19,
    }


@pytest.mark.parametrize("field", ["rare_token_filter", "pds_filter"])
def test_jsonl_source_requires_explicit_nullable_filters(field: str) -> None:
    raw_config = _jsonl_pretokenization_config().model_dump(mode="json")
    del raw_config["sources"][0][field]

    with pytest.raises(ValueError, match=field):
        config_module.JsonlPretokenizationConfig.model_validate(raw_config)


def test_jsonl_config_rejects_duplicate_source_paths() -> None:
    raw_source = {
        "jsonl_dir": "/data/source",
        "weight": 1.0,
        "percentage_data": 1.0,
        "sample_type": "pretrain",
        "filter": "pretrain_v4__lobstral",
        "rare_token_filter": None,
        "pds_filter": None,
    }

    with pytest.raises(
        ValueError,
        match=r"Duplicate JSONL source path /data/source in sources\[0\] and sources\[1\]",
    ):
        config_module.JsonlPretokenizationConfig.model_validate(
            {
                "tokenizer_path": "/tokenizers/lobstenizer_hf",
                "jsonl_root_data_dir": "/data",
                "sources": [raw_source, raw_source.copy()],
            }
        )


def test_sorted_source_configs_match_mistral_source_order() -> None:
    base_source = _jsonl_pretokenization_config().sources[0]
    sources = [
        base_source.model_copy(
            update={
                "jsonl_dir": "/data/z-image",
                "sample_type": config_module.SampleType.PRETRAIN_IMAGE,
            }
        ),
        base_source.model_copy(update={"jsonl_dir": "/data/z-text"}),
        base_source.model_copy(
            update={
                "jsonl_dir": "/data/a-image",
                "sample_type": config_module.SampleType.PRETRAIN_IMAGE,
            }
        ),
        base_source.model_copy(update={"jsonl_dir": "/data/a-text"}),
    ]

    sorted_sources = config_module._sorted_source_configs(sources)

    assert [source.jsonl_dir for source in sorted_sources] == [
        "/data/a-text",
        "/data/z-text",
        "/data/a-image",
        "/data/z-image",
    ]


def test_resolve_config_source_uses_source_level_jsonl_root() -> None:
    source = config_module.JsonlSourceConfig.model_validate(
        {
            "jsonl_dir": "/nvidia-root/english/math",
            "jsonl_root_data_dir": "/nvidia-root",
            "weight": 1.0,
            "percentage_data": 1.0,
            "sample_type": "pretrain",
            "filter": "pretrain_v4__lobstral",
            "rare_token_filter": None,
            "pds_filter": None,
        }
    )

    _, pretok_dir = config_module.resolve_config_source(
        source,
        jsonl_root=Path("/mistral-root"),
        tokenizer_name=config_module.TokenizerName.PRETRAIN_V3_LOBSTENIZER,
        pretokenized_data_dir=Path("/pretok"),
    )

    assert pretok_dir.parts[-2:] == ("english", "math")
    assert "nvidia-root" not in pretok_dir.parts


def test_explicit_null_filters_preserve_legacy_cache_path() -> None:
    config = _jsonl_pretokenization_config()

    assert config.sources[0].rare_token_filter is None
    assert config.sources[0].pds_filter is None
    assert config.sources[0].bad_document_filter is None
    _, pretok_dir = config_module.resolve_config_source(
        config.sources[0],
        jsonl_root=Path(config.jsonl_root_data_dir),
        tokenizer_name=config_module.TokenizerName.PRETRAIN_V3_LOBSTENIZER,
        pretokenized_data_dir=Path("/pretok"),
    )

    assert pretok_dir == Path(
        "/pretok/pretrain_v3_lobstenizer/pretrain_v4__lobstral/source"
    )


def test_rare_token_and_bad_document_filters_are_added_to_cache_path() -> None:
    config = _jsonl_pretokenization_config()
    source = config.sources[0].model_copy(
        update={
            "rare_token_filter": RareTokenFilter.RARE_TOKENS_DEBUG,
            "bad_document_filter": BadDocumentFilter.NVIDIA_BAD_DOCUMENTS_DEBUG,
        }
    )

    _, pretok_dir = config_module.resolve_config_source(
        source,
        jsonl_root=Path(config.jsonl_root_data_dir),
        tokenizer_name=config_module.TokenizerName.PRETRAIN_V3_LOBSTENIZER,
        pretokenized_data_dir=Path("/pretok"),
    )

    assert pretok_dir == Path(
        "/pretok/pretrain_v3_lobstenizer/"
        "pretrain_v4__lobstral_rare_tokens_DEBUG_nvidia_bad_documents_DEBUG/source"
    )


def _multimodal_config(
    *,
    tokenizer_path: str,
    filter_name: str,
    sample_type: str = "pretrain_ocr",
    rare_token_filter: RareTokenFilter | None,
    pds_filter: PdsFilter | None,
) -> config_module.JsonlPretokenizationConfig:
    source = {
        "jsonl_dir": "/data/source",
        "weight": 1.0,
        "percentage_data": 1.0,
        "sample_type": sample_type,
        "filter": filter_name,
        "rare_token_filter": (
            rare_token_filter.value if rare_token_filter is not None else None
        ),
        "pds_filter": pds_filter.value if pds_filter is not None else None,
    }
    return config_module.JsonlPretokenizationConfig.model_validate(
        {
            "tokenizer_path": tokenizer_path,
            "jsonl_root_data_dir": "/data",
            "sources": [source],
        }
    )


def _grounding_config(
    tokenizer_path: str,
    *,
    rare_token_filter: RareTokenFilter | None = (
        RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1
    ),
    pds_filter: PdsFilter | None = PdsFilter.NVIDIA_PDS_QUARANTINE_V2,
) -> config_module.JsonlPretokenizationConfig:
    return _multimodal_config(
        tokenizer_path=tokenizer_path,
        filter_name="multimodal_v1__grounding",
        rare_token_filter=rare_token_filter,
        pds_filter=pds_filter,
    )


@pytest.mark.parametrize(
    ("tokenizer_path", "expected_name"),
    [
        (
            "/tokenizers/n4-v6-262k.json",
            config_module.TokenizerName.N4_V6_262K_ML4_REGEX,
        ),
        ("/tokenizers/ml5_tokenizer_hf", config_module.TokenizerName.ML5_HF_TOKENIZER),
    ],
)
@pytest.mark.parametrize(
    "filter_name",
    ["multimodal_v4__lobstral", "multimodal_v1__grounding"],
)
def test_n4_and_ml5_share_allowed_multimodal_profiles_and_filters(
    tokenizer_path: str,
    expected_name: config_module.TokenizerName,
    filter_name: str,
) -> None:
    config = _multimodal_config(
        tokenizer_path=tokenizer_path,
        filter_name=filter_name,
        rare_token_filter=RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
        pds_filter=PdsFilter.NVIDIA_PDS_QUARANTINE_V2,
    )

    assert config_module._tokenizer_name_from_config(config) is expected_name
    assert (
        config.sources[0].rare_token_filter is RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1
    )
    assert config.sources[0].pds_filter is PdsFilter.NVIDIA_PDS_QUARANTINE_V2


def test_lobstenizer_can_select_pds_filter_explicitly() -> None:
    config = _multimodal_config(
        tokenizer_path="/tokenizers/lobstenizer_hf",
        filter_name="multimodal_v4__lobstral",
        rare_token_filter=None,
        pds_filter=PdsFilter.NVIDIA_PDS_QUARANTINE_V1,
    )

    assert config.sources[0].pds_filter is PdsFilter.NVIDIA_PDS_QUARANTINE_V1
    _, pretok_dir = config_module.resolve_config_source(
        config.sources[0],
        jsonl_root=Path(config.jsonl_root_data_dir),
        tokenizer_name=config_module.TokenizerName.PRETRAIN_V3_LOBSTENIZER,
        pretokenized_data_dir=Path("/pretok"),
    )
    assert pretok_dir == Path(
        "/pretok/pretrain_v3_lobstenizer/"
        "multimodal_v4__lobstral_nvidia_pds_quarantine_v1/source"
    )


@pytest.mark.parametrize(
    "tokenizer_path",
    ["/tokenizers/n4-v6-262k.json", "/tokenizers/ml5_tokenizer_hf"],
)
def test_n4_and_ml5_accept_explicit_null_pds_filter_for_text_sources(
    tokenizer_path: str,
) -> None:
    config = _multimodal_config(
        tokenizer_path=tokenizer_path,
        filter_name="multimodal_v4__lobstral",
        sample_type="pretrain",
        rare_token_filter=RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
        pds_filter=None,
    )

    assert config.sources[0].pds_filter is None


@pytest.mark.parametrize(
    ("tokenizer_path", "tokenizer_name"),
    [
        (
            "/tokenizers/n4-v6-262k.json",
            config_module.TokenizerName.N4_V6_262K_ML4_REGEX,
        ),
        (
            "/tokenizers/ml5_tokenizer_hf",
            config_module.TokenizerName.ML5_HF_TOKENIZER,
        ),
    ],
)
def test_no_max_length_profile_has_versioned_ml4_cache_path(
    tokenizer_path: str,
    tokenizer_name: config_module.TokenizerName,
) -> None:
    config = _multimodal_config(
        tokenizer_path=tokenizer_path,
        filter_name="pretrain_v4__lobstral_no_max_length",
        sample_type="pretrain",
        rare_token_filter=RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
        pds_filter=None,
    )

    _, pretok_dir = config_module.resolve_config_source(
        config.sources[0],
        jsonl_root=Path(config.jsonl_root_data_dir),
        tokenizer_name=tokenizer_name,
        pretokenized_data_dir=Path("/pretok"),
    )

    assert pretok_dir == Path(
        f"/pretok/{tokenizer_name.value}/"
        "pretrain_v4__lobstral_no_max_length_rare_tokens_n4_v6_262k_v1/source"
    )


def test_text_source_rejects_non_null_pds_filter() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "pretrain is not multimodal and requires pds_filter=null, "
            "got nvidia_pds_quarantine_v2"
        ),
    ):
        _multimodal_config(
            tokenizer_path="/tokenizers/n4-v6-262k.json",
            filter_name="multimodal_v4__lobstral",
            sample_type="pretrain",
            rare_token_filter=RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
            pds_filter=PdsFilter.NVIDIA_PDS_QUARANTINE_V2,
        )


def test_tokenizer_policy_registry_is_explicit_and_ml5_matches_n4() -> None:
    policies = cache_key.RARE_TOKEN_FILTER_POLICY_BY_TOKENIZER
    ml4_policy = policies[config_module.TokenizerName.N4_V6_262K_ML4_REGEX]

    assert set(policies) == set(config_module.TokenizerName)
    assert set(policies[config_module.TokenizerName.PRETRAIN_V3_LOBSTENIZER]) == set(
        FilterArgsName
    )
    assert policies[config_module.TokenizerName.ML5_HF_TOKENIZER] is ml4_policy
    assert set(ml4_policy) == {
        FilterArgsName.PRETRAIN_V4__LOBSTRAL_NO_MAX_LENGTH,
        FilterArgsName.MULTIMODAL_V4__LOBSTRAL,
        FilterArgsName.MULTIMODAL_V1__GROUNDING,
    }
    pds_policies = cache_key.PDS_FILTER_POLICY_BY_TOKENIZER
    assert set(pds_policies) == set(config_module.TokenizerName)
    assert pds_policies[config_module.TokenizerName.PRETRAIN_V3_LOBSTENIZER] is None
    assert (
        pds_policies[config_module.TokenizerName.ML5_HF_TOKENIZER]
        is pds_policies[config_module.TokenizerName.N4_V6_262K_ML4_REGEX]
        is PdsFilter.NVIDIA_PDS_QUARANTINE_V2
    )


@pytest.mark.parametrize(
    "tokenizer_path",
    ["/tokenizers/n4-v6-262k.json", "/tokenizers/ml5_tokenizer_hf"],
)
def test_n4_and_ml5_reject_the_same_unsupported_filter_profile(
    tokenizer_path: str,
) -> None:
    with pytest.raises(ValueError, match="requires one of filter profiles"):
        _multimodal_config(
            tokenizer_path=tokenizer_path,
            filter_name="pretrain_v4__lobstral",
            rare_token_filter=RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
            pds_filter=PdsFilter.NVIDIA_PDS_QUARANTINE_V2,
        )


@pytest.mark.parametrize(
    ("tokenizer_path", "tokenizer_name", "expected_cache_path"),
    [
        (
            "/tokenizers/n4-v6-262k.json",
            config_module.TokenizerName.N4_V6_262K_ML4_REGEX,
            "/pretok/n4_v6_262k_ml4_regex/"
            "multimodal_v1__grounding_rare_tokens_n4_v6_262k_v1_"
            "nvidia_pds_quarantine_v2/source",
        ),
        (
            "/tokenizers/ml5_tokenizer_hf",
            config_module.TokenizerName.ML5_HF_TOKENIZER,
            "/pretok/ml5_hf_tokenizer/"
            "multimodal_v1__grounding_rare_tokens_n4_v6_262k_v1_"
            "nvidia_pds_quarantine_v2/source",
        ),
    ],
)
def test_n4_and_ml5_use_same_explicit_filters_and_cache_suffix(
    tokenizer_path: str,
    tokenizer_name: config_module.TokenizerName,
    expected_cache_path: str,
) -> None:
    config = _grounding_config(tokenizer_path)

    assert (
        config.sources[0].rare_token_filter is RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1
    )
    assert config.sources[0].pds_filter is PdsFilter.NVIDIA_PDS_QUARANTINE_V2
    _, pretok_dir = config_module.resolve_config_source(
        config.sources[0],
        jsonl_root=Path(config.jsonl_root_data_dir),
        tokenizer_name=tokenizer_name,
        pretokenized_data_dir=Path("/pretok"),
    )
    assert pretok_dir == Path(expected_cache_path)


@pytest.mark.parametrize(
    ("tokenizer_path", "tokenizer_name"),
    [
        ("/tokenizers/n4-v6-262k.json", "n4_v6_262k_ml4_regex"),
        ("/tokenizers/ml5_tokenizer_hf", "ml5_hf_tokenizer"),
    ],
)
def test_n4_and_ml5_reject_incompatible_rare_token_filter(
    tokenizer_path: str, tokenizer_name: str
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            f"multimodal_v1__grounding with {tokenizer_name} requires "
            "rare_token_filter=rare_tokens_n4_v6_262k_v1, got rare_tokens_DEBUG"
        ),
    ):
        _grounding_config(
            tokenizer_path,
            rare_token_filter=RareTokenFilter.RARE_TOKENS_DEBUG,
        )


@pytest.mark.parametrize(
    ("rare_token_filter", "pds_filter", "match"),
    [
        (
            None,
            PdsFilter.NVIDIA_PDS_QUARANTINE_V2,
            "requires rare_token_filter=rare_tokens_n4_v6_262k_v1, got null",
        ),
        (
            RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
            None,
            "requires pds_filter=nvidia_pds_quarantine_v2, got null",
        ),
        (
            RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
            PdsFilter.NVIDIA_PDS_QUARANTINE_V1,
            "requires pds_filter=nvidia_pds_quarantine_v2, "
            "got nvidia_pds_quarantine_v1",
        ),
    ],
)
def test_n4_requires_explicit_filters(
    rare_token_filter: RareTokenFilter | None,
    pds_filter: PdsFilter | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _grounding_config(
            "/tokenizers/n4-v6-262k.json",
            rare_token_filter=rare_token_filter,
            pds_filter=pds_filter,
        )


@pytest.mark.parametrize(
    ("rare_token_filter", "pds_filter", "match"),
    [
        (
            None,
            PdsFilter.NVIDIA_PDS_QUARANTINE_V2,
            "requires rare_token_filter=rare_tokens_n4_v6_262k_v1, got null",
        ),
        (
            RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
            None,
            "requires pds_filter=nvidia_pds_quarantine_v2, got null",
        ),
        (
            RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
            PdsFilter.NVIDIA_PDS_QUARANTINE_V1,
            "requires pds_filter=nvidia_pds_quarantine_v2, "
            "got nvidia_pds_quarantine_v1",
        ),
    ],
)
def test_cache_key_rechecks_explicit_filters(
    rare_token_filter: RareTokenFilter | None,
    pds_filter: PdsFilter | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        cache_key.compute_cache_key(
            profile=FilterArgsName.MULTIMODAL_V4__LOBSTRAL,
            tokenizer_name=config_module.TokenizerName.N4_V6_262K_ML4_REGEX,
            sample_type=config_module.SampleType.PRETRAIN_IMAGE,
            rare_token_filter=rare_token_filter,
            pds_filter=pds_filter,
        )
