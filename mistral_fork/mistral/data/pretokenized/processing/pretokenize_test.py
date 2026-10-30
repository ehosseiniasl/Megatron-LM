from __future__ import annotations

import json
from pathlib import Path

import pytest

from mistral_fork.mistral.data.pretokenized import document_filters
from mistral_fork.mistral.data.pretokenized.cache_key import TokenizerName
from mistral_fork.mistral.data.pretokenized.document_filters import (
    BadDocumentFilter,
    PdsFilter,
    RareTokenFilter,
)
from mistral_fork.mistral.data.pretokenized.processing.config import JsonlSourceConfig
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
    ChunkStats,
    PretokenizedDatasourceMetadataV2,
)
from mistral_fork.mistral.data.pretokenized.processing.merge import merge_source
from mistral_fork.mistral.data.pretokenized.processing.paths_helpers import (
    PrefixPaths,
    get_datasource_metadata_v2_path,
    get_single_chunk_dir,
    get_single_chunk_prefix,
)
from mistral_fork.mistral.data.pretokenized.processing.pretokenize import (
    _simple_tokenizer_name_from_config,
    _source_pretok_dir,
    build_metadata,
    pretokenize_sample,
    tokenize_file_to_chunk,
)


class _Tokenizer:
    bos_id = 1
    eos_id = 2

    def __init__(self) -> None:
        self.tokenized_texts: list[str] = []

    def tokenize(self, text: str) -> list[int]:
        self.tokenized_texts.append(text)
        return [99 if text == "rare" else 3]


@pytest.mark.parametrize(
    ("directory", "expected_name"),
    [
        ("lobstenizer_hf", TokenizerName.PRETRAIN_V3_LOBSTENIZER),
        ("ml5_tokenizer_hf", TokenizerName.ML5_HF_TOKENIZER),
    ],
)
def test_lightweight_config_uses_huggingface_cache_key(
    directory: str, expected_name: TokenizerName
) -> None:
    tokenizer_name = _simple_tokenizer_name_from_config(
        {"tokenizer_path": f"/tokenizers/{directory}"}
    )

    assert tokenizer_name == expected_name.value


def test_lightweight_config_rejects_unregistered_huggingface_tokenizer() -> None:
    with pytest.raises(KeyError, match="unregistered_hf"):
        _simple_tokenizer_name_from_config(
            {"tokenizer_path": "/tokenizers/unregistered_hf"}
        )


def test_lightweight_config_requires_tokenizer_path() -> None:
    with pytest.raises(KeyError, match="tokenizer_path"):
        _simple_tokenizer_name_from_config({})


def _source_config(
    jsonl_path: Path,
    *,
    rare_token_filter: RareTokenFilter | None,
    bad_document_filter: BadDocumentFilter | None,
) -> JsonlSourceConfig:
    raw_config = {
        "jsonl_dir": str(jsonl_path),
        "weight": 1.0,
        "percentage_data": 1.0,
        "sample_type": "pretrain",
        "filter": "pretrain_v4__lobstral",
        "rare_token_filter": (
            rare_token_filter.value if rare_token_filter is not None else None
        ),
        "pds_filter": None,
    }
    if bad_document_filter is not None:
        raw_config["bad_document_filter"] = bad_document_filter.value
    return JsonlSourceConfig.model_validate(raw_config)


def _write_jsonl(path: Path, texts: list[str]) -> list[int]:
    lines = [json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts]
    offsets = []
    byte_offset = 0
    for line in lines:
        offsets.append(byte_offset)
        byte_offset += len(line.encode("utf-8"))
    path.write_text("".join(lines), encoding="utf-8")
    return offsets


def test_pretokenize_midtraining_segments() -> None:
    source_config = JsonlSourceConfig.model_validate(
        {
            "jsonl_dir": "/data/midtraining.jsonl",
            "weight": 1.0,
            "percentage_data": 1.0,
            "sample_type": "pretrain",
            "filter": "pretrain_v4__lobstral",
            "masking_strategy": "MASK_MIDTRAINING_SEGMENTS",
            "rare_token_filter": None,
            "pds_filter": None,
        }
    )
    metadata = build_metadata(
        source_config=source_config,
        input_path=Path(source_config.jsonl_dir),
        n_chunks=1,
        tokenizer_path="/tokenizer",
        validation_fraction=0.0,
        guarantee_validation_per_chunk=False,
        tokenizer_name=TokenizerName.PRETRAIN_V3_LOBSTENIZER,
    )
    tokenizer = _Tokenizer()

    document = pretokenize_sample(
        sample={
            "source": "midtraining",
            "text": "maskedtrained",
            "mask_segments": [[0, 6, False], [6, 13, True]],
        },
        source_config=source_config,
        tokenizer=tokenizer,
        source_metadata=metadata,
    )

    assert document is not None
    assert tokenizer.tokenized_texts == ["masked", "trained"]
    assert document.tokens == [1, 3, 3, 2]
    assert document.mask is not None
    assert document.mask.sizes == [1, 1, 2]
    assert document.mask.mask == [True, False, True]


def _tokenize_chunk(
    *,
    tmp_path: Path,
    jsonl_path: Path,
    rare_token_filter: RareTokenFilter | None = None,
    bad_document_filter: BadDocumentFilter | None = None,
) -> tuple[ChunkStats, _Tokenizer, Path, PretokenizedDatasourceMetadataV2]:
    source_config = _source_config(
        jsonl_path,
        rare_token_filter=rare_token_filter,
        bad_document_filter=bad_document_filter,
    )
    source_dir = tmp_path / "pretokenized-source"
    chunks_dir = source_dir / "chunks"
    dst_single_chunk_dir = get_single_chunk_dir(chunks_dir, jsonl_path)
    metadata = build_metadata(
        source_config=source_config,
        input_path=jsonl_path,
        n_chunks=1,
        tokenizer_path="/tokenizer",
        validation_fraction=0.0,
        guarantee_validation_per_chunk=False,
        tokenizer_name=TokenizerName.PRETRAIN_V3_LOBSTENIZER,
    )
    tokenizer = _Tokenizer()
    stats = tokenize_file_to_chunk(
        jsonl_path=jsonl_path,
        dst_single_chunk_dir=dst_single_chunk_dir,
        source_config=source_config,
        tokenizer=tokenizer,
        source_metadata=metadata,
        validation_fraction=0.0,
        guarantee_validation_per_chunk=False,
    )
    return stats, tokenizer, source_dir, metadata


@pytest.mark.parametrize(
    ("rare_token_filter", "bad_document_filter", "expected_profile_dir"),
    [
        (None, None, "pretrain_v4__lobstral"),
        ("rare_tokens_DEBUG", None, "pretrain_v4__lobstral_rare_tokens_DEBUG"),
        (
            None,
            "nvidia_bad_documents_DEBUG",
            "pretrain_v4__lobstral_nvidia_bad_documents_DEBUG",
        ),
        (
            "rare_tokens_DEBUG",
            "nvidia_bad_documents_DEBUG",
            "pretrain_v4__lobstral_rare_tokens_DEBUG_nvidia_bad_documents_DEBUG",
        ),
    ],
)
def test_lightweight_missing_check_uses_optional_filter_cache_path(
    rare_token_filter: str | None,
    bad_document_filter: str | None,
    expected_profile_dir: str,
) -> None:
    source = {
        "jsonl_dir": "/data/source",
        "sample_type": "pretrain",
        "filter": "pretrain_v4__lobstral",
        "rare_token_filter": rare_token_filter,
        "pds_filter": None,
    }
    if bad_document_filter is not None:
        source["bad_document_filter"] = bad_document_filter

    pretok_dir = _source_pretok_dir(
        source,
        jsonl_root=Path("/data"),
        pretokenized_data_dir=Path("/pretok"),
        tokenizer_name=TokenizerName.PRETRAIN_V3_LOBSTENIZER,
    )

    assert pretok_dir == Path(
        "/pretok/pretrain_v3_lobstenizer", expected_profile_dir, "source"
    )


def test_nvidia_pds_quarantine_v2_extends_v1_manifests() -> None:
    v1_paths = document_filters.PDS_FILTER_PATHS[
        PdsFilter.NVIDIA_PDS_QUARANTINE_V1
    ]
    v2_paths = document_filters.PDS_FILTER_PATHS[
        PdsFilter.NVIDIA_PDS_QUARANTINE_V2
    ]

    assert set(document_filters.PDS_FILTER_PATHS) == set(PdsFilter)
    assert v2_paths[: len(v1_paths)] == v1_paths
    assert v2_paths[len(v1_paths) :] == (
        Path(
            "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/"
            "nv-00-10603-sandia/quarantine-0826/pulse_quarantine_0826.parquet"
        ),
        Path(
            "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/"
            "nv-00-10603-sandia/pulse_quarantine_0831/"
            "pulse_quarantine_0831.parquet"
        ),
    )


@pytest.mark.parametrize("field", ["rare_token_filter", "pds_filter"])
def test_lightweight_cache_path_requires_explicit_filters(field: str) -> None:
    source = {
        "jsonl_dir": "/data/source",
        "sample_type": "pretrain",
        "filter": "pretrain_v4__lobstral",
        "rare_token_filter": None,
        "pds_filter": None,
    }
    del source[field]

    with pytest.raises(KeyError, match=field):
        _source_pretok_dir(
            source,
            jsonl_root=Path("/data"),
            pretokenized_data_dir=Path("/pretok"),
            tokenizer_name=TokenizerName.PRETRAIN_V3_LOBSTENIZER,
        )


@pytest.mark.parametrize(
    ("tokenizer_name", "sample_type", "expected_pds_suffix"),
    [
        (TokenizerName.ML5_HF_TOKENIZER, "pretrain", ""),
        (
            TokenizerName.ML5_HF_TOKENIZER,
            "pretrain_ocr",
            "_nvidia_pds_quarantine_v2",
        ),
        (TokenizerName.N4_V6_262K_ML4_REGEX, "pretrain", ""),
        (
            TokenizerName.N4_V6_262K_ML4_REGEX,
            "pretrain_ocr",
            "_nvidia_pds_quarantine_v2",
        ),
    ],
)
def test_lightweight_cache_path_checks_explicit_filters(
    tokenizer_name: TokenizerName,
    sample_type: str,
    expected_pds_suffix: str,
) -> None:
    source = {
        "jsonl_dir": "/data/source",
        "sample_type": sample_type,
        "filter": "multimodal_v4__lobstral",
        "rare_token_filter": RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
        "pds_filter": (
            PdsFilter.NVIDIA_PDS_QUARANTINE_V2
            if expected_pds_suffix
            else None
        ),
    }

    pretok_dir = _source_pretok_dir(
        source,
        jsonl_root=Path("/data"),
        pretokenized_data_dir=Path("/pretok"),
        tokenizer_name=tokenizer_name,
    )

    assert pretok_dir == Path(
        "/pretok",
        tokenizer_name.value,
        f"multimodal_v4__lobstral_rare_tokens_n4_v6_262k_v1{expected_pds_suffix}",
        "source",
    )


def test_explicit_null_filters_preserve_pretokenization_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl_path = tmp_path / "no-document-filter.jsonl"
    _write_jsonl(jsonl_path, ["rare"])
    monkeypatch.setattr(document_filters, "RARE_TOKEN_IDS_DEBUG", frozenset({99}))

    stats, _, _, metadata = _tokenize_chunk(
        tmp_path=tmp_path,
        jsonl_path=jsonl_path,
    )

    assert stats.n_docs_train == 1
    assert stats.n_docs_filtered == 0
    assert metadata.rare_token_filter is None
    assert metadata.pds_filter is None
    assert metadata.bad_document_filter is None
    filter_args = json.loads(metadata.filter_args_json)
    assert "rare_token_filter" not in filter_args
    assert "pds_filter" not in filter_args
    assert "bad_document_filter" not in filter_args


def test_rare_token_filter_skips_tokenized_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl_path = tmp_path / "rare-tokens.jsonl"
    _write_jsonl(jsonl_path, ["keep", "rare"])
    monkeypatch.setattr(document_filters, "RARE_TOKEN_IDS_DEBUG", frozenset({99}))

    stats, tokenizer, _, metadata = _tokenize_chunk(
        tmp_path=tmp_path,
        jsonl_path=jsonl_path,
        rare_token_filter=RareTokenFilter.RARE_TOKENS_DEBUG,
    )

    assert tokenizer.tokenized_texts == ["keep", "rare"]
    assert stats.n_docs_total == 2
    assert stats.n_docs_train == 1
    assert stats.n_docs_filtered == 1
    assert metadata.rare_token_filter is RareTokenFilter.RARE_TOKENS_DEBUG
    assert metadata.bad_document_filter is None
    assert json.loads(metadata.filter_args_json)["rare_token_filter"] == {
        "name": "rare_tokens_DEBUG"
    }


def test_nvidia_bad_document_filter_uses_path_and_byte_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl_path = tmp_path / "nvidia.jsonl"
    offsets = _write_jsonl(jsonl_path, ["multibyte é", "bad"])
    monkeypatch.setattr(
        document_filters,
        "NVIDIA_BAD_DOCUMENT_BYTE_OFFSETS_DEBUG",
        {str(jsonl_path): frozenset({offsets[1]})},
    )

    stats, tokenizer, _, _ = _tokenize_chunk(
        tmp_path=tmp_path,
        jsonl_path=jsonl_path,
        bad_document_filter=BadDocumentFilter.NVIDIA_BAD_DOCUMENTS_DEBUG,
    )

    assert offsets[1] > 1
    assert tokenizer.tokenized_texts == ["multibyte é"]
    assert stats.n_docs_total == 2
    assert stats.n_docs_train == 1
    assert stats.n_docs_filtered == 1


def test_rare_token_and_bad_document_filters_can_be_combined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl_path = tmp_path / "combined.jsonl"
    offsets = _write_jsonl(jsonl_path, ["bad", "rare", "keep"])
    monkeypatch.setattr(document_filters, "RARE_TOKEN_IDS_DEBUG", frozenset({99}))
    monkeypatch.setattr(
        document_filters,
        "NVIDIA_BAD_DOCUMENT_BYTE_OFFSETS_DEBUG",
        {str(jsonl_path): frozenset({offsets[0]})},
    )

    stats, tokenizer, _, metadata = _tokenize_chunk(
        tmp_path=tmp_path,
        jsonl_path=jsonl_path,
        rare_token_filter=RareTokenFilter.RARE_TOKENS_DEBUG,
        bad_document_filter=BadDocumentFilter.NVIDIA_BAD_DOCUMENTS_DEBUG,
    )

    assert tokenizer.tokenized_texts == ["rare", "keep"]
    assert stats.n_docs_total == 3
    assert stats.n_docs_train == 1
    assert stats.n_docs_filtered == 2
    assert metadata.rare_token_filter is RareTokenFilter.RARE_TOKENS_DEBUG
    assert metadata.bad_document_filter is BadDocumentFilter.NVIDIA_BAD_DOCUMENTS_DEBUG
    filter_args = json.loads(metadata.filter_args_json)
    assert filter_args["rare_token_filter"] == {"name": "rare_tokens_DEBUG"}
    assert filter_args["bad_document_filter"] == {"name": "nvidia_bad_documents_DEBUG"}


def test_fully_filtered_chunk_is_valid_and_mergeable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl_path = tmp_path / "fully-filtered.jsonl"
    _write_jsonl(jsonl_path, ["rare"])
    monkeypatch.setattr(document_filters, "RARE_TOKEN_IDS_DEBUG", frozenset({99}))

    stats, _, source_dir, metadata = _tokenize_chunk(
        tmp_path=tmp_path,
        jsonl_path=jsonl_path,
        rare_token_filter=RareTokenFilter.RARE_TOKENS_DEBUG,
    )
    chunks_dir = source_dir / "chunks"
    chunk_paths = PrefixPaths.from_prefix(
        get_single_chunk_prefix(get_single_chunk_dir(chunks_dir, jsonl_path))
    )

    assert stats.n_docs_train == 0
    assert stats.n_docs_filtered == 1
    assert chunk_paths.bin.stat().st_size == 0
    assert chunk_paths.index.is_file()

    get_datasource_metadata_v2_path(chunks_dir).write_text(
        metadata.model_dump_json(indent=2), encoding="utf-8"
    )
    merged_dir = merge_source(source_dir=source_dir, max_group_size=1024)
    merged_bin = next(merged_dir.glob("chunks_*.bin"))

    assert merged_bin.stat().st_size == 0
    assert merged_bin.with_suffix(".idx").is_file()
