import json
from pathlib import Path

import pytest

from mistral_fork.mistral.data.pretokenized.cache_key import TokenizerName
from mistral_fork.mistral.data.pretokenized.components import filters
from mistral_fork.mistral.data.pretokenized.document_filters import (
    PdsFilter,
    RareTokenFilter,
)
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.processing.config import JsonlSourceConfig
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
    ImageAnnotationType,
    MaskingStrategy,
    PretokenizedDatasourceMetadataV2,
    required_image_annotations_for_transform,
)
from mistral_fork.mistral.data.pretokenized.processing.pretokenize import (
    _geometry_text,
    _tokenize_multimodal_document,
    build_metadata,
    pretokenize_sample,
    tokenize_file_to_chunk,
)
from mistral_fork.mistral.data.source import SampleType


class _CharTokenizer:
    bos_id = 1
    eos_id = 2

    def tokenize(self, text: str) -> list[int]:
        return [ord(char) for char in text]


def _metadata() -> PretokenizedDatasourceMetadataV2:
    return PretokenizedDatasourceMetadataV2(
        filter_name=FilterArgsName.MULTIMODAL_V1__GROUNDING,
        tokenizer_name=TokenizerName.PRETRAIN_V3_LOBSTENIZER,
        filter_args_json="{}",
        tokenizer_path="/tokenizer",
        original_source="/source",
        n_chunks=1,
        validation_fraction=0.001,
        guarantee_validation_per_chunk=True,
        has_image_metadata=True,
        sample_type=SampleType.PRETRAIN_OCR,
        masking_strategy=MaskingStrategy.MASK_MMDOC_CONTENT,
        available_image_annotations=set(),
    )


def _expanded_mask(sizes: list[int], values: list[bool]) -> list[bool]:
    expanded = []
    for size, value in zip(sizes, values, strict=True):
        expanded.extend([value] * size)
    return expanded


def test_grounding_default_requires_l2_image_annotation() -> None:
    assert required_image_annotations_for_transform("grounding_default") == {
        ImageAnnotationType.L2_RECONSTRUCTION_SCORE
    }


def test_sometimes_high_res_default_does_not_require_l2_image_annotation() -> None:
    assert (
        required_image_annotations_for_transform("sometimes_high_res_default") == set()
    )


def test_mask_mmdoc_content_masks_masked_text() -> None:
    doc = _tokenize_multimodal_document(
        {
            "multimodal_document": {
                "content": [
                    "visible",
                    {"type": "masked_text", "text": "hidden"},
                    "visible2",
                ]
            }
        },
        _CharTokenizer(),
        ["visible", {"type": "masked_text", "text": "hidden"}, "visible2"],
        _metadata(),
        MaskingStrategy.MASK_MMDOC_CONTENT,
    )

    assert doc.mask is not None
    text = "".join(chr(token) for token in doc.tokens[1:-1])
    mask = _expanded_mask(doc.mask.sizes, doc.mask.mask)

    assert text == "visiblehiddenvisible2"
    assert len(mask) == len(doc.tokens)
    assert mask[1 : 1 + len("visible")] == [True] * len("visible")
    start = 1 + len("visible")
    assert mask[start : start + len("hidden")] == [False] * len("hidden")
    start += len("hidden")
    assert mask[start : start + len("visible2")] == [True] * len("visible2")


def test_grounding_geometry_serializes_like_mmdoc_content() -> None:
    assert (
        _geometry_text(
            [
                {
                    "top_left": [0.1, 0.2],
                    "bottom_right": [0.3, 0.4],
                    "label": "Cat",
                    "image_idx": 0,
                }
            ],
            fields=None,
            cur_image_idx=1,
        )
        == '[{"box_2d": [100, 200, 300, 400], "label": "cat"}]'
    )


def _grounding_source_config(jsonl_dir: str = "/source") -> JsonlSourceConfig:
    return JsonlSourceConfig.model_validate(
        {
            "jsonl_dir": jsonl_dir,
            "weight": 1.0,
            "percentage_data": 1.0,
            "sample_type": "pretrain_ocr",
            "filter": "multimodal_v1__grounding",
            "masking_strategy": "MASK_MMDOC_CONTENT",
            "rare_token_filter": RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
            "pds_filter": PdsFilter.NVIDIA_PDS_QUARANTINE_V2,
        }
    )


def _raw_image() -> dict:
    return {
        "doc_tar_offset": {
            "tar_path": "s3://images/image.tar",
            "offset": 0,
            "size": 50,
        },
        "width": 10,
        "height": 10,
        "hash": "image-hash",
    }


def test_grounding_end_to_end_serialization_and_loss_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        filters,
        "_load_pds_key_set",
        lambda pds_filter: (frozenset(), frozenset(), {}),
    )
    geometry = {
        "top_left": [0.1, 0.2],
        "bottom_right": [0.3, 0.4],
        "label": "Cat",
        "is_masked": True,
    }

    doc = pretokenize_sample(
        {
            "multimodal_document": {
                "content": [_raw_image(), geometry, "visible"]
            }
        },
        _grounding_source_config(),
        _CharTokenizer(),
        _metadata(),
    )

    assert doc is not None
    assert doc.mask is not None
    rendered = "".join(chr(token) for token in doc.tokens[1:-1])
    geometry_text = '[{"box_2d": [100, 200, 300, 400], "label": "cat"}]'
    assert rendered == geometry_text + "visible"
    expanded_mask = _expanded_mask(doc.mask.sizes, doc.mask.mask)
    geometry_start = 1
    assert expanded_mask[
        geometry_start : geometry_start + len(geometry_text)
    ] == [False] * len(geometry_text)
    text_start = geometry_start + len(geometry_text)
    assert expanded_mask[text_start : text_start + len("visible")] == [True] * len(
        "visible"
    )


def test_grounding_quarantine_runs_before_ordinary_image_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        filters,
        "_load_pds_key_set",
        lambda pds_filter: (
            frozenset({("images/image.tar", 0, 50)}),
            frozenset(),
            {},
        ),
    )

    def fail_if_called(*args: object, **kwargs: object) -> None:
        pytest.fail("ordinary image filtering ran before NV quarantine")

    monkeypatch.setattr(
        "mistral_fork.mistral.data.pretokenized.processing.pretokenize."
        "filter_multimodal_content",
        fail_if_called,
    )

    assert (
        pretokenize_sample(
            {
                "multimodal_document": {
                    "content": [_raw_image(), "visible"]
                }
            },
            _grounding_source_config(),
            _CharTokenizer(),
            _metadata(),
        )
        is None
    )


def test_grounding_quarantine_increments_dedicated_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        filters,
        "_load_nv_quarantine_key_set",
        lambda: (
            frozenset({("images/image.tar", 0, 50)}),
            frozenset(),
            {},
        ),
    )
    jsonl_path = tmp_path / "grounding.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "multimodal_document": {
                    "content": [_raw_image(), "visible"],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_config = _grounding_source_config(str(jsonl_path))
    metadata = build_metadata(
        source_config=source_config,
        input_path=jsonl_path,
        n_chunks=1,
        tokenizer_path="/tokenizer",
        validation_fraction=0.0,
        guarantee_validation_per_chunk=False,
        tokenizer_name=TokenizerName.PRETRAIN_V3_LOBSTENIZER,
    )

    stats = tokenize_file_to_chunk(
        jsonl_path=jsonl_path,
        dst_single_chunk_dir=tmp_path / "chunk",
        source_config=source_config,
        tokenizer=_CharTokenizer(),
        source_metadata=metadata,
        validation_fraction=0.0,
        guarantee_validation_per_chunk=False,
    )

    assert stats.n_docs_total == 1
    assert stats.n_docs_train == 0
    assert stats.n_docs_filtered == 1
    assert stats.n_docs_nv_quarantine_filtered == 1
