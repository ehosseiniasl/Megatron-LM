import json
from types import SimpleNamespace
from typing import Any

import pytest

from mistral_fork.mistral.data.pretokenized.components import filters
from mistral_fork.mistral.data.pretokenized.document_filters import (
    PdsFilter,
    RareTokenFilter,
)
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
    MaskingStrategy,
)


def _source(
    profile: FilterArgsName = FilterArgsName.MULTIMODAL_V1__GROUNDING,
    *,
    masking_strategy: MaskingStrategy | None = None,
    rare_token_filter: RareTokenFilter | None = None,
    pds_filter: PdsFilter | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        filter=profile,
        masking_strategy=masking_strategy,
        rare_token_filter=rare_token_filter,
        pds_filter=pds_filter,
        bad_document_filter=None,
    )


def _image(index: int = 0, **updates: Any) -> dict[str, Any]:
    image = {
        "doc_tar_offset": {
            "tar_path": f"s3://images/image-{index}.tar",
            "offset": index * 100,
            "size": 50,
        },
        "width": 10,
        "height": 10,
        "hash": f"hash-{index}",
    }
    image.update(updates)
    return image


def _geometries(content: list[Any]) -> list[dict[str, Any]]:
    geometries = []
    for item in content:
        if not isinstance(item, dict) or not filters.is_geometry_item(item):
            continue
        geometries.extend(filters._geometry_items(item))
    return geometries


def test_grounding_keeps_supported_geometry_formats_and_masking_fields() -> None:
    geometries = [
        {
            "top_left": [0.1, 0.2],
            "bottom_right": [0.3, 0.4],
            "is_masked": True,
        },
        {"box_2d": [100, 200, 300, 400], "label": "Cat"},
        {"point": {"x": 0.2, "y": 0.3}},
        {"point_2d": [200, 300]},
        {"center": [0.5, 0.5], "w": 0.2, "h": 0.3, "angle": 0.1},
        {"obb_2d": [500, 500, 200, 300, 45]},
        {
            "polygons": [
                {"vertices": [[0.1, 0.1], [0.3, 0.1], [0.2, 0.3]]}
            ]
        },
        {"mask_2d": [[100, 100, 300, 100, 200, 300]]},
    ]
    document = {
        "content": [
            _image(),
            geometries[0],
            {"geometries": geometries[1:], "container_field": "preserved"},
            "abc",
        ]
    }

    content = filters.filter_multimodal_content(document, _source())
    kept = _geometries(content)

    assert len(kept) == len(geometries)
    assert all(geometry["image_idx"] == 0 for geometry in kept)
    assert kept[0]["is_masked"] is True
    container = next(item for item in content if isinstance(item, dict) and "geometries" in item)
    assert container["container_field"] == "preserved"
    assert kept[1]["label"] == "Cat"


def test_grounding_filters_invalid_and_more_than_500_geometries() -> None:
    valid_points = [{"point": [0.5, 0.5]} for _ in range(501)]
    invalid = [
        {"point": [1.1, 0.5]},
        {"center": [0.5, 0.5], "w": 2.0, "h": 1.0},
        {"top_left": [0.2, 0.2], "bottom_right": [0.2, 0.2]},
        {"top_left": [0.8, 0.8], "bottom_right": [0.2, 0.2]},
        {"obb_2d": [500, 500, 0, 300, 45]},
        {"center": [0.5, 0.5], "w": -0.2, "h": -0.3, "angle": 0.1},
    ]
    document = {"content": [_image(), *invalid, *valid_points, "abc"]}

    kept = _geometries(filters.filter_multimodal_content(document, _source()))

    assert len(kept) == 500
    assert all("point" in geometry for geometry in kept)


def test_grounding_drops_and_reindexes_geometry_when_images_are_filtered() -> None:
    low_quality = _image(
        0,
        vision_classifier_output={"classes_summarized": {"quality": "garbage"}},
    )
    document = {
        "content": [
            low_quality,
            {"box_2d": [100, 100, 200, 200], "image_idx": 0},
            _image(1),
            {
                "geometries": [
                    {"point_2d": [100, 100], "image_idx": 0},
                    {"point_2d": [200, 200], "image_idx": 1},
                ]
            },
            {"point": [0.3, 0.3], "image_idx": 1},
            "abc",
        ]
    }

    content = filters.filter_multimodal_content(document, _source())
    images = [item for item in content if filters.is_image_item(item)]
    kept = _geometries(content)

    assert [image["hash"] for image in images] == ["hash-1"]
    assert len(kept) == 2
    assert all(geometry["image_idx"] == 0 for geometry in kept)
    assert all(geometry.get("point_2d") != [100, 100] for geometry in kept)


def test_grounding_resolves_missing_image_idx_in_direct_and_container_geometry() -> None:
    document = {
        "content": [
            _image(0),
            {"box_2d": [100, 100, 200, 200]},
            "middle",
            _image(1),
            {"geometries": [{"point": [0.4, 0.4]}]},
            "abc",
        ]
    }

    kept = _geometries(filters.filter_multimodal_content(document, _source()))

    assert [geometry["image_idx"] for geometry in kept] == [0, 1]


@pytest.mark.parametrize(
    ("text_length", "expected"),
    [(2, False), (3, True), (40_000, True), (40_001, False)],
)
def test_grounding_character_density_boundaries(
    text_length: int, expected: bool
) -> None:
    document = {"content": [_image(), "x" * text_length]}
    content = filters.filter_multimodal_content(document, _source())

    assert (
        filters.is_valid_filtered_multimodal_content(document, content, _source())
        is expected
    )


def test_grounding_requires_a_surviving_image() -> None:
    document = {"content": ["text only"]}
    content = filters.filter_multimodal_content(document, _source())

    assert not filters.is_valid_filtered_multimodal_content(
        document, content, _source()
    )


def test_grounding_bypasses_v4_alignment_bucket_ocr_and_canary_checks() -> None:
    image = _image(
        doc_tar_offset={
            "tar_path": "s3://laion2b-en-images-by-jsonl-112824/chunk.tar",
            "offset": 0,
            "size": 50,
        },
        filter_args={"image_class_not_aligned": True},
    )
    text = f"abc {filters.BENCHMARK_CANARIES[0]}"
    document = {"content": [image, text], "overlap_score": 0.0}

    grounding_content = filters.filter_multimodal_content(document, _source())

    assert any(filters.is_image_item(item) for item in grounding_content)
    assert filters.is_valid_filtered_multimodal_content(
        document, grounding_content, _source()
    )
    assert filters.is_valid_text_chunks(
        [text], _source(), multimodal_content=grounding_content
    )

    v4_source = _source(FilterArgsName.MULTIMODAL_V4__LOBSTRAL)
    assert not any(
        filters.is_image_item(item)
        for item in filters.filter_multimodal_content(document, v4_source)
    )
    assert not filters.is_valid_text_chunks([text], v4_source)


def test_grounding_bbox_serialization_contributes_to_density_and_length() -> None:
    bbox = {
        "box_2d": [100, 200, 300, 400],
        "label": "Cat",
    }
    document = {"content": [_image(), bbox, ""]}
    content = filters.filter_multimodal_content(document, _source())

    assert filters._grounding_bbox_serialized_length(content) == len(
        '[BBOX]100,200,300,400, label="cat"[/BBOX]'
    )
    assert filters.is_valid_filtered_multimodal_content(document, content, _source())
    assert filters.is_valid_text_chunks(
        [], _source(), multimodal_content=content
    )


def test_no_max_length_profile_accepts_oversized_text_and_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(filters, "MAX_DOC_LENGTH", 5)
    default_source = _source(FilterArgsName.PRETRAIN_V4__LOBSTRAL)
    no_max_length_source = _source(
        FilterArgsName.PRETRAIN_V4__LOBSTRAL_NO_MAX_LENGTH
    )

    assert not filters.is_valid_text_chunks(["123456"], default_source)
    assert filters.is_valid_text_chunks(["123456"], no_max_length_source)
    assert json.loads(filters.filter_args_json(default_source))["length"][
        "max_chars"
    ] == 5
    assert json.loads(filters.filter_args_json(no_max_length_source))["length"][
        "max_chars"
    ] is None


@pytest.mark.parametrize(
    ("masking_strategy", "remove_laziness"),
    [
        (MaskingStrategy.MASK_FILE_SEPARATOR, True),
        (MaskingStrategy.MASK_FILE_SEPARATOR_NO_LAZINESS, False),
        (MaskingStrategy.MASK_LOCODIFF_MARKERS, True),
        (None, False),
    ],
)
def test_github_laziness_filter_follows_masking_strategy(
    masking_strategy: MaskingStrategy | None,
    remove_laziness: bool,
) -> None:
    source = _source(
        FilterArgsName.MULTIMODAL_V4__LOBSTRAL,
        masking_strategy=masking_strategy,
    )

    assert (
        filters.is_valid_text_chunks(["Write your code here"], source)
        is not remove_laziness
    )
    filter_args = json.loads(filters.filter_args_json(source))
    assert filter_args["github"]["remove_laziness"] is remove_laziness


def test_pds_quarantine_is_enabled_by_source_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quarantined = _image(
        doc_tar_offset={
            "tar_path": "s3://images/quarantined.tar",
            "offset": 10,
            "size": 20,
        },
        is_safe=False,
    )
    monkeypatch.setattr(
        filters,
        "_load_pds_key_set",
        lambda pds_filter: (
            frozenset({("images/quarantined.tar", 10, 20)}),
            frozenset(),
            {},
        ),
    )
    document = {"content": [quarantined, "abc"]}

    assert filters.multimodal_document_has_pds_quarantined_image(
        document, PdsFilter.NVIDIA_PDS_QUARANTINE_V2
    )
    assert not filters.multimodal_document_has_pds_quarantined_image(document, None)


@pytest.mark.parametrize(
    ("tar_offset", "expected"),
    [
        (("s3://images/quarantined.tar", 10, 20, False), True),
        (("images/quarantined.tar", -502, 1024, True), True),
        (("images/other.tar", 10, 20, False), False),
        (None, False),
    ],
)
def test_stored_image_metadata_matches_pds_quarantine(
    monkeypatch: pytest.MonkeyPatch,
    tar_offset: tuple[str, int, int, bool] | None,
    expected: bool,
) -> None:
    image = SimpleNamespace(tar_offset=None)
    if tar_offset is not None:
        image.tar_offset = SimpleNamespace(
            tar_path=tar_offset[0],
            offset=tar_offset[1],
            size=tar_offset[2],
            new_tar_format=tar_offset[3],
        )
    monkeypatch.setattr(
        filters,
        "_load_pds_key_set",
        lambda pds_filter: (
            frozenset({("images/quarantined.tar", 10, 20)}),
            frozenset({("images/quarantined.tar", -502, 1024)}),
            {},
        ),
    )

    assert (
        filters.image_index_entry_is_pds_quarantined(
            image, PdsFilter.NVIDIA_PDS_QUARANTINE_V3
        )
        is expected
    )


def test_retransfer_quarantine_path_is_normalized() -> None:
    path = filters._normalize_quarantine_tar_path(
        "nv-00-10603-sandia/image_buckets_retransfer/images/file.tar"
    )
    assert path == "images/file.tar"


@pytest.mark.parametrize(
    "doc_tar_offset",
    [None, {"tar_path": "s3://images/incomplete.tar", "offset": 10}],
)
def test_pds_quarantine_warns_and_skips_missing_tar_annotations(
    doc_tar_offset: dict[str, Any] | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    document = {"content": [_image(doc_tar_offset=doc_tar_offset), "abc"]}

    assert not filters.multimodal_document_has_pds_quarantined_image(
        document, PdsFilter.NVIDIA_PDS_QUARANTINE_V2
    )
    assert "image cannot be checked against the PDS manifests" in caplog.text


def test_grounding_filter_metadata_records_effective_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quarantine = {
        "manifests": [{"path": "/manifest", "sha256": "abc123"}],
        "match_fields": ["tar_path", "offset", "size"],
    }
    monkeypatch.setattr(filters, "pds_filter_args", lambda pds_filter: quarantine)

    args = json.loads(
        filters.filter_args_json(
            _source(
                rare_token_filter=RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1,
                pds_filter=PdsFilter.NVIDIA_PDS_QUARANTINE_V2,
            )
        )
    )

    assert args["contamination"] is None
    assert args["rare_token_filter"] == {"name": "rare_tokens_n4_v6_262k_v1"}
    assert args["pds_filter"] == {
        "name": "nvidia_pds_quarantine_v2",
        **quarantine,
    }
    assert args["image"]["min_images"] == 1
    assert args["image"]["min_chars_per_image"] == 3
    assert args["image"]["max_chars_per_image"] == 40_000
    assert args["image"]["max_consecutive_images"] == 100_000
    assert args["image"]["geometry_filter_args"] == {
        "avoid_duplicates": False,
        "filter_out_of_bounds": True,
        "max_area": 1.0,
        "max_geom_per_image": 500,
        "min_area": 0.0,
        "skip_no_label": False,
    }
