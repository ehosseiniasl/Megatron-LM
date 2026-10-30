"""Tests for serialization compatibility between Mistral and Megatron fork.

Verifies that TransformImageMetadata (with ImageBoundingBox and
l2_reconstruction_score) serialized by Mistral's ImageIndexEntry can be
deserialized by the fork, and vice versa.

The Mistral side uses Pydantic-based ImageBoundingBox serialized via
model_dump_json(); the fork uses a plain dataclass with to_mistral_json().
Both sides use msgspec for the outer ImageIndexEntry envelope.
"""

import json
from pathlib import Path

import msgspec
import numpy as np
import pytest

from mistral_fork.mistral.data.pretokenized.image_transform_metadata import (
    ImageBoundingBox,
    TransformImageMetadata,
)
from mistral_fork.mistral.data.pretokenized.metadata import (
    DocumentIndexEntry,
    ImageIndexEntry,
    ScatteredMetadataReader,
    TextIndexEntry,
    _mask_size,
    _MaskCursor,
    _MetadataReader,
    _split_mask,
    _TarOffset,
    dump_metadata,
)


# ---------------------------------------------------------------------------
# ImageBoundingBox serialization
# ---------------------------------------------------------------------------


class TestImageBoundingBoxSerialization:
    """Test ImageBoundingBox round-trips through Mistral's JSON format."""

    def test_to_mistral_dict(self) -> None:
        bbox = ImageBoundingBox(
            top_left_x=0.1, top_left_y=0.2, bottom_right_x=0.8, bottom_right_y=0.9
        )
        d = bbox.to_mistral_dict()
        assert d["top_left"] == [0.1, 0.2]
        assert d["bottom_right"] == [0.8, 0.9]
        assert d["label"] is None
        assert d["is_masked"] is False

    def test_to_mistral_json_is_valid_json(self) -> None:
        bbox = ImageBoundingBox(
            top_left_x=0.0, top_left_y=0.0, bottom_right_x=1.0, bottom_right_y=1.0
        )
        s = bbox.to_mistral_json()
        parsed = json.loads(s)
        assert parsed["top_left"] == [0.0, 0.0]
        assert parsed["bottom_right"] == [1.0, 1.0]

    def test_from_mistral_dict(self) -> None:
        d = {
            "top_left": [0.1, 0.2],
            "bottom_right": [0.8, 0.9],
            "label": None,
            "caption": None,
            "image_idx": None,
            "meta": None,
            "is_masked": False,
        }
        bbox = ImageBoundingBox.from_mistral_dict(d)
        assert bbox.top_left_x == pytest.approx(0.1)
        assert bbox.top_left_y == pytest.approx(0.2)
        assert bbox.bottom_right_x == pytest.approx(0.8)
        assert bbox.bottom_right_y == pytest.approx(0.9)

    def test_from_mistral_dict_normalizes_old_absolute_coordinates(self) -> None:
        """Old Mistral metadata stores pixel coords plus page dimensions."""
        d = {
            "top_left": [10.0, 20.0],
            "bottom_right": [90.0, 80.0],
            "page_height": 100.0,
            "page_width": 100.0,
            "label": None,
            "caption": None,
            "image_idx": None,
            "meta": None,
            "is_masked": False,
        }
        bbox = ImageBoundingBox.from_mistral_dict(d)
        assert bbox.top_left_x == pytest.approx(0.1)
        assert bbox.top_left_y == pytest.approx(0.2)
        assert bbox.bottom_right_x == pytest.approx(0.9)
        assert bbox.bottom_right_y == pytest.approx(0.8)

    def test_round_trip_via_json(self) -> None:
        """Serialize to Mistral JSON, then deserialize back."""
        original = ImageBoundingBox(
            top_left_x=0.123,
            top_left_y=0.456,
            bottom_right_x=0.789,
            bottom_right_y=0.012,
        )
        json_str = original.to_mistral_json()
        restored = ImageBoundingBox.from_mistral_dict(json.loads(json_str))
        assert original == restored

    def test_from_dict_auto_detects_format(self) -> None:
        """from_dict should auto-detect Mistral format vs flat format."""
        # Mistral format
        d_mistral = {"top_left": [0.1, 0.2], "bottom_right": [0.8, 0.9]}
        bbox1 = ImageBoundingBox.from_dict(d_mistral)
        assert bbox1.top_left_x == pytest.approx(0.1)

        # Flat format
        d_flat = {
            "top_left_x": 0.1,
            "top_left_y": 0.2,
            "bottom_right_x": 0.8,
            "bottom_right_y": 0.9,
        }
        bbox2 = ImageBoundingBox.from_dict(d_flat)
        assert bbox2 == bbox1

    def test_from_mistral_dict_ignores_extra_fields(self) -> None:
        """Mistral's ImageBoundingBox has extra GroundedGeometry fields."""
        d = {
            "top_left": [0.1, 0.2],
            "bottom_right": [0.8, 0.9],
            "label": "some_label",
            "caption": "a caption",
            "image_idx": 42,
            "meta": {"key": "value"},
            "is_masked": True,
        }
        bbox = ImageBoundingBox.from_mistral_dict(d)
        assert bbox.top_left_x == pytest.approx(0.1)
        assert bbox.bottom_right_y == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# ImageIndexEntry enc/dec hooks
# ---------------------------------------------------------------------------


class TestImageIndexEntryHooks:
    """Test that ImageIndexEntry _enc_hook / _dec_hook handle ImageBoundingBox."""

    def _make_entry(self, bbox: ImageBoundingBox | None = None) -> ImageIndexEntry:
        meta = TransformImageMetadata(
            width=640,
            height=480,
            l2_reconstruction_score={"image_min_size=224,l2_resize_factor=2": 0.005},
        )
        if bbox is not None:
            meta.img_bbox_for_trim = bbox
        return ImageIndexEntry(
            img_id_in_doc=0,
            tar_offset=_TarOffset(offset=100, size=5000, tar_path="/data/images.tar"),
            pos_in_text=42,
            metadata=meta,
            size=64,
            seed=(1, 2, 3),
        )

    def test_enc_hook_produces_json_string(self) -> None:
        bbox = ImageBoundingBox(
            top_left_x=0.1, top_left_y=0.2, bottom_right_x=0.8, bottom_right_y=0.9
        )
        result = ImageIndexEntry._enc_hook(bbox)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["top_left"] == [0.1, 0.2]
        assert parsed["bottom_right"] == [0.8, 0.9]

    def test_dec_hook_from_string(self) -> None:
        json_str = json.dumps(
            {
                "top_left": [0.1, 0.2],
                "bottom_right": [0.8, 0.9],
                "label": None,
                "caption": None,
                "image_idx": None,
                "meta": None,
                "is_masked": False,
            }
        )
        result = ImageIndexEntry._dec_hook(ImageBoundingBox, json_str)
        assert isinstance(result, ImageBoundingBox)
        assert result.top_left_x == pytest.approx(0.1)

    def test_dec_hook_from_dict(self) -> None:
        d = {"top_left": [0.1, 0.2], "bottom_right": [0.8, 0.9]}
        result = ImageIndexEntry._dec_hook(ImageBoundingBox, d)
        assert isinstance(result, ImageBoundingBox)
        assert result.top_left_x == pytest.approx(0.1)

    def test_dec_hook_from_old_absolute_bbox_dict(self) -> None:
        d = {
            "top_left": [10.0, 20.0],
            "bottom_right": [90.0, 80.0],
            "page_height": 100.0,
            "page_width": 100.0,
        }
        result = ImageIndexEntry._dec_hook(ImageBoundingBox, d)
        assert isinstance(result, ImageBoundingBox)
        assert result.top_left_x == pytest.approx(0.1)
        assert result.bottom_right_y == pytest.approx(0.8)

    def test_round_trip_without_bbox(self) -> None:
        """Entries without bbox should round-trip through to_bytes/from_bytes."""
        entry = self._make_entry(bbox=None)
        data = ImageIndexEntry.to_bytes(entry)
        restored = ImageIndexEntry.from_bytes(data)
        assert restored.img_id_in_doc == entry.img_id_in_doc
        assert restored.pos_in_text == entry.pos_in_text
        assert restored.metadata.width == 640
        assert restored.metadata.height == 480
        assert restored.metadata.l2_reconstruction_score == {
            "image_min_size=224,l2_resize_factor=2": 0.005
        }

    def test_round_trip_with_bbox(self) -> None:
        """Entries with bbox should round-trip through to_bytes/from_bytes."""
        bbox = ImageBoundingBox(
            top_left_x=0.1, top_left_y=0.2, bottom_right_x=0.8, bottom_right_y=0.9
        )
        entry = self._make_entry(bbox=bbox)
        data = ImageIndexEntry.to_bytes(entry)
        restored = ImageIndexEntry.from_bytes(data)
        assert restored.metadata._img_bbox_for_trim is not None
        assert restored.metadata._img_bbox_for_trim.top_left_x == pytest.approx(0.1)
        assert restored.metadata._img_bbox_for_trim.bottom_right_y == pytest.approx(0.9)

    def test_to_bytes_writes_mistral_bbox_json_string(self) -> None:
        bbox = ImageBoundingBox(
            top_left_x=0.1, top_left_y=0.2, bottom_right_x=0.8, bottom_right_y=0.9
        )
        entry = self._make_entry(bbox=bbox)
        data = ImageIndexEntry.to_bytes(entry)
        raw = msgspec.json.decode(data, type=dict)

        assert "content_modified" not in raw["metadata"]
        assert "max_image_size" not in raw["metadata"]
        encoded_bbox = raw["metadata"]["_img_bbox_for_trim"]
        assert isinstance(encoded_bbox, str)
        parsed_bbox = json.loads(encoded_bbox)
        assert parsed_bbox["top_left"] == [0.1, 0.2]
        assert parsed_bbox["bottom_right"] == [0.8, 0.9]
        assert "top_left_x" not in parsed_bbox
        assert "bottom_right_y" not in parsed_bbox

    def test_document_to_bytes_writes_mistral_bbox_json_string(self) -> None:
        bbox = ImageBoundingBox(
            top_left_x=0.1, top_left_y=0.2, bottom_right_x=0.8, bottom_right_y=0.9
        )
        image_entry = self._make_entry(bbox=bbox)
        document_entry = DocumentIndexEntry(
            doc_id=0,
            text=TextIndexEntry(offset=0, size=128),
            imgs=[image_entry],
            size=128 + image_entry.size,
        )

        data = DocumentIndexEntry.to_bytes([document_entry])
        raw = msgspec.json.decode(data, type=list[dict])

        metadata = raw[0]["imgs"][0]["metadata"]
        assert "content_modified" not in metadata
        assert "max_image_size" not in metadata
        encoded_bbox = raw[0]["imgs"][0]["metadata"]["_img_bbox_for_trim"]
        assert isinstance(encoded_bbox, str)
        parsed_bbox = json.loads(encoded_bbox)
        assert parsed_bbox["top_left"] == [0.1, 0.2]
        assert parsed_bbox["bottom_right"] == [0.8, 0.9]
        assert "top_left_x" not in parsed_bbox
        assert "bottom_right_y" not in parsed_bbox

    def test_round_trip_list(self) -> None:
        """Lists of entries should round-trip."""
        bbox = ImageBoundingBox(
            top_left_x=0.5, top_left_y=0.5, bottom_right_x=1.0, bottom_right_y=1.0
        )
        entries = [self._make_entry(bbox=None), self._make_entry(bbox=bbox)]
        data = ImageIndexEntry.to_bytes(entries)
        restored = ImageIndexEntry.list_from_bytes(data)
        assert len(restored) == 2
        assert restored[0].metadata._img_bbox_for_trim is None
        assert restored[1].metadata._img_bbox_for_trim is not None
        assert restored[1].metadata._img_bbox_for_trim.top_left_x == pytest.approx(0.5)

    def test_deserialize_mistral_format(self) -> None:
        """Simulate what Mistral's ImageIndexEntry.to_bytes() would produce.

        Mistral serializes ImageBoundingBox via model_dump_json() which includes
        all GroundedGeometry fields (label, caption, image_idx, meta, is_masked).
        """
        # Build a raw dict mimicking Mistral's JSON output
        mistral_bbox_json = json.dumps(
            {
                "top_left": [0.15, 0.25],
                "bottom_right": [0.85, 0.95],
                "label": None,
                "caption": None,
                "image_idx": None,
                "meta": None,
                "is_masked": False,
            }
        )
        raw = {
            "img_id_in_doc": 3,
            "tar_path": "/data/test.tar",
            "tar_offset": {"offset": 200, "size": 3000},
            "pos_in_text": 10,
            "metadata": {
                "width": 800,
                "height": 600,
                "l2_reconstruction_score": {
                    "image_min_size=224,l2_resize_factor=2": 0.007
                },
                "_img_bbox_for_trim": mistral_bbox_json,
                "img_bbox_for_trim_is_computed": True,
                "content_modified": False,
                "max_image_size": None,
            },
            "size": 128,
            "seed": [5, 6, 7],
        }
        data = msgspec.json.encode(raw)
        entry = ImageIndexEntry.from_bytes(data)
        assert entry.img_id_in_doc == 3
        assert entry.metadata.width == 800
        assert entry.metadata.height == 600
        assert entry.metadata._img_bbox_for_trim is not None
        assert entry.metadata._img_bbox_for_trim.top_left_x == pytest.approx(0.15)
        assert entry.metadata._img_bbox_for_trim.bottom_right_y == pytest.approx(0.95)

    def test_deserialize_none_bbox_sentinel(self) -> None:
        raw = {
            "img_id_in_doc": 0,
            "tar_offset": {"offset": 0, "size": 100, "tar_path": "/test.tar"},
            "pos_in_text": 0,
            "metadata": {
                "width": 640,
                "height": 480,
                "_img_bbox_for_trim": "none_bbox",
                "img_bbox_for_trim_is_computed": True,
            },
        }
        entry = ImageIndexEntry.from_bytes(msgspec.json.encode(raw))
        assert entry.metadata.img_bbox_for_trim_is_computed
        assert entry.metadata.img_bbox_for_trim is None

    def test_deserialize_legacy_format(self) -> None:
        """Legacy entries use metadata_before_transform + maybe_transform_output."""
        raw = {
            "img_id_in_doc": 0,
            "tar_path": "/data/legacy.tar",
            "tar_offset": {"offset": 0, "size": 1000},
            "pos_in_text": 0,
            "metadata_before_transform": {
                "width": 320,
                "height": 240,
            },
            "maybe_transform_output": {
                "seed": [1, 2],
                "n_toks": 32,
            },
        }
        data = msgspec.json.encode(raw)
        entry = ImageIndexEntry.from_bytes(data)
        assert entry.metadata.width == 320
        assert entry.seed == (1, 2)
        assert entry.size == 32


# ---------------------------------------------------------------------------
# TransformImageMetadata
# ---------------------------------------------------------------------------


class TestTransformImageMetadata:
    def test_img_bbox_setter_accepts_dict(self) -> None:
        meta = TransformImageMetadata(width=100, height=100)
        meta.img_bbox_for_trim = {
            "top_left_x": 0.1,
            "top_left_y": 0.2,
            "bottom_right_x": 0.8,
            "bottom_right_y": 0.9,
        }
        assert meta.img_bbox_for_trim is not None
        assert isinstance(meta.img_bbox_for_trim, ImageBoundingBox)
        assert meta.img_bbox_for_trim.top_left_x == pytest.approx(0.1)

    def test_img_bbox_setter_accepts_bbox(self) -> None:
        meta = TransformImageMetadata(width=100, height=100)
        bbox = ImageBoundingBox(
            top_left_x=0.1, top_left_y=0.2, bottom_right_x=0.8, bottom_right_y=0.9
        )
        meta.img_bbox_for_trim = bbox
        assert meta.img_bbox_for_trim is bbox

    def test_img_bbox_setter_accepts_none_bbox_sentinel(self) -> None:
        meta = TransformImageMetadata(width=100, height=100)
        meta.img_bbox_for_trim = "none_bbox"
        assert meta.img_bbox_for_trim_is_computed
        assert meta.img_bbox_for_trim is None

    def test_img_bbox_getter_raises_if_not_computed(self) -> None:
        meta = TransformImageMetadata(width=100, height=100)
        with pytest.raises(ValueError, match="not computed"):
            _ = meta.img_bbox_for_trim

    def test_l2_reconstruction_score(self) -> None:
        meta = TransformImageMetadata(
            width=1000,
            height=800,
            l2_reconstruction_score={"image_min_size=224,l2_resize_factor=2": 0.005},
        )
        assert meta.l2_reconstruction_score[
            "image_min_size=224,l2_resize_factor=2"
        ] == pytest.approx(0.005)

    def test_msgspec_round_trip(self) -> None:
        """TransformImageMetadata should survive msgspec encode/decode as part of ImageIndexEntry."""
        meta = TransformImageMetadata(
            width=640,
            height=480,
            l2_reconstruction_score={"key1": 0.1, "key2": 0.9},
        )
        entry = ImageIndexEntry(
            img_id_in_doc=0,
            tar_offset=_TarOffset(offset=0, size=100, tar_path="/test.tar"),
            pos_in_text=0,
            metadata=meta,
        )
        data = ImageIndexEntry.to_bytes(entry)
        restored = ImageIndexEntry.from_bytes(data)
        assert restored.metadata.width == 640
        assert restored.metadata.height == 480
        assert restored.metadata.l2_reconstruction_score == {"key1": 0.1, "key2": 0.9}

    def test_deserialize_legacy_megatron_transform_context_fields(self) -> None:
        raw = {
            "img_id_in_doc": 0,
            "tar_offset": {"offset": 0, "size": 100, "tar_path": "/test.tar"},
            "pos_in_text": 0,
            "metadata": {
                "width": 640,
                "height": 480,
                "l2_reconstruction_score": {"key1": 0.1},
                "content_modified": True,
                "max_image_size": 512,
            },
        }
        restored = ImageIndexEntry.from_bytes(msgspec.json.encode(raw))
        assert restored.metadata.width == 640
        assert restored.metadata.height == 480
        assert restored.metadata.l2_reconstruction_score == {"key1": 0.1}
        assert not hasattr(restored.metadata, "content_modified")
        assert not hasattr(restored.metadata, "max_image_size")


def _metadata_entry(img_id: int) -> ImageIndexEntry:
    return ImageIndexEntry(
        img_id_in_doc=img_id,
        tar_offset=_TarOffset(offset=img_id, size=100 + img_id, tar_path="/tmp/a.tar"),
        pos_in_text=img_id,
        metadata=TransformImageMetadata(width=10 + img_id, height=20 + img_id),
        size=4,
    )


def test_metadata_reader_persistent_index_and_raw_bytes(tmp_path: Path) -> None:
    metadata_path = tmp_path / "chunks.metadata"
    original_images = [[_metadata_entry(i)] for i in range(5)]
    dump_metadata(original_images, metadata_path)

    index_path = metadata_path.with_name(metadata_path.name + ".idx")
    assert index_path.exists()

    reader = _MetadataReader(metadata_path)
    try:
        assert len(reader) == len(original_images)
        assert ImageIndexEntry.list_from_bytes(reader.get_raw_bytes(3)) == [
            _metadata_entry(3)
        ]
    finally:
        reader.close()

    index_path.write_bytes(b"bad index")
    rebuilt_reader = _MetadataReader(metadata_path)
    try:
        assert rebuilt_reader[4] == [_metadata_entry(4)]
    finally:
        rebuilt_reader.close()


def test_metadata_reader_supports_empty_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "empty.metadata"
    dump_metadata([], metadata_path)

    reader = _MetadataReader(metadata_path)
    try:
        assert len(reader) == 0
        with pytest.raises(IndexError, match="Line index out of range"):
            reader.get_raw_bytes(0)
    finally:
        reader.close()


def test_scattered_metadata_reader_roundtrip(tmp_path: Path) -> None:
    metadata_path = tmp_path / "chunks.metadata"
    original_images = [[_metadata_entry(i)] for i in range(20)]
    dump_metadata(original_images, metadata_path)

    scatter_dir = tmp_path / "scatter"
    scatter_dir.mkdir()
    doc_ids = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])
    rng = np.random.default_rng(7)
    rng.shuffle(doc_ids)

    scattered = ScatteredMetadataReader(
        metadata_path=metadata_path,
        doc_ids=doc_ids,
        tmp_dir=scatter_dir,
        max_bucket_bytes=128,
    )
    try:
        assert scattered._n_buckets > 1
        assert len(scattered) == len(doc_ids)
        for pos, doc_id in enumerate(doc_ids):
            assert scattered[pos] == [_metadata_entry(int(doc_id))]
        with pytest.raises(AssertionError, match="out of range"):
            _ = scattered[len(doc_ids)]
    finally:
        scattered.cleanup()

    assert not list(scatter_dir.glob("bucket_*.bin"))


def test_document_index_truncate_trailing_masked() -> None:
    doc = DocumentIndexEntry(
        doc_id=1,
        text=TextIndexEntry(offset=7, size=10),
        imgs=[],
        size=10,
        mask=([4, 6], [1, 0]),
    )

    truncated, removed_toks, removed_imgs = doc.truncate_trailing_masked()

    assert truncated is not None
    assert removed_toks == 6
    assert removed_imgs == 0
    assert truncated.text == TextIndexEntry(offset=7, size=4)
    assert truncated.size == 4
    assert truncated.mask == ([4], [1])


@pytest.mark.parametrize(
    ("mask", "splits"),
    [
        (([5, 3, 7], [0, 1, 0]), [(4, 1), (6, 1), (7, 0)]),
        (([3, 4, 5], [0, 1, 0]), [(3, 0), (4, 1), (6, 0)]),
        (([1, 2, 1, 8], [1, 0, 1, 0]), [(1, 1), (5, 1), (8, 0)]),
    ],
)
def test_mask_cursor_matches_repeated_mask_splits(
    mask: tuple[list[int], list[int]], splits: list[tuple[int, int]]
) -> None:
    cursor = _MaskCursor(mask)
    remaining_mask = mask

    for size, overlap in splits:
        expected, remaining_mask = _split_mask(remaining_mask, size, overlap)

        assert cursor.take(size, overlap) == expected
        assert cursor.remaining == _mask_size(remaining_mask)

    assert cursor.remaining == 0
