import io
from unittest.mock import MagicMock, patch

from PIL import Image

from mistral_fork.mistral.data.pretokenized.image_transform_metadata import TransformImageMetadata
from mistral_fork.mistral.data.pretokenized.image_utils import (
    _MAX_GAP_BYTES,
    _ImageEntry,
    _split_into_contiguous_spans,
    load_contents_grouped,
)
from mistral_fork.mistral.data.pretokenized.metadata import ImageIndexEntry, _TarOffset
from mistral_fork.shared.common_utils import TarOffset

_MOCK_CLUSTERIFY = "mistral_fork.mistral.data.pretokenized.metadata.clusterify_path"
_MOCK_LOAD_BYTES = (
    "mistral_fork.mistral.data.pretokenized.image_utils.load_bytes"
)
_MOCK_GET_TAR_BYTES = "mistral_fork.mistral.data.pretokenized.metadata.get_tar_bytes_helper"


def _make_entry(tar_path: str, offset: int, size: int) -> ImageIndexEntry:
    return ImageIndexEntry(
        img_id_in_doc=0,
        tar_offset=_TarOffset(offset=offset, size=size, tar_path=tar_path),
        pos_in_text=0,
        metadata=TransformImageMetadata(width=10, height=10),
        size=16,
    )


def _make_1x1_png(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=color).save(buf, format="PNG")
    return buf.getvalue()


# -- _split_into_contiguous_spans --


def test_split_single_entry() -> None:
    entry = _ImageEntry(0, MagicMock(), TarOffset(offset=0, size=100))
    spans = _split_into_contiguous_spans([entry])
    assert len(spans) == 1
    assert spans[0] == [entry]


def test_split_exact_threshold_stays_together() -> None:
    entries = [
        _ImageEntry(0, MagicMock(), TarOffset(offset=0, size=100)),
        _ImageEntry(1, MagicMock(), TarOffset(offset=100 + _MAX_GAP_BYTES, size=100)),
    ]
    spans = _split_into_contiguous_spans(entries)
    assert len(spans) == 1


def test_split_over_threshold_splits() -> None:
    entries = [
        _ImageEntry(0, MagicMock(), TarOffset(offset=0, size=100)),
        _ImageEntry(
            1, MagicMock(), TarOffset(offset=100 + _MAX_GAP_BYTES + 1, size=100)
        ),
    ]
    spans = _split_into_contiguous_spans(entries)
    assert len(spans) == 2


def test_split_duplicate_offsets_stay_together() -> None:
    """Duplicate entries (same offset) should be kept in the same span."""
    entries = [
        _ImageEntry(0, MagicMock(), TarOffset(offset=100, size=50)),
        _ImageEntry(1, MagicMock(), TarOffset(offset=100, size=50)),
        _ImageEntry(2, MagicMock(), TarOffset(offset=100, size=50)),
    ]
    spans = _split_into_contiguous_spans(entries)
    assert len(spans) == 1
    assert len(spans[0]) == 3


# -- load_contents_grouped --


def test_load_contents_grouped_empty() -> None:
    assert load_contents_grouped([]) == []


def test_load_contents_grouped_single_image() -> None:
    png = _make_1x1_png()
    entry = _make_entry("s3://bucket/test.tar", offset=0, size=len(png))

    with patch(_MOCK_GET_TAR_BYTES, return_value=png) as mock_tar:
        results = load_contents_grouped([entry])

    assert len(results) == 1
    assert isinstance(results[0], Image.Image)
    mock_tar.assert_called_once()


def test_load_contents_grouped_different_tars() -> None:
    png = _make_1x1_png()
    e1 = _make_entry("s3://bucket/a.tar", offset=0, size=len(png))
    e2 = _make_entry("s3://bucket/b.tar", offset=0, size=len(png))

    with patch(_MOCK_GET_TAR_BYTES, return_value=png) as mock_tar:
        results = load_contents_grouped([e1, e2])

    assert len(results) == 2
    assert all(isinstance(r, Image.Image) for r in results)
    assert mock_tar.call_count == 2


def test_load_contents_grouped_same_tar_batched() -> None:
    png = _make_1x1_png()
    size = len(png)

    e1 = _make_entry("s3://bucket/test.tar", offset=0, size=size)
    e2 = _make_entry("s3://bucket/test.tar", offset=size + 50, size=size)
    spanning = png + b"\x00" * 50 + png

    with (
        patch(_MOCK_LOAD_BYTES, return_value=spanning) as mock_load,
        patch(_MOCK_CLUSTERIFY, side_effect=lambda p: p),
    ):
        results = load_contents_grouped([e1, e2])

    assert len(results) == 2
    assert all(isinstance(r, Image.Image) for r in results)
    mock_load.assert_called_once()


def test_load_contents_grouped_preserves_order() -> None:
    red_png = _make_1x1_png(color=(255, 0, 0))
    blue_png = _make_1x1_png(color=(0, 0, 255))
    assert len(red_png) == len(blue_png)
    size = len(red_png)

    # e1 has higher offset but comes first in input — should get blue
    # e2 has lower offset — should get red
    e1 = _make_entry("s3://bucket/test.tar", offset=size + 10, size=size)
    e2 = _make_entry("s3://bucket/test.tar", offset=0, size=size)
    spanning = red_png + b"\x00" * 10 + blue_png

    with (
        patch(_MOCK_LOAD_BYTES, return_value=spanning) as mock_load,
        patch(_MOCK_CLUSTERIFY, side_effect=lambda p: p),
    ):
        results = load_contents_grouped([e1, e2])

    assert len(results) == 2
    assert results[0].getpixel((0, 0)) == (0, 0, 255), (
        "e1 (higher offset) should be blue"
    )
    assert results[1].getpixel((0, 0)) == (255, 0, 0), "e2 (lower offset) should be red"
    mock_load.assert_called_once()


def test_load_contents_grouped_mixed_tars() -> None:
    """Test loading images from multiple tars, with batching within each tar."""
    png = _make_1x1_png()
    size = len(png)

    e1 = _make_entry("s3://bucket/a.tar", offset=0, size=size)
    e2 = _make_entry("s3://bucket/a.tar", offset=size + 10, size=size)
    e3 = _make_entry("s3://bucket/b.tar", offset=0, size=size)
    spanning = png + b"\x00" * 10 + png

    with (
        patch(_MOCK_LOAD_BYTES, return_value=spanning) as mock_load,
        patch(_MOCK_GET_TAR_BYTES, return_value=png) as mock_tar,
        patch(_MOCK_CLUSTERIFY, side_effect=lambda p: p),
    ):
        results = load_contents_grouped([e1, e2, e3])

    assert len(results) == 3
    mock_load.assert_called_once()  # a.tar: 2 images batched into 1 read
    mock_tar.assert_called_once()  # b.tar: 1 image, 1 individual read
    assert all(isinstance(r, Image.Image) for r in results)


def test_load_contents_grouped_duplicate_offsets() -> None:
    """Duplicate image entries (same tar + offset) should return the same image."""
    png = _make_1x1_png(color=(0, 255, 0))
    size = len(png)

    # Three entries pointing to the same offset
    e1 = _make_entry("s3://bucket/test.tar", offset=0, size=size)
    e2 = _make_entry("s3://bucket/test.tar", offset=0, size=size)
    e3 = _make_entry("s3://bucket/test.tar", offset=0, size=size)

    with (
        patch(_MOCK_LOAD_BYTES, return_value=png) as mock_load,
        patch(_MOCK_CLUSTERIFY, side_effect=lambda p: p),
    ):
        results = load_contents_grouped([e1, e2, e3])

    assert len(results) == 3
    # All three should be the same image (green)
    for r in results:
        assert isinstance(r, Image.Image)
        assert r.getpixel((0, 0)) == (0, 255, 0)
    # Only one read should have been issued
    mock_load.assert_called_once()
    # All results should be the exact same object (reused)
    assert results[0] is results[1] is results[2]
