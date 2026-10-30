"""Image loading utilities for pretokenized data.

This module provides functions for efficiently loading images from tar archives,
with support for batched reads to reduce I/O calls.
"""

import io
import logging
import typing
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, UnidentifiedImageError

from mistral_fork.shared.common_utils import TarOffset
from mistral_fork.shared.image_utils import convert_to_rgb
from mistral_fork.shared.tartar import (
    TAR_HEADER_SIZE,
    _read_bytes,
    get_tar_bytes_helper,
    load_bytes,
)

if TYPE_CHECKING:
    from mistral_fork.mistral.data.pretokenized.metadata import ImageIndexEntry

# Maximum gap between consecutive images in the same tar to still batch them
# into a single ranged read. Observed gaps for contiguous images are 4-12KB
# (tar headers + alignment padding + metadata entries). 512KB is generous.
_MAX_GAP_BYTES = 512 * 1024

logger = logging.getLogger(__name__)


class _ImageEntry(typing.NamedTuple):
    idx: int
    image: "ImageIndexEntry"
    tar_offset: TarOffset


def load_contents_grouped(
    images: list["ImageIndexEntry"], max_group_workers: int = 1
) -> list[Image.Image]:
    """Load images, batching reads for images sharing the same tar into a single I/O call.

    When multiple images share a tar file and are contiguous (or nearly so),
    a single ranged read replaces N individual reads.
    """
    if not images:
        return []
    if max_group_workers < 1:
        raise ValueError(f"max_group_workers must be positive, got {max_group_workers}")

    results: list[Image.Image | None] = [None] * len(images)

    # Group by resolved tar_path
    groups: dict[str, list[tuple[int, "ImageIndexEntry"]]] = defaultdict(list)
    for idx, img in enumerate(images):
        groups[img.tar_path].append((idx, img))

    def load_group(item: tuple[str, list[tuple[int, "ImageIndexEntry"]]]) -> None:
        tar_path, group = item
        if len(group) == 1:
            idx, img = group[0]
            results[idx] = img.load_content()
        else:
            _load_same_tar_images(tar_path, group, results)

    if max_group_workers == 1 or len(groups) == 1:
        for item in groups.items():
            load_group(item)
    else:
        with ThreadPoolExecutor(max_workers=min(max_group_workers, len(groups))) as pool:
            list(pool.map(load_group, groups.items()))

    loaded: list[Image.Image] = []
    for r in results:
        assert r is not None, "some images were not loaded"
        loaded.append(r)
    return loaded


def _load_same_tar_images(
    tar_path: str,
    group: list[tuple[int, "ImageIndexEntry"]],
    results: list[Image.Image | None],
) -> None:
    """Batch-read contiguous images from one tar, splitting on large gaps."""
    entries = [
        _ImageEntry(idx, img, img.tar_offset.to_pydantic()) for idx, img in group
    ]

    for span in _split_into_contiguous_spans(entries):
        if len(span) == 1:
            results[span[0].idx] = span[0].image.load_content()
            continue

        _batch_read_span(tar_path, span, results)


def _split_into_contiguous_spans(
    entries: list[_ImageEntry],
) -> list[list[_ImageEntry]]:
    """Sort entries by offset and split into spans where consecutive gaps <= _MAX_GAP_BYTES.

    Duplicate entries (same offset) are kept together in the same span.
    """
    sorted_entries = sorted(entries, key=lambda e: e.tar_offset.offset)
    spans: list[list[_ImageEntry]] = [[sorted_entries[0]]]
    for entry in sorted_entries[1:]:
        prev = spans[-1][-1].tar_offset
        cur = entry.tar_offset
        # Duplicate entries have offset == prev.offset, so gap will be negative.
        # Treat duplicates as gap=0 (they'll share the same loaded image).
        gap = cur.offset - (prev.offset + prev.size)
        is_duplicate = cur.offset == prev.offset and cur.size == prev.size
        if is_duplicate:
            gap = 0
        assert gap >= 0, f"Overlapping entries: {prev=}, {cur=}"
        if gap > _MAX_GAP_BYTES:
            spans.append([entry])
        else:
            spans[-1].append(entry)
    return spans


def _decode_image(img_bytes: bytes) -> Image.Image:
    with io.BytesIO(img_bytes) as f:
        pil_img: Image.Image = Image.open(f)
        pil_img.load()
    pil_img = convert_to_rgb(pil_img)
    assert pil_img.mode == "RGB"
    return pil_img


@lru_cache(maxsize=1)
def _load_image_not_found() -> Image.Image:
    return _decode_image(Path(__file__).with_name("image_not_found.png").read_bytes())


def _fallback_image(
    image: "ImageIndexEntry", error: BaseException, operation: str
) -> Image.Image:
    logger.warning(
        "Failed to %s image %d at %s offset %d; using the image-not-found "
        "fallback while preserving the original token span: %s",
        operation,
        image.img_id_in_doc,
        image.tar_path,
        image.tar_offset.offset,
        error,
    )
    return _load_image_not_found().copy()


def _decode_image_or_fallback(img_bytes: bytes, image: "ImageIndexEntry") -> Image.Image:
    try:
        return _decode_image(img_bytes)
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        return _fallback_image(image, error, "decode")


def _load_image_or_fallback(image: "ImageIndexEntry") -> Image.Image:
    try:
        img_bytes = get_tar_bytes_helper(
            tar_path=image.tar_path,
            offset=image.tar_offset.to_pydantic(),
            is_compressed=False,
            read_full_tar=True,
        )
        if img_bytes is None:
            raise OSError("image tar entry is empty")
    except OSError as error:
        return _fallback_image(image, error, "retrieve")
    return _decode_image_or_fallback(img_bytes, image)


def _batch_read_span(
    tar_path: str,
    span: list[_ImageEntry],
    results: list[Image.Image | None],
) -> None:
    """Issue a single ranged read for a contiguous span of images.

    Duplicate entries (same offset) reuse the same decoded image.
    """
    min_offset = span[0].tar_offset.offset
    last = span[-1].tar_offset
    max_end = last.offset + last.size

    total_size = max_end - min_offset
    try:
        span_bytes = load_bytes(tar_path, min_offset, total_size)
    except OSError as error:
        for entry in span:
            results[entry.idx] = _fallback_image(entry.image, error, "retrieve")
        return

    # Cache decoded images by offset to handle duplicates
    decoded_cache: dict[int, Image.Image] = {}

    for entry in span:
        offset = entry.tar_offset.offset
        if offset in decoded_cache:
            results[entry.idx] = decoded_cache[offset]
            continue

        rel_start = offset - min_offset
        img_raw = span_bytes[rel_start : rel_start + entry.tar_offset.size]

        if entry.tar_offset.new_tar_format:
            content = img_raw[TAR_HEADER_SIZE:]
            img_bytes: bytes = _read_bytes(is_compressed=False, bytes=content)
        else:
            img_bytes = img_raw

        img = _decode_image_or_fallback(img_bytes, entry.image)
        decoded_cache[offset] = img
        results[entry.idx] = img
