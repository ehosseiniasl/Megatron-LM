"""Image transform utilities: geometry helpers, rotation, trim, resize, normalize.

Copied from:
  - shared/image_utils.py (geometry helpers)
  - mistral/model/multimodal/image_transforms/utils.py (ImageRotator, Trim, ResizeKeepRatio)
  - mistral/model/args.py (normalization constants)

Only the functions needed for crawl_default_no_high_res.
"""

import dataclasses

import numpy as np

from mistral_fork.mistral.data.pretokenized.image_transform_metadata import (
    ImageBoundingBox,
    TransformImageMetadata,
)

# ---------------------------------------------------------------------------
# Constants — copied from shared/image_utils.py and mistral/model/args.py
# ---------------------------------------------------------------------------

CRAWL_L2_RESIZE_THRESHOLD: float = 0.008773181770004912
L2_RESIZE_KEY = "image_min_size=224,l2_resize_factor=2"

CLIP_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclasses.dataclass
class TransformContext:
    max_image_size: int | None = None
    content_modified: bool = False


# ---------------------------------------------------------------------------
# Geometry helpers — copied from shared/image_utils.py
# ---------------------------------------------------------------------------


def get_ratio(source_size: tuple[int, int], max_image_size: int) -> float:
    """Copied from: shared/image_utils.py (get_ratio)"""
    h, w = source_size
    return max(h / max_image_size, w / max_image_size)


def round_up(input_size: tuple[int, int], resize_multiple: int) -> tuple[int, int]:
    """Copied from: shared/image_utils.py (round_up)"""
    h, w = input_size
    new_h = ((h - 1) // resize_multiple + 1) * resize_multiple
    new_w = ((w - 1) // resize_multiple + 1) * resize_multiple
    return (new_h, new_w)


def get_transformed_size(
    original_size: tuple[int, int],
    max_image_size: int,
    resize_multiple: int,
) -> tuple[int, int]:
    """Copied from: shared/image_utils.py (get_transformed_size)"""
    ratio = get_ratio(source_size=original_size, max_image_size=max_image_size)
    ratio = max(ratio, 1.0)
    size = (round(original_size[0] / ratio), round(original_size[1] / ratio))
    return round_up(input_size=size, resize_multiple=resize_multiple)


def get_downsampled_image_size(
    resize_factor: int,
    original_size: tuple[int, int],
    min_length: int = 224,
) -> tuple[int, int]:
    """Copied from: shared/image_utils.py (get_downsampled_image_size)"""
    original_width, original_height = original_size
    aspect_ratio = original_width / original_height

    downsample_width = int(original_width / resize_factor)
    downsample_height = int(original_height / resize_factor)

    if max(downsample_width, downsample_height) < min_length:
        if downsample_width > downsample_height:
            downsample_width = min_length
            downsample_height = int(downsample_width / aspect_ratio)
        else:
            downsample_height = min_length
            downsample_width = int(downsample_height * aspect_ratio)

    return max(1, downsample_width), max(1, downsample_height)


# ---------------------------------------------------------------------------
# Resize metadata — copied from utils.py (ResizeKeepRatio._get_transformed_metadata)
# ---------------------------------------------------------------------------


def compute_resize_metadata(
    meta: TransformImageMetadata,
    rng: np.random.Generator,
    max_image_size: int,
    patch_size: int,
    spatial_merge_size: int,
    variable_size: bool,
    random_scale_prob: float,
    random_scale_range: tuple[float, float],
) -> TransformImageMetadata:
    """Compute output dimensions after aspect-ratio-preserving resize.

    Copied from: utils.py (ResizeKeepRatio._get_transformed_metadata)
    """
    source_size = (meta.height, meta.width)
    ratio = get_ratio(source_size=source_size, max_image_size=max_image_size)

    if variable_size and ratio <= 1.0:
        size: tuple[int, int] = source_size
    else:
        ratio_factor = (1.0, 1.0)
        if random_scale_prob > 0 and rng.random() < random_scale_prob:
            factor = rng.uniform(random_scale_range[0], random_scale_range[1])
            ratio_factor = (factor, factor)

        size = (
            round(source_size[0] * ratio_factor[0] / ratio),
            round(source_size[1] * ratio_factor[1] / ratio),
        )
        ratio2 = get_ratio(source_size=size, max_image_size=max_image_size)
        if ratio2 > 1.0:
            size = (round(size[0] / ratio2), round(size[1] / ratio2))

    resize_multiple = spatial_merge_size * patch_size
    size = get_transformed_size(
        original_size=size,
        max_image_size=max_image_size,
        resize_multiple=resize_multiple,
    )
    assert size[0] >= 1 and size[1] >= 1
    return dataclasses.replace(meta, height=size[0], width=size[1])


# ---------------------------------------------------------------------------
# ImageRotator — copied from utils.py (ImageRotator)
# ---------------------------------------------------------------------------


def sample_rotation(
    rng: np.random.Generator,
    rotate90_prob: float,
    rotate180_prob: float,
    rotate270_prob: float,
) -> int:
    """Copied from: utils.py (ImageRotator._sample_k)"""
    p_none = 1.0 - (rotate90_prob + rotate180_prob + rotate270_prob)
    return int(
        rng.choice(a=4, p=(p_none, rotate90_prob, rotate180_prob, rotate270_prob))
    )


def apply_rotation_metadata(
    meta: TransformImageMetadata, k: int
) -> TransformImageMetadata:
    """Copied from: utils.py (ImageRotator._rotate_meta)"""
    if k % 2 == 1:
        meta = dataclasses.replace(meta, width=meta.height, height=meta.width)
        if meta.img_bbox_for_trim_is_computed and meta._img_bbox_for_trim is not None:
            bbox = meta._img_bbox_for_trim
            if k == 1:
                new_bbox = ImageBoundingBox(
                    top_left_x=bbox.top_left_y,
                    top_left_y=1.0 - bbox.bottom_right_x,
                    bottom_right_x=bbox.bottom_right_y,
                    bottom_right_y=1.0 - bbox.top_left_x,
                )
            else:  # k == 3
                new_bbox = ImageBoundingBox(
                    top_left_x=1.0 - bbox.bottom_right_y,
                    top_left_y=bbox.top_left_x,
                    bottom_right_x=1.0 - bbox.top_left_y,
                    bottom_right_y=bbox.bottom_right_x,
                )
            meta = dataclasses.replace(meta, _img_bbox_for_trim=new_bbox)
    elif (
        k == 2
        and meta.img_bbox_for_trim_is_computed
        and meta._img_bbox_for_trim is not None
    ):
        bbox = meta._img_bbox_for_trim
        new_bbox = ImageBoundingBox(
            top_left_x=1.0 - bbox.bottom_right_x,
            top_left_y=1.0 - bbox.bottom_right_y,
            bottom_right_x=1.0 - bbox.top_left_x,
            bottom_right_y=1.0 - bbox.top_left_y,
        )
        meta = dataclasses.replace(meta, _img_bbox_for_trim=new_bbox)
    return meta


# ---------------------------------------------------------------------------
# Trim — copied from utils.py (Trim)
# ---------------------------------------------------------------------------


def apply_trim_metadata(
    meta: TransformImageMetadata,
    rng: np.random.Generator,
    trim_prob: float,
    trim_pad: int,
) -> tuple[TransformImageMetadata, ImageBoundingBox | None]:
    """Copied from: utils.py (Trim._get_transformed_metadata)"""
    if trim_prob == 0 or rng.random() > trim_prob:
        return meta, None

    if not meta.img_bbox_for_trim_is_computed:
        return meta, None

    img_bbox = meta.img_bbox_for_trim
    if img_bbox is None:
        return meta, None

    height, width = meta.height, meta.width

    # Denormalize from [0,1] to pixel coordinates
    left = img_bbox.top_left_x * width
    upper = img_bbox.top_left_y * height
    right = img_bbox.bottom_right_x * width
    lower = img_bbox.bottom_right_y * height

    # Apply padding
    left = max(0, left - trim_pad)
    upper = max(0, upper - trim_pad)
    right = min(width, right + trim_pad)
    lower = min(height, lower + trim_pad)

    # Store as normalized [0,1] coordinates for pixel application
    result_bbox = ImageBoundingBox(
        top_left_x=left / width,
        top_left_y=upper / height,
        bottom_right_x=right / width,
        bottom_right_y=lower / height,
    )

    new_width = int(right) - int(left)
    new_height = int(lower) - int(upper)
    meta = dataclasses.replace(
        meta,
        width=new_width,
        height=new_height,
        _img_bbox_for_trim=None,
        img_bbox_for_trim_is_computed=True,
    )

    return meta, result_bbox
