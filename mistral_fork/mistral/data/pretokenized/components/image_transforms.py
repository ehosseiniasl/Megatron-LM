"""Image transform registry, protocol, and factory.

ImageTransformType identifies the image transform pipeline used during
epoch index building and training. String values are stable — changing
a value invalidates cached epoch indexes on disk (via the image encoder uuid).

Transform implementations live in:
  mistral_fork/mistral/model/multimodal/image_transforms/
"""

import dataclasses
import typing
from enum import StrEnum
from typing import Protocol

import numpy as np
from PIL import Image

from mistral_fork.mistral.data.pretokenized.image_transform_metadata import TransformImageMetadata
from mistral_fork.mistral.model.multimodal.image_transforms.image_transforms import (
    CrawlDefaultNoHighRes,
    GroundingDefault,
    SometimesHighResDefault,
)
from mistral_fork.mistral_common.tokens.tokenizers.image import image_size_to_num_tokens

DEFAULT_MAX_IMAGE_SIZE = 1024
DEFAULT_IMAGE_PATCH_SIZE = 14
DEFAULT_SPATIAL_MERGE_SIZE = 2


@dataclasses.dataclass(frozen=True)
class SpecialImageIDs:
    """Special token IDs for image encoding."""

    img: int
    img_break: int
    img_end: int


class ImageTransformType(StrEnum):
    NO_TRANSFORM = "no_transform"
    CRAWL_DEFAULT_NO_HIGH_RES = "crawl_default_no_high_res"
    CRAWL_DEFAULT_NO_HIGH_RES_NO_L2_RESIZE = "crawl_default_no_high_res_no_l2_resize"
    GROUNDING_DEFAULT = "grounding_default"
    GROUNDING_DEFAULT_NO_L2_RESIZE = "grounding_default_no_l2_resize"
    SOMETIMES_HIGH_RES_DEFAULT = "sometimes_high_res_default"


class ImageTransform(Protocol):
    def precompute_transforms(
        self, metadata: TransformImageMetadata, seed: tuple[int, ...]
    ) -> tuple[int, TransformImageMetadata]:
        """Return (n_tokens, updated_metadata) without loading the image."""
        ...

    def apply_transforms(
        self,
        image: Image.Image,
        metadata: TransformImageMetadata,
        seed: tuple[int, ...],
    ) -> tuple[np.ndarray, TransformImageMetadata]:
        """Apply transforms to the image and return (transformed_image, updated_metadata)."""
        ...


# ---------------------------------------------------------------------------
# _NoTransform — trivial, lives here to avoid an extra file
# ---------------------------------------------------------------------------


class _NoTransform:
    """Identity transform: no image modification, token count from Mistral formula."""

    def __init__(
        self,
        special_ids: SpecialImageIDs,
        max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
        patch_size: int = DEFAULT_IMAGE_PATCH_SIZE,
        spatial_merge_size: int = DEFAULT_SPATIAL_MERGE_SIZE,
    ) -> None:
        self._special_ids = special_ids
        self._max_image_size = max_image_size
        self._patch_size = patch_size
        self._spatial_merge_size = spatial_merge_size

    def _compute_n_tokens(self, width: int, height: int) -> int:
        w_tokens, h_tokens = image_size_to_num_tokens(
            size=(width, height),
            max_image_size=self._max_image_size,
            patch_size=self._patch_size,
            spatial_merge_size=self._spatial_merge_size,
        )
        return (w_tokens + 1) * h_tokens

    def precompute_transforms(
        self, metadata: TransformImageMetadata, seed: tuple[int, ...]
    ) -> tuple[int, TransformImageMetadata]:
        n_toks = self._compute_n_tokens(metadata.width, metadata.height)
        return n_toks, metadata

    def apply_transforms(
        self,
        image: Image.Image,
        metadata: TransformImageMetadata,
        seed: tuple[int, ...],
    ) -> tuple[np.ndarray, TransformImageMetadata]:
        return np.array(image), metadata


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def image_transforms_factory(
    transform_type: ImageTransformType,
    special_ids: SpecialImageIDs,
) -> ImageTransform:
    """Return an ImageTransform for the given type."""
    match transform_type:
        case ImageTransformType.NO_TRANSFORM:
            return _NoTransform(special_ids=special_ids)
        case ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES:
            return CrawlDefaultNoHighRes()
        case ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES_NO_L2_RESIZE:
            return CrawlDefaultNoHighRes(apply_dynamic_proportion=0.0)
        case ImageTransformType.GROUNDING_DEFAULT:
            return GroundingDefault()
        case ImageTransformType.GROUNDING_DEFAULT_NO_L2_RESIZE:
            return GroundingDefault(apply_dynamic_proportion=0.0)
        case ImageTransformType.SOMETIMES_HIGH_RES_DEFAULT:
            return SometimesHighResDefault()
        case _ as unreachable:
            typing.assert_never(unreachable)
