"""Simple PretokenizedImageEncoder backed by pluggable ImageTransforms.

This is not in components/ because the encoder is always the same wrapper —
the variable part is the ImageTransform (which has its own enum/factory).
"""

import hashlib
from typing import Any

import numpy as np
from PIL import Image

from mistral_fork.mistral.data.pretokenized.components.image_transforms import (
    ImageTransform,
    ImageTransformType,
    SpecialImageIDs,
    image_transforms_factory,
)
from mistral_fork.mistral.data.pretokenized.datasource import PretokenizedImageEncoder
from mistral_fork.mistral.data.pretokenized.image_transform_metadata import TransformImageMetadata
from mistral_fork.mistral_common.tokens.tokenizers.image import image_size_to_num_tokens
from mistral_fork.shared.image_utils import TrainingImageEncoding


class SimpleImageEncoder(PretokenizedImageEncoder):
    """PretokenizedImageEncoder that delegates to an ImageTransform.

    - precompute_transforms: delegates to transforms (used during epoch index building)
    - apply_transforms_and_encode_image: applies transforms, then builds image tokens
      using the same layout as Mistral: ([img]*w + [img_break])*h with last = img_end
    """

    def __init__(
        self,
        transform_type: ImageTransformType,
        special_ids: SpecialImageIDs,
        max_image_size: int = 1024,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
    ) -> None:
        self._transform_type = transform_type
        self._special_ids = special_ids
        self._max_image_size = max_image_size
        self._patch_size = patch_size
        self._spatial_merge_size = spatial_merge_size
        self._transforms = image_transforms_factory(transform_type, special_ids)

    def uuid(self) -> str:
        # Must match Mistral's PretokenizedFlexibleImageEncoder.uuid() so we
        # find the same cached epoch indexes.
        ids_hash = hashlib.md5(str(self._special_ids).encode()).hexdigest()[:8]
        return f"{self._transform_type.value}_{ids_hash}"

    def to_infos_dict(self) -> dict[str, Any]:
        return {
            "image_process_args": _image_process_args_for_mistral_infos(
                self._transform_type
            ),
            "special_ids": {
                "img": self._special_ids.img,
                "img_break": self._special_ids.img_break,
                "img_end": self._special_ids.img_end,
            },
            "is_train": True,
        }

    def _get_image_tokens(self, h_pixels: int, w_pixels: int) -> list[int]:
        """Generate image token IDs matching Mistral's layout."""
        if self._transform_type is ImageTransformType.NO_TRANSFORM:
            w_tokens, h_tokens = image_size_to_num_tokens(
                size=(w_pixels, h_pixels),
                max_image_size=self._max_image_size,
                patch_size=self._patch_size,
                spatial_merge_size=self._spatial_merge_size,
            )
        else:
            w_tokens = w_pixels // self._patch_size
            h_tokens = h_pixels // self._patch_size
            assert h_tokens % self._spatial_merge_size == 0
            assert w_tokens % self._spatial_merge_size == 0
            h_tokens //= self._spatial_merge_size
            w_tokens //= self._spatial_merge_size
        tokens = (
            [self._special_ids.img] * w_tokens + [self._special_ids.img_break]
        ) * h_tokens
        if tokens:
            tokens[-1] = self._special_ids.img_end
        return tokens

    def precompute_transforms(
        self, metadata: TransformImageMetadata, seed: tuple[int, ...]
    ) -> tuple[int, TransformImageMetadata]:
        return self._transforms.precompute_transforms(metadata, seed)

    def apply_transforms_and_encode_image(
        self,
        image: Image.Image,
        metadata: TransformImageMetadata,
        seed: tuple[int, ...],
    ) -> tuple[TrainingImageEncoding, TransformImageMetadata]:
        transformed_image, new_metadata = self._transforms.apply_transforms(
            image, metadata, seed
        )
        tokens = self._get_image_tokens(
            h_pixels=new_metadata.height, w_pixels=new_metadata.width
        )
        return TrainingImageEncoding(
            tokens=tokens, image=transformed_image
        ), new_metadata

    def encode_image_metadata(
        self, metadata: TransformImageMetadata, seed: tuple[int, ...]
    ) -> list[int]:
        """Build the native image-token sequence without loading image pixels."""
        _, transformed_metadata = self.precompute_transforms(metadata, seed)
        return self._get_image_tokens(
            h_pixels=transformed_metadata.height, w_pixels=transformed_metadata.width
        )

    def get_transforms(self) -> ImageTransform:
        return self._transforms


def _image_process_args_for_mistral_infos(
    transform_type: ImageTransformType,
) -> dict[str, Any]:
    downsampling_args = {
        "always_high_res_charts_plots": False,
        "apply_dynamic_proportion": 0.9,
        "apply_high_res_proportion": 0.0,
        "high_res_charts_plots_threshold": 1.0,
        "high_res_image_size": 2048,
        "l2_resize_factor": 2,
        "l2_resize_threshold": 0.008773181770004912,
        "min_size": 224,
        "spatial_merge_size": 2,
    }
    match transform_type:
        case (
            ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES
            | ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES_NO_L2_RESIZE
            | ImageTransformType.GROUNDING_DEFAULT
            | ImageTransformType.GROUNDING_DEFAULT_NO_L2_RESIZE
        ):
            is_grounding = transform_type in (
                ImageTransformType.GROUNDING_DEFAULT,
                ImageTransformType.GROUNDING_DEFAULT_NO_L2_RESIZE,
            )
            no_l2_resize = transform_type in (
                ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES_NO_L2_RESIZE,
                ImageTransformType.GROUNDING_DEFAULT_NO_L2_RESIZE,
            )
            apply_dynamic_proportion = (
                0.0 if no_l2_resize else downsampling_args["apply_dynamic_proportion"]
            )
            return {
                "centering": "default",
                "color_jitter_args": None,
                "downsampling_args": {
                    **downsampling_args,
                    "apply_dynamic_proportion": apply_dynamic_proportion,
                    "apply_high_res_proportion": 0.1 if is_grounding else 0.0,
                    "high_res_image_size": 1540,
                },
                "enforce_image_first": not is_grounding,
                "enforce_single_image": not is_grounding,
                "eval_image_size": None,
                "image_size": 1036,
                "interpolation": "bicubic",
                "max_patches_per_side": None,
                "max_tokens": None,
                "num_channels": 3,
                "ocr_args": None,
                "patch_size": 14,
                "per_source_transform_args_str": None,
                "random_scale_aspect_args": {
                    "random_aspect_prob": 0.0,
                    "random_aspect_range_max": 1.11,
                    "random_aspect_range_min": 0.9,
                    "random_scale_prob": 0.5,
                    "random_scale_range_max": 1.11,
                    "random_scale_range_min": 0.9,
                },
                "schedule_args": None,
                "transform_type": "sometimes_high_res",
                "use_cv2_resize": True,
                "variable_size": True,
            }
        case ImageTransformType.SOMETIMES_HIGH_RES_DEFAULT:
            return {
                "centering": "default",
                "color_jitter_args": None,
                "downsampling_args": {
                    "always_high_res_charts_plots": False,
                    "apply_dynamic_proportion": 0.0,
                    "apply_high_res_proportion": 0.1,
                    "high_res_charts_plots_threshold": 1.0,
                    "high_res_image_size": 1540,
                    "l2_resize_factor": 2,
                    "l2_resize_threshold": 0.008773181770004912,
                    "min_size": 224,
                    "spatial_merge_size": 2,
                },
                "enforce_image_first": True,
                "enforce_single_image": True,
                "eval_image_size": None,
                "image_size": 1036,
                "interpolation": "bicubic",
                "max_patches_per_side": None,
                "max_tokens": None,
                "num_channels": 3,
                "ocr_args": None,
                "patch_size": 14,
                "per_source_transform_args_str": None,
                "random_crop_args": None,
                "random_scale_aspect_args": {
                    "random_aspect_prob": 0.0,
                    "random_aspect_range_max": 1.11,
                    "random_aspect_range_min": 0.9,
                    "random_scale_prob": 0.5,
                    "random_scale_range_max": 1.0,
                    "random_scale_range_min": 0.5,
                },
                "schedule_args": None,
                "transform_type": "sometimes_high_res",
                "use_cv2_resize": True,
                "variable_size": True,
            }
        case ImageTransformType.NO_TRANSFORM:
            return {
                "centering": "default",
                "color_jitter_args": None,
                "downsampling_args": None,
                "enforce_image_first": True,
                "enforce_single_image": True,
                "eval_image_size": None,
                "image_size": 1024,
                "interpolation": "bicubic",
                "max_patches_per_side": None,
                "max_tokens": None,
                "num_channels": 3,
                "ocr_args": None,
                "patch_size": 14,
                "per_source_transform_args_str": None,
                "random_scale_aspect_args": None,
                "schedule_args": None,
                "transform_type": "default",
                "use_cv2_resize": True,
                "variable_size": True,
            }
        case _ as unreachable:
            raise AssertionError(f"Unsupported transform type: {unreachable}")
