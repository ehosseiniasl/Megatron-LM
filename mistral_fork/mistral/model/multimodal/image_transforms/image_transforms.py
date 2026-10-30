"""Crawl image transform implementation.

Copied from:
  - mistral/model/multimodal/image_transforms/image_transforms.py
    (SometimesHighResTransform, DynamicResizeTransform)
  - mistral/model/multimodal/image_transforms/transform_registry.py
    (_build_crawl_default)

These classes implement the fork's ImageTransform protocol
(precompute_transforms / apply_transforms) using the same logic
and RNG consumption order as the upstream Mistral classes.

Pixel operations in apply_transforms are copy-pasted from upstream
to ensure bitwise parity. Each block is annotated with the upstream
source file and line numbers.
"""


import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import ToTensor

from mistral_fork.mistral.data.pretokenized.image_transform_metadata import TransformImageMetadata
from mistral_fork.shared.image_utils import convert_to_rgb

from .utils import (
    CLIP_DATASET_MEAN,
    CLIP_DATASET_STD,
    CRAWL_L2_RESIZE_THRESHOLD,
    L2_RESIZE_KEY,
    TransformContext,
    apply_rotation_metadata,
    apply_trim_metadata,
    compute_resize_metadata,
    get_downsampled_image_size,
    round_up,
)

# Upstream: utils.py:295 — torchvision.transforms.v2.ToTensor
# Instantiate once (stateless).
_TO_TENSOR = ToTensor()


# ---------------------------------------------------------------------------
# Shared token count computation
# ---------------------------------------------------------------------------


def _compute_n_tokens(
    width: int,
    height: int,
    patch_size: int,
    spatial_merge_size: int,
) -> int:
    w_tokens = width // patch_size
    h_tokens = height // patch_size
    assert h_tokens % spatial_merge_size == 0
    assert w_tokens % spatial_merge_size == 0
    h_tokens //= spatial_merge_size
    w_tokens //= spatial_merge_size
    return (w_tokens + 1) * h_tokens


# ---------------------------------------------------------------------------
# CrawlDefaultNoHighRes
#
# Copied from: image_transforms.py (SometimesHighResTransform + DynamicResizeTransform)
# Config from: transform_registry.py (_build_crawl_default(with_high_res=False))
# ---------------------------------------------------------------------------


class CrawlDefaultNoHighRes:
    """Crawl transform without high-res upsampling.

    Pipeline: optional L2 dynamic downsampling -> aspect-ratio resize -> normalize.

    Args:
        apply_dynamic_proportion: Probability of consulting the L2 score for
            dynamic downsampling. When omitted, the concrete preset's class
            default is used; setting it to zero skips only the L2-guided step.
    """

    IMAGE_SIZE = 1036
    PATCH_SIZE = 14
    SPATIAL_MERGE_SIZE = 2
    L2_THRESHOLD = CRAWL_L2_RESIZE_THRESHOLD
    L2_RESIZE_FACTOR = 2
    L2_MIN_SIZE = 224
    HIGH_RES_IMAGE_SIZE = 1540
    APPLY_HIGH_RES_PROPORTION = 0.0
    APPLY_DYNAMIC_PROPORTION = 0.9
    RANDOM_SCALE_PROB = 0.5
    RANDOM_SCALE_RANGE = (0.9, 1.11)

    def __init__(self, *, apply_dynamic_proportion: float | None = None) -> None:
        if apply_dynamic_proportion is None:
            apply_dynamic_proportion = self.APPLY_DYNAMIC_PROPORTION
        if not 0.0 <= apply_dynamic_proportion <= 1.0:
            raise ValueError(
                "apply_dynamic_proportion must be between 0 and 1, got "
                f"{apply_dynamic_proportion}"
            )
        self._apply_dynamic_proportion = apply_dynamic_proportion

    def _update_context_max_image_size(
        self,
        meta: TransformImageMetadata,
        ctx: TransformContext,
        rng: np.random.Generator,
    ) -> None:
        """Copied from: SometimesHighResTransform._get_max_size + ComputeDynamicMaxSize."""
        image_max_size = self.IMAGE_SIZE

        # High-res decision: matches Mistral's SometimesHighResTransform._get_max_size.
        if rng.random() < self.APPLY_HIGH_RES_PROPORTION:
            image_max_size = round_up(
                (self.HIGH_RES_IMAGE_SIZE, self.HIGH_RES_IMAGE_SIZE),
                resize_multiple=self.PATCH_SIZE,
            )[0]

        # Dynamic downsampling decision
        p = rng.random()
        max_pil_image_size = max(meta.height, meta.width)
        if (
            p < self._apply_dynamic_proportion
            and max_pil_image_size / self.L2_RESIZE_FACTOR < image_max_size
            and max_pil_image_size > self.L2_MIN_SIZE
        ):
            if ctx.content_modified:
                raise ValueError(
                    "The image was modified, so we cannot use the precomputed "
                    "L2 reconstruction score."
                )

            l2_score = meta.l2_reconstruction_score.get(L2_RESIZE_KEY)
            if l2_score is None:
                raise ValueError(
                    f"l2_reconstruction_score is not precomputed for key "
                    f"'{L2_RESIZE_KEY}'. Metadata must contain this score."
                )

            if l2_score < self.L2_THRESHOLD:
                new_max = max(
                    get_downsampled_image_size(
                        resize_factor=self.L2_RESIZE_FACTOR,
                        original_size=(meta.width, meta.height),
                        min_length=self.L2_MIN_SIZE,
                    )
                )
                assert new_max <= image_max_size
                image_max_size = new_max
                image_max_size = round_up(
                    (image_max_size, image_max_size),
                    resize_multiple=self.PATCH_SIZE,
                )[0]

        ctx.max_image_size = image_max_size

    def _get_output_metadata(
        self, metadata: TransformImageMetadata, rng: np.random.Generator
    ) -> TransformImageMetadata:
        ctx = TransformContext()
        self._update_context_max_image_size(metadata, ctx, rng)
        assert ctx.max_image_size is not None, "Expected max image size to be set"
        meta = compute_resize_metadata(
            meta=metadata,
            rng=rng,
            max_image_size=ctx.max_image_size,
            patch_size=self.PATCH_SIZE,
            spatial_merge_size=self.SPATIAL_MERGE_SIZE,
            variable_size=True,
            random_scale_prob=self.RANDOM_SCALE_PROB,
            random_scale_range=self.RANDOM_SCALE_RANGE,
        )
        return meta

    def precompute_transforms(
        self, metadata: TransformImageMetadata, seed: tuple[int, ...]
    ) -> tuple[int, TransformImageMetadata]:
        rng = np.random.default_rng(seed)
        meta = self._get_output_metadata(metadata, rng)
        n_toks = _compute_n_tokens(
            meta.width,
            meta.height,
            self.PATCH_SIZE,
            self.SPATIAL_MERGE_SIZE,
        )
        return n_toks, meta

    def apply_transforms(
        self,
        image: Image.Image,
        metadata: TransformImageMetadata,
        seed: tuple[int, ...],
    ) -> tuple[np.ndarray, TransformImageMetadata]:
        import cv2  # noqa: F401
        rng = np.random.default_rng(seed)
        meta = self._get_output_metadata(metadata, rng)
        target_size = (meta.height, meta.width)

        # --- Pixel operations: copy-pasted from upstream Mistral ---

        # ToTensor (upstream: utils.py:295-307)
        img = _TO_TENSOR(convert_to_rgb(image))

        # ResizeKeepRatio._forward, use_cv2_resize=True (upstream: utils.py:890-895)
        np_img = img.permute((1, 2, 0)).numpy()
        np_img = cv2.resize(
            np_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA
        )
        img = torch.from_numpy(np_img).permute((2, 0, 1)).contiguous()

        # Normalize (upstream: utils.py:323-332, via torchvision.transforms.v2.Normalize)
        mean = torch.as_tensor(CLIP_DATASET_MEAN, dtype=img.dtype)
        std = torch.as_tensor(CLIP_DATASET_STD, dtype=img.dtype)
        img = img.clone()
        img.sub_(mean[:, None, None]).div_(std[:, None, None])

        return img.numpy(), meta


class SometimesHighResDefault(CrawlDefaultNoHighRes):
    """Mistral's sometimes_high_res_default transform preset."""

    APPLY_HIGH_RES_PROPORTION = 0.1
    APPLY_DYNAMIC_PROPORTION = 0.0
    RANDOM_SCALE_RANGE = (0.5, 1.0)


class GroundingDefault(CrawlDefaultNoHighRes):
    """Grounding default transform.

    Same pixel pipeline as crawl default, but enables the upstream high-res
    branch used by SourceTransform.GROUNDING_DEFAULT.
    """

    APPLY_HIGH_RES_PROPORTION = 0.1
