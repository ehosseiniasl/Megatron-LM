# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Image preprocessing for multimodal inference servers.

Shared between vlm_server.py and the coordinator/engine image dispatch in
run_dynamic_text_generation_server.py. Lives in core/inference so the engine
can import it without circular dependencies.
"""

import io
import math
from typing import Optional

import torch

from megatron.core.inference.config import ImageProcessingConfig
from megatron.core.models.vision.encoder_registry import REGISTRY as _ENCODER_REGISTRY

_MISTRAL_CRAWL_EVAL_SPECS = {
    "mistral-vit": (1036, 14, 2),
    "mistral-vit-410m": (1036, 14, 2),
    "mistral-vit-1p6b": (1036, 14, 2),
}


def _resolve_pixel_stats(vision_model_type: str):
    """Return (pixel_mean, pixel_std) for a vision encoder.

    Reads from the canonical encoder registry so training and inference share
    one source of truth. Falls back to CLIP-style stats for unknown encoders
    (matching the registry's own EncoderSpec defaults).
    """
    spec = _ENCODER_REGISTRY.get(vision_model_type)
    if spec is not None:
        return list(spec.pixel_mean), list(spec.pixel_std)
    # Fall back to CLIP defaults pulled from the registry's dataclass field
    # defaults rather than instantiating (which would need the four required
    # geometry fields) or copying locally.
    from megatron.core.models.vision.encoder_registry import EncoderSpec

    fields = EncoderSpec.__dataclass_fields__
    return list(fields["pixel_mean"].default), list(fields["pixel_std"].default)


def dynamic_res_preprocess(
    image,
    min_patches=1,
    max_patches=128,
    res_step=16,
    factor_max=1.0,
    pixel_shuffle=False,
    spatial_merge_size=1,
):
    """Resize image to fit within [min_patches, max_patches] preserving aspect ratio.

    For pixel_shuffle, patch grid dimensions are rounded to even numbers for
    compatibility.

    NOTE: Training uses ``DynamicResolutionImageTilingStrategy._process_single``
    (in megatron.energon.task_encoder.multimodal.image_tiling) as the canonical
    resize. The math here is intentionally a subset of that strategy and could
    drift if energon's implementation changes (e.g. ``min_side`` floor, tiling
    augmentation). For full parity, inference should call into the energon
    strategy directly — TODO once we have a clean way to import it that doesn't
    require energon at engine-drain time.
    """
    orig_width, orig_height = image.size

    # Use math.ceil, not round(x + 0.5) — the latter is banker's rounding and
    # produces off-by-one, non-monotonic patch counts on exactly-aligned sides.
    closest_patch_height = math.ceil(orig_height / res_step)
    closest_patch_width = math.ceil(orig_width / res_step)
    patches = closest_patch_height * closest_patch_width

    factor = min(math.sqrt(max_patches / patches), factor_max)
    target_patch_height = math.floor(factor * closest_patch_height)
    target_patch_width = math.floor(factor * closest_patch_width)

    if target_patch_height * target_patch_width < min_patches:
        up_factor = math.sqrt(min_patches / max(target_patch_height * target_patch_width, 1))
        target_patch_height = math.ceil(up_factor * target_patch_height)
        target_patch_width = math.ceil(up_factor * target_patch_width)

    grid_multiple = max(2 if pixel_shuffle else 1, spatial_merge_size)
    if grid_multiple > 1:
        if target_patch_height % grid_multiple:
            increase = grid_multiple - target_patch_height % grid_multiple
            if (target_patch_height + increase) * target_patch_width <= max_patches:
                target_patch_height += increase
            else:
                target_patch_height -= target_patch_height % grid_multiple
        if target_patch_width % grid_multiple:
            increase = grid_multiple - target_patch_width % grid_multiple
            if target_patch_height * (target_patch_width + increase) <= max_patches:
                target_patch_width += increase
            else:
                target_patch_width -= target_patch_width % grid_multiple

        target_patch_height = max(grid_multiple, target_patch_height)
        target_patch_width = max(grid_multiple, target_patch_width)

    assert target_patch_height * target_patch_width <= max_patches

    resized_img = image.resize((target_patch_width * res_step, target_patch_height * res_step))
    return resized_img


def _preprocess_mistral_crawl_eval(image, config: ImageProcessingConfig) -> torch.Tensor:
    """Apply the deterministic evaluation counterpart of Mistral crawl training."""
    import cv2
    import numpy as np
    from torchvision.transforms.v2 import ToTensor

    try:
        max_image_size, patch_size, spatial_merge_size = _MISTRAL_CRAWL_EVAL_SPECS[
            config.vision_model_type
        ]
    except KeyError as exc:
        raise ValueError(
            f"mistral-crawl-eval does not support vision model type {config.vision_model_type!r}"
        ) from exc

    if config.patch_dim != patch_size or config.spatial_merge_size != spatial_merge_size:
        raise ValueError(
            "mistral-crawl-eval expects "
            f"patch_dim={patch_size} and spatial_merge_size={spatial_merge_size}"
        )

    from mistral_fork.mistral.model.multimodal.image_transforms.utils import (
        CLIP_DATASET_MEAN,
        CLIP_DATASET_STD,
        get_transformed_size,
    )

    target_h, target_w = get_transformed_size(
        original_size=(image.height, image.width),
        max_image_size=max_image_size,
        resize_multiple=patch_size * spatial_merge_size,
    )
    image_tensor = ToTensor()(image)
    image_array = cv2.resize(
        image_tensor.permute((1, 2, 0)).numpy(), (target_w, target_h), interpolation=cv2.INTER_AREA
    )
    image_tensor = torch.from_numpy(np.asarray(image_array)).permute((2, 0, 1)).contiguous()
    mean = torch.as_tensor(CLIP_DATASET_MEAN, dtype=image_tensor.dtype)
    std = torch.as_tensor(CLIP_DATASET_STD, dtype=image_tensor.dtype)
    return image_tensor.sub(mean[:, None, None]).div(std[:, None, None])


def preprocess_image_bytes(
    image_bytes: bytes,
    config: ImageProcessingConfig,
    target_hw=None,
    device: Optional[torch.device] = None,
) -> tuple:
    """Preprocess raw image bytes into tensors for dynamic-resolution inference.

    Args:
        image_bytes: Raw image file bytes (e.g. JPEG/PNG).
        config: Image preprocessing configuration.
        target_hw: Optional (H, W) tuple in pixels. If given, resize to exactly
            this size instead of running dynamic_res_preprocess. Used to keep
            all images in a multi-image request at the same patch dimensions.
        device: Optional device for the image patch tensor. Image-size metadata
            intentionally remains on CPU for shape arithmetic.

    Returns:
        ``imgs`` on ``device`` when provided and CPU ``imgs_sizes`` metadata.
        imgs shape: [1, num_patches, C*patch_dim*patch_dim]
        imgs_sizes shape: [1, 2] with [H, W] in pixels.
    """
    from PIL import Image

    try:
        from torchvision import transforms as T
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for VLM image preprocessing. Install a "
            "torchvision build matching your torch version, or use the NGC "
            "PyTorch container that ships one."
        ) from exc

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    patch_dim = config.patch_dim

    if target_hw is not None:
        target_h, target_w = target_hw
        img = img.resize((target_w, target_h))
        img_tensor = None
    elif config.preprocess_preset == "mistral-crawl-eval":
        img_tensor = _preprocess_mistral_crawl_eval(img, config)
    elif config.preprocess_preset != "checkpoint":
        raise ValueError(f"Unknown VLM image preprocessing preset: {config.preprocess_preset}")
    else:
        img = dynamic_res_preprocess(
            img,
            min_patches=config.dynamic_resolution_min_patches,
            max_patches=config.dynamic_resolution_max_patches,
            res_step=patch_dim,
            pixel_shuffle=config.pixel_shuffle,
            spatial_merge_size=config.spatial_merge_size,
        )
        img_tensor = None

    if img_tensor is None:
        vision_type = config.vision_model_type
        pixel_mean = config.pixel_mean
        pixel_std = config.pixel_std
        if pixel_mean is None or pixel_std is None:
            pixel_mean, pixel_std = _resolve_pixel_stats(vision_type)

        transform = T.Compose([T.ToTensor(), T.Normalize(mean=pixel_mean, std=pixel_std)])

        img_tensor = transform(img)  # [C, H, W]
    C, H, W = img_tensor.shape

    py, px = H // patch_dim, W // patch_dim
    patches = img_tensor.reshape(C, py, patch_dim, px, patch_dim)
    patches = patches.permute(1, 3, 0, 2, 4).contiguous()
    patches = patches.reshape(py * px, C * patch_dim * patch_dim)

    images = patches.unsqueeze(0)
    imgs_sizes = torch.tensor([[H, W]], dtype=torch.int32)

    if device is not None:
        return images.to(device), imgs_sizes
    return images, imgs_sizes


def preprocess_image_bytes_list(
    image_bytes_list, config: ImageProcessingConfig, device: Optional[torch.device] = None
) -> dict:
    """Preprocess a list of raw image bytes into engine.add_request image kwargs.

    Selects the dynamic-resolution or tiling path from the inference config.
    Each image is preprocessed independently so its aspect ratio is preserved.

    Args:
        image_bytes_list: List of raw image bytes (one entry per image).
        config: Image preprocessing configuration.
        device: Optional target device for the returned tensors. If None,
            tensors are returned on CPU and the caller is responsible for
            transfer.

    Returns:
        dict suitable for ``**kwargs`` to ``DynamicInferenceEngine.add_request``.
    """
    if not image_bytes_list:
        return {}

    dynamic_res = config.dynamic_resolution and not config.use_tiling

    if not dynamic_res:
        # Static tiling used to live here as ``preprocess_image_bytes_tiled``,
        # but it delegated to ``examples.multimodal.image_processing.ImageTransform``,
        # a bad dependency direction (``megatron/core`` importing from
        # ``examples/``). No in-tree caller currently hits this branch
        # (all supported encoders are on the dynamic-resolution path), so we
        # drop the tiling helper rather than move its 168-line dep into core.
        # Wire clients that need static tiling should preprocess bytes
        # themselves and submit a tensor payload
        # (``multi_modal_data['image'] = {'imgs': Tensor, 'num_tiles': Tensor,
        # 'num_img_embeddings_per_tile': int}``); the engine already accepts
        # that shape without touching examples.
        raise NotImplementedError(
            "Wire-side static-tiling preprocessing has moved out of "
            "``megatron/core``. Submit a preprocessed tensor payload as "
            "``multi_modal_data['image']`` "
            "({'imgs': Tensor, 'num_tiles': Tensor, "
            "'num_img_embeddings_per_tile': int}), or set "
            "``ImageProcessingConfig.dynamic_resolution=True`` to use the "
            "dynamic-resolution path that stays in-core."
        )

    # Preprocess each image independently so its aspect ratio is preserved.
    # Downstream (llava_model._preprocess_data / vision encoder pack) handles
    # per-image cu_seqlens, so ragged patch counts are fine.
    all_imgs, all_sizes = [], []
    for image_bytes in image_bytes_list:
        imgs, imgs_sizes = preprocess_image_bytes(image_bytes, config, device=device)
        all_imgs.append(imgs)
        all_sizes.append(imgs_sizes)
    imgs = torch.cat(all_imgs, dim=1) if len(all_imgs) > 1 else all_imgs[0]
    imgs_sizes = torch.cat(all_sizes, dim=0) if len(all_sizes) > 1 else all_sizes[0]
    return {"imgs": imgs, "imgs_sizes": imgs_sizes}
