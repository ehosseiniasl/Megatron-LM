# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for image-token expansion in :class:`VLMInferenceWrapper`.

These exercise ``expand_image_tokens`` against a stub model module, so they
cover the native image-token layout without building a real LLaVA model.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)

IMAGE_TOKEN = 100
IMAGE_BREAK_TOKEN = 101
IMAGE_END_TOKEN = 102


def _module(
    dynamic_resolution=True,
    pixel_shuffle=False,
    spatial_merge_size=1,
    patch_dim=2,
    class_token_len=0,
    drop_vision_class_token=True,
    use_native_image_tokens=False,
    image_end_token_index=None,
    image_break_token_index=None,
):
    module = SimpleNamespace(
        image_token_index=IMAGE_TOKEN,
        dynamic_resolution=dynamic_resolution,
        _patch_dim=patch_dim,
        _pixel_shuffle=pixel_shuffle,
        _spatial_merge_size=spatial_merge_size,
        _class_token_len=class_token_len,
        _drop_vision_class_token=drop_vision_class_token,
        img_seq_len=4,
    )
    if use_native_image_tokens:
        module.use_native_image_tokens = True
    if image_end_token_index is not None:
        module.image_end_token_index = image_end_token_index
    if image_break_token_index is not None:
        module.image_break_token_index = image_break_token_index
    return module


def _expand(module, tokens, imgs_sizes=None, num_tiles=None):
    # ``expand_image_tokens`` only reaches through ``self.model``; build the
    # minimal stub the accessor expects rather than a full wrapper.
    stub_self = SimpleNamespace(model=SimpleNamespace(module=module))
    return VLMInferenceWrapper.expand_image_tokens(stub_self, tokens, num_tiles, imgs_sizes)


def test_dynamic_resolution_expansion_is_reachable():
    """The gate used to read a ``_dynamic_resolution`` attribute LLaVAModel never
    set, so every dynamic-resolution request raised NotImplementedError."""
    # 4x4 image, patch_dim 2 -> 2x2 = 4 embeddings.
    imgs_sizes = torch.tensor([[4, 4]])
    tokens = [[1, IMAGE_TOKEN, 2]]

    expanded, mask = _expand(_module(), tokens, imgs_sizes=imgs_sizes)

    assert expanded == [[1, -1, -1, -1, -1, 2]]
    assert mask == [[None, 0, 1, 2, 3, None]]


def test_dynamic_resolution_rejected_without_model_support():
    imgs_sizes = torch.tensor([[4, 4]])

    with pytest.raises(NotImplementedError, match="dynamic_resolution=True"):
        _expand(_module(dynamic_resolution=False), [[IMAGE_TOKEN]], imgs_sizes=imgs_sizes)


def test_native_image_tokens_interleave_break_and_end():
    """Each patch row is terminated by image-break, the last row by image-end."""
    # 4x4 image, patch_dim 2 -> 2 rows x 2 columns.
    imgs_sizes = torch.tensor([[4, 4]])
    tokens = [[1, IMAGE_TOKEN, 2]]
    module = _module(
        use_native_image_tokens=True,
        image_end_token_index=IMAGE_END_TOKEN,
        image_break_token_index=IMAGE_BREAK_TOKEN,
    )

    expanded, mask = _expand(module, tokens, imgs_sizes=imgs_sizes)

    assert expanded == [[1, -1, -1, IMAGE_BREAK_TOKEN, -1, -1, IMAGE_END_TOKEN, 2]]
    # Break/end tokens are real vocabulary entries, so they carry no embedding index.
    assert mask == [[None, 0, 1, None, 2, 3, None, None]]


def test_native_image_end_only_appends_single_terminator():
    """Without an image-break token the run is flat, terminated by image-end."""
    imgs_sizes = torch.tensor([[4, 4]])
    module = _module(use_native_image_tokens=True, image_end_token_index=IMAGE_END_TOKEN)

    expanded, mask = _expand(module, [[IMAGE_TOKEN]], imgs_sizes=imgs_sizes)

    assert expanded == [[-1, -1, -1, -1, IMAGE_END_TOKEN]]
    assert mask == [[0, 1, 2, 3, None]]


def test_native_image_tokens_reject_layout_mismatch():
    """A class token makes the embedding count exceed rows*columns."""
    imgs_sizes = torch.tensor([[4, 4]])
    module = _module(
        drop_vision_class_token=False,
        class_token_len=1,
        use_native_image_tokens=True,
        image_end_token_index=IMAGE_END_TOKEN,
        image_break_token_index=IMAGE_BREAK_TOKEN,
    )

    with pytest.raises(ValueError, match="Native image-token layout does not match"):
        _expand(module, [[IMAGE_TOKEN]], imgs_sizes=imgs_sizes)


def test_native_image_tokens_multiple_images_offset_correctly():
    # Two 4x4 images -> 4 embeddings each, second image's mask starts at 4.
    imgs_sizes = torch.tensor([[4, 4], [4, 4]])
    tokens = [[IMAGE_TOKEN, 5, IMAGE_TOKEN]]
    module = _module(
        use_native_image_tokens=True,
        image_end_token_index=IMAGE_END_TOKEN,
        image_break_token_index=IMAGE_BREAK_TOKEN,
    )

    expanded, mask = _expand(module, tokens, imgs_sizes=imgs_sizes)

    assert expanded == [
        [
            -1,
            -1,
            IMAGE_BREAK_TOKEN,
            -1,
            -1,
            IMAGE_END_TOKEN,
            5,
            -1,
            -1,
            IMAGE_BREAK_TOKEN,
            -1,
            -1,
            IMAGE_END_TOKEN,
        ]
    ]
    assert mask == [[0, 1, None, 2, 3, None, None, 4, 5, None, 6, 7, None]]


def test_spatial_merge_size_halves_the_layout():
    # 8x8 image, patch_dim 2 -> 4x4 patches; spatial_merge_size 2 -> 2x2 layout,
    # 16 // 4 = 4 embeddings.
    imgs_sizes = torch.tensor([[8, 8]])
    module = _module(
        spatial_merge_size=2,
        use_native_image_tokens=True,
        image_end_token_index=IMAGE_END_TOKEN,
        image_break_token_index=IMAGE_BREAK_TOKEN,
    )

    expanded, mask = _expand(module, [[IMAGE_TOKEN]], imgs_sizes=imgs_sizes)

    assert expanded == [[-1, -1, IMAGE_BREAK_TOKEN, -1, -1, IMAGE_END_TOKEN]]
    assert mask == [[0, 1, None, 2, 3, None]]


def test_non_native_expansion_is_unchanged():
    """Models without native tokens keep the plain pad-run expansion."""
    imgs_sizes = torch.tensor([[4, 4]])

    expanded, mask = _expand(_module(), [[IMAGE_TOKEN, 7]], imgs_sizes=imgs_sizes)

    assert expanded == [[-1, -1, -1, -1, 7]]
    assert mask == [[0, 1, 2, 3, None]]
