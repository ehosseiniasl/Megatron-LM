# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional, Tuple, Union

import torch

from megatron.core.packed_seq_params import PackedSeqParams


def dynamic_patch_grid(
    imgs_sizes: Union[List[Tuple[int, int]], torch.Tensor], patch_dim: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-image patch grids and sequence lengths for dynamic image sizes."""
    if torch.is_tensor(imgs_sizes):
        sizes = imgs_sizes.to(device=device, dtype=torch.int64)
    else:
        sizes = torch.tensor(imgs_sizes, device=device, dtype=torch.int64)
    patch_hw = sizes // patch_dim
    seq_lens = patch_hw[:, 0] * patch_hw[:, 1]
    return patch_hw, seq_lens


def dynamic_patch_grid_lists(
    imgs_sizes: Union[List[Tuple[int, int]], torch.Tensor], patch_dim: int
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Return dynamic patch metadata without synchronizing CUDA to the host."""
    if torch.is_tensor(imgs_sizes) and imgs_sizes.is_cuda:
        raise ValueError("imgs_sizes must remain on CPU to avoid CUDA-to-host synchronization")
    patch_hw = [
        (height // patch_dim, width // patch_dim) for height, width in image_size_tuples(imgs_sizes)
    ]
    return patch_hw, [height * width for height, width in patch_hw]


def image_size_tuples(
    imgs_sizes: Union[List[Tuple[int, int]], torch.Tensor],
) -> List[Tuple[int, int]]:
    """Convert tensor or list image sizes to Python ``(height, width)`` tuples."""
    if torch.is_tensor(imgs_sizes):
        return [tuple(int(v) for v in size.tolist()) for size in imgs_sizes]
    return [(int(h), int(w)) for h, w in imgs_sizes]


def run_vision_transformer(
    decoder: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    rotary_pos_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run an MCore transformer block on batch-first vision hidden states."""
    hidden_states = hidden_states.transpose(0, 1).contiguous()
    hidden_states = decoder(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
    )
    return hidden_states.transpose(0, 1).contiguous()
