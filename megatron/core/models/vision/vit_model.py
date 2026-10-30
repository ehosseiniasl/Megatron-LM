# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""
Native Megatron-Core Vision Transformer.

Provides ViTModel for Mistral vision encoders using mcore TransformerBlock,
with no HuggingFace dependency at runtime.

TODO: Optimize dynamic-resolution Mistral patch merging with a fused/custom
kernel. The current per-image merge path is simple but can create CPU syncs and
many small kernel launches.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.models.vision.vit_utils import dynamic_patch_grid_lists, run_vision_transformer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


class _TorchNorm(nn.Module):
    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float):
        super().__init__()
        if config.normalization == "RMSNorm":
            self.norm = nn.RMSNorm(hidden_size, eps=eps)
            self.norm.weight.is_rmsnorm_weight = True
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured torch normalization layer."""
        return self.norm(x)


try:
    from megatron.core.extensions.transformer_engine import HAVE_TE, TENorm

    _NORM_IMPL = TENorm if HAVE_TE else _TorchNorm
except Exception:
    _NORM_IMPL = _TorchNorm


class PatchEmbedding(nn.Module):
    """Conv2d patch extractor — no positional embedding."""

    def __init__(self, in_channels: int, hidden_size: int, patch_dim: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_dim, stride=patch_dim, bias=bias
        )
        self.patch_dim = patch_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """(B, C, H, W) → (B, N, hidden), (h_patches, w_patches)"""
        x = self.proj(x.to(self.proj.weight.dtype))
        h_patches, w_patches = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden)
        return x, h_patches, w_patches

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-patchified input -> (B, N, hidden)."""
        weight = self.proj.weight.flatten(1)
        return F.linear(x.to(weight.dtype), weight, self.proj.bias)


def _cat_rope(chunks: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-image RoPE tensors into packed-sequence order."""
    return torch.cat(chunks, dim=0)


class Mistral2DRotaryEmbedding(nn.Module):
    """
    2D RoPE for Mistral-native vision encoders.

    Mistral/vLLM applies RoPE as complex multiplication over adjacent hidden
    dimension pairs.  This returns repeat-interleaved angles and must be used
    with ``rotary_interleaved=True`` in TransformerConfig.
    """

    def __init__(self, head_dim: int, max_patches_per_side: int, rope_theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_patches_per_side = max_patches_per_side

        # freq_dim = head_dim // 4 (half for h, half for w in the head_dim//2 space).
        # Like LLM RoPE, keep frequencies outside module buffers so Float16Module
        # cannot cast them to BF16/FP16 before FP32 angle and sin/cos computation.
        self.freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    def forward(self, h_patches: int, w_patches: int, device: torch.device) -> torch.Tensor:
        """
        Returns freqs of shape (h_patches * w_patches, 1, 1, head_dim).
        This is the format mcore's apply_rotary_pos_emb expects for rotary_pos_emb.
        mcore handles cos/sin internally.
        """
        # Cache the device copy; module.to() does not move this constant tensor.
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device=device)
        base_freqs = self.freqs
        h = torch.arange(h_patches, device=device, dtype=base_freqs.dtype)
        w = torch.arange(w_patches, device=device, dtype=base_freqs.dtype)

        freqs_h = torch.outer(h, base_freqs[::2])  # (h_patches, head_dim//4)
        freqs_w = torch.outer(w, base_freqs[1::2])  # (w_patches, head_dim//4)

        angles = torch.cat(
            [
                freqs_h[:, None, :].expand(-1, w_patches, -1),
                freqs_w[None, :, :].expand(h_patches, -1, -1),
            ],
            dim=-1,
        ).reshape(-1, self.head_dim // 2)

        # MCore interleaved RoPE pairs adjacent dims, so each complex-pair angle
        # is repeated into the real and imaginary slots.
        freqs = angles.repeat_interleave(2, dim=-1)
        return freqs[:, None, None, :]  # (N, 1, 1, head_dim) — mcore rotary_pos_emb format


class MistralPatchMerger(nn.Module):
    """Spatial patch merger for Mistral vision encoders.

    Applies RMSNorm on pre-merge tokens, folds spatial blocks into a single
    merge_size**2 * hidden vector, then projects back to hidden_size with a
    bias-less Linear.
    Matches Mistral-Large-3's `pre_mm_projector_norm` + `patch_merger.merging_layer`.
    """

    def __init__(self, transformer_config: TransformerConfig, spatial_merge_size: int = 2):
        super().__init__()
        if spatial_merge_size < 1:
            raise ValueError(f"spatial_merge_size must be >= 1, got {spatial_merge_size}")
        self.spatial_merge_size = spatial_merge_size
        hidden = transformer_config.hidden_size
        self.pre_norm = _NORM_IMPL(
            config=transformer_config, hidden_size=hidden, eps=transformer_config.layernorm_epsilon
        )
        self.linear_fc1 = nn.Linear(hidden * spatial_merge_size**2, hidden, bias=False)

    def forward(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """(B, h_p*w_p, H) → (B, h_p/m*w_p/m, H)."""
        m = self.spatial_merge_size
        if h_patches % m != 0 or w_patches % m != 0:
            raise ValueError(
                "Patch grid must be divisible by spatial_merge_size, got "
                f"{h_patches}x{w_patches} and {m}"
            )
        B, _, H = x.shape
        h_out, w_out = h_patches // m, w_patches // m
        x = self.pre_norm(x)
        # Match vLLM PatchMerger/unfold order: each 2x2 block is flattened as
        # (hidden, merge_h, merge_w), not patch-major (merge_h, merge_w, hidden).
        x = x.reshape(B, h_out, m, w_out, m, H)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()  # (B, h_out, w_out, H, m, m)
        x = x.reshape(B, h_out * w_out, H * m * m)  # (B, N_out, 4H)
        return self.linear_fc1(x)  # (B, N_out, H)


class ViTModel(VisionModule):
    """
    Generic Vision Transformer — native mcore, no transformers dependency.

    Supports:
      - Mistral ViT (2D RoPE, SwiGLU, RMSNorm, no CLS, no bias)
      - Mistral ViT with a 2×2 patch merger after the transformer stack
      - CLIP/SigLIP (learned absolute pos, GELU, LayerNorm, CLS token)
        when add_class_token=True and pos_emb_type='learned_absolute'

    Args:
        transformer_config: Standard mcore TransformerConfig. Configure:
            - normalization, norm_epsilon
            - gated_linear_unit, activation_func
            - add_bias_linear
            - num_layers, hidden_size, num_attention_heads, ffn_hidden_size
        transformer_layer_spec: Layer spec for TransformerBlock (bidirectional).
        patch_dim: Patch size in pixels.
        img_h / img_w: Maximum image dimensions used to size the RoPE table.
            Dynamic-resolution images may be smaller and can differ within a
            packed batch.
        add_class_token: Prepend a learnable CLS token (CLIP style).
        class_token_len: Width of CLS token.
        ln_pre: Apply RMSNorm/LayerNorm before the transformer stack.
        ln_pre_eps: Epsilon for pre-transformer norm (defaults to norm_epsilon).
        pos_emb_type: 'rope2d' | 'learned_absolute' | 'none'.
        rope_theta: RoPE base frequency.
        use_merger: If True, append a Mistral-style spatial patch merger that
            keeps the output width at hidden_size.
        spatial_merge_size: Merger block size (default 2 means 2×2, a 4× token reduction).
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        patch_dim: int = 16,
        img_h: int = 1024,
        img_w: int = 1024,
        in_channels: int = 3,
        patch_embed_bias: bool = False,
        add_class_token: bool = False,
        class_token_len: int = 0,
        ln_pre: bool = True,
        ln_pre_eps: Optional[float] = None,
        pos_emb_type: str = 'rope2d',
        rope_theta: float = 10000.0,
        use_merger: bool = False,
        spatial_merge_size: int = 2,
        pg_collection=None,
        vp_stage: Optional[int] = None,
        name: str | None = None,
    ):
        super().__init__(config=transformer_config)

        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w
        self.add_class_token = add_class_token
        self.class_token_len = class_token_len if add_class_token else 0
        self.pos_emb_type = pos_emb_type
        hidden_size = transformer_config.hidden_size

        # Patch embedding (state dict: patch_embed.proj.*)
        self.patch_embed = PatchEmbedding(
            in_channels, hidden_size, patch_dim, bias=patch_embed_bias
        )

        # Pre-transformer norm (matches Mistral's ln_pre and CLIP's ln_pre)
        if ln_pre:
            # TENorm auto-dispatches to RMSNorm or LayerNorm based on config.normalization
            self.ln_pre = _NORM_IMPL(
                config=transformer_config,
                hidden_size=hidden_size,
                eps=ln_pre_eps if ln_pre_eps is not None else transformer_config.layernorm_epsilon,
            )
        else:
            self.ln_pre = None

        # CLS token (CLIP style)
        if add_class_token:
            self.class_token = nn.Parameter(torch.zeros(1, class_token_len, hidden_size))

        # Positional embedding
        max_patches = (img_h // patch_dim) * (img_w // patch_dim)
        if pos_emb_type == 'learned_absolute':
            self.position_embeddings = nn.Embedding(max_patches + self.class_token_len, hidden_size)
        elif pos_emb_type == 'rope2d':
            head_dim = hidden_size // transformer_config.num_attention_heads
            max_patches_per_side = max(img_h, img_w) // patch_dim
            self.rope = Mistral2DRotaryEmbedding(head_dim, max_patches_per_side, rope_theta)
        # 'none': no positional encoding

        # Transformer
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            name=(name + ".decoder") if name is not None else None,
        )

        # Optional Mistral 2×2 patch merger after the transformer stack.
        # Keeps output width at hidden_size; sequence length shrinks by merge_size**2.
        self.spatial_merge_size = spatial_merge_size if use_merger else 1
        if use_merger:
            self.merger = MistralPatchMerger(transformer_config, spatial_merge_size)
        else:
            self.merger = None
        # Read by llava_model to size the vision projection input.
        self.out_hidden_size = hidden_size

    @property
    def num_patches_per_image(self) -> int:
        """Number of patches produced by the fixed-resolution path."""
        return (self.img_h // self.patch_dim) * (self.img_w // self.patch_dim)

    def set_input_tensor(self, input_tensor):
        """Pass pipeline-parallel input tensors to the transformer block."""
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        imgs_sizes=None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: Fixed-resolution images as (B, C, H, W), where every
                image in the tensor shares H/W, or packed dynamic-resolution
                patches as (1, total_patches, C * patch_dim * patch_dim).
            attention_mask: optional mask for TransformerBlock
            imgs_sizes: Per-image (H, W) values for packed dynamic-resolution
                inputs. Images in this path may have different resolutions.

        Returns:
            (B, N, hidden_size)  where N includes CLS tokens if add_class_token=True
        """
        dynamic_resolution = pixel_values.dim() == 3
        if dynamic_resolution:
            assert imgs_sizes is not None, "imgs_sizes is required for dynamic-resolution ViTModel"
            assert not self.add_class_token, "dynamic-resolution CLS handling is not implemented"
            patch_hw, seq_lens = dynamic_patch_grid_lists(imgs_sizes, self.patch_dim)
            x = self.patch_embed.forward_patches(pixel_values)
        else:
            # Fixed-resolution path is kept for standalone encoder parity checks.
            # The VLM training path should pass packed dynamic-resolution images.
            B = pixel_values.shape[0]
            x, h_patches, w_patches = self.patch_embed(pixel_values)

        # 2. Pre-transformer norm
        if self.ln_pre is not None:
            x = self.ln_pre(x)

        # 3. Positional embedding
        rotary_pos_emb = None
        if self.pos_emb_type == 'learned_absolute':
            assert not dynamic_resolution, "learned absolute ViT positions are fixed-size only"
            pos = torch.arange(x.shape[1], device=pixel_values.device)
            if self.add_class_token:
                pos = pos + self.class_token_len
            x = x + self.position_embeddings(pos)
        elif self.pos_emb_type == 'rope2d':
            # Shape: (N, 1, 1, head_dim//2) — mcore rotary_pos_emb format
            if dynamic_resolution:
                rotary_pos_emb = _cat_rope(
                    [self.rope(h, w, pixel_values.device) for h, w in patch_hw]
                )
            else:
                rotary_pos_emb = self.rope(h_patches, w_patches, pixel_values.device)

        # 4. CLS token
        if self.add_class_token:
            cls = self.class_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        # 5. TransformerBlock: expects (S, B, hidden)
        x = run_vision_transformer(
            self.decoder,
            x,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        # 6. Optional 2×2 patch merger. Merger only runs on the
        #    patch tokens — CLS tokens (when present) would need separate handling,
        #    but Mistral ViT configs have no CLS tokens so we skip that branch.
        if self.merger is not None:
            if dynamic_resolution:
                chunks = torch.split(x, seq_lens, dim=1)
                x = torch.cat(
                    [self.merger(chunk, int(h), int(w)) for chunk, (h, w) in zip(chunks, patch_hw)],
                    dim=1,
                )
            else:
                x = self.merger(x, h_patches, w_patches)

        return x
