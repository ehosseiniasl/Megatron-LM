#!/usr/bin/env /usr/bin/python3
"""Compare this branch's Mistral and RADIO encoders against exported goldens.

The goldens are produced by the pre-vlm-05 vLLM/HF exporter at the
projector-independent ``features["encoder_tokens"]`` boundary.  This checker
keeps only the encoders implemented in this branch:

* ``mistral``: Mistral 410M vision tokens after native 2x2 patch merging.
* ``c_radio_vit_h``: standalone C-RADIO patch tokens after class-token drop.
* ``post-c-radio-omni``: Nemotron Omni RADIO tokens after dynamic pixel shuffle.

Custom goldens may be checked with ``--encoder NAME --golden PATH``.  The file
must be a ``torch.save`` payload with ``inputs`` and ``features`` dictionaries.
Only ``features["encoder_tokens"]`` is required for pass/fail; extra feature
keys such as ``vision_raw``, ``radio_patch_features``, or ``pixel_shuffled`` are
compared as diagnostics when present.
"""

import argparse
import gc
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from megatron.core import parallel_state
from megatron.core.activations import fast_gelu
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.multimodal.llava_model import pixel_shuffle
from megatron.core.models.vision import vit_model as vit_module
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.models.vision.vit_model import ViTModel
from megatron.core.models.vision.vit_utils import image_size_tuples, run_vision_transformer
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.typed_torch import not_none

DATA_ROOT = Path("/home/sasatheesh/data")
GOLDEN_DIR = (
    DATA_ROOT
    / "checkpoints"
    / "encoder_goldens"
    / "model_leaderboard_spreadsheet_vllm_processor_20260504"
)
CKPT_ROOT = DATA_ROOT / "checkpoints" / "encoders_v2"
MODEL_DTYPE = torch.bfloat16

DEFAULT_ENCODERS = ("c_radio_vit_h", "post-c-radio-omni")

ENCODER_CASES = {
    "mistral": ("mistral-vit", "vision_encoder_410M_mcore", "mistral.pt"),
    "c_radio_vit_h": ("radio", "c_radio_vit_h", "c_radio_vit_h_standalone.pt"),
    "post-c-radio-omni": ("radio", "post-c-radio-omni", "c_radio_vit_h.pt"),
}

LEGACY_GOLDEN_FILES = {
    # Existing exporter/checkpoint asset name for this Mistral variant.
    "mistral": "mistral_410m.pt",
}

LEGACY_CHECKPOINT_NAMES = {}

REQUIRED_INPUT_KEYS = {
    "mistral": ("images",),
    "c_radio_vit_h": ("pixel_values_flat", "imgs_sizes"),
    "post-c-radio-omni": ("pixel_values_flat", "imgs_sizes"),
}

ENCODER_TOKEN_WIDTH = {
    "mistral": 1024,
    "c_radio_vit_h": 1280,
    "post-c-radio-omni": 5120,
}

MISTRAL_IMAGE_SPECS = {
    "mistral": {
        "patch_dim": 14,
        "img_h": 1540,
        "img_w": 1540,
        "spatial_merge_size": 2,
    },
}

GOLDEN_ENCODER_ALIASES = {
    "mistral": (
        "mistral",
        "mistral-vit",
        "mistral_410m",
        "mistral-vit-410m",
        "pixtral_410m",
        "pixtral-vit-410m",
    ),
    "c_radio_vit_h": ("c_radio_vit_h", "radio"),
    "post-c-radio-omni": ("post-c-radio-omni", "c_radio_vit_h", "radio"),
}

GOLDEN_SCHEMA = """\
Golden files are torch.save payloads with this shape:

{
    "encoder": "mistral" | "c_radio_vit_h" | "post-c-radio-omni",
    "inputs": {
        # mistral:
        "images": FloatTensor[1, 3, H, W],

        # c_radio_vit_h and post-c-radio-omni:
        "pixel_values_flat": BFloat16Tensor[1, num_patches, 768],
        "imgs_sizes": LongTensor[1, 2],  # [[height_px, width_px]]
    },
    "features": {
        # Required. This is the pass/fail comparison boundary.
        "encoder_tokens": FloatTensor[num_tokens, hidden],

        # Optional diagnostics, compared when present:
        "vision_raw": FloatTensor[raw_tokens, hidden],               # Mistral
        "radio_patch_features": FloatTensor[1, raw_tokens, 1280],   # RADIO
        "pixel_shuffled": FloatTensor[1, num_tokens, 5120],          # post-c-radio-omni
    },
}

Shape rules:
- mistral: H and W must be divisible by 14, and H/14 and W/14 must be even.
  encoder_tokens shape is [(H/14)*(W/14)/4, 1024].
- c_radio_vit_h: num_patches must equal (height_px/16)*(width_px/16).
  encoder_tokens shape is [num_patches, 1280].
- post-c-radio-omni: num_patches must equal (height_px/16)*(width_px/16), and
  height_px/16 and width_px/16 must be even for 2x pixel shuffle.
  encoder_tokens shape is [num_patches/4, 5120].
"""

CASE_NOTES = {
    "mistral": (
        "Compares Mistral 410M vision after pre-transformer norm, transformer, and native "
        "2x2 patch merger. Missing rope.freqs is expected because it is a "
        "deterministic non-persistent buffer."
    ),
    "c_radio_vit_h": (
        "Standalone nvidia/C-RADIOv4-H comparison at class-token-dropped RADIO patch "
        "features. This path has no pixel shuffle or conv merging."
    ),
    "post-c-radio-omni": (
        "Nemotron Omni embedded RADIO comparison after class-token drop and dynamic "
        "pixel shuffle, before the Omni mlp1 projector."
    ),
}


def init_single_rank(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        port = 29600 + (os.getpid() % 1000)
        torch.distributed.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=1,
        )
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    if device.type == "cuda":
        model_parallel_cuda_manual_seed(1234, force_reset_rng=True)


def local_vit_layer_spec() -> ModuleSpec:
    mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        ),
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=not_none(WrappedTorchNorm),
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=not_none(WrappedTorchNorm),
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def base_config(
    *,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    ffn_hidden_size: int,
    kv_channels: int,
) -> TransformerConfig:
    return TransformerConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_attention_heads,
        ffn_hidden_size=ffn_hidden_size,
        kv_channels=kv_channels,
        params_dtype=MODEL_DTYPE,
        bf16=True,
        perform_initialization=False,
        transformer_impl="local",
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


def mistral_config(name: str) -> TransformerConfig:
    if name == "mistral":
        params = {
            "num_layers": 24,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "ffn_hidden_size": 4096,
            "kv_channels": 64,
        }
    else:
        raise ValueError(f"unknown Mistral encoder case {name!r}")

    config = base_config(
        num_layers=params["num_layers"],
        hidden_size=params["hidden_size"],
        num_attention_heads=params["num_attention_heads"],
        ffn_hidden_size=params["ffn_hidden_size"],
        kv_channels=params["kv_channels"],
    )
    config.normalization = "RMSNorm"
    config.layernorm_epsilon = 1e-5
    config.activation_func = torch.nn.functional.silu
    config.gated_linear_unit = True
    config.add_bias_linear = False
    config.add_qkv_bias = False
    config.rotary_interleaved = True
    config.layernorm_zero_centered_gamma = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.apply_rope_fusion = False
    return config


def radio_config() -> TransformerConfig:
    config = base_config(
        num_layers=32,
        hidden_size=1280,
        num_attention_heads=16,
        ffn_hidden_size=5120,
        kv_channels=80,
    )
    config.normalization = "LayerNorm"
    config.layernorm_epsilon = 1e-6
    config.activation_func = fast_gelu
    config.qk_layernorm = False
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.gated_linear_unit = False
    config.layernorm_zero_centered_gamma = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.apply_rope_fusion = False
    return config


def build_mistral(name: str) -> torch.nn.Module:
    config = mistral_config(name)
    image_spec = MISTRAL_IMAGE_SPECS[name]
    vit_module._NORM_IMPL = WrappedTorchNorm
    return ViTModel(
        transformer_config=config,
        transformer_layer_spec=local_vit_layer_spec(),
        patch_dim=image_spec["patch_dim"],
        img_h=image_spec["img_h"],
        img_w=image_spec["img_w"],
        pos_emb_type="rope2d",
        ln_pre=True,
        use_merger=image_spec["spatial_merge_size"] > 1,
        spatial_merge_size=image_spec["spatial_merge_size"],
    )


def build_mistral_model() -> torch.nn.Module:
    return build_mistral("mistral")


def build_radio() -> torch.nn.Module:
    config = radio_config()
    return RADIOViTModel(
        transformer_config=config,
        transformer_layer_spec=local_vit_layer_spec(),
        img_h=512,
        img_w=512,
        max_img_h=2048,
        max_img_w=2048,
        class_token_len=10,
        patch_dim=16,
        add_class_token=True,
        dynamic_resolution=True,
        force_eval_mode=True,
    )


def remap_te_norm_key(key: str) -> str:
    key = re.sub(
        r"(decoder\.layers\.\d+)\.self_attention\.linear_qkv\.layer_norm_weight$",
        r"\1.input_layernorm.weight",
        key,
    )
    key = re.sub(
        r"(decoder\.layers\.\d+)\.self_attention\.linear_qkv\.layer_norm_bias$",
        r"\1.input_layernorm.bias",
        key,
    )
    key = re.sub(
        r"(decoder\.layers\.\d+)\.mlp\.linear_fc1\.layer_norm_weight$",
        r"\1.pre_mlp_layernorm.weight",
        key,
    )
    key = re.sub(
        r"(decoder\.layers\.\d+)\.mlp\.linear_fc1\.layer_norm_bias$",
        r"\1.pre_mlp_layernorm.bias",
        key,
    )
    return key


def load_vision_dcp(model: torch.nn.Module, ckpt_dir: Path) -> dict:
    reader = FileSystemReader(str(ckpt_dir))
    metadata = reader.read_metadata().state_dict_metadata
    prefix = "model.vision_model."
    load_sd = {
        key: torch.empty(meta.size, dtype=meta.properties.dtype)
        for key, meta in metadata.items()
        if key.startswith(prefix) and isinstance(meta, TensorStorageMetadata)
    }
    dcp.load(load_sd, storage_reader=reader)

    tensors = dict(model.named_parameters())
    tensors.update(dict(model.named_buffers()))

    loaded = []
    skipped = []
    for ckpt_key, tensor in load_sd.items():
        rel_key = remap_te_norm_key(ckpt_key.removeprefix(prefix))
        target = tensors.get(rel_key)
        if target is None:
            skipped.append(rel_key)
            continue
        target.data.copy_(tensor.to(dtype=target.dtype))
        loaded.append(rel_key)

    missing = sorted(set(tensors) - set(loaded))
    return {
        "loaded": len(loaded),
        "num_checkpoint_tensors": len(load_sd),
        "num_model_tensors": len(tensors),
        "missing": missing,
        "skipped": sorted(skipped),
    }


def tensor_desc(value) -> str:
    if not torch.is_tensor(value):
        return type(value).__name__
    return f"Tensor{tuple(value.shape)} {value.dtype}"


def require_tensor(container: dict, key: str, where: str, golden_path: Path) -> torch.Tensor:
    if key not in container:
        raise ValueError(f"{golden_path}: missing {where}[{key!r}]")
    value = container[key]
    if not torch.is_tensor(value):
        raise ValueError(f"{golden_path}: {where}[{key!r}] must be a tensor, got {type(value)}")
    return value


def require_encoder_tokens(
    data: dict,
    name: str,
    golden_path: Path,
    *,
    expected_tokens: int | None = None,
) -> torch.Tensor:
    features = data["features"]
    encoder_tokens = require_tensor(features, "encoder_tokens", "features", golden_path)
    expected_width = ENCODER_TOKEN_WIDTH[name]
    if encoder_tokens.ndim != 2 or encoder_tokens.shape[-1] != expected_width:
        raise ValueError(
            f"{golden_path}: features['encoder_tokens'] must have shape "
            f"[num_tokens, {expected_width}] for {name}, got {tensor_desc(encoder_tokens)}"
        )
    if expected_tokens is not None and encoder_tokens.shape[0] != expected_tokens:
        raise ValueError(
            f"{golden_path}: features['encoder_tokens'] must have {expected_tokens} tokens "
            f"for {name}, got {encoder_tokens.shape[0]}"
        )
    return encoder_tokens


def validate_mistral_golden(data: dict, name: str, golden_path: Path) -> None:
    image_spec = MISTRAL_IMAGE_SPECS[name]
    patch_dim = image_spec["patch_dim"]
    spatial_merge_size = image_spec["spatial_merge_size"]
    images = require_tensor(data["inputs"], "images", "inputs", golden_path)
    if images.ndim != 4 or images.shape[0] != 1 or images.shape[1] != 3:
        raise ValueError(
            f"{golden_path}: inputs['images'] must have shape [1, 3, H, W], "
            f"got {tensor_desc(images)}"
        )
    h, w = int(images.shape[-2]), int(images.shape[-1])
    if h % patch_dim != 0 or w % patch_dim != 0:
        raise ValueError(
            f"{golden_path}: Mistral image H,W must be divisible by {patch_dim}, got {h},{w}"
        )
    h_patches, w_patches = h // patch_dim, w // patch_dim
    if spatial_merge_size > 1 and (
        h_patches % spatial_merge_size != 0 or w_patches % spatial_merge_size != 0
    ):
        raise ValueError(
            f"{golden_path}: Mistral patch grid must be divisible by "
            f"spatial_merge_size={spatial_merge_size}, got {h_patches}x{w_patches}"
        )
    expected_tokens = h_patches * w_patches
    expected_tokens //= spatial_merge_size * spatial_merge_size
    require_encoder_tokens(
        data,
        name,
        golden_path,
        expected_tokens=expected_tokens,
    )


def validate_mistral_model_golden(data: dict, golden_path: Path) -> None:
    validate_mistral_golden(data, "mistral", golden_path)


def radio_patch_grid(data: dict, name: str, golden_path: Path) -> tuple[int, int, int]:
    pixel_values = require_tensor(data["inputs"], "pixel_values_flat", "inputs", golden_path)
    imgs_sizes = require_tensor(data["inputs"], "imgs_sizes", "inputs", golden_path)
    if pixel_values.ndim != 3 or pixel_values.shape[0] != 1 or pixel_values.shape[-1] != 768:
        raise ValueError(
            f"{golden_path}: inputs['pixel_values_flat'] must have shape "
            f"[1, num_patches, 768] for {name}, got {tensor_desc(pixel_values)}"
        )
    if imgs_sizes.shape != (1, 2):
        raise ValueError(
            f"{golden_path}: inputs['imgs_sizes'] must have shape [1, 2], "
            f"got {tensor_desc(imgs_sizes)}"
        )
    h_pixels, w_pixels = [int(v) for v in imgs_sizes[0].tolist()]
    if h_pixels % 16 != 0 or w_pixels % 16 != 0:
        raise ValueError(
            f"{golden_path}: RADIO imgs_sizes must be divisible by 16, got {h_pixels},{w_pixels}"
        )
    h_patches, w_patches = h_pixels // 16, w_pixels // 16
    expected_patches = h_patches * w_patches
    if pixel_values.shape[1] != expected_patches:
        raise ValueError(
            f"{golden_path}: inputs['pixel_values_flat'] has {pixel_values.shape[1]} patches, "
            f"but imgs_sizes imply {expected_patches}"
        )
    return h_patches, w_patches, expected_patches


def validate_radio_standalone_golden(data: dict, golden_path: Path) -> None:
    _, _, expected_patches = radio_patch_grid(data, "c_radio_vit_h", golden_path)
    require_encoder_tokens(
        data,
        "c_radio_vit_h",
        golden_path,
        expected_tokens=expected_patches,
    )


def validate_radio_omni_golden(data: dict, golden_path: Path) -> None:
    h_patches, w_patches, expected_patches = radio_patch_grid(
        data,
        "post-c-radio-omni",
        golden_path,
    )
    if h_patches % 2 != 0 or w_patches % 2 != 0:
        raise ValueError(
            f"{golden_path}: post-c-radio-omni patch grid must be even for 2x pixel shuffle, "
            f"got {h_patches}x{w_patches}"
        )
    require_encoder_tokens(
        data,
        "post-c-radio-omni",
        golden_path,
        expected_tokens=expected_patches // 4,
    )


VALIDATORS = {
    "mistral": validate_mistral_model_golden,
    "c_radio_vit_h": validate_radio_standalone_golden,
    "post-c-radio-omni": validate_radio_omni_golden,
}


def validate_golden(data: dict, name: str, golden_path: Path) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{golden_path}: golden must be a dict, got {type(data)}")
    for group in ("inputs", "features"):
        if group not in data or not isinstance(data[group], dict):
            raise ValueError(f"{golden_path}: golden must contain a dict at {group!r}")
    missing = sorted(set(REQUIRED_INPUT_KEYS[name]) - set(data["inputs"]))
    if missing:
        raise ValueError(f"{golden_path}: missing required inputs for {name}: {missing}")
    declared_encoder = data.get("encoder")
    if declared_encoder is not None and declared_encoder not in GOLDEN_ENCODER_ALIASES[name]:
        raise ValueError(
            f"{golden_path}: payload encoder {declared_encoder!r} does not match requested {name!r}"
        )
    VALIDATORS[name](data, golden_path)


def golden(path: Path, name: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    validate_golden(data, name, path)
    return data


def token_cosine_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if actual.ndim < 2 or actual.shape[-1] == 0:
        return {}
    actual_tokens = actual.reshape(-1, actual.shape[-1])
    expected_tokens = expected.reshape(-1, expected.shape[-1])
    per_token = F.cosine_similarity(actual_tokens, expected_tokens, dim=-1)
    return {
        "token_cosine_min": float(per_token.min().item()),
        "token_cosine_p01": float(torch.quantile(per_token, 0.01).item()),
        "token_cosine_p50": float(torch.quantile(per_token, 0.50).item()),
        "token_cosine_p99": float(torch.quantile(per_token, 0.99).item()),
        "token_cosine_mean": float(per_token.mean().item()),
    }


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    out = {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "shape_match": list(actual.shape) == list(expected.shape),
    }
    if not out["shape_match"]:
        return out
    diff = actual - expected
    out.update(
        {
            "max_abs": float(diff.abs().max().item()),
            "mean_abs": float(diff.abs().mean().item()),
            "rms": float(diff.pow(2).mean().sqrt().item()),
            "cosine": float(F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()),
            "actual_first8": actual.flatten()[:8].tolist(),
            "expected_first8": expected.flatten()[:8].tolist(),
        }
    )
    out.update(token_cosine_stats(actual, expected))
    return out


def run_mistral(model: ViTModel, data: dict, device: torch.device) -> dict[str, torch.Tensor]:
    images = data["inputs"]["images"].to(device=device, dtype=MODEL_DTYPE)
    x, h_patches, w_patches = model.patch_embed(images)
    if model.ln_pre is not None:
        x = model.ln_pre(x)
    rotary_pos_emb = model.rope(h_patches, w_patches, device)
    x = model.decoder(
        hidden_states=x.transpose(0, 1).contiguous(),
        attention_mask=None,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=None,
    ).transpose(0, 1).contiguous()
    encoder_tokens = model.merger(x, h_patches, w_patches) if model.merger is not None else x
    return {
        "vision_raw": x.squeeze(0),
        "encoder_tokens": encoder_tokens.squeeze(0),
    }


def run_radio_single_sequence(
    model: RADIOViTModel,
    pixel_values: torch.Tensor,
    imgs_sizes: torch.Tensor,
) -> torch.Tensor:
    sizes = image_size_tuples(imgs_sizes)
    if len(sizes) != 1:
        raise ValueError("RADIO parity checker currently covers one image at a time")

    seq_len = (sizes[0][0] // model.patch_dim) * (sizes[0][1] // model.patch_dim)
    x, _ = model.embedder(pixel_values)
    if x.shape[1] != seq_len:
        raise ValueError(f"RADIO patch count mismatch: {x.shape[1]} != {seq_len}")

    x = model.apply_pos_enc(x, input_size=sizes[0])[0]
    if model.add_class_token:
        class_token = model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)
    if model.ln_pre:
        x = model.ln_pre(x)
    x = run_vision_transformer(model.decoder, x, attention_mask=None, packed_seq_params=None)
    if model.ln_post:
        x = model.ln_post(x)
    return x


def run_radio_standalone(
    model: RADIOViTModel,
    data: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    imgs_sizes = data["inputs"]["imgs_sizes"].to(device=device, dtype=torch.int64)
    pixel_values = data["inputs"]["pixel_values_flat"].to(device=device, dtype=MODEL_DTYPE)
    encoded = run_radio_single_sequence(model, pixel_values, imgs_sizes)
    patch_features = encoded[:, model.class_token_len :, :]
    return {
        "radio_patch_features": patch_features,
        "encoder_tokens": patch_features.squeeze(0),
    }


def run_radio_omni(
    model: RADIOViTModel,
    data: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    imgs_sizes = data["inputs"]["imgs_sizes"].to(device=device, dtype=torch.int64)
    pixel_values = data["inputs"]["pixel_values_flat"].to(device=device, dtype=MODEL_DTYPE)
    encoded = run_radio_single_sequence(model, pixel_values, imgs_sizes)
    patch_features = encoded[:, model.class_token_len :, :]
    seq_lens = torch.prod(imgs_sizes // model.patch_dim, dim=-1)
    chunks = torch.split(patch_features, seq_lens.tolist(), dim=-2)
    shuffled = torch.cat(
        [
            pixel_shuffle(
                chunk,
                h=int(size[0].item()) // model.patch_dim,
                w=int(size[1].item()) // model.patch_dim,
            )
            for chunk, size in zip(chunks, imgs_sizes)
        ],
        dim=-2,
    )
    return {
        "radio_patch_features": patch_features,
        "encoder_tokens": shuffled.squeeze(0),
    }


BUILDERS: dict[str, Callable[[], torch.nn.Module]] = {
    "mistral": build_mistral_model,
    "c_radio_vit_h": build_radio,
    "post-c-radio-omni": build_radio,
}

RUNNERS: dict[str, Callable[[torch.nn.Module, dict, torch.device], dict[str, torch.Tensor]]] = {
    "mistral": run_mistral,
    "c_radio_vit_h": run_radio_standalone,
    "post-c-radio-omni": run_radio_omni,
}


def golden_path_for_case(name: str, args: argparse.Namespace) -> Path:
    _, _, golden_file = ENCODER_CASES[name]
    if args.golden is not None:
        return args.golden
    path = args.golden_dir / golden_file
    if path.exists():
        return path
    legacy_file = LEGACY_GOLDEN_FILES.get(name)
    if legacy_file is not None:
        legacy_path = args.golden_dir / legacy_file
        if legacy_path.exists():
            return legacy_path
    return path


def checkpoint_path_for_case(name: str, args: argparse.Namespace) -> Path:
    _, ckpt_name, _ = ENCODER_CASES[name]
    path = args.ckpt_root / ckpt_name
    if path.exists():
        return path
    legacy_name = LEGACY_CHECKPOINT_NAMES.get(name)
    if legacy_name is not None:
        legacy_path = args.ckpt_root / legacy_name
        if legacy_path.exists():
            return legacy_path
    return path


def run_case(name: str, args: argparse.Namespace, device: torch.device) -> dict:
    vision_model_type, ckpt_name, _ = ENCODER_CASES[name]
    golden_path = golden_path_for_case(name, args)
    ckpt_path = checkpoint_path_for_case(name, args)
    print(f"running parity for {name}", flush=True)
    data = golden(golden_path, name)
    model = BUILDERS[name]()
    load_info = load_vision_dcp(model, ckpt_path)
    model = model.to(device=device, dtype=MODEL_DTYPE).eval()

    with torch.inference_mode():
        actual = RUNNERS[name](model, data, device)

    comparisons = {}
    for key, value in actual.items():
        expected = data["features"].get(key)
        if expected is not None:
            comparisons[key] = metrics(value, expected)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "encoder": name,
        "vision_model_type": vision_model_type,
        "checkpoint": str(ckpt_path),
        "default_checkpoint_name": ckpt_name,
        "golden": str(golden_path),
        "note": CASE_NOTES[name],
        "load_info": load_info,
        "comparisons": comparisons,
    }


def case_passed(result: dict, min_cosine: float, min_token_cosine: float | None) -> bool:
    encoder_tokens = result["comparisons"].get("encoder_tokens")
    if encoder_tokens is None or not encoder_tokens.get("shape_match", False):
        return False
    if encoder_tokens.get("cosine", -1.0) < min_cosine:
        return False
    if min_token_cosine is not None:
        token_min = encoder_tokens.get("token_cosine_min")
        if token_min is None or token_min < min_token_cosine:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR)
    parser.add_argument(
        "--golden",
        type=Path,
        help="Custom .pt golden file. Requires exactly one --encoder.",
    )
    parser.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=GOLDEN_DIR / "radio_mistral_mcore_parity_results.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-cosine", type=float, default=0.98)
    parser.add_argument("--min-token-cosine", type=float, default=None)
    parser.add_argument(
        "--encoder",
        action="append",
        choices=sorted(ENCODER_CASES),
        help="Encoder to check. May be specified multiple times; defaults to all.",
    )
    parser.add_argument("--no-fail", action="store_true")
    parser.add_argument(
        "--print-golden-schema",
        action="store_true",
        help="Print the expected custom golden .pt schema and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_golden_schema:
        print(GOLDEN_SCHEMA, flush=True)
        return
    encoders = args.encoder or list(DEFAULT_ENCODERS)
    if args.golden is not None and len(encoders) != 1:
        raise SystemExit("--golden requires exactly one --encoder")
    device = torch.device(args.device)
    try:
        init_single_rank(device)
        results = {
            "host": socket.gethostname(),
            "torch_version": torch.__version__,
            "golden_dir": str(args.golden_dir),
            "custom_golden": str(args.golden) if args.golden is not None else None,
            "ckpt_root": str(args.ckpt_root),
            "device": str(device),
            "dtype": str(MODEL_DTYPE),
            "min_cosine": args.min_cosine,
            "min_token_cosine": args.min_token_cosine,
            "cases": {},
        }
        for name in encoders:
            result = run_case(name, args, device)
            result["passed"] = case_passed(result, args.min_cosine, args.min_token_cosine)
            results["cases"][name] = result
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
                f.write("\n")
        print(json.dumps(results, indent=2), flush=True)
        if not args.no_fail and not all(result["passed"] for result in results["cases"].values()):
            raise SystemExit(1)
    finally:
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
