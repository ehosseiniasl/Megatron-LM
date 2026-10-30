# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Vision-encoder-only warm start from a raw torch DCP checkpoint."""

from __future__ import annotations

import os
import re

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import _is_process_group_member as is_process_group_member
from megatron.core.utils import unwrap_model

_VISION_DCP_PREFIX = "model.vision_model."


def _remap_vision_model_key(key: str) -> str:
    """Map fused TE norm names to their separate-norm equivalents."""
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


def _resolve_ckpt_iter_dir(ckpt_dir: str) -> str:
    """Resolve a tracker-style checkpoint root to its iteration directory."""
    tracker = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tracker):
        with open(tracker) as tracker_file:
            content = tracker_file.read().strip()
        if content and content != "release":
            iter_dir = os.path.join(ckpt_dir, f"iter_{int(content):07d}")
            if os.path.isdir(iter_dir):
                return iter_dir
    return ckpt_dir


def _vision_tp_slice(
    tensor: torch.Tensor, param_shape: torch.Size, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """Slice a full TP=1 tensor for a tensor-parallel encoder rank."""
    if tp_size == 1 or tuple(tensor.shape) == tuple(param_shape):
        return tensor
    if tensor.shape[0] != param_shape[0]:
        start = tp_rank * param_shape[0]
        return tensor[start : start + param_shape[0], ...]
    if tensor.ndim > 1 and tensor.shape[1] != param_shape[1]:
        start = tp_rank * param_shape[1]
        return tensor[:, start : start + param_shape[1]]
    return tensor


def load_vision_only_from_dcp(
    mimo_model: torch.nn.Module, ckpt_dir: str, topology, encoder_name: str
) -> None:
    """Load every persistent vision tensor while leaving language/projector init intact."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    iter_dir = _resolve_ckpt_iter_dir(ckpt_dir)
    world_rank = dist.get_rank() if dist.is_initialized() else 0

    encoder_grid = topology.grids.get(encoder_name) if topology is not None else None
    if encoder_grid is None:
        available = sorted(topology.grids) if topology is not None else None
        raise RuntimeError(f"no encoder grid named {encoder_name!r}; available grids: {available}")
    if not encoder_grid.is_current_rank_in_grid():
        return

    submodules = getattr(mimo_model, "modality_submodules", None)
    if submodules is None or encoder_name not in submodules:
        raise RuntimeError(f"encoder rank has no modality_submodules[{encoder_name!r}]")
    submodule = unwrap_model(submodules[encoder_name])
    encoders = getattr(submodule, "encoders", None)
    if encoders is None or encoder_name not in encoders:
        raise RuntimeError(f"vision submodule has no encoders[{encoder_name!r}]")
    encoder_wrapper = encoders[encoder_name]
    vision_model = getattr(encoder_wrapper, "vit_model", None) or getattr(
        encoder_wrapper, "radio_model", None
    )
    if vision_model is None:
        vision_model = encoder_wrapper

    module_pgs = topology.module_pgs.get(encoder_name)
    tp_pg = getattr(module_pgs, "tp", None)
    dp_cp_pg = getattr(module_pgs, "dp_cp", None)
    if dp_cp_pg is None or not is_process_group_member(dp_cp_pg):
        raise RuntimeError(
            f"encoder {encoder_name!r} has no local dp_cp process group for DCP load"
        )
    tp_rank = dist.get_rank(tp_pg) if tp_pg is not None else 0
    tp_size = dist.get_world_size(tp_pg) if tp_pg is not None else 1

    reader = FileSystemReader(iter_dir)
    checkpoint_metadata = reader.read_metadata().state_dict_metadata
    load_state = {
        key: torch.empty(metadata.size, dtype=metadata.properties.dtype)
        for key, metadata in checkpoint_metadata.items()
        if key.startswith(_VISION_DCP_PREFIX) and isinstance(metadata, TensorStorageMetadata)
    }
    if not load_state:
        raise RuntimeError(f"no {_VISION_DCP_PREFIX!r} tensors found in raw DCP {iter_dir}")

    dcp.load(load_state, storage_reader=reader, process_group=dp_cp_pg)

    model_tensors = dict(vision_model.named_parameters())
    model_tensors.update(dict(vision_model.named_buffers()))
    written_model_keys: set[str] = set()
    skipped_extra: list[str] = []

    for checkpoint_key, tensor in load_state.items():
        relative_key = checkpoint_key[len(_VISION_DCP_PREFIX) :]
        model_key = relative_key
        param = model_tensors.get(model_key)
        if param is None:
            remapped = _remap_vision_model_key(relative_key)
            if remapped != relative_key:
                model_key = remapped
                param = model_tensors.get(model_key)
        if param is None:
            skipped_extra.append(relative_key)
            continue
        tensor = _vision_tp_slice(tensor, param.shape, tp_rank, tp_size)
        if tuple(tensor.shape) != tuple(param.shape):
            raise RuntimeError(
                f"vision tensor {relative_key!r} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(param.shape)} on TP rank {tp_rank}/{tp_size}"
            )
        param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))
        written_model_keys.add(model_key)

    persistent_keys = {key for key in vision_model.state_dict() if "_extra_state" not in key}
    unwritten = sorted(persistent_keys - written_model_keys)
    if unwritten:
        raise RuntimeError(
            f"{len(unwritten)} persistent vision tensors were not written: {unwritten[:30]}"
        )

    if skipped_extra and dist.get_rank(dp_cp_pg) == 0:
        print(
            f"[load-vision-from] ignored {len(skipped_extra)} unowned DCP keys: "
            f"{sorted(skipped_extra)[:16]}",
            flush=True,
        )
    print(
        f"[load-vision-from] rank={world_rank} loaded "
        f"{len(written_model_keys)} vision tensors from {iter_dir} "
        f"(tp_rank={tp_rank}/{tp_size})",
        flush=True,
    )
