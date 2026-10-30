# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training and checkpoint policy for simplified sparse attention."""

from copy import copy
from functools import lru_cache
from typing import Any

import torch

from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedBase


def is_simplified_sparse_attention(config: Any) -> bool:
    """Whether this run explicitly selects the simplified sparse implementation."""
    return getattr(config, "experimental_attention_variant", None) == "ssa"


def is_simplified_sparse_checkpoint(config: Any) -> bool:
    """Recognize current SSA and earlier learned-Q/K checkpoint metadata only."""
    return is_simplified_sparse_attention(config) or (
        getattr(config, "experimental_attention_variant", None) == "dsa"
        and (
            getattr(config, "dsa_gqa_backend", None) == "cute"
            or getattr(config, "dsa_min_memory_backend", None) == "cute"
            or getattr(config, "dsa_indexer_mode", None) == "simplified"
        )
    )


def checkpoint_spec(spec: list[str] | None) -> list[str] | None:
    """Translate the two former sparse spec names when reading older checkpoints."""
    old_module = "megatron.core.transformer.experimental_attention_variant.dsa_layer_specs"
    if spec and len(spec) == 2 and spec[0] == old_module:
        names = {"dsa_stack_spec": "ssa_stack_spec", "gdp_dsa_stack_spec": "gdp_ssa_stack_spec"}
        if spec[1] in names:
            return [
                "megatron.core.transformer.experimental_attention_variant."
                "simplified_sparse_attention_layer_specs",
                names[spec[1]],
            ]
    return spec


@lru_cache(maxsize=None)
def num_indexer_layers(num_layers: int, hybrid_layer_pattern: str | None) -> int:
    """Count active indexers once for each model architecture used by this process."""
    if not hybrid_layer_pattern:
        return num_layers
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, parse_hybrid_pattern

    pattern = parse_hybrid_pattern(hybrid_layer_pattern)
    count = pattern.main_pattern.count(Symbols.ATTENTION)
    if pattern.mtp_pattern:
        count += pattern.mtp_num_depths * pattern.mtp_pattern.count(Symbols.ATTENTION)
    return count


def is_indexer_parameter(name: str) -> bool:
    """Match the indexer component without matching similarly named backbone modules."""
    return name.startswith("indexer.") or ".indexer." in name


def freeze_backbone_for_indexer_training(model_chunks: list[torch.nn.Module], config: Any) -> None:
    """Freeze before DDP/optimizer setup, preserving explicitly frozen indexer parameters."""
    if not is_simplified_sparse_attention(config) or not config.ssa_train_indexer_only:
        return

    local_indexers = 0
    for model in model_chunks:
        for name, parameter in model.named_parameters():
            is_indexer = is_indexer_parameter(name)
            if not is_indexer:
                parameter.requires_grad_(False)
            local_indexers += int(is_indexer)
        # Expert biases are buffers, updated outside the optimizer.
        for module in model.modules():
            if hasattr(module, "frozen_expert_bias"):
                module.frozen_expert_bias = True

    # A pipeline stage can contain no indexer. Reject only a globally incompatible model.
    if torch.distributed.is_initialized():
        device = "cuda" if torch.distributed.get_backend() == "nccl" else "cpu"
        count = torch.tensor(local_indexers, dtype=torch.int64, device=device)
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        local_indexers = count.item()
    if local_indexers == 0:
        raise RuntimeError("Indexer-only training found no sparse-attention indexer parameters.")


def initializing_indexer_from_dense(args: Any, checkpoint_args: Any, load_optimizer: bool) -> bool:
    """Only an explicit phase-1 model-only load may introduce fresh indexer weights."""
    return (
        is_simplified_sparse_attention(args)
        and args.ssa_train_indexer_only
        and getattr(checkpoint_args, "experimental_attention_variant", None) is None
        and not load_optimizer
    )


def validate_checkpoint(args: Any, checkpoint_args: Any, *, load_optimizer: bool) -> None:
    """Validate architecture separately from the optimizer's trainable parameter set."""
    if not (
        is_simplified_sparse_attention(args) or is_simplified_sparse_checkpoint(checkpoint_args)
    ):
        return
    if initializing_indexer_from_dense(args, checkpoint_args, load_optimizer):
        return

    assert is_simplified_sparse_checkpoint(checkpoint_args), (
        "A dense checkpoint can initialize simplified sparse attention only in phase 1 "
        "with --no-load-optim or --finetune."
    )
    assert is_simplified_sparse_attention(
        args
    ), "A simplified sparse-attention checkpoint requires experimental_attention_variant='ssa'."
    assert not getattr(
        checkpoint_args, "multi_latent_attention", False
    ), "An MLA checkpoint cannot initialize simplified sparse attention."
    # Older checkpoints exposed these architecture choices. The standalone
    # implementation fixes them; no runtime compatibility flags are needed.
    for name, expected in (
        ("dsa_indexer_mode", "simplified"),
        ("dsa_simplified_use_learned_k", True),
    ):
        assert (
            getattr(checkpoint_args, name, expected) == expected
        ), f"Checkpoint {name} must be {expected!r} for simplified sparse attention."
    assert checkpoint_args.dsa_indexer_topk == args.dsa_indexer_topk, (
        "dsa_indexer_topk differs between checkpoint "
        f"({checkpoint_args.dsa_indexer_topk}) and runtime ({args.dsa_indexer_topk})."
    )
    if load_optimizer:
        assert not getattr(checkpoint_args, "dsa_train_main_only", False), (
            "A main-only checkpoint has a different trainable parameter set. "
            "Use --no-load-optim or --finetune."
        )
        checkpoint_phase1 = getattr(
            checkpoint_args,
            "ssa_train_indexer_only",
            getattr(checkpoint_args, "dsa_train_indexer_only", False),
        )
        assert checkpoint_phase1 == args.ssa_train_indexer_only, (
            "Sparse-attention training phase changed while loading optimizer state. "
            "Use --no-load-optim or --finetune when changing the trainable parameters."
        )


def retain_initialized_indexer(sharded_state_dict: dict) -> None:
    """Keep new indexer tensors local when loading a distributed dense checkpoint."""

    def retain(value: Any) -> Any:
        if isinstance(value, ShardedBase) and is_indexer_parameter(value.key):
            return LocalNonpersistentObject(value.data)
        return value

    dict_list_map_inplace(retain, sharded_state_dict)


def add_initialized_indexer_state(module: torch.nn.Module, state_dict: dict) -> dict:
    """Fill only new indexer keys; strict model loading still validates every backbone key."""
    state_dict = copy(state_dict)
    for key, value in module.state_dict().items():
        if is_indexer_parameter(key) and key not in state_dict:
            state_dict[key] = value
    return state_dict
