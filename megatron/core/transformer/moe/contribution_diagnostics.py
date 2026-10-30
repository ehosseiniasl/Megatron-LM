# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Detached, additive summaries of the actual weighted MoE outputs."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from megatron.core.tensor_observation import is_observing_tensor, observe_tensor

SOURCE = "moe_contributions"
# [tokens, shared ssq, routed ssq, dot, zero shared, zero routed, zero weight sum],
# followed by k squared sums, k normalized-weight sums, and k zero-output counts.
HEADER_SIZE = 7
_MTP_DEPTH: ContextVar[int | None] = ContextVar("moe_contribution_mtp_depth", default=None)


@contextmanager
def moe_contribution_mtp_depth(depth: int) -> Iterator[None]:
    """Label a repeated MTP invocation by its one-based prediction depth."""
    if not is_observing_tensor(SOURCE):
        yield
        return
    token = _MTP_DEPTH.set(depth)
    try:
        yield
    finally:
        _MTP_DEPTH.reset(token)


def validate_moe_contribution_config(config: object) -> None:
    """Reject layouts that do not expose complete individual expert outputs."""
    dispatcher = getattr(config, "moe_token_dispatcher_type", None)
    hybrid = (
        dispatcher == "flex" and getattr(config, "moe_flex_dispatcher_backend", None) == "hybridep"
    )
    if dispatcher != "alltoall" and not hybrid:
        raise NotImplementedError("moe-contributions requires alltoall or flex/hybridep.")
    if hybrid and getattr(config, "expert_tensor_parallel_size", 1) != 1:
        # Squared norms of TP partial outputs cannot be summed to recover the norm
        # of their sum. Ultra uses ETP=1, so each row is a complete expert output.
        raise NotImplementedError("HybridEP contributions require expert tensor parallel size 1.")
    if hybrid and getattr(config, "moe_hybridep_pad_uneven_dispatch_inputs", False):
        raise NotImplementedError(
            "HybridEP contributions currently require equal per-rank token counts."
        )
    if hybrid and getattr(config, "moe_expert_rank_capacity_factor", None) is not None:
        raise NotImplementedError("HybridEP contributions require dropless dynamic capacity.")
    if getattr(config, "moe_expert_capacity_factor", None) is not None:
        raise NotImplementedError("moe-contributions currently requires dropless routing.")
    if getattr(config, "moe_latent_size", None) and getattr(
        config, "moe_shared_expert_overlap", False
    ):
        raise NotImplementedError(
            "Latent MoE contributions require shared expert overlap disabled."
        )


@dataclass
class ContributionRouting:
    """Small metadata surviving dispatch; no expert activations are retained."""

    ranks: torch.Tensor
    weight_sums: torch.Tensor
    valid_tokens: torch.Tensor
    zero_weight_tokens: torch.Tensor
    expert_indices: torch.Tensor | None = None


@torch.no_grad()
def contribution_rank_map(
    probs: torch.Tensor, routing_map: torch.Tensor, topk: int
) -> tuple[torch.Tensor, ContributionRouting]:
    """Rank selected experts by their final mixture weights, before dispatch padding.

    Zero-weight selected experts remain selected. Ties use ascending expert id.
    Rank zero denotes padding, and real ranks are one-based. Weight shares divide
    out the common routing scale; actual contribution outputs retain that scale.
    """
    if probs.ndim != 2 or probs.shape != routing_map.shape or not 0 < topk <= probs.shape[1]:
        raise ValueError("Expected matching [tokens, experts] probabilities/mask and a valid topk.")
    # Only sort k candidates, not a full int64 [tokens, experts] permutation.
    selected = probs.detach().masked_fill(~routing_map, -float("inf"))
    indices = selected.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    weights = selected.gather(1, indices)
    order = weights.argsort(dim=-1, descending=True, stable=True)
    indices = indices.gather(1, order)
    valid = routing_map.sum(dim=-1) == topk
    weights = probs.detach().gather(1, indices).float().masked_fill(~valid[:, None], 0)
    totals = weights.sum(dim=-1)
    shares = weights / torch.where(totals != 0, totals, 1)[:, None]
    rank_map = torch.zeros_like(probs, dtype=torch.float32)
    ranks = torch.arange(1, topk + 1, device=probs.device, dtype=torch.float32)
    rank_map.scatter_(1, indices, ranks.expand_as(weights) * valid[:, None])
    return rank_map, ContributionRouting(
        ranks=rank_map.new_empty(0),
        weight_sums=shares.double().sum(dim=0),
        valid_tokens=valid,
        zero_weight_tokens=((totals == 0) & valid).sum(),
        expert_indices=indices,
    )


@torch.no_grad()
def row_moments(left: torch.Tensor, right: torch.Tensor | None = None) -> torch.Tensor:
    """Return [ssq, zero] or paired [left ssq, right ssq, dot, left zero, right zero].

    CUDA uses a fused reduction and never materializes an FP32 copy of an entire
    expert output. CPU is a small-reference implementation for unit tests.
    """
    if left.ndim != 2 or left.shape[1] == 0 or (right is not None and right.shape != left.shape):
        raise ValueError("Expected matching [tokens, hidden] tensors with nonempty hidden size.")
    if left.is_cuda:
        from megatron.core.fusions.moe_contribution import contribution_row_moments

        return contribution_row_moments(left, right)
    left = left.detach().float()
    left_zero = (left == 0).all(dim=-1).float()
    if right is None:
        return torch.stack((left.square().sum(dim=-1), left_zero))
    right = right.detach().float()
    return torch.stack(
        (
            left.square().sum(-1),
            right.square().sum(-1),
            (left * right).sum(-1),
            left_zero,
            (right == 0).all(dim=-1).float(),
        )
    )


@torch.no_grad()
def ranked_contribution_sums(
    outputs: torch.Tensor, routing: ContributionRouting
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce complete weighted expert outputs by rank, excluding padding."""
    if routing.ranks.shape != outputs.shape[:1]:
        raise ValueError("Permuted rank labels must match expert output rows.")
    topk = routing.weight_sums.numel()
    energy, zero = row_moments(outputs).unbind()
    ranks = routing.ranks.long()
    # Bin zero is padding. In particular, do not classify real zero-weight
    # assignments as padding just because their weighted output is zero.
    sums = torch.zeros(topk + 1, dtype=torch.float64, device=outputs.device)
    zeros = torch.zeros_like(sums)
    sums.scatter_add_(0, ranks, energy.double())
    zeros.scatter_add_(0, ranks, zero.double())
    return sums[1:], zeros[1:]


@torch.no_grad()
def contribution_energy_payload(outputs: torch.Tensor) -> torch.Tensor:
    """Encode row energy and exact zeros in HybridEP's single FP32 scalar channel.

    Negative one denotes an exactly zero output. Squared norms are nonnegative,
    including nonzero outputs whose squared norm underflows, so this preserves
    exact-zero counts without an additional collective. Only ETP=1 is supported.
    """
    energy, zero = row_moments(outputs).unbind()
    return energy.masked_fill(zero.bool(), -1).contiguous()


@torch.no_grad()
def returned_contribution_sums(
    payload: torch.Tensor, routing: ContributionRouting
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce returned token/expert energies using the original top-k selections.

    HybridEP returns a dense [tokens, EP, local_experts] scalar map. Gather only
    the real selections, preserving zero-weight selections and ignoring padding.
    """
    if routing.expert_indices is None:
        raise ValueError("Original expert selections are required for HybridEP contributions.")
    tokens = routing.expert_indices.shape[0]
    payload = payload[:tokens].flatten(1)
    ranked = payload.gather(1, routing.expert_indices)
    valid = routing.valid_tokens[:, None]
    zero = (ranked == -1) & valid
    # Mask rather than compact: boolean indexing would synchronize the GPU to
    # determine the output size on every observed MoE invocation.
    return ranked.masked_fill(zero | ~valid, 0).double().sum(0), zero.double().sum(0)


@torch.no_grad()
def observe_moe_contributions(
    owner: object,
    routed: torch.Tensor,
    shared: torch.Tensor | None,
    routing: ContributionRouting,
    rank_sums: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Emit one compact summary before the shared and routed branches are added."""
    if not is_observing_tensor(SOURCE):
        return
    routed = routed.detach().reshape(-1, routed.shape[-1])
    valid = routing.valid_tokens
    if valid.shape != routed.shape[:1]:
        raise ValueError("Routing token mask must match the combined output.")
    if shared is None:
        routed_ssq, routed_zero = row_moments(routed).unbind()
        shared_ssq = dot = torch.zeros_like(routed_ssq)
        shared_zero = torch.ones_like(routed_zero)
    else:
        shared = shared.detach().reshape_as(routed)
        shared_ssq, routed_ssq, dot, shared_zero, routed_zero = row_moments(shared, routed).unbind()
    summary = torch.cat(
        (
            torch.stack(
                (
                    valid.sum().double(),
                    shared_ssq.masked_fill(~valid, 0).double().sum(),
                    routed_ssq.masked_fill(~valid, 0).double().sum(),
                    dot.masked_fill(~valid, 0).double().sum(),
                    shared_zero.masked_fill(~valid, 0).double().sum(),
                    routed_zero.masked_fill(~valid, 0).double().sum(),
                    routing.zero_weight_tokens.double(),
                )
            ),
            rank_sums[0],
            routing.weight_sums,
            rank_sums[1],
        )
    )
    depth = _MTP_DEPTH.get()
    name = SOURCE if depth is None else f"{SOURCE}.mtp_{depth}"
    if getattr(getattr(owner, "config", None), "moe_latent_size", None):
        name += ".latent"
    observe_tensor(
        owner,
        name,
        SOURCE,
        summary,
        tp_shard_dim=(
            0 if getattr(getattr(owner, "config", None), "sequence_parallel", False) else None
        ),
    )
