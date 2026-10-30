# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor adapters for the optional simplified sparse attention kernel package."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch
from packaging.version import Version

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

from . import dsa_layout, dsa_masking


@lru_cache(maxsize=1)
def _check_kernel_package_version() -> None:
    """Check the optional packed-kernel dependency once, when SSA executes."""
    requirement = "simplified-sparse-attention >=0.1.0.dev6"
    try:
        installed = version("simplified-sparse-attention")
    except PackageNotFoundError as exc:
        raise RuntimeError(f"The CuTe backend requires {requirement}.") from exc
    if Version(installed) < Version("0.1.0.dev6"):
        raise RuntimeError(f"The CuTe backend requires {requirement}; found {installed}.")


@dataclass
class AttentionLayout:
    """Sequence metadata reused by SSA layers within one packed microbatch."""

    key_starts: torch.Tensor
    key_ends: torch.Tensor
    query_valid: torch.Tensor
    num_real_tokens: torch.Tensor
    key_reorder: torch.Tensor | None
    cu_seqlens: torch.Tensor
    max_seqlen: int
    packed: bool
    cp_size: int
    cp_rank: int
    phase1_segments: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


def build_attention_layout(
    query: torch.Tensor,
    packed_seq_params: PackedSeqParams | None,
    cp_group: torch.distributed.ProcessGroup | None,
) -> AttentionLayout:
    """Map local zigzag query rows to globally ordered, document-local causal keys.

    Packed cumulative lengths retain real document lengths; padded cumulative
    lengths describe physical storage, with padding at each document's suffix.
    Language-model loss masks do not describe validity: image/prompt tokens may
    be real inputs without a prediction objective.
    """
    local_rows, device = query.shape[0], query.device
    cp_size = cp_group.size() if cp_group is not None else 1
    cp_rank = cp_group.rank() if cp_size > 1 else 0
    global_rows = local_rows * cp_size
    packed = packed_seq_params is not None
    if packed:
        if packed_seq_params.qkv_format != "thd":
            raise ValueError("Simplified sparse attention requires THD packed metadata.")
        if packed_seq_params.cp_group is not None and packed_seq_params.cp_group is not cp_group:
            raise ValueError("Packed SSA metadata must use the attention CP group.")
        if packed_seq_params.local_cp_size not in (None, cp_size):
            raise ValueError("Packed SSA metadata has a different CP size.")
        cu_q, cu_k = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
        # This is self-attention: Q/K use the same document boundaries. The
        # producer may store them in distinct but equivalent tensors.
        if cu_q.shape != cu_k.shape:
            raise ValueError("SSA self-attention requires matching Q/K document boundaries.")
        logical_cu = packed_seq_params.cu_seqlens_q
        if logical_cu is None or logical_cu.shape != cu_q.shape:
            raise ValueError("Packed SSA requires matching logical and physical document lengths.")
        if packed_seq_params.max_seqlen_q is None:
            raise ValueError("Packed SSA requires max_seqlen_q.")
        cu_seqlens = cu_q
        max_seqlen = packed_seq_params.max_seqlen_q
    else:
        cu_seqlens = torch.tensor((0, global_rows), dtype=torch.int32, device=device)
        max_seqlen = global_rows
        query_valid = torch.ones(local_rows, dtype=torch.bool, device=device)

    if cp_size > 1:
        if packed:
            positions, reorder = (
                dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder(
                    cu_seqlens,
                    cu_seqlens,
                    cp_size,
                    cp_rank,
                    device,
                    local_output_size=local_rows,
                    global_output_size=global_rows,
                    query_cu_seqlens_cover_output=True,
                    key_cu_seqlens_cover_output=True,
                )
            )
        else:
            positions = dsa_layout.build_zigzag_cp_local_positions(
                global_rows, cp_size, cp_rank, device
            )
            reorder = dsa_layout.build_zigzag_allgather_cp_key_reorder(local_rows, cp_size, device)
    else:
        positions = torch.arange(local_rows, dtype=torch.int64, device=device)
        reorder = None
    starts, ends = dsa_masking.generate_varlen_mask_params_for_positions(cu_seqlens, positions)
    if packed:
        # Derive validity once from the same physical positions used by CP.
        # Real query rows precede each document's padding, so their causal K/V
        # prefixes remain valid with physical boundaries in both training phases.
        sequence_ids = torch.searchsorted(
            cu_seqlens[1:].to(device=device).contiguous(), positions, right=True
        )
        logical_lengths = logical_cu.to(device=device, dtype=torch.int64).diff()
        query_valid = positions < starts + logical_lengths.index_select(0, sequence_ids)
        num_real_tokens = query_valid.sum(dtype=torch.float32).reshape(1)
        if cp_size > 1:
            torch.distributed.all_reduce(num_real_tokens, group=cp_group)
        # An all-padding microbatch contributes zero auxiliary loss.
        num_real_tokens = num_real_tokens.clamp_min(1)
    else:
        num_real_tokens = torch.full((1,), float(global_rows), dtype=torch.float32, device=device)
    return AttentionLayout(
        starts,
        ends,
        query_valid,
        num_real_tokens,
        reorder,
        cu_seqlens,
        max_seqlen,
        packed,
        cp_size,
        cp_rank,
    )


def _phase1_flash_inputs(
    key: torch.Tensor, value: torch.Tensor, layout: AttentionLayout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Express each zigzag query chunk as bottom-right causal FlashAttention.

    A rank owns the front and mirrored back chunk of each document. Give each
    chunk its causal K/V prefix; FlashAttention's rectangular mask then exactly
    matches its global query positions. Dense teacher tensors are detached.
    """
    if layout.cp_size == 1:
        return (
            key,
            value,
            layout.cu_seqlens,
            layout.cu_seqlens,
            layout.max_seqlen,
            layout.max_seqlen,
        )
    if layout.phase1_segments is None:
        segments = dsa_layout.build_packed_cp_indexer_layout(
            layout.cu_seqlens,
            layout.cu_seqlens,
            cp_size=layout.cp_size,
            cp_rank=layout.cp_rank,
            key_size=key.shape[0],
        )
        layout.phase1_segments = (
            segments.source_indices,
            segments.segment_cu_q.to(dtype=torch.int32),
            segments.segment_cu_k.to(dtype=torch.int32),
        )
    source_indices, cu_q, cu_k = layout.phase1_segments
    divisor = 2 * layout.cp_size
    return (
        key.index_select(0, source_indices),
        value.index_select(0, source_indices),
        cu_q,
        cu_k,
        layout.max_seqlen // divisor,
        layout.max_seqlen * (divisor - layout.cp_rank) // divisor,
    )


def run_cute_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    *,
    topk: int,
    loss_coeff: float,
    calculate_per_token_loss: bool,
    train_indexer_only: bool,
    cp_group: torch.distributed.ProcessGroup | None = None,
    layout: AttentionLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run packed THD or SBHD B1 with local queries and globally ordered keys.

    Phase 1 uses dense teacher attention and full-support indexer KL. Phase 2
    uses sparse attention and selected-support KL. The package applies the
    coefficient and global sum/mean normalization exactly once. Standard
    differentiable CP gathers preserve BF16 in forward and backward reduction.
    """
    packed_input = query.ndim == 3
    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")
    value, _ = dsa_layout.ensure_sbhd(value, "value")
    q_indexer, _ = dsa_layout.ensure_sbhd(q_indexer, "q_indexer")
    k_indexer, _ = dsa_layout.ensure_sbhd(k_indexer, "k_indexer")
    if query.shape[1] != 1:
        raise ValueError("CuTe sparse attention requires batch size 1.")
    sequence_length = query.shape[0]
    if key.shape != (sequence_length, 1, 1, 256) or value.shape != key.shape:
        raise ValueError("CuTe sparse attention requires local self-attention K/V [S,1,1,256].")
    if q_indexer.shape != (sequence_length, 1, 1, 128) or k_indexer.shape != q_indexer.shape:
        raise ValueError("CuTe sparse attention requires local indexer Q/K [S,1,1,128].")
    if layout is None:
        if packed_input:
            raise ValueError("THD sparse attention requires packed layout metadata.")
        layout = build_attention_layout(query, None, cp_group)

    _check_kernel_package_version()
    try:
        from simplified_sparse_attention import RowMetadata, sparse_attention
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The CuTe backend requires simplified-sparse-attention >=0.1.0.dev6 "
            "and its matching CuTe runtime."
        ) from exc

    row_metadata = RowMetadata(
        layout.key_starts,
        layout.key_ends,
        layout.query_valid,
        (
            torch.ones_like(layout.num_real_tokens)
            if calculate_per_token_loss
            else layout.num_real_tokens
        ),
        triangular_scores=layout.cp_size == 1,
        mask_invalid_rows=layout.packed,
    )
    query, key, value = query[:, 0], key[:, 0], value[:, 0]
    q_indexer = q_indexer[:, 0, 0].contiguous()
    k_indexer = k_indexer[:, 0, 0].contiguous()
    if train_indexer_only:
        # Neither the dense teacher nor its targets participate in backbone AD.
        query, key, value = query.detach(), key.detach(), value.detach()
    if layout.cp_size > 1:

        def gather(tensor):
            gathered = gather_from_sequence_parallel_region(tensor, group=cp_group)
            return gathered.index_select(0, layout.key_reorder)

        key, value, k_indexer = gather(key), gather(value), gather(k_indexer)
    if train_indexer_only:
        from simplified_sparse_attention.phase1 import dense_attention_flash4, dense_indexer_kl

        teacher_key, teacher_value, cu_q, cu_k, max_q, max_k = _phase1_flash_inputs(
            key, value, layout
        )
        output, teacher_lse = dense_attention_flash4(
            query,
            teacher_key,
            teacher_value,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            query_valid=row_metadata.query_valid,
            softmax_scale=256**-0.5,
        )
        loss = None
        if loss_coeff > 0:
            loss = dense_indexer_kl(
                query,
                key,
                q_indexer,
                k_indexer,
                key_starts=row_metadata.key_starts,
                key_ends=row_metadata.key_ends,
                query_valid=row_metadata.query_valid,
                loss_denominator=row_metadata.loss_denominator,
                teacher_softmax_scale=256**-0.5,
                student_softmax_scale=128**-0.5,
                loss_coeff=loss_coeff,
                teacher_lse=teacher_lse,
                use_kf=True,
            )
    else:
        output, loss = sparse_attention(
            query, key, value, q_indexer, k_indexer, row_metadata, topk, loss_coeff=loss_coeff
        )
    output = (
        output.reshape(sequence_length, -1)
        if packed_input
        else output.reshape(sequence_length, 1, -1)
    )
    return output, loss
