# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused scalar statistics without tracing concurrently executing model code."""

import torch
import triton
import triton.language as tl


@triton.jit
def _moment_sums_kernel(values, partials, count, blocks, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(values + offsets, offsets < count, other=0)
    absolute = tl.abs(value)
    tl.store(partials + block, tl.sum(value, axis=0))
    tl.store(partials + blocks + block, tl.sum(absolute, axis=0))
    tl.store(partials + 2 * blocks + block, tl.sum(absolute * absolute, axis=0))


def moment_sums(tensor: torch.Tensor) -> torch.Tensor:
    """Return sum, absolute sum, and squared sum for a CUDA FP32/FP64 tensor.

    The caller selects the accumulation dtype. One input pass computes all three
    block reductions; only the small partial-sum table needs a second reduction.
    No Dynamo tracing occurs, including on first use or when tensor shapes change.
    """
    values = tensor.contiguous().view(-1)
    count = values.numel()
    if count == 0:
        return values.new_zeros(3)
    blocks = triton.cdiv(count, 2048)
    partials = values.new_empty((3, blocks))
    _moment_sums_kernel[(blocks,)](values, partials, count, blocks, BLOCK=2048)
    return partials[:, 0] if blocks == 1 else partials.sum(dim=1)
