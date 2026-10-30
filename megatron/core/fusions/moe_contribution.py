# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Row reductions for MoE diagnostics with O(tokens) temporary storage."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_moments(
    left,
    right,
    output,
    rows,
    hidden,
    left_stride,
    right_stride,
    PAIRED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    a = tl.load(left + row * left_stride + col, col < hidden, other=0).to(tl.float32)
    tl.store(output + row, tl.sum(a * a, axis=0))
    if PAIRED:
        b = tl.load(right + row * right_stride + col, col < hidden, other=0).to(tl.float32)
        tl.store(output + rows + row, tl.sum(b * b, axis=0))
        tl.store(output + 2 * rows + row, tl.sum(a * b, axis=0))
        tl.store(output + 3 * rows + row, tl.max(tl.abs(a), axis=0) == 0)
        tl.store(output + 4 * rows + row, tl.max(tl.abs(b), axis=0) == 0)
    else:
        tl.store(output + rows + row, tl.max(tl.abs(a), axis=0) == 0)


def contribution_row_moments(left: torch.Tensor, right: torch.Tensor | None) -> torch.Tensor:
    """Compute FP32 squared norms and dot products without copying hidden vectors."""
    if left.stride(-1) != 1 or (right is not None and right.stride(-1) != 1):
        raise ValueError("MoE contribution reductions require a contiguous hidden dimension.")
    rows, hidden = left.shape
    result = torch.empty(
        (5 if right is not None else 2, rows), dtype=torch.float32, device=left.device
    )
    if rows:
        _row_moments[(rows,)](
            left,
            left if right is None else right,
            result,
            rows,
            hidden,
            left.stride(0),
            0 if right is None else right.stride(0),
            PAIRED=right is not None,
            BLOCK=triton.next_power_of_2(hidden),
        )
    return result
