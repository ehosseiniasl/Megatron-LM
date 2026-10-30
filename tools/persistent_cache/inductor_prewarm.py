"""Opt-in startup warmup for the weighted ReLU indexing specializations.

Disk caches cannot populate a new worker's in-memory Dynamo guards. Exercise
the ordinary compiled functions before model allocation; retain every guard
and compiler setting. Importing this module does not import Torch.
"""

import functools
import math
import os
import sys
import time
from argparse import Namespace
from types import ModuleType


def _plan(args, rank):
    if not (
        getattr(args, "bf16", False)
        and getattr(args, "squared_relu", False)
        and getattr(args, "use_fused_weighted_squared_relu", False)
        and getattr(args, "num_experts", 0)
    ):
        return None, "requires BF16 fused squared-ReLU MoE"
    if getattr(args, "use_te_activation_func", False):
        return None, "Transformer Engine activation path bypasses the compiled function"
    # Do not reject use_transformer_engine_op_fuser: per-layer BF16 precision
    # overrides can select this ordinary weighted function even with that flag.
    # The explicit prewarm opt-in should be used for recipes reaching this path.
    if getattr(args, "moe_apply_probs_on_input", False):
        return None, "input-weighted MoE bypasses the weighted activation function"
    clamp = getattr(args, "activation_func_tanh_clamp_scale", None)
    if not isinstance(clamp, (float, int)) or not math.isfinite(clamp) or clamp <= 0:
        return None, "requires a finite positive tanh clamp"
    width = getattr(args, "moe_ffn_hidden_size", None)
    tp = getattr(args, "expert_tensor_parallel_size", None)
    if not isinstance(width, int) or not isinstance(tp, int) or width <= 0 or tp <= 0 or width % tp:
        return None, "requires a validated local expert width"
    if hasattr(args, "mimo_llm_offset"):
        offset = args.mimo_llm_offset
        size = getattr(args, "mimo_llm_world_size", None)
        if not isinstance(offset, int) or not isinstance(size, int) or offset < 0 or size <= 0:
            return None, "requires a validated MIMO language rank interval"
        if not offset <= rank < offset + size:
            return None, "rank does not own the language model"
    width //= tp
    high = (((2**31 + width - 1) // width + 255) // 256) * 256
    rows = [max(256, (high // 4 // 256) * 256), max(512, (high // 3 // 256) * 256), high]
    if not rows[0] < rows[1] < rows[2]:
        return None, "unsupported expert width for indexing warmup"
    return {"width": width, "rows": rows, "clamp": float(clamp)}, None


def _log(message):
    print(f"[INDUCTOR PREWARM] {message}", file=sys.stderr, flush=True)


def _run_shape(torch, forward, backward, rows, width, clamp):
    x_base = grad_base = x = weights = grad = output = dx = dw = None
    try:
        # Keep activation/gradient views, but allocate weights without a base.
        # Checkpoint recomputation detaches weights, removing their view base.
        # Warming a weights view adds weights._base.size() guards that reject
        # those detached tensors. Base-free weights also cover ordinary views.
        with torch.enable_grad():
            x_base = torch.ones((rows, width), device="cuda", dtype=torch.bfloat16)
            weights = torch.ones((rows, 1), device="cuda", dtype=torch.float32, requires_grad=True)
            grad_base = torch.ones_like(x_base)
            grad = grad_base.view(rows, width)
        for requires_grad in (False, True):
            with torch.enable_grad():
                x_base.requires_grad_(requires_grad)
                x = x_base.view(rows, width)
            with torch.no_grad():
                output = forward(x, weights, clamp)
            output = None
        with torch.no_grad():
            dx, dw = backward(grad, x, weights, clamp)
    finally:
        # Also release references held by an exception traceback before the
        # outer cleanup runs. No retry with smaller shapes or altered settings.
        x_base = grad_base = x = weights = grad = output = dx = dw = None


def warmup(args: Namespace) -> bool:
    """Warm supported functions once at startup when explicitly enabled.

    Insufficient memory or an unsupported recipe skips the optional warmup.
    Execution errors propagate; CUDA or compiler failures must not be hidden.
    """
    if os.environ.get("MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU") != "1":
        return False
    rank = getattr(args, "rank", None)
    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    plan, reason = _plan(args, rank)
    if plan is None:
        _log(f"rank={rank} skipped: {reason}")
        return False

    import torch

    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    # Allow four FP32-sized temporaries in addition to live BF16 inputs/outputs,
    # plus headroom for compiler work and other startup allocations.
    estimated_bytes = 32 * max(plan["rows"]) * plan["width"]
    required_bytes = estimated_bytes + max(2 * 1024**3, total_bytes // 10)
    if free_bytes < required_bytes:
        _log(f"rank={rank} skipped: free_bytes={free_bytes} required_bytes={required_bytes}")
        return False

    from megatron.core.fusions.fused_weighted_squared_relu import (
        weighted_clamped_squared_relu,
        weighted_clamped_squared_relu_back,
    )

    before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    start = time.monotonic()
    try:
        with torch.random.fork_rng(devices=[device]):
            for rows in plan["rows"]:
                _run_shape(
                    torch,
                    weighted_clamped_squared_relu,
                    weighted_clamped_squared_relu_back,
                    rows,
                    plan["width"],
                    plan["clamp"],
                )
            torch.cuda.synchronize(device)
    finally:
        torch.cuda.empty_cache()
    compiled = torch._dynamo.utils.counters["stats"]["unique_graphs"] - before
    _log(
        f"rank={rank} completed width={plan['width']} rows={plan['rows']} "
        f"seconds={time.monotonic() - start:.3f} unique_graphs={compiled}"
    )
    return True


def install(module: ModuleType) -> bool:
    """Attach the opt-in warmup after Megatron's existing startup JIT warmup."""
    if os.environ.get("MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU") != "1":
        return False
    original = module._warmup_jit_function
    if getattr(original, "_mcore_inductor_prewarm", False):
        return False
    attempted = False

    @functools.wraps(original)
    def warmed(*args, **kwargs):
        nonlocal attempted
        result = original(*args, **kwargs)
        if not attempted:
            attempted = True
            warmup(module.get_args())
        return result

    warmed._mcore_inductor_prewarm = True
    module._warmup_jit_function = warmed
    return True
