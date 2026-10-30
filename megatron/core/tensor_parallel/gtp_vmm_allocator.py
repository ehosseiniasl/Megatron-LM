# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CUDA VMM allocator for the GTP symmetric-memory pools.

Implements NCCL's requirements on user-allocated communication buffers
(https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#memory-allocator)
minimally: VMM allocations at the recommended granularity, exportable handle
types (POSIX FD, plus FABRIC when supported — dropped on retry if cuMemCreate
rejects it), and GPUDirect-RDMA-capable physical memory when supported.

Unlike ``ncclMemAlloc``, the memory is mapped only on the allocation's device
— the requirements ask for no more, and ncclMemAlloc's extra mappings on every
P2P-visible peer GPU slow CPU-side kernel launching for the whole training
step.

Window registration is still delegated to ``nccl_allocator.register_mem_pool``
(which calls ``ncclCommWindowRegister``), which accepts this memory and runs its
symmetric kernels on it.
"""

import logging
import os

import torch

# This import is needed for the cpp extension to work.
# pylint: disable=unused-import
from torch.utils import cpp_extension

import megatron.core.nccl_allocator as nccl_allocator
from megatron.core.nccl_allocator import get_func_args
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_allocator = None


def _load_vmm_allocator_extension():
    """Build/load the extension without creating an allocator (also used when baking).

    A configured extension-cache root must belong to the matching image/toolchain.
    Keep Torch's source and build-option checks when consuming cached files.
    """
    # load_inline writes main.cpp before acquiring PyTorch's build lock, so ranks
    # can overwrite the source while another rank compiles it. Use a packaged
    # source file that is never rewritten at runtime; load serializes the build.
    source_path = os.path.join(os.path.dirname(__file__), "csrc", "gtp_vmm_allocator.cpp")
    extension_root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if extension_root is None:
        module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        extension_root = os.path.join(module_dir, "build")
    build_dir = os.path.join(extension_root, "gtp_vmm_allocator")
    os.makedirs(build_dir, exist_ok=True)
    try:
        return torch.utils.cpp_extension.load(
            name="gtp_vmm_allocator",
            sources=[source_path],
            with_cuda=True,
            extra_ldflags=["-lcuda"],
            verbose=True,
            is_python_module=True,
            build_directory=build_dir,
        )
    except Exception as e:
        raise RuntimeError(
            "[GTP] Failed to build the GTP VMM allocator extension; "
            "--gtp-remat-nccl-ub/--gtp-expert-remat-nccl-ub require nvcc and libcuda "
            "at runtime."
        ) from e


def _build_vmm_allocator():
    global _allocator
    if _allocator is not None:
        return
    _allocator = _load_vmm_allocator_extension().get_vmm_allocator()


def create_vmm_mem_pool() -> torch.cuda.MemPool:
    """
    Create a symmetric memory pool using the VMM allocator. GTP symmetric pools are
    always symmetric (register_gtp_symm_pool enforces the torch >= 2.9 floor).
    """
    _build_vmm_allocator()
    assert _allocator is not None, "VMM allocator is not initialized"
    if 'symmetric' in get_func_args(torch.cuda.MemPool):
        # PyTorch >= 2.9.0a0 and before PyTorch PR #161238 takes the symmetric knob at
        # MemPool construction; since #161238 it lives in the registration function.
        return torch.cuda.MemPool(_allocator, symmetric=True)
    if 'symm_mem' in get_func_args(torch.cuda.MemPool):
        # Argument name divergence between nvidia pytorch and the official pytorch.
        return torch.cuda.MemPool(_allocator, symm_mem=True)
    # The symmetric knob is in the registration function.
    return torch.cuda.MemPool(_allocator)


def init() -> None:
    """
    Initialize the VMM allocator, including the NCCL environment its pools are
    registered under (same settings as nccl_allocator.init()).
    """
    # Enables NCCL NVLS algorithm
    os.environ["NCCL_NVLS_ENABLE"] = "1"
    # Disables the use of the tensor register allocator hook
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "0"
    _build_vmm_allocator()
    log_single_rank(logger, logging.INFO, "[MCORE][GTP] Initialized the VMM Allocator")


def register_mem_pool(pool: torch.cuda.MemPool, group) -> None:
    """
    Window-register a VMM pool's segments on ``group`` (always symmetric).
    Delegating to nccl_allocator is safe because its (de)registration walks the
    pool's segments and never touches the allocator that produced them.
    """
    nccl_allocator.register_mem_pool(pool, group, symmetric=True)


def deregister_mem_pool(pool: torch.cuda.MemPool, group) -> None:
    """
    Deregister a VMM pool's windows from ``group``. Delegates to nccl_allocator.
    """
    nccl_allocator.deregister_mem_pool(pool, group)
