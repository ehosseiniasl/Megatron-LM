# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``NCCL_ALGO=""`` must survive torch determinism, or NCCL loses its symmetric kernels.

Rationale and measurements: docs/user-guide/deterministic-training.md, section
"Why NCCL_ALGO defaults to empty".
"""

import ctypes
import os

import pytest
import torch

from megatron.training.determinism import (
    NCCL_ALGO_DEFAULT,
    apply_determinism_env,
    assert_nvls_not_banned,
    libc_getenv,
)
from tests.unit_tests.test_utilities import Utils

_libc = ctypes.CDLL(None)
_libc.unsetenv.argtypes = [ctypes.c_char_p]


def _clear_nccl_algo():
    """Clear in libc as well as os.environ — PyTorch's write bypasses os.environ."""
    os.environ.pop("NCCL_ALGO", None)
    _libc.unsetenv(b"NCCL_ALGO")


def _require_torchrun():
    if "RANK" not in os.environ:
        pytest.skip("launch under torchrun (needs RANK/WORLD_SIZE)")


def _new_group_all_ranks():
    return torch.distributed.new_group(list(range(torch.distributed.get_world_size())))


class TestNcclAlgoDefault:
    def test_default_is_empty_not_absent(self):
        """Absent lets ProcessGroupNCCL export ^NVLS; any real value loses NVLS."""
        env = {}
        apply_determinism_env(env)
        assert env["NCCL_ALGO"] == NCCL_ALGO_DEFAULT == ""

    def test_launcher_can_opt_back_into_ring(self):
        """Ring restores the by-construction guarantee, at the cost of the kernels."""
        env = {"NCCL_ALGO": "Ring"}
        apply_determinism_env(env)
        assert env["NCCL_ALGO"] == "Ring"

    def test_empty_passes_validation(self):
        apply_determinism_env({"NCCL_ALGO": ""})

    def test_unaccepted_token_still_rejected(self):
        with pytest.raises(AssertionError, match="NCCL_ALGO"):
            apply_determinism_env({"NCCL_ALGO": "Tree"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestNvlsBanSuppression:
    """PyTorch's ``set_env("NCCL_ALGO", "^NVLS")`` must never run for our groups.

    It is guarded by ``if (!get_env("NCCL_ALGO").has_value())``, so an empty value keeps
    it unreached. Both arms build a real ``ProcessGroupNCCL`` — that constructor is the
    only place the write happens.
    """

    def test_ban_fires_when_nccl_algo_is_unset(self, monkeypatch):
        """Negative control: without our default, PyTorch DOES write ^NVLS.

        Without this the positive test below could pass for the wrong reason — e.g. if a
        PyTorch version stopped banning NVLS entirely, or if no group was really built.
        """
        _require_torchrun()
        Utils.initialize_distributed()
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            _clear_nccl_algo()
            torch.use_deterministic_algorithms(True)
            group = _new_group_all_ranks()
            assert libc_getenv("NCCL_ALGO") == "^NVLS", (
                f"expected PyTorch to ban NVLS, got NCCL_ALGO={libc_getenv("NCCL_ALGO")!r}. "
                "If upstream dropped the ban, the suppression below is now a no-op and "
                "megatron/training/determinism.py should be simplified."
            )
            torch.distributed.destroy_process_group(group)
        finally:
            torch.use_deterministic_algorithms(was_deterministic)
            _clear_nccl_algo()

    def test_empty_nccl_algo_suppresses_the_ban(self, monkeypatch):
        """Positive: an empty value keeps the write unreached, for every group."""
        _require_torchrun()
        Utils.initialize_distributed()
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            monkeypatch.setenv("NCCL_ALGO", "")
            torch.use_deterministic_algorithms(True)

            # Several constructions, including one "late" group standing in for the
            # lazily-built eval/inference/expert groups that run after startup.
            for _ in range(2):
                group = _new_group_all_ranks()
                assert libc_getenv("NCCL_ALGO") == "", (
                    f"NCCL_ALGO is {libc_getenv("NCCL_ALGO")!r} after building a group under "
                    "determinism; the symmetric NVLS/LDMC kernels are lost from here on"
                )
                # Also exercise the shipped guard, not just our own assertion.
                assert_nvls_not_banned()
                torch.distributed.destroy_process_group(group)
        finally:
            torch.use_deterministic_algorithms(was_deterministic)
            _clear_nccl_algo()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestSymmetricKernelIsActuallyUsed:
    """End to end: the symmetric kernel must actually run, not just be permitted.

    The tests above check the env var survives. This one checks the consequence -- NCCL
    dispatches the reduce-scatter to ``ncclSymkDevKernel_*`` rather than
    ``ncclDevKernel_*_RING`` -- by reading the kernel name out of torch.profiler.

    Needs NVLink multicast and a registered symmetric window, so it skips where the
    control arm cannot produce a symmetric kernel either.
    """

    @staticmethod
    def _reduce_scatter_kernels(group, count=688128, dtype=torch.bfloat16):
        """Run a reduce-scatter on registered pool memory; return the NCCL kernel names."""
        from torch.profiler import ProfilerActivity, profile

        from megatron.core.tensor_parallel.gtp_symmetric_memory import (
            deregister_and_clear_gtp_symm_pools,
            gtp_symm_pool_ctx,
            register_gtp_symm_pool,
        )

        try:
            register_gtp_symm_pool(group)
        except Exception as exc:  # no NCCL window registration here
            pytest.skip(f"NCCL symmetric registration unavailable: {exc}")

        world = torch.distributed.get_world_size()
        try:
            with gtp_symm_pool_ctx(group):
                inp = torch.ones(count * world, dtype=dtype, device="cuda")
                out = torch.empty(count, dtype=dtype, device="cuda")

            def reduce_scatter():
                for _ in range(3):
                    torch.distributed.reduce_scatter_tensor(out, inp, group=group)

            # Warm up outside the profiler: the first call pays one-time tuner and
            # channel setup and can land on a different kernel than steady state.
            reduce_scatter()
            torch.cuda.synchronize()

            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                reduce_scatter()
                torch.cuda.synchronize()

            names = sorted({e.name for e in prof.events() if "nccl" in e.name.lower()})
            del inp, out
            torch.cuda.synchronize()
            return names
        finally:
            deregister_and_clear_gtp_symm_pools()

    def test_empty_nccl_algo_keeps_the_symmetric_kernel(self, monkeypatch):
        """The kernel must actually be symmetric, measured against a control.

        The control decides whether this box can produce a symmetric kernel at all (it
        needs NVLink multicast and enough ranks). Only if it can does the deterministic
        arm's result become an assertion -- otherwise a machine without multicast would
        fail for the wrong reason, and a blanket skip would let a real regression pass.
        """
        _require_torchrun()
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            # Control: no determinism, NCCL_ALGO untouched -> should reach LDMC.
            _clear_nccl_algo()
            torch.use_deterministic_algorithms(False)
            control_group = _new_group_all_ranks()
            control = self._reduce_scatter_kernels(control_group)
            torch.distributed.destroy_process_group(control_group)
            if not any("Symk" in k for k in control):
                pytest.skip(f"no symmetric kernel available on this machine: {control}")

            # Arm: determinism on, NCCL_ALGO="" -> must still reach LDMC.
            _clear_nccl_algo()
            monkeypatch.setenv("NCCL_ALGO", "")
            torch.use_deterministic_algorithms(True)
            group = _new_group_all_ranks()
            kernels = self._reduce_scatter_kernels(group)
            torch.distributed.destroy_process_group(group)

            assert any(
                "Symk" in k for k in kernels
            ), f"determinism lost the symmetric kernel: control={control} arm={kernels}"
            assert libc_getenv("NCCL_ALGO") == "", "NCCL_ALGO was overwritten"
        finally:
            torch.use_deterministic_algorithms(was_deterministic)
            _clear_nccl_algo()
