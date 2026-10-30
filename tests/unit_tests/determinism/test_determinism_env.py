# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Env-var handling in ``apply_determinism_env``.

These run on a plain dict rather than ``os.environ``, so they need no GPU and no
process group -- which is the point: the SPCX rule below guards against a silent
multi-node hang, and a guard that can only be exercised on 64 nodes never gets
exercised.
"""

import pytest

from megatron.training.determinism import DETERMINISM_ENV_VAR_DEFAULTS, apply_determinism_env


class TestDeterminismEnvDefaults:
    def test_fills_canonical_defaults_when_unset(self):
        env = {}
        apply_determinism_env(env)
        for name, value in DETERMINISM_ENV_VAR_DEFAULTS.items():
            assert env[name] == value

    def test_launcher_value_wins_over_default(self):
        env = {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
        apply_determinism_env(env)
        assert env["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


class TestRingSpcxTripwire:
    """Ring on the spcx plugin hangs silently; fail at startup instead."""

    def test_ring_with_spcx_and_no_remedy_fails(self):
        with pytest.raises(AssertionError, match="known deadlock"):
            apply_determinism_env({"NCCL_ALGO": "Ring", "NCCL_NET_PLUGIN": "spcx"})

    @pytest.mark.parametrize(
        "remedy", [{"NCCL_IB_SKIP_RECV_COMPLETION": "0"}, {"NCCL_PROTO": "simple"}]
    )
    def test_either_measured_remedy_is_accepted(self, remedy):
        apply_determinism_env({"NCCL_ALGO": "Ring", "NCCL_NET_PLUGIN": "spcx", **remedy})

    def test_default_empty_algo_is_unaffected(self):
        """The default never trips it — that is the point of not selecting Ring."""
        apply_determinism_env({"NCCL_NET_PLUGIN": "spcx"})

    def test_ring_without_spcx_is_fine(self):
        apply_determinism_env({"NCCL_ALGO": "Ring", "NCCL_NET_PLUGIN": "none"})
