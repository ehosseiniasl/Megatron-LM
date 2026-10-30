# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU safety checks for the explicitly enabled startup prewarm hook."""

import importlib.util
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location(
    "inductor_prewarm_tested", REPO_ROOT / "tools/persistent_cache/inductor_prewarm.py"
)
prewarm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prewarm)


def recipe(**changes):
    values = dict(
        bf16=True,
        squared_relu=True,
        use_fused_weighted_squared_relu=True,
        num_experts=512,
        activation_func_tanh_clamp_scale=16.0,
        moe_ffn_hidden_size=4096,
        expert_tensor_parallel_size=1,
        rank=68,
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"bf16": False},
        {"squared_relu": False},
        {"use_fused_weighted_squared_relu": False},
        {"use_te_activation_func": True},
        {"moe_apply_probs_on_input": True},
        {"num_experts": None},
        {"activation_func_tanh_clamp_scale": 0},
        {"activation_func_tanh_clamp_scale": float("nan")},
        {"activation_func_tanh_clamp_scale": float("inf")},
        {"moe_ffn_hidden_size": None},
        {"expert_tensor_parallel_size": 3},
        {"expert_tensor_parallel_size": 0},
        {"mimo_llm_offset": 64},
    ],
)
def test_unsupported_or_unvalidated_recipes_skip(changes):
    plan, reason = prewarm._plan(recipe(**changes), 68)
    assert plan is None and reason


def test_mimo_only_warms_validated_language_interval():
    args = recipe(mimo_llm_offset=64, mimo_llm_world_size=64)
    for rank in (0, 63, 128):
        assert prewarm._plan(args, rank)[0] is None
    for rank in (64, 127):
        assert prewarm._plan(args, rank)[0]["width"] == 4096


def test_real_cli_fusion_flag_with_precision_override_is_supported():
    # bias_activation_fusion is a derived config field, absent from CLI args.
    # Op-fuser can be requested while BF16 overrides select the unfused path.
    args = recipe(
        use_transformer_engine_op_fuser=True,
        te_precision_config_file="bf16_overrides.yaml",
        mimo_llm_offset=64,
        mimo_llm_world_size=64,
    )
    assert not hasattr(args, "bias_activation_fusion")
    plan, reason = prewarm._plan(args, 68)
    assert reason is None
    assert plan["width"] == 4096
    assert plan["rows"][-1] == 524288


def test_plan_exercises_both_indexing_ranges_at_local_expert_width():
    plan, reason = prewarm._plan(
        recipe(moe_ffn_hidden_size=8192, expert_tensor_parallel_size=2), 68
    )
    assert reason is None
    assert plan["width"] == 4096
    assert len(set(plan["rows"])) == 3
    assert all(row % 256 == 0 for row in plan["rows"])
    assert all(row * plan["width"] < 2**31 for row in plan["rows"][:2])
    assert plan["rows"][-1] * plan["width"] >= 2**31


def test_disabled_path_does_not_need_torch_or_megatron(monkeypatch):
    monkeypatch.delenv("MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert prewarm.warmup(recipe()) is False
    assert prewarm.install(ModuleType("empty_module")) is False


def test_install_preserves_native_call_and_runs_once_after_it(monkeypatch):
    monkeypatch.setenv("MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU", "1")
    calls = []
    args = recipe()
    module = ModuleType("initialize")
    module.get_args = lambda: args

    def original(*positional, **keywords):
        calls.append(("native", positional, keywords))
        return "native-result"

    module._warmup_jit_function = original
    monkeypatch.setattr(prewarm, "warmup", lambda args: calls.append(("optional", args)))
    assert prewarm.install(module)
    assert not prewarm.install(module)
    assert module._warmup_jit_function(tp_size=2) == "native-result"
    assert module._warmup_jit_function(tp_size=4) == "native-result"
    assert [call[0] for call in calls] == ["native", "optional", "native"]
    assert calls[0][2] == {"tp_size": 2}
    assert calls[1][1] is args


class FakeTensor:
    def __init__(self, torch, shape, dtype, requires_grad=False, base=None):
        self.torch = torch
        self.shape = shape
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.base = base
        self.view_created_with_grad = torch.state["grad"] if base is not None else None

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def view(self, *shape):
        return FakeTensor(self.torch, shape, self.dtype, self.requires_grad, self)


class FakeTorch(ModuleType):
    def __init__(self, free=256 * 1024**3):
        super().__init__("torch")
        self.state = {"grad": False, "rng": 17}
        self.calls = []
        self.bfloat16, self.float32 = "bf16", "fp32"
        self.cuda = SimpleNamespace(
            current_device=lambda: 0,
            mem_get_info=lambda device: (free, 288 * 1024**3),
            synchronize=lambda device: self.calls.append("synchronize"),
            empty_cache=lambda: self.calls.append("empty_cache"),
        )
        self.random = SimpleNamespace(fork_rng=self.fork_rng)
        self._dynamo = SimpleNamespace(utils=SimpleNamespace(counters={"stats": Counter()}))

    def ones(self, shape, *, device, dtype, requires_grad=False):
        self.calls.append("allocate")
        return FakeTensor(self, shape, dtype, requires_grad)

    def ones_like(self, tensor):
        return self.ones(tensor.shape, device="cuda", dtype=tensor.dtype)

    @contextmanager
    def fork_rng(self, devices):
        previous = self.state["rng"]
        self.calls.append("fork_rng")
        try:
            yield
        finally:
            self.state["rng"] = previous
            self.calls.append("restore_rng")

    @contextmanager
    def _grad_mode(self, enabled):
        previous = self.state["grad"]
        self.state["grad"] = enabled
        try:
            yield
        finally:
            self.state["grad"] = previous

    def enable_grad(self):
        return self._grad_mode(True)

    def no_grad(self):
        return self._grad_mode(False)


def test_memory_preflight_skips_without_allocations(monkeypatch):
    monkeypatch.setenv("MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU", "1")
    torch = FakeTorch(free=40 * 1024**3)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert not prewarm.warmup(recipe())
    assert not torch.calls


def test_installed_hook_calls_real_warmup_after_native_initialization(monkeypatch):
    monkeypatch.setenv("MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU", "1")
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    fusions = ModuleType("megatron.core.fusions.fused_weighted_squared_relu")
    fusions.weighted_clamped_squared_relu = object()
    fusions.weighted_clamped_squared_relu_back = object()
    monkeypatch.setitem(sys.modules, fusions.__name__, fusions)
    native_calls, rows_seen = [], []
    args = recipe(
        use_transformer_engine_op_fuser=True, te_precision_config_file="bf16_overrides.yaml"
    )
    module = ModuleType("initialize")
    module.get_args = lambda: args

    def native(tp_size=None):
        native_calls.append(tp_size)
        return "original-result"

    def record_shape(torch, forward, backward, rows, width, clamp):
        assert native_calls == [2]
        rows_seen.append(rows)

    module._warmup_jit_function = native
    monkeypatch.setattr(prewarm, "_run_shape", record_shape)
    assert prewarm.install(module)
    assert module._warmup_jit_function(tp_size=2) == "original-result"
    assert module._warmup_jit_function(tp_size=3) == "original-result"
    assert native_calls == [2, 3]
    assert rows_seen == [131072, 174592, 524288]
    assert torch.calls.count("fork_rng") == 1
    assert torch.calls.count("empty_cache") == 1


@pytest.mark.parametrize("fail", [False, True])
def test_warmup_preserves_context_and_native_function_modes(monkeypatch, fail):
    monkeypatch.setenv("MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU", "1")
    torch = FakeTorch()
    module = ModuleType("megatron.core.fusions.fused_weighted_squared_relu")
    observed = []

    def forward(x, weights, clamp):
        assert torch.state["grad"] is False
        assert x.view_created_with_grad and x.base is not None
        # Checkpoint recomputation detaches router weights. Warming a view
        # creates Dynamo guards on a base absent from the recomputed weights.
        assert weights.base is None and weights.view_created_with_grad is None
        assert weights.shape == (x.shape[0], 1) and weights.dtype == torch.float32
        assert weights.requires_grad and clamp == 16
        observed.append(("forward", x.requires_grad))
        torch.state["rng"] += 1
        if fail:
            raise RuntimeError("compiler failure")
        return FakeTensor(torch, x.shape, x.dtype)

    def backward(grad, x, weights, clamp):
        assert torch.state["grad"] is False
        assert x.requires_grad and weights.requires_grad and not grad.requires_grad
        assert weights.base is None and weights.shape == (x.shape[0], 1)
        assert grad.view_created_with_grad and grad.base is not None
        observed.append(("backward", x.requires_grad))
        return FakeTensor(torch, x.shape, x.dtype), FakeTensor(torch, weights.shape, weights.dtype)

    module.weighted_clamped_squared_relu = forward
    module.weighted_clamped_squared_relu_back = backward
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    if fail:
        with pytest.raises(RuntimeError, match="compiler failure"):
            prewarm.warmup(recipe())
        assert observed == [("forward", False)]
    else:
        assert prewarm.warmup(recipe())
        assert observed == [("forward", False), ("forward", True), ("backward", True)] * 3
    assert torch.state == {"grad": False, "rng": 17}
    assert torch.calls[-1] == "empty_cache"
    assert "restore_rng" in torch.calls
