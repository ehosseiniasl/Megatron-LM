# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Expert-only checkpoint boundaries, gradients, and real TE saved activation coverage."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import megatron.core.extensions.transformer_engine as te_extension
import megatron.core.transformer.moe.experts as experts_module
from megatron.core.activations import squared_relu
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class _OffloadPolicy:
    """Keep real saved-tensor hooks while replacing CUDA transfers with a policy decision."""

    def __init__(self, resident, cpu_offload):
        self.resident = resident
        self.cpu_offload = cpu_offload
        self.scopes = []

    def __call__(self, enabled, tensor, name):
        policy = self

        class Scope:
            def __init__(self):
                self.saved = []
                self.decisions = 0
                self.commits = []

            def __enter__(self):
                if enabled:
                    self.previous = policy.cpu_offload.CPUOffloadEnabled
                    policy.cpu_offload.CPUOffloadEnabled = True

                    def pack(value):
                        self.saved.append(
                            (tuple(value.shape), value.numel() * value.element_size())
                        )
                        return value.detach()

                    self.hooks = torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value)
                    self.hooks.__enter__()
                return tensor

            def __exit__(self, *args):
                if enabled:
                    self.hooks.__exit__(*args)
                    policy.cpu_offload.CPUOffloadEnabled = self.previous

            def should_recompute_instead_of_offload(self):
                self.decisions += 1
                return policy.resident

            def group_offload(self, output, forced_released_tensors=None, **kwargs):
                if enabled:
                    self.commits.append(forced_released_tensors)
                return output

        scope = Scope()
        if enabled:
            assert name == "fused_group_mlp"
            self.scopes.append(scope)
        return scope


class _GroupedLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, skip_bias_add, cpu_offload):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, output_size, input_size) * 0.1)
        self.bias = torch.nn.Parameter(torch.randn(2, output_size) * 0.1)
        self.skip_bias_add = skip_bias_add
        self.use_bias = True
        self.cpu_offload = cpu_offload
        self.calls = []

    def forward(self, hidden, counts):
        self.calls.append((torch.is_grad_enabled(), self.cpu_offload.CPUOffloadEnabled))
        splits = counts.tolist() if isinstance(counts, torch.Tensor) else counts
        output = torch.cat(
            [
                torch.nn.functional.linear(
                    chunk, self.weight[index], None if self.skip_bias_add else self.bias[index]
                )
                for index, chunk in enumerate(torch.split(hidden, splits))
            ]
        )
        bias = self.bias if isinstance(counts, torch.Tensor) else list(self.bias.unbind())
        return output, bias if self.skip_bias_add else None


def _toy_experts(fused, grouped_counts, cpu_offload, recompute):
    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        recompute_granularity="selective",
        recompute_modules=["moe_act", "moe_experts"] if recompute else [],
        fp8="e4m3" if fused else None,
        fp4=None,
        moe_paged_stash=False,
        moe_apply_probs_on_input=False,
        gated_linear_unit=False,
        use_te_activation_func=True,
        delay_offload_until_cuda_graph=False,
    )
    module.tp_group = object()
    module._with_fused_impl = fused
    module._use_grouped_tensor = grouped_counts
    module.offload_fused_group_mlp = fused
    module.offload_expert_fc1 = not fused
    module.offload_moe_act = not fused
    module.activation_recompute = False
    module.activation_func = squared_relu
    module.linear_fc1 = _GroupedLinear(4, 8, False, cpu_offload)
    module.linear_fc2 = _GroupedLinear(8, 4, True, cpu_offload)

    def ops(hidden, counts, probs, *fc2_inputs):
        fc1_output, _ = module.linear_fc1(hidden, counts)
        activation = squared_relu(fc1_output) * probs.unsqueeze(-1)
        output, bias = module.linear_fc2(activation, counts)
        assert len(fc2_inputs) == 2
        return TEGroupedMLP._apply_bias(output, bias, counts, probs)

    module._fused_ops = (ops,)
    return module


@pytest.mark.parametrize("fused,grouped_counts", [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize("resident", [False, True])
def test_disabled_expert_recompute_preserves_math_hooks_and_commits(
    monkeypatch, fused, grouped_counts, resident
):
    """Ordinary expert execution must not decide a policy or enter a new checkpoint."""
    cpu_offload = SimpleNamespace(CPUOffloadEnabled=False)
    monkeypatch.setattr(te_extension, "cpu_offload", cpu_offload)
    policy = _OffloadPolicy(resident, cpu_offload)
    monkeypatch.setattr(experts_module, "off_interface", policy)
    monkeypatch.setattr(experts_module, "skip_routed_expert_padding", lambda config: True)

    def unexpected_checkpoint(*args, **kwargs):
        raise AssertionError("Disabled moe_experts must not checkpoint expert compute")

    monkeypatch.setattr(TEGroupedMLP, "_checkpoint_expert_mlp", unexpected_checkpoint)
    reference = _toy_experts(fused, grouped_counts, cpu_offload, False)
    target = _toy_experts(fused, grouped_counts, cpu_offload, False)
    target.load_state_dict(reference.state_dict())
    hidden = torch.randn(5, 4, requires_grad=True)
    probs = torch.rand(5, requires_grad=True)
    actual_hidden = hidden.detach().clone().requires_grad_(True)
    actual_probs = probs.detach().clone().requires_grad_(True)
    counts = torch.tensor([2, 3])

    # Independent grouped-MLP reference, including FC2 bias scaled by router scores.
    expected_chunks = []
    for index, (chunk, scores) in enumerate(
        zip(torch.split(hidden, counts.tolist()), torch.split(probs, counts.tolist()))
    ):
        fc1 = torch.nn.functional.linear(
            chunk, reference.linear_fc1.weight[index], reference.linear_fc1.bias[index]
        )
        activation = squared_relu(fc1) * scores.unsqueeze(-1)
        fc2 = torch.nn.functional.linear(activation, reference.linear_fc2.weight[index])
        expected_chunks.append(fc2 + reference.linear_fc2.bias[index] * scores.unsqueeze(-1))
    expected = torch.cat(expected_chunks)
    actual, output_bias = target(actual_hidden, counts, actual_probs)
    expected.square().sum().backward()
    actual.square().sum().backward()

    assert output_bias is None
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_hidden.grad, hidden.grad)
    torch.testing.assert_close(actual_probs.grad, probs.grad)
    for actual_param, expected_param in zip(target.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_param.grad, expected_param.grad)
    assert len(policy.scopes) == 1
    scope = policy.scopes[0]
    assert scope.decisions == 0
    assert len(scope.commits) == 1
    assert len(scope.commits[0]) == (1 if fused else 2)
    assert scope.commits[0][0] is actual_hidden
    assert target.linear_fc1.calls == [(True, True)]
    # The unfused FC2 must remain outside the group hooks, as before this feature.
    assert target.linear_fc2.calls == [(True, fused)]
    assert not cpu_offload.CPUOffloadEnabled


@pytest.mark.parametrize("fused,grouped_counts", [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize("resident", [False, True])
def test_expert_recompute_repeated_forward_gradients_and_hook_isolation(
    monkeypatch, fused, grouped_counts, resident
):
    """Different MTP counts/probs survive two forwards; replay never joins another group."""
    cpu_offload = SimpleNamespace(CPUOffloadEnabled=False)
    monkeypatch.setattr(te_extension, "cpu_offload", cpu_offload)
    policy = _OffloadPolicy(resident, cpu_offload)
    monkeypatch.setattr(experts_module, "off_interface", policy)
    monkeypatch.setattr(experts_module, "skip_routed_expert_padding", lambda config: True)
    checkpoint_calls = []

    def te_checkpoint(function, distribute, tracker, tp_group, *args, **kwargs):
        assert distribute is False
        assert kwargs == {"use_reentrant": True}
        checkpoint_calls.append((tp_group, args))
        return checkpoint(function, *args, use_reentrant=True)

    monkeypatch.setattr(experts_module, "te_checkpoint", te_checkpoint)
    reference = _toy_experts(fused, grouped_counts, cpu_offload, False)
    target = _toy_experts(fused, grouped_counts, cpu_offload, True)
    target.activation_recompute = resident
    target.load_state_dict(reference.state_dict())
    base_hidden = [torch.randn(5, 4), torch.randn(5, 4)]
    base_probs = [torch.rand(5), torch.rand(5)]
    counts = [torch.tensor([2, 3]), torch.tensor([4, 1])]

    def run(module):
        hidden = [value.clone().requires_grad_(True) for value in base_hidden]
        probs = [value.clone().requires_grad_(True) for value in base_probs]
        outputs = [module(x, c, p)[0] for x, c, p in zip(hidden, counts, probs)]
        loss = sum(output.square().sum() for output in outputs)
        foreign_saves = []

        def foreign_pack(tensor):
            foreign_saves.append(tensor.shape)
            return tensor.detach()

        cpu_offload.CPUOffloadEnabled = True
        with torch.autograd.graph.saved_tensors_hooks(foreign_pack, lambda value: value):
            loss.backward()
        assert cpu_offload.CPUOffloadEnabled
        cpu_offload.CPUOffloadEnabled = False
        return outputs, hidden, probs, foreign_saves

    expected = run(reference)
    policy.scopes.clear()
    actual = run(target)
    for group_index, (actual_group, expected_group) in enumerate(zip(actual[:3], expected[:3])):
        for got, want in zip(actual_group, expected_group):
            torch.testing.assert_close(
                got if group_index == 0 else got.grad, want if group_index == 0 else want.grad
            )
    for target_param, reference_param in zip(target.parameters(), reference.parameters()):
        torch.testing.assert_close(target_param.grad, reference_param.grad)
    assert actual[3] == []
    assert len(policy.scopes) == 2  # Replay must not create offload start/commit records.
    assert all(scope.decisions == 1 and len(scope.commits) == 1 for scope in policy.scopes)
    if resident:
        assert len(checkpoint_calls) == 2
        assert all(group is target.tp_group for group, _ in checkpoint_calls)
        assert all(scope.commits == [[]] for scope in policy.scopes)
        assert all(len(scope.saved) == 3 for scope in policy.scopes)
        assert target.linear_fc1.calls == [(False, False)] * 2 + [(True, False)] * 2
        # Activation-only checkpointing must never be entered by the whole-MLP replay.
        assert not hasattr(target, "activation_checkpoint")
    else:
        assert checkpoint_calls == []
        assert len(target.linear_fc1.calls) == 2
        assert all(scope.commits[0] for scope in policy.scopes)


@pytest.mark.parametrize("num_tokens", [0, 5])
def test_expert_recompute_preserves_parameter_gradients_with_frozen_inputs(monkeypatch, num_tokens):
    cpu_offload = SimpleNamespace(CPUOffloadEnabled=False)
    monkeypatch.setattr(te_extension, "cpu_offload", cpu_offload)
    policy = _OffloadPolicy(True, cpu_offload)
    monkeypatch.setattr(experts_module, "off_interface", policy)
    monkeypatch.setattr(experts_module, "skip_routed_expert_padding", lambda config: True)
    monkeypatch.setattr(
        experts_module,
        "te_checkpoint",
        lambda function, distribute, tracker, tp_group, *args, **kwargs: checkpoint(
            function, *args, **kwargs
        ),
    )
    module = _toy_experts(True, True, cpu_offload, True)
    output, _ = module(
        torch.randn(num_tokens, 4), torch.tensor([0, num_tokens]), torch.rand(num_tokens)
    )
    output.square().sum().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert policy.scopes[0].decisions == 1
    assert policy.scopes[0].commits == [[]]
    assert policy.scopes[0].saved[-1] == ((0,), 0)


def test_expert_recompute_masks_and_restores_te_offload_context(monkeypatch):
    cpu_offload = SimpleNamespace(CPUOffloadEnabled=True)
    monkeypatch.setattr(te_extension, "cpu_offload", cpu_offload)
    monkeypatch.setattr(
        experts_module,
        "te_checkpoint",
        lambda function, distribute, tracker, tp_group, *args, **kwargs: checkpoint(
            function, *args, **kwargs
        ),
    )
    module = _toy_experts(False, True, cpu_offload, True)
    weight = torch.nn.Parameter(torch.ones(4))
    replay_states = []

    def forward(hidden, counts, probs):
        replay_states.append(cpu_offload.CPUOffloadEnabled)
        return hidden * weight * probs[:, None]

    hidden = torch.ones(2, 4, requires_grad=True)
    output = module._checkpoint_expert_mlp(forward, hidden, torch.tensor([1, 1]), torch.ones(2))
    output.sum().backward()
    assert replay_states == [False, False]
    assert cpu_offload.CPUOffloadEnabled
    torch.testing.assert_close(weight.grad, torch.full((4,), 2.0))

    def raise_error(*args):
        raise RuntimeError("expert forward failed")

    with pytest.raises(RuntimeError, match="expert forward failed"):
        module._checkpoint_expert_mlp(raise_error, hidden, torch.tensor([1, 1]), torch.ones(2))
    assert cpu_offload.CPUOffloadEnabled


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_expert_recompute_uses_te_phase_for_gtp(monkeypatch):
    """GTP's replay/prefetch detector must see BF16 recomputation as well as FP8."""
    te_distributed = pytest.importorskip("transformer_engine.pytorch.distributed")
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(1, 1)
    try:
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        policy = _OffloadPolicy(True, te_extension.cpu_offload)
        monkeypatch.setattr(experts_module, "off_interface", policy)
        monkeypatch.setattr(experts_module, "skip_routed_expert_padding", lambda config: True)
        module = _toy_experts(False, True, te_extension.cpu_offload, True).cuda().bfloat16()
        module.tp_group = get_default_pg_collection().expt_tp
        phases = []
        module.linear_fc1.register_forward_pre_hook(
            lambda *args: phases.append(te_distributed.in_fp8_activation_recompute_phase())
        )
        hidden = torch.randn(5, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        probs = torch.rand(5, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        output, _ = module(hidden, torch.tensor([2, 3], device="cuda"), probs)
        output.float().square().sum().backward()
        assert phases == [False, True]
        assert hidden.grad is not None and probs.grad is not None
        assert all(parameter.grad is not None for parameter in module.parameters())
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_real_te_mxfp8_expert_recompute_gradients_and_saved_bytes(monkeypatch):
    """Run the real clamped fused kernel and TE checkpoint; retain only boundary tensors."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 fused grouped MLP requires Blackwell")
    if int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
        pytest.skip("Set NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 before importing TE")
    te = pytest.importorskip("transformer_engine.pytorch")
    grouped_mlp = pytest.importorskip("transformer_engine.pytorch.ops.fused.grouped_mlp")
    from transformer_engine.common.recipe import MXFP8BlockScaling

    if not grouped_mlp.GroupedMLP_CuTeGEMMUnary.is_supported():
        pytest.skip("Installed TE reports fused unary grouped MLP unsupported")
    if not experts_module._te_supports_scaled_tanh_srelu():
        pytest.skip("Installed TE cannot fuse clamped squared ReLU")
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(1, 1)
    try:
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            ffn_hidden_size=512,
            num_attention_heads=4,
            num_moe_experts=2,
            add_bias_linear=False,
            activation_func=squared_relu,
            use_fused_weighted_squared_relu=True,
            activation_func_tanh_clamp_scale=16.0,
            use_transformer_engine_op_fuser=True,
            moe_grouped_gemm=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            moe_router_topk=1,
            fp8="e4m3",
            fp8_recipe="mxfp8",
            fine_grained_activation_offloading=True,
            offload_modules=["fused_group_mlp"],
            recompute_granularity="selective",
            recompute_modules=["moe_act", "moe_experts"],
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(2, moe_grouped_gemm=True).mlp
        )
        experts = TEGroupedMLP(
            2,
            config,
            get_submodules(submodules.experts),
            1,
            pg_collection=get_default_pg_collection(),
        ).cuda()
        assert experts._with_fused_impl
        policy = _OffloadPolicy(False, te_extension.cpu_offload)
        monkeypatch.setattr(experts_module, "off_interface", policy)
        base_hidden = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
        base_probs = torch.rand(512, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn_like(base_hidden) * 0.1
        counts = torch.tensor([256, 256], device="cuda", dtype=torch.int64)

        def run(resident):
            policy.resident = resident
            policy.scopes.clear()
            experts.zero_grad(set_to_none=True)
            x = base_hidden.clone().requires_grad_(True)
            p = base_probs.clone().requires_grad_(True)
            with te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
                output, _ = experts(x, counts, p)
            saved = list(policy.scopes[0].saved)
            output.backward(grad_output)
            grads = [
                param.grad.detach().clone() for param in experts.parameters() if param.requires_grad
            ]
            return output.detach(), x.grad, p.grad, grads, saved

        baseline = run(False)
        target = run(True)
        selected = [
            type(entry[0] if isinstance(entry, (tuple, list)) else entry).__name__
            for group in experts._fused_ops[0]._module_groups
            for entry in group._forward_ops
        ]
        assert "GroupedMLP_CuTeGEMMUnary" in selected, selected
        for got, want in zip(target[:3], baseline[:3]):
            torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-4)
        for got, want in zip(target[3], baseline[3]):
            torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-4)
        assert len(target[4]) == 3
        expected_bytes = sum(
            t.numel() * t.element_size() for t in (base_hidden, counts, base_probs)
        )
        assert sum(size for _, size in target[4]) == expected_bytes
        assert sum(size for _, size in baseline[4]) > 2 * expected_bytes
        assert policy.scopes[0].commits == [[]]
    finally:
        Utils.destroy_model_parallel()
