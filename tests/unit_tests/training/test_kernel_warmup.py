# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from argparse import Namespace

import pytest
import torch

from megatron.core.fusions import fused_cross_entropy
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.training import kernel_warmup
from megatron.training.kernel_warmup import warmup_training_kernels
from tests.unit_tests.test_utilities import Utils


def configuration(tp_group, dtype=torch.float32):
    return Namespace(
        seq_length=8,
        context_parallel_size=1,
        micro_batch_size=2,
        padded_vocab_size=128 * tp_group.size(),
        logit_dtype=dtype,
        params_dtype=dtype,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="native",
        squared_relu=True,
        quick_geglu=False,
        use_te_activation_func=False,
        use_fused_weighted_squared_relu=True,
        activation_func_tanh_clamp_scale=16.0,
        swiglu=False,
        bias_swiglu_fusion=True,
        bias_gelu_fusion=True,
        add_bias_linear=False,
        ffn_hidden_size=64 * tp_group.size(),
        moe_shared_expert_intermediate_size=32 * tp_group.size(),
        hidden_size=32,
        bias_dropout_fusion=True,
        sequence_parallel=False,
        fp32_residual_connection=False,
        hidden_dropout=0.1,
        activation_func_fp8_input_store=False,
        activation_func_clamp_value=None,
    )


@pytest.fixture(scope="module")
def tp_group():
    # Utils guards on is_initialized, since the runner and earlier tests may
    # already hold the default group. Teardown belongs to conftest's session
    # cleanup: destroying the group here would pull it out from under every
    # test that runs afterwards.
    Utils.initialize_distributed()
    return torch.distributed.group.WORLD


@pytest.fixture(autouse=True)
def reset_compilation_cache():
    # Each case represents a separate training configuration. Accumulating all
    # dtype/layout variants can exceed Dynamo's limit and silently test eager code.
    torch._dynamo.reset()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_warmup_preserves_rng_and_live_storage(tp_group, dtype, monkeypatch):
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    live = torch.full((1024,), 7.0, pin_memory=True)
    address = live.data_ptr()
    calls = []
    original = torch.accelerator.memory.empty_host_cache

    def trim():
        calls.append(True)
        original()

    monkeypatch.setattr(torch.accelerator.memory, "empty_host_cache", trim)
    layouts = []
    scale_gradients = fused_cross_entropy._scale_gradients_inplace

    def observe_gradient(grad_input, grad_output):
        layouts.append((grad_output.shape, grad_output.stride()))
        return scale_gradients(grad_input, grad_output)

    monkeypatch.setattr(fused_cross_entropy, "_scale_gradients_inplace", observe_gradient)
    args = configuration(tp_group, dtype)
    args.seq_length = 16
    args.context_parallel_size = 2
    warmup_training_kernels(args, tp_group)
    warmup_layouts = layouts.copy()
    layouts.clear()
    assert calls == [True]
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert live.data_ptr() == address and torch.all(live == 7.0)

    # Reuse the warmed shape with different values and compare distributed CE
    # forward/backward against an unsharded PyTorch reference.
    vocab = 128 * tp_group.size()
    full = torch.linspace(-1, 1, vocab, device="cuda", dtype=dtype).expand(8, 2, -1)
    local = full[..., tp_group.rank() * 128 : (tp_group.rank() + 1) * 128]
    logits = local.clone().requires_grad_(True)
    labels = torch.arange(16, device="cuda").view(8, 2) % vocab
    # Exercise the actual model loss wrapper, rather than assuming the layout
    # of the gradient that it supplies to the fused CE backward.
    model = Namespace(config=args, pg_collection=Namespace(tp=tp_group))
    loss = LanguageModule.compute_language_model_loss(model, labels.T.contiguous(), logits)
    grad = torch.autograd.grad(loss, logits, grad_outputs=torch.ones_like(loss))[0]
    assert warmup_layouts == layouts * 2
    assert layouts == [(torch.Size([8, 2]), (1, 8))]
    reference = full.float().contiguous().requires_grad_(True)
    expected = (
        torch.nn.functional.cross_entropy(
            reference.flatten(0, 1), labels.flatten(), reduction="none"
        )
        .view(8, 2)
        .T.contiguous()
    )
    expected_grad = torch.autograd.grad(expected, reference, torch.ones_like(expected))[0]
    expected_grad = expected_grad[..., tp_group.rank() * 128 : (tp_group.rank() + 1) * 128]
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad, expected_grad.to(dtype), rtol=1e-5, atol=1e-5)


def test_warmup_rejects_invalid_shape(tp_group):
    args = configuration(tp_group)
    args.seq_length = 0
    with pytest.raises(ValueError, match="positive"):
        warmup_training_kernels(args, tp_group)
    if tp_group.size() > 1:
        args.seq_length = 8
        args.padded_vocab_size = 129
        with pytest.raises(ValueError, match="divisible"):
            warmup_training_kernels(args, tp_group)


@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
def test_other_configurations_and_later_compilation(tp_group, activation, monkeypatch):
    args = configuration(tp_group, torch.bfloat16)
    args.cross_entropy_loss_fusion = False
    args.squared_relu = False
    args.activation_func_tanh_clamp_scale = None
    args.swiglu = activation == "swiglu"
    args.add_bias_linear = True
    args.sequence_parallel = True
    args.fp32_residual_connection = True

    def unused_ce(*unused):
        pytest.fail("Disabled CE must not be warmed up")

    monkeypatch.setattr(kernel_warmup, "_warmup_cross_entropy", unused_ce)
    warmup_training_kernels(args, tp_group)

    # A new specialization must still compile after startup workers shut down.
    @torch.compile
    def later(x):
        return torch.cos(x) * 3.25 + 1

    x = torch.linspace(-1, 1, 97, device="cuda")
    torch.testing.assert_close(later(x), torch.cos(x) * 3.25 + 1)


@pytest.mark.parametrize("activation", ["te", "quick_geglu"])
def test_does_not_warmup_unused_activation(tp_group, activation, monkeypatch):
    args = configuration(tp_group)
    args.cross_entropy_loss_fusion = False
    args.bias_dropout_fusion = False
    args.squared_relu = False
    args.activation_func_tanh_clamp_scale = None
    args.use_te_activation_func = activation == "te"
    args.bias_gelu_fusion = activation != "te"
    if activation == "quick_geglu":
        args.quick_geglu = True
        args.add_bias_linear = True

    def unused_activation(*unused):
        pytest.fail("Must not warm up a native activation that the model will not use")

    monkeypatch.setattr(kernel_warmup, "_warmup_activation", unused_activation)
    warmup_training_kernels(args, tp_group)
