# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Simplified sparse-attention phase, precision, and gradient contracts."""

import sys
from argparse import ArgumentParser
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    dsa_layout,
    simplified_sparse_attention,
    simplified_sparse_attention_cute,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import core_transformer_config_from_args
from megatron.training.arguments import add_megatron_arguments


def _config(**overrides):
    values = dict(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=16,
        num_query_groups=1,
        kv_channels=256,
        add_bias_linear=False,
        experimental_attention_variant="ssa",
        dsa_indexer_topk=512,
        dsa_indexer_loss_coeff=0.1,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _inputs(sequence_length=16, heads=16, device="cpu"):
    return tuple(
        torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        for shape in (
            (sequence_length, 1, heads, 256),
            (sequence_length, 1, 1, 256),
            (sequence_length, 1, 1, 256),
            (sequence_length, 1, 1, 128),
            (sequence_length, 1, 1, 128),
        )
    )


def test_cute_rejects_deterministic_mode():
    with pytest.raises(ValueError, match="uses atomics"):
        _config(deterministic_mode=True)


@pytest.mark.parametrize(
    "pattern,selector,expected",
    [
        (Symbols.ATTENTION, "ssa", "ssa"),
        (Symbols.DS_ATTENTION, None, "dsa"),
        (Symbols.DS_ATTENTION, "ssa", "error"),
    ],
)
def test_hybrid_argument_conversion_preserves_attention_selection(pattern, selector, expected):
    parser = ArgumentParser()
    add_megatron_arguments(parser)
    options = [
        "--num-layers",
        "1",
        "--hidden-size",
        "256",
        "--num-attention-heads",
        "16",
        "--group-query-attention",
        "--num-query-groups",
        "1",
        "--kv-channels",
        "256",
        "--disable-bias-linear",
        "--no-bias-gelu-fusion",
        "--no-rope-fusion",
        "--bf16",
        "--dsa-indexer-topk",
        "512",
        "--dsa-indexer-loss-coeff",
        "0.1",
        "--hybrid-layer-pattern",
        pattern,
    ]
    if selector is not None:
        options.extend(["--experimental-attention-variant", selector])
    args = parser.parse_args(options)
    # validate_args normally derives params_dtype from --bf16.
    args.params_dtype = torch.bfloat16

    if expected == "error":
        with pytest.raises(ValueError, match="cannot be combined with MLA DSA 'D' layers"):
            core_transformer_config_from_args(args)
    else:
        config = core_transformer_config_from_args(args)
        assert config.experimental_attention_variant == expected


def _metadata(starts, ends, valid, denominator, **kwargs):
    return SimpleNamespace(
        key_starts=starts, key_ends=ends, query_valid=valid, loss_denominator=denominator, **kwargs
    )


@pytest.mark.parametrize("installed", [None, "0.1.0.dev5", "0.1.0.dev6", "0.1.0"])
def test_cute_kernel_package_requirement(monkeypatch, installed):
    adapter = simplified_sparse_attention_cute

    def package_version(name):
        assert name == "simplified-sparse-attention"
        if installed is None:
            raise adapter.PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(adapter, "version", package_version)
    adapter._check_kernel_package_version.cache_clear()
    try:
        if installed in (None, "0.1.0.dev5"):
            with pytest.raises(
                RuntimeError, match=r"requires simplified-sparse-attention >=0.1.0.dev6"
            ):
                adapter._check_kernel_package_version()
        else:
            adapter._check_kernel_package_version()
    finally:
        adapter._check_kernel_package_version.cache_clear()


@pytest.mark.parametrize("coefficient", [0.0, 0.03])
@pytest.mark.parametrize("per_token", [False, True])
def test_adapter_applies_coefficient_and_normalization_once(monkeypatch, coefficient, per_token):
    monkeypatch.setattr(
        simplified_sparse_attention_cute, "_check_kernel_package_version", lambda: None
    )
    inputs = _inputs()
    seen = {}

    def sparse_attention(q, k, v, qi, ki, layout, topk, *, loss_coeff):
        seen.update(q=q, k=k, v=v, qi=qi, ki=ki, layout=layout)
        return (
            q + k.expand_as(q) + v.expand_as(q),
            (qi.float().square().sum() + ki.float().square().sum())
            * loss_coeff
            / layout.loss_denominator.squeeze(),
        )

    monkeypatch.setitem(
        sys.modules,
        "simplified_sparse_attention",
        SimpleNamespace(RowMetadata=_metadata, sparse_attention=sparse_attention),
    )
    output, loss = simplified_sparse_attention_cute.run_cute_sparse_attention(
        *inputs,
        topk=512,
        loss_coeff=coefficient,
        calculate_per_token_loss=per_token,
        train_indexer_only=False,
    )
    denominator = 1 if per_token else 16
    assert output.shape == (16, 1, 4096)
    for name, original in zip(("q", "k", "v", "qi", "ki"), inputs):
        assert seen[name].data_ptr() == original.data_ptr()
    torch.testing.assert_close(seen["layout"].key_starts, torch.zeros(16, dtype=torch.int64))
    torch.testing.assert_close(seen["layout"].key_ends, torch.arange(1, 17, dtype=torch.int64))
    (output.float().sum() + loss).backward()
    torch.testing.assert_close(inputs[0].grad, torch.ones_like(inputs[0]))
    for tensor in inputs[1:3]:
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 16))
    for tensor in inputs[3:]:
        expected = (2 * tensor.detach().float() * coefficient / denominator).to(tensor.dtype)
        torch.testing.assert_close(tensor.grad, expected)


def test_phase1_adapter_reuses_flash4_teacher_lse_and_full_support(monkeypatch):
    monkeypatch.setattr(
        simplified_sparse_attention_cute, "_check_kernel_package_version", lambda: None
    )
    inputs = _inputs()
    calls = {}
    teacher_output = inputs[0].detach()[:, 0].clone()
    teacher_lse = torch.randn(16, 16)

    def teacher(q, k, v, **kwargs):
        calls["teacher"] = kwargs
        return teacher_output, teacher_lse

    def dense_kl(q, k, qi, ki, **kwargs):
        calls["kl"] = kwargs
        return (qi.float().square().sum() + ki.float().square().sum()) * 0.03 / 16

    def unexpected_sparse(*args, **kwargs):
        pytest.fail("Phase 1 must not invoke selected-support sparse attention")

    monkeypatch.setitem(
        sys.modules,
        "simplified_sparse_attention",
        SimpleNamespace(RowMetadata=_metadata, sparse_attention=unexpected_sparse),
    )
    monkeypatch.setitem(
        sys.modules,
        "simplified_sparse_attention.phase1",
        SimpleNamespace(dense_attention_flash4=teacher, dense_indexer_kl=dense_kl),
    )
    output, loss = simplified_sparse_attention_cute.run_cute_sparse_attention(
        *inputs, topk=512, loss_coeff=0.03, calculate_per_token_loss=False, train_indexer_only=True
    )
    assert calls["kl"]["teacher_lse"] is teacher_lse
    assert calls["kl"]["use_kf"] is True
    assert calls["kl"]["loss_coeff"] == 0.03
    torch.testing.assert_close(calls["kl"]["key_ends"], torch.arange(1, 17))
    torch.testing.assert_close(
        calls["teacher"]["cu_seqlens_q"], torch.tensor([0, 16], dtype=torch.int32)
    )
    assert not output.requires_grad
    loss.backward()
    assert all(tensor.grad is None for tensor in inputs[:3])
    assert all(tensor.grad is not None for tensor in inputs[3:])


class _Linear(torch.nn.Module):
    def __init__(self, input_size, output_size, *, config, **kwargs):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(
            torch.randn(output_size, input_size, dtype=config.params_dtype) * 0.05
        )

    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight), None


def _mock_core(
    monkeypatch, *, phase1=False, coefficient=0.1, per_token=False, layer_number=1, **overrides
):
    monkeypatch.setattr(simplified_sparse_attention, "TELinear", _Linear)
    monkeypatch.setattr(
        simplified_sparse_attention,
        "get_fp8_disabled_context",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        simplified_sparse_attention.DSAIndexerLossAutoScaler, "main_loss_backward_scale", None
    )
    config = _config(
        hidden_size=16,
        ssa_train_indexer_only=phase1,
        dsa_indexer_loss_coeff=coefficient,
        calculate_per_token_loss=per_token,
        **overrides,
    )
    group = SimpleNamespace(size=lambda: 1)
    return simplified_sparse_attention.SimplifiedSparseAttentionCore(
        config,
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=group, cp=group),
        softmax_scale=config.softmax_scale,
    )


def test_indexer_shape_is_fixed_and_dsa_only_options_do_not_change_it(monkeypatch):
    core = _mock_core(
        monkeypatch,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_rotate_activation=True,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_k_norm_epsilon=1e-4,
        dsa_indexer_k_norm_fp32=True,
    )
    assert core.indexer.linear_q.weight.shape == (128, 16)
    assert core.indexer.linear_k.weight.shape == (128, 16)
    assert set(dict(core.named_parameters())) == {
        "indexer.linear_q.weight",
        "indexer.linear_k.weight",
    }


def test_core_rejects_nonstandard_scale_at_construction(monkeypatch):
    with pytest.raises(ValueError, match="softmax scale"):
        _mock_core(monkeypatch, softmax_scale=0.125)


@pytest.mark.parametrize("phase1", [False, True])
def test_both_phases_pass_normalized_indexer_input(phase1):
    owner = SimpleNamespace(config=_config(ssa_train_indexer_only=phase1))
    hidden = torch.randn(16, 1, 256)
    normalized = torch.randn_like(hidden)
    kwargs = simplified_sparse_attention.SimplifiedSparseAttention._get_core_attention_extra_kwargs(
        owner, hidden_states=hidden, qkv_layernorm_output=normalized, inference_context=None
    )
    assert kwargs["indexer_input"] is normalized
    assert set(kwargs) == {"indexer_input"}


def test_ssa_factory_preserves_dense_spec_and_sets_loss_scale_hook():
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        _get_self_attention_module_spec,
        get_experimental_attention_variant_module_spec,
    )
    from megatron.core.transformer.attention import SelfAttention

    config = _config()
    dense_before = _get_self_attention_module_spec(config)
    selected = get_experimental_attention_variant_module_spec(config)
    dense_after = _get_self_attention_module_spec(config)
    assert selected.module is simplified_sparse_attention.SimplifiedSparseAttention
    assert dense_before.module is dense_after.module is SelfAttention
    assert selected.submodules == dense_before.submodules == dense_after.submodules
    assert selected.metainfo == dense_before.metainfo == dense_after.metainfo
    assert config.experimental_attention_variant_loss_scale_func == (
        simplified_sparse_attention.DSAIndexerLossAutoScaler.set_loss_scale
    )
    assert (
        _config(experimental_attention_variant=None).experimental_attention_variant_loss_scale_func
        is None
    )


def _fake_attention(q, k, v, qi, ki, **kwargs):
    output = q + k.expand_as(q) + v.expand_as(q)
    if kwargs["train_indexer_only"]:
        output = output.detach()
    denominator = 1 if kwargs["calculate_per_token_loss"] else q.size(0)
    loss = qi.float().square().sum() + ki.float().square().sum()
    return output.flatten(2), loss * kwargs["loss_coeff"] / denominator


@pytest.mark.parametrize("coefficient", [0.0, 0.2])
@pytest.mark.parametrize("per_token", [False, True])
def test_core_auxiliary_gradients_are_detached_and_logging_is_mean(
    monkeypatch, coefficient, per_token
):
    core = _mock_core(monkeypatch, coefficient=coefficient, per_token=per_token)
    logged = {}
    monkeypatch.setattr(
        simplified_sparse_attention.DSAIndexerLossLoggingHelper,
        "save_loss_to_tracker",
        lambda **kw: logged.update(kw),
    )
    monkeypatch.setattr(simplified_sparse_attention, "run_cute_sparse_attention", _fake_attention)
    hidden = torch.randn(16, 1, 16, dtype=torch.bfloat16, requires_grad=True)
    query, key, value, _, _ = _inputs()
    output = core(query, key, value, None, indexer_input=hidden)
    output.float().sum().backward()
    assert hidden.grad is None
    assert set(dict(core.named_parameters())) == {
        "indexer.linear_q.weight",
        "indexer.linear_k.weight",
    }
    for parameter in core.indexer.parameters():
        assert parameter.grad is not None
        assert bool(torch.count_nonzero(parameter.grad)) == (coefficient > 0)
    qi, ki = core.indexer(hidden)
    expected_log = (qi.float().square().sum() + ki.float().square().sum()) * coefficient / 16
    torch.testing.assert_close(logged["loss"], expected_log)
    torch.testing.assert_close(query.grad, torch.ones_like(query))


def test_phase1_residual_backward_reaches_every_indexer(monkeypatch):
    cores = [_mock_core(monkeypatch, phase1=True, layer_number=i + 1) for i in range(3)]
    monkeypatch.setattr(
        simplified_sparse_attention.DSAIndexerLossLoggingHelper,
        "save_loss_to_tracker",
        lambda **kw: None,
    )
    monkeypatch.setattr(simplified_sparse_attention, "run_cute_sparse_attention", _fake_attention)
    simplified_sparse_attention.DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(0.25))
    hidden = torch.randn(16, 1, 16, dtype=torch.bfloat16)
    projection = torch.randn(16, 16 * 256, dtype=torch.bfloat16)
    for core in cores:
        query = (hidden @ projection).reshape(16, 1, 16, 256)
        key = query[:, :, :1]
        output = core(query, key, key, None, indexer_input=hidden)
        hidden = hidden + output[..., :16]
    hidden.float().sum().backward()
    for core in cores:
        for parameter in core.indexer.parameters():
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) > 0
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("precision", ["bf16", "mxfp8", "mxfp8_params"])
@pytest.mark.parametrize("recompute", ["none", "core_attn", "core_attn_output", "qkv_proj"])
def test_transformer_layer_reuses_norm_and_preserves_bf16_indexer(
    monkeypatch, precision, recompute
):
    from megatron.core.enums import Fp8Recipe
    from megatron.core.fp8_utils import get_fp8_context, is_mxfp8tensor
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_layer_specs import (
        ssa_stack_spec,
    )
    from megatron.core.transformer.spec_utils import build_module
    from tests.unit_tests.test_utilities import Utils

    if not torch.cuda.is_available():
        pytest.skip("Transformer Engine requires CUDA")
    mxfp8 = precision != "bf16"
    if mxfp8 and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell")
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(1224)
        config = _config(
            ffn_hidden_size=512,
            recompute_granularity="selective" if recompute != "none" else None,
            recompute_modules=(
                ["core_attn_output", "qkv_proj"]
                if recompute == "qkv_proj"
                else [recompute] if recompute != "none" else []
            ),
            fp8="hybrid" if mxfp8 else None,
            fp8_recipe=Fp8Recipe.mxfp8,
            fp8_param=precision == "mxfp8_params",
        )
        # Enter the real outer parameter-quantization context: the indexer's
        # initialization guard must override it while QKV remains quantized.
        with get_fp8_context(config, is_init=True):
            layer = build_module(ssa_stack_spec.submodules.attention_layer, config=config).cuda()
        attention = layer.self_attention
        assert is_mxfp8tensor(attention.linear_qkv.weight) == config.fp8_param
        captured = {"qkv_calls": 0, "core_calls": 0}

        def capture_projection(module, inputs, output):
            captured["qkv_calls"] += 1
            captured["norm"] = output[2]

        attention.linear_qkv.register_forward_hook(capture_projection)
        attention.core_attention.register_forward_hook(
            lambda module, inputs, output: captured.update(core_calls=captured["core_calls"] + 1)
        )
        attention.core_attention.indexer.linear_q.register_forward_pre_hook(
            lambda module, inputs: captured.update(indexer=inputs[0])
        )
        monkeypatch.setattr(
            simplified_sparse_attention.DSAIndexerLossLoggingHelper,
            "save_loss_to_tracker",
            lambda **kw: None,
        )
        monkeypatch.setattr(
            simplified_sparse_attention.DSAIndexerLossAutoScaler, "main_loss_backward_scale", None
        )
        monkeypatch.setattr(
            simplified_sparse_attention, "run_cute_sparse_attention", _fake_attention
        )
        hidden = torch.randn(32, 1, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        with get_fp8_context(config):
            output, _ = layer(hidden, attention_mask=None)
        output.float().square().mean().backward()
        assert captured["qkv_calls"] == (2 if recompute == "qkv_proj" else 1)
        assert captured["core_calls"] == (2 if recompute != "none" else 1)
        assert captured["indexer"].data_ptr() == captured["norm"].data_ptr()
        assert not captured["indexer"].requires_grad
        assert hidden.grad is not None
        for name, parameter in layer.named_parameters():
            if ".indexer." in name or "linear_qkv.weight" in name:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
            if ".indexer." in name:
                assert parameter.dtype == torch.bfloat16
                assert type(parameter).__name__ == "Parameter"
        assert attention.core_attention.indexer.linear_q.config.fp8 is None
        assert attention.core_attention.indexer.linear_q.config.quant_recipe is None
        if precision == "bf16" and recompute == "none":
            from megatron.core.transformer.attention import InferenceMode

            def unexpected_decode(*args, **kwargs):
                pytest.fail("SSA must reject caching before entering inherited decode paths")

            monkeypatch.setattr(InferenceMode, "is_active", lambda: True)
            monkeypatch.setattr(attention, "flash_decode", unexpected_decode)
            monkeypatch.setattr(attention, "_adjust_key_value_for_inference", unexpected_decode)
            config.flash_decode = True
            context = SimpleNamespace(
                is_dynamic_batching=lambda: False,
                is_static_batching=lambda: True,
                is_decode_only=lambda: True,
            )
            with pytest.raises(NotImplementedError, match="decode caching"):
                attention.forward_pre_attn_and_core_attn(
                    hidden, attention_mask=None, inference_context=context
                )
    finally:
        Utils.destroy_model_parallel()


def _attention_oracle(inputs, coefficient, denominator, topk=None):
    q, k, v, qi, ki = (tensor.float() for tensor in inputs)
    q, k, v = q[:, 0], k[:, 0, 0], v[:, 0, 0]
    scores = torch.einsum("qhd,kd->hqk", q, k) * 256**-0.5
    causal = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
    logits = qi[:, 0, 0] @ ki[:, 0, 0].T * 128**-0.5
    if topk is not None:
        indices = logits.detach().masked_fill(~causal, -torch.inf).topk(topk, dim=-1).indices
        selected = torch.zeros_like(causal).scatter_(1, indices, True)
        causal &= selected
    probability = scores.masked_fill(~causal, -torch.inf).softmax(-1)
    output = torch.einsum("hqk,kd->qhd", probability, v).reshape(inputs[0].size(0), 1, -1)
    target = probability.detach().mean(0)
    log_prediction = logits.masked_fill(~causal, -torch.inf).log_softmax(-1)
    terms = target * (target.clamp_min(1e-10).log() - log_prediction)
    loss = terms.masked_fill(~causal, 0).sum() * coefficient / denominator
    return output, loss


@pytest.mark.parametrize("recompute", ["none", "core_attn", "core_attn_output", "qkv_proj"])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8", "mxfp8_params"])
def test_phase1_frozen_transformer_layers_retain_all_indexer_gradients(
    monkeypatch, recompute, precision
):
    from megatron.core.enums import Fp8Recipe
    from megatron.core.fp8_utils import get_fp8_context, is_mxfp8tensor
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_layer_specs import (
        ssa_stack_spec,
    )
    from megatron.core.transformer.spec_utils import build_module
    from tests.unit_tests.test_utilities import Utils

    if not torch.cuda.is_available():
        pytest.skip("Transformer Engine requires CUDA")
    mxfp8 = precision != "bf16"
    if mxfp8 and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell")
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(1234)
        config = _config(
            ffn_hidden_size=512,
            ssa_train_indexer_only=True,
            recompute_granularity="selective" if recompute != "none" else None,
            recompute_modules=(
                ["core_attn_output", "qkv_proj"]
                if recompute == "qkv_proj"
                else [recompute] if recompute != "none" else []
            ),
            fp8="hybrid" if mxfp8 else None,
            fp8_recipe=Fp8Recipe.mxfp8,
            fp8_param=precision == "mxfp8_params",
        )
        with get_fp8_context(config, is_init=True):
            layers = torch.nn.ModuleList(
                [
                    build_module(
                        ssa_stack_spec.submodules.attention_layer,
                        config=config,
                        layer_number=number,
                    )
                    for number in (1, 2)
                ]
            ).cuda()
        for name, parameter in layers.named_parameters():
            parameter.requires_grad_(".indexer." in name)
        for layer in layers:
            assert is_mxfp8tensor(layer.self_attention.linear_qkv.weight) == config.fp8_param
        logged_layers = []
        monkeypatch.setattr(
            simplified_sparse_attention.DSAIndexerLossLoggingHelper,
            "save_loss_to_tracker",
            lambda **kw: logged_layers.append(kw["layer_number"]),
        )
        monkeypatch.setattr(
            simplified_sparse_attention.DSAIndexerLossAutoScaler, "main_loss_backward_scale", None
        )
        monkeypatch.setattr(
            simplified_sparse_attention, "run_cute_sparse_attention", _fake_attention
        )
        hidden = torch.randn(32, 1, 256, dtype=torch.bfloat16, device="cuda")
        with get_fp8_context(config):
            for layer in layers:
                hidden, _ = layer(hidden, attention_mask=None)
        hidden.float().square().mean().backward()
        assert sorted(logged_layers) == [1, 2]
        for name, parameter in layers.named_parameters():
            if ".indexer." in name:
                assert parameter.dtype == torch.bfloat16
                assert type(parameter).__name__ == "Parameter"
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                assert torch.count_nonzero(parameter.grad) > 0, name
            else:
                assert parameter.grad is None, name
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "phase1,coefficient,per_token,sequence_length",
    [
        (True, 0.03, False, 128),
        (True, 0.2, True, 128),
        (False, 0.0, False, 128),
        (False, 0.03, False, 128),
        (False, 0.2, True, 128),
        (False, 0.03, False, 1024),
    ],
)
def test_packaged_attention_matches_output_loss_and_gradients(
    phase1, coefficient, per_token, sequence_length
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("The packaged kernels require SM103")
    pytest.importorskip("simplified_sparse_attention")
    torch.manual_seed(654)
    if phase1:
        pytest.importorskip("flash_attn.cute.interface")
    inputs = _inputs(sequence_length=sequence_length, device="cuda")
    selected_support = sequence_length > 512
    if selected_support:
        # Exact BF16 inputs give strictly ordered indexer scores. The selected
        # keys are unambiguous; no tolerance hides different top-k membership.
        with torch.no_grad():
            inputs[3].zero_()
            inputs[3][..., :2] = 1
            positions = torch.arange(sequence_length, device="cuda")
            inputs[4][:, 0, 0, 0] = (positions // 32) * 0.125
            inputs[4][:, 0, 0, 1] = (positions % 32) / 256
    refs = tuple(tensor.detach().float().requires_grad_() for tensor in inputs)
    output, loss = simplified_sparse_attention_cute.run_cute_sparse_attention(
        *inputs,
        topk=512,
        loss_coeff=coefficient,
        calculate_per_token_loss=per_token,
        train_indexer_only=phase1,
    )
    expected_output, expected_loss = _attention_oracle(
        refs,
        coefficient,
        1 if per_token else sequence_length,
        topk=512 if selected_support else None,
    )
    if selected_support:
        dense_output, _ = _attention_oracle(refs, coefficient, sequence_length)
        assert not torch.allclose(expected_output[512:], dense_output[512:], rtol=2e-2, atol=2e-2)
    upstream = torch.randn_like(output)
    if phase1:
        assert not output.requires_grad
        loss.backward()
        expected_loss.backward()
    else:
        ((output.float() * upstream).sum() + loss).backward()
        ((expected_output * upstream).sum() + expected_loss).backward()
    torch.testing.assert_close(output.float(), expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(loss, expected_loss, rtol=2e-2, atol=2e-4)
    for index, (actual, expected) in enumerate(zip(inputs, refs)):
        if phase1 and index < 3:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad.float(), expected.grad, rtol=4e-2, atol=4e-2)
            reference_norm = torch.linalg.vector_norm(expected.grad)
            if reference_norm.item() == 0:
                assert torch.count_nonzero(actual.grad) == 0
            else:
                relative_error = (
                    torch.linalg.vector_norm(actual.grad.float() - expected.grad) / reference_norm
                )
                assert relative_error.item() < 0.08


def test_core_allows_cp1_without_an_explicit_group(monkeypatch):
    monkeypatch.setattr(simplified_sparse_attention, "TELinear", _Linear)
    monkeypatch.setattr(
        simplified_sparse_attention,
        "get_fp8_disabled_context",
        lambda *args, **kwargs: nullcontext(),
    )
    core = simplified_sparse_attention.SimplifiedSparseAttentionCore(
        _config(hidden_size=16),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=None, cp=None),
    )
    monkeypatch.setattr(simplified_sparse_attention, "run_cute_sparse_attention", _fake_attention)
    monkeypatch.setattr(
        simplified_sparse_attention.DSAIndexerLossLoggingHelper,
        "save_loss_to_tracker",
        lambda **kw: None,
    )
    hidden = torch.randn(16, 1, 16, dtype=torch.bfloat16)
    output = core(*_inputs()[:3], None, indexer_input=hidden)
    assert output.shape == (16, 1, 4096)


@pytest.mark.parametrize("cp_size", [2, 8, 64])
@pytest.mark.parametrize(
    "query_chunks,key_chunks",
    [([1, 3], [1, 3]), ([0, 2, 0, 2, 0], [2, 0, 1, 1, 0]), ([1, 3], [2, 6]), ([0, 0], [0, 0])],
)
def test_complete_packed_cp_mapping_matches_generic(monkeypatch, cp_size, query_chunks, key_chunks):
    """Direct mapping preserves empty documents, rectangular layouts and every owner."""
    device = torch.device("cpu")
    cu_q = torch.tensor([0] + query_chunks, dtype=torch.int32).cumsum(0) * (2 * cp_size)
    cu_k = torch.tensor([0] + key_chunks, dtype=torch.int32).cumsum(0) * (2 * cp_size)
    local_q, local_k = 2 * sum(query_chunks), 2 * sum(key_chunks)
    build = dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder
    gathered_keys = torch.cat(
        [
            dsa_layout.build_packed_allgather_cp_local_positions(
                cu_k, cp_size, rank, device, local_k, cu_seqlens_cover_output=True
            )
            for rank in range(cp_size)
        ]
    )
    for rank in range(cp_size):
        sizes = dict(
            local_output_size=local_q,
            # SSA supplies only local_output_size for equal-sized Q/K.
            key_local_output_size=None if local_q == local_k else local_k,
            global_output_size=local_k * cp_size,
        )
        expected = build(cu_q, cu_k, cp_size, rank, device, **sizes)
        with monkeypatch.context() as patch:
            patch.setattr(
                dsa_layout,
                "build_packed_allgather_cp_local_positions",
                lambda *args, **kwargs: pytest.fail("Complete mapping used per-owner construction"),
            )
            actual = build(
                cu_q,
                cu_k,
                cp_size,
                rank,
                device,
                **sizes,
                query_cu_seqlens_cover_output=True,
                key_cu_seqlens_cover_output=True,
            )
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
            assert result.dtype == torch.int64 and result.is_contiguous()
        torch.testing.assert_close(
            gathered_keys.index_select(0, actual[1]), torch.arange(local_k * cp_size)
        )


def test_incomplete_packed_cp_mapping_keeps_padding_fallback(monkeypatch):
    monkeypatch.setattr(
        dsa_layout,
        "_build_complete_packed_cp_positions_and_reorder",
        lambda *args, **kwargs: pytest.fail("Incomplete metadata used complete mapping"),
    )
    cu = torch.tensor([0, 4, 12], dtype=torch.int32)
    positions, reorder = dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder(
        cu, cu, 2, 0, torch.device("cpu"), local_output_size=8, global_output_size=16
    )
    torch.testing.assert_close(positions, torch.tensor([0, 3, 4, 5, 10, 11, 12, 13]))
    gathered = torch.tensor([0, 3, 4, 5, 10, 11, 12, 13, 1, 2, 6, 7, 8, 9, 20, 21])
    torch.testing.assert_close(gathered[reorder], torch.tensor(list(range(14)) + [20, 21]))


def test_packed_cp_layout_uses_changed_physical_boundaries_and_logical_lengths(monkeypatch):
    from megatron.core.packed_seq_params import PackedSeqParams

    group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
    physical = torch.tensor([0, 4, 16], dtype=torch.int32)
    logical = torch.tensor([0, 2, 10], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=12,
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, **kwargs: tensor.fill_(10))
    query = torch.empty(8, 1, 1, 1)
    for boundaries, starts, positions, valid in (
        (
            [0, 4, 16],
            [0, 0, 4, 4, 4, 4, 4, 4],
            [1, 2, 7, 8, 9, 10, 11, 12],
            [True, False, True, True, True, True, True, False],
        ),
        (
            [0, 8, 16],
            [0, 0, 0, 0, 8, 8, 8, 8],
            [2, 3, 4, 5, 10, 11, 12, 13],
            [False, False, False, False, True, True, True, True],
        ),
    ):
        physical.copy_(torch.tensor(boundaries, dtype=torch.int32))
        layout = simplified_sparse_attention_cute.build_attention_layout(query, params, group)
        torch.testing.assert_close(layout.key_starts, torch.tensor(starts))
        torch.testing.assert_close(layout.key_ends, torch.tensor(positions) + 1)
        torch.testing.assert_close(layout.query_valid, torch.tensor(valid))
        torch.testing.assert_close(layout.num_real_tokens, torch.tensor([10.0]))


def test_core_reuses_packed_layout_between_layers_and_invalidates_changed_lengths(monkeypatch):
    from megatron.core.packed_seq_params import PackedSeqParams

    cores = [_mock_core(monkeypatch, per_token=True, layer_number=i + 1) for i in range(2)]
    # Layers in a model share the same process-group object.
    cores[1].cp_group = cores[0].cp_group
    logical = torch.tensor([0, 13], dtype=torch.int32)
    physical = torch.tensor([0, 16], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=16,
        max_seqlen_kv=16,
    )
    build = simplified_sparse_attention.build_attention_layout
    builds, logged = [], []

    def record_build(*args):
        result = build(*args)
        builds.append(result)
        return result

    monkeypatch.setattr(simplified_sparse_attention, "build_attention_layout", record_build)
    monkeypatch.setattr(simplified_sparse_attention, "run_cute_sparse_attention", _fake_attention)
    monkeypatch.setattr(
        simplified_sparse_attention.DSAIndexerLossLoggingHelper,
        "save_loss_to_tracker",
        lambda **kw: logged.append(kw["loss"]),
    )
    hidden = torch.randn(16, 1, 16, dtype=torch.bfloat16)
    inputs = _inputs()[:3]
    for core in cores:
        core(*inputs, None, packed_seq_params=params, indexer_input=hidden)
    assert len(builds) == 1
    assert builds[0].num_real_tokens.item() == 13
    qi, ki = cores[0].indexer(hidden)
    expected_log = (qi.float().square().sum() + ki.float().square().sum()) * 0.1 / 13
    torch.testing.assert_close(logged[0], expected_log)
    logical[1] = 12
    cores[0](*inputs, None, packed_seq_params=params, indexer_input=hidden)
    assert len(builds) == 2
    assert builds[-1].num_real_tokens.item() == 12
