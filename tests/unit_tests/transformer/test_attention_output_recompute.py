# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Output-discarding attention replay, including shortcut and repeated-layer lifetimes."""

from unittest import mock

import pytest
import torch

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    CheckpointWithoutOutputManager,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _build_attention(recompute_module, fp8_recipe, num_query_groups=None):
    if recompute_module is None:
        recompute_modules = None
    else:
        recompute_modules = (
            [recompute_module] if isinstance(recompute_module, str) else list(recompute_module)
        )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=num_query_groups,
        kv_channels=64,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        recompute_granularity="selective" if recompute_module is not None else None,
        recompute_modules=recompute_modules,
        fp8="e4m3" if fp8_recipe else None,
        fp8_recipe=fp8_recipe or "delayed",
    )
    return SelfAttention(
        config,
        get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    ).cuda()


@pytest.mark.parametrize("recompute_module", [None, "core_attn"], ids=["none", "legacy_core"])
@pytest.mark.parametrize("fp8_recipe", [None, "mxfp8"])
def test_disabled_attention_output_recompute_preserves_legacy_path(recompute_module, fp8_recipe):
    """Opting out must keep QKV/output storage and the legacy replay call signature."""
    if fp8_recipe == "mxfp8":
        te = pytest.importorskip("transformer_engine.pytorch")
        available, reason = te.is_mxfp8_available(return_reason=True)
        if not available:
            pytest.skip(reason)

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        with mock.patch(
            "megatron.core.transformer.attention.tensor_parallel.CheckpointWithoutOutput",
            side_effect=AssertionError("Disabled attention recompute created an output checkpoint"),
        ):
            module = _build_attention(recompute_module, fp8_recipe, num_query_groups=1)
            assert not module.checkpoint_core_attention_output
            assert not module.checkpoint_qkv_projection
            hidden = torch.randn(
                128, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            with (
                mock.patch.object(
                    module.core_attention, "forward", wraps=module.core_attention.forward
                ) as core,
                mock.patch.object(
                    module.linear_qkv, "forward", wraps=module.linear_qkv.forward
                ) as qkv,
                mock.patch.object(
                    module,
                    "_checkpointed_attention_forward",
                    wraps=module._checkpointed_attention_forward,
                ) as legacy_checkpoint,
            ):
                with get_fp8_context(module.config):
                    core_output = module.forward_pre_attn_and_core_attn(hidden, attention_mask=None)
                    assert not hasattr(core_output, "_core_attn_output_checkpoint")
                    assert not hasattr(core_output, "_qkv_projection_checkpoint")
                    original_storage = core_output.untyped_storage().data_ptr()
                    original_bytes = core_output.untyped_storage().nbytes()
                    # Disabled stage two must not consume tensor attributes owned by
                    # any other caller or interpret them as enabled checkpoint handles.
                    sentinel = object()
                    core_output._core_attn_output_checkpoint = sentinel
                    core_output._qkv_projection_checkpoint = sentinel
                    output, _ = module.forward_post_core_attn(core_output)
                assert core_output._core_attn_output_checkpoint is sentinel
                assert core_output._qkv_projection_checkpoint is sentinel
                assert core_output.untyped_storage().data_ptr() == original_storage
                assert core_output.untyped_storage().nbytes() == original_bytes
                output.backward(torch.randn_like(output))
                assert qkv.call_count == 1
                assert core.call_count == (2 if recompute_module == "core_attn" else 1)
                assert legacy_checkpoint.call_count == int(recompute_module == "core_attn")
                for call in legacy_checkpoint.call_args_list:
                    assert "checkpoint_without_output" not in call.kwargs
            assert torch.isfinite(output).all()
            assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
            for name, parameter in module.named_parameters():
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
    finally:
        Utils.destroy_model_parallel()


def _run_repeated_attention(module, inputs, output_grads, packed_seq_params, early_backward):
    """Keep separate stage-one calls pending, or backward before reusing an MTP layer."""
    inputs = [value.detach().clone().requires_grad_(True) for value in inputs]
    core_outputs, outputs, checkpoints = [], [], []
    discard_output = module.checkpoint_core_attention_output
    discard_qkv = "qkv_proj" in module.config.recompute_modules
    qkv_checkpoints, qkv_outputs, expected_qkv = [], {}, {}

    def project_and_check(core_output):
        with get_fp8_context(module.config):
            output, _ = module.forward_post_core_attn(core_output)
        assert not hasattr(core_output, "_core_attn_output_checkpoint")
        assert not hasattr(core_output, "_qkv_projection_checkpoint")
        assert (core_output.untyped_storage().size() == 0) == discard_output
        if discard_qkv:
            assert all(
                value.untyped_storage().size() == 0 for value in qkv_outputs[id(core_output)]
            )
        return output

    core_forward = module.core_attention.forward

    def checked_core_forward(query, key, value, *args, **kwargs):
        # A reversed restore-hook order must fail here, before launching an attention kernel.
        assert all(tensor.untyped_storage().size() > 0 for tensor in (query, key, value))
        return core_forward(query, key, value, *args, **kwargs)

    with (
        mock.patch.object(
            module.core_attention, "forward", side_effect=checked_core_forward
        ) as core,
        mock.patch.object(
            module.linear_qkv, "forward", wraps=module.linear_qkv.forward
        ) as projection,
    ):
        for value, output_grad in zip(inputs, output_grads):
            with get_fp8_context(module.config):
                core_output = module.forward_pre_attn_and_core_attn(
                    value, attention_mask=None, packed_seq_params=packed_seq_params
                )
            assert core_output.untyped_storage().size() > 0
            if discard_output:
                checkpoints.append(core_output._core_attn_output_checkpoint)
            if discard_qkv:
                checkpoint = core_output._qkv_projection_checkpoint
                qkv_checkpoints.append(checkpoint)
                qkv_outputs[id(core_output)] = tuple(checkpoint.outputs)
                expected_qkv[id(core_output)] = [
                    value.detach().clone() for value in checkpoint.outputs
                ]
            core_outputs.append(core_output)
            if early_backward:
                output = project_and_check(core_output)
                outputs.append(output)
                output.backward(output_grad)

        if not early_backward:
            # Stage one may be called again before an earlier invocation reaches projection.
            # A module-global checkpoint handle would discard or restore the wrong output.
            outputs = [project_and_check(value) for value in core_outputs]
            assert core.call_count == len(inputs)
            torch.autograd.backward(outputs, output_grads)

        assert core.call_count == 2 * len(inputs)
        assert projection.call_count == (2 if discard_qkv else 1) * len(inputs)

    if discard_output:
        assert len({id(checkpoint) for checkpoint in checkpoints}) == len(inputs)
    if discard_qkv:
        assert len({id(checkpoint) for checkpoint in qkv_checkpoints}) == len(inputs)
        for core_output in core_outputs:
            for observed, expected in zip(
                qkv_outputs[id(core_output)], expected_qkv[id(core_output)]
            ):
                assert observed.untyped_storage().size() > 0
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert all(value.untyped_storage().size() > 0 for value in core_outputs)
    gradients = {
        name: parameter.grad.detach().clone() for name, parameter in module.named_parameters()
    }
    return (
        [output.detach().clone() for output in outputs],
        [value.grad for value in inputs],
        gradients,
    )


@pytest.mark.parametrize("fp8_recipe", [None, "mxfp8"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("early_backward", [False, True])
@pytest.mark.parametrize("qkv_proj", [False, True], ids=["core_output", "qkv_and_core_output"])
def test_attention_output_discard_and_replay_parity(fp8_recipe, packed, early_backward, qkv_proj):
    """Release projection inputs and preserve outputs/all gradients across repeated forwards."""
    if fp8_recipe == "mxfp8":
        te = pytest.importorskip("transformer_engine.pytorch")
        available, reason = te.is_mxfp8_available(return_reason=True)
        if not available:
            pytest.skip(reason)

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        query_groups = 1 if qkv_proj else None
        baseline = _build_attention(
            "core_attn_output" if qkv_proj else "core_attn", fp8_recipe, query_groups
        )
        recompute_modules = ["core_attn_output"] + (["qkv_proj"] if qkv_proj else [])
        recomputed = _build_attention(recompute_modules, fp8_recipe, query_groups)
        recomputed.load_state_dict(baseline.state_dict())
        if fp8_recipe:
            assert recomputed.linear_proj.save_original_input

        sequence_length = 128
        inputs = [
            torch.randn(sequence_length, 1, 256, dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        output_grads = [torch.randn_like(value) for value in inputs]
        packed_seq_params = None
        if packed:
            boundaries = torch.tensor([0, 64, sequence_length], dtype=torch.int32, device="cuda")
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=boundaries,
                cu_seqlens_kv=boundaries,
                max_seqlen_q=64,
                max_seqlen_kv=64,
            )

        reference = _run_repeated_attention(
            baseline, inputs, output_grads, packed_seq_params, early_backward
        )
        actual = _run_repeated_attention(
            recomputed, inputs, output_grads, packed_seq_params, early_backward
        )
        for expected, observed in zip(reference[0], actual[0]):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        for expected, observed in zip(reference[1], actual[1]):
            torch.testing.assert_close(observed, expected, rtol=1e-2, atol=1e-3)
        assert reference[2].keys() == actual[2].keys()
        for name, expected in reference[2].items():
            torch.testing.assert_close(actual[2][name], expected, rtol=1e-2, atol=1e-3, msg=name)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("qkv_proj", [False, True])
def test_attention_output_recompute_no_grad_retains_output(training, qkv_proj):
    """Inference/no-grad calls must never attach a replay handle or discard their output."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        recompute_modules = ["core_attn_output"] + (["qkv_proj"] if qkv_proj else [])
        module = _build_attention(recompute_modules, None, num_query_groups=1)
        module.train(training)
        with torch.no_grad():
            hidden = torch.randn(128, 1, 256, device="cuda", dtype=torch.bfloat16)
            core_output = module.forward_pre_attn_and_core_attn(hidden, attention_mask=None)
            output, _ = module.forward_post_core_attn(core_output)
        assert not hasattr(core_output, "_core_attn_output_checkpoint")
        assert not hasattr(core_output, "_qkv_projection_checkpoint")
        assert core_output.untyped_storage().size() > 0
        assert torch.isfinite(output).all()
    finally:
        Utils.destroy_model_parallel()


def test_qkv_recompute_preserves_weight_gradients_with_frozen_input():
    """A nondifferentiable input must still replay QKV and train every attention weight."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        baseline = _build_attention("core_attn_output", None, num_query_groups=1)
        recomputed = _build_attention(["core_attn_output", "qkv_proj"], None, num_query_groups=1)
        recomputed.load_state_dict(baseline.state_dict())
        hidden = torch.randn(128, 1, 256, device="cuda", dtype=torch.bfloat16)
        output_grad = torch.randn_like(hidden)
        results = []
        for module, expected_projection_calls in ((baseline, 1), (recomputed, 2)):
            with mock.patch.object(
                module.linear_qkv, "forward", wraps=module.linear_qkv.forward
            ) as projection:
                output, _ = module(hidden, attention_mask=None)
                assert output.requires_grad
                assert torch.isfinite(output).all()
                output.backward(output_grad)
                assert projection.call_count == expected_projection_calls
            gradients = {}
            for name, parameter in module.named_parameters():
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                gradients[name] = parameter.grad.detach().clone()
            results.append((output.detach().clone(), gradients))
        assert hidden.grad is None
        torch.testing.assert_close(results[1][0], results[0][0], rtol=0, atol=0)
        assert results[1][1].keys() == results[0][1].keys()
        for name, expected in results[0][1].items():
            torch.testing.assert_close(
                results[1][1][name], expected, rtol=1e-2, atol=1e-3, msg=name
            )
    finally:
        Utils.destroy_model_parallel()


def test_qkv_recompute_restores_discarded_normalization_input():
    """Residual-block replay restores the norm input before QKV and core-attention replay."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        baseline = _build_attention("core_attn_output", None, num_query_groups=1)
        recomputed = _build_attention(["core_attn_output", "qkv_proj"], None, num_query_groups=1)
        recomputed.load_state_dict(baseline.state_dict())
        norms = [torch.nn.LayerNorm(256, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        norms[1].load_state_dict(norms[0].state_dict())
        hidden = torch.randn(128, 1, 256, device="cuda", dtype=torch.bfloat16)
        output_grad = torch.randn_like(hidden)
        results = []
        for module, norm, discard_norm in (
            (baseline, norms[0], False),
            (recomputed, norms[1], True),
        ):
            x = hidden.detach().clone().requires_grad_(True)
            events = []
            norm_forward = norm.forward
            projection_forward = module.linear_qkv.forward
            core_forward = module.core_attention.forward

            def tracked_norm(*args, **kwargs):
                events.append("norm")
                return norm_forward(*args, **kwargs)

            def checked_projection(states, *args, **kwargs):
                assert states.untyped_storage().size() > 0
                events.append("qkv")
                return projection_forward(states, *args, **kwargs)

            def checked_core(query, key, value, *args, **kwargs):
                assert all(tensor.untyped_storage().size() > 0 for tensor in (query, key, value))
                events.append("core")
                return core_forward(query, key, value, *args, **kwargs)

            with (
                mock.patch.object(norm, "forward", side_effect=tracked_norm),
                mock.patch.object(module.linear_qkv, "forward", side_effect=checked_projection),
                mock.patch.object(module.core_attention, "forward", side_effect=checked_core),
            ):
                if discard_norm:
                    manager = CheckpointWithoutOutputManager()
                    checkpoint = CheckpointWithoutOutput(ckpt_manager=manager)
                    normalized = checkpoint.checkpoint(norm, x)
                else:
                    normalized = norm(x)
                expected_normalized = normalized.detach().clone()
                core_output = module.forward_pre_attn_and_core_attn(normalized, attention_mask=None)
                if discard_norm:
                    qkv_outputs = tuple(core_output._qkv_projection_checkpoint.outputs)
                projected, _ = module.forward_post_core_attn(core_output)
                # A downstream residual output is the block manager's replay boundary.
                output = projected + x
                if discard_norm:
                    manager.discard_all_outputs_and_register_unified_recompute(output)
                    assert normalized.untyped_storage().size() == 0
                    assert core_output.untyped_storage().size() == 0
                    assert all(value.untyped_storage().size() == 0 for value in qkv_outputs)
                output.backward(output_grad)
                if discard_norm:
                    assert events == ["norm", "qkv", "core", "norm", "qkv", "core"]
                    torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=0)
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in module.named_parameters()
            }
            gradients.update(
                {
                    f"norm.{name}": parameter.grad.detach().clone()
                    for name, parameter in norm.named_parameters()
                }
            )
            results.append((output.detach().clone(), x.grad.clone(), gradients))
        torch.testing.assert_close(results[1][0], results[0][0], rtol=0, atol=0)
        torch.testing.assert_close(results[1][1], results[0][1], rtol=1e-2, atol=1e-3)
        assert results[1][2].keys() == results[0][2].keys()
        for name, expected in results[0][2].items():
            torch.testing.assert_close(
                results[1][2][name], expected, rtol=1e-2, atol=1e-3, msg=name
            )
    finally:
        Utils.destroy_model_parallel()
