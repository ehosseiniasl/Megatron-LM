# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""--rerun-large-grad-norm-steps is validated with the rest of the training arguments."""

from types import SimpleNamespace

import pytest

from megatron.training.arguments import _validate_rerun_large_grad_norm_steps_args


def _args(**overrides):
    values = dict(
        rerun_large_grad_norm_steps=True,
        rerun_grad_norm_threshold=None,
        rerun_mode="validate_results",
        clip_grad=1.0,
        use_distributed_optimizer=True,
        use_megatron_fsdp=False,
        use_torch_fsdp2=False,
        optimizer_cuda_graph=False,
        cuda_graph_impl="none",
        bf16=True,
        fp16=False,
        loss_scale=None,
        fp8="e4m3",
        fp8_recipe="mxfp8",
        moe_router_load_balancing_type="quantile_balancing",
        moe_router_quantile_balancing_estimation_scope="global_batch",
        mtp_use_repeated_layer=True,
        mtp_num_layers=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_supported_configuration_passes():
    _validate_rerun_large_grad_norm_steps_args(_args())


def test_disabled_flag_skips_every_check():
    _validate_rerun_large_grad_norm_steps_args(
        _args(rerun_large_grad_norm_steps=False, rerun_mode="disabled")
    )


@pytest.mark.parametrize("threshold", [0.2, 1.0, 4.0])
def test_explicit_threshold_passes(threshold):
    _validate_rerun_large_grad_norm_steps_args(_args(rerun_grad_norm_threshold=threshold))


def test_explicit_threshold_does_not_need_clipping():
    _validate_rerun_large_grad_norm_steps_args(_args(clip_grad=0.0, rerun_grad_norm_threshold=0.2))


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(rerun_mode="disabled"), "validate_results"),
        (dict(clip_grad=0.0), "clip-grad"),
        (dict(use_distributed_optimizer=False), "distributed-optimizer"),
        (dict(use_megatron_fsdp=True), "FSDP"),
        (dict(optimizer_cuda_graph=True), "optimizer-cuda-graph"),
        (dict(cuda_graph_impl="local"), "CUDA graphs"),
        (dict(bf16=False), "bf16"),
        (dict(moe_router_load_balancing_type="aux_loss"), "quantile_balancing"),
        (dict(mtp_use_repeated_layer=False), "repeated-layer MTP"),
        (dict(rerun_grad_norm_threshold=0.0), "must be positive"),
        (
            dict(rerun_large_grad_norm_steps=False, rerun_grad_norm_threshold=4.0),
            "requires --rerun-large-grad-norm-steps",
        ),
    ],
)
def test_unsupported_configurations_fail(overrides, message):
    with pytest.raises(AssertionError, match=message):
        _validate_rerun_large_grad_norm_steps_args(_args(**overrides))
