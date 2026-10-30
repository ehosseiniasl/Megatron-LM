# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Frozen activations must not suppress gradients of enclosed trainable modules."""

import pytest
import torch

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    CheckpointWithoutOutputManager,
    checkpoint,
    model_parallel_cuda_manual_seed,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("mode", ["reentrant", "discard", "manager"])
@pytest.mark.parametrize("requires_input_grad", [False, True])
def test_closed_over_parameter_gradients(mode, requires_input_grad):
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    try:
        x = torch.randn(4, 8, device="cuda", requires_grad=requires_input_grad)
        weight = torch.randn(8, 3, device="cuda", requires_grad=True)
        expected_x = x.detach().clone().requires_grad_(requires_input_grad)
        expected_weight = weight.detach().clone().requires_grad_(True)
        expected = (expected_x @ expected_weight).square().sum()
        expected.backward()

        manager = CheckpointWithoutOutputManager() if mode == "manager" else None
        handle = CheckpointWithoutOutput(fp8=None, ckpt_manager=manager)
        if mode == "reentrant":
            output = checkpoint(lambda value: value @ weight, False, x)
        else:
            output = handle.checkpoint(lambda value: value @ weight, x)
        loss = output.square().sum()
        if mode == "discard":
            handle.discard_output_and_register_recompute(loss)
        elif manager is not None:
            manager.discard_all_outputs_and_register_unified_recompute(loss)
        loss.backward()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(weight.grad, expected_weight.grad)
        if requires_input_grad:
            torch.testing.assert_close(x.grad, expected_x.grad)
        else:
            assert x.grad is None
    finally:
        Utils.destroy_model_parallel()


def test_reused_checkpoint_after_no_grad_forward():
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    try:
        handle = CheckpointWithoutOutput(fp8=None)
        x = torch.randn(4, 8, device="cuda", requires_grad=True)
        with torch.no_grad():
            output = handle.checkpoint(torch.sigmoid, x)
            handle.discard_output_and_register_recompute(output)
            assert not output.requires_grad
            torch.testing.assert_close(output, torch.sigmoid(x))
        output = handle.checkpoint(torch.sigmoid, x)
        loss = output.square().sum()
        handle.discard_output_and_register_recompute(loss)
        loss.backward()
        sigmoid = x.detach().sigmoid()
        torch.testing.assert_close(x.grad, 2 * sigmoid.square() * (1 - sigmoid))
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_manager", [False, True])
def test_frozen_operation_with_differentiable_dummy_input(use_manager):
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    try:
        x = torch.randn(4, 8, device="cuda")
        dummy = torch.empty(0, device="cuda", requires_grad=True)
        weight = torch.randn(4, 8, device="cuda", requires_grad=True)
        expected_weight_grad = x.square()
        manager = CheckpointWithoutOutputManager() if use_manager else None
        handle = CheckpointWithoutOutput(fp8=None, ckpt_manager=manager)
        output = handle.checkpoint(lambda value, unused: value.square(), x, dummy)
        loss = (output * weight).sum()
        if use_manager:
            manager.discard_all_outputs_and_register_unified_recompute(loss)
        else:
            handle.discard_output_and_register_recompute(loss)
        loss.backward()
        assert x.grad is None
        assert dummy.grad is None
        torch.testing.assert_close(weight.grad, expected_weight_grad)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_manager", [False, True])
def test_mixed_differentiable_and_detached_recomputed_outputs(use_manager):
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    try:
        x = torch.randn(4, 8, device="cuda", requires_grad=True)
        expected_grad = 2 * x.detach()
        manager = CheckpointWithoutOutputManager() if use_manager else None
        handle = CheckpointWithoutOutput(fp8=None, ckpt_manager=manager)
        differentiable, detached = handle.checkpoint(
            lambda value: (value.square(), value.detach().cos()), x
        )
        loss = differentiable.sum() + detached.sum()
        if use_manager:
            manager.discard_all_outputs_and_register_unified_recompute(loss)
        else:
            handle.discard_output_and_register_recompute(loss)
        loss.backward()
        torch.testing.assert_close(x.grad, expected_grad)
    finally:
        Utils.destroy_model_parallel()
