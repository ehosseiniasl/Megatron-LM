# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare fused metric statistics with an independent high-precision reference."""

import os

import pytest
import torch

from megatron.training.tensor_metrics.definitions import MeanAbsoluteRmsMetric


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("shape", ((), (0,), (1,), (31,), (2049,), (65539,), (33, 257)))
def test_fused_moments_match_logical_values(dtype, shape):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    generator = torch.Generator(device="cuda").manual_seed(123)
    values = torch.randn(shape, dtype=dtype, device="cuda", generator=generator)
    if values.ndim == 2:
        values = values.transpose(0, 1)
        assert not values.is_contiguous()
    actual = MeanAbsoluteRmsMetric().contribution(values)
    reference = values.double()
    expected = torch.stack(
        (
            reference.sum(),
            reference.abs().sum(),
            reference.square().sum(),
            reference.new_tensor(reference.numel()),
        )
    )
    assert actual.dtype == dtype
    tolerances = (
        {"rtol": 1e-12, "atol": 2e-10} if dtype == torch.float64 else {"rtol": 2e-6, "atol": 2e-4}
    )
    torch.testing.assert_close(actual.double(), expected, **tolerances)


@pytest.mark.parametrize("fill", (0.0, float("inf"), float("nan")))
def test_fused_moments_preserve_zero_and_nonfinite_values(fill):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    values = torch.full((4097,), fill, device="cuda")
    actual = MeanAbsoluteRmsMetric().contribution(values)
    expected = torch.stack(
        (values.sum(), values.abs().sum(), values.square().sum(), values.new_tensor(values.numel()))
    )
    torch.testing.assert_close(actual, expected, equal_nan=True)
