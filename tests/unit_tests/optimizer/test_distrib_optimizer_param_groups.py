# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Param-group bookkeeping in DistributedOptimizer, exercised on a mixed-dtype group.

``_build_model_and_main_param_groups`` rebuilds each group's inner ``params`` list as
native-FP32 shards followed by FP32 copies of the reduced-precision params, which is not the
grad-buffer traversal order ``_build_optimizer_group_ranges`` records. The optimizer's
checkpoint save/load path indexes ``param_groups[gi]["params"][go]`` with the recorded order,
so the two must agree. ``--moe-router-fp32-weights`` creates a group that mixes dtypes.
"""

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.optimizer.optimizer_config import OptimizerConfig


def _build_groups(param_dtypes):
    """Run the real classmethod over one bucket holding ``param_dtypes``, unsharded.

    Returns ``(params, opt_group_ranges, index_map)``.
    """
    params = [
        torch.nn.Parameter(torch.randn(4, 8, dtype=dtype, device='cuda')) for dtype in param_dtypes
    ]

    param_map = {p: {"param": Range(0, p.numel())} for p in params}
    gbuf_ranges = [{torch.float32: [{"param_map": param_map}]}]
    param_gbuf_map = {p: (0, torch.float32, 0) for p in params}

    orig_group = {"params": list(params), "wd_mult": 1.0}
    opt_group_ranges = [{"params": list(params), "orig_group": orig_group}]

    config = OptimizerConfig(bf16=True, use_distributed_optimizer=True)
    *_, index_map = DistributedOptimizer._build_model_and_main_param_groups(
        gbuf_ranges, param_gbuf_map, opt_group_ranges, config
    )
    return params, opt_group_ranges, index_map


def _assert_map_is_consistent(params, opt_group_ranges, index_map):
    """Every model param's recorded slot must hold that param's own shard."""
    for model_param in params:
        group_index, group_order = index_map[model_param]
        main_param = opt_group_ranges[group_index]["orig_group"]["params"][group_order]
        assert main_param.numel() == model_param.numel()
        torch.testing.assert_close(
            main_param.view(-1).float(), model_param.detach().view(-1).float(), rtol=0, atol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "param_dtypes",
    [
        pytest.param([torch.bfloat16, torch.bfloat16], id="all_bf16"),
        pytest.param([torch.float32, torch.float32], id="all_fp32"),
        pytest.param([torch.bfloat16, torch.float32], id="bf16_then_fp32"),
        pytest.param([torch.float32, torch.bfloat16], id="fp32_then_bf16"),
        pytest.param(
            [torch.bfloat16, torch.float32, torch.bfloat16, torch.float32], id="interleaved"
        ),
    ],
)
def test_param_group_index_map_tracks_rebuilt_order(param_dtypes):
    params, opt_group_ranges, index_map = _build_groups(param_dtypes)
    _assert_map_is_consistent(params, opt_group_ranges, index_map)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_param_group_index_map_matches_traversal_order_when_homogeneous():
    """Single-dtype groups, which already worked, must be left where they were."""
    for dtype in (torch.bfloat16, torch.float32):
        params, _, index_map = _build_groups([dtype] * 3)
        assert [index_map[p] for p in params] == [(0, 0), (0, 1), (0, 2)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_group_puts_fp32_params_first():
    """Pins the ordering the index map encodes."""
    params, _, index_map = _build_groups([torch.bfloat16, torch.float32, torch.bfloat16])
    bf16_a, fp32, bf16_b = params
    assert index_map[fp32] == (0, 0)
    assert index_map[bf16_a] == (0, 1)
    assert index_map[bf16_b] == (0, 2)
