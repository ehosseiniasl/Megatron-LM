# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for MimoOptimizer module-local gradient clipping."""

import os
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version
from torch.utils._python_dispatch import TorchDispatchMode

from examples.mimo.training.topology import ModuleGridSpec, create_topology
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    param_and_grad_buffer,
)
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.optimizer import (
    MimoOptimizer,
    ModuleOptimizerInfo,
    get_mimo_optimizer,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.dist_utils import _ddp_wrap
from tests.unit_tests.test_utilities import Utils

ENCODER_NAME = "images"
# Gradient norm of each module; the joint norm is 5.
GRAD_NORMS = {ENCODER_NAME: 3.0, MIMO_LANGUAGE_MODULE_KEY: 4.0}


@pytest.fixture
def topology():
    """Encoder on the first half of the ranks, LLM on the second half, DP over each half."""
    Utils.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    if world_size < 2 or world_size % 2:
        pytest.skip("Requires an even distributed world size >= 2")
    module_size = world_size // 2
    topology = create_topology(
        [
            ModuleGridSpec(name=ENCODER_NAME, num_ranks=module_size),
            ModuleGridSpec(
                name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=module_size, rank_offset=module_size
            ),
        ]
    )
    yield topology
    topology.destroy()
    Utils.destroy_model_parallel()


def _mimo_optimizer(topology, module, config):
    """Build a MimoOptimizer for this rank's DDP-wrapped module through get_mimo_optimizer."""
    (name,) = topology.schedule_pg_collection.keys()
    model = torch.nn.Module()
    model.mimo_config = SimpleNamespace(module_to_grid_map=topology.grids)
    model.language_model = module if name == MIMO_LANGUAGE_MODULE_KEY else None
    model.modality_submodules = torch.nn.ModuleDict(
        {} if name == MIMO_LANGUAGE_MODULE_KEY else {name: module}
    )
    return get_mimo_optimizer(model, config)


def _fill_grads(module, norm):
    """Fill every main_grad with one value so the module's gradient norm is ``norm``."""
    params = list(module.parameters())
    value = norm / sum(param.numel() for param in params) ** 0.5
    for param in params:
        param.main_grad.fill_(value)


def _record_collective_groups(patch):
    """Record the ranks of the group of every collective that a module step runs."""
    group_ranks = []
    # (module, collective, positional index of its group argument)
    targets = [
        (torch.distributed, "all_reduce", 2),
        (param_and_grad_buffer, "dist_all_gather_func", 2),
    ]
    for module, name, index in targets:

        def record(*args, _collective=getattr(module, name), _index=index, **kwargs):
            group = kwargs.get("group", args[_index] if len(args) > _index else None)
            group = torch.distributed.group.WORLD if group is None else group
            group_ranks.append(set(torch.distributed.get_process_group_ranks(group)))
            return _collective(*args, **kwargs)

        patch.setattr(module, name, record)
    return group_ranks


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 ranks.")
@pytest.mark.parametrize("local_clip", [False, True])
def test_clip_by_module_norm(topology, local_clip):
    """Default clips both modules by the joint norm; local clips each by its own norm."""
    (name,) = topology.schedule_pg_collection.keys()
    pg_collection = topology.module_pgs[name]
    linear = torch.nn.Linear(16, 1, bias=False, device="cuda")
    linear.weight.data.zero_()
    linear.pg_collection = pg_collection
    module = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(),
        linear,
        pg_collection=pg_collection,
    )
    optimizer = _mimo_optimizer(
        topology,
        module,
        OptimizerConfig(
            optimizer="sgd",
            lr=1.0,
            weight_decay=0.0,
            sgd_momentum=0.0,
            clip_grad=1.0,
            bf16=True,
            mimo_module_local_grad_clip=local_clip,
        ),
    )
    _fill_grads(linear, GRAD_NORMS[name])

    success, grad_norm, _ = optimizer.step()

    expected_norm = GRAD_NORMS[name] if local_clip else 5.0
    assert success is True
    assert float(grad_norm) == pytest.approx(expected_norm)
    # SGD with lr 1 from zero weights: the weight is minus the clipped gradient.
    expected_weight = -GRAD_NORMS[name] / 16**0.5 / expected_norm
    torch.testing.assert_close(
        linear.weight.detach(), torch.full_like(linear.weight, expected_weight)
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 ranks.")
@pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip layer-wise optimizer for LTS test",
)
@pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package is not installed"
)
@pytest.mark.parametrize("local_clip", [False, True])
def test_layer_wise_chain_collectives_stay_in_module(topology, monkeypatch, local_clip):
    """With [LayerWise(Muon), DistOpt(Adam)] module optimizers, as in NM4, the local step
    reduces and gathers only inside the module's ranks."""
    (name,) = topology.schedule_pg_collection.keys()
    pg_collection = topology.module_pgs[name]
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)).bfloat16().cuda()
    model.config = TransformerConfig(num_attention_heads=1, num_layers=1, bf16=True)
    model.pg_collection = pg_collection
    (module,) = _ddp_wrap(
        [model],
        data_parallel_random_init=False,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True, grad_reduce_in_fp32=True
        ),
        overlap_param_gather_with_optimizer_step=False,
        pg_collection=pg_collection,
        use_layer_wise_distributed_optimizer=True,
    )
    optimizer = _mimo_optimizer(
        topology,
        module,
        OptimizerConfig(
            optimizer="muon",
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=True,
            use_layer_wise_distributed_optimizer=True,
            muon_scalar_optimizer="adam",
            clip_grad=1.0,
            mimo_module_local_grad_clip=local_clip,
        ),
    )
    kinds = {type(leaf) for leaf in optimizer.chained_optimizers}
    assert {LayerWiseDistributedOptimizer, DistributedOptimizer} <= kinds
    _fill_grads(model, GRAD_NORMS[name])

    with monkeypatch.context() as patch:
        group_ranks = _record_collective_groups(patch)
        success, grad_norm, _ = optimizer.step()

    assert success is True
    assert float(grad_norm) == pytest.approx(GRAD_NORMS[name] if local_clip else 5.0, rel=1e-4)
    grid = topology.grids[name]
    module_ranks = set(range(grid.rank_offset, grid.rank_offset + grid.size))
    if local_clip:
        assert module_ranks in group_ranks
        assert all(ranks <= module_ranks for ranks in group_ranks)
    else:
        assert set(range(torch.distributed.get_world_size())) in group_ranks


@pytest.mark.parametrize(
    "results, expected",
    [
        ([(True, 3.0, 1), (True, 4.0, 2)], (True, 5.0, 3)),
        ([(True, 3.0, None), (False, None, None)], (False, None, None)),
        ([], (True, None, None)),
    ],
)
def test_combine_module_results(results, expected):
    """A rank with several or no active modules combines their step results."""
    module_infos = {
        f"module{index}": ModuleOptimizerInfo(
            SimpleNamespace(step=lambda result=result: result), None, None, True
        )
        for index, result in enumerate(results)
    }
    optimizer = MimoOptimizer(module_infos, OptimizerConfig(mimo_module_local_grad_clip=True))
    assert optimizer.step() == expected


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
        ),
    ],
)
@pytest.mark.parametrize(
    "norms, tensor_indices, expected",
    [
        ([3.0], [0], 3.0),
        ([3.0, 4.0], [0, 1], 5.0),
        ([3.0, 4.0, 12.0], [0, 2], 13.0),
        ([3.0, 4.0], [1], 5.0),
        ([0.0, 0.0], [0, 1], 0.0),
        ([1e20, 1e20], [0, 1], 2**0.5 * 1e20),
        ([float("inf"), float("nan")], [0, 1], float("inf")),
        ([float("nan"), 3.0], [0, 1], float("nan")),
        ([None, 3.0], [1], None),
    ],
)
def test_combine_tensor_norms_without_scalar_extraction(device, norms, tensor_indices, expected):
    """Aggregating module norms must not force device scalars onto the host."""

    class RejectScalarExtraction(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                pytest.fail("Module-local norm aggregation extracted a tensor scalar")
            return func(*args, **(kwargs or {}))

    norms = [
        (
            torch.tensor(norm if index % 2 == 0 else [norm], device=device)
            if index in tensor_indices
            else norm
        )
        for index, norm in enumerate(norms)
    ]
    module_infos = {
        f"module{index}": ModuleOptimizerInfo(
            SimpleNamespace(step=lambda norm=norm: (True, norm, None)), None, None, True
        )
        for index, norm in enumerate(norms)
    }
    optimizer = MimoOptimizer(module_infos, OptimizerConfig(mimo_module_local_grad_clip=True))

    with RejectScalarExtraction():
        success, grad_norm, num_zeros = optimizer.step()

    assert success is True
    assert num_zeros is None
    if expected is None:
        assert grad_norm is None
    else:
        assert isinstance(grad_norm, torch.Tensor)
        assert grad_norm.device.type == device
        torch.testing.assert_close(grad_norm, torch.full_like(grad_norm, expected), equal_nan=True)
        if len(norms) == 1:
            assert grad_norm is norms[0]


def test_split_step_api_rejects_local_clip():
    """The split step API runs the joint path, so it refuses the module-local option."""
    optimizer = MimoOptimizer({}, OptimizerConfig(mimo_module_local_grad_clip=True))
    with pytest.raises(NotImplementedError, match="mimo_module_local_grad_clip"):
        optimizer.prepare_grads_for_step()


@pytest.mark.parametrize(
    "kwargs",
    [{"fp16": True}, {"bf16": True, "loss_scale": 1.0}, {"grad_norm_skip_threshold": 10.0}],
)
@pytest.mark.parametrize("optimizer_factory", [MimoOptimizer, get_mimo_optimizer])
def test_rejects_configs_where_one_module_can_skip(kwargs, optimizer_factory):
    """Loss scaling and a finite skip threshold could skip one module's update alone."""
    config = OptimizerConfig(mimo_module_local_grad_clip=True, **kwargs)
    # The factory must reject the config before accessing the model or building optimizers.
    with pytest.raises(ValueError, match="mimo_module_local_grad_clip"):
        optimizer_factory({}, config)


@pytest.mark.parametrize(
    "kwargs",
    [{"fp16": True}, {"bf16": True, "loss_scale": 1.0}, {"grad_norm_skip_threshold": 10.0}],
)
def test_joint_clip_allows_configs_where_modules_can_skip(kwargs):
    """The joint path synchronizes skips, so these settings remain supported."""
    optimizer = MimoOptimizer({}, OptimizerConfig(**kwargs))
    assert optimizer.config.mimo_module_local_grad_clip is False
