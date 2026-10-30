# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the MimoOptimizer step result and cross-grid parameter norm."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.optimizer import MimoOptimizer, ModuleOptimizerInfo
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from tests.unit_tests.test_utilities import Utils


def _optimizer(step_succeeds: bool, ssa: bool = False) -> MimoOptimizer:
    module = SimpleNamespace(get_parameters=lambda: [], step_with_ready_grads=lambda: step_succeeds)
    config = OptimizerConfig(
        log_num_zeros_in_grad=False,
        ssa_separate_indexer_grad_clip=ssa,
        grad_norm_skip_threshold=3.0,
    )
    opt = MimoOptimizer({"encoder": ModuleOptimizerInfo(module, None, None, True)}, config)
    opt._prepared_main_grad_norm = 2.0
    return opt


class _NoDistributed:
    def __getattr__(self, name):
        raise AssertionError(f"torch.distributed.{name} used after the module steps")


@pytest.mark.parametrize(
    "ssa,grad_norm,success", [(False, 4.0, True), (True, 2.0, True), (True, 4.0, False)]
)
def test_step_with_finite_grads_needs_no_collective(monkeypatch, ssa, grad_norm, success):
    """After the module steps or a skipped update, no rank waits for the other modules."""
    opt = _optimizer(True, ssa)
    monkeypatch.setattr(torch, "distributed", _NoDistributed())
    assert opt.step_with_prepared_grads(False, grad_norm) == (success, grad_norm, None)


@pytest.mark.parametrize("ssa", [False, True])
def test_failed_module_step_with_finite_grads_raises(ssa):
    """A module step that is not skipped and fails with finite gradients raises on its own rank."""
    with pytest.raises(RuntimeError, match="failed with finite gradients"):
        _optimizer(False, ssa).step_with_prepared_grads(False, 2.0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 ranks.")
def test_param_norm_combines_disjoint_module_grids(monkeypatch):
    """Parameter logging sums module norms without reading global model-parallel state."""
    from examples.mimo.training.topology import ModuleGridSpec, create_topology
    from megatron.training.utils import common_utils

    Utils.initialize_distributed()
    topology = None
    try:
        world_size = torch.distributed.get_world_size()
        if world_size < 2 or world_size % 2:
            pytest.skip("Requires an even distributed world size >= 2")
        module_size = world_size // 2
        encoder_name = "encoder"
        topology = create_topology(
            [
                ModuleGridSpec(name=encoder_name, num_ranks=module_size, tp=module_size),
                ModuleGridSpec(
                    name=MIMO_LANGUAGE_MODULE_KEY,
                    num_ranks=module_size,
                    tp=module_size,
                    rank_offset=module_size,
                ),
            ]
        )

        model = torch.nn.Module()
        model.mimo_config = SimpleNamespace(module_to_grid_map=topology.grids)
        model.language_model = None
        model.modality_submodules = torch.nn.ModuleDict()

        active_names = list(topology.schedule_pg_collection.keys())
        assert len(active_names) == 1
        active_name = active_names[0]
        active_module = torch.nn.Linear(2, 2, bias=False, device="cuda")
        active_module.weight.data.fill_(1.0 if active_name == encoder_name else 2.0)
        if active_name == MIMO_LANGUAGE_MODULE_KEY:
            model.language_model = active_module
        else:
            model.modality_submodules[active_name] = active_module

        monkeypatch.setattr(
            common_utils, "get_args", lambda: SimpleNamespace(bf16=False, use_megatron_fsdp=False)
        )
        assert common_utils.calc_params_l2_norm(
            model, pg_collection=topology.schedule_pg_collection
        ) == pytest.approx(20**0.5)
    finally:
        if topology is not None:
            topology.destroy()
        Utils.destroy_model_parallel()
