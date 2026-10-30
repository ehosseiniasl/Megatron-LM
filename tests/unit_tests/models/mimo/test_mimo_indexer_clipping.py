# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Separate indexer clipping through MIMO's real cross-grid optimizer step."""

import math

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.mimo.optimizer import MimoOptimizer, ModuleOptimizerInfo
from megatron.core.optimizer.optimizer import ChainedOptimizer, FP32Optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.tensor_parallel import layers
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Synthetic gradients require no downloaded data."""


@pytest.fixture
def module_groups(monkeypatch):
    if not torch.cuda.is_available() or Utils.world_size < 4 or Utils.world_size % 2:
        pytest.skip("Requires an even distributed world size >= 4")
    Utils.initialize_distributed()
    assert not parallel_state.model_parallel_is_initialized()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    size = world_size // 2
    groups = [
        torch.distributed.new_group(list(range(0, size))),
        torch.distributed.new_group(list(range(size, world_size))),
    ]
    singletons = [torch.distributed.new_group([r]) for r in range(world_size)]

    def no_mpu(*_args, **_kwargs):
        pytest.fail("MIMO clipping must use explicit process groups")

    for name in (
        "get_model_parallel_group",
        "get_tensor_model_parallel_group",
        "get_context_parallel_group",
    ):
        monkeypatch.setattr(parallel_state, name, no_mpu)
    for name in (
        "get_tensor_model_parallel_rank",
        "get_gtp_weight_remat_rank",
        "get_expert_gtp_weight_remat_rank",
    ):
        monkeypatch.setattr(layers, name, no_mpu)
    try:
        yield rank >= size, groups[rank // size], singletons[rank]
    finally:
        for group in reversed(groups + singletons):
            if group != torch.distributed.GroupMember.NON_GROUP_MEMBER:
                torch.distributed.destroy_process_group(group)
        Utils.destroy_model_parallel()


def _parameter(gradient, group=None):
    param = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    param.grad = torch.full_like(param, gradient)
    if group is not None:
        param.grad_norm_group = group
    return param


def _build_optimizer(module_groups, config, frozen_encoder, include_mtp=False):
    language_rank, stats_group, singleton = module_groups
    params = []
    originals = []

    def leaf(parameters):
        inner = torch.optim.SGD(parameters, lr=1.0) if parameters else None
        opt = FP32Optimizer(inner, config, init_state_fn=lambda _: None)
        opt.grad_stats_parallel_group = stats_group
        opt.tp_group = opt.expert_tp_group = singleton
        opt.gtp_group = opt.expert_gtp_group = singleton
        return opt

    opt = None
    active = language_rank or not frozen_encoder
    if active:
        # Sharded gradients have total main norm 5 across the active module grids.
        main_norm = 5.0 if frozen_encoder else (3.0 if language_rank else 4.0)
        main = _parameter(main_norm / math.sqrt(stats_group.size()))
        params.append(main)
        indexer_params = []
        if language_rank and stats_group.rank() == stats_group.size() - 1:
            # All other language ranks have an empty local indexer optimizer shard.
            indexer_params.append(
                _parameter(12.0, "dsa_indexer" if config.ssa_separate_indexer_grad_clip else None)
            )
            if include_mtp:
                indexer_params.append(_parameter(100.0, "mtp"))
        params.extend(indexer_params)
        opt = (
            ChainedOptimizer([leaf([main]), leaf(indexer_params)])
            if language_rank
            else leaf(params)
        )
        originals = [param.grad.detach().clone() for param in params]

    infos = {
        "encoder": ModuleOptimizerInfo(
            opt if not language_rank else None, None, None, active and not language_rank
        ),
        "language": ModuleOptimizerInfo(opt if language_rank else None, None, None, language_rank),
    }
    return MimoOptimizer(infos, config), params, originals


def _scale(limit, norm):
    return min(1.0, limit / (norm + 1e-6)) if limit > 0 else 1.0


@pytest.mark.parametrize("frozen_encoder", [True, False])
@pytest.mark.parametrize(
    "main_clip,indexer_clip", [(1.0, 1.0), (1.0, 0.25), (0.0, 0.25), (1.0, 0.0)]
)
def test_separate_clipping_across_disjoint_grids(
    module_groups, frozen_encoder, main_clip, indexer_clip
):
    config = OptimizerConfig(
        optimizer="sgd",
        lr=1.0,
        clip_grad=main_clip,
        ssa_separate_indexer_grad_clip=True,
        ssa_indexer_clip_grad=indexer_clip,
    )
    optimizer, params, gradients = _build_optimizer(module_groups, config, frozen_encoder)
    for _ in range(2):
        before = [param.detach().clone() for param in params]
        for param, gradient in zip(params, gradients):
            param.grad = gradient.clone()
        successful, norm, _ = optimizer.step()
        assert successful
        assert norm == pytest.approx(13.0, rel=1e-6)
        assert optimizer.grad_norms_by_group["dsa_indexer"] == pytest.approx(12.0)
        for param, initial, gradient in zip(params, before, gradients):
            indexer = getattr(param, "grad_norm_group", None) == "dsa_indexer"
            scale = _scale(indexer_clip, 12.0) if indexer else _scale(main_clip, 5.0)
            torch.testing.assert_close(param, initial - gradient * scale)


@pytest.mark.parametrize("frozen_encoder", [True, False])
def test_shared_clipping_is_unchanged(module_groups, monkeypatch, frozen_encoder):
    config = OptimizerConfig(optimizer="sgd", lr=1.0, clip_grad=1.0)
    optimizer, params, gradients = _build_optimizer(module_groups, config, frozen_encoder)

    def no_separate_norms():
        pytest.fail("Flag-off MIMO must not add separate-norm collectives")

    monkeypatch.setattr(optimizer, "_compute_grad_norms_by_group", no_separate_norms)
    successful, norm, _ = optimizer.step()
    assert successful
    assert norm == pytest.approx(13.0, rel=1e-6)
    for param, gradient in zip(params, gradients):
        torch.testing.assert_close(param, -gradient * _scale(1.0, 13.0))


@pytest.mark.parametrize("frozen_encoder", [True, False])
def test_skip_threshold_includes_indexer_on_every_rank(module_groups, frozen_encoder):
    config = OptimizerConfig(
        optimizer="sgd", lr=1.0, ssa_separate_indexer_grad_clip=True, grad_norm_skip_threshold=6.0
    )
    optimizer, params, _ = _build_optimizer(module_groups, config, frozen_encoder)
    successful, norm, _ = optimizer.step()
    assert not successful
    assert norm == pytest.approx(13.0, rel=1e-6)
    for param in params:
        assert param.item() == 0.0


def test_separate_mtp_norm_stays_out_of_reporting_and_skip(module_groups):
    config = OptimizerConfig(
        optimizer="sgd",
        lr=1.0,
        clip_grad=1.0,
        ssa_separate_indexer_grad_clip=True,
        ssa_indexer_clip_grad=0.25,
        grad_norm_skip_threshold=20.0,
    )
    optimizer, params, gradients = _build_optimizer(module_groups, config, True, include_mtp=True)
    successful, norm, _ = optimizer.step()
    assert successful
    assert norm == pytest.approx(13.0, rel=1e-6)
    for param, gradient in zip(params, gradients):
        group = getattr(param, "grad_norm_group", None)
        limit, group_norm = {None: (1.0, 5.0), "dsa_indexer": (0.25, 12.0), "mtp": (1.0, 100.0)}[
            group
        ]
        torch.testing.assert_close(param, -gradient * _scale(limit, group_norm))


@pytest.mark.parametrize("phase1", [False, True])
def test_prepared_step_reports_indexer_and_reuses_current_norms(module_groups, monkeypatch, phase1):
    config = OptimizerConfig(
        optimizer="sgd",
        lr=1.0,
        clip_grad=1.0,
        ssa_separate_indexer_grad_clip=True,
        ssa_indexer_clip_grad=0.25,
        grad_norm_skip_threshold=6.0,
    )
    optimizer, params, gradients = _build_optimizer(module_groups, config, frozen_encoder=phase1)
    if phase1:
        for param in params:
            if getattr(param, "grad_norm_group", None) != "dsa_indexer":
                param.requires_grad_(False)
                param.grad = None

    found_inf, reported_norm = optimizer.prepare_grads_for_step()
    assert not found_inf
    assert reported_norm == pytest.approx(12.0 if phase1 else 13.0)
    assert reported_norm > 6.0  # The native pre-update rerun gate must see the indexer spike.
    for param in params:
        assert param.item() == 0.0  # Preparation must not update parameters.

    # A rerun discards prepared gradients and measures fresh values.
    optimizer.discard_prepared_grads()
    for param, gradient in zip(params, gradients):
        if getattr(param, "grad_norm_group", None) == "dsa_indexer":
            param.grad = gradient / 6.0  # Indexer norm falls from 12 to 2.
        elif not phase1:
            param.grad = gradient.clone()
    found_inf, reported_norm = optimizer.prepare_grads_for_step()
    assert not found_inf
    assert reported_norm == pytest.approx(2.0 if phase1 else math.sqrt(29))

    def no_repeated_preparation():
        pytest.fail("Commit must reuse the current prepared main and group norms")

    monkeypatch.setattr(optimizer, "prepare_grads", no_repeated_preparation)
    monkeypatch.setattr(optimizer, "get_grad_norm", no_repeated_preparation)
    monkeypatch.setattr(optimizer, "_compute_grad_norms_by_group", no_repeated_preparation)
    successful, norm, _ = optimizer.step_with_prepared_grads(found_inf, reported_norm)
    assert successful
    assert norm == reported_norm
    for param, gradient in zip(params, gradients):
        if getattr(param, "grad_norm_group", None) == "dsa_indexer":
            torch.testing.assert_close(param, -gradient / 6.0 * _scale(0.25, 2.0))
        elif phase1:
            assert param.item() == 0.0
        else:
            torch.testing.assert_close(param, -gradient * _scale(1.0, 5.0))
