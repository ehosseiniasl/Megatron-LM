# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Separate indexer clipping must change gradients, not optimizer grouping or state."""

import math
from argparse import ArgumentParser

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, _get_param_groups, get_megatron_optimizer
from megatron.core.optimizer.optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    copy_optimizer_param_metadata,
)
from megatron.core.transformer import TransformerConfig
from megatron.training.arguments import _add_regularization_args
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _param(grad, group=None, dtype=torch.float32):
    param = torch.nn.Parameter(torch.zeros(1, device='cuda', dtype=dtype))
    param.grad = torch.full_like(param, grad)
    if group is not None:
        param.grad_norm_group = group
    return param


def _optimizer(params, config, kind='fp32', stats_group=None):
    def make_optimizer(parameters):
        inner = torch.optim.SGD(parameters, lr=1.0) if parameters else None
        if kind == 'bf16':
            optimizer = Float16OptimizerWithFloat16Params(
                inner, config, grad_scaler=None, init_state_fn=lambda _: None
            )
        else:
            optimizer = FP32Optimizer(inner, config, init_state_fn=lambda _: None)
        if stats_group is not None:
            optimizer.grad_stats_parallel_group = stats_group
        return optimizer

    if kind == 'chained':
        # Dense/expert optimizers must share one backbone norm, while the indexer
        # has its own norm even when it belongs to the dense optimizer.
        return ChainedOptimizer([make_optimizer(params[::2]), make_optimizer(params[1::2])])
    return make_optimizer(params)


def _scale(limit, norm):
    return min(1.0, limit / (norm + 1.0e-6)) if limit > 0 else 1.0


@pytest.mark.parametrize('kind', ['fp32', 'chained', 'bf16'])
@pytest.mark.parametrize('separate', [False, True])
def test_joint_and_separate_clipping_update_parameters(kind, separate):
    """Actual SGD updates match the joint or independent clipping equations."""
    dtype = torch.bfloat16 if kind == 'bf16' else torch.float32
    backbone = _param(10.0, dtype=dtype)
    indexer = _param(1.0, 'dsa_indexer' if separate else None, dtype=dtype)
    config = OptimizerConfig(
        optimizer='sgd',
        lr=1.0,
        clip_grad=1.0,
        bf16=kind == 'bf16',
        ssa_separate_indexer_grad_clip=separate,
    )
    optimizer = _optimizer([backbone, indexer], config, kind)

    successful, norm, _ = optimizer.step()

    assert successful
    assert float(norm) == pytest.approx(math.sqrt(101), rel=1e-6)
    expected_backbone = -10 * _scale(1.0, 10.0 if separate else math.sqrt(101))
    expected_indexer = -_scale(1.0, 1.0 if separate else math.sqrt(101))
    torch.testing.assert_close(backbone, torch.full_like(backbone, expected_backbone))
    torch.testing.assert_close(indexer, torch.full_like(indexer, expected_indexer))
    if separate:
        assert float(optimizer.grad_norms_by_group['dsa_indexer']) == pytest.approx(1.0)
    if kind == 'bf16' and separate:
        assert indexer.main_param.grad_norm_group == 'dsa_indexer'


@pytest.mark.parametrize('kind', ['fp32', 'chained'])
@pytest.mark.parametrize('main_clip,indexer_clip', [(1.0, 0.25), (0.0, 0.25), (1.0, 0.0)])
def test_independent_clip_thresholds(kind, main_clip, indexer_clip):
    backbone = _param(10.0)
    indexer = _param(2.0, 'dsa_indexer')
    config = OptimizerConfig(
        optimizer='sgd',
        lr=1.0,
        clip_grad=main_clip,
        ssa_separate_indexer_grad_clip=True,
        ssa_indexer_clip_grad=indexer_clip,
    )
    optimizer = _optimizer([backbone, indexer], config, kind)

    successful, norm, _ = optimizer.step()

    assert successful
    assert float(norm) == pytest.approx(math.sqrt(104), rel=1e-6)
    torch.testing.assert_close(backbone, torch.full_like(backbone, -10 * _scale(main_clip, 10)))
    torch.testing.assert_close(indexer, torch.full_like(indexer, -2 * _scale(indexer_clip, 2)))


@pytest.mark.parametrize('kind', ['fp32', 'chained'])
def test_phase_one_frozen_backbone_has_same_indexer_update(kind):
    """Freezing the backbone leaves only the indexer, so equal limits give equal updates."""
    outcomes = []
    for separate in (False, True):
        frozen = torch.nn.Parameter(torch.zeros(1, device='cuda'), requires_grad=False)
        indexer = _param(2.0, 'dsa_indexer' if separate else None)
        config = OptimizerConfig(optimizer='sgd', lr=1.0, ssa_separate_indexer_grad_clip=separate)
        optimizer = _optimizer([indexer], config, kind)

        successful, norm, _ = optimizer.step()

        assert successful
        assert float(norm) == pytest.approx(2.0)
        assert frozen.item() == 0
        outcomes.append(indexer.detach().clone())
    torch.testing.assert_close(*outcomes)


def test_chained_dense_and_expert_share_backbone_norm_with_independent_mtp():
    dense = _param(3.0)
    expert = _param(4.0)
    indexer = _param(2.0, 'dsa_indexer')
    mtp = _param(100.0, 'mtp')
    config = OptimizerConfig(
        optimizer='sgd',
        lr=1.0,
        clip_grad=1.0,
        ssa_separate_indexer_grad_clip=True,
        ssa_indexer_clip_grad=0.5,
        grad_norm_skip_threshold=6.0,
    )
    optimizer = _optimizer([dense, expert, indexer, mtp], config, 'chained')

    successful, norm, _ = optimizer.step()

    assert successful  # The MTP norm of 100 must not affect the skip check.
    assert float(norm) == pytest.approx(math.sqrt(29), rel=1e-6)
    assert float(optimizer.grad_norms_by_group['mtp']) == pytest.approx(100.0)
    torch.testing.assert_close(dense, torch.full_like(dense, -3 * _scale(1.0, 5)))
    torch.testing.assert_close(expert, torch.full_like(expert, -4 * _scale(1.0, 5)))
    torch.testing.assert_close(indexer, torch.full_like(indexer, -2 * _scale(0.5, 2)))
    torch.testing.assert_close(mtp, torch.full_like(mtp, -100 * _scale(1.0, 100)))


def test_chained_skip_threshold_includes_indexer_norm():
    backbone = _param(0.5)
    indexer = _param(10.0, 'dsa_indexer')
    config = OptimizerConfig(
        optimizer='sgd',
        lr=1.0,
        clip_grad=1.0,
        ssa_separate_indexer_grad_clip=True,
        grad_norm_skip_threshold=1.0,
    )
    optimizer = _optimizer([backbone, indexer], config, 'chained')

    successful, norm, _ = optimizer.step()

    assert not successful
    assert float(norm) == pytest.approx(math.sqrt(100.25), rel=1e-6)
    assert backbone.item() == 0.0
    assert indexer.item() == 0.0


@pytest.mark.parametrize('kind', ['fp32', 'chained'])
def test_global_indexer_norm_when_only_one_rank_owns_indexer(kind):
    """Ranks without an indexer shard must participate in its norm reductions."""
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        pytest.skip('Needs at least two ranks to exercise empty local indexer shards')
    backbone = _param(3.0 / math.sqrt(world_size))
    params = [backbone]
    if torch.distributed.get_rank() == 0:
        params.append(_param(4.0, 'dsa_indexer'))
    config = OptimizerConfig(
        optimizer='sgd',
        lr=1.0,
        clip_grad=1.0,
        ssa_separate_indexer_grad_clip=True,
        ssa_indexer_clip_grad=0.25,
    )
    optimizer = _optimizer(params, config, kind, torch.distributed.group.WORLD)

    successful, norm, _ = optimizer.step()

    assert successful
    assert float(norm) == pytest.approx(5.0, rel=1e-6)
    assert float(optimizer.grad_norms_by_group['dsa_indexer']) == pytest.approx(4.0)
    torch.testing.assert_close(
        backbone, torch.full_like(backbone, -3.0 / math.sqrt(world_size) * _scale(1.0, 3.0))
    )
    if len(params) == 2:
        torch.testing.assert_close(params[1], torch.full_like(params[1], -4 * _scale(0.25, 4)))


def test_chained_global_skip_when_other_ranks_have_no_trainable_parameters():
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        pytest.skip('Needs at least two ranks to exercise stub optimizers')
    params = [_param(10.0, 'dsa_indexer')] if torch.distributed.get_rank() == 0 else []
    config = OptimizerConfig(
        optimizer='sgd', lr=1.0, ssa_separate_indexer_grad_clip=True, grad_norm_skip_threshold=1.0
    )
    optimizer = _optimizer(params, config, 'chained', torch.distributed.group.WORLD)

    successful, norm, _ = optimizer.step()

    assert not successful
    assert float(norm) == pytest.approx(10.0)
    if params:
        assert params[0].item() == 0.0


@pytest.mark.parametrize('separate', [False, True])
def test_distributed_optimizer_clips_actual_ddp_gradient_shards(separate):
    """Exercise automatic tagging, backward, DP reduction, master shards and Adam."""

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16, device='cuda')
            self.indexer = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16, device='cuda')

        def forward(self, inputs):
            return 10 * self.backbone(inputs).sum() + self.indexer(inputs).sum()

    model = Model()
    model = DistributedDataParallel(
        TransformerConfig(num_layers=1, num_attention_heads=1, hidden_size=16),
        DistributedDataParallelConfig(use_distributed_optimizer=True),
        model,
    )
    config = OptimizerConfig(
        optimizer='adam',
        lr=0.01,
        weight_decay=0.0,
        bf16=True,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        ssa_separate_indexer_grad_clip=separate,
        ssa_indexer_clip_grad=0.25 if separate else None,
    )
    optimizer = get_megatron_optimizer(config, [model])
    model.zero_grad_buffer()
    model(torch.ones(1, 16, dtype=torch.bfloat16, device='cuda')).backward()
    model.finish_grad_sync()

    successful, norm, _ = optimizer.step()

    expected_norm = math.sqrt(256 * (10**2 + 1))
    assert successful
    assert float(norm) == pytest.approx(expected_norm, rel=1e-5)
    owned_indexer_elements = 0
    for param in optimizer.get_parameters():
        if param.grad is None:
            continue
        if getattr(param, 'grad_norm_group', None) == 'dsa_indexer':
            owned_indexer_elements += param.numel()
            expected = _scale(0.25, 16)
            torch.testing.assert_close(param.grad, torch.full_like(param.grad, expected))
        elif separate:
            expected = 10 * _scale(1.0, 160)
            torch.testing.assert_close(param.grad, torch.full_like(param.grad, expected))
        else:
            # A shard may contain either backbone or indexer gradients.
            allowed = torch.tensor([1.0, 10.0], device='cuda') * _scale(1.0, expected_norm)
            distance = (param.grad.flatten()[:, None] - allowed).abs().min(dim=1).values
            assert torch.all(distance < 1e-6)
    tagged_count = torch.tensor(owned_indexer_elements, device='cuda')
    torch.distributed.all_reduce(tagged_count)
    assert tagged_count.item() == (256 if separate else 0)


@pytest.mark.parametrize('separate', [False, True])
def test_parameter_tagging_preserves_optimizer_groups(separate):
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(1, 1, bias=False, device='cuda')
    model.indexer = torch.nn.Linear(1, 1, bias=False, device='cuda')
    model.block = torch.nn.Module()
    model.block.indexer = torch.nn.Linear(1, 1, bias=False, device='cuda')
    model.frozen = torch.nn.Parameter(torch.ones(1, device='cuda'), requires_grad=False)
    config = OptimizerConfig(lr=1.0, ssa_separate_indexer_grad_clip=separate)

    groups = _get_param_groups([model], config, config_overrides={})

    assert len(groups) == 1  # No extra scheduler/checkpoint parameter-group identity.
    assert {id(p) for p in groups[0]['params']} == {
        id(model.backbone.weight),
        id(model.indexer.weight),
        id(model.block.indexer.weight),
    }
    assert getattr(model.backbone.weight, 'grad_norm_group', None) is None
    for param in (model.indexer.weight, model.block.indexer.weight):
        assert getattr(param, 'grad_norm_group', None) == ('dsa_indexer' if separate else None)
    if separate:
        config.ssa_separate_indexer_grad_clip = False
        rebuilt_groups = _get_param_groups([model], config, config_overrides={})
        assert len(rebuilt_groups) == 1
        for param in rebuilt_groups[0]['params']:
            assert getattr(param, 'grad_norm_group', None) is None


def test_existing_indexer_norm_group_is_not_silently_overwritten():
    model = torch.nn.Module()
    model.indexer = torch.nn.Linear(1, 1, bias=False, device='cuda')
    model.indexer.weight.grad_norm_group = 'mtp'
    config = OptimizerConfig(lr=1.0, ssa_separate_indexer_grad_clip=True)

    with pytest.raises(ValueError, match='overlapping clipping groups'):
        _get_param_groups([model], config, config_overrides={})


def test_indexer_metadata_survives_optimizer_shard_copy():
    source = _param(1.0, 'dsa_indexer')
    source.allreduce = True
    destination = source.detach().clone()

    copy_optimizer_param_metadata(destination, source)

    assert destination.grad_norm_group == 'dsa_indexer'
    assert destination.allreduce is True


def test_indexer_clip_cli_defaults_and_explicit_values():
    parser = _add_regularization_args(ArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.ssa_separate_indexer_grad_clip is False
    assert defaults.ssa_indexer_clip_grad is None
    explicit = parser.parse_args(
        [
            '--ssa-separate-indexer-grad-clip',
            '--ssa-indexer-clip-grad',
            '0.25',
            '--clip-grad',
            '0.5',
        ]
    )
    assert explicit.ssa_separate_indexer_grad_clip is True
    assert explicit.ssa_indexer_clip_grad == 0.25
    assert explicit.clip_grad == 0.5


@pytest.mark.parametrize('limit', [-1.0, float('inf'), float('nan')])
def test_invalid_indexer_clip_threshold(limit):
    with pytest.raises(ValueError, match='finite and non-negative'):
        OptimizerConfig(ssa_separate_indexer_grad_clip=True, ssa_indexer_clip_grad=limit)


def test_indexer_threshold_requires_separate_clipping():
    with pytest.raises(ValueError, match='requires ssa_separate_indexer_grad_clip'):
        OptimizerConfig(ssa_indexer_clip_grad=0.25)


@pytest.mark.parametrize('phase1', [False, True])
def test_chained_prepared_norm_includes_indexer_and_refreshes_on_rerun(monkeypatch, phase1):
    indexer = _param(12.0, 'dsa_indexer')
    backbone = None if phase1 else _param(5.0)
    params = [indexer] if phase1 else [backbone, indexer]
    config = OptimizerConfig(
        optimizer='sgd',
        lr=1.0,
        clip_grad=1.0,
        ssa_separate_indexer_grad_clip=True,
        ssa_indexer_clip_grad=0.25,
        grad_norm_skip_threshold=6.0,
    )
    optimizer = _optimizer(params, config, 'chained')
    found_inf, reported_norm = optimizer.prepare_grads_for_step()
    assert not found_inf
    assert reported_norm == pytest.approx(12.0 if phase1 else 13.0)
    assert reported_norm > 6.0  # Matches the value consumed by the training rerun gate.
    for param in params:
        assert param.item() == 0.0

    optimizer.discard_prepared_grads()
    indexer.grad = torch.full_like(indexer, 2.0)
    if backbone is not None:
        backbone.grad = torch.full_like(backbone, 5.0)
    found_inf, reported_norm = optimizer.prepare_grads_for_step()
    assert not found_inf
    assert reported_norm == pytest.approx(2.0 if phase1 else math.sqrt(29))

    def no_repeated_preparation():
        pytest.fail('Commit must reuse the current prepared main and group norms')

    monkeypatch.setattr(optimizer, 'prepare_grads', no_repeated_preparation)
    monkeypatch.setattr(optimizer, 'get_grad_norm', no_repeated_preparation)
    monkeypatch.setattr(optimizer, '_compute_grad_norms_by_group', no_repeated_preparation)
    successful, norm, _ = optimizer.step_with_prepared_grads(found_inf, reported_norm)
    assert successful
    assert norm == reported_norm
    torch.testing.assert_close(indexer, torch.full_like(indexer, -2 * _scale(0.25, 2)))
    if backbone is not None:
        torch.testing.assert_close(backbone, torch.full_like(backbone, -5 * _scale(1.0, 5)))
