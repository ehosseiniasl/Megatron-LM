# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small-gradient metric reference, placement, integration and memory-lifetime tests."""

import weakref
from argparse import ArgumentParser
from unittest import mock

import pytest
import torch

from megatron.training.arguments import _add_logging_args
from megatron.training.tensor_metrics import small_wgrad
from megatron.training.tensor_metrics.core import (
    FlatShard,
    MetricSite,
    MetricTensor,
    Owned,
    RankRelation,
    Replica,
    SegmentedFlatShard,
    Shard,
    TensorMetricExecutor,
)
from megatron.training.tensor_metrics.observer import (
    _prepare_optimizer_metric_values,
    build_tensor_metric_observer,
    parse_tensor_metric_specs,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.training.test_tensor_metric_observer import (
    _fp32_optimizer,
    _layer_model,
    _pg_collection,
)


def _value(data, name='decoder.layers.0.weight', relations=(), **kwargs):
    tensor = data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.float64)
    return MetricTensor(tensor, (MetricSite(name, 'wgrad'),), relations, **kwargs)


def _metrics(absolute=(0, 1e-8, 0.5), relative=(0, 0.01, 1)):
    return tuple(
        s.metric
        for s in parse_tensor_metric_specs(
            [
                f'{name}:1'
                for name in (
                    'wgrad-small-fraction',
                    'wgrad-small-fraction-individual',
                    'wgrad-relative-small-fraction',
                    'wgrad-relative-small-fraction-individual',
                )
            ],
            wgrad_small_thresholds=absolute,
            wgrad_relative_small_thresholds=relative,
        )
    )


def _run(metrics, batches, groups=None):
    executor = TensorMetricExecutor(groups or {})
    prepared = _prepare_optimizer_metric_values(
        executor, metrics, batches(), replay_batches=batches
    )
    return {
        metric.name: {
            r.label: r.tensor for r in executor.complete(metric, metric.start(prepared[id(metric)]))
        }
        for metric in metrics
    }


def test_threshold_cli_validation_and_labels():
    parser = ArgumentParser()
    _add_logging_args(parser)
    args = parser.parse_args(
        [
            '--tensor-metrics-wgrad-small-thresholds',
            '0',
            '1e-8',
            '--tensor-metrics-wgrad-relative-small-thresholds',
            '0.01',
            '0.1',
        ]
    )
    assert args.tensor_metrics_wgrad_small_thresholds == [0, 1e-8]
    assert args.tensor_metrics_wgrad_relative_small_thresholds == [0.01, 0.1]
    assert parser.parse_args([]).tensor_metrics_wgrad_relative_small_thresholds == [0.01]
    metric = parse_tensor_metric_specs(
        ['wgrad-small-fraction:2'], wgrad_small_thresholds=[1e-8, 0, 1e-8]
    )[0].metric
    assert metric.thresholds == (0, 1e-8)
    result = _run((metric,), lambda: [[_value([0, 1e-8])]])
    assert result[metric.name]['global/le_1e-8'] == 1
    assert result[metric.name]['global/le_0'] == 0.5
    default = parse_tensor_metric_specs(['wgrad-relative-small-fraction:3'])[0]
    assert default.metric.thresholds == (0.01,)
    assert default.is_due(2) and not default.is_due(1)


@pytest.mark.parametrize('thresholds', [(), (-1,), (float('nan'),), (float('inf'),)])
@pytest.mark.parametrize('relative', [False, True])
def test_invalid_thresholds(thresholds, relative):
    name = 'wgrad-relative-small-fraction' if relative else 'wgrad-small-fraction'
    key = 'wgrad_relative_small_thresholds' if relative else 'wgrad_small_thresholds'
    with pytest.raises(ValueError, match='finite, nonnegative'):
        parse_tensor_metric_specs([f'{name}:1'], **{key: thresholds})


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_counts_match_reference_and_weighted_scopes(device, monkeypatch):
    monkeypatch.setattr(small_wgrad, '_CHUNK_SIZE', 2)
    tensors = [
        torch.tensor([0, 1e-8, -0.5, 2.0], dtype=torch.float64, device=device),
        torch.tensor([100.0], dtype=torch.float64, device=device),
    ]
    names = ['decoder.layers.0.weight', 'decoder.layers.1.weight']
    metrics = _metrics()
    result = _run(metrics, lambda: [[_value(t, n)] for t, n in zip(tensors, names)])
    for metric in metrics:
        prefix = 'rms_fraction_' if metric.relative else 'le_'
        for threshold in metric.thresholds:
            counts = []
            for t, name in zip(tensors, names):
                limit = threshold * t.square().mean().sqrt() if metric.relative else threshold
                counts.append((t.abs() <= limit).sum())
                label = prefix + small_wgrad._threshold_label(threshold)
                if metric.name.endswith('-individual'):
                    assert result[metric.name][f'{name}/{label}'] == counts[-1] / t.numel()
            if not metric.name.endswith('-individual'):
                expected = sum(counts).double() / 5
                torch.testing.assert_close(result[metric.name][f'global/{label}'], expected)
                torch.testing.assert_close(
                    result[metric.name][f'family/decoder.layers.ALL.weight/{label}'], expected
                )
    # One tensor's large RMS must not change another tensor's relative cutoff.
    assert result['wgrad-relative-small-fraction']['global/rms_fraction_0.01'] == 0.4


@pytest.mark.parametrize('scale', [1e-30, 1e-10, 1, 1e20])
def test_relative_scale_invariance_and_zero_tensor(scale):
    values = [
        _value(torch.tensor([0, 0.001, -1, 2], dtype=torch.float32) * scale),
        _value([0, 0], 'decoder.layers.1.weight'),
    ]
    results = _run(_metrics(), lambda: [values])
    rel = results['wgrad-relative-small-fraction-individual']
    assert rel['decoder.layers.0.weight/rms_fraction_0.01'] == 0.5
    assert rel['decoder.layers.1.weight/rms_fraction_0'] == 1


@pytest.mark.parametrize('bad', [float('inf'), -float('inf'), float('nan')])
def test_nonfinite_never_small_and_relative_invalidates_containing_scope(bad):
    values = [_value([0, bad]), _value([0, 0], 'decoder.layers.1.weight')]
    results = _run(_metrics(absolute=[1e100]), lambda: [values])
    assert results['wgrad-small-fraction']['global/le_1e100'] == 0.75
    assert results['wgrad-relative-small-fraction']['global/rms_fraction_0.01'].isnan()
    individual = results['wgrad-relative-small-fraction-individual']
    assert individual['decoder.layers.0.weight/rms_fraction_0.01'].isnan()
    assert individual['decoder.layers.1.weight/rms_fraction_0.01'] == 1


def test_empty_and_strided_inputs():
    values = [_value(torch.arange(12.0).reshape(3, 4).t()), _value([], 'empty')]
    results = _run(_metrics(absolute=[0]), lambda: [values, []])
    assert results['wgrad-small-fraction']['global/le_0'] == 1 / 12
    assert results['wgrad-small-fraction-individual']['empty/le_0'] == 0


def test_two_pass_releases_semantic_copies_and_shares_work():
    visits, references = [], []

    def batches():
        visits.append(1)
        for i in range(3):
            assert all(ref() is None for ref in references)
            tensor = torch.tensor([0.0, 1.0])
            references.append(weakref.ref(tensor))
            yield [_value(tensor, f'decoder.layers.{i}.weight')]
            del tensor

    metrics = _metrics()
    calls = []
    original = small_wgrad._local_statistics

    def counted_statistics(*args, **kwargs):
        calls.append(1)  # Do not retain the tensors in mock.call_args_list.
        return original(*args, **kwargs)

    with mock.patch.object(small_wgrad, '_local_statistics', counted_statistics):
        _run(metrics, batches)
    assert len(visits) == 2
    assert len(calls) == 6  # one per tensor/pass, not one per output or threshold
    assert all(ref() is None for ref in references)
    visits.clear()
    _run(metrics[:2], batches)
    assert len(visits) == 1


def test_observer_schedule_and_zero_equivalence(monkeypatch):
    from contextlib import nullcontext

    import megatron.training.tensor_metrics.observer as observer_mod

    monkeypatch.setattr(observer_mod, 'get_nvtx_range', lambda: lambda *a, **kw: nullcontext())
    records = {}
    observer = build_tensor_metric_observer(
        [
            'wgrad-small-fraction:2',
            'wgrad-small-fraction-individual:3',
            'wgrad-relative-small-fraction:2',
            'wgrad-zero-fraction:2',
        ],
        result_sink=lambda m, r, step: records.setdefault(step, {}).update(
            {m.name: {v.label: v.tensor for v in r}}
        ),
        iterations=[5],
        wgrad_small_thresholds=[0],
    )
    model = _layer_model()
    for parameter in model.parameters():
        parameter.main_grad = parameter.detach().clone()
    for iteration in range(5):
        observer(
            model=[model],
            optimizer=_fp32_optimizer(model),
            iteration=iteration,
            pg_collection=_pg_collection(),
        )
    assert set(records) == {2, 3, 4, 5}
    assert set(records[3]) == {'wgrad-small-fraction-individual'}
    for step in (2, 4, 5):
        for scope, zero in records[step]['wgrad-zero-fraction'].items():
            assert records[step]['wgrad-small-fraction'][scope + '/le_0'] == zero


@pytest.mark.parametrize('placement', ['shard', 'flat', 'segmented', 'owned', 'replica'])
def test_distributed_logical_rms_and_count_sharing(placement):
    Utils.initialize_distributed()
    rank, size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    if size < 2:
        pytest.skip('Requires multiple ranks.')
    device = torch.device('cuda', torch.cuda.current_device())
    full = torch.tensor([0.0, 0.1, 1.0, 100.0], device=device)
    if placement in ('owned', 'replica'):
        local = full if rank == 0 or placement == 'replica' else full[:0]
        relation = Owned(0) if placement == 'owned' else Replica()
    else:
        start, end = (4 * rank) // size, (4 * (rank + 1)) // size
        local = full[start:end]
        relation = {
            'shard': Shard(0),
            'flat': FlatShard((4,), start, end),
            'segmented': SegmentedFlatShard((4,), ((start, end),)),
        }[placement]
    value = _value(
        local,
        relations=(RankRelation('test', relation),),
        is_placeholder=placement == 'owned' and rank != 0,
    )
    metrics = _metrics(absolute=[0, 1], relative=[0.01, 0.1])
    with mock.patch.object(
        torch.distributed, 'all_reduce', wraps=torch.distributed.all_reduce
    ) as calls:
        results = _run(metrics, lambda: [[value]], {'test': torch.distributed.group.WORLD})
    assert results['wgrad-relative-small-fraction']['global/rms_fraction_0.01'] == 0.5
    assert results['wgrad-relative-small-fraction']['global/rms_fraction_0.1'] == 0.75
    assert results['wgrad-small-fraction']['global/le_0'] == 0.25
    # Moments + absolute counts + relative counts; summary/individual reuse all reductions.
    assert calls.call_count == (0 if placement == 'replica' else 3)


def test_counts_keep_integer_precision_for_large_populations():
    metric = _metrics(absolute=[0])[0]
    population = 2**30 + 7
    count = population - 1
    result = metric.finalize(torch.tensor([count, population, 0], dtype=torch.int64))
    assert result.item() == count / population
    assert result.item() < 1


def test_distributed_composed_tp_gtp_axes():
    Utils.initialize_distributed()
    rank, size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    if size < 4 or size % 2:
        pytest.skip('Requires an even world of at least four ranks.')
    groups = {}
    created = []
    for ranks in [list(range(i, i + 2)) for i in range(0, size, 2)]:
        group = torch.distributed.new_group(ranks)
        created.append(group)
        if rank in ranks:
            groups['tp'] = group
    for ranks in [list(range(i, size, 2)) for i in range(2)]:
        group = torch.distributed.new_group(ranks)
        created.append(group)
        if rank in ranks:
            groups['gtp'] = group
    try:
        full = torch.arange(size, dtype=torch.float64, device='cuda').square()
        full[-1] = 1000
        value = _value(
            full[rank : rank + 1],
            relations=(
                RankRelation('tp', Shard(0)),
                RankRelation('gtp', FlatShard((size // 2,), rank // 2, rank // 2 + 1)),
            ),
        )
        results = _run(_metrics(), lambda: [[value]], groups)
        rms = full.square().mean().sqrt()
        expected = (full.abs() <= 0.01 * rms).double().mean()
        torch.testing.assert_close(
            results['wgrad-relative-small-fraction']['global/rms_fraction_0.01'], expected
        )
    finally:
        for group in created:
            if group != torch.distributed.GroupMember.NON_GROUP_MEMBER:
                torch.distributed.destroy_process_group(group)
