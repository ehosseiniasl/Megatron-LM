# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.contribution_diagnostics import (
    contribution_rank_map,
    moe_contribution_mtp_depth,
    observe_moe_contributions,
    ranked_contribution_sums,
)
from megatron.training.tensor_metrics.observer import (
    TrainingTensorMetricObserver,
    _validate_forward_observation_model,
    parse_tensor_metric_specs,
)


def test_contribution_metric_registration_and_cadence():
    (scheduled,) = parse_tensor_metric_specs(['moe-contributions:100'])
    assert scheduled.metric.source_kinds == frozenset({'moe_contributions'})
    assert scheduled.metric.name == 'moe-contributions'
    assert not scheduled.is_due(98)
    assert scheduled.is_due(99)


@pytest.mark.parametrize(
    'graph_modules,accepted',
    [
        ([], False),
        ([CudaGraphModule.moe], False),
        ([CudaGraphModule.moe_preprocess], False),
        ([CudaGraphModule.moe_router], False),
        ([CudaGraphModule.attn], True),
    ],
)
def test_contribution_metric_requires_eager_moe_but_allows_attention_graphs(
    graph_modules, accepted
):
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        num_moe_experts=512,
        moe_token_dispatcher_type='alltoall',
        cuda_graph_impl='local',
        cuda_graph_modules=graph_modules,
    )
    if accepted:
        _validate_forward_observation_model([model], frozenset({'moe_contributions'}))
    else:
        with pytest.raises(NotImplementedError, match='eager MoE'):
            _validate_forward_observation_model([model], frozenset({'moe_contributions'}))


def test_contribution_metric_rejects_flex_before_forward():
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_moe_experts=512, moe_token_dispatcher_type='flex')
    with pytest.raises(NotImplementedError, match='alltoall'):
        _validate_forward_observation_model([model], frozenset({'moe_contributions'}))


def test_observer_collects_only_due_steps_and_logs_separate_mtp_depths(monkeypatch):
    import megatron.training.tensor_metrics.observer as observer_module

    monkeypatch.setattr(observer_module, 'get_nvtx_range', lambda: lambda *a, **k: nullcontext())
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    owner = model.decoder.layers[0]
    model.config = owner.config = SimpleNamespace(
        num_moe_experts=2, moe_token_dispatcher_type='alltoall', sequence_parallel=False
    )
    pg = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    groups = SimpleNamespace(
        **{
            name: pg
            for name in (
                'tp',
                'pp',
                'cp',
                'gtp_remat',
                'dp_cp',
                'ep',
                'expt_tp',
                'expt_dp',
                'expt_gtp_remat',
            )
        }
    )
    logged = []
    observer = TrainingTensorMetricObserver(
        parse_tensor_metric_specs(['moe-contributions:100']),
        result_sink=lambda metric, results, step: logged.append((results, step)),
    )
    for iteration in (98, 99):
        with observer.observe_forward_backward(
            model=[model], iteration=iteration, pg_collection=groups
        ):
            if iteration == 98:
                # The helper must exit before touching any tensors when the metric is inactive.
                observe_moe_contributions(None, None, None, None, None)
            else:
                for _ in range(2):
                    for depth in (1, 2):
                        p = torch.tensor([[0.75, 0.25]])
                        _, meta = contribution_rank_map(p, torch.ones_like(p, dtype=torch.bool), 2)
                        meta.ranks = torch.tensor([1.0, 2.0])
                        sums = ranked_contribution_sums(torch.tensor([[3.0], [1.0]]), meta)
                        with moe_contribution_mtp_depth(depth):
                            observe_moe_contributions(
                                owner, torch.tensor([[4.0]]), torch.tensor([[2.0]]), meta, sums
                            )
        observer(model=[model], optimizer=None, iteration=iteration, pg_collection=groups)
    assert len(logged) == 1
    results, step = logged[0]
    assert step == 100
    values = {r.label: r.tensor for r in results}
    assert len(values) == 46
    for depth in (1, 2):
        prefix = f'decoder.layers.0/mtp_{depth}/'
        assert values[prefix + 'tokens'] == 2
        assert values[prefix + 'shared-over-routed-rms'] == 0.5
