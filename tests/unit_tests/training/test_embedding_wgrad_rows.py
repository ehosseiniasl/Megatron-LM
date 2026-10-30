# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Check row statistics against dense references, including distributed optimizer layouts."""

import os
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch

import megatron.training.tensor_metrics.observer as observer_mod
from megatron.core.optimizer import LayerWiseDistributedOptimizer
from megatron.training.arguments import _add_logging_args
from megatron.training.tensor_metrics import (
    FlatShard,
    MetricSite,
    MetricTensor,
    RankRelation,
    TensorMetricExecutor,
)
from megatron.training.tensor_metrics.embedding_metrics import (
    EmbeddingWgradRowL2Metric,
    _embedding_row_squared_l2,
)
from megatron.training.tensor_metrics.observer import build_tensor_metric_observer
from tests.unit_tests.training.test_tensor_metric_sources import (
    _disable_tensor_metric_profiling,
    _distributed_optimizer,
    _fp32_optimizer,
    _pg_collection,
)


@pytest.mark.parametrize("start,end", [(0, 35), (1, 34), (4, 6), (5, 10), (35, 35)])
def test_partial_rows_match_dense_reference(start, end):
    gradient = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    before = gradient.clone()
    shard = gradient.flatten()[start:end]
    got = _embedding_row_squared_l2(
        shard, (7, 5), (RankRelation("dp", FlatShard((7, 5), start, end)),), 2, 8, torch.tensor(4.0)
    )
    dense = torch.zeros_like(gradient)
    dense.flatten()[start:end] = shard
    expected = torch.zeros(8)
    expected[2:] = (dense[:6].double() / 4).square().sum(1).float()
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(gradient, before, rtol=0, atol=0)


def _model(parameter):
    model = torch.nn.Module()
    model.embedding = torch.nn.Module()
    model.embedding.word_embeddings = torch.nn.Module()
    model.embedding.word_embeddings.weight = parameter
    return model


def test_executor_returns_labeled_row_scalars_without_observer():
    value = MetricTensor(
        torch.tensor([0.0, 9.0, 16.0, 9.0]),
        (MetricSite("embedding.word_embeddings.weight", "embedding_wgrad_row_squared_l2"),),
        (),
    )

    results = TensorMetricExecutor({}).run(EmbeddingWgradRowL2Metric(), [value])

    assert [result.label for result in results] == (
        ["mean"] + [f"top_{k}/l2" for k in range(1, 5)] + [f"top_{k}/token_id" for k in range(1, 5)]
    )
    assert all(result.tensor.ndim == 0 for result in results)
    torch.testing.assert_close(
        torch.stack([result.tensor for result in results]),
        torch.tensor([2.5, 4.0, 3.0, 3.0, 0.0, 2.0, 1.0, 3.0, 0.0], dtype=torch.float64),
    )


def _check_observer(monkeypatch, parameter, optimizer, pg, reference, scale, has_tokenizer=True):
    monkeypatch.setattr(observer_mod, "get_args", lambda: SimpleNamespace())

    def tokenizer():
        if not has_tokenizer:
            raise AssertionError("tokenizer is not initialized")
        return SimpleNamespace(vocab_size=len(reference))

    monkeypatch.setattr(observer_mod, "get_tokenizer", tokenizer)
    optimizer.get_loss_scale = lambda: torch.tensor([scale], device=reference.device)
    captured = {}
    observer = build_tensor_metric_observer(
        ["embedding-wgrad-row-l2:2"],
        result_sink=lambda metric, results, iteration: captured.update(
            {r.label: r.tensor for r in results}
        ),
    )
    original = parameter.main_grad.clone()
    observer(model=[_model(parameter)], optimizer=optimizer, iteration=0, pg_collection=pg)
    assert not captured
    observer(model=[_model(parameter)], optimizer=optimizer, iteration=1, pg_collection=pg)
    norms = reference.double().norm(dim=1)
    expected_ids = sorted(range(len(norms)), key=lambda i: (-norms[i].item(), i))[:10]
    assert len(captured) == 1 + 2 * min(10, len(reference))
    torch.testing.assert_close(captured["mean"], norms.mean(), rtol=2e-6, atol=1e-7)
    for k, token in enumerate(expected_ids, 1):
        torch.testing.assert_close(captured[f"top_{k}/l2"], norms[token], rtol=2e-6, atol=1e-7)
        assert captured[f"top_{k}/token_id"].item() == token
    torch.testing.assert_close(parameter.main_grad, original, rtol=0, atol=0, equal_nan=True)


def test_explicit_vocab_size_without_global_tokenizer(monkeypatch):
    parser = ArgumentParser()
    _add_logging_args(parser)
    args = parser.parse_args(["--tensor-metrics-embedding-vocab-size", "262144"])
    monkeypatch.setattr(observer_mod, "get_args", lambda: args)

    def no_tokenizer():
        raise AssertionError("tokenizer is not initialized")

    monkeypatch.setattr(observer_mod, "get_tokenizer", no_tokenizer)
    observer = build_tensor_metric_observer(["embedding-wgrad-row-l2:1"])
    assert observer._embedding_vocab_size == 262144
    args.tensor_metrics_embedding_vocab_size = None
    observer = build_tensor_metric_observer(["embedding-wgrad-row-l2:1"])
    assert observer._embedding_vocab_size is None
    args.tensor_metrics_embedding_vocab_size = 0
    with pytest.raises(ValueError, match="must be positive"):
        build_tensor_metric_observer(["embedding-wgrad-row-l2:1"])


@pytest.mark.parametrize("vocab_size", [1, 6, 15])
@pytest.mark.parametrize("all_zero", [False, True])
def test_mean_includes_zero_rows_and_excludes_padding(monkeypatch, vocab_size, all_zero):
    reference = torch.zeros(vocab_size, 5)
    if not all_zero:
        reference[-1] = 3
    parameter = torch.nn.Parameter(torch.zeros(vocab_size + 3, 5))
    parameter.main_grad = torch.cat((reference * 8, torch.full((3, 5), float("nan"))))
    _check_observer(
        monkeypatch, parameter, _fp32_optimizer(parameter), _pg_collection(), reference, 8
    )


@pytest.mark.parametrize(
    "layout", ["replica", "tp", "tp_hidden", "gtp", "flat", "owner", "tp_gtp", "gtp_flat"]
)
@pytest.mark.parametrize("has_tokenizer", [True, False])
def test_distributed_row_ids_and_norms(monkeypatch, layout, has_tokenizer):
    if int(os.environ.get("WORLD_SIZE", "1")) not in (4, 8):
        pytest.skip("Requires four or eight GPU ranks for nested shard coverage.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    world = torch.distributed.group.WORLD
    rank, size = world.rank(), world.size()
    assert size % 2 == 0
    device = torch.device("cuda", torch.cuda.current_device())
    pg = _pg_collection()
    reference = torch.arange(15 * 8, dtype=torch.float32, device=device).reshape(15, 8)
    reference[0] = 0
    reference[1] = reference[14]  # exact tie across distant vocabulary shards
    padding_value = float("nan") if has_tokenizer else 1000.0
    padded = torch.cat((reference, torch.full((1, 8), padding_value, device=device)))
    if not has_tokenizer:
        # Vocabulary alignment rows count in the fallback, including in the top ten;
        # NaN-filled physical GTP padding below must still be excluded.
        reference = padded
    local = padded
    attributes = {}
    if layout in ("tp", "tp_hidden"):
        dim = 0 if layout == "tp" else 1
        pg.tp = world
        local = padded.chunk(size, dim=dim)[rank].contiguous()
        attributes = dict(tensor_model_parallel=True, partition_dim=dim, partition_stride=1)
    elif layout == "gtp":
        pg.gtp_remat = world
        rows = (16 + size - 1) // size + 1
        physical = torch.full((rows * size, 8), float("nan"), device=device)
        physical[:16] = padded
        local = physical.chunk(size)[rank].contiguous()
        attributes = dict(is_gtp_weight_remat=True, pad_length=rows * size - 16)
    elif layout in ("tp_gtp", "gtp_flat"):
        # Outer rank coordinate selects TP rows (or GTP rows for gtp_flat).
        inner, outer = None, None
        for start in range(0, size, 2):
            group = torch.distributed.new_group([start, start + 1])
            if rank in (start, start + 1):
                inner = group
        for parity in range(2):
            group = torch.distributed.new_group(list(range(parity, size, 2)))
            if rank % 2 == parity:
                outer = group
        if layout == "tp_gtp":
            pg.tp, pg.gtp_remat = outer, inner
            tp_rows = 16 // outer.size()
            rows = tp_rows // 2 + 1
            physical = torch.full((rows * 2, 8), float("nan"), device=device)
            physical[:tp_rows] = padded.chunk(outer.size())[outer.rank()]
            local = physical.chunk(2)[inner.rank()].contiguous()
            attributes = dict(
                tensor_model_parallel=True,
                partition_dim=0,
                partition_stride=1,
                is_gtp_weight_remat=True,
                pad_length=2,
            )
        else:
            pg.gtp_remat, pg.dp_cp = outer, inner
            rows = 16 // outer.size() + 1
            physical = torch.full((rows * outer.size(), 8), float("nan"), device=device)
            physical[:16] = padded
            local = physical.chunk(outer.size())[outer.rank()].contiguous()
            attributes = dict(is_gtp_weight_remat=True, pad_length=outer.size())
    parameter = torch.nn.Parameter(torch.zeros_like(local))
    for key, value in attributes.items():
        setattr(parameter, key, value)
    parameter.main_grad = local * 8
    if layout in ("flat", "gtp_flat"):
        if layout == "flat":
            pg.dp_cp = world
        group = pg.dp_cp
        # Uneven intervals intentionally split rows, including a tiny first shard.
        bounds = [0, 3] + [
            3 + (parameter.numel() - 3) * i // (group.size() - 1) for i in range(1, group.size())
        ]
        start, end = bounds[group.rank() : group.rank() + 2]
        optimizer = _distributed_optimizer(
            parameter, torch.nn.Parameter(parameter.flatten()[start:end]), start, end
        )
    elif layout == "owner":
        pg.dp_cp = world
        optimizer = object.__new__(LayerWiseDistributedOptimizer)
        optimizer.chained_optimizers = [_fp32_optimizer(*([parameter] if rank == 0 else []))]
        optimizer.dp_cp_params_list = [[parameter]] + [[] for _ in range(size - 1)]
        optimizer.expt_dp_params_list = None
    else:
        optimizer = _fp32_optimizer(parameter)
        if layout == "replica":
            pg.dp_cp = world
    _check_observer(monkeypatch, parameter, optimizer, pg, reference, 8, has_tokenizer)
