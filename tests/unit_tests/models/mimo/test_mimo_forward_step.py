# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for MIMO forward-step helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from examples.mimo.training.batch import move_batch_to_cuda
from examples.mimo.training.step import loss_func
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_observation import capture_tensor_observations
from megatron.training.tensor_metrics.data_source_loss import capture_data_source_metrics


def test_loss_func_returns_int_num_tokens_three_tuple():
    output = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

    loss_sum, num_tokens, loss_dict = loss_func(output, loss_mask=loss_mask)

    assert isinstance(num_tokens, torch.Tensor)
    assert not num_tokens.is_floating_point()
    assert num_tokens.dtype in (torch.int32, torch.int64, torch.int16)
    assert int(num_tokens.item()) == 3

    assert isinstance(loss_sum, torch.Tensor)
    assert loss_sum.shape == torch.Size([])
    assert torch.allclose(loss_sum, torch.tensor(1.0 + 2.0 + 4.0))

    assert set(loss_dict.keys()) == {"lm loss"}
    logged = loss_dict["lm loss"]
    assert logged.shape == torch.Size([2])
    assert torch.allclose(logged[0], loss_sum.detach())
    assert torch.allclose(logged[1], num_tokens.detach().float())


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("validation", (False, True))
def test_training_source_loss_preserves_gradients_and_validation_reporting(
    monkeypatch, enabled, validation
):
    from examples.mimo.training import step
    from mistral_adapter.per_source_validation import (
        PER_SOURCE_LOSS_KEY,
        PER_SOURCE_NUM_SOURCES_KEY,
    )

    monkeypatch.setattr(step, "move_batch_to_cuda", lambda batch: batch)
    observed = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, input_ids, labels, loss_mask):
            return labels.float() * self.weight, loss_mask

    model = Model()
    batch = {
        "input_ids": torch.tensor([[1, 3, 99]]),
        "labels": torch.tensor([[2, 4, 7]]),
        "loss_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "data_source_ids": torch.tensor([[0, 1, 1]]),
    }
    if validation:
        batch["source_ids"] = torch.tensor([[1, 0, 0]])
        batch[PER_SOURCE_NUM_SOURCES_KEY] = 2
    with (
        capture_tensor_observations(
            lambda *args: observed.append(args[3]), {"data_source_loss"} if enabled else set()
        ),
        capture_data_source_metrics(("a", "b") if enabled else (), SimpleNamespace()),
    ):
        output, closure = step.mimo_forward_step(iter([batch]), model)
        loss, count, report = closure(output)
        loss.backward()
    assert loss == 12 and count == 2 and model.weight.grad == 6
    assert set(report) == ({"lm loss", PER_SOURCE_LOSS_KEY} if validation else {"lm loss"})
    if validation:
        torch.testing.assert_close(
            report[PER_SOURCE_LOSS_KEY], torch.tensor([[8, 1, 2], [4, 1, 1]], dtype=torch.float64)
        )
    if enabled:
        torch.testing.assert_close(observed[0], torch.tensor([[4, 1], [8, 1]], dtype=torch.float64))
        assert not observed[0].requires_grad
    else:
        assert observed == []


def test_training_source_loss_requires_provenance():
    with capture_tensor_observations(lambda *args: None, {"data_source_loss"}):
        with pytest.raises(ValueError, match="token-aligned source IDs"):
            loss_func(torch.ones(1, 2), loss_mask=torch.ones(1, 2), model=torch.nn.Module())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_move_batch_to_cuda_recurses_dict_list_tuple():
    t_top = torch.tensor([1.0])
    t_in_list = torch.tensor([2.0])
    t_in_tuple = torch.tensor([3.0])
    t_nested = torch.tensor([4.0])

    batch = {
        "input_ids": t_top,
        "a_list": [t_in_list, "not a tensor", 7],
        "a_tuple": (t_in_tuple,),
        "nested": {"deep": t_nested},
        "scalar": 5,
    }

    out = move_batch_to_cuda(batch)

    assert isinstance(out, dict)
    assert isinstance(out["a_list"], list)
    assert isinstance(out["a_tuple"], tuple)
    assert out["scalar"] == 5
    assert out["a_list"][1] == "not a tensor"
    assert out["input_ids"].is_cuda
    assert out["a_list"][0].is_cuda
    assert out["a_tuple"][0].is_cuda
    assert out["nested"]["deep"].is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_move_batch_to_cuda_handles_packed_seq_params():
    cu_q = torch.tensor([0, 4, 8], dtype=torch.int32)
    cu_kv = torch.tensor([0, 4, 8], dtype=torch.int32)
    psp = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv, max_seqlen_q=8, max_seqlen_kv=8
    )

    batch = {"packing": psp}
    out = move_batch_to_cuda(batch)

    assert out["packing"] is psp
    assert psp.qkv_format == "thd"
    assert psp.max_seqlen_q == 8
    assert psp.cu_seqlens_q.is_cuda
    assert psp.cu_seqlens_kv.is_cuda
