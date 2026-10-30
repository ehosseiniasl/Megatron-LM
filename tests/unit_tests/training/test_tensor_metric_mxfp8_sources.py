# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise metric placeholders with real TE MXFP8 model parameters."""

import os

import pytest
import torch

from megatron.core.fp8_utils import copy_tensor_to_quantized_param, is_mxfp8tensor
from megatron.core.optimizer import LayerWiseDistributedOptimizer
from megatron.core.parameter_names import CanonicalParameterNameMap
from megatron.training.tensor_metrics.observer import build_tensor_metric_observer
from megatron.training.tensor_metrics.optimizer_sources import _optimizer_tensor_views
from tests.unit_tests.training.test_tensor_metric_sources import (
    _chained_optimizer,
    _disable_tensor_metric_profiling,
    _float16_optimizer,
    _mimo_optimizer,
    _pg_collection,
)


@pytest.mark.parametrize("mimo", (False, True))
def test_mxfp8_remote_optimizer_slots_and_metric_reductions(mimo):
    """Quantized model weights must yield ordinary neutral slots on non-owner ranks."""
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("Requires at least two ranks to exercise remote ownership.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell.")
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    world = torch.distributed.group.WORLD
    rank = torch.distributed.get_rank()
    size = torch.distributed.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    pg = _pg_collection()
    pg.dp_cp = world
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(size)])
    parameters = []
    expected_parameters = []
    expected_gradients = []
    for index, layer in enumerate(model.decoder.layers):
        values = torch.full((64, 64), float(index + 1), device=device)
        gradient = values.clone()
        gradient[::2] = 0
        quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        parameter = torch.nn.Parameter(
            quantizer.make_empty((64, 64), dtype=torch.bfloat16, device=device)
        )
        copy_tensor_to_quantized_param(parameter, values)
        assert is_mxfp8tensor(parameter)
        parameter.main_grad = gradient
        layer.weight = parameter
        parameters.append(parameter)
        expected_parameters.append(values)
        expected_gradients.append(gradient)

    child = object.__new__(LayerWiseDistributedOptimizer)
    child.chained_optimizers = [
        _float16_optimizer(parameters[rank], torch.nn.Parameter(expected_parameters[rank]))
    ]
    child.dp_cp_params_list = [[parameter] for parameter in parameters]
    child.expt_dp_params_list = None
    optimizer = _mimo_optimizer(_chained_optimizer(child), pg) if mimo else child
    views = _optimizer_tensor_views(CanonicalParameterNameMap(model), optimizer, pg)
    for view in views:
        assert type(view.parameter) is torch.Tensor
        assert type(view.wgrad) is torch.Tensor
        if view.model_parameter is parameters[rank]:
            assert view.parameter.dtype is torch.float32
            assert not view.is_placeholder
        else:
            assert view.parameter.numel() == view.wgrad.numel() == 0
            assert view.parameter.dtype is torch.bfloat16
            assert view.is_placeholder

    names = ("param-l2", "wgrad-l2", "wgrad-abs-max", "wgrad-zero-fraction")
    captured = {}
    observer = build_tensor_metric_observer(
        [f"{name}{suffix}:1" for name in names for suffix in ("", "-individual")],
        result_sink=lambda metric, results, iteration: captured.update(
            {metric.name: {result.label: result.tensor for result in results}}
        ),
    )
    observer(model=[model], optimizer=optimizer, iteration=0, pg_collection=pg)
    family = "family/decoder.layers.ALL.weight"
    for name in names:
        tensors = expected_parameters if name == "param-l2" else expected_gradients
        groups = {"global": list(range(size)), family: list(range(size))}
        groups.update({f"decoder.layers.{i}": [i] for i in range(size)})
        for suffix, selections in (
            ("", groups),
            ("-individual", {f"decoder.layers.{i}.weight": [i] for i in range(size)}),
        ):
            results = captured[name + suffix]
            assert results.keys() == selections.keys()
            for label, indices in selections.items():
                values = torch.cat([tensors[i].reshape(-1) for i in indices]).double()
                if name.endswith("l2"):
                    expected = values.norm()
                elif name.endswith("abs-max"):
                    expected = values.abs().max()
                else:
                    expected = (values == 0).double().mean()
                torch.testing.assert_close(results[label].double(), expected)
