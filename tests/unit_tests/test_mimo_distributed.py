# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from datetime import timedelta
from unittest.mock import MagicMock

import torch.distributed as dist

from examples.mimo.training.distributed import initialize_distributed
from megatron.core import parallel_state


def test_initialize_distributed_is_lazy_and_honors_timeout(monkeypatch):
    timeout = timedelta(minutes=30)
    init_process_group = MagicMock()

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr("torch.cuda.set_device", MagicMock())
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    monkeypatch.setattr(dist, "barrier", MagicMock())
    monkeypatch.setattr(
        "examples.mimo.training.distributed.assert_parallel_state_uninitialized", MagicMock()
    )
    monkeypatch.setattr(parallel_state, "get_global_memory_buffer", lambda: object())

    initialize_distributed(timeout=timeout)

    init_process_group.assert_called_once_with(backend="nccl", timeout=timeout)
    assert "device_id" not in init_process_group.call_args.kwargs
