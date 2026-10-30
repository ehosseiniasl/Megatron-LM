# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed dense-to-indexer initialization retains strict backbone loading."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
from megatron.training import checkpointing
from megatron.training.arguments import parse_args
from megatron.training.simplified_sparse_attention import (
    freeze_backbone_for_indexer_training,
    retain_initialized_indexer,
)
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _sharded(key, value):
    return ShardedTensor.from_rank_offsets(key, value, replica_id=Utils.rank)


class _Model(torch.nn.Module):
    def __init__(self, with_indexer=True):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2, bias=False)
        if with_indexer:
            self.indexer = torch.nn.Linear(2, 2, bias=False)


@pytest.mark.parametrize("legacy_sparse_metadata", [False, True])
def test_training_loader_dense_initialization_resume_and_phase_transition(
    model_parallel, tmp_path_dist_ckpt, monkeypatch, legacy_sparse_metadata
):
    """Exercise the real training loader, including file IO and strict module loading."""
    with TempNamedDir(tmp_path_dist_ckpt / "ssa_training_loader") as directory:
        args = parse_args(ignore_unknown_args=True)
        init_basic_mock_args(args, 1, 1, bf16=False)
        init_checkpointing_mock_args(args, str(directory))
        args.ckpt_format = "torch"
        args.use_dist_ckpt = False
        args.use_distributed_optimizer = False
        args.no_load_optim = args.no_load_rng = True
        args.experimental_attention_variant = "ssa"
        args.ssa_train_indexer_only = True
        args.dsa_indexer_topk = 512
        monkeypatch.setattr(checkpointing, "get_args", lambda: args)
        monkeypatch.setattr(checkpointing, "update_num_microbatches", lambda **_: None)
        monkeypatch.setattr(checkpointing, "has_nvidia_modelopt", False)
        monkeypatch.setattr(checkpointing, "_CHECKPOINT_VERSION", None)
        dense = _Model(False)
        with torch.no_grad():
            dense.backbone.weight.fill_(2.0)
        dense_args = deepcopy(args)
        dense_args.experimental_attention_variant = None
        dense_args.consumed_train_samples = 148

        def write_checkpoint(saved_args, model_state, optimizer_state=None):
            path = checkpointing.get_checkpoint_name(str(directory), 37)
            if Utils.rank == 0:
                from pathlib import Path

                Path(path).parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "args": saved_args,
                    "checkpoint_version": 3.1,
                    "iteration": 37,
                    "model": model_state,
                }
                if optimizer_state is not None:
                    state["optimizer"] = optimizer_state
                torch.save(state, path)
                (directory / "latest_checkpointed_iteration.txt").write_text("37")
            torch.distributed.barrier()

        write_checkpoint(dense_args, dense.state_dict())
        sparse = _Model()
        initial_indexer = sparse.indexer.weight.detach().clone()
        iteration, _ = checkpointing.load_checkpoint([sparse], None, None)
        assert iteration == 37 and args.consumed_train_samples == 148
        torch.testing.assert_close(sparse.backbone.weight, dense.backbone.weight)
        torch.testing.assert_close(sparse.indexer.weight, initial_indexer)

        # A missing backbone key must fail even though new indexer keys are allowed.
        write_checkpoint(dense_args, {})
        with pytest.raises(RuntimeError, match="backbone.weight"):
            checkpointing.load_checkpoint([sparse], None, None)

        with torch.no_grad():
            sparse.indexer.weight.fill_(3.0)
        source_optimizer = torch.optim.AdamW([sparse.indexer.weight], lr=0.1)
        sparse.indexer(torch.ones(2)).sum().backward()
        source_optimizer.step()
        saved_sparse_args = deepcopy(args)
        if legacy_sparse_metadata:
            saved_sparse_args.experimental_attention_variant = "dsa"
            saved_sparse_args.dsa_gqa_backend = "cute"
            saved_sparse_args.dsa_train_indexer_only = True
            del saved_sparse_args.ssa_train_indexer_only
        write_checkpoint(saved_sparse_args, sparse.state_dict(), source_optimizer.state_dict())
        resumed = _Model()
        resumed_optimizer = torch.optim.AdamW([resumed.indexer.weight], lr=0.1)
        resumed_optimizer.is_stub_optimizer = False
        args.no_load_optim = False
        checkpointing.load_checkpoint([resumed], resumed_optimizer, None)
        torch.testing.assert_close(resumed.indexer.weight, sparse.indexer.weight)
        torch.testing.assert_close(
            resumed_optimizer.state[resumed.indexer.weight]["exp_avg"],
            source_optimizer.state[sparse.indexer.weight]["exp_avg"],
        )
        assert resumed_optimizer.state[resumed.indexer.weight]["step"].item() == 1

        args.ssa_train_indexer_only = False
        with pytest.raises(AssertionError, match="training phase changed"):
            checkpointing.load_checkpoint([resumed], resumed_optimizer, None)
        args.no_load_optim = True
        phase2 = _Model()
        checkpointing.load_checkpoint([phase2], None, None)
        for key, value in sparse.state_dict().items():
            torch.testing.assert_close(phase2.state_dict()[key], value)


def test_phase1_pipeline_stage_without_indexer_participates_in_loss_logging():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
    helper = DSAIndexerLossLoggingHelper
    helper.tracker.clear()
    try:
        owns_indexer = parallel_state.get_pipeline_model_parallel_rank() == 1
        model = _Model(with_indexer=owns_indexer)
        freeze_backbone_for_indexer_training(
            [model],
            SimpleNamespace(experimental_attention_variant="ssa", ssa_train_indexer_only=True),
        )
        if owns_indexer:
            helper.save_loss_to_tracker(torch.tensor(2.0, device="cuda"), 2, 2)
        total = {}
        helper.track_indexer_metrics(
            1.0, 1, None, total_loss_dict=total, num_layers=2, num_indexer_layers=1, loss_coeff=0.1
        )
        torch.testing.assert_close(total["indexer loss"], torch.tensor(2.0, device="cuda"))
        torch.testing.assert_close(total["indexer raw loss"], torch.tensor(20.0, device="cuda"))
        assert not model.backbone.weight.requires_grad
    finally:
        helper.tracker.clear()
        Utils.destroy_model_parallel()


def test_dense_initialization_and_sparse_resume(model_parallel, tmp_path_dist_ckpt):
    with TempNamedDir(tmp_path_dist_ckpt / "dense_to_indexer") as directory:
        dense_path = directory / "dense"
        phase1_path = directory / "phase1"
        dense_path.mkdir(exist_ok=True)
        phase1_path.mkdir(exist_ok=True)
        source = torch.arange(4, dtype=torch.float32).view(2, 2)
        dist_checkpointing.save(
            {"model": {"backbone.weight": _sharded("backbone.weight", source)}}, dense_path
        )
        initialized = torch.full((2, 2), 3.0)
        template = {
            "backbone.weight": _sharded("backbone.weight", torch.empty_like(source)),
            "core_attention.indexer.linear_q.weight": _sharded(
                "core_attention.indexer.linear_q.weight", initialized.clone()
            ),
        }
        retain_initialized_indexer(template)
        phase1 = dist_checkpointing.load({"model": template}, dense_path, strict="raise_all")
        torch.testing.assert_close(phase1["model"]["backbone.weight"], source)
        torch.testing.assert_close(
            phase1["model"]["core_attention.indexer.linear_q.weight"], initialized
        )

        # A subsequent sparse resume requests and restores the saved indexer;
        # it must not reuse the fresh initialization used by dense conversion.
        trained_indexer = initialized + 1
        dist_checkpointing.save(
            {
                "model": {
                    "backbone.weight": _sharded("backbone.weight", source),
                    "core_attention.indexer.linear_q.weight": _sharded(
                        "core_attention.indexer.linear_q.weight", trained_indexer
                    ),
                }
            },
            phase1_path,
        )
        resume = {
            "backbone.weight": _sharded("backbone.weight", torch.empty_like(source)),
            "core_attention.indexer.linear_q.weight": _sharded(
                "core_attention.indexer.linear_q.weight", torch.empty_like(initialized)
            ),
        }
        loaded = dist_checkpointing.load({"model": resume}, phase1_path, strict="raise_all")
        torch.testing.assert_close(
            loaded["model"]["core_attention.indexer.linear_q.weight"], trained_indexer
        )


def _shard_checkpoint_state(state):
    return {
        key: (
            ShardedTensor.from_rank_offsets(key, value, replica_id=Utils.rank)
            if isinstance(value, torch.Tensor)
            else ShardedObject(key, value, (1,), (0,), replica_id=Utils.rank)
        )
        for key, value in state.items()
    }


class _IndexerWithExtraState(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2, bias=False)
        self.extra_state = {"initialized": True}

    def get_extra_state(self):
        return self.extra_state

    def set_extra_state(self, state):
        self.extra_state = state


class _NestedWrapperModel(torch.nn.Module):
    """Reproduce MIMO's nested wrapper save/load prefix asymmetry."""

    def __init__(self, with_indexer=True):
        super().__init__()
        language_model = _Model(with_indexer)
        language_model.mtp = _Model(with_indexer)
        if with_indexer:
            for model in (language_model, language_model.mtp):
                model.indexer = _IndexerWithExtraState()
        # Use the real DDP state_dict forwarding implementation. Outer Module
        # load_state_dict traverses registered children and needs both prefixes.
        self.language_model = _BaseDataParallel(None, _BaseDataParallel(None, language_model))

    @property
    def inner(self):
        return self.language_model.module.module

    def checkpoint_state(self):
        return {
            "language_model.module.module." + key: value
            for key, value in self.inner.state_dict().items()
        }

    def sharded_state_dict(self, **kwargs):
        return _shard_checkpoint_state(self.checkpoint_state())


@pytest.fixture
def native_loader(tmp_path_dist_ckpt, monkeypatch, request):
    Utils.initialize_model_parallel()
    try:
        with TempNamedDir(tmp_path_dist_ckpt / "ssa_nested_training_loader") as directory:
            args = parse_args(ignore_unknown_args=True)
            init_basic_mock_args(args, 1, 1, bf16=False)
            init_checkpointing_mock_args(args, str(directory))
            args.ckpt_format = request.param
            args.use_dist_ckpt = request.param == "torch_dist"
            args.dist_ckpt_strictness = "raise_all"
            args.use_distributed_optimizer = False
            args.no_load_optim = args.no_load_rng = True
            args.experimental_attention_variant = "ssa"
            args.ssa_train_indexer_only = True
            args.dsa_indexer_topk = 512
            monkeypatch.setattr(checkpointing, "get_args", lambda: args)
            monkeypatch.setattr(checkpointing, "update_num_microbatches", lambda **_: None)
            monkeypatch.setattr(checkpointing, "has_nvidia_modelopt", False)
            monkeypatch.setattr(checkpointing, "_CHECKPOINT_VERSION", None)

            def write_checkpoint(saved_args, model_state, iteration, optimizer_state=None):
                path = Path(
                    checkpointing.get_checkpoint_name(
                        str(directory), iteration, return_base_dir=args.use_dist_ckpt
                    )
                )
                state = {
                    "args": saved_args,
                    "checkpoint_version": 3.1,
                    "iteration": iteration,
                    "model": model_state,
                }
                if optimizer_state is not None:
                    state["optimizer"] = optimizer_state
                if args.use_dist_ckpt:
                    if Utils.rank == 0:
                        path.mkdir(parents=True)
                    torch.distributed.barrier()
                    state["model"] = _shard_checkpoint_state(model_state)
                    dist_checkpointing.save(state, path)
                elif Utils.rank == 0:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(state, path)
                if Utils.rank == 0:
                    (directory / "latest_checkpointed_iteration.txt").write_text(str(iteration))
                torch.distributed.barrier()

            yield args, write_checkpoint
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("native_loader", ["torch_dist"], indirect=True)
def test_dense_initialization_preserves_nested_wrapper_indexer_keys(native_loader):
    args, write_checkpoint = native_loader
    dense = _NestedWrapperModel(False)
    with torch.no_grad():
        dense.inner.backbone.weight.fill_(2.0)
    dense_args = deepcopy(args)
    dense_args.experimental_attention_variant = None
    write_checkpoint(dense_args, dense.checkpoint_state(), 37)
    args.finetune = True
    sparse = _NestedWrapperModel()
    initialized_indexers = [
        model.indexer.weight.detach().clone() for model in (sparse.inner, sparse.inner.mtp)
    ]
    unwrapped_key = "language_model.indexer.weight"
    wrapped_key = "language_model.module.module.indexer.weight"
    assert unwrapped_key in sparse.state_dict()
    assert wrapped_key not in sparse.state_dict()
    assert wrapped_key in sparse.sharded_state_dict()
    assert "language_model.module.module.mtp.indexer.weight" in sparse.sharded_state_dict()

    iteration, _ = checkpointing.load_checkpoint([sparse], None, None, strict=False)

    assert iteration == 0
    torch.testing.assert_close(sparse.inner.backbone.weight, dense.inner.backbone.weight)
    for model, initialized_indexer in zip((sparse.inner, sparse.inner.mtp), initialized_indexers):
        torch.testing.assert_close(model.indexer.weight, initialized_indexer)
        assert model.indexer.extra_state == {"initialized": True}

    # The wrapper fix must preserve strict backbone validation during conversion.
    incomplete = dense.checkpoint_state()
    del incomplete["language_model.module.module.backbone.weight"]
    write_checkpoint(dense_args, incomplete, 38)
    with pytest.raises((RuntimeError, CheckpointingException), match="backbone.weight"):
        checkpointing.load_checkpoint([_NestedWrapperModel()], None, None, strict=False)

    # Sparse loading must restore saved indexer weights and extra state, not retain init.
    with torch.no_grad():
        for model in (sparse.inner, sparse.inner.mtp):
            model.indexer.weight.fill_(3.0)
            model.indexer.extra_state = {"trained": True}
    write_checkpoint(deepcopy(args), sparse.checkpoint_state(), 39)
    restored = _NestedWrapperModel()
    checkpointing.load_checkpoint([restored], None, None)
    for model, original in zip(
        (restored.inner, restored.inner.mtp), (sparse.inner, sparse.inner.mtp)
    ):
        torch.testing.assert_close(model.indexer.weight, original.indexer.weight)
        assert model.indexer.extra_state == {"trained": True}

    incomplete = sparse.checkpoint_state()
    del incomplete[wrapped_key]
    write_checkpoint(deepcopy(args), incomplete, 40)
    args.dist_ckpt_strictness = "log_unexpected"
    with pytest.raises((RuntimeError, CheckpointingException), match="indexer.weight"):
        checkpointing.load_checkpoint([_NestedWrapperModel()], None, None, strict=False)
