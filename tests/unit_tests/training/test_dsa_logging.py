# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Indexer logging includes empty pipeline stages and reports raw KL consistently."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """These logging tests use synthetic tensors and need no downloaded datasets."""


@contextmanager
def _explicit_indexer_groups(cp_size, gtp_remat=False, pp_size=2):
    """Build real groups while leaving Megatron's global groups unset."""
    Utils.initialize_distributed()
    parallel_state.destroy_model_parallel()
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    if world < 4 or world % (pp_size * cp_size):
        pytest.skip("Requires at least four ranks, divisible by PP times CP.")
    stage_size = world // pp_size
    dp_size = stage_size // cp_size
    groups = ProcessGroupCollection()
    created = []

    def create(name, members):
        group = torch.distributed.new_group(members, backend="nccl")
        if rank in members:
            setattr(groups, name, group)
            created.append(group)

    try:
        for offset in range(stage_size):
            create("pp", list(range(offset, world, stage_size)))
        for start in range(0, world, cp_size):
            create("cp", list(range(start, start + cp_size)))
        for pp_rank in range(pp_size):
            for cp_rank in range(cp_size):
                start = pp_rank * stage_size + cp_rank
                create("dp", list(range(start, start + stage_size, cp_size)))
        if gtp_remat:
            # Ordinary DP excludes the remat peers; loss averaging must include them.
            groups.dp_gtp_remat = groups.dp
            for member in range(world):
                create("dp", [member])
        groups.mp = groups.pp
        assert not parallel_state.model_parallel_is_initialized()
        assert not parallel_state.is_initialized()
        yield groups, dp_size
        assert parallel_state.get_pipeline_model_parallel_group(False) is None
        assert parallel_state.get_context_parallel_group(False) is None
        assert not parallel_state.is_initialized()
    finally:
        DSAIndexerLossLoggingHelper.tracker.clear()
        torch.distributed.barrier()
        for group in reversed(created):
            torch.distributed.destroy_process_group(group)


def _seed_explicit_indexer_loss(groups, multiplier):
    # Only the last stage has an indexer; earlier stages still join all collectives.
    if groups.pp.rank() == groups.pp.size() - 1:
        dp_group = groups.dp_gtp_remat or groups.dp
        partial_loss = multiplier * (groups.cp.rank() + 1) * (dp_group.rank() + 1)
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            torch.tensor(float(partial_loss), device="cuda"),
            layer_number=2,
            num_layers=2,
            reduce_group=groups.cp if groups.cp.size() > 1 else None,
        )


@pytest.mark.parametrize(
    "cp_size,gtp_remat,pp_size", [(1, False, 2), (2, False, 2), (1, True, 2), (2, True, 1)]
)
def test_indexer_logging_without_global_groups(cp_size, gtp_remat, pp_size):
    helper = DSAIndexerLossLoggingHelper
    helper.tracker.clear()
    with _explicit_indexer_groups(cp_size, gtp_remat, pp_size) as (groups, dp_size):
        if gtp_remat:
            assert groups.dp.size() == 1
            assert groups.dp_gtp_remat.size() > 1
            # CP=2 and PP=1 leave two distinct-data GTP-remat peers on four GPUs.
            # The CP and GTP-remat groups intersect only at the current rank.
            cp_ranks = set(torch.distributed.get_process_group_ranks(groups.cp))
            remat_ranks = set(torch.distributed.get_process_group_ranks(groups.dp_gtp_remat))
            assert cp_ranks & remat_ranks == {torch.distributed.get_rank()}
        for iteration in (1, 2):
            _seed_explicit_indexer_loss(groups, iteration)
            total = {}
            helper.track_indexer_metrics(
                loss_scale=0.5,
                iteration=iteration,
                writer=None,
                total_loss_dict=total,
                num_layers=2,
                num_indexer_layers=1,
                loss_coeff=0.1,
                cp_group=groups.cp if cp_size > 1 else None,
                pg_collection=groups,
            )
            expected = torch.tensor(
                iteration * 0.5 * cp_size * (cp_size + 1) / 2 * (dp_size + 1) / 2, device="cuda"
            )
            torch.testing.assert_close(total["indexer loss"], expected)
            torch.testing.assert_close(total["indexer raw loss"], expected / 0.1)
            assert torch.count_nonzero(helper.tracker["values"]) == 0


@pytest.mark.parametrize("missing_group", ["pp", "dp"])
def test_indexer_logging_rejects_missing_explicit_groups(missing_group):
    with _explicit_indexer_groups(1) as (groups, _):
        _seed_explicit_indexer_loss(groups, 1)
        setattr(groups, missing_group, None)
        with pytest.raises(ValueError, match="PP and DP process groups"):
            DSAIndexerLossLoggingHelper.track_indexer_metrics(
                loss_scale=1.0,
                iteration=1,
                writer=None,
                num_layers=2,
                num_indexer_layers=1,
                pg_collection=groups,
            )


def _training_log_args(monkeypatch, cp_size, dp_size):
    from megatron.training import training

    args = SimpleNamespace(
        log_interval=100,
        timing_log_level=0,
        perform_rl_step=False,
        micro_batch_size=1,
        data_parallel_size=dp_size,
        gtp_weight_remat_size=1,
        world_size=torch.distributed.get_world_size(),
        seq_length=8,
        freeze_all_layers=False,
        num_experts=None,
        mtp_num_layers=None,
        experimental_attention_variant="ssa",
        dsa_indexer_loss_coeff=0.1,
        hybrid_layer_pattern="M*",
        num_layers=2,
        context_parallel_size=cp_size,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    for name in (
        "get_timers",
        "get_tensorboard_writer",
        "get_wandb_writer",
        "get_one_logger",
        "get_energy_monitor",
    ):
        monkeypatch.setattr(training, name, lambda: None)
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(training.one_logger_utils, "get_one_logger", lambda: None)
    return training


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("log_indexer_loss", [False, True])
def test_training_log_without_global_groups(monkeypatch, cp_size, log_indexer_loss):
    helper = DSAIndexerLossLoggingHelper
    helper.tracker.clear()
    with _explicit_indexer_groups(cp_size) as (groups, dp_size):
        training = _training_log_args(monkeypatch, cp_size, dp_size)
        if log_indexer_loss:
            _seed_explicit_indexer_loss(groups, 1)
            local_groups = groups
        else:
            # An encoder-only rank has no language-model CP/PP/DP collection.
            local_groups = ProcessGroupCollection()
            local_groups.mp = groups.mp
        total = {}
        result = training.training_log(
            loss_dict={},
            total_loss_dict=total,
            learning_rate=0.001,
            iteration=1,
            loss_scale=1.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            max_attention_logit=None,
            pg_collection=local_groups,
            log_indexer_loss=log_indexer_loss,
        )
        assert result is False
        if log_indexer_loss:
            expected = torch.tensor(cp_size * (cp_size + 1) / 2 * (dp_size + 1) / 2, device="cuda")
            torch.testing.assert_close(total["indexer loss"], expected)
            torch.testing.assert_close(total["indexer raw loss"], expected / 0.1)
        else:
            assert "indexer loss" not in total
            assert not helper.tracker


@pytest.mark.parametrize("context_parallel_size", [1, 2])
def test_empty_pipeline_stage_participates_in_indexer_loss_logging(context_parallel_size):
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=2, context_parallel_size=context_parallel_size
    )
    helper = DSAIndexerLossLoggingHelper
    helper.tracker.clear()
    try:
        cp_group = parallel_state.get_context_parallel_group()
        owns_indexer = parallel_state.get_pipeline_model_parallel_rank() == 1
        # Each CP rank contributes a different, globally normalized partial loss.
        partial_loss = float(parallel_state.get_context_parallel_rank() + 1)
        for _ in range(2):
            if owns_indexer:
                helper.save_loss_to_tracker(
                    torch.tensor(partial_loss, device="cuda"),
                    2,
                    2,
                    reduce_group=cp_group if context_parallel_size > 1 else None,
                )
            total = {}
            helper.track_indexer_metrics(
                1.0,
                1,
                None,
                total_loss_dict=total,
                num_indexer_layers=1,
                num_layers=2,
                loss_coeff=0.1,
                cp_group=cp_group if context_parallel_size > 1 else None,
            )
            expected = torch.tensor(
                context_parallel_size * (context_parallel_size + 1) / 2, device="cuda"
            )
            torch.testing.assert_close(total["indexer loss"], expected)
            torch.testing.assert_close(total["indexer raw loss"], expected / 0.1)
    finally:
        helper.tracker.clear()
        Utils.destroy_model_parallel()
