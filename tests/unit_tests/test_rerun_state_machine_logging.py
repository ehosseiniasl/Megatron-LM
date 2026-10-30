# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Logging and exit behavior of RerunStateMachine.validate_result."""

import logging

import pytest
import torch

from megatron.core import rerun_state_machine as rsm


def _validate(machine, fatal=False, **kwargs):
    machine.validate_result(
        result=torch.tensor(2.0),
        rejection_func=lambda value: bool(value > 1.0),
        message="global grad norm exceeded clip threshold 1.0",
        fatal=fatal,
        allow_first_iteration=not fatal,
        **kwargs,
    )


@pytest.fixture
def machine_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    tracker = tmp_path / "rejected.txt"

    def make(mode):
        machine = rsm.RerunStateMachine(mode=mode, result_rejected_tracker_filename=str(tracker))
        machine.state = rsm.RerunState.INITIAL_RUN
        return machine

    return make, tracker


@pytest.mark.parametrize("mode", [rsm.RerunMode.DISABLED, rsm.RerunMode.VALIDATE_RESULTS])
@pytest.mark.parametrize("last_rank", [False, True])
def test_only_the_last_rank_logs(machine_factory, monkeypatch, caplog, mode, last_rank):
    make, tracker = machine_factory
    monkeypatch.setattr(rsm, "_is_last_rank", lambda: last_rank)
    machine = make(mode)
    with caplog.at_level(logging.WARNING, logger=rsm.logger.name):
        _validate(machine, log_on_all_ranks=False)
    logged = [r for r in caplog.records if "exceeded clip threshold" in r.getMessage()]
    assert bool(logged) == last_rank
    # The constructor creates the tracker file empty; only a logging rank writes to it.
    assert bool(tracker.read_text()) == last_rank
    if mode == rsm.RerunMode.VALIDATE_RESULTS:
        # The validation itself still runs on every rank: each one requests the rerun.
        assert machine.rerun_requested


def test_default_logs_on_every_rank(machine_factory, monkeypatch, caplog):
    make, tracker = machine_factory
    monkeypatch.setattr(rsm, "_is_last_rank", lambda: False)
    with caplog.at_level(logging.WARNING, logger=rsm.logger.name):
        _validate(make(rsm.RerunMode.VALIDATE_RESULTS))
    assert any("exceeded clip threshold" in r.getMessage() for r in caplog.records)
    assert tracker.read_text()


def test_nonfatal_reproducible_rerun_does_not_ask_for_another_gpu(
    machine_factory, monkeypatch, caplog
):
    make, _ = machine_factory
    machine = make(rsm.RerunMode.VALIDATE_RESULTS)
    _validate(machine)
    machine.state = rsm.RerunState.RERUNNING_IN_PLACE
    machine.validation_counts.clear()
    with caplog.at_level(logging.WARNING, logger=rsm.logger.name):
        _validate(machine)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "reproducible within the tolerance" in messages
    assert "Continuing, since this validation is not fatal" in messages
    assert "different GPU" not in messages
    assert machine.continue_requested


def test_fatal_reproducible_rerun_exits_without_checkpoint(machine_factory, monkeypatch, caplog):
    make, _ = machine_factory
    machine = make(rsm.RerunMode.VALIDATE_RESULTS)
    machine.first_iteration_complete = True
    monkeypatch.setattr(machine, "_reduce_any", lambda value: tuple(value))
    monkeypatch.setattr(machine, "_restore_state", lambda: None)
    _validate(machine, fatal=True)
    machine.state = rsm.RerunState.RERUNNING_IN_PLACE
    machine.validation_counts.clear()
    with caplog.at_level(logging.WARNING, logger=rsm.logger.name):
        _validate(machine, fatal=True)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "Exiting, since this validation is fatal" in messages
    assert not machine.checkpoint_requested and not machine.continue_requested
    assert not machine.should_run_forward_backward(None)
    assert machine.should_checkpoint_and_exit() == (
        False,
        True,
        rsm.EXIT_CODE_FAILED_ON_RESULT_VALIDATION,
    )


@pytest.mark.parametrize("mode", [rsm.RerunMode.DISABLED, rsm.RerunMode.VALIDATE_RESULTS])
def test_reported_iteration_matches_one_indexed_training_iteration(
    machine_factory, monkeypatch, caplog, mode
):
    make, tracker = machine_factory
    machine = make(mode)
    machine.state = rsm.RerunState.NOT_RUNNING_YET
    monkeypatch.setattr(machine, "_save_state", lambda: None)
    assert machine.current_iteration == 0
    # Starting the first training iteration (iteration 1 in the training loop) advances it to 1.
    assert machine.should_run_forward_backward(None)
    assert machine.current_iteration == 1
    with caplog.at_level(logging.WARNING, logger=rsm.logger.name):
        _validate(machine)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "iteration 1:" in messages or "iteration #1 " in messages
    # The tracker file feeds --iterations-to-skip, which the training loop compares against
    # its 1-indexed iteration.
    assert "iteration=1 " in tracker.read_text()


def test_rerun_from_checkpoint_advances_to_failed_iteration(machine_factory):
    make, _ = machine_factory
    machine = make(rsm.RerunMode.VALIDATE_RESULTS)
    # A checkpoint saved for a rerun holds the iterations completed before the failed one, and
    # train() sets current_iteration to that count after loading it.
    machine.state = rsm.RerunState.WILL_RERUN_FROM_CHECKPOINT
    machine.current_iteration = 41
    assert machine.should_run_forward_backward(None)
    assert machine.state == rsm.RerunState.RERUNNING_FROM_CHECKPOINT
    assert machine.current_iteration == 42
