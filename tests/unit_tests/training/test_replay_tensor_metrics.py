# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Clipping reruns must enable observations before replaying an unscheduled step."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.training import ft_integration, training


@pytest.fixture
def step_harness(monkeypatch):
    args = SimpleNamespace(
        rerun_large_grad_norm_steps=True,
        optimizer_cuda_graph=False,
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        vision_pretraining=False,
        empty_unused_memory_level=0,
        seq_length=4,
        micro_batch_size=1,
        decoder_seq_length=None,
        barrier_with_L1_time=False,
        qk_clip=False,
        log_max_attention_logit=False,
        log_num_zeros_in_grad=False,
        data_parallel_size=1,
        gtp_weight_remat_size=1,
    )
    groups = SimpleNamespace(mp=object(), pp=object(), dp_cp=object())
    model = SimpleNamespace(zero_grad_buffer=Mock(), pg_collection=groups)
    optimizer = SimpleNamespace(
        config=SimpleNamespace(clip_grad=1.0),
        zero_grad=Mock(),
        capture_grad_preparation_state=Mock(return_value=None),
        restore_grad_preparation_state=Mock(),
        discard_prepared_grads=Mock(),
        prepare_grads_for_step=Mock(return_value=(False, 2.0)),
        step_with_prepared_grads=Mock(return_value=(True, 2.0, 0)),
    )
    machine = SimpleNamespace(
        get_mode=lambda: training.RerunMode.VALIDATE_RESULTS,
        should_checkpoint_and_exit=lambda: (False, False, 0),
        attempt=0,
        requested=False,
    )

    def should_run(_):
        if machine.attempt == 0 or (machine.attempt == 1 and machine.requested):
            machine.attempt += 1
            return True
        return False

    def validate_result(*, result, rejection_func, **kwargs):
        if machine.attempt == 1:
            machine.requested = rejection_func(result)

    machine.should_run_forward_backward = should_run
    machine.validate_result = validate_result
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: Mock())
    monkeypatch.setattr(training, "get_rerun_state_machine", lambda: machine)
    monkeypatch.setattr(training, "get_telemetry", lambda: None)
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(training, "get_moe_router_tracer", lambda: None)
    monkeypatch.setattr(training, "get_moe_metrics_tracker", lambda: {})
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(training, "capture_global_batch_quantile_router_state", lambda _: [])
    monkeypatch.setattr(training, "restore_global_batch_quantile_router_state", Mock())
    monkeypatch.setattr(training.MTPLossLoggingHelper, "clean_metrics_in_tracker", Mock())
    monkeypatch.setattr(
        training, "_forward_backward_grad_context", lambda _: (nullcontext(), False)
    )
    monkeypatch.setattr(training, "is_pp_last_stage", lambda _: False)
    monkeypatch.setattr(training, "logical_and_across_model_parallel_group", lambda x, **_: x)
    monkeypatch.setattr(training, "reduce_max_stat_across_model_parallel_group", lambda x, **_: x)
    spans = SimpleNamespace(
        forward_backward=lambda _: nullcontext(), optimizer=lambda: nullcontext()
    )
    monkeypatch.setattr(training._otel, "TrainingStepSpans", lambda _: spans)
    ft_hook = Mock()
    monkeypatch.setattr(ft_integration, "on_tensor_metrics_start", ft_hook)
    observer = Mock()
    observer.observe_forward_backward.side_effect = lambda **_: nullcontext()
    forward_backward = Mock(return_value=[])

    def run(*, metric_observer=observer):
        machine.attempt = 0
        machine.requested = False
        return training.train_step(
            None,
            None,
            [model],
            optimizer,
            Mock(),
            None,
            forward_backward,
            iteration=6,
            tensor_metric_observer=metric_observer,
        )

    return SimpleNamespace(
        run=run,
        optimizer=optimizer,
        observer=observer,
        ft_hook=ft_hook,
        forward_backward=forward_backward,
        machine=machine,
    )


def test_clipping_replay_forces_both_observation_and_commit_only_for_that_step(step_harness):
    h = step_harness
    h.run()
    assert h.forward_backward.call_count == 2
    observations = h.observer.observe_forward_backward.call_args_list
    assert "force" not in observations[0].kwargs
    assert observations[1].kwargs["force"] is True
    assert h.observer.call_count == 1
    assert h.observer.call_args.kwargs["force"] is True
    assert h.observer.call_args.kwargs["iteration"] == 6
    h.ft_hook.assert_called_once_with()

    # Retry the same iteration normally: no persistent override may survive.
    h.optimizer.prepare_grads_for_step.return_value = (False, 0.5)
    h.observer.reset_mock()
    h.ft_hook.reset_mock()
    h.run()
    assert h.observer.observe_forward_backward.call_count == 1
    assert "force" not in h.observer.observe_forward_backward.call_args.kwargs
    assert "force" not in h.observer.call_args.kwargs
    h.ft_hook.assert_not_called()


@pytest.mark.parametrize("norm", (0.5, 1.0, float("inf"), float("nan")))
def test_steps_without_finite_clipping_do_not_force_metrics(step_harness, norm):
    h = step_harness
    h.optimizer.prepare_grads_for_step.return_value = (False, norm)
    h.run()
    assert h.forward_backward.call_count == 1
    assert "force" not in h.observer.observe_forward_backward.call_args.kwargs
    assert "force" not in h.observer.call_args.kwargs
    h.ft_hook.assert_not_called()


def test_unrelated_replay_does_not_force_metrics(step_harness):
    h = step_harness
    h.optimizer.prepare_grads_for_step.return_value = (False, 0.5)
    h.machine.validate_result = lambda **_: setattr(h.machine, "requested", True)
    h.run()
    assert h.forward_backward.call_count == 2
    assert all("force" not in c.kwargs for c in h.observer.observe_forward_backward.call_args_list)
    assert "force" not in h.observer.call_args.kwargs
    h.ft_hook.assert_not_called()


def test_existing_replay_with_infinite_norm_forces_metrics(step_harness):
    h = step_harness
    h.optimizer.prepare_grads_for_step.return_value = (False, float("inf"))
    # Another validation already requested replay; infinity alone does not trigger it.
    h.machine.validate_result = lambda **_: setattr(h.machine, "requested", True)
    h.run()
    assert h.forward_backward.call_count == 2
    observations = h.observer.observe_forward_backward.call_args_list
    assert "force" not in observations[0].kwargs
    assert observations[1].kwargs["force"] is True
    assert h.observer.call_count == 1
    assert h.observer.call_args.kwargs["force"] is True
    h.ft_hook.assert_called_once_with()


def test_clipping_replay_without_configured_metrics(step_harness):
    h = step_harness
    h.run(metric_observer=None)
    assert h.forward_backward.call_count == 2
    h.observer.assert_not_called()
    h.ft_hook.assert_not_called()


def test_replay_exit_does_not_emit_metrics(step_harness):
    h = step_harness
    h.machine.should_checkpoint_and_exit = lambda: (True, True, 17)
    result = h.run()
    assert h.forward_backward.call_count == 2
    assert h.observer.observe_forward_backward.call_args.kwargs["force"] is True
    h.observer.assert_not_called()
    assert result[2:5] == (True, True, 17)


@pytest.mark.parametrize("initial_section", ("step", "tensor_metrics", None))
def test_replay_switches_ft_section_without_advancing_step_count(monkeypatch, initial_section):
    client = Mock()
    monkeypatch.setattr(ft_integration, "_GLOBAL_RANK_MONITOR_CLIENT", client)
    monkeypatch.setattr(ft_integration, "_is_step_section_open", initial_section is not None)
    monkeypatch.setattr(ft_integration, "_step_section_name", initial_section or "step")
    monkeypatch.setattr(ft_integration, "_seen_tr_iters_cnt", 10)
    monkeypatch.setattr(ft_integration, "_NUM_WARMUP_ITERS", 5)

    ft_integration.on_tensor_metrics_start()
    if initial_section == "step":
        client.end_section.assert_called_once_with("step")
        client.start_section.assert_called_once_with("tensor_metrics")
    else:
        client.end_section.assert_not_called()
        client.start_section.assert_not_called()
    assert ft_integration._seen_tr_iters_cnt == 10

    if initial_section is not None:
        client.reset_mock()
        ft_integration.on_training_step_end()
        client.end_section.assert_called_once_with("tensor_metrics")
        assert ft_integration._seen_tr_iters_cnt == 11
