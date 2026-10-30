# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pre-reduce-scatter NaN/Inf flags name every offending GTP weight and work with reruns."""

import pytest
import torch

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import (
    _ParamAndGradBucket,
    _ParamAndGradBucketGroup,
)
from megatron.core.rerun_state_machine import (
    RerunMode,
    destroy_rerun_state_machine,
    get_rerun_state_machine,
    initialize_rerun_state_machine,
)
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

pytestmark = [
    pytest.mark.skipif(not HAVE_GTP, reason="GTP requires TransformerEngine"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="The flags live on the GPU"),
]

if HAVE_GTP:
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        NonFiniteWgrads,
        attach_nonfinite_wgrad_tracker,
        check_gtp_wgrads_finite,
    )


@pytest.fixture(autouse=True)
def rerun_disabled():
    initialize_rerun_state_machine(mode=RerunMode.DISABLED)
    yield
    destroy_rerun_state_machine()


class _GTPModel(torch.nn.Module):
    """Two dense GTP weights and one routed-expert param covering two expert weights."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Parameter(torch.zeros(4, 4, device="cuda"))
        self.fc2 = torch.nn.Parameter(torch.zeros(4, 4, device="cuda"))
        self.expert0 = torch.nn.Parameter(torch.zeros(4, 4, device="cuda"))
        self.expert1 = torch.nn.Parameter(torch.zeros(4, 4, device="cuda"))
        for param in self.parameters():
            param.is_gtp_weight_remat = True
        # As GTP wraps grouped experts: every expert is routed, only the first holds the list.
        self.expert0.is_routed_expert = True
        self.expert1.is_routed_expert = True
        self.expert0.weight_list = [self.expert0, self.expert1]
        self.expert1.weight_list = None


def _record(param, *values):
    wgrads = [torch.full((4, 4), value, device="cuda") for value in values]
    param._nonfinite_wgrad_tracker.record(param._nonfinite_wgrad_slots, wgrads)


@pytest.fixture
def model():
    model = _GTPModel()
    attach_nonfinite_wgrad_tracker(model)
    return model


def test_every_weight_shares_one_tracker(model):
    trackers = {id(p._nonfinite_wgrad_tracker) for p in (model.fc1, model.fc2, model.expert0)}
    assert len(trackers) == 1
    assert model.expert0._nonfinite_wgrad_slots.tolist() == [2, 3]
    # A non-first expert has no weight_list and covers only its own slot.
    assert model.expert1._nonfinite_wgrad_slots.tolist() == [3]


def test_finite_wgrads_pass(model):
    _record(model.fc1, 1.0)
    _record(model.expert0, 1.0, 2.0)
    check_gtp_wgrads_finite(model)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_every_offending_weight_is_named_and_flags_clear(model, bad):
    _record(model.fc1, 1.0)
    _record(model.expert0, 1.0, bad)  # Only expert1 is bad within the batched call.
    _record(model.fc2, bad)
    with pytest.raises(RuntimeError, match=r"2 GTP weight\(s\).*fc2.*expert1"):
        check_gtp_wgrads_finite(model)
    check_gtp_wgrads_finite(model)  # Reported once: the flags were cleared.


def test_later_clean_reduce_scatter_does_not_clear_a_flag(model):
    # A weight can be reduce-scattered more than once per step (microbatches, MTP); flags stick.
    _record(model.fc1, float("nan"))
    _record(model.fc1, 1.0)
    with pytest.raises(RuntimeError, match="fc1"):
        check_gtp_wgrads_finite(model)


def test_check_uses_the_tracker_recorded_on_the_model(model, monkeypatch):
    assert model._gtp_nonfinite_wgrad_tracker is model.fc1._nonfinite_wgrad_tracker

    def walk_parameters(*args, **kwargs):
        raise AssertionError("check_gtp_wgrads_finite walked the parameters")

    monkeypatch.setattr(model, "parameters", walk_parameters)
    _record(model.fc2, float("nan"))
    with pytest.raises(RuntimeError, match="fc2"):
        check_gtp_wgrads_finite(model)


def test_names_tell_model_chunks_apart():
    chunks = [_GTPModel(), _GTPModel()]
    attach_nonfinite_wgrad_tracker(chunks)
    names = chunks[0].fc1._nonfinite_wgrad_tracker.names
    assert names[:2] == ["chunk0.fc1", "chunk0.fc2"]
    assert names[4:6] == ["chunk1.fc1", "chunk1.fc2"]


def test_model_without_gtp_weights_is_a_no_op():
    model = torch.nn.Linear(4, 4, device="cuda")
    attach_nonfinite_wgrad_tracker(model)
    assert not hasattr(model.weight, "_nonfinite_wgrad_tracker")
    assert model._gtp_nonfinite_wgrad_tracker is None
    check_gtp_wgrads_finite(model)


def test_result_prints_at_most_five_names():
    weights = tuple((slot, f"layer{slot}.weight") for slot in range(8))
    assert repr(NonFiniteWgrads(weights)) == (
        "8 GTP weight(s): layer0.weight, layer1.weight, layer2.weight, layer3.weight, "
        "layer4.weight and 3 more"
    )
    assert repr(NonFiniteWgrads(())) == "no GTP weights"
    assert not NonFiniteWgrads(())


def _rerun_machine(monkeypatch, mode):
    """The live rerun state machine in ``mode``, without RNG snapshots or collectives."""
    machine = get_rerun_state_machine()
    machine.set_mode(mode)
    machine.first_iteration_complete = True
    monkeypatch.setattr(machine, "_save_state", lambda: None)
    monkeypatch.setattr(machine, "_restore_state", lambda: None)
    monkeypatch.setattr(machine, "_reduce_any", lambda value: value)
    return machine


@pytest.mark.parametrize(
    "initial_slots, rerun_slots, expected_status",
    [
        ((0,), (), "first_rerun_not_reproducible"),
        ((0,), (0,), "first_rerun_reproducible"),
        ((0,), (1,), "first_rerun_not_reproducible"),
        ((0,), (0, 1), "first_rerun_not_reproducible"),
        ((0, 1), (0,), "first_rerun_not_reproducible"),
    ],
)
def test_rerun_compares_the_flagged_weights(
    model, monkeypatch, initial_slots, rerun_slots, expected_status
):
    machine = _rerun_machine(monkeypatch, RerunMode.VALIDATE_RESULTS)
    statuses = []
    monkeypatch.setattr(
        machine,
        "_log_validation_error_to_file",
        lambda **kwargs: statuses.append(kwargs["status"].value),
    )
    tracker = model.fc1._nonfinite_wgrad_tracker
    # Equal names must not hide a switch between two different weights.
    tracker.names[1] = tracker.names[0]

    assert machine.should_run_forward_backward(None)
    for slot in initial_slots:
        tracker.flags[slot] = 1
    check_gtp_wgrads_finite(model)
    assert machine.should_run_forward_backward(None)  # The flagged step is rerun.
    for slot in rerun_slots:
        tracker.flags[slot] = 1
    check_gtp_wgrads_finite(model)
    assert not machine.should_run_forward_backward(None)

    assert statuses == ["initial_run", expected_status]
    # Fatal either way. Whether a checkpoint is saved first is the state machine's policy.
    _, should_exit, _ = machine.should_checkpoint_and_exit()
    assert should_exit


def test_determinism_stats_pairs_a_clean_check_with_a_flagged_rerun(model, monkeypatch):
    machine = _rerun_machine(monkeypatch, RerunMode.REPORT_DETERMINISM_STATS)
    monkeypatch.setattr(machine, "_maybe_report_stats", lambda: None)

    assert machine.should_run_forward_backward(None)
    check_gtp_wgrads_finite(model)
    assert machine.should_run_forward_backward(None)
    _record(model.fc1, float("nan"))
    check_gtp_wgrads_finite(model)  # Must find the initial run's clean result, not assert.
    assert not machine.should_run_forward_backward(None)
    assert machine.should_checkpoint_and_exit() == (False, False, 0)


def _bucket_group(param):
    """A real DDP bucket group; check_grads itself needs no distributed collective."""
    bucket = _ParamAndGradBucket(
        params=[param],
        param_data=None,
        grad_data=torch.zeros_like(param).flatten(),
        offset=0,
        numel_unpadded=param.numel(),
        gradient_scaling_factor=1.0,
        bucket_id=0,
        param_index_map={param: (0, param.numel(), 0)},
        params_with_extra_main_grads=[],
    )
    return _ParamAndGradBucketGroup(
        buckets=[bucket],
        ddp_config=DistributedDataParallelConfig(overlap_grad_reduce=True),
        collective_group=None,
        collective_group_size=1,
    )


@pytest.mark.parametrize(
    "first_name, other_name, final_model_check",
    [("fc1", "fc2", False), ("expert0", "expert1", False), ("fc1", "fc2", True)],
)
def test_ddp_check_preserves_other_bucket_flags(model, first_name, other_name, final_model_check):
    """A bucket reports only its weights and leaves the others for their own check."""
    first, other = getattr(model, first_name), getattr(model, other_name)
    if first_name == "expert0":
        _record(first, float("nan"), float("nan"))  # One RS records both experts.
    else:
        _record(first, float("nan"))
        _record(other, float("nan"))

    with pytest.raises(RuntimeError, match=rf"1 GTP weight\(s\): {first_name}"):
        _bucket_group(first).check_grads(check_for_nan_or_inf=True, check_for_large=False)
    with pytest.raises(RuntimeError, match=rf"1 GTP weight\(s\): {other_name}"):
        if final_model_check:
            check_gtp_wgrads_finite(model)
        else:
            _bucket_group(other).check_grads(check_for_nan_or_inf=True, check_for_large=False)
    check_gtp_wgrads_finite(model)  # Both checked slots were cleared.


def test_ddp_rerun_ignores_other_bucket_completion_order(model, monkeypatch):
    """Model B's flag becoming visible after A's check initially, but before it on rerun."""
    groups = [_bucket_group(model.fc1), _bucket_group(model.fc2)]
    machine = _rerun_machine(monkeypatch, RerunMode.VALIDATE_RESULTS)
    statuses = []
    monkeypatch.setattr(
        machine,
        "_log_validation_error_to_file",
        lambda **kwargs: statuses.append(kwargs["status"].value),
    )
    for rerun in (False, True):
        assert machine.should_run_forward_backward(None)
        _record(model.fc1, float("nan"))
        if rerun:
            _record(model.fc2, float("nan"))
        groups[0].check_grads(check_for_nan_or_inf=True, check_for_large=False)
        if not rerun:
            _record(model.fc2, float("nan"))
        groups[1].check_grads(check_for_nan_or_inf=True, check_for_large=False)
    assert not machine.should_run_forward_backward(None)
    assert statuses == ["initial_run", "first_rerun_reproducible"]


def test_captured_records_survive_separate_bucket_checks(model):
    """Replay writes the same flag storage, and checking one bucket preserves the other."""
    tracker = model.fc1._nonfinite_wgrad_tracker
    groups = [_bucket_group(model.fc1), _bucket_group(model.fc2)]
    bad = torch.full_like(model.fc1, float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for param in (model.fc1, model.fc2):
            tracker.record(param._nonfinite_wgrad_slots, [bad])
    stream.synchronize()
    tracker.flags.zero_()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for param in (model.fc1, model.fc2):
            tracker.record(param._nonfinite_wgrad_slots, [bad])

    for _ in range(2):
        graph.replay()
        for index, group in enumerate(groups, start=1):
            with pytest.raises(RuntimeError, match=rf"1 GTP weight\(s\): fc{index}"):
                group.check_grads(check_for_nan_or_inf=True, check_for_large=False)
        check_gtp_wgrads_finite(model)
