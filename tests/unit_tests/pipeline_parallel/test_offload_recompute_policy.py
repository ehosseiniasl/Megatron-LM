# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise early offload decisions with real checkpointing and CUDA transfers."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for offload/recompute tests."
)


@pytest.fixture
def offload_runtime():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    off_interface.reset_instance()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


def _expert_math(tensor):
    # Direct execution saves two tensors, whereas checkpointing saves only its
    # input. This difference must not change the policy learned during warmup.
    return tensor.sin().sin()


def _init_chunk(fraction, min_size=1):
    off_interface.init_chunk_handler(
        pp_rank=0,
        vp_size=None,
        vp_stage=None,
        min_offloaded_tensor_size=min_size,
        delta_offload_bytes_across_pp_ranks=0,
        activation_offload_fraction=fraction,
    )


@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.75, 1.0])
def test_early_policy_keeps_checkpoint_inputs_and_reload_slots(offload_runtime, fraction):
    """Decide once, avoid a second fraction pass, and preserve backward parity."""
    warmup_indices = {
        0.0: [],
        0.5: list(range(0, 70, 2)),
        0.75: [i for i in range(70) if i % 4 != 3],
        1.0: list(range(70)),
    }[fraction]
    steady_indices = warmup_indices
    reference_input = torch.full((16, 16), 0.1, device="cuda", requires_grad=True)
    reference_output = reference_input
    for _ in range(70):
        reference_output = _expert_math(reference_output)
    reference_output.sum().backward()
    expected_output = reference_output.detach().clone()
    expected_grad = reference_input.grad.detach().clone()

    # A second steady iteration verifies that cached groups retain their policy
    # after checkpoint inputs and offloaded intermediates have been consumed.
    for expected_indices in (warmup_indices, steady_indices, steady_indices):
        input_tensor = reference_input.detach().clone().requires_grad_()
        _init_chunk(fraction)
        output = input_tensor
        sources = []
        decisions = []
        for index in range(70):
            scope = off_interface(True, output, "fused_group_mlp")
            with scope as activation:
                recompute = scope.should_recompute_instead_of_offload()
                assert scope.should_recompute_instead_of_offload() == recompute
                decisions.append(recompute)
                source = activation.clone()
                sources.append(source)
                if recompute:
                    output = checkpoint(_expert_math, source, use_reentrant=True)
                else:
                    output = _expert_math(source)
            output = scope.group_offload(output, forced_released_tensors=[source])

        torch.cuda.synchronize()
        chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
        assert len(chunk.offload_groups) == 70
        assert chunk._warmup_decided_groups == 70
        assert chunk._warmup_eligible_groups == 0
        assert len(chunk._groups_to_reload) == 70
        assert not chunk._groups_to_offload
        actual_indices = [
            i
            for i, group in enumerate(chunk.offload_groups)
            if any(isinstance(state, tuple) for state in group._tensors.values())
        ]
        assert actual_indices == expected_indices
        assert [i for i, recompute in enumerate(decisions) if not recompute] == expected_indices
        for i, (group, source) in enumerate(zip(chunk.offload_groups, sources)):
            assert group.offload_policy_decided
            assert len(group._tensors) == (2 if i in expected_indices else 1)
            # A resident checkpoint input must survive forced-release handling.
            expected_bytes = 0 if i in expected_indices else input_tensor.numel() * 4
            assert source.untyped_storage().nbytes() == expected_bytes

        torch.testing.assert_close(output, expected_output)
        output.sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(input_tensor.grad, expected_grad)
        assert not chunk._groups_to_reload
        assert all(not group._tensors for group in chunk.offload_groups)
        del output, sources, source, input_tensor
        off_interface.reset()
        assert [
            i for i, group in enumerate(chunk.offload_groups) if group.offload
        ] == steady_indices


def test_early_policy_is_independent_of_empty_inputs_and_copy_threshold(offload_runtime):
    """Changing routed input shapes must preserve the GTP replay schedule."""
    layouts_by_iteration = (
        ["empty", "tiny", "full", "empty", "full", "full", "full"],
        ["full", "empty", "empty", "full", "tiny", "full", "full"],
    )
    expected_choices = [0, 2, 4, 6]
    expected_copies = ([2, 4, 6], [0, 6])
    for layouts, copies in zip(layouts_by_iteration, expected_copies):
        reference_input = torch.full((16, 16), 0.1, device="cuda", requires_grad=True)
        reference_output = reference_input
        for layout in layouts:
            group_input = reference_output[:0] if layout == "empty" else reference_output
            if layout == "tiny":
                group_input = reference_output[:1, :1]
            reference_output = reference_output + _expert_math(group_input).sum() / 256
        reference_output.sum().backward()
        _init_chunk(0.5, min_size=128)
        input_tensor = torch.full((16, 16), 0.1, device="cuda", requires_grad=True)
        output = input_tensor
        sources = []
        actual_choices = []
        for index, layout in enumerate(layouts):
            group_input = output[:0] if layout == "empty" else output
            if layout == "tiny":
                group_input = output[:1, :1]
            scope = off_interface(True, group_input, "fused_group_mlp")
            with scope as activation:
                recompute = scope.should_recompute_instead_of_offload()
                assert scope.should_recompute_instead_of_offload() == recompute
                if not recompute:
                    actual_choices.append(index)
                source = activation.clone()
                sources.append(source)
                result = (
                    checkpoint(_expert_math, source, use_reentrant=True)
                    if recompute
                    else _expert_math(source)
                )
            result = scope.group_offload(result, forced_released_tensors=[source])
            # Keep all group boundaries on one dependency chain, including empties.
            output = output + result.sum() / 256

        torch.cuda.synchronize()
        chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
        assert actual_choices == expected_choices
        assert chunk._warmup_decided_groups == 7
        assert chunk._warmup_eligible_groups == 0
        assert len(chunk._groups_to_reload) == len(layouts)
        actual_copies = [
            i
            for i, group in enumerate(chunk.offload_groups)
            if any(isinstance(state, tuple) for state in group._tensors.values())
        ]
        assert actual_copies == copies
        for i, (source, layout) in enumerate(zip(sources, layouts)):
            original_bytes = {"empty": 0, "tiny": 4, "full": 16 * 16 * 4}[layout]
            assert source.untyped_storage().nbytes() == (0 if i in copies else original_bytes)
        output.sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, reference_output)
        torch.testing.assert_close(input_tensor.grad, reference_input.grad)
        assert not chunk._groups_to_reload
        assert all(not group._tensors for group in chunk.offload_groups)
        del output, result, sources, source, input_tensor
        off_interface.reset()


def test_decided_terminal_group_waits_for_d2h_before_fallback_reload(offload_runtime, monkeypatch):
    """Wait before direct reload in discovery and cached-policy iterations."""
    reference_input = torch.full((16, 16), 0.1, device="cuda", requires_grad=True)
    reference_output = _expert_math(reference_input)
    reference_output.sum().backward()

    for is_warmup in (True, False, False):
        input_tensor = reference_input.detach().clone().requires_grad_()
        _init_chunk(0.65)

        with monkeypatch.context() as patch:
            scope = off_interface(True, input_tensor, "fused_group_mlp")
            with scope as activation:
                chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
                assert chunk.is_warmup == is_warmup
                offload = chunk.offload

                def delayed_offload(tensor, **kwargs):
                    # Queue bounded work on the D2H stream to exercise a backlogged
                    # copy without host-thread gates that could deadlock cleanup.
                    torch.cuda._sleep(50_000_000)
                    return offload(tensor, **kwargs)

                patch.setattr(chunk, "offload", delayed_offload)
                assert not scope.should_recompute_instead_of_offload()
                source = activation.clone()
                output = _expert_math(source)
            output = scope.group_offload(output, forced_released_tensors=[source])

            assert len(chunk.offload_groups) == 1
            group = chunk.offload_groups[-1]
            assert group.offload_policy_decided and group.offload
            assert len(group._tensors) == 2
            assert all(isinstance(state, tuple) for state in group._tensors.values())
            assert not chunk._reloading_group
            wait_offload_event = group.wait_offload_event
            reload = chunk.reload
            calls = []

            def checked_wait(stream):
                assert stream == torch.cuda.current_stream()
                wait_offload_event(stream)
                calls.append("wait")

            def checked_reload(state, non_blocking=None):
                # Assert the dependency, not event.query(): a fast device or
                # slow host may finish D2H before backward reaches this point.
                # The old fallback fails here even when numerical parity happens
                # to pass because its asynchronous copy completed in time.
                assert calls and calls[-1] == "wait"
                calls.append("reload")
                return reload(state, non_blocking=non_blocking)

            patch.setattr(group, "wait_offload_event", checked_wait)
            patch.setattr(chunk, "reload", checked_reload)
            # Do not synchronize before backward: the first group has no
            # prefetched reload and must establish its own D2H dependency.
            output.sum().backward()
            torch.cuda.synchronize()
            assert calls == ["wait", "reload", "wait", "reload"]
            torch.testing.assert_close(output, reference_output)
            torch.testing.assert_close(input_tensor.grad, reference_input.grad)
            assert not group._tensors
            assert not chunk._groups_to_reload

        del output, source, input_tensor
        off_interface.reset()
        assert group.offload


def test_disabled_interface_chooses_recompute_without_initializing_manager(monkeypatch):
    def unexpected_manager_access():
        raise AssertionError("Disabled offload must not access the manager")

    monkeypatch.setattr(PipelineOffloadManager, "get_instance", unexpected_manager_access)
    scope = off_interface(False, torch.ones(1), "fused_group_mlp")
    with scope:
        assert scope.should_recompute_instead_of_offload()
