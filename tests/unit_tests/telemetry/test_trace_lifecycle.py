# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for loop-pass trace roots and checkpoint finalization topology."""

import ast
import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
from opentelemetry.context import Context

REPO_ROOT = Path(__file__).resolve().parents[3]
TELEMETRY_PATH = REPO_ROOT / "megatron/core/telemetry/telemetry.py"
TRAINING_PATH = REPO_ROOT / "megatron/training/training.py"
ASYNC_UTILS_PATH = REPO_ROOT / "megatron/training/async_utils.py"
CHECKPOINTING_PATH = REPO_ROOT / "megatron/training/checkpointing.py"

EXPECTED_LIFECYCLE_SPANS = {
    "SPAN_TRAINING_ITER_BLOCK": "nv.dl.training.iter_block",
    "SPAN_CHECKPOINT_EXPOSED_SAVE": "nv.dl.training.checkpoint.exposed_save",
    "SPAN_CHECKPOINT_SAVE": "nv.dl.training.checkpoint.save",
    "SPAN_CHECKPOINT_SAVE_FINALIZE": "nv.dl.training.checkpoint.save.finalize",
    "SPAN_CHECKPOINT_EXIT_FINALIZE": "nv.mlm.checkpoint.exit_finalize",
}
NVRX_FINALIZE_SPAN = "nv.nvrx.ckpt.save.finalize"
NVRX_CALL_IDX = "nv.nvrx.ckpt.call_idx"


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _function(path, name):
    return next(
        node for node in _tree(path).body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _calls(node, name):
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call) and _call_name(candidate.func) == name
    ]


def _otel_constant(node):
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "_otel"
    ):
        return node.attr
    return None


def _module_string_constants(path):
    constants = {}
    for node in _tree(path).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = node.value.value
    return constants


def _import_nvrx_async_queue():
    core = pytest.importorskip("nvidia_resiliency_ext.checkpointing.async_ckpt.core")
    nvrx_telemetry = pytest.importorskip("nvidia_resiliency_ext.shared_utils.telemetry")
    return core.AsyncCallsQueue, core.AsyncRequest, nvrx_telemetry


class _CompletedAsyncCaller:
    """In-process caller stub that leaves queue topology and finalization in NVRx."""

    def __init__(self):
        self.scheduled = []
        self.completion_checks = []

    def schedule_async_call(self, async_request):
        self.scheduled.append(async_request)

    def is_current_async_call_done(self, blocking, no_dist):
        self.completion_checks.append((blocking, no_dist))
        return True


def _test_module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _test_package(monkeypatch, name, path):
    module = _test_module(monkeypatch, name)
    module.__path__ = [str(path)]
    return module


def _load_test_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _import_mcore_strategy(monkeypatch):
    """Load the production MCore selector and queue without importing Megatron's Triton surface."""
    core_root = REPO_ROOT / "megatron/core"
    dist_root = core_root / "dist_checkpointing"
    strategy_root = dist_root / "strategies"
    _test_package(monkeypatch, "megatron", REPO_ROOT / "megatron")
    _test_package(monkeypatch, "megatron.core", core_root)
    _test_package(monkeypatch, "megatron.core.dist_checkpointing", dist_root)
    _test_package(monkeypatch, "megatron.core.dist_checkpointing.strategies", strategy_root)
    _test_module(monkeypatch, "megatron.core._rank_utils", safe_get_rank=lambda: 0)
    _test_module(monkeypatch, "megatron.core.utils", log_single_rank=lambda *args, **kwargs: None)

    @contextmanager
    def debug_time(*args, **kwargs):
        yield

    _test_module(monkeypatch, "megatron.core.dist_checkpointing.utils", debug_time=debug_time)
    mcore_async = _load_test_module(
        monkeypatch,
        "megatron.core.dist_checkpointing.strategies.async_utils",
        strategy_root / "async_utils.py",
    )

    placeholder = type("Placeholder", (), {})
    _test_module(
        monkeypatch, "megatron.core.dist_checkpointing.core", CheckpointingException=Exception
    )
    _test_module(
        monkeypatch, "megatron.core.dist_checkpointing.dict_utils", nested_values=lambda value: ()
    )
    _test_module(
        monkeypatch,
        "megatron.core.dist_checkpointing.mapping",
        ShardedBase=placeholder,
        ShardedObject=placeholder,
        ShardedTensor=placeholder,
        StateDict=dict,
        ShardedStateDict=dict,
        is_main_replica=lambda replica_id: True,
    )
    _test_module(
        monkeypatch,
        "megatron.core.dist_checkpointing.strategies.checkpointable",
        CheckpointableShardedTensor=placeholder,
        LocalShardsContainer=placeholder,
    )
    _test_module(
        monkeypatch,
        "megatron.core.dist_checkpointing.strategies.nvrx",
        has_nvrx_async_support=lambda: False,
        make_nvrx_async_request=lambda *args, **kwargs: None,
    )

    class MultiStorageClientFeature:
        @staticmethod
        def is_enabled():
            return False

    _test_module(
        monkeypatch, "megatron.core.msc_utils", MultiStorageClientFeature=MultiStorageClientFeature
    )
    _test_module(
        monkeypatch,
        "megatron.core.dist_checkpointing.strategies.cached_metadata_filesystem_reader",
        CachedMetadataFileSystemReader=placeholder,
    )
    _test_module(
        monkeypatch,
        "megatron.core.dist_checkpointing.strategies.filesystem_async",
        FileSystemWriterAsync=placeholder,
        _results_queue=None,
        get_write_results_queue=lambda: None,
    )
    _test_module(
        monkeypatch,
        "megatron.core.dist_checkpointing.strategies.state_dict_saver",
        save_state_dict_async_finalize=lambda: None,
        save_state_dict_async_plan=lambda: None,
    )
    strategy = _load_test_module(
        monkeypatch, "megatron.core.dist_checkpointing.strategies.torch", strategy_root / "torch.py"
    )
    return strategy, mcore_async


def _finalize_nvrx_requests(monkeypatch, count):
    core = pytest.importorskip("nvidia_resiliency_ext.checkpointing.async_ckpt.core")
    async_calls_queue, async_request, _ = _import_nvrx_async_queue()
    queue = async_calls_queue(persistent=False)
    caller = _CompletedAsyncCaller()
    monkeypatch.setattr(queue, "_get_async_caller", lambda: caller)
    monkeypatch.setattr(core.torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(core.torch.distributed, "all_reduce", lambda tensor, op: None)
    try:
        for _ in range(count):
            queue.schedule_async_request(async_request(None, (), []))
        return queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
    finally:
        queue.close(abort=True)
        queue.async_calls.clear()


def _record_nvrx_spans(monkeypatch, nvrx_telemetry, tracer):
    @contextmanager
    def nvrx_span(group, name, tracer=None, **attributes):
        del group, tracer
        with span_exporter_tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span

    span_exporter_tracer = tracer
    monkeypatch.setattr(nvrx_telemetry, "_managed_span", nvrx_span)


@pytest.fixture
def span_exporter():
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export")
    try:
        in_memory_export = importlib.import_module("opentelemetry.sdk.trace.export")
        exporter_type = in_memory_export.InMemorySpanExporter
    except AttributeError:
        in_memory_export = importlib.import_module(
            "opentelemetry.sdk.trace.export.in_memory_span_exporter"
        )
        exporter_type = in_memory_export.InMemorySpanExporter

    exporter = exporter_type()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    try:
        yield provider.get_tracer("megatron-trace-lifecycle-test"), exporter
    finally:
        provider.shutdown()


def test_loop_pass_chain_and_finalize_topology(monkeypatch, span_exporter, telemetry_module):
    telemetry = telemetry_module
    _, _, nvrx_telemetry = _import_nvrx_async_queue()
    tracer, exporter = span_exporter
    monkeypatch.setattr(telemetry, "is_enabled", lambda group: group == telemetry.JOB)

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        assert group == telemetry.CKPT
        with span_exporter[0].start_as_current_span(name, attributes=attributes) as span:
            yield span

    monkeypatch.setattr(telemetry, "managed_span", managed_span)

    # Patch only NVRx's Lens adapter. Names, attributes, queue ordering, and the
    # number of finalize spans remain owned by the pinned AsyncCallsQueue code.
    _record_nvrx_spans(monkeypatch, nvrx_telemetry, tracer)

    startup = tracer.start_span(telemetry.SPAN_TRAINING_STARTUP, context=Context())
    startup_context = startup.get_span_context()
    startup.end()

    expected_completions = (0, 1, 3)
    lifecycle = telemetry.LoopPassSpanLifecycle()

    with tracer.start_as_current_span("ambient"):
        for training_step, completion_count in zip((17, 18, 19), expected_completions):
            lifecycle.start(tracer, training_step, initial_link_context=startup_context)
            with telemetry.managed_span(telemetry.CKPT, telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE):
                completed = _finalize_nvrx_requests(monkeypatch, completion_count)
            assert len(completed) == completion_count
            lifecycle.end()

        completed = telemetry.run_checkpoint_exit_finalize(
            _finalize_nvrx_requests, tracer, monkeypatch, 2
        )
        assert len(completed) == 2

    spans = list(exporter.get_finished_spans())
    ambient = next(span for span in spans if span.name == "ambient")
    startup_span = next(span for span in spans if span.name == telemetry.SPAN_TRAINING_STARTUP)
    loop_roots = sorted(
        (span for span in spans if span.name == telemetry.SPAN_TRAINING_ITER_BLOCK),
        key=lambda span: span.attributes[telemetry.TRAINING_STEP],
    )
    exit_root = next(span for span in spans if span.name == telemetry.SPAN_CHECKPOINT_EXIT_FINALIZE)

    assert [span.attributes[telemetry.TRAINING_STEP] for span in loop_roots] == [17, 18, 19]
    assert all(span.parent is None for span in (*loop_roots, exit_root))
    trace_ids = {
        ambient.context.trace_id,
        startup_span.context.trace_id,
        *(span.context.trace_id for span in loop_roots),
        exit_root.context.trace_id,
    }
    assert len(trace_ids) == 6
    assert [span.links[0].context.span_id for span in loop_roots] == [
        startup_span.context.span_id,
        loop_roots[0].context.span_id,
        loop_roots[1].context.span_id,
    ]
    assert all(len(span.links) == 1 for span in loop_roots)
    assert exit_root.links == ()
    for root, completion_count in zip(loop_roots, expected_completions):
        wrappers = [
            span
            for span in spans
            if span.name == telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE
            and span.parent is not None
            and span.parent.span_id == root.context.span_id
        ]
        assert len(wrappers) == 1
        children = [
            span
            for span in spans
            if span.name == NVRX_FINALIZE_SPAN
            and span.parent is not None
            and span.parent.span_id == wrappers[0].context.span_id
        ]
        assert [span.attributes[NVRX_CALL_IDX] for span in children] == list(
            range(completion_count)
        )

    exit_wrappers = [
        span
        for span in spans
        if span.name == telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE
        and span.parent is not None
        and span.parent.span_id == exit_root.context.span_id
    ]
    assert len(exit_wrappers) == 1
    exit_children = [
        span
        for span in spans
        if span.name == NVRX_FINALIZE_SPAN
        and span.parent is not None
        and span.parent.span_id == exit_wrappers[0].context.span_id
    ]
    assert [span.attributes[NVRX_CALL_IDX] for span in exit_children] == [0, 1]


def test_loop_pass_lifecycle_ends_active_root_on_exception(
    monkeypatch, span_exporter, telemetry_module
):
    telemetry = telemetry_module
    tracer, exporter = span_exporter
    monkeypatch.setattr(telemetry, "is_enabled", lambda group: group == telemetry.JOB)
    lifecycle = telemetry.LoopPassSpanLifecycle()

    with pytest.raises(RuntimeError, match="training failed"):
        try:
            lifecycle.start(tracer, 31)
            with tracer.start_as_current_span("iteration-child"):
                raise RuntimeError("training failed")
        finally:
            lifecycle.end()

    spans = list(exporter.get_finished_spans())
    root = next(span for span in spans if span.name == telemetry.SPAN_TRAINING_ITER_BLOCK)
    child = next(span for span in spans if span.name == "iteration-child")
    assert root.attributes[telemetry.TRAINING_STEP] == 31
    assert child.parent.span_id == root.context.span_id


@pytest.mark.parametrize("async_request_count", [0, 2], ids=["sync", "async"])
def test_checkpoint_save_cost_spans_runtime(
    monkeypatch, span_exporter, async_request_count, telemetry_module
):
    telemetry = telemetry_module
    tracer, exporter = span_exporter
    async_calls_queue, async_request, nvrx_telemetry = _import_nvrx_async_queue()
    monkeypatch.setattr(telemetry, "is_enabled", lambda group: group == telemetry.CKPT)

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        assert group == telemetry.CKPT
        with span_exporter[0].start_as_current_span(name, attributes=attributes) as span:
            yield span

    monkeypatch.setattr(telemetry, "managed_span", managed_span)
    _record_nvrx_spans(monkeypatch, nvrx_telemetry, tracer)

    queue = async_calls_queue(persistent=False)
    caller = _CompletedAsyncCaller()
    monkeypatch.setattr(queue, "_get_async_caller", lambda: caller)
    try:
        with telemetry.checkpoint_exposed_save_span(23):
            with telemetry.checkpoint_save_span(23):
                for _ in range(async_request_count):
                    queue.schedule_async_request(async_request(None, (), []))
    finally:
        queue.close(abort=True)
        queue.async_calls.clear()

    spans = list(exporter.get_finished_spans())
    exposed = next(span for span in spans if span.name == telemetry.SPAN_CHECKPOINT_EXPOSED_SAVE)
    save = next(span for span in spans if span.name == telemetry.SPAN_CHECKPOINT_SAVE)
    assert exposed.attributes == {telemetry.TRAINING_STEP: 23}
    assert save.attributes == {telemetry.TRAINING_STEP: 23}
    assert save.parent.span_id == exposed.context.span_id
    assert NVRX_CALL_IDX not in exposed.attributes
    assert NVRX_CALL_IDX not in save.attributes

    schedules = [span for span in spans if span.name == "nv.nvrx.ckpt.save.schedule"]
    assert [span.attributes[NVRX_CALL_IDX] for span in schedules] == list(
        range(async_request_count)
    )
    assert all(span.parent.span_id == save.context.span_id for span in schedules)


def test_exit_finalize_closes_root_and_wrapper_on_exception(
    monkeypatch, span_exporter, telemetry_module
):
    telemetry = telemetry_module
    tracer, exporter = span_exporter
    monkeypatch.setattr(telemetry, "is_enabled", lambda group: group == telemetry.JOB)

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        assert group == telemetry.CKPT
        with span_exporter[0].start_as_current_span(name, attributes=attributes) as span:
            yield span

    monkeypatch.setattr(telemetry, "managed_span", managed_span)

    def fail_drain():
        raise RuntimeError("drain failed")

    with pytest.raises(RuntimeError, match="drain failed"):
        telemetry.run_checkpoint_exit_finalize(fail_drain, tracer)

    spans = list(exporter.get_finished_spans())
    exit_root = next(span for span in spans if span.name == telemetry.SPAN_CHECKPOINT_EXIT_FINALIZE)
    wrapper = next(span for span in spans if span.name == telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE)
    assert exit_root.parent is None
    assert wrapper.parent.span_id == exit_root.context.span_id


def test_finalize_span_tags_drain_step_and_durable_checkpoint_step(
    monkeypatch, span_exporter, telemetry_module
):
    telemetry = telemetry_module
    tracer, exporter = span_exporter

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        assert group == telemetry.CKPT
        with drain_tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span

    drain_tracer = tracer
    monkeypatch.setattr(telemetry, "managed_span", managed_span)
    monkeypatch.setattr(telemetry, "_DURABLE_CHECKPOINT_STEP", None)

    with telemetry.checkpoint_save_finalize_span(31):
        pass
    with telemetry.checkpoint_save_finalize_span(32):
        telemetry.record_checkpoint_durable_step(31)
    # A drain that finalizes nothing must not inherit the previous durable step.
    with telemetry.checkpoint_save_finalize_span(33):
        pass
    with telemetry.checkpoint_save_finalize_span():
        pass

    drains = [
        span
        for span in exporter.get_finished_spans()
        if span.name == telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE
    ]
    assert [dict(span.attributes) for span in drains] == [
        {telemetry.TRAINING_STEP: 31},
        {telemetry.TRAINING_STEP: 32, telemetry.CHECKPOINT_LAST_ITERATION_STEP: 31},
        {telemetry.TRAINING_STEP: 33},
        {},
    ]


def test_durable_step_recorded_between_drains_is_exported_by_the_next_drain(
    monkeypatch, span_exporter, telemetry_module
):
    telemetry = telemetry_module
    tracer, exporter = span_exporter

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        assert group == telemetry.CKPT
        with drain_tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span

    drain_tracer = tracer
    monkeypatch.setattr(telemetry, "managed_span", managed_span)
    monkeypatch.setattr(telemetry, "_DURABLE_CHECKPOINT_STEP", None)

    # Reusing shared-memory buffers makes NVRx finalize the pending request while
    # preparing the next save, so the callback fires between drains rather than
    # inside one. The step must survive until a drain span can export it.
    telemetry.record_checkpoint_durable_step(40)
    with telemetry.checkpoint_save_finalize_span(51):
        pass
    with telemetry.checkpoint_save_finalize_span(52):
        pass

    drains = [
        span
        for span in exporter.get_finished_spans()
        if span.name == telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE
    ]
    assert [dict(span.attributes) for span in drains] == [
        {telemetry.TRAINING_STEP: 51, telemetry.CHECKPOINT_LAST_ITERATION_STEP: 40},
        {telemetry.TRAINING_STEP: 52},
    ]


def test_durable_step_keeps_the_newest_checkpoint_when_reports_coalesce(
    monkeypatch, span_exporter, telemetry_module
):
    telemetry = telemetry_module
    tracer, exporter = span_exporter

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        assert group == telemetry.CKPT
        with drain_tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span

    drain_tracer = tracer
    monkeypatch.setattr(telemetry, "managed_span", managed_span)
    monkeypatch.setattr(telemetry, "_DURABLE_CHECKPOINT_STEP", None)

    # Two requests can report durability before a drain exports either one; the
    # restart point is the newer checkpoint regardless of report order.
    telemetry.record_checkpoint_durable_step(60)
    telemetry.record_checkpoint_durable_step(50)
    with telemetry.checkpoint_save_finalize_span(61):
        pass

    drain = next(
        span
        for span in exporter.get_finished_spans()
        if span.name == telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE
    )
    assert drain.attributes[telemetry.CHECKPOINT_LAST_ITERATION_STEP] == 60


def test_durable_checkpoint_step_survives_nvrx_per_request_child_spans(
    monkeypatch, span_exporter, telemetry_module
):
    telemetry = telemetry_module
    tracer, exporter = span_exporter
    _, _, nvrx_telemetry = _import_nvrx_async_queue()

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        assert group == telemetry.CKPT
        with span_exporter[0].start_as_current_span(name, attributes=attributes) as span:
            yield span

    monkeypatch.setattr(telemetry, "managed_span", managed_span)
    monkeypatch.setattr(telemetry, "_DURABLE_CHECKPOINT_STEP", None)
    _record_nvrx_spans(monkeypatch, nvrx_telemetry, tracer)

    with telemetry.checkpoint_save_finalize_span(44):
        telemetry.record_checkpoint_durable_step(43)
        assert len(_finalize_nvrx_requests(monkeypatch, 1)) == 1

    spans = list(exporter.get_finished_spans())
    drain = next(span for span in spans if span.name == telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE)
    nvrx_child = next(span for span in spans if span.name == NVRX_FINALIZE_SPAN)
    assert nvrx_child.parent.span_id == drain.context.span_id
    assert telemetry.CHECKPOINT_LAST_ITERATION_STEP not in nvrx_child.attributes
    assert drain.attributes[telemetry.CHECKPOINT_LAST_ITERATION_STEP] == 43
    assert drain.attributes[telemetry.TRAINING_STEP] == 44


def test_mcore_strategy_queue_runs_inside_one_broad_finalize_span(
    monkeypatch, span_exporter, telemetry_module
):
    from opentelemetry import trace

    telemetry = telemetry_module
    tracer, exporter = span_exporter
    strategy_module, mcore_async = _import_mcore_strategy(monkeypatch)
    strategy_name, strategy = strategy_module.get_async_strategy("mcore")
    assert strategy_name == "mcore"
    assert strategy["AsyncCallsQueue"] is mcore_async.AsyncCallsQueue
    assert strategy["AsyncRequest"] is mcore_async.AsyncRequest

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        assert group == telemetry.CKPT
        with span_exporter[0].start_as_current_span(name, attributes=attributes) as span:
            yield span

    monkeypatch.setattr(telemetry, "managed_span", managed_span)
    monkeypatch.setattr(mcore_async.torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(mcore_async.torch.distributed, "all_reduce", lambda tensor, op: None)

    finalized = []

    def record_finalize(request_index):
        current_span_id = trace.get_current_span().get_span_context().span_id
        finalized.append((request_index, current_span_id))

    queue = strategy["AsyncCallsQueue"](persistent=False)
    caller = _CompletedAsyncCaller()
    monkeypatch.setattr(queue, "_get_async_caller", lambda: caller)
    scheduled_indices = []
    try:
        for request_index in range(2):
            request = strategy["AsyncRequest"](
                None, (), [lambda request_index=request_index: record_finalize(request_index)]
            )
            scheduled_indices.append(queue.schedule_async_request(request))

        with telemetry.managed_span(telemetry.CKPT, telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE):
            completed_indices = queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
    finally:
        queue.close(abort=True)

    spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name == telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE
    ]
    assert scheduled_indices == [0, 1]
    assert completed_indices == [0, 1]
    assert queue.get_num_unfinalized_calls() == 0
    assert len(spans) == 1
    assert finalized == [(0, spans[0].context.span_id), (1, spans[0].context.span_id)]
    assert caller.completion_checks == [(True, False), (True, False)]
    assert [request.call_idx for request in caller.scheduled] == [0, 1]
    assert all(request.finalize_fns == [] for request in caller.scheduled)


def test_nvrx_cpu_shm_reuse_drain_keeps_nvrx_topology(
    monkeypatch, span_exporter, tmp_path, telemetry_module
):
    telemetry = telemetry_module
    torch = pytest.importorskip("torch")
    metadata = pytest.importorskip("torch.distributed.checkpoint.metadata")
    planner = pytest.importorskip("torch.distributed.checkpoint.planner")
    async_calls_queue, async_request, nvrx_telemetry = _import_nvrx_async_queue()
    filesystem_async = pytest.importorskip(
        "nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async"
    )
    tracer, exporter = span_exporter
    _record_nvrx_spans(monkeypatch, nvrx_telemetry, tracer)

    source = torch.tensor(1.0)
    write_item = planner.WriteItem(
        index=metadata.MetadataIndex(fqn="tensor"), type=planner.WriteItemType.TENSOR
    )

    class Planner:
        def resolve_data(self, item):
            assert item == write_item
            return source

    queue = async_calls_queue(persistent=False, cpu_shm_mode=True)
    caller = _CompletedAsyncCaller()
    monkeypatch.setattr(queue, "_get_async_caller", lambda: caller)
    monkeypatch.setattr(torch.Tensor, "share_memory_", lambda self: self)
    monkeypatch.setattr(filesystem_async, "get_write_results_queue", lambda: object())
    filesystem_async.FileSystemWriterAsync.cleanup_tensor_caches()
    try:
        first_writer = filesystem_async.FileSystemWriterAsync(
            tmp_path, use_cached_data_structure=True, use_cpu_shm_for_gpu_tensors=True
        )
        first_writer.prepare_write_data(planner.SavePlan([write_item]), Planner())
        queue.schedule_async_request(async_request(None, (), []))

        source.fill_(2.0)
        second_writer = filesystem_async.FileSystemWriterAsync(
            tmp_path, use_cached_data_structure=True, use_cpu_shm_for_gpu_tensors=True
        )
        second_writer.prepare_write_data(planner.SavePlan([write_item]), Planner())
    finally:
        queue.close(abort=True)
        queue.async_calls.clear()
        filesystem_async.FileSystemWriterAsync.register_shm_drain_callback(None)
        filesystem_async.FileSystemWriterAsync.cleanup_tensor_caches()

    spans = list(exporter.get_finished_spans())
    shm_drain = next(span for span in spans if span.name == "nv.nvrx.ckpt.save.shm_drain")
    finalize = next(span for span in spans if span.name == NVRX_FINALIZE_SPAN)
    assert finalize.parent.span_id == shm_drain.context.span_id
    assert finalize.attributes[NVRX_CALL_IDX] == 0
    assert not any(span.name == telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE for span in spans)


def test_lifecycle_constants_and_call_sites_are_canonical():
    constants = _module_string_constants(TELEMETRY_PATH)
    assert {name: constants[name] for name in EXPECTED_LIFECYCLE_SPANS} == EXPECTED_LIFECYCLE_SPANS

    executable_strings = {
        node.value
        for node in ast.walk(_tree(TRAINING_PATH))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for constant, value in EXPECTED_LIFECYCLE_SPANS.items():
        assert constant in TELEMETRY_PATH.read_text()
        assert value not in executable_strings


def test_training_opens_exactly_one_root_at_the_top_of_each_loop_pass():
    train = _function(TRAINING_PATH, "train")
    loop = next(node for node in train.body if isinstance(node, ast.While))
    current_iteration_assignment = loop.body[0]
    assert isinstance(current_iteration_assignment, ast.Assign)
    assert isinstance(current_iteration_assignment.targets[0], ast.Name)
    assert current_iteration_assignment.targets[0].id == "current_iteration"
    assert ast.unparse(current_iteration_assignment.value) == "completed_iterations + 1"

    first_statement = loop.body[1]

    assert isinstance(first_statement, ast.Expr)
    assert isinstance(first_statement.value, ast.Call)
    assert _call_name(first_statement.value.func) == "_otel.start_training_loop_pass"
    assert isinstance(first_statement.value.args[0], ast.Name)
    assert first_statement.value.args[0].id == "current_iteration"
    assert len(_calls(loop, "_otel.start_training_loop_pass")) == 1

    start_helper = _function(TELEMETRY_PATH, "start_training_loop_pass")
    helper_source = ast.unparse(start_helper)
    assert "save_interval" not in helper_source
    assert "_ACTIVE_TRAINER_HANDLE._lifecycle.start_loop_pass" in helper_source
    lifecycle = next(
        node
        for node in _tree(TELEMETRY_PATH).body
        if isinstance(node, ast.ClassDef) and node.name == "LoopPassSpanLifecycle"
    )
    lifecycle_source = ast.unparse(lifecycle)
    assert "SPAN_TRAINING_ITER_BLOCK" in lifecycle_source
    assert "training_step=training_step" in lifecycle_source


def test_final_loop_pass_ends_before_the_path_specific_terminal_drain():
    train = _function(TRAINING_PATH, "train")
    loop_index = next(index for index, node in enumerate(train.body) if isinstance(node, ast.While))
    next_statement = train.body[loop_index + 1]
    assert isinstance(next_statement, ast.Expr)
    assert _call_name(next_statement.value.func) == "_otel.end_training_loop_pass"

    terminal_helper = _function(TELEMETRY_PATH, "finalize_training_exit")
    helper_calls = _calls(terminal_helper, "run_checkpoint_exit_finalize")
    assert len(helper_calls) == 1
    helper_end_calls = _calls(terminal_helper, "end_training_loop_pass")
    assert len(helper_end_calls) == 1
    assert helper_end_calls[0].lineno < helper_calls[0].lineno
    assert isinstance(helper_calls[0].args[0], ast.Name)
    assert helper_calls[0].args[0].id == "callback"

    lifecycle = next(
        node
        for node in _tree(TELEMETRY_PATH).body
        if isinstance(node, ast.ClassDef) and node.name == "_TrainerLifecycle"
    )
    close = next(
        node
        for node in lifecycle.body
        if isinstance(node, ast.FunctionDef) and node.name == "close"
    )
    assert len(_calls(close, "self.end_loop_pass")) == 1

    pretrain = _function(TRAINING_PATH, "pretrain")
    final_save = _calls(pretrain, "save_checkpoint_and_time")[-1]
    normal_terminal_drain = _calls(pretrain, "_otel.finalize_training_exit")[-1]
    assert final_save.lineno < normal_terminal_drain.lineno
    assert (
        next(
            keyword.value.value
            for keyword in normal_terminal_drain.keywords
            if keyword.arg == "terminate"
        )
        is True
    )

    early_terminal_drains = _calls(train, "_otel.finalize_training_exit")
    assert len(early_terminal_drains) == 1
    assert (
        next(
            keyword.value.value
            for keyword in early_terminal_drains[0].keywords
            if keyword.arg == "terminate"
        )
        is True
    )
    assert next_statement.value.lineno < early_terminal_drains[0].lineno

    training_exception = next(
        handler
        for handler in ast.walk(pretrain)
        if isinstance(handler, ast.ExceptHandler) and _calls(handler, "_otel.shutdown_training")
    )
    assert len(_calls(training_exception, "_otel.shutdown_training")) == 1


def test_poll_intermediate_drain_and_terminal_drains_wrap_complete_queue_calls():
    train = _function(TRAINING_PATH, "train")
    parents = {
        child: parent for parent in ast.walk(train) for child in ast.iter_child_nodes(parent)
    }
    finalize_contexts = [
        node
        for node in ast.walk(train)
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and _call_name(node.items[0].context_expr.func) == "_otel.checkpoint_save_finalize_span"
    ]
    assert len(finalize_contexts) == 2
    assert {
        ast.unparse(context.items[0].context_expr.args[0]) for context in finalize_contexts
    } == {"current_iteration", "completed_iterations"}
    arguments = []
    for context in finalize_contexts:
        parent = parents[context]
        statements = next(
            value
            for _field, value in ast.iter_fields(parent)
            if isinstance(value, list) and context in value
        )
        assignment = statements[statements.index(context) - 1]
        assert ast.unparse(assignment) == "_finalize_async_save = maybe_finalize_async_save"
        call = context.body[0].value
        assert ast.unparse(call.func) == "_finalize_async_save"
        arguments.append({keyword.arg: keyword.value.value for keyword in call.keywords})
    assert sorted(arguments, key=lambda item: item["blocking"]) == [
        {"blocking": False},
        {"blocking": True, "terminate": False},
    ]

    terminal_helper = _function(TELEMETRY_PATH, "finalize_training_exit")
    exit_calls = _calls(terminal_helper, "run_checkpoint_exit_finalize")
    assert len(exit_calls) == 1
    assert isinstance(exit_calls[0].args[0], ast.Name)
    assert exit_calls[0].args[0].id == "callback"
    assert all(
        ast.unparse(call.args[0]) == "maybe_finalize_async_save"
        for call in _calls(train, "_otel.finalize_training_exit")
    )


def test_durable_step_is_registered_before_the_logits_finalize_handoff():
    save_checkpoint = _function(CHECKPOINTING_PATH, "save_checkpoint")
    registrations = [
        node
        for node in _calls(save_checkpoint, "async_save_request.add_finalize_fn")
        if "_otel.record_checkpoint_durable_step" in ast.unparse(node.args[0])
    ]
    assert len(registrations) == 1
    # The step is bound as a default argument so the callback reports the iteration it
    # was registered for, not whatever the name holds when the queue drains.
    assert ast.unparse(registrations[0].args[0]) == (
        "lambda i=iteration: _otel.record_checkpoint_durable_step(i)"
    )

    # Distillation moves the checkpoint request's finalize_fns onto the logits request so
    # success callbacks only fire once both writes land. Registering after that handoff
    # would report durability from the checkpoint request alone, before the tracker file
    # is written.
    handoffs = [
        node
        for node in ast.walk(save_checkpoint)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "async_save_request.finalize_fns.clear"
    ]
    assert len(handoffs) == 1
    assert registrations[0].lineno < handoffs[0].lineno


def test_checkpoint_cost_spans_carry_step_without_request_index():
    save = _function(TRAINING_PATH, "save_checkpoint_and_time")
    exposed_wrapper = _calls(save, "_otel.checkpoint_exposed_save_span")[0]
    assert isinstance(exposed_wrapper.args[0], ast.Name)
    assert exposed_wrapper.args[0].id == "iteration"

    save_wrapper = _calls(save, "_otel.checkpoint_save_span")[0]
    assert isinstance(save_wrapper.args[0], ast.Name)
    assert save_wrapper.args[0].id == "iteration"

    for helper_name, span_constant in (
        ("checkpoint_exposed_save_span", "SPAN_CHECKPOINT_EXPOSED_SAVE"),
        ("checkpoint_save_span", "SPAN_CHECKPOINT_SAVE"),
    ):
        shim_source = ast.unparse(_function(TELEMETRY_PATH, helper_name))
        assert "is_enabled(CKPT)" in shim_source
        assert f"managed_span(CKPT, {span_constant}" in shim_source
        assert "TRAINING_STEP" in shim_source

    finalize_source = ast.unparse(_function(TELEMETRY_PATH, "checkpoint_save_finalize_span"))
    assert "managed_span(CKPT, SPAN_CHECKPOINT_SAVE_FINALIZE" in finalize_source
    assert "TRAINING_STEP" in finalize_source
    assert "CHECKPOINT_LAST_ITERATION_STEP" in finalize_source

    combined_source = TRAINING_PATH.read_text() + ASYNC_UTILS_PATH.read_text()
    assert "_tag_current_span_call_idx" not in combined_source
    assert "nvrx.call_idx" not in combined_source
    assert NVRX_CALL_IDX not in combined_source


def test_async_queue_results_are_not_propagated_into_trainer_telemetry():
    schedule = _function(ASYNC_UTILS_PATH, "schedule_async_save")
    finalize = _function(ASYNC_UTILS_PATH, "maybe_finalize_async_save")
    schedule_calls = [
        call
        for call in ast.walk(schedule)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    ]
    finalize_calls = [
        call
        for call in ast.walk(finalize)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    ]

    assert sum(call.func.attr == "schedule_async_request" for call in schedule_calls) == 1
    assert sum(call.func.attr == "maybe_finalize_async_calls" for call in finalize_calls) == 1
    assert not any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in {"schedule_async_request", "maybe_finalize_async_calls"}
            for call in ast.walk(node)
        )
        for node in (*ast.walk(schedule), *ast.walk(finalize))
    )


def test_default_preset_keeps_coarse_root_without_iteration_or_detail_groups(telemetry_module):
    constants = _module_string_constants(TELEMETRY_PATH)
    assert constants["SPAN_TRAINING_ITER_BLOCK"] == "nv.dl.training.iter_block"

    telemetry = telemetry_module
    assert telemetry.JOB in telemetry.PRESETS["default"]
    assert telemetry.TRAIN not in telemetry.PRESETS["default"]
    assert telemetry.DETAIL not in telemetry.PRESETS["default"]
