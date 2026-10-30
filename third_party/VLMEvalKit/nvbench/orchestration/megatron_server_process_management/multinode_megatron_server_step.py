"""Coordinate one Megatron torchrun group across Slurm server nodes."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..configured_eval_run import ResolvedEvalRun
from ..eval_run_path_resolution import (
    PreparedMegatronDynamicServerVlmFewshotRun,
    prepare_vlmeval_run_directory,
)
from ..resolved_run_manifest import (
    MANIFEST_FILENAME,
    canonical_config_fingerprint,
    write_resolved_run_manifest,
)
from ..utils import (
    JsonArtifactError,
    SHA256_PATTERN,
    SLURM_JOB_ID_PATTERN,
    read_exact_json,
    write_json_once,
)
from .megatron_dynamic_server_launch import (
    LocalTorchrunServerProcess,
    MegatronDynamicServerLaunch,
    raise_on_termination_signal,
)
from .split_megatron_server_step import (
    EnclosingAllocationError,
    PreparedServerStep,
    SplitMegatronServerStepError,
    SplitMegatronServerStepResult,
    SplitServerStepControlPaths,
    SplitServerStepWaitError,
    TerminalServerStep,
    create_split_server_step_control_paths,
    read_terminal_server_step,
    wait_for_prepared_server_step,
    wait_for_server_step_shutdown_request,
    wait_for_terminal_server_step,
    write_prepared_server_step,
    write_terminal_server_step,
)


@dataclass(frozen=True)
class MultinodeAllocation:
    """Slurm identity needed by one node-local supervisor."""

    job_id: str
    nodes: int
    node_rank: int


def _slurm_integer(environment: Mapping[str, str], name: str, *, allow_zero: bool = False) -> int:
    value = environment.get(name, "")
    if type(value) is not str or not value.isdecimal() or (
        not allow_zero and value.startswith("0")
    ):
        raise EnclosingAllocationError(f"{name} is not a valid Slurm integer")
    parsed = int(value)
    if parsed < int(not allow_zero):
        raise EnclosingAllocationError(f"{name} is not a valid Slurm integer")
    return parsed


def validate_multinode_allocation(
    resolved: ResolvedEvalRun,
    environment: Mapping[str, str] | None = None,
) -> MultinodeAllocation:
    """Require one supervisor task per configured server node."""

    if type(resolved) is not ResolvedEvalRun:
        raise EnclosingAllocationError("resolved run has the wrong record type")
    values = os.environ if environment is None else environment
    job_id = values.get("SLURM_JOB_ID", "")
    if SLURM_JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise EnclosingAllocationError("SLURM_JOB_ID is invalid")
    run = resolved.configured.run
    nodes = _slurm_integer(values, "SLURM_NNODES")
    node_rank = _slurm_integer(values, "SLURM_NODEID", allow_zero=True)
    valid = (
        run.launch_mode == "slurm_multinode_torchrun"
        and run.nodes > 1
        and nodes == run.nodes == _slurm_integer(values, "SLURM_JOB_NUM_NODES")
        and nodes == _slurm_integer(values, "SLURM_NTASKS")
        and run.gpus_per_node == _slurm_integer(values, "SLURM_GPUS_ON_NODE")
        and node_rank < nodes
        and _slurm_integer(values, "SLURM_LOCALID", allow_zero=True) == 0
        and _slurm_integer(values, "SLURM_PROCID", allow_zero=True) == node_rank
    )
    if not valid:
        raise EnclosingAllocationError("allocation must provide one supervisor per server node")
    return MultinodeAllocation(job_id, nodes, node_rank)


def _node_log(paths: SplitServerStepControlPaths, rank: int) -> Path:
    return paths.attempt_directory / f"megatron_dynamic_server_node_{rank:03d}.log"


def _node_terminal(paths: SplitServerStepControlPaths, rank: int) -> Path:
    return paths.attempt_directory / f"server_step_node_{rank:03d}_terminal.json"


def _write_node_terminal(
    paths: SplitServerStepControlPaths,
    rank: int,
    terminal: TerminalServerStep,
) -> Path:
    try:
        return write_json_once(
            _node_terminal(paths, rank), {"schema_version": 1, **asdict(terminal)}
        )
    except JsonArtifactError as error:
        raise SplitMegatronServerStepError("node terminal state could not be recorded") from error


def _read_node_terminal(paths: SplitServerStepControlPaths, rank: int) -> TerminalServerStep:
    try:
        payload = read_exact_json(
            _node_terminal(paths, rank),
            {"schema_version", "config_fingerprint", "status", "process_returncode"},
        )
        if payload.pop("schema_version") != 1:
            raise ValueError
        terminal = TerminalServerStep(**payload)  # type: ignore[arg-type]
        if (
            SHA256_PATTERN.fullmatch(terminal.config_fingerprint) is None
            or terminal.status not in {"clean", "failed"}
            or (
                terminal.process_returncode is not None
                and type(terminal.process_returncode) is not int
            )
        ):
            raise ValueError
        return terminal
    except (JsonArtifactError, TypeError, ValueError) as error:
        raise SplitServerStepWaitError("node terminal state is malformed") from error


def _wait_for_node_terminals(
    paths: SplitServerStepControlPaths,
    fingerprint: str,
    nodes: int,
    timeout_seconds: float = 60.0,
) -> tuple[TerminalServerStep, ...]:
    deadline = time.monotonic() + timeout_seconds
    while not all(_node_terminal(paths, rank).exists() for rank in range(nodes)):
        if time.monotonic() >= deadline:
            raise SplitServerStepWaitError("timed out waiting for node cleanup")
        time.sleep(1.0)
    terminals = tuple(_read_node_terminal(paths, rank) for rank in range(nodes))
    if any(terminal.config_fingerprint != fingerprint for terminal in terminals):
        raise SplitServerStepWaitError("node terminal fingerprint mismatch")
    return terminals


@dataclass(frozen=True)
class MultinodeServerStepDependencies:
    """Small injectable process boundary for lifecycle tests."""

    allocation_validator: Callable[[ResolvedEvalRun], MultinodeAllocation]
    process_factory: Callable[[MegatronDynamicServerLaunch], object]
    shutdown_waiter: Callable[..., None]


def _production_dependencies() -> MultinodeServerStepDependencies:
    return MultinodeServerStepDependencies(
        validate_multinode_allocation,
        LocalTorchrunServerProcess,
        wait_for_server_step_shutdown_request,
    )


def execute_multinode_megatron_server_step(
    prepared: PreparedMegatronDynamicServerVlmFewshotRun,
    *,
    dependencies: MultinodeServerStepDependencies | None = None,
) -> SplitMegatronServerStepResult:
    """Run one node supervisor and let rank zero aggregate cleanup state."""

    if type(prepared) is not PreparedMegatronDynamicServerVlmFewshotRun:
        raise SplitMegatronServerStepError("prepared run has the wrong record type")
    deps = _production_dependencies() if dependencies is None else dependencies
    allocation = deps.allocation_validator(prepared.resolved)
    fingerprint = canonical_config_fingerprint(prepared.resolved)
    timeout = prepared.resolved.configured.model.server.health_timeout_seconds

    if allocation.node_rank == 0:
        directory = prepare_vlmeval_run_directory(prepared.resolved)
        manifest = write_resolved_run_manifest(
            prepared.resolved, slurm_job_id=allocation.job_id
        )
        paths = create_split_server_step_control_paths(directory.path, allocation.job_id)
        launch = replace(prepared.launch, server_log_path=str(_node_log(paths, 0)))
        prepared_artifact = write_prepared_server_step(
            paths,
            PreparedServerStep(
                fingerprint, launch.endpoint, launch.health_endpoint, launch.server_log_path
            ),
        )
    else:
        paths, published = wait_for_prepared_server_step(
            Path(prepared.resolved.paths.run_directory),
            allocation.job_id,
            fingerprint,
            timeout,
        )
        launch = replace(
            prepared.launch,
            server_log_path=str(_node_log(paths, allocation.node_rank)),
        )
        if (published.endpoint, published.health_endpoint) != (
            launch.endpoint,
            launch.health_endpoint,
        ):
            raise SplitMegatronServerStepError("rank-zero endpoint differs across supervisors")
        manifest = Path(prepared.resolved.paths.run_directory) / MANIFEST_FILENAME
        prepared_artifact = paths.prepared

    process: object | None = None
    returncode: int | None = None
    error: Exception | None = None
    try:
        process = deps.process_factory(launch)
        process.start()  # type: ignore[attr-defined]
        with raise_on_termination_signal(process.forward_signal):  # type: ignore[attr-defined]
            deps.shutdown_waiter(paths, fingerprint, process.poll)  # type: ignore[attr-defined]
    except Exception as server_error:
        error = server_error
    finally:
        if process is not None:
            try:
                returncode = process.terminate()  # type: ignore[attr-defined]
            except Exception as cleanup_error:
                error = error or cleanup_error
        terminal = TerminalServerStep(
            fingerprint, "clean" if error is None else "failed", returncode
        )
        _write_node_terminal(paths, allocation.node_rank, terminal)

    if allocation.node_rank == 0:
        try:
            node_terminals = _wait_for_node_terminals(paths, fingerprint, allocation.nodes)
            if any(terminal.status != "clean" for terminal in node_terminals):
                raise SplitServerStepWaitError("one or more node supervisors failed")
        except Exception as aggregate_error:
            error = error or aggregate_error
        terminal_artifact = write_terminal_server_step(
            paths,
            TerminalServerStep(
                fingerprint, "clean" if error is None else "failed", returncode
            ),
        )
    else:
        global_terminal = wait_for_terminal_server_step(paths, fingerprint, 60.0)
        terminal_artifact = paths.terminal
        if global_terminal.status != "clean":
            error = error or SplitServerStepWaitError("rank-zero aggregation failed")

    if error is not None:
        raise SplitMegatronServerStepError("multi-node server step failed") from error
    return SplitMegatronServerStepResult(
        manifest,
        prepared_artifact,
        terminal_artifact,
        Path(launch.server_log_path),
        returncode,
    )
