"""Own the complete lifecycle of the background Megatron server job step.

This module publishes cross-step control artifacts, supervises one local
torchrun launcher, and records cleanup truth for the client. Torchrun and
Megatron retain ownership of their distributed and frontend descendants.
"""

from __future__ import annotations

import math
import os
import stat
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


CONTROL_DIRECTORY_NAME = "megatron_dynamic_server_step_attempts"
PREPARED_FILENAME = "server_step_prepared.json"
SHUTDOWN_REQUEST_FILENAME = "server_step_shutdown_request.json"
TERMINAL_FILENAME = "server_step_terminal.json"
_CONTROL_REQUIRED_MODE = stat.S_IRGRP | stat.S_IXGRP


class EnclosingAllocationError(RuntimeError):
    """Raised when the active allocation differs from the resolved run contract."""


@dataclass(frozen=True)
class EnclosingAllocation:
    """Validated identity and topology of the one enclosing Slurm task."""

    job_id: str
    nodes: int
    tasks: int
    gpus_per_node: int


def _positive_decimal(environment: Mapping[str, str], name: str) -> int:
    value = environment.get(name)
    if type(value) is not str or not value.isdecimal() or value.startswith("0"):
        raise EnclosingAllocationError(f"{name} must be a positive decimal integer")
    parsed = int(value)
    if parsed <= 0:
        raise EnclosingAllocationError(f"{name} must be positive")
    return parsed


def validate_enclosing_allocation(
    resolved: ResolvedEvalRun,
    environment: Mapping[str, str] | None = None,
) -> EnclosingAllocation:
    """Validate the enclosing Slurm allocation without querying the scheduler.

    ``resolved`` supplies the required node and GPU shape. ``environment`` may
    provide a read-only mapping for tests; otherwise the current process
    environment is inspected. The function performs no mutation or scheduler
    calls and returns the validated job identity and topology. Missing,
    malformed, multi-task, or mismatched fields raise
    :class:`EnclosingAllocationError`.
    """
    if type(resolved) is not ResolvedEvalRun:
        raise EnclosingAllocationError("resolved run has the wrong record type")
    values = os.environ if environment is None else environment
    job_id = values.get("SLURM_JOB_ID")
    if type(job_id) is not str or SLURM_JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise EnclosingAllocationError("SLURM_JOB_ID must identify one allocation task")
    nodes = _positive_decimal(values, "SLURM_NNODES")
    declared_nodes = _positive_decimal(values, "SLURM_JOB_NUM_NODES")
    tasks = _positive_decimal(values, "SLURM_NTASKS")
    gpus_per_node = _positive_decimal(values, "SLURM_GPUS_ON_NODE")
    if nodes != declared_nodes or nodes != resolved.configured.run.nodes:
        raise EnclosingAllocationError("allocated node count differs from the run config")
    if tasks != 1:
        raise EnclosingAllocationError("the local supervisor requires exactly one enclosing task")
    if gpus_per_node != resolved.configured.run.gpus_per_node:
        raise EnclosingAllocationError("allocated GPUs per node differ from the run config")
    return EnclosingAllocation(job_id, nodes, tasks, gpus_per_node)


class SplitServerStepControlError(RuntimeError):
    """Raised when split-step control state is unsafe, malformed, or contradictory."""


@dataclass(frozen=True)
class SplitServerStepControlPaths:
    """Fixed immutable paths for one job-ID-scoped server-step attempt."""

    attempt_directory: Path
    prepared: Path
    shutdown_request: Path
    terminal: Path


@dataclass(frozen=True)
class PreparedServerStep:
    """Immutable endpoint/log evidence published before the local launcher starts."""

    config_fingerprint: str
    endpoint: str
    health_endpoint: str
    server_log_path: str


@dataclass(frozen=True)
class TerminalServerStep:
    """Immutable final launcher status published after bounded cleanup."""

    config_fingerprint: str
    status: str
    process_returncode: int | None


def _require_fingerprint(value: object) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise SplitServerStepControlError("config fingerprint must be lowercase 64-hex")
    return value


def _require_nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise SplitServerStepControlError(f"{label} must be a nonempty string")
    return value


def _shared_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        mode = path.stat().st_mode
    except (OSError, RuntimeError) as error:
        raise SplitServerStepControlError(f"{label} is unavailable") from error
    if (
        resolved != path
        or not stat.S_ISDIR(mode)
        or mode & _CONTROL_REQUIRED_MODE != _CONTROL_REQUIRED_MODE
        or not os.access(path, os.R_OK | os.W_OK | os.X_OK)
    ):
        raise SplitServerStepControlError(f"{label} must be a real shared writable directory")
    return resolved


def _control_paths(attempt: Path) -> SplitServerStepControlPaths:
    return SplitServerStepControlPaths(
        attempt,
        attempt / PREPARED_FILENAME,
        attempt / SHUTDOWN_REQUEST_FILENAME,
        attempt / TERMINAL_FILENAME,
    )


def create_split_server_step_control_paths(
    run_directory: Path,
    job_id: str,
) -> SplitServerStepControlPaths:
    """Create fixed paths for one new shared job-ID attempt.

    Only the control root and new attempt directory are created. Invalid paths,
    job IDs, ownership modes, or duplicate attempts raise
    :class:`SplitServerStepControlError`.
    """
    run_directory = _shared_directory(run_directory, "run directory")
    if type(job_id) is not str or SLURM_JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise SplitServerStepControlError("job ID must be a positive decimal integer")
    control_root = run_directory / CONTROL_DIRECTORY_NAME
    try:
        control_root.mkdir(mode=0o750)
        created_root = True
    except FileExistsError:
        created_root = False
    except OSError as error:
        raise SplitServerStepControlError("cannot create server-step control root") from error
    if created_root:
        os.chmod(control_root, 0o750)
    _shared_directory(control_root, "server-step control root")
    attempt = control_root / job_id
    try:
        attempt.mkdir(mode=0o750)
        os.chmod(attempt, 0o750)
    except OSError as error:
        raise SplitServerStepControlError("server-step attempt already exists or failed") from error
    return _control_paths(_shared_directory(attempt, "server-step attempt"))


def resolve_split_server_step_control_paths(
    run_directory: Path,
    job_id: str,
) -> SplitServerStepControlPaths:
    """Return one existing real attempt's fixed paths without filesystem mutation."""
    run_directory = _shared_directory(run_directory, "run directory")
    if type(job_id) is not str or SLURM_JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise SplitServerStepControlError("job ID must be a positive decimal integer")
    attempt = run_directory / CONTROL_DIRECTORY_NAME / job_id
    return _control_paths(_shared_directory(attempt, "server-step attempt"))


def _validate_control_paths(paths: SplitServerStepControlPaths) -> SplitServerStepControlPaths:
    if type(paths) is not SplitServerStepControlPaths:
        raise SplitServerStepControlError("control paths have the wrong record type")
    attempt = _shared_directory(paths.attempt_directory, "server-step attempt")
    if paths != _control_paths(attempt):
        raise SplitServerStepControlError("control paths do not use the fixed filenames")
    return paths


def _write_once(destination: Path, payload: dict[str, object]) -> Path:
    try:
        return write_json_once(destination, payload)
    except JsonArtifactError as error:
        raise SplitServerStepControlError(
            f"cannot create immutable control artifact: {destination.name}"
        ) from error


def _read_exact(path: Path, expected_keys: set[str]) -> dict[str, object]:
    try:
        return read_exact_json(path, expected_keys)
    except JsonArtifactError as error:
        raise SplitServerStepControlError(
            f"control artifact is unavailable or malformed: {path.name}"
        ) from error


def _validated_prepared(value: object) -> PreparedServerStep:
    if type(value) is not PreparedServerStep:
        raise SplitServerStepControlError("prepared state has the wrong record type")
    _require_fingerprint(value.config_fingerprint)
    for field, label in zip(
        (value.endpoint, value.health_endpoint, value.server_log_path),
        ("endpoint", "health endpoint", "server log path"),
        strict=True,
    ):
        _require_nonempty_string(field, label)
    return value


def _validated_terminal(value: object) -> TerminalServerStep:
    if type(value) is not TerminalServerStep:
        raise SplitServerStepControlError("terminal state has the wrong record type")
    _require_fingerprint(value.config_fingerprint)
    if value.status not in {"clean", "failed"}:
        raise SplitServerStepControlError("terminal status must be clean or failed")
    if value.process_returncode is not None and type(value.process_returncode) is not int:
        raise SplitServerStepControlError("process return code must be an integer or null")
    return value


def write_prepared_server_step(
    paths: SplitServerStepControlPaths,
    prepared: PreparedServerStep,
) -> Path:
    """Validate and write immutable prepared state, returning its artifact path."""
    _validate_control_paths(paths)
    _validated_prepared(prepared)
    return _write_once(paths.prepared, {"schema_version": 1, **asdict(prepared)})


def read_prepared_server_step(paths: SplitServerStepControlPaths) -> PreparedServerStep:
    """Read and validate immutable prepared state without mutating its artifact."""
    _validate_control_paths(paths)
    payload = _read_exact(
        paths.prepared,
        {"schema_version", "config_fingerprint", "endpoint", "health_endpoint", "server_log_path"},
    )
    if payload.pop("schema_version") != 1:
        raise SplitServerStepControlError("unsupported prepared-state schema")
    return _validated_prepared(PreparedServerStep(**payload))  # type: ignore[arg-type]


def write_server_step_shutdown_request(
    paths: SplitServerStepControlPaths,
    config_fingerprint: str,
) -> Path:
    """Write one immutable fingerprint-bound shutdown request and return its path."""
    _validate_control_paths(paths)
    _require_fingerprint(config_fingerprint)
    return _write_once(
        paths.shutdown_request,
        {"config_fingerprint": config_fingerprint, "schema_version": 1},
    )


def read_server_step_shutdown_request(paths: SplitServerStepControlPaths) -> str:
    """Return a validated shutdown fingerprint without mutating its artifact."""
    _validate_control_paths(paths)
    payload = _read_exact(paths.shutdown_request, {"schema_version", "config_fingerprint"})
    if payload["schema_version"] != 1:
        raise SplitServerStepControlError("unsupported shutdown-request schema")
    return _require_fingerprint(payload["config_fingerprint"])


def write_terminal_server_step(
    paths: SplitServerStepControlPaths,
    terminal: TerminalServerStep,
) -> Path:
    """Validate and write immutable terminal state, returning its artifact path."""
    _validate_control_paths(paths)
    _validated_terminal(terminal)
    return _write_once(paths.terminal, {"schema_version": 1, **asdict(terminal)})


def read_terminal_server_step(paths: SplitServerStepControlPaths) -> TerminalServerStep:
    """Read and validate immutable terminal state without mutating its artifact."""
    _validate_control_paths(paths)
    payload = _read_exact(
        paths.terminal,
        {"schema_version", "config_fingerprint", "status", "process_returncode"},
    )
    if payload.pop("schema_version") != 1:
        raise SplitServerStepControlError("unsupported terminal-state schema")
    return _validated_terminal(TerminalServerStep(**payload))  # type: ignore[arg-type]


class SplitServerStepWaitError(RuntimeError):
    """Raised on timeout, early server exit, or cross-step identity mismatch."""


Clock = Callable[[], float]
Sleeper = Callable[[float], None]
ProcessPoll = Callable[[], int | None]


def _timeout_values(timeout_seconds: float, poll_seconds: float) -> tuple[float, float]:
    values = (timeout_seconds, poll_seconds)
    if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
        raise SplitServerStepWaitError("wait intervals must be finite numbers")
    if timeout_seconds <= 0 or poll_seconds <= 0 or poll_seconds > timeout_seconds:
        raise SplitServerStepWaitError("wait intervals must be positive and bounded")
    return float(timeout_seconds), float(poll_seconds)


def wait_for_prepared_server_step(
    run_directory: Path,
    job_id: str,
    expected_fingerprint: str,
    timeout_seconds: float,
    *,
    poll_seconds: float = 1.0,
    monotonic: Clock = time.monotonic,
    sleeper: Sleeper = time.sleep,
) -> tuple[SplitServerStepControlPaths, PreparedServerStep]:
    """Wait for matching prepared state without mutating cross-step artifacts."""
    timeout_seconds, poll_seconds = _timeout_values(timeout_seconds, poll_seconds)
    deadline = monotonic() + timeout_seconds
    attempt = run_directory / CONTROL_DIRECTORY_NAME / job_id
    prepared_path = attempt / PREPARED_FILENAME
    terminal_path = attempt / TERMINAL_FILENAME
    while True:
        if prepared_path.exists():
            paths = resolve_split_server_step_control_paths(run_directory, job_id)
            prepared = read_prepared_server_step(paths)
            if prepared.config_fingerprint != expected_fingerprint:
                raise SplitServerStepWaitError("prepared-state fingerprint mismatch")
            return paths, prepared
        if terminal_path.exists():
            paths = resolve_split_server_step_control_paths(run_directory, job_id)
            terminal = read_terminal_server_step(paths)
            if terminal.config_fingerprint != expected_fingerprint:
                raise SplitServerStepWaitError("terminal-state fingerprint mismatch")
            raise SplitServerStepWaitError(
                "server step failed before prepared state: "
                f"status={terminal.status} returncode={terminal.process_returncode}"
            )
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise SplitServerStepWaitError("timed out waiting for server prepared state")
        sleeper(min(poll_seconds, remaining))


def wait_for_server_step_shutdown_request(
    paths: SplitServerStepControlPaths,
    expected_fingerprint: str,
    process_poll: ProcessPoll,
    *,
    poll_seconds: float = 5.0,
    sleeper: Sleeper = time.sleep,
) -> None:
    """Wait for matching client shutdown or fail on direct launcher exit."""
    if type(poll_seconds) not in (int, float) or not math.isfinite(poll_seconds):
        raise SplitServerStepWaitError("poll interval must be a finite number")
    if poll_seconds <= 0:
        raise SplitServerStepWaitError("poll interval must be positive")
    while True:
        if paths.shutdown_request.exists():
            observed = read_server_step_shutdown_request(paths)
            if observed != expected_fingerprint:
                raise SplitServerStepWaitError("shutdown-request fingerprint mismatch")
            return
        returncode = process_poll()
        if returncode is not None:
            raise SplitServerStepWaitError(
                f"server process exited before shutdown request: {returncode}"
            )
        sleeper(float(poll_seconds))


def wait_for_terminal_server_step(
    paths: SplitServerStepControlPaths,
    expected_fingerprint: str,
    timeout_seconds: float,
    *,
    poll_seconds: float = 1.0,
    monotonic: Clock = time.monotonic,
    sleeper: Sleeper = time.sleep,
) -> TerminalServerStep:
    """Wait for matching terminal cleanup state without mutating its artifact."""
    timeout_seconds, poll_seconds = _timeout_values(timeout_seconds, poll_seconds)
    deadline = monotonic() + timeout_seconds
    while True:
        if paths.terminal.exists():
            terminal = read_terminal_server_step(paths)
            if terminal.config_fingerprint != expected_fingerprint:
                raise SplitServerStepWaitError("terminal-state fingerprint mismatch")
            return terminal
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise SplitServerStepWaitError("timed out waiting for server cleanup")
        sleeper(min(poll_seconds, remaining))


def external_server_process_poll(
    paths: SplitServerStepControlPaths,
    expected_fingerprint: str,
) -> int | None:
    """Adapt terminal evidence to the health-check process-poll callback."""
    if not paths.terminal.exists():
        return None
    terminal = read_terminal_server_step(paths)
    if terminal.config_fingerprint != expected_fingerprint:
        raise SplitServerStepWaitError("terminal-state fingerprint mismatch")
    if terminal.process_returncode is not None:
        return terminal.process_returncode
    return 0 if terminal.status == "clean" else 1


class SplitMegatronServerStepError(RuntimeError):
    """Raised for an invalid request, failed lifecycle, or terminal-record failure."""


@dataclass(frozen=True)
class SplitMegatronServerStepResult:
    """Durable evidence returned only after a clean server-step lifecycle.

    Paths identify the prepared manifest, prepared and terminal control records,
    and server log. ``process_returncode`` is the reaped torchrun status, which
    may be ``None`` only when the supplied process implementation returns it.
    """

    prepared_manifest: Path
    prepared_control_artifact: Path
    terminal_control_artifact: Path
    server_log_artifact: Path
    process_returncode: int | None


@dataclass(frozen=True)
class SplitMegatronServerStepDependencies:
    """Injectable side-effect boundary used by orchestration tests.

    The callbacks validate allocation shape, create a process exposing
    ``start``, ``poll``, ``forward_signal``, and ``terminate``, and wait for the
    matching client shutdown request.
    """

    allocation_validator: Callable[[ResolvedEvalRun], EnclosingAllocation]
    process_factory: Callable[[MegatronDynamicServerLaunch], object]
    shutdown_waiter: Callable[..., None]


def production_split_server_step_dependencies() -> SplitMegatronServerStepDependencies:
    """Return the allocation, process, and shutdown production adapters."""

    return SplitMegatronServerStepDependencies(
        validate_enclosing_allocation,
        LocalTorchrunServerProcess,
        wait_for_server_step_shutdown_request,
    )


def _attempt_launch(
    launch: MegatronDynamicServerLaunch,
    paths: SplitServerStepControlPaths,
) -> MegatronDynamicServerLaunch:
    """Bind this Slurm attempt's contained log path without mutating the input."""

    log_path = paths.attempt_directory / "megatron_dynamic_server.log"
    return replace(launch, server_log_path=str(log_path))


def execute_split_megatron_server_step(
    prepared: PreparedMegatronDynamicServerVlmFewshotRun,
    *,
    dependencies: SplitMegatronServerStepDependencies | None = None,
) -> SplitMegatronServerStepResult:
    """Run one prepared server until its matching client requests shutdown.

    ``prepared`` must contain one resolved run and immutable launch description.
    Optional ``dependencies`` replace the production I/O boundary as one exact
    dependency record. Before process startup, the function validates
    allocation identity, creates the run directory, and writes the prepared
    manifest and control record.

    After prepared state exists, process construction, startup, signal
    forwarding, shutdown waiting, termination, and terminal recording form one
    guarded lifecycle. The child is always reaped if it was constructed.
    Success returns durable artifact paths without transferring ownership of
    mutable state. Execution failures raise ``SplitMegatronServerStepError``
    after a failed terminal record is attempted; earlier precondition errors
    propagate before any terminal claim is made.
    """
    # Validate caller-owned records and external identity before filesystem
    # mutation or process construction.
    if type(prepared) is not PreparedMegatronDynamicServerVlmFewshotRun:
        raise SplitMegatronServerStepError("prepared run has the wrong record type")
    resolved = prepared.resolved
    deps = production_split_server_step_dependencies() if dependencies is None else dependencies
    if type(deps) is not SplitMegatronServerStepDependencies:
        raise SplitMegatronServerStepError("server-step dependencies have the wrong type")
    allocation = deps.allocation_validator(resolved)

    # Publish the shared run identity and attempt-specific endpoints/log path.
    # The client waits for this prepared record before contacting the server.
    directory = prepare_vlmeval_run_directory(resolved)
    manifest = write_resolved_run_manifest(resolved, slurm_job_id=allocation.job_id)
    fingerprint = canonical_config_fingerprint(resolved)
    paths = create_split_server_step_control_paths(directory.path, allocation.job_id)
    launch = _attempt_launch(prepared.launch, paths)
    prepared_artifact = write_prepared_server_step(
        paths,
        PreparedServerStep(
            fingerprint,
            launch.endpoint,
            launch.health_endpoint,
            launch.server_log_path,
        ),
    )

    # From process construction onward, every exit flows through termination
    # and a terminal cleanup record.
    process: object | None = None
    process_returncode: int | None = None
    error: Exception | None = None
    try:
        process = deps.process_factory(launch)
        process.start()  # type: ignore[attr-defined]
        with raise_on_termination_signal(process.forward_signal):  # type: ignore[attr-defined]
            deps.shutdown_waiter(paths, fingerprint, process.poll)  # type: ignore[attr-defined]
    except Exception as server_error:
        error = server_error
    finally:
        # Reap the torchrun launcher before claiming clean or failed terminal
        # state. Torchrun owns rank cleanup; Megatron owns frontend cleanup. A
        # terminal-write failure is itself a server-step failure.
        if process is not None:
            try:
                process_returncode = process.terminate()  # type: ignore[attr-defined]
            except Exception as cleanup_error:
                if error is None:
                    error = cleanup_error
        terminal = TerminalServerStep(
            fingerprint,
            "clean" if error is None else "failed",
            process_returncode,
        )
        try:
            terminal_artifact = write_terminal_server_step(paths, terminal)
        except Exception as terminal_error:
            raise SplitMegatronServerStepError(
                "server step could not record terminal cleanup state"
            ) from terminal_error

    if error is not None:
        raise SplitMegatronServerStepError("server step failed") from error

    # Return only immutable paths and the final reaped process status.
    return SplitMegatronServerStepResult(
        manifest,
        prepared_artifact,
        terminal_artifact,
        Path(launch.server_log_path),
        process_returncode,
    )
