"""Validate and render a deterministic local-torchrun server launch.
Construction is side-effect free: this module starts no process and writes no artifact.
"""

from __future__ import annotations

import math
import os
import re
import signal
import stat
import subprocess
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from ..configured_eval_run import ResolvedEvalRun

_ARGUMENT_NAME = re.compile(r"[a-z][a-z0-9-]*")
_LOOPBACK_HOST = "127.0.0.1"
_INFERENCE_CONTEXT_ARGUMENT = "inference-max-seq-length"
_DYNAMIC_BATCH_TOKEN_BUDGET_ARGUMENT = "inference-dynamic-batching-max-tokens"
_MINIMUM_DYNAMIC_BATCH_TOKEN_BUDGET = 16384
_OWNED_SERVER_ARGUMENTS = frozenset(
    (
        "load",
        "ckpt-step",
        "host",
        "port",
        "inference-coordinator-port",
        _INFERENCE_CONTEXT_ARGUMENT,
        _DYNAMIC_BATCH_TOKEN_BUDGET_ARGUMENT,
    )
)


class MegatronDynamicServerLaunchError(ValueError):
    """Raised when a launch cannot be constructed from the resolved contract."""


@dataclass(frozen=True)
class MegatronDynamicServerLaunch:
    """Immutable process, endpoint, environment, and logging contract.

    ``argv`` and ``working_directory`` are consumed by the supervisor.
    ``environment_overrides`` owns only integration values; others remain
    inherited. Endpoints use the selected local or rendezvous host, and
    ``server_log_path`` is the sole durable process-output destination.
    """
    argv: tuple[str, ...]
    working_directory: str
    environment_overrides: tuple[tuple[str, str], ...]
    endpoint: str
    health_endpoint: str
    server_log_path: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MegatronDynamicServerLaunchError(message)


def _absolute(value: object, label: str) -> Path:
    valid = type(value) is str and bool(value) and "\0" not in value
    _require(valid, f"{label} must be a nonempty string")
    path = Path(value)
    _require(path.is_absolute(), f"{label} must be absolute")
    _require(str(path) == value and ".." not in path.parts, f"{label} must be normalized")
    return path


def _validate_topology(resolved: ResolvedEvalRun) -> int:
    config = resolved.configured
    _require(
        config.run.launch_mode in {"local_torchrun", "slurm_multinode_torchrun"},
        "unsupported launch mode",
    )
    expected_nodes = (
        config.run.nodes == 1
        if config.run.launch_mode == "local_torchrun"
        else config.run.nodes > 1
    )
    _require(type(config.run.nodes) is int and expected_nodes, "invalid node topology")
    workers = config.run.gpus_per_node
    _require(type(workers) is int and workers > 0, "invalid GPU topology")
    expert_parallel = config.run.expert_model_parallel_size
    _require(
        type(expert_parallel) is int
        and expert_parallel > 0
        and config.run.nodes * workers % expert_parallel == 0,
        "invalid expert parallel topology",
    )
    return workers


def _validate_paths(
    resolved: ResolvedEvalRun,
    run_directory: object,
    python_executable: object,
    compiler_cache_directory: object,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    paths = resolved.paths
    runtime_root = _absolute(paths.runtime_root, "runtime root")
    entrypoint = _absolute(paths.runtime_entrypoint, "runtime entrypoint")
    checkpoint = _absolute(paths.checkpoint_root, "checkpoint root")
    output_root = _absolute(paths.output_root, "output root")
    _absolute(paths.run_directory, "resolved run directory")
    _require(isinstance(run_directory, (str, Path)), "run directory must be a path")
    requested_run = _absolute(str(run_directory), "run directory")
    interpreter = _absolute(python_executable, "Python executable")
    cache = _absolute(str(compiler_cache_directory), "compiler cache directory")
    _require(str(requested_run) == paths.run_directory, "run directory differs from resolved path")
    _require(requested_run != output_root, "run directory must be below the output root")
    _require(requested_run.is_relative_to(output_root), "run directory escapes the output root")
    _require(entrypoint.is_relative_to(runtime_root), "runtime entrypoint escapes the runtime root")
    if resolved.configured.checkpoint.trusted_standalone_common_state:
        common_state = Path(paths.checkpoint_metadata).parent / "common.pt"
        _require(
            common_state.is_file() and os.access(common_state, os.R_OK),
            "trusted legacy common state requires a readable standalone common.pt",
        )
    try:
        local_tmp = Path("/tmp").resolve(strict=True)
        resolved_cache = cache.resolve(strict=True)
        cache_mode = resolved_cache.stat().st_mode
    except OSError as error:
        raise MegatronDynamicServerLaunchError("compiler cache is unavailable") from error
    _require(resolved_cache == cache, "compiler cache directory must not be a symlink")
    _require(
        resolved_cache != local_tmp and resolved_cache.is_relative_to(local_tmp),
        "compiler cache directory must be a unique directory below /tmp",
    )
    _require(stat.S_ISDIR(cache_mode), "compiler cache path must be a directory")
    _require(os.access(resolved_cache, os.R_OK | os.W_OK | os.X_OK), "compiler cache is unusable")
    return runtime_root, entrypoint, checkpoint, requested_run, interpreter, resolved_cache


def _render_override(name: object, value: object) -> tuple[str, ...]:
    _require(type(name) is str and _ARGUMENT_NAME.fullmatch(name) is not None, "invalid override")
    _require(name not in _OWNED_SERVER_ARGUMENTS, f"override duplicates launcher argument: {name}")
    flag = f"--{name}"
    if type(value) is bool:
        _require(value, f"false boolean override is unsupported: {name}")
        return (flag,)
    valid_scalar = type(value) in (str, int, float)
    _require(valid_scalar, f"override has unsupported value type: {name}")
    if type(value) is float:
        _require(math.isfinite(value), f"override has non-finite value: {name}")
    rendered = str(value)
    _require(bool(rendered) and "\0" not in rendered, f"override has an empty value: {name}")
    return (flag, rendered)


def _context_bindings(context_length: int | None) -> tuple[tuple[str, str], ...]:
    """Bind an explicit request limit and a compatible aggregate token budget."""

    if context_length is None:
        return ()
    batch_token_budget = max(_MINIMUM_DYNAMIC_BATCH_TOKEN_BUDGET, context_length)
    return (
        (_INFERENCE_CONTEXT_ARGUMENT, str(context_length)),
        (_DYNAMIC_BATCH_TOKEN_BUDGET_ARGUMENT, str(batch_token_budget)),
    )


def _resolved_overrides(resolved: ResolvedEvalRun) -> tuple[str, ...]:
    overrides = resolved.resolved_cli_overrides
    _require(type(overrides) is tuple, "resolved CLI overrides must be a tuple")
    names: list[str] = []
    rendered: list[str] = []
    for item in overrides:
        _require(type(item) is tuple and len(item) == 2, "malformed resolved CLI override")
        name, value = item
        rendered.extend(_render_override(name, value))
        names.append(name)
    _require(len(names) == len(set(names)), "duplicate resolved CLI override")
    return tuple(rendered)


def _environment_overrides(
    runtime_root: Path,
    compiler_cache_directory: Path,
    overrides: tuple[tuple[str, object], ...],
    trusted_standalone_common_state: bool,
) -> tuple[tuple[str, str], ...]:
    """Return integration-owned Python, cache, logging, and deterministic values.

    Megatron-owned inference defaults remain absent.
    """

    environment = [
        ("PYTHONPATH", str(runtime_root)),
        ("PYTHONNOUSERSITE", "1"),
        ("PYTHONDONTWRITEBYTECODE", "1"),
        ("PYTHONUNBUFFERED", "1"),
        ("WANDB_MODE", "disabled"),
        ("TRITON_CACHE_DIR", str(compiler_cache_directory / "triton")),
        ("TORCHINDUCTOR_CACHE_DIR", str(compiler_cache_directory / "torchinductor")),
        (
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
            "1" if trusted_standalone_common_state else "0",
        ),
    ]
    if trusted_standalone_common_state:
        environment.append(("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0"))
    if ("deterministic-mode", True) in overrides:
        environment.extend(
            (
                ("NCCL_ALGO", "Ring"),
                ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
                ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0"),
            )
        )
    return tuple(environment)


def build_megatron_dynamic_server_launch(
    resolved: ResolvedEvalRun,
    run_directory: str | Path,
    *,
    host: str,
    port: int,
    coordinator_port: int,
    python_executable: str,
    compiler_cache_directory: str | Path,
    node_rank: int | None = None,
    master_addr: str | None = None,
    rendezvous_port: int | None = None,
) -> MegatronDynamicServerLaunch:
    """Build a launch record from one fully resolved evaluation.

    ``resolved`` supplies checkpoint, runtime, topology, context, and model
    overrides. ``run_directory`` must equal its resolved destination. The HTTP
    endpoint is loopback locally and the rendezvous host in multi-node mode.
    Ports must be distinct and unprivileged. The interpreter must be absolute
    and normalized; the cache must be a real directory below node-local
    ``/tmp``.

    Returns an immutable, fully rendered launch. It starts no process and
    changes no filesystem state. Raises ``MegatronDynamicServerLaunchError``
    for any inconsistent input.
    """
    _require(type(resolved) is ResolvedEvalRun, "resolved must be an exact ResolvedEvalRun")
    workers = _validate_topology(resolved)
    run = resolved.configured.run
    if run.launch_mode == "local_torchrun":
        _require(host == _LOOPBACK_HOST and type(host) is str, "host must be 127.0.0.1")
    else:
        _require(host == master_addr, "multi-node endpoint must use the rendezvous host")
    ports = (port, coordinator_port)
    valid_ports = all(type(value) is int and 1024 <= value <= 65535 for value in ports)
    _require(valid_ports and port != coordinator_port, "ports must be distinct integers")
    runtime, entrypoint, checkpoint, run_path, interpreter, cache = _validate_paths(
        resolved, run_directory, python_executable, compiler_cache_directory
    )
    context_length = resolved.configured.model.server.context_length
    _require(
        context_length is None or (type(context_length) is int and context_length > 0),
        "invalid context length",
    )
    iteration = resolved.configured.checkpoint.iteration
    _require(type(iteration) is int and iteration > 0, "invalid checkpoint iteration")
    if run.launch_mode == "local_torchrun":
        _require(
            node_rank is None and master_addr is None and rendezvous_port is None,
            "local launch cannot configure distributed rendezvous",
        )
        torchrun = ("--standalone", "--nnodes", "1", "--nproc-per-node", str(workers))
    else:
        _require(
            type(node_rank) is int and 0 <= node_rank < run.nodes,
            "node rank must be within the configured node count",
        )
        _require(
            type(master_addr) is str
            and bool(master_addr)
            and master_addr not in {_LOOPBACK_HOST, "localhost", "::1"}
            and not any(character.isspace() for character in master_addr)
            and "\0" not in master_addr,
            "master address must be a nonempty hostname or address",
        )
        _require(
            type(rendezvous_port) is int
            and 1024 <= rendezvous_port <= 65535
            and rendezvous_port not in ports,
            "rendezvous port must be a third distinct unprivileged port",
        )
        torchrun = (
            "--nnodes",
            str(run.nodes),
            "--nproc-per-node",
            str(workers),
            "--node-rank",
            str(node_rank),
            "--master-addr",
            master_addr,
            "--master-port",
            str(rendezvous_port),
        )
    argv = (
        str(interpreter),
        "-u",
        "-m",
        "torch.distributed.run",
        *torchrun,
        "--max-restarts",
        "0",
        str(entrypoint),
    )
    context_bindings = _context_bindings(context_length)
    bindings = (
        ("load", str(checkpoint)),
        ("ckpt-step", str(iteration)),
        ("port", str(port)),
        ("inference-coordinator-port", str(coordinator_port)),
        *context_bindings,
    )
    if run.launch_mode == "local_torchrun":
        bindings = bindings[:2] + (("host", host),) + bindings[2:]
    argv += tuple(token for name, value in bindings for token in (f"--{name}", value))
    argv += _resolved_overrides(resolved)
    environment = _environment_overrides(
        runtime,
        cache,
        resolved.resolved_cli_overrides,
        resolved.configured.checkpoint.trusted_standalone_common_state,
    )
    return MegatronDynamicServerLaunch(
        argv=argv,
        working_directory=str(runtime),
        environment_overrides=environment,
        endpoint=f"http://{host}:{port}/v1/chat/completions",
        health_endpoint=f"http://{host}:{port}/v1/health",
        server_log_path=str(run_path / "megatron_dynamic_server.log"),
    )


def render_megatron_dynamic_server_launch(
    launch: MegatronDynamicServerLaunch,
) -> dict[str, object]:
    """Return a detached JSON-safe view of an exact launch record.

    Mutating the returned lists or dictionaries cannot change ``launch``.
    Invalid record types raise ``MegatronDynamicServerLaunchError``.
    """
    _require(type(launch) is MegatronDynamicServerLaunch, "invalid launch record")
    return {
        "argv": list(launch.argv),
        "working_directory": launch.working_directory,
        "environment_overrides": [
            {"name": name, "value": value} for name, value in launch.environment_overrides
        ],
        "endpoint": launch.endpoint,
        "health_endpoint": launch.health_endpoint,
        "server_log_path": launch.server_log_path,
    }


_REMOVED_ENVIRONMENT_NAMES = frozenset({"PYTHONHOME", "PYTHONPATH"})


class LocalTorchrunServerProcessError(RuntimeError):
    """Raised when local server process lifecycle management fails."""


def _sanitized_environment(
    launch: MegatronDynamicServerLaunch,
    source: Mapping[str, str],
) -> dict[str, str]:
    """Return the owned child environment without mutating ``source``.

    Only Python import-root variables are removed. Launch overrides are applied
    once; malformed input or duplicate overrides raise
    :class:`LocalTorchrunServerProcessError`.
    """
    environment: dict[str, str] = {}
    for name, value in source.items():
        if type(name) is not str or type(value) is not str or "\0" in name + value:
            raise LocalTorchrunServerProcessError("base environment must contain strings")
        if name not in _REMOVED_ENVIRONMENT_NAMES:
            environment[name] = value
    override_names: set[str] = set()
    for pair in launch.environment_overrides:
        if type(pair) is not tuple or len(pair) != 2:
            raise LocalTorchrunServerProcessError("malformed environment override")
        name, value = pair
        if type(name) is not str or type(value) is not str or not name or "\0" in name + value:
            raise LocalTorchrunServerProcessError("invalid environment override")
        if name in override_names:
            raise LocalTorchrunServerProcessError("duplicate environment override")
        override_names.add(name)
        environment[name] = value
    return environment


class LocalTorchrunServerProcess:
    """Single-use observer of one torchrun launcher child and its log.

    ``launch`` is an exact immutable launch record. Construction only snapshots
    a sanitized environment; ``start`` creates the direct child and its mode-
    ``0640`` log. This supervisor never manages torchrun descendants. Lifecycle
    contract failures raise :class:`LocalTorchrunServerProcessError`.
    """

    def __init__(
        self,
        launch: MegatronDynamicServerLaunch,
        *,
        terminate_timeout_seconds: float = 10.0,
        kill_timeout_seconds: float = 5.0,
        base_environment: Mapping[str, str] | None = None,
    ) -> None:
        if type(launch) is not MegatronDynamicServerLaunch:
            raise LocalTorchrunServerProcessError("launch must be an exact launch record")
        timeouts = (terminate_timeout_seconds, kill_timeout_seconds)
        valid_timeouts = all(
            type(value) in (int, float) and math.isfinite(value) and value > 0
            for value in timeouts
        )
        if not valid_timeouts:
            raise LocalTorchrunServerProcessError("cleanup timeouts must be finite and positive")
        self.launch = launch
        self.terminate_timeout_seconds = float(terminate_timeout_seconds)
        self.kill_timeout_seconds = float(kill_timeout_seconds)
        source = os.environ if base_environment is None else base_environment
        if not isinstance(source, Mapping):
            raise LocalTorchrunServerProcessError("base environment must be a mapping")
        self._environment = _sanitized_environment(launch, source)
        self._process: subprocess.Popen[bytes] | None = None
        self._log_stream: BinaryIO | None = None
        self._started = False

    def _open_log(self) -> BinaryIO:
        """Create or truncate the sole mode-0640 regular server log."""
        path = Path(self.launch.server_log_path)
        if not path.is_absolute() or not path.parent.is_dir():
            raise LocalTorchrunServerProcessError("server log parent must be an existing directory")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        try:
            descriptor = os.open(path, flags, 0o640)
            os.fchmod(descriptor, 0o640)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("server log is not a regular file")
            return os.fdopen(descriptor, "wb")
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
            raise LocalTorchrunServerProcessError("server log cannot be opened") from error

    def _close_log(self) -> None:
        """Close the owned log stream if it is open."""
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def start(self) -> None:
        """Start the torchrun child once and redirect output to its owned log.

        The child receives a copy of the sanitized environment in the launch
        directory. Repeated starts or child-creation failures raise
        :class:`LocalTorchrunServerProcessError`.
        """
        if self._started:
            raise LocalTorchrunServerProcessError("server process can only be started once")
        self._started = True
        self._log_stream = self._open_log()
        try:
            self._process = subprocess.Popen(
                self.launch.argv,
                cwd=self.launch.working_directory,
                env=self._environment.copy(),
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
            )
        except Exception as error:
            self._close_log()
            raise LocalTorchrunServerProcessError("server process failed to start") from error

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise LocalTorchrunServerProcessError("server process has not started")
        return self._process

    def poll(self) -> int | None:
        """Return the child status and close its log after child exit."""
        process = self._require_process()
        result = process.poll()
        if result is not None:
            self._close_log()
        return result

    @property
    def returncode(self) -> int | None:
        """Return ``None`` before or during execution, else the child exit code."""
        return None if self._process is None else self.poll()

    def forward_signal(self, signum: int | signal.Signals) -> None:
        """Forward SIGINT/SIGTERM to the direct child without signaling descendants."""
        if type(signum) not in (int, signal.Signals) or signum not in (
            signal.SIGINT,
            signal.SIGTERM,
        ):
            raise LocalTorchrunServerProcessError("only SIGINT and SIGTERM can be forwarded")
        process = self._require_process()
        if process.poll() is None:
            process.send_signal(signum)
        else:
            self._close_log()

    def terminate(self) -> int | None:
        """Reap the child with bounded TERM-then-KILL cleanup.

        Returns the final child status or ``None`` before child creation. Calls
        after reaping are safe. Forced or failed cleanup raises
        :class:`LocalTorchrunServerProcessError`; the owned log always closes.
        """
        if self._process is None:
            self._close_log()
            return None
        process = self._process
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    return process.wait(timeout=self.terminate_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=self.kill_timeout_seconds)
                    except subprocess.TimeoutExpired as error:
                        raise LocalTorchrunServerProcessError(
                            "server process could not be reaped"
                        ) from error
                    raise LocalTorchrunServerProcessError(
                        "torchrun launcher required forced cleanup"
                    )
            try:
                return process.wait(timeout=self.kill_timeout_seconds)
            except subprocess.TimeoutExpired as error:
                raise LocalTorchrunServerProcessError(
                    "server process could not be reaped"
                ) from error
        finally:
            self._close_log()

    def __enter__(self) -> "LocalTorchrunServerProcess":
        """Start the child and return this supervisor."""
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Terminate the child and never suppress the enclosing exception."""
        self.terminate()


class TerminationSignalReceived(RuntimeError):
    """Raised after an optional direct-child signal forward."""


@contextmanager
def raise_on_termination_signal(
    forward: Callable[[int], None] | None = None,
) -> Iterator[None]:
    """Temporarily raise on SIGINT/SIGTERM and optionally forward to one child.

    Existing signal handlers are restored on every exit. The process signal
    handlers are the only mutation; received signals raise
    :class:`TerminationSignalReceived`.
    """
    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def handle(signum: int, _frame: object) -> None:
        if forward is not None:
            forward(signum)
        raise TerminationSignalReceived(f"received signal {signum}")

    for signum in previous:
        signal.signal(signum, handle)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
