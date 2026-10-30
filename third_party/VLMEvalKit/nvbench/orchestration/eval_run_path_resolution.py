"""Resolve and validate the configured filesystem inputs for an evaluation."""

from __future__ import annotations
import os
import re
import stat
from dataclasses import dataclass, replace
from pathlib import Path

from .benchmark_array_selection import (
    benchmark_array_run_component,
    bind_benchmark_array_selection,
    common_generation_config,
    configured_benchmarks,
)
from .chat_template_policy import configured_chat_template_path
from .configured_eval_run import (
    ConditionalPerplexityBenchmarkConfig,
    ConfiguredEvalRun,
    EvalConfigError,
    ResolvedEvalRun,
    ResolvedPaths,
    bind_resolved_cli_paths,
    validate_configured_eval_run,
)
from .eval_run_config_loading import load_configured_eval_run
from .megatron_server_process_management.megatron_dynamic_server_launch import (
    MegatronDynamicServerLaunch,
    build_megatron_dynamic_server_launch,
)

_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")


class VLMEvalRunDirectoryPreparationError(RuntimeError):
    """Raised when resolved output directories cannot be created safely."""


@dataclass(frozen=True)
class PreparedVLMEvalRunDirectory:
    """One prepared run directory and whether this call created it."""

    path: Path
    is_new: bool


@dataclass(frozen=True)
class PreparedMegatronDynamicServerVlmFewshotRun:
    """One resolved evaluation and its side-effect-free server launch."""

    resolved: ResolvedEvalRun
    launch: MegatronDynamicServerLaunch


def _expand_workspace(configured: str, workspace: Path, label: str) -> str:
    placeholders = set(_PLACEHOLDER.findall(configured))
    if unsupported := sorted(placeholders - {"workspace"}):
        raise EvalConfigError(f"{label} uses unsupported placeholders: {unsupported}")
    expanded = configured.replace("${workspace}", str(workspace))
    if "$" in expanded:
        raise EvalConfigError(f"{label} contains an unresolved placeholder")
    return expanded


def _require_group_directory(path: Path, label: str) -> None:
    if not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
        raise EvalConfigError(f"{label} is not a readable, traversable directory: {path}")
    mode = path.stat().st_mode
    if not mode & stat.S_IRGRP or not mode & stat.S_IXGRP:
        raise EvalConfigError(f"{label} is not group-readable and group-traversable: {path}")


def _require_group_file(path: Path, label: str) -> None:
    if not path.is_file() or not os.access(path, os.R_OK):
        raise EvalConfigError(f"{label} is not a readable file: {path}")
    if not path.stat().st_mode & stat.S_IRGRP:
        raise EvalConfigError(f"{label} is not group-readable: {path}")


def _absolute_existing(configured: str, workspace: Path, label: str) -> Path:
    expanded = Path(_expand_workspace(configured, workspace, label))
    if not expanded.is_absolute():
        raise EvalConfigError(f"{label} must resolve from an absolute path")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as error:
        raise EvalConfigError(f"{label} does not resolve: {expanded}") from error
    return resolved


def _safe_relative(value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise EvalConfigError(f"{label} must be a safe relative path: {value}")
    if "$" in value:
        raise EvalConfigError(f"{label} contains an unsupported placeholder")
    return relative


def _file_beneath(root: Path, relative: str, label: str) -> Path:
    try:
        child = (root / _safe_relative(relative, label)).resolve(strict=True)
    except OSError as error:
        raise EvalConfigError(f"{label} does not resolve beneath {root}") from error
    if not child.is_relative_to(root):
        raise EvalConfigError(f"{label} escapes its configured root")
    _require_group_file(child, label)
    parent = child.parent
    while True:
        _require_group_directory(parent, f"{label} parent")
        if parent == root:
            break
        parent = parent.parent
    return child


def _runtime_root(repository_root: str | Path | None) -> Path:
    candidate = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[4]
    )
    if not candidate.is_absolute():
        raise EvalConfigError("current runtime repository must be absolute")
    try:
        root = candidate.resolve(strict=True)
    except OSError as error:
        raise EvalConfigError("current runtime repository is unavailable") from error
    _require_group_directory(root, "runtime repository")
    return root


def _resolve_output(configured: str, workspace: Path, label: str) -> Path:
    expanded = Path(_expand_workspace(configured, workspace, label))
    if not expanded.is_absolute():
        raise EvalConfigError(f"{label} must be absolute after expansion")
    resolved = expanded.resolve(strict=False)
    if resolved == workspace or not resolved.is_relative_to(workspace):
        raise EvalConfigError(f"{label} must remain beneath the resolved workspace")
    if resolved.exists():
        _require_group_directory(resolved, label)
    return resolved


def _run_directory(output_root: Path, name: str) -> Path:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise EvalConfigError("run.name must be one safe path component")
    resolved = (output_root / name).resolve(strict=False)
    if not resolved.is_relative_to(output_root):
        raise EvalConfigError("run directory escapes output root")
    return resolved


def _resolve_chat_template(config: ConfiguredEvalRun, runtime_root: Path) -> Path | None:
    """Resolve the normalized generation template without leaking policy branches."""
    benchmarks = configured_benchmarks(config)
    if all(type(item) is ConditionalPerplexityBenchmarkConfig for item in benchmarks):
        return None
    relative = configured_chat_template_path(common_generation_config(config).chat_template)
    if relative is None:
        return None
    return _file_beneath(
        runtime_root,
        f"third_party/VLMEvalKit/nvbench/{relative}",
        "chat template",
    )


def resolve_eval_run(
    config: ConfiguredEvalRun,
    workspace: str | Path,
    *,
    repository_root: str | Path | None = None,
    validate_destination: bool = True,
) -> ResolvedEvalRun:
    """Resolve configured filesystem inputs into one immutable run.

    ``workspace`` and every configured source must resolve through real,
    group-accessible paths. The function creates no output. Invalid
    configuration, paths, or an occupied destination raise
    :class:`EvalConfigError`.
    """
    validate_configured_eval_run(config)
    workspace_path = Path(workspace)
    if not workspace_path.is_absolute():
        raise EvalConfigError("workspace must be absolute")
    try:
        workspace_path = workspace_path.resolve(strict=True)
    except OSError as error:
        raise EvalConfigError(f"workspace does not resolve: {workspace_path}") from error
    _require_group_directory(workspace_path, "workspace")

    runtime_root = _runtime_root(repository_root)
    entrypoint = _file_beneath(
        runtime_root, "tools/run_dynamic_text_generation_server.py", "runtime entrypoint"
    )
    chat_template = _resolve_chat_template(config, runtime_root)

    checkpoint = _absolute_existing(
        config.checkpoint.configured_path, workspace_path, "checkpoint root"
    )
    _require_group_directory(checkpoint, "checkpoint root")
    try:
        iteration_dir = (checkpoint / f"iter_{config.checkpoint.iteration:07d}").resolve(
            strict=True
        )
    except OSError as error:
        raise EvalConfigError("checkpoint iteration directory does not resolve") from error
    if not iteration_dir.is_relative_to(checkpoint):
        raise EvalConfigError("checkpoint iteration directory escapes checkpoint root")
    _require_group_directory(iteration_dir, "checkpoint iteration")
    metadata = _file_beneath(iteration_dir, ".metadata", "checkpoint metadata")

    cache = _absolute_existing(config.paths.cache_root, workspace_path, "cache root")
    _require_group_directory(cache, "cache root")
    output = _resolve_output(config.output.configured_root, workspace_path, "output root")
    run_directory = _run_directory(output, benchmark_array_run_component(config))
    resolved_config = config
    if type(config.benchmark) is ConditionalPerplexityBenchmarkConfig:
        benchmark_data = _absolute_existing(
            config.benchmark.configured_data_path, workspace_path, "conditional-perplexity data"
        )
        _require_group_file(benchmark_data, "conditional-perplexity data")
        image_archive = benchmark_data.with_suffix(".tar")
        _require_group_file(image_archive, "conditional-perplexity image archive")
        resolved_config = replace(
            config, benchmark=replace(config.benchmark, configured_data_path=str(benchmark_data))
        )
    paths = ResolvedPaths(
        workspace=str(workspace_path),
        runtime_root=str(runtime_root),
        runtime_entrypoint=str(entrypoint),
        chat_template=None if chat_template is None else str(chat_template),
        checkpoint_root=str(checkpoint),
        checkpoint_metadata=str(metadata),
        cache_root=str(cache),
        output_root=str(output),
        run_directory=str(run_directory),
    )
    resolved_run = ResolvedEvalRun(
        configured=resolved_config,
        paths=paths,
        resolved_cli_overrides=bind_resolved_cli_paths(config, paths),
    )
    if validate_destination:
        from .resolved_run_manifest import validate_output_destination

        validate_output_destination(resolved_run)
    return resolved_run


def _shared_output_directory(path: Path, label: str, *, writable: bool) -> None:
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise VLMEvalRunDirectoryPreparationError(f"cannot inspect {label}") from error
    required = stat.S_IRGRP | stat.S_IXGRP
    access = os.R_OK | os.X_OK | (os.W_OK if writable else 0)
    if not stat.S_ISDIR(mode) or mode & required != required or not os.access(path, access):
        raise VLMEvalRunDirectoryPreparationError(
            f"{label} must be accessible and group-readable/traversable"
        )


def _create_missing_output_chain(target: Path, workspace: Path) -> None:
    missing: list[Path] = []
    candidate = target
    while not candidate.exists():
        if candidate == workspace or not candidate.is_relative_to(workspace):
            raise VLMEvalRunDirectoryPreparationError("output creation escaped the workspace")
        missing.append(candidate)
        candidate = candidate.parent
    _shared_output_directory(candidate, "output ancestor", writable=True)
    for path in reversed(missing):
        try:
            os.mkdir(path, 0o750)
            os.chmod(path, 0o750)
        except OSError as error:
            raise VLMEvalRunDirectoryPreparationError(
                f"cannot create output directory: {path.name}"
            ) from error


def prepare_vlmeval_run_directory(resolved: ResolvedEvalRun) -> PreparedVLMEvalRunDirectory:
    """Create only the resolved output and run directories with shared modes.

    ``resolved`` is not mutated. The function may create the configured output
    chain and run directory with mode 0750. Changed, occupied, inaccessible, or
    escaping paths raise :class:`VLMEvalRunDirectoryPreparationError`.
    """
    if type(resolved) is not ResolvedEvalRun:
        raise VLMEvalRunDirectoryPreparationError("resolved run has the wrong record type")
    from .resolved_run_manifest import validate_output_destination

    try:
        validate_output_destination(resolved)
    except EvalConfigError as error:
        raise VLMEvalRunDirectoryPreparationError(
            "resolved run output destination is no longer valid"
        ) from error
    workspace = Path(resolved.paths.workspace)
    output = Path(resolved.paths.output_root)
    run_directory = Path(resolved.paths.run_directory)
    if not all(path.is_absolute() for path in (workspace, output, run_directory)):
        raise VLMEvalRunDirectoryPreparationError("resolved output paths must be absolute")
    if output == workspace or not output.is_relative_to(workspace):
        raise VLMEvalRunDirectoryPreparationError("output root must remain below the workspace")
    if run_directory == output or not run_directory.is_relative_to(output):
        raise VLMEvalRunDirectoryPreparationError("run directory must remain below output root")

    _create_missing_output_chain(output, workspace)
    try:
        if output.resolve(strict=True) != output:
            raise VLMEvalRunDirectoryPreparationError("output root changed during creation")
    except OSError as error:
        raise VLMEvalRunDirectoryPreparationError("output root is unavailable") from error
    _shared_output_directory(output, "output root", writable=True)

    is_new = not run_directory.exists()
    if is_new:
        try:
            os.mkdir(run_directory, 0o750)
            os.chmod(run_directory, 0o750)
        except OSError as error:
            raise VLMEvalRunDirectoryPreparationError("cannot create run directory") from error
    try:
        if run_directory.resolve(strict=True) != run_directory:
            raise VLMEvalRunDirectoryPreparationError("run directory changed during creation")
    except OSError as error:
        raise VLMEvalRunDirectoryPreparationError("run directory is unavailable") from error
    _shared_output_directory(run_directory, "run directory", writable=True)
    return PreparedVLMEvalRunDirectory(run_directory, is_new)


def resolve_megatron_dynamic_server_vlm_fewshot_run(
    run_config: str | Path,
    workspace: str | Path,
    *,
    benchmark_index: int | None = None,
    scored_limit: int | None = None,
    validate_destination: bool = True,
) -> ResolvedEvalRun:
    """Load, optionally bind one array entry, and resolve shared inputs.

    ``benchmark_index`` is a zero-based index into the configured plan.
    ``scored_limit`` optionally caps rows for a bounded debug run. The function
    creates no output and starts no process. Configuration, path, and
    destination failures propagate through their documented contract types.
    """
    configured = load_configured_eval_run(run_config)
    if benchmark_index is not None:
        configured = bind_benchmark_array_selection(
            configured, benchmark_index, scored_limit=scored_limit
        )
    elif scored_limit is not None:
        raise EvalConfigError("scored limit requires a benchmark index")
    return resolve_eval_run(configured, workspace, validate_destination=validate_destination)


def prepare_megatron_dynamic_server_vlm_fewshot_run(
    run_config: str | Path,
    workspace: str | Path,
    *,
    benchmark_index: int | None = None,
    scored_limit: int | None = None,
    host: str,
    port: int,
    coordinator_port: int,
    python_executable: str,
    compiler_cache_directory: str | Path,
    node_rank: int | None = None,
    master_addr: str | None = None,
    rendezvous_port: int | None = None,
    validate_destination: bool = True,
) -> PreparedMegatronDynamicServerVlmFewshotRun:
    """Resolve one run and return its side-effect-free server launch.

    The run configuration supplies immutable model inputs. Host, ports, Python,
    and compiler cache define the local torchrun command. The function creates
    no directory and starts no process; resolution and launch errors propagate.
    """
    resolved = resolve_megatron_dynamic_server_vlm_fewshot_run(
        run_config,
        workspace,
        benchmark_index=benchmark_index,
        scored_limit=scored_limit,
        validate_destination=validate_destination,
    )
    launch = build_megatron_dynamic_server_launch(
        resolved,
        resolved.paths.run_directory,
        host=host,
        port=port,
        coordinator_port=coordinator_port,
        python_executable=python_executable,
        compiler_cache_directory=compiler_cache_directory,
        node_rank=node_rank,
        master_addr=master_addr,
        rendezvous_port=rendezvous_port,
    )
    return PreparedMegatronDynamicServerVlmFewshotRun(resolved, launch)
