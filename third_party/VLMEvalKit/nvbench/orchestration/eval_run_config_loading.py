"""Compose an evaluation run from one run, model, and suite YAML."""

from __future__ import annotations

from pathlib import Path

from .configured_eval_run import (
    CheckpointConfig,
    ConfiguredEvalRun,
    ConfiguredPaths,
    EvalConfigError,
    OutputConfig,
    cluster_resource,
    run_for_resource,
    validate_configured_eval_run,
)
from .eval_component_config_loading import (
    load_benchmark_suite,
    load_cluster_config,
    load_model_config,
)
from .strict_eval_yaml_loading import (
    expect_mapping,
    expect_type,
    load_yaml_file,
    resolve_config_path,
)


def load_configured_eval_run(path: str | Path) -> ConfiguredEvalRun:
    """Load one strict run config and its minimal model and suite files."""
    try:
        run_path = Path(path).resolve(strict=True)
        config_root = run_path.parent.parent.resolve(strict=True)
    except OSError as error:
        raise EvalConfigError(f"cannot resolve run config {path}: {error}") from error
    raw = expect_mapping(
        load_yaml_file(run_path),
        {
            "schema_version",
            "run",
            "model_config",
            "suite_config",
            "cluster_config",
            "checkpoint",
            "paths",
            "output",
        },
        set(),
        "eval run",
    )
    if expect_type(raw["schema_version"], int, "schema_version") != 1:
        raise EvalConfigError("unsupported run-config schema version")
    model_path = resolve_config_path(
        run_path.parent, config_root, raw["model_config"], "model config"
    )
    suite_path = resolve_config_path(
        run_path.parent, config_root, raw["suite_config"], "suite config"
    )
    cluster_path = resolve_config_path(
        run_path.parent, config_root, raw["cluster_config"], "cluster config"
    )
    run = expect_mapping(raw["run"], {"name"}, set(), "run")
    name = expect_type(run["name"], str, "run.name")
    cluster = load_cluster_config(cluster_path)
    suite = load_benchmark_suite(suite_path)
    initial_resource = cluster_resource(cluster, suite.benchmarks[0].resource)
    run_config = run_for_resource(name, initial_resource)
    checkpoint = expect_mapping(
        raw["checkpoint"],
        {"path", "iteration"},
        {"trusted_standalone_common_state"},
        "checkpoint",
    )
    paths = expect_mapping(raw["paths"], {"cache_root"}, set(), "paths")
    output = expect_mapping(raw["output"], {"root", "resume"}, set(), "output")
    config = ConfiguredEvalRun(
        run=run_config,
        cluster=cluster,
        model=load_model_config(model_path),
        benchmark=None,
        benchmark_suite=suite,
        checkpoint=CheckpointConfig(
            configured_path=expect_type(checkpoint["path"], str, "checkpoint.path"),
            iteration=expect_type(checkpoint["iteration"], int, "checkpoint.iteration"),
            trusted_standalone_common_state=expect_type(
                checkpoint.get("trusted_standalone_common_state", False),
                bool,
                "checkpoint.trusted_standalone_common_state",
            ),
        ),
        paths=ConfiguredPaths(
            cache_root=expect_type(paths["cache_root"], str, "paths.cache_root")
        ),
        output=OutputConfig(
            configured_root=expect_type(output["root"], str, "output.root"),
            resume=expect_type(output["resume"], bool, "output.resume"),
        ),
    )
    validate_configured_eval_run(config)
    return config


__all__ = ("load_configured_eval_run",)
