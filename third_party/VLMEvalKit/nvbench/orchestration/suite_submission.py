"""Build resource-grouped Slurm submissions for one evaluation suite."""

from __future__ import annotations

import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .benchmark_array_selection import configured_benchmarks
from .configured_eval_run import (
    BenchmarkConfig,
    ConfiguredEvalRun,
    ResourceConfig,
    configured_resource,
)
from .eval_run_config_loading import load_configured_eval_run


class SuiteSubmissionError(RuntimeError):
    """Raised when a suite cannot be planned or submitted safely."""


SBATCH_JOB_ID_PATTERN = re.compile(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?")


@dataclass(frozen=True)
class SuiteSubmission:
    """One resource-homogeneous sparse Slurm array submission."""

    resource: str
    benchmark_indices: tuple[int, ...]
    datasets: tuple[str, ...]
    argv: tuple[str, ...]
    environment_overrides: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class JudgeScoringSubmission:
    """One CPU-only judge array paired with a GPU resource submission."""

    source_resource: str
    benchmark_indices: tuple[int, ...]
    datasets: tuple[str, ...]
    argv: tuple[str, ...]


def _absolute_file(value: str | Path, label: str) -> Path:
    try:
        path = Path(value).resolve(strict=True)
    except OSError as error:
        raise SuiteSubmissionError(f"{label} does not resolve") from error
    if not path.is_file():
        raise SuiteSubmissionError(f"{label} must be a file")
    return path


def _absolute_directory(value: str | Path, label: str) -> Path:
    try:
        path = Path(value).resolve(strict=True)
    except OSError as error:
        raise SuiteSubmissionError(f"{label} does not resolve") from error
    if not path.is_dir():
        raise SuiteSubmissionError(f"{label} must be a directory")
    return path


def _array(indices: tuple[int, ...], concurrency: int) -> str:
    return f"{','.join(str(index) for index in indices)}%{concurrency}"


def _submission(
    configured: ConfiguredEvalRun,
    resource: ResourceConfig,
    indices: tuple[int, ...],
    run_config: Path,
    workspace: Path,
    launcher: Path,
    limit: int | None,
) -> SuiteSubmission:
    benchmarks = configured_benchmarks(configured)
    datasets = tuple(benchmarks[index].dataset for index in indices)
    total_cpus = resource.server_cpus_per_node + resource.client_cpus
    argv = (
        "sbatch",
        "--parsable",
        f"--account={configured.cluster.account}",
        f"--partition={configured.cluster.partition}",
        f"--nodes={resource.nodes}",
        "--ntasks-per-node=1",
        f"--gpus-per-node={resource.gpus_per_node}",
        f"--cpus-per-task={total_cpus}",
        f"--mem={resource.allocation_memory_per_node}",
        f"--time={resource.time}",
        f"--array={_array(indices, resource.array_concurrency)}",
        f"--job-name=vlmeval-{configured.run.name}-{resource.name}",
        str(launcher),
        "--run-config",
        str(run_config),
        "--workspace",
        str(workspace),
    )
    if limit is not None:
        argv += ("--limit", str(limit))
    environment = (
        ("NVBENCH_SERVER_CPUS", str(resource.server_cpus_per_node)),
        ("NVBENCH_SERVER_MEMORY", resource.server_memory_per_node),
        ("NVBENCH_CLIENT_CPUS", str(resource.client_cpus)),
        ("NVBENCH_CLIENT_MEMORY", resource.client_memory),
    )
    return SuiteSubmission(resource.name, indices, datasets, argv, environment)


def build_suite_submission_plan(
    run_config: str | Path,
    workspace: str | Path,
    launcher: str | Path,
    *,
    benchmark_index: int | None = None,
    limit: int | None = None,
) -> tuple[SuiteSubmission, ...]:
    """Group suite entries by resource and render deterministic sbatch commands."""
    if benchmark_index is not None and type(benchmark_index) is not int:
        raise SuiteSubmissionError("benchmark index must be an integer")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise SuiteSubmissionError("limit must be a positive integer")
    run_path = _absolute_file(run_config, "run config")
    workspace_path = _absolute_directory(workspace, "workspace")
    launcher_path = _absolute_file(launcher, "launcher")
    configured = load_configured_eval_run(run_path)
    benchmarks = configured_benchmarks(configured)
    if benchmark_index is not None and not 0 <= benchmark_index < len(benchmarks):
        raise SuiteSubmissionError("benchmark index is outside the configured suite")
    selected = (
        tuple(enumerate(benchmarks))
        if benchmark_index is None
        else ((benchmark_index, benchmarks[benchmark_index]),)
    )
    groups: dict[str, list[int]] = defaultdict(list)
    for index, benchmark in selected:
        groups[benchmark.resource].append(index)
    return tuple(
        _submission(
            configured,
            configured_resource(configured, resource.name),
            tuple(groups[resource.name]),
            run_path,
            workspace_path,
            launcher_path,
            limit,
        )
        for resource in configured.cluster.resources
        if resource.name in groups
    )


def build_judge_scoring_submission_plan(
    run_config: str | Path,
    workspace: str | Path,
    launcher: str | Path,
    *,
    benchmark_index: int | None = None,
    limit: int | None = None,
) -> tuple[JudgeScoringSubmission, ...]:
    """Render CPU-only follow-ups for the selected LLM-judged benchmarks."""
    run_path = _absolute_file(run_config, "run config")
    workspace_path = _absolute_directory(workspace, "workspace")
    launcher_path = _absolute_file(launcher, "judge launcher")
    configured = load_configured_eval_run(run_path)
    benchmarks = configured_benchmarks(configured)
    if benchmark_index is not None and (
        type(benchmark_index) is not int
        or not 0 <= benchmark_index < len(benchmarks)
    ):
        raise SuiteSubmissionError("benchmark index is outside the configured suite")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise SuiteSubmissionError("limit must be a positive integer")
    selected = (
        tuple(enumerate(benchmarks))
        if benchmark_index is None
        else ((benchmark_index, benchmarks[benchmark_index]),)
    )
    groups: dict[str, list[int]] = defaultdict(list)
    for index, benchmark in selected:
        if type(benchmark) is BenchmarkConfig and benchmark.judge != "exact_matching":
            groups[benchmark.resource].append(index)
    scoring = configured.cluster.judge_scoring
    if groups and scoring is None:
        raise SuiteSubmissionError("cluster has no CPU judge-scoring resource")
    if scoring is None:
        return ()
    plans = []
    for resource in configured.cluster.resources:
        if resource.name not in groups:
            continue
        indices = tuple(groups[resource.name])
        datasets = tuple(benchmarks[index].dataset for index in indices)
        argv = (
            "sbatch",
            "--parsable",
            f"--account={configured.cluster.account}",
            f"--partition={scoring.partition}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={scoring.cpus}",
            f"--mem={scoring.memory}",
            f"--time={scoring.time}",
            f"--array={_array(indices, scoring.array_concurrency)}",
            f"--job-name=vlmeval-{configured.run.name}-judge-{resource.name}",
            str(launcher_path),
            "--run-config",
            str(run_path),
            "--workspace",
            str(workspace_path),
        )
        if limit is not None:
            argv += ("--limit", str(limit))
        plans.append(JudgeScoringSubmission(resource.name, indices, datasets, argv))
    return tuple(plans)


def submit_suite_plan(
    plan: tuple[SuiteSubmission, ...],
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Submit a validated plan and return Slurm job IDs in resource order."""
    source = os.environ if environment is None else environment
    for required in ("NVBENCH_SERVER_IMAGE", "NVBENCH_CLIENT_IMAGE"):
        if not source.get(required):
            raise SuiteSubmissionError(f"{required} must be exported before submission")
    job_ids: list[str] = []
    for submission in plan:
        child_environment = dict(source)
        child_environment.update(submission.environment_overrides)
        try:
            completed = subprocess.run(
                submission.argv,
                check=True,
                capture_output=True,
                text=True,
                env=child_environment,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise SuiteSubmissionError(
                f"submission failed for resource {submission.resource}"
            ) from error
        job_id = completed.stdout.strip()
        if not SBATCH_JOB_ID_PATTERN.fullmatch(job_id):
            raise SuiteSubmissionError("sbatch returned an invalid job ID")
        job_ids.append(job_id)
    return tuple(job_ids)


def submit_judge_scoring_plan(
    plan: tuple[JudgeScoringSubmission, ...],
    generation_plan: tuple[SuiteSubmission, ...],
    generation_job_ids: tuple[str, ...],
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Submit CPU judge arrays after their corresponding GPU arrays terminate."""
    if len(generation_plan) != len(generation_job_ids):
        raise SuiteSubmissionError("generation plan and job IDs differ in length")
    dependencies = {
        submission.resource: job_id
        for submission, job_id in zip(generation_plan, generation_job_ids, strict=True)
    }
    if len(dependencies) != len(generation_plan):
        raise SuiteSubmissionError("generation resources are not unique")
    source = os.environ if environment is None else environment
    if plan and not source.get("NVBENCH_CLIENT_IMAGE"):
        raise SuiteSubmissionError("NVBENCH_CLIENT_IMAGE must be exported before submission")
    job_ids = []
    for submission in plan:
        dependency = dependencies.get(submission.source_resource)
        if dependency is None or not SBATCH_JOB_ID_PATTERN.fullmatch(dependency):
            raise SuiteSubmissionError("judge scoring has no valid generation dependency")
        dependency_job_id = dependency.partition(";")[0]
        argv = submission.argv[:2] + (
            f"--dependency=afterany:{dependency_job_id}",
        ) + submission.argv[2:]
        try:
            completed = subprocess.run(
                argv,
                check=True,
                capture_output=True,
                text=True,
                env=dict(source),
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise SuiteSubmissionError(
                f"judge submission failed for resource {submission.source_resource}"
            ) from error
        job_id = completed.stdout.strip()
        if not SBATCH_JOB_ID_PATTERN.fullmatch(job_id):
            raise SuiteSubmissionError("sbatch returned an invalid judge job ID")
        job_ids.append(job_id)
    return tuple(job_ids)


__all__ = (
    "JudgeScoringSubmission",
    "SuiteSubmission",
    "SuiteSubmissionError",
    "build_judge_scoring_submission_plan",
    "build_suite_submission_plan",
    "submit_judge_scoring_plan",
    "submit_suite_plan",
)
