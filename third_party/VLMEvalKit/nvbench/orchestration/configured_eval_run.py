"""Immutable evaluation configuration records and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeAlias

from .utils import SAFE_IDENTIFIER_PATTERN, VLMEvalRowIndex

ConfigScalar: TypeAlias = str | int | float | bool
ArgumentRecord: TypeAlias = tuple[str, ConfigScalar]
_HUGGING_FACE_REPOSITORY = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)
_HUGGING_FACE_REVISION = re.compile(r"[0-9a-f]{40}")


class EvalConfigError(ValueError):
    """Raised when evaluation configuration is invalid."""


@dataclass(frozen=True)
class RunConfig:
    """Launch identity and resources."""

    name: str
    launch_mode: str
    nodes: int
    gpus_per_node: int
    api_nproc: int
    expert_model_parallel_size: int
    resource: str


@dataclass(frozen=True)
class ResourceConfig:
    """One reusable Slurm allocation and step-resource class."""

    name: str
    nodes: int
    gpus_per_node: int
    api_nproc: int
    expert_model_parallel_size: int
    server_cpus_per_node: int
    server_memory_per_node: str
    client_cpus: int
    client_memory: str
    allocation_memory_per_node: str
    time: str
    array_concurrency: int


@dataclass(frozen=True)
class JudgeScoringResourceConfig:
    """CPU-only Slurm resources for external judge scoring."""

    partition: str
    cpus: int
    memory: str
    time: str
    array_concurrency: int


@dataclass(frozen=True)
class ClusterConfig:
    """Cluster submission settings and named resource classes."""

    name: str
    source_path: str
    account: str
    partition: str
    resources: tuple[ResourceConfig, ...]
    judge_scoring: JudgeScoringResourceConfig | None = None


@dataclass(frozen=True)
class TokenizerConfig:
    """Pinned Hugging Face tokenizer selected for evaluation."""

    repository: str
    revision: str


@dataclass(frozen=True)
class ServerConfig:
    """Server limits and deliberate checkpoint overrides."""

    context_length: int | None
    health_timeout_seconds: int
    request_timeout_seconds: int
    cli_overrides: tuple[ArgumentRecord, ...]


@dataclass(frozen=True)
class ModelConfig:
    """Minimal model-serving configuration not owned by the checkpoint."""

    name: str
    tokenizer: TokenizerConfig
    server: ServerConfig


@dataclass(frozen=True)
class GenerationConfig:
    """Generation and result-judging parameters for one benchmark."""

    temperature: float
    retry: int
    image_size: int
    image_detail: str
    max_new_tokens: int
    stop: tuple[str, ...]
    enable_thinking: bool
    chat_template: str


@dataclass(frozen=True)
class ConditionalPerplexityRequestConfig:
    """Model request parameters that define conditional perplexity."""

    instruction: str
    temperature: float
    top_p: float
    retry: int


@dataclass(frozen=True)
class BenchmarkConfig:
    """Native dataset, few-shot, result, and generation contract."""

    name: str
    dataset: str
    upstream_class: str
    modality: str
    num_shot: int
    demonstration_type: str
    demonstration_answer_format: str
    include_demonstration_images: bool
    prepend_reasoning_cue: bool
    scored_limit: int | None
    result_reader: str
    expected_result_suffix: str
    result_metric_unit: str
    judge: str
    resource: str
    generation: GenerationConfig
    server_context_length: int | None = None
    prompt_adapter: str | None = None
    demonstration_indices: tuple[VLMEvalRowIndex, ...] = ()

    @property
    def demonstration_source(self) -> str:
        """Return whether examples come from benchmark rows or nvbench assets."""
        return "benchmark" if self.demonstration_type == "benchmark" else "packaged"


@dataclass(frozen=True)
class ConditionalPerplexityBenchmarkConfig:
    """One packed dataset whose reference descriptions are scored."""

    name: str
    dataset: str
    data_format: str
    configured_data_path: str
    category_field: str
    resource: str
    scored_limit: int | None
    request: ConditionalPerplexityRequestConfig


BenchmarkPlanEntry: TypeAlias = BenchmarkConfig | ConditionalPerplexityBenchmarkConfig


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    """Ordered benchmark plan loaded from one suite YAML."""

    name: str
    source_path: str
    benchmarks: tuple[BenchmarkPlanEntry, ...]


@dataclass(frozen=True)
class BenchmarkArraySelection:
    """Source plan and selected position for one Slurm array element."""

    source_suite_name: str | None
    source_suite_path: str | None
    benchmark_index: int
    benchmark_count: int
    ordered_datasets: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointConfig:
    """Exact checkpoint root and iteration to load."""

    configured_path: str
    iteration: int
    trusted_standalone_common_state: bool = False


@dataclass(frozen=True)
class ConfiguredPaths:
    """Shared VLMEvalKit cache root."""

    cache_root: str


@dataclass(frozen=True)
class OutputConfig:
    """Output root and resume authorization."""

    configured_root: str
    resume: bool


@dataclass(frozen=True)
class ConfiguredEvalRun:
    """Complete evaluation configuration before path resolution."""

    run: RunConfig
    cluster: ClusterConfig
    model: ModelConfig
    benchmark: BenchmarkPlanEntry | None
    checkpoint: CheckpointConfig
    paths: ConfiguredPaths
    output: OutputConfig
    benchmark_suite: BenchmarkSuiteConfig | None = None
    benchmark_array_selection: BenchmarkArraySelection | None = None


@dataclass(frozen=True)
class ResolvedPaths:
    """Resolved filesystem paths used by one evaluation run."""

    workspace: str
    runtime_root: str
    runtime_entrypoint: str
    chat_template: str | None
    checkpoint_root: str
    checkpoint_metadata: str
    cache_root: str
    output_root: str
    run_directory: str


@dataclass(frozen=True)
class ResolvedEvalRun:
    """Configured run plus resolved paths and server arguments."""

    configured: ConfiguredEvalRun
    paths: ResolvedPaths
    resolved_cli_overrides: tuple[ArgumentRecord, ...]


def bind_resolved_cli_paths(
    config: ConfiguredEvalRun, paths: ResolvedPaths
) -> tuple[ArgumentRecord, ...]:
    """Return server arguments with the resolved tokenizer and topology."""
    generated: tuple[ArgumentRecord, ...] = (
        ("tensor-model-parallel-size", 1),
        ("expert-tensor-parallel-size", 1),
        ("expert-model-parallel-size", config.run.expert_model_parallel_size),
        ("pipeline-model-parallel-size", 1),
        ("tokenizer-type", "HuggingFaceTokenizer"),
        ("tokenizer-model", config.model.tokenizer.repository),
        ("tokenizer-hf-revision", config.model.tokenizer.revision),
        ("no-use-tokenizer-model-from-checkpoint-args", True),
        ("inference-max-requests", 1),
        ("dist-ckpt-strictness", "log_all"),
        ("num-speculative-tokens", 0),
    )
    checkpoint_context = (
        (("inference-context-from-checkpoint", True),)
        if config.model.server.context_length is None
        else ()
    )
    chat_template = (
        () if paths.chat_template is None else (("chat-template", paths.chat_template),)
    )
    return generated + checkpoint_context + chat_template + config.model.server.cli_overrides


def cluster_resource(cluster: ClusterConfig, name: str) -> ResourceConfig:
    """Return one named cluster resource or raise a configuration error."""
    matches = tuple(item for item in cluster.resources if item.name == name)
    if len(matches) != 1:
        raise EvalConfigError(f"resource class is not configured exactly once: {name}")
    return matches[0]


def configured_resource(config: ConfiguredEvalRun, name: str) -> ResourceConfig:
    """Return one named resource from a configured evaluation."""
    return cluster_resource(config.cluster, name)


def run_for_resource(name: str, resource: ResourceConfig) -> RunConfig:
    """Bind immutable run identity to one resource class."""
    return RunConfig(
        name=name,
        launch_mode=("local_torchrun" if resource.nodes == 1 else "slurm_multinode_torchrun"),
        nodes=resource.nodes,
        gpus_per_node=resource.gpus_per_node,
        api_nproc=resource.api_nproc,
        expert_model_parallel_size=resource.expert_model_parallel_size,
        resource=resource.name,
    )


def _configured_absolute(path: str) -> bool:
    return path == "${workspace}" or path.startswith("${workspace}/") or path.startswith("/")


def validate_configured_eval_run(config: ConfiguredEvalRun) -> None:
    """Validate one complete configuration without accessing the filesystem."""
    run = config.run
    if not SAFE_IDENTIFIER_PATTERN.fullmatch(run.name):
        raise EvalConfigError("run name must be lowercase and filesystem-safe")
    valid_topology = (run.launch_mode == "local_torchrun" and run.nodes == 1) or (
        run.launch_mode == "slurm_multinode_torchrun" and run.nodes > 1
    )
    if type(run.nodes) is not int or not valid_topology:
        raise EvalConfigError("launch mode and node count are inconsistent")
    resources = (run.gpus_per_node, run.api_nproc, run.expert_model_parallel_size)
    if any(type(value) is not int or value <= 0 for value in resources):
        raise EvalConfigError("run resource counts must be positive integers")
    if run.nodes * run.gpus_per_node % run.expert_model_parallel_size:
        raise EvalConfigError("expert parallel size must divide the total server GPU count")
    cluster = config.cluster
    if not SAFE_IDENTIFIER_PATTERN.fullmatch(cluster.name):
        raise EvalConfigError("cluster name must be lowercase and filesystem-safe")
    if not cluster.source_path.startswith("/") or not cluster.account or not cluster.partition:
        raise EvalConfigError("cluster identity and source must be configured")
    resource_names = tuple(item.name for item in cluster.resources)
    if not resource_names or len(resource_names) != len(set(resource_names)):
        raise EvalConfigError("cluster resource names must be nonempty and unique")
    if run.resource not in resource_names:
        raise EvalConfigError("run resource is absent from the cluster config")
    for resource in cluster.resources:
        if not SAFE_IDENTIFIER_PATTERN.fullmatch(resource.name):
            raise EvalConfigError("resource names must be lowercase and filesystem-safe")
        counts = (
            resource.nodes,
            resource.gpus_per_node,
            resource.api_nproc,
            resource.expert_model_parallel_size,
            resource.server_cpus_per_node,
            resource.client_cpus,
            resource.array_concurrency,
        )
        if any(type(value) is not int or value <= 0 for value in counts):
            raise EvalConfigError("cluster resource counts must be positive integers")
        if resource.nodes * resource.gpus_per_node % resource.expert_model_parallel_size:
            raise EvalConfigError("resource expert parallel size must divide total GPUs")
        if any(
            type(value) is not str or not value
            for value in (
                resource.server_memory_per_node,
                resource.client_memory,
                resource.allocation_memory_per_node,
                resource.time,
            )
        ):
            raise EvalConfigError("cluster resource memory and time must be configured")
    scoring = cluster.judge_scoring
    if scoring is not None:
        if not scoring.partition:
            raise EvalConfigError("judge-scoring partition must be configured")
        if any(
            type(value) is not int or value <= 0
            for value in (scoring.cpus, scoring.array_concurrency)
        ):
            raise EvalConfigError("judge-scoring counts must be positive integers")
        if not scoring.memory or not scoring.time:
            raise EvalConfigError("judge-scoring memory and time must be configured")
    expected_run = run_for_resource(run.name, cluster_resource(cluster, run.resource))
    if run != expected_run:
        raise EvalConfigError("run topology must be derived from its resource class")
    if type(config.checkpoint.iteration) is not int or config.checkpoint.iteration <= 0:
        raise EvalConfigError("checkpoint iteration must be positive")
    if type(config.checkpoint.trusted_standalone_common_state) is not bool:
        raise EvalConfigError("checkpoint trusted-standalone-common-state flag must be boolean")
    tokenizer = config.model.tokenizer
    if not _HUGGING_FACE_REPOSITORY.fullmatch(tokenizer.repository):
        raise EvalConfigError("tokenizer repository must be an owner/name Hugging Face ID")
    if not _HUGGING_FACE_REVISION.fullmatch(tokenizer.revision):
        raise EvalConfigError("tokenizer revision must be a full lowercase commit SHA")
    roots = (
        config.checkpoint.configured_path,
        config.paths.cache_root,
        config.output.configured_root,
    )
    if any(not _configured_absolute(path) for path in roots):
        raise EvalConfigError("configured data and output roots must be absolute")
    server = config.model.server
    if (
        server.context_length is not None
        and (type(server.context_length) is not int or server.context_length <= 0)
    ):
        raise EvalConfigError("server context override must be a positive integer")
    if any(
        type(value) is not int or value <= 0
        for value in (server.health_timeout_seconds, server.request_timeout_seconds)
    ):
        raise EvalConfigError("server limits and timeouts must be positive integers")
    names = tuple(name for name, _ in server.cli_overrides)
    if len(names) != len(set(names)):
        raise EvalConfigError("server overrides must be unique")
    generated_names = {
        "tensor-model-parallel-size",
        "expert-tensor-parallel-size",
        "expert-model-parallel-size",
        "pipeline-model-parallel-size",
        "tokenizer-type",
        "tokenizer-model",
        "tokenizer-hf-revision",
        "no-use-tokenizer-model-from-checkpoint-args",
        "inference-max-requests",
        "inference-context-from-checkpoint",
        "dist-ckpt-strictness",
        "num-speculative-tokens",
        "chat-template",
    }
    if generated_names.intersection(names):
        raise EvalConfigError("server overrides duplicate evaluator-owned arguments")
    from .benchmark_array_selection import validate_benchmark_plan

    validate_benchmark_plan(config)
