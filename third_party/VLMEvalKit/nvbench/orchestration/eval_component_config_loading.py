"""Load the minimal model and benchmark-suite configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chat_template_policy import parse_chat_template_policy
from .configured_eval_run import (
    BenchmarkConfig,
    BenchmarkSuiteConfig,
    ClusterConfig,
    ConditionalPerplexityBenchmarkConfig,
    ConditionalPerplexityRequestConfig,
    EvalConfigError,
    GenerationConfig,
    JudgeScoringResourceConfig,
    ModelConfig,
    ResourceConfig,
    ServerConfig,
    TokenizerConfig,
)
from .fewshot_dataset_adaptation.prompt_adapters import supported_prompt_adapters
from .strict_eval_yaml_loading import (
    expect_mapping,
    expect_number,
    expect_scalar,
    expect_string_mapping,
    expect_string_tuple,
    expect_type,
    load_yaml_file,
)
from .utils import (
    VLMEvalRowIdentityError,
    require_unique_vlmeval_row_indices,
)


def _argument_records(raw: Any, label: str) -> tuple[tuple[str, Any], ...]:
    values = expect_string_mapping(raw, label)
    return tuple(
        (key, expect_scalar(value, f"{label}.{key}")) for key, value in sorted(values.items())
    )


def load_model_config(path: Path) -> ModelConfig:
    """Load tokenizer selection and intentional server overrides."""
    raw = expect_mapping(load_yaml_file(path), {"schema_version", "model"}, set(), "model config")
    if expect_type(raw["schema_version"], int, "schema_version") != 1:
        raise EvalConfigError("unsupported model-config schema version")
    model = expect_mapping(raw["model"], {"name", "tokenizer", "server"}, set(), "model")
    tokenizer = expect_mapping(
        model["tokenizer"], {"repository", "revision"}, set(), "tokenizer"
    )
    server = expect_mapping(model["server"], set(), {"context_length", "overrides"}, "server")
    context_length = server.get("context_length")
    if context_length is not None:
        context_length = expect_type(context_length, int, "server.context_length")
    return ModelConfig(
        name=expect_type(model["name"], str, "model.name"),
        tokenizer=TokenizerConfig(
            repository=expect_type(tokenizer["repository"], str, "tokenizer.repository"),
            revision=expect_type(tokenizer["revision"], str, "tokenizer.revision"),
        ),
        server=ServerConfig(
            context_length=context_length,
            health_timeout_seconds=1800,
            request_timeout_seconds=300,
            cli_overrides=_argument_records(server.get("overrides", {}), "server.overrides"),
        ),
    )


def load_cluster_config(path: Path) -> ClusterConfig:
    """Load reusable Slurm settings and named resource classes."""
    raw = expect_mapping(
        load_yaml_file(path), {"schema_version", "cluster"}, set(), "cluster config"
    )
    if expect_type(raw["schema_version"], int, "schema_version") != 1:
        raise EvalConfigError("unsupported cluster-config schema version")
    cluster = expect_mapping(
        raw["cluster"],
        {"name", "slurm", "resources"},
        {"judge_scoring"},
        "cluster",
    )
    slurm = expect_mapping(
        cluster["slurm"], {"account", "partition"}, set(), "cluster.slurm"
    )
    resources = expect_string_mapping(cluster["resources"], "cluster.resources")
    scoring = None
    if "judge_scoring" in cluster:
        raw_scoring = expect_mapping(
            cluster["judge_scoring"],
            {"partition", "cpus", "memory", "time", "array_concurrency"},
            set(),
            "cluster.judge_scoring",
        )
        scoring = JudgeScoringResourceConfig(
            partition=expect_type(
                raw_scoring["partition"], str, "cluster.judge_scoring.partition"
            ),
            cpus=expect_type(raw_scoring["cpus"], int, "cluster.judge_scoring.cpus"),
            memory=expect_type(
                raw_scoring["memory"], str, "cluster.judge_scoring.memory"
            ),
            time=expect_type(
                raw_scoring["time"], str, "cluster.judge_scoring.time"
            ),
            array_concurrency=expect_type(
                raw_scoring["array_concurrency"],
                int,
                "cluster.judge_scoring.array_concurrency",
            ),
        )
    if not resources:
        raise EvalConfigError("cluster.resources must not be empty")
    loaded: list[ResourceConfig] = []
    required = {
        "nodes",
        "gpus_per_node",
        "api_nproc",
        "expert_model_parallel_size",
        "server_cpus_per_node",
        "server_memory_per_node",
        "client_cpus",
        "client_memory",
        "allocation_memory_per_node",
        "time",
        "array_concurrency",
    }
    for name, value in resources.items():
        item = expect_mapping(value, required, set(), f"cluster.resources.{name}")
        loaded.append(
            ResourceConfig(
                name=name,
                nodes=expect_type(item["nodes"], int, f"resources.{name}.nodes"),
                gpus_per_node=expect_type(
                    item["gpus_per_node"], int, f"resources.{name}.gpus_per_node"
                ),
                api_nproc=expect_type(item["api_nproc"], int, f"resources.{name}.api_nproc"),
                expert_model_parallel_size=expect_type(
                    item["expert_model_parallel_size"],
                    int,
                    f"resources.{name}.expert_model_parallel_size",
                ),
                server_cpus_per_node=expect_type(
                    item["server_cpus_per_node"], int, f"resources.{name}.server_cpus_per_node"
                ),
                server_memory_per_node=expect_type(
                    item["server_memory_per_node"], str, f"resources.{name}.server_memory_per_node"
                ),
                client_cpus=expect_type(item["client_cpus"], int, f"resources.{name}.client_cpus"),
                client_memory=expect_type(
                    item["client_memory"], str, f"resources.{name}.client_memory"
                ),
                allocation_memory_per_node=expect_type(
                    item["allocation_memory_per_node"],
                    str,
                    f"resources.{name}.allocation_memory_per_node",
                ),
                time=expect_type(item["time"], str, f"resources.{name}.time"),
                array_concurrency=expect_type(
                    item["array_concurrency"], int, f"resources.{name}.array_concurrency"
                ),
            )
        )
    return ClusterConfig(
        name=expect_type(cluster["name"], str, "cluster.name"),
        source_path=str(path),
        account=expect_type(slurm["account"], str, "cluster.slurm.account"),
        partition=expect_type(slurm["partition"], str, "cluster.slurm.partition"),
        resources=tuple(loaded),
        judge_scoring=scoring,
    )


_GENERATION_REQUEST_FIELDS = {
    "chat_template",
    "temperature",
    "retry",
    "image_size",
    "image_detail",
    "max_new_tokens",
    "stop",
    "enable_thinking",
}


def _load_request(
    raw: Any,
    *,
    label: str = "suite.request",
    defaults: GenerationConfig | None = None,
) -> GenerationConfig:
    """Load a complete request or apply one benchmark's partial overrides."""
    required = _GENERATION_REQUEST_FIELDS if defaults is None else set()
    request = expect_mapping(
        raw,
        required,
        set() if defaults is None else _GENERATION_REQUEST_FIELDS,
        label,
    )

    def value(name: str) -> Any:
        if name in request:
            return request[name]
        if defaults is None:
            raise EvalConfigError(f"{label} is missing: {name}")
        return getattr(defaults, name)

    if "stop" in request:
        stop = expect_string_tuple(request["stop"], f"{label}.stop")
    elif defaults is not None:
        stop = defaults.stop
    else:
        raise EvalConfigError(f"{label} is missing: stop")

    return GenerationConfig(
        temperature=float(expect_number(value("temperature"), f"{label}.temperature")),
        retry=expect_type(value("retry"), int, f"{label}.retry"),
        image_size=expect_type(value("image_size"), int, f"{label}.image_size"),
        image_detail=expect_type(value("image_detail"), str, f"{label}.image_detail"),
        max_new_tokens=expect_type(
            value("max_new_tokens"), int, f"{label}.max_new_tokens"
        ),
        stop=stop,
        enable_thinking=expect_type(
            value("enable_thinking"), bool, f"{label}.enable_thinking"
        ),
        chat_template=parse_chat_template_policy(
            value("chat_template"), f"{label}.chat_template"
        ),
    )


def load_benchmark_suite(path: Path) -> BenchmarkSuiteConfig:
    """Load one direct ordered suite with dataset-owned demonstration settings."""
    raw = expect_mapping(load_yaml_file(path), {"schema_version", "suite"}, set(), "suite")
    if expect_type(raw["schema_version"], int, "schema_version") != 1:
        raise EvalConfigError("unsupported suite schema version")
    suite = expect_mapping(raw["suite"], {"name", "request", "benchmarks"}, {"kind"}, "suite")
    kind = expect_type(suite.get("kind", "generation"), str, "suite.kind")
    if kind not in {"generation", "conditional_perplexity"}:
        raise EvalConfigError(f"unsupported suite kind: {kind}")
    if kind == "conditional_perplexity":
        return _load_conditional_perplexity_suite(path, suite)
    generation = _load_request(suite["request"])
    entries = expect_type(suite["benchmarks"], list, "suite.benchmarks")
    if not entries:
        raise EvalConfigError("suite must contain at least one benchmark")
    benchmarks: list[BenchmarkConfig] = []
    for index, raw_entry in enumerate(entries):
        label = f"suite.benchmarks[{index}]"
        entry = expect_mapping(
            raw_entry,
            {
                "dataset",
                "class",
                "num_shot",
                "demonstration",
                "answer_format",
                "result_reader",
                "result_suffix",
                "result_metric_unit",
                "judge",
                "resource",
            },
            {"prompt_adapter", "request", "server"},
            label,
        )
        dataset = expect_type(entry["dataset"], str, f"{label}.dataset")
        prompt_adapter = entry.get("prompt_adapter")
        if prompt_adapter is not None:
            prompt_adapter = expect_type(
                prompt_adapter, str, f"{label}.prompt_adapter"
            )
            if prompt_adapter not in supported_prompt_adapters():
                raise EvalConfigError(
                    f"{label}.prompt_adapter is unsupported: {prompt_adapter}"
                )
        server = expect_mapping(
            entry.get("server", {}),
            set(),
            {"context_length"},
            f"{label}.server",
        )
        context_length = server.get("context_length")
        if context_length is not None:
            context_length = expect_type(
                context_length, int, f"{label}.server.context_length"
            )
        demonstration = expect_mapping(
            entry["demonstration"],
            {"type", "include_images"},
            {"indices", "prepend_reasoning_cue"},
            f"{label}.demonstration",
        )
        raw_demonstration_indices = expect_type(
            demonstration.get("indices", []),
            list,
            f"{label}.demonstration.indices",
        )
        try:
            demonstration_indices = require_unique_vlmeval_row_indices(
                tuple(raw_demonstration_indices),
                label=f"{label}.demonstration.indices",
            )
        except VLMEvalRowIdentityError as error:
            raise EvalConfigError(str(error)) from error
        benchmarks.append(
            BenchmarkConfig(
                name=dataset.lower().replace("-", "_").replace(" ", "_"),
                dataset=dataset,
                upstream_class=expect_type(entry["class"], str, f"{label}.class"),
                modality="multimodal",
                num_shot=expect_type(entry["num_shot"], int, f"{label}.num_shot"),
                demonstration_type=expect_type(
                    demonstration["type"], str, f"{label}.demonstration.type"
                ),
                demonstration_answer_format=expect_type(
                    entry["answer_format"], str, f"{label}.answer_format"
                ),
                include_demonstration_images=expect_type(
                    demonstration["include_images"],
                    bool,
                    f"{label}.demonstration.include_images",
                ),
                prepend_reasoning_cue=expect_type(
                    demonstration.get("prepend_reasoning_cue", False),
                    bool,
                    f"{label}.demonstration.prepend_reasoning_cue",
                ),
                scored_limit=None,
                result_reader=expect_type(entry["result_reader"], str, f"{label}.result_reader"),
                expected_result_suffix=expect_type(
                    entry["result_suffix"], str, f"{label}.result_suffix"
                ),
                result_metric_unit=expect_type(
                    entry["result_metric_unit"], str, f"{label}.result_metric_unit"
                ),
                judge=expect_type(entry["judge"], str, f"{label}.judge"),
                resource=expect_type(entry["resource"], str, f"{label}.resource"),
                generation=_load_request(
                    entry.get("request", {}),
                    label=f"{label}.request",
                    defaults=generation,
                ),
                server_context_length=context_length,
                prompt_adapter=prompt_adapter,
                demonstration_indices=demonstration_indices,
            )
        )
    return BenchmarkSuiteConfig(
        name=expect_type(suite["name"], str, "suite.name"),
        source_path=str(path),
        benchmarks=tuple(benchmarks),
    )


def _load_conditional_perplexity_suite(path: Path, suite: dict[str, Any]) -> BenchmarkSuiteConfig:
    """Load one conditional-perplexity suite without generation-only fields."""
    request_raw = expect_mapping(
        suite["request"], {"instruction", "temperature", "top_p", "retry"}, set(), "suite.request"
    )
    request = ConditionalPerplexityRequestConfig(
        instruction=expect_type(request_raw["instruction"], str, "request.instruction"),
        temperature=float(expect_number(request_raw["temperature"], "request.temperature")),
        top_p=float(expect_number(request_raw["top_p"], "request.top_p")),
        retry=expect_type(request_raw["retry"], int, "request.retry"),
    )
    entries = expect_type(suite["benchmarks"], list, "suite.benchmarks")
    if not entries:
        raise EvalConfigError("suite must contain at least one benchmark")
    benchmarks: list[ConditionalPerplexityBenchmarkConfig] = []
    for index, raw_entry in enumerate(entries):
        label = f"suite.benchmarks[{index}]"
        entry = expect_mapping(
            raw_entry, {"dataset", "format", "data", "category", "resource"}, set(), label
        )
        dataset = expect_type(entry["dataset"], str, f"{label}.dataset")
        benchmarks.append(
            ConditionalPerplexityBenchmarkConfig(
                name=dataset.lower().replace("-", "_").replace(" ", "_"),
                dataset=dataset,
                data_format=expect_type(entry["format"], str, f"{label}.format"),
                configured_data_path=expect_type(entry["data"], str, f"{label}.data"),
                category_field=expect_type(entry["category"], str, f"{label}.category"),
                resource=expect_type(entry["resource"], str, f"{label}.resource"),
                scored_limit=None,
                request=request,
            )
        )
    return BenchmarkSuiteConfig(
        name=expect_type(suite["name"], str, "suite.name"),
        source_path=str(path),
        benchmarks=tuple(benchmarks),
    )


__all__ = ("load_benchmark_suite", "load_cluster_config", "load_model_config")
