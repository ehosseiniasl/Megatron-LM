"""Access configured benchmark plans and bind one entry to a Slurm array element."""

from __future__ import annotations

import math
from dataclasses import replace

from .chat_template_policy import configured_chat_template_path
from .configured_eval_run import (
    BenchmarkArraySelection,
    BenchmarkConfig,
    BenchmarkSuiteConfig,
    ConditionalPerplexityBenchmarkConfig,
    ConfiguredEvalRun,
    EvalConfigError,
    GenerationConfig,
    configured_resource,
    run_for_resource,
    validate_configured_eval_run,
)
from .utils import (
    SAFE_IDENTIFIER_PATTERN,
    VLMEvalRowIdentityError,
    require_unique_vlmeval_row_indices,
)
from .vlmeval_result_contract import validate_benchmark_result_contract


def configured_benchmarks(
    config: ConfiguredEvalRun,
) -> tuple[BenchmarkConfig | ConditionalPerplexityBenchmarkConfig, ...]:
    """Return the ordered benchmark plan from ``config`` without mutating it.

    ``config`` must be an exact configured-run record that selects either one
    benchmark or a nonempty all-``BenchmarkConfig`` suite. Variant violations
    raise :class:`EvalConfigError`.
    """
    if type(config) is not ConfiguredEvalRun:
        raise EvalConfigError("configured run has the wrong record type")
    single = config.benchmark
    suite = config.benchmark_suite
    supported = (BenchmarkConfig, ConditionalPerplexityBenchmarkConfig)
    if type(single) in supported and suite is None:
        return (single,)
    if single is None and type(suite) is BenchmarkSuiteConfig:
        benchmarks = suite.benchmarks
        if not benchmarks or any(type(item) not in supported for item in benchmarks):
            raise EvalConfigError("benchmark suite contains unsupported entry records")
        if any(type(item) is not type(benchmarks[0]) for item in benchmarks[1:]):
            raise EvalConfigError("benchmark suite cannot mix evaluation kinds")
        return benchmarks
    raise EvalConfigError("run must select exactly one benchmark or benchmark suite")


def configured_benchmark_suite(config: ConfiguredEvalRun) -> BenchmarkSuiteConfig | None:
    """Return ``config``'s validated suite, or ``None`` for a single run.

    The input record is never mutated. An invalid single-versus-suite variant
    raises :class:`EvalConfigError`.
    """
    configured_benchmarks(config)
    return config.benchmark_suite


def common_generation_config(config: ConfiguredEvalRun) -> GenerationConfig:
    """Return the generation contract for one executable benchmark.

    Suite planning may contain dataset-specific request settings, but array
    binding normally reduces every executable run to one benchmark. An
    unbound suite remains executable only when all entries share one request
    contract. The input record is never mutated.
    """
    benchmarks = configured_benchmarks(config)
    if any(type(item) is not BenchmarkConfig for item in benchmarks):
        raise EvalConfigError("conditional-perplexity suites have no generation config")
    generation = benchmarks[0].generation
    if any(item.generation != generation for item in benchmarks[1:]):
        raise EvalConfigError("mixed benchmark request settings require one selected benchmark")
    return generation


def _validate_benchmark(
    config: ConfiguredEvalRun, benchmark: BenchmarkConfig | ConditionalPerplexityBenchmarkConfig
) -> None:
    if type(benchmark) is ConditionalPerplexityBenchmarkConfig:
        configured_resource(config, benchmark.resource)
        if benchmark.data_format != "packed_jsonl_tar":
            raise EvalConfigError("unsupported conditional-perplexity data format")
        if benchmark.category_field != "image_type":
            raise EvalConfigError("unsupported conditional-perplexity category field")
        if not benchmark.configured_data_path.startswith(("/", "${workspace}/")):
            raise EvalConfigError("conditional-perplexity data path must be absolute")
        if benchmark.scored_limit is not None and benchmark.scored_limit <= 0:
            raise EvalConfigError("scored_limit must be positive")
        request = benchmark.request
        if not request.instruction or not request.instruction.strip():
            raise EvalConfigError("conditional-perplexity instruction must be nonempty")
        if not math.isfinite(request.temperature) or request.temperature != 0:
            raise EvalConfigError("conditional perplexity requires temperature zero")
        if not math.isfinite(request.top_p) or request.top_p != 1:
            raise EvalConfigError("conditional perplexity requires top_p one")
        if request.retry <= 0:
            raise EvalConfigError("retry must be a positive maximum attempt count")
        return
    validate_benchmark_result_contract(benchmark)
    configured_resource(config, benchmark.resource)
    if type(benchmark.num_shot) is not int or benchmark.num_shot < 0:
        raise EvalConfigError("num_shot must be nonnegative")
    if benchmark.scored_limit is not None and benchmark.scored_limit <= 0:
        raise EvalConfigError("scored_limit must be positive")
    if benchmark.server_context_length is not None and (
        type(benchmark.server_context_length) is not int or benchmark.server_context_length <= 0
    ):
        raise EvalConfigError("benchmark server context override must be a positive integer")
    if benchmark.demonstration_type not in {"benchmark", "reasoning", "rich", "trivial"}:
        raise EvalConfigError("unsupported demonstration type")
    try:
        demonstration_indices = require_unique_vlmeval_row_indices(
            benchmark.demonstration_indices, label="demonstration indices"
        )
    except VLMEvalRowIdentityError as error:
        raise EvalConfigError(str(error)) from error
    if demonstration_indices:
        if benchmark.demonstration_type != "benchmark":
            raise EvalConfigError("explicit demonstration indices require benchmark demonstrations")
        if len(demonstration_indices) != benchmark.num_shot:
            raise EvalConfigError("explicit demonstration index count must equal num_shot")
    if type(benchmark.include_demonstration_images) is not bool:
        raise EvalConfigError("include_demonstration_images must be boolean")
    if type(benchmark.prepend_reasoning_cue) is not bool:
        raise EvalConfigError("prepend_reasoning_cue must be boolean")
    if benchmark.prepend_reasoning_cue and benchmark.demonstration_type != "reasoning":
        raise EvalConfigError("prepend_reasoning_cue requires reasoning demonstrations")
    if benchmark.demonstration_answer_format not in {
        "raw",
        "label",
        "value",
        "screenspot_click",
        "screenspot_click_normalized",
    }:
        raise EvalConfigError("unsupported demonstration answer format")
    screenspot_formats = {"screenspot_click", "screenspot_click_normalized"}
    if benchmark.demonstration_answer_format in screenspot_formats and not (
        benchmark.upstream_class == "ScreenSpot"
        and benchmark.dataset in {"ScreenSpot_Mobile", "ScreenSpot_Desktop", "ScreenSpot_Web"}
    ):
        raise EvalConfigError("ScreenSpot click formatting requires a standard ScreenSpot dataset")
    marker_template = "assets/screenspot_shared_gui_end_marker_template.jinja"
    if benchmark.generation.chat_template == marker_template and benchmark.num_shot == 0:
        raise EvalConfigError("ScreenSpot marker template requires few-shot demonstrations")
    if benchmark.prepend_reasoning_cue and benchmark.demonstration_answer_format != "raw":
        raise EvalConfigError("prepend_reasoning_cue requires raw demonstration answers")
    generation = benchmark.generation
    configured_chat_template_path(generation.chat_template)
    if not math.isfinite(generation.temperature) or generation.temperature < 0:
        raise EvalConfigError("temperature must be finite and nonnegative")
    if generation.retry <= 0:
        raise EvalConfigError("retry must be a positive maximum attempt count")
    if generation.image_size == 0 or generation.image_size < -1:
        raise EvalConfigError("image_size must be -1 or positive")
    if type(generation.image_detail) is not str or generation.image_detail not in {"high", "low"}:
        raise EvalConfigError("image_detail must be high or low")
    if generation.max_new_tokens <= 0:
        raise EvalConfigError("max_new_tokens must be positive")
    context_length = benchmark.server_context_length
    if context_length is None:
        context_length = config.model.server.context_length
    if context_length is not None and generation.max_new_tokens > context_length:
        raise EvalConfigError("max_new_tokens exceeds the server context")
    if not generation.stop or any(not item for item in generation.stop):
        raise EvalConfigError("generation stop strings must be nonempty")
    if not benchmark.judge.strip():
        raise EvalConfigError("benchmark judge must be nonempty")
    if config.benchmark_array_selection is not None and config.run.resource != benchmark.resource:
        raise EvalConfigError("selected benchmark and run resource differ")
    if (
        config.benchmark_array_selection is not None
        and benchmark.server_context_length is not None
        and config.model.server.context_length != benchmark.server_context_length
    ):
        raise EvalConfigError("selected benchmark and server context override differ")


def validate_benchmark_plan(config: ConfiguredEvalRun) -> None:
    """Validate ``config``'s benchmark variant and plan without mutation.

    Every benchmark, shared generation setting, suite identity, and any array
    selection must agree. Invalid plan records raise :class:`EvalConfigError`.
    """
    benchmarks = configured_benchmarks(config)
    for benchmark in benchmarks:
        _validate_benchmark(config, benchmark)
    validate_benchmark_array_selection(config)
    suite = config.benchmark_suite
    if suite is None:
        return
    if not SAFE_IDENTIFIER_PATTERN.fullmatch(suite.name):
        raise EvalConfigError("benchmark suite name must be filesystem-safe")
    if not suite.source_path.startswith("/"):
        raise EvalConfigError("suite source path must be absolute")
    datasets = tuple(item.dataset for item in benchmarks)
    if len(datasets) != len(set(datasets)):
        raise EvalConfigError("benchmark suite datasets must be unique")


def _selection_for_index(
    config: ConfiguredEvalRun, benchmark_index: int
) -> BenchmarkArraySelection:
    benchmarks = configured_benchmarks(config)
    if type(benchmark_index) is not int:
        raise EvalConfigError("benchmark array index must be an integer")
    if benchmark_index < 0 or benchmark_index >= len(benchmarks):
        raise EvalConfigError(
            f"benchmark array index {benchmark_index} is outside " f"0..{len(benchmarks) - 1}"
        )
    suite = config.benchmark_suite
    return BenchmarkArraySelection(
        source_suite_name=None if suite is None else suite.name,
        source_suite_path=None if suite is None else suite.source_path,
        benchmark_index=benchmark_index,
        benchmark_count=len(benchmarks),
        ordered_datasets=tuple(item.dataset for item in benchmarks),
    )


def validate_benchmark_array_selection(config: ConfiguredEvalRun) -> None:
    """Validate an already-bound selection without reading files or mutating data.

    An unbound run is accepted. A bound run must contain exactly one benchmark,
    its original ordered dataset identity, and either complete source-suite
    provenance or the single-benchmark sentinel. Violations raise
    :class:`EvalConfigError`.
    """
    selection = config.benchmark_array_selection
    if selection is None:
        return
    if type(selection) is not BenchmarkArraySelection:
        raise EvalConfigError("benchmark array selection has the wrong record type")
    if config.benchmark is None or config.benchmark_suite is not None:
        raise EvalConfigError("an array-bound run must contain exactly one benchmark")
    index = selection.benchmark_index
    count = selection.benchmark_count
    if type(index) is not int or type(count) is not int or count <= 0:
        raise EvalConfigError("benchmark array index must be an integer and count must be positive")
    if index < 0 or index >= count:
        raise EvalConfigError("benchmark array index is outside its recorded plan")
    datasets = selection.ordered_datasets
    valid_datasets = (
        type(datasets) is tuple
        and len(datasets) == count
        and all(type(item) is str and item and item == item.strip() for item in datasets)
        and len(datasets) == len(set(datasets))
    )
    if not valid_datasets:
        raise EvalConfigError("benchmark array dataset plan is invalid")
    if datasets[index] != config.benchmark.dataset:
        raise EvalConfigError("selected benchmark differs from the array dataset plan")
    suite_values = (selection.source_suite_name, selection.source_suite_path)
    if all(value is None for value in suite_values):
        if count != 1:
            raise EvalConfigError("multi-benchmark selection lacks suite provenance")
        return
    name, path = suite_values
    valid_suite = (
        type(name) is str
        and bool(SAFE_IDENTIFIER_PATTERN.fullmatch(name))
        and type(path) is str
        and path.startswith("/")
    )
    if not valid_suite:
        raise EvalConfigError("suite array selection has incomplete source provenance")


def bind_benchmark_array_selection(
    config: ConfiguredEvalRun, benchmark_index: int, *, scored_limit: int | None = None
) -> ConfiguredEvalRun:
    """Return an independently executable config for one array element.

    The complete input plan is validated before selection. The returned frozen
    record owns one benchmark and immutable provenance for the original ordered
    plan; neither ``config`` nor any nested source record is mutated. Invalid or
    repeated binding raises :class:`EvalConfigError`. ``scored_limit`` is an
    optional positive row cap for bounded debug runs.
    """
    if type(config) is not ConfiguredEvalRun:
        raise EvalConfigError("configured run has the wrong record type")
    if config.benchmark_array_selection is not None:
        raise EvalConfigError("benchmark array selection is already bound")
    if scored_limit is not None and (type(scored_limit) is not int or scored_limit <= 0):
        raise EvalConfigError("scored limit must be a positive integer")
    validate_configured_eval_run(config)
    selection = _selection_for_index(config, benchmark_index)
    benchmark = configured_benchmarks(config)[benchmark_index]
    if scored_limit is not None:
        benchmark = replace(benchmark, scored_limit=scored_limit)
    model = config.model
    if type(benchmark) is BenchmarkConfig and benchmark.server_context_length is not None:
        model = replace(
            model, server=replace(model.server, context_length=benchmark.server_context_length)
        )
    selected = replace(
        config,
        run=run_for_resource(config.run.name, configured_resource(config, benchmark.resource)),
        model=model,
        benchmark=benchmark,
        benchmark_suite=None,
        benchmark_array_selection=selection,
    )
    validate_configured_eval_run(selected)
    return selected


def benchmark_array_run_component(config: ConfiguredEvalRun) -> str:
    """Return a safe run-directory component without mutating ``config``.

    Unbound runs retain their configured name; bound runs add deterministic array
    position and count. Invalid bindings or names longer than 255 bytes raise
    :class:`EvalConfigError`.
    """
    selection = config.benchmark_array_selection
    if selection is None:
        return config.run.name
    validate_benchmark_array_selection(config)
    width = max(2, len(str(selection.benchmark_count)))
    index = f"{selection.benchmark_index:0{width}d}"
    count = f"{selection.benchmark_count:0{width}d}"
    component = f"{config.run.name}__array-{index}-of-{count}"
    if len(component.encode("utf-8")) > 255:
        raise EvalConfigError("benchmark array run-directory component is too long")
    return component


__all__ = (
    "benchmark_array_run_component",
    "bind_benchmark_array_selection",
    "common_generation_config",
    "configured_benchmark_suite",
    "configured_benchmarks",
    "validate_benchmark_array_selection",
    "validate_benchmark_plan",
)
