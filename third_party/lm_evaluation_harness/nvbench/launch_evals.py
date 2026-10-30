import argparse
import copy
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml
from assemble_results import load_results, tabulate_results


def parse_endpoint_preset(preset: str) -> tuple:
    """Parse an endpoint preset string into (engine, dynamic, concurrency, batch_size).

    Format: {engine}[_dyn[_c{concurrency}][_b{batch_size}]]

    Bare numbers after _dyn are interpreted as concurrency (backward compat).
    Prefixed parameters: c{N} = concurrency, b{N} = batch_size.
    batch_size=None means "use the task's configured batch_size".

    Examples:
        vllm              -> ("vllm", False, 1, None)
        megatron          -> ("megatron", False, 1, None)
        vllm_dyn          -> ("vllm", True, 64, 8)
        vllm_dyn_32       -> ("vllm", True, 32, 8)
        vllm_dyn_c32      -> ("vllm", True, 32, 8)
        vllm_dyn_b4       -> ("vllm", True, 64, 4)
        vllm_dyn_c64_b8   -> ("vllm", True, 64, 8)
        megatron_dyn_b4   -> ("megatron", True, 64, 4)
    """
    DEFAULT_CONCURRENCY = 64
    DEFAULT_DYN_BATCH_SIZE = 8
    VALID_ENGINES = ("vllm", "megatron")
    parts = preset.split("_")
    engine = parts[0]
    assert engine in VALID_ENGINES, (
        f"Unknown engine '{engine}'. Expected one of {VALID_ENGINES}"
    )
    dynamic = len(parts) >= 2 and parts[1] == "dyn"
    if not dynamic:
        return engine, False, 1, None

    concurrency = DEFAULT_CONCURRENCY
    batch_size = DEFAULT_DYN_BATCH_SIZE
    for part in parts[2:]:
        if part.startswith("c"):
            concurrency = int(part[1:])
        elif part.startswith("b"):
            batch_size = int(part[1:])
        else:
            # Bare number = concurrency (backward compat)
            concurrency = int(part)
    return engine, dynamic, concurrency, batch_size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help=(
            "Path to nvbench yaml config. "
            "See https://gitlab-master.nvidia.com/swdl-nemollm-mlops/evals/lm-evaluation-harness/-/tree/adlr/nemo5/nvbench "
            "for examples."
        ),
    )
    parser.add_argument(
        "-r", "--results_dir", type=str, required=True, help="Path to save results"
    )
    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        required=True,
        help=(
            "Endpoint preset: {engine}[_dyn[_c{concurrency}][_b{batch_size}]]. "
            "Examples: vllm, megatron, vllm_dyn, vllm_dyn_c64, vllm_dyn_b8, "
            "vllm_dyn_c64_b8. In dynamic mode (_dyn), requests are sent with N "
            "concurrent in-flight (default c=64, b=8). Bare number after _dyn "
            "is interpreted as concurrency for backward compat (vllm_dyn_64)."
        ),
    )
    parser.add_argument(
        "-i",
        "--ip",
        type=str,
        default="127.0.0.1",
        required=True,
        help="IP address of model server",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        default="5000",
        required=True,
        help="Port number of model server",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to evaluate, used as a key for caching results",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face tokenizer ID or local path. Required for "
            "megatron/notok-completions loglikelihood_rolling tasks; vllm "
            "defaults to the model name when omitted."
        ),
    )
    parser.add_argument(
        "--llm-compression-data-dir",
        default=None,
        help="Local LC26 subset directory; avoids the private NVIDIA HF dataset.",
    )
    parser.add_argument(
        "--mistral-ppl-data-dir",
        default=None,
        help="Local raw arXiv-math and AFP-crawl evaluation data directory.",
    )
    parser.add_argument(
        "-t",
        dest="task_selection_range",
        type=str,
        required=False,
        help=(
            "Task selection range using Python slice notation. Examples: "
            "-t 0 (select task 0), -t 0:3 (select tasks 0,1,2), "
            "-t 2: (select from task 2 to end), -t :5 (select first 5 tasks), "
            "-t 1:9:2 (select tasks 1,3,5,7 with step=2)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit each selected task to this many examples (or a fraction below 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size configured for every selected task.",
    )
    parser.add_argument(
        "--log-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-sample outputs in addition to aggregate metrics.",
    )
    return parser.parse_args()


def megatron_main_model_args(
    ip: str,
    port: str,
    model_name: str,
    batch_size: int,
    max_concurrent_per_client: int = 1,
    tokenizer: str | None = None,
) -> str:
    endpoints_str = f"http://{ip}:{port}"
    model_args = [
        f"base_url={endpoints_str}",
        f"model={model_name}",
    ]
    if tokenizer:
        model_args.append(f"tokenizer={tokenizer}")
    model_args.append(f"max_concurrent_per_client={max_concurrent_per_client}")

    return (
        "--model notok-completions "
        f"--model_args {','.join(model_args)}"
    )


def vllm_model_args(
    ip: str,
    port: str,
    model_name: str,
    batch_size: int,
    max_concurrent_per_client: int = 1,
    tokenizer: str | None = None,
) -> str:
    tokenizer_name = tokenizer or model_name
    return (
        "--model local-completions "
        f"--model_args base_url=http://{ip}:{port}/v1,"
        f"model={model_name},"
        f"tokenizer={tokenizer_name},"
        f"tokenizer_backend=huggingface,"
        f"max_concurrent_per_client={max_concurrent_per_client}"
    )


def parse_task_range(range_spec: str) -> slice:
    """Parse task range specification like '0', '0:3', '0:9' into proper slice"""
    if ":" not in range_spec:
        # Single index: "0" -> slice(0, 1)
        idx = int(range_spec)
        return slice(idx, idx + 1)

    # Range specification: "0:3" -> slice(0, 3)
    parts = range_spec.split(":")
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if len(parts) > 1 and parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None

    return slice(start, end, step)


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Get results dir from config
    output_path_ = Path(args.results_dir) / args.model_name
    results = load_results(output_path_, config, filter_to_config=False)

    # Get list of completed tasks from results
    completed_tasks = {k[0] for k in results}

    engine, dynamic, concurrency, preset_batch_size = parse_endpoint_preset(args.endpoint)
    model_args_fn = {
        "megatron": megatron_main_model_args,
        "vllm": vllm_model_args,
    }[engine]

    print(f"Endpoint preset: engine={engine}, dynamic={dynamic}, concurrency={concurrency}, batch_size={preset_batch_size}")

    if args.task_selection_range is None:
        tasks = config["tasks"]
    else:
        task_selector = parse_task_range(args.task_selection_range)
        tasks = config["tasks"][task_selector]

    failed_tasks = []

    # Loop through tasks
    for task in tasks:
        task_args = copy.deepcopy(task)
        task_name = task_args.pop("task_name")
        batch_size = task_args.pop("batch_size", None)
        use_cache = task_args.pop("use_cache", True)  # default to caching
        overrides = task_args.pop("overrides", {})
        generation_kwargs = task_args.pop("generation_kwargs", None)
        use_cache &= task_args.pop(
            "cache_requests", False
        )  # dont cache requests, only responses.
        for k in ["report_metric", "task_group"]:
            task_args.pop(k, None)

        effective_batch_size = args.batch_size or batch_size or preset_batch_size or 1
        model_args = model_args_fn(
            args.ip, args.port, args.model_name, effective_batch_size,
            max_concurrent_per_client=concurrency,
            tokenizer=args.tokenizer,
        )

        # Skip if already completed
        if task_name in completed_tasks:
            print(f"Skipping {task_name} - already completed")
            continue

        args_str = ""
        for key, value in task_args.items():
            args_str += f"--{key}={value} "

        if args.limit is not None:
            args_str += f"--limit {args.limit} "

        if use_cache:
            args_str += f"--use_cache {output_path_}/cache_{task_name}_ "

        overrides_dict = overrides if isinstance(overrides, dict) else {}
        if isinstance(generation_kwargs, dict):
            for key, value in generation_kwargs.items():
                overrides_dict[f"generation_kwargs.{key}"] = value

        if overrides_dict:
            override_str = " ".join(f"{k}={v}" for k, v in overrides_dict.items())
            args_str += f"--override \"{override_str}\""

        bash_cmd = (
            f"{sys.executable} -m lm_eval "
            f"{model_args} "
            f"--tasks {task_name} "
            f"--batch_size {effective_batch_size} "
            f"{args_str} "
            f"{'--log_samples ' if args.log_samples else ''}"
            f"--output_path {output_path_} "
        )

        # Run evaluation
        try:
            env = os.environ.copy()
            env["HF_ALLOW_CODE_EVAL"] = "1"
            if args.llm_compression_data_dir:
                env["LM_EVAL_LLM_COMPRESSION_2026_DATA_DIR"] = (
                    args.llm_compression_data_dir
                )
            if args.mistral_ppl_data_dir:
                env["LM_EVAL_MISTRAL_PPL_DATA_DIR"] = args.mistral_ppl_data_dir
            print(f"Launching evaluation for {task_name} with command: {bash_cmd}")
            subprocess.run(shlex.split(bash_cmd), check=True, env=env)
            print(f"Completed evaluation for {task_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {task_name}: {e}")
            failed_tasks.append(task_name)

    results = load_results(output_path_, config)
    tabulate_results(results)
    if failed_tasks:
        raise SystemExit(f"Failed evaluation tasks: {', '.join(failed_tasks)}")


if __name__ == "__main__":
    main()
