#!/usr/bin/env python3

import argparse
import asyncio
import hashlib
import json
import math
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from lm_eval.api.judge import (
    DEFAULT_JUDGE_CONCURRENCY,
    JUDGE_PROVIDERS,
    AsyncOpenAIJudge,
    get_judge_provider,
)
from lm_eval.tasks.parametric_qa.grading import (
    FACTS_PARAMETRIC,
    GRADER_SPECS,
    SIMPLEQA_VERIFIED,
    build_judge_request,
    canonical_task_name,
    parse_structured_grade,
)


TASK_NAMES = {
    SIMPLEQA_VERIFIED: "simpleqa_verified_25shot",
    FACTS_PARAMETRIC: "facts_parametric_25shot",
}
EXPECTED_SAMPLES = {
    SIMPLEQA_VERIFIED: 1000,
    FACTS_PARAMETRIC: 1052,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Grade logged SimpleQA Verified or FACTS Parametric generations "
            "after GPU-backed inference has finished."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Sample JSONL files or directories to search recursively.",
    )
    parser.add_argument(
        "--judge-model",
        default="auto",
        help=("Judge model slug. 'auto' uses openai/gpt-oss-120b."),
    )
    parser.add_argument(
        "--provider",
        choices=sorted(JUDGE_PROVIDERS),
        default="openrouter",
        help="OpenAI-compatible judge provider.",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible judge endpoint; overrides the provider default.",
    )
    parser.add_argument(
        "--api-key-env",
        help="Environment variable containing the endpoint API key.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_JUDGE_CONCURRENCY,
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument(
        "--replicates",
        type=int,
        help="Override judge calls per example (defaults: SimpleQA=1, FACTS=3).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Judge temperature. Omitted by default to preserve provider defaults.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        help="Cache path; valid only when grading one sample file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for judged results JSON (default: beside each sample file).",
    )
    parser.add_argument(
        "--filter-name",
        help="Filter suffix used in lm-eval result metric names (defaults to provider).",
    )
    parser.add_argument(
        "--no-local-fast-path",
        action="store_true",
        help="Send empty and exact-reference responses to the judge as well.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report pending call counts without using the API.",
    )
    args = parser.parse_args()
    provider = get_judge_provider(args.provider)
    args.base_url = args.base_url or provider.base_url
    args.api_key_env = args.api_key_env or provider.api_key_envs
    args.supports_response_format = provider.supports_response_format
    args.default_headers = provider.default_headers
    args.filter_name = args.filter_name or args.provider
    return args


def discover_sample_files(inputs):
    files = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            files.extend(path.rglob("samples_simpleqa_verified_25shot_*.jsonl"))
            files.extend(path.rglob("samples_facts_parametric_25shot_*.jsonl"))
            continue
        raise FileNotFoundError(path)
    return sorted(set(path.resolve() for path in files))


def extract_prediction(sample):
    responses = sample.get("filtered_resps", [])
    if not responses:
        return ""
    response = responses[0]
    while isinstance(response, list):
        if not response:
            return ""
        response = response[0]
    return str(response)


def load_samples(path):
    samples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            doc = sample.get("doc", {})
            question = doc.get("question") or doc.get("problem") or doc.get("query")
            gold_answer = doc.get("answer") or sample.get("target")
            if question is None or gold_answer is None:
                raise ValueError(
                    f"{path}:{line_number}: missing question or gold answer"
                )
            samples.append(
                {
                    "doc_id": sample.get("doc_id", len(samples)),
                    "task_hint": doc.get("_parametric_qa_task", path.name),
                    "question": str(question),
                    "gold_answer": str(gold_answer),
                    "prediction": extract_prediction(sample),
                }
            )
    if not samples:
        raise ValueError(f"No samples found in {path}")
    task = canonical_task_name(samples[0]["task_hint"])
    if any(canonical_task_name(sample["task_hint"]) != task for sample in samples):
        raise ValueError(f"Mixed tasks in {path}")
    return task, samples


def load_source_run_metadata(path, task, num_samples):
    task_name = TASK_NAMES[task]
    prefix = f"samples_{task_name}_"
    suffix = path.name.removeprefix(prefix).removesuffix(".jsonl")
    result_path = path.with_name(f"results_{suffix}.json")
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            sample_counts = result["n-samples"][task_name]
            return {
                "limit": result["config"].get("limit"),
                "original": sample_counts["original"],
                "effective": sample_counts["effective"],
                "source_results": str(result_path),
            }
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Could not read source-run metadata from {result_path}: {error}"
            ) from error

    expected = EXPECTED_SAMPLES[task]
    return {
        "limit": None if num_samples == expected else num_samples,
        "original": expected,
        "effective": num_samples,
        "source_results": None,
    }


def request_key(task, model, prompt_version, sample, replicate):
    payload = {
        "task": task,
        "model": model,
        "prompt_version": prompt_version,
        "question": sample["question"],
        "gold_answer": sample["gold_answer"],
        "prediction": sample["prediction"],
        "replicate": replicate,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_cache(path):
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if record.get("status") == "ok" and record.get("grade"):
                records[record["key"]] = record
    return records


def local_fast_grade(task, sample):
    prediction = sample["prediction"].strip()
    if not prediction:
        return "NOT_ATTEMPTED"
    prediction = " ".join(
        unicodedata.normalize("NFKC", prediction).casefold().split()
    ).rstrip(" .,!?:;")
    gold_answer = " ".join(
        unicodedata.normalize("NFKC", sample["gold_answer"]).casefold().split()
    ).rstrip(" .,!?:;")
    if prediction == gold_answer:
        return "CORRECT"
    return None


def metric_values(task, records_by_item, num_samples):
    grades = [
        record["grade"]
        for item_records in records_by_item.values()
        for record in item_records
    ]
    correct = sum(grade == "CORRECT" for grade in grades)
    not_attempted = sum(grade == "NOT_ATTEMPTED" for grade in grades)
    unknown = sum(grade == "UNKNOWN" for grade in grades)
    total = len(grades)
    attempted = total - not_attempted

    accuracy = correct / total if total else 0.0
    attempted_accuracy = correct / attempted if attempted else 0.0
    denominator = accuracy + attempted_accuracy
    f1 = 2 * accuracy * attempted_accuracy / denominator if denominator else 0.0
    metrics = {
        "judge_accuracy": accuracy,
        "judge_attempted_accuracy": attempted_accuracy,
        "judge_f1": f1,
        "judge_hedging_rate": not_attempted / total if total else 0.0,
    }
    if task == FACTS_PARAMETRIC:
        metrics["judge_unknown_rate"] = unknown / total if total else 0.0
        metrics["judge_strict_accuracy"] = (
            sum(
                len(item_records) > 0
                and all(record["grade"] == "CORRECT" for record in item_records)
                for item_records in records_by_item.values()
            )
            / num_samples
        )
    return metrics


def write_results(
    output_path,
    task,
    model,
    prompt_version,
    replicates,
    metrics,
    usage,
    filter_name,
    sample_path,
    source_run,
    base_url,
    temperature,
):
    task_name = TASK_NAMES[task]
    task_results = {
        f"{metric},{filter_name}": value for metric, value in metrics.items()
    }
    task_results["alias"] = task_name
    higher_is_better = {
        f"{metric},{filter_name}": metric
        not in {"judge_hedging_rate", "judge_unknown_rate"}
        for metric in metrics
    }
    result = {
        "results": {task_name: task_results},
        "group_subtasks": {},
        "configs": {task_name: {"num_fewshot": 25}},
        "versions": {task_name: 1.0},
        "n-shot": {task_name: 25},
        "higher_is_better": {task_name: higher_is_better},
        "n-samples": {
            task_name: {
                "original": source_run["original"],
                "effective": source_run["effective"],
            }
        },
        "config": {
            "limit": source_run["limit"],
            "deferred_judge": {
                "endpoint": base_url,
                "model": model,
                "prompt_version": prompt_version,
                "replicates": replicates,
                "temperature": temperature,
                "source_samples": str(sample_path),
                "source_results": source_run["source_results"],
                "usage": usage,
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


async def grade_file(path, args):
    task, samples = load_samples(path)
    source_run = load_source_run_metadata(path, task, len(samples))
    spec = GRADER_SPECS[task]
    model = spec.default_model if args.judge_model == "auto" else args.judge_model
    replicates = args.replicates or spec.replicates
    if replicates < 1:
        raise ValueError("--replicates must be positive")

    provider = getattr(args, "provider", "openrouter")
    cache_path = args.cache_path or path.with_suffix(f".{provider}-cache.jsonl")
    cache = load_cache(cache_path)
    records_by_item = {index: [] for index in range(len(samples))}
    pending = []

    for index, sample in enumerate(samples):
        for replicate in range(replicates):
            key = request_key(task, model, spec.prompt_version, sample, replicate)
            if key in cache:
                records_by_item[index].append(cache[key])
                continue
            fast_grade = (
                None if args.no_local_fast_path else local_fast_grade(task, sample)
            )
            if fast_grade is not None:
                record = {
                    "key": key,
                    "status": "ok",
                    "source": "local_fast_path",
                    "task": task,
                    "model": model,
                    "prompt_version": spec.prompt_version,
                    "doc_id": sample["doc_id"],
                    "sample_index": index,
                    "replicate": replicate,
                    "grade": fast_grade,
                    "raw_response": "",
                    "usage": {},
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                records_by_item[index].append(record)
                pending.append((None, record))
                continue
            pending.append((key, index, replicate, sample))

    api_call_count = sum(len(item) == 4 for item in pending)
    cached_count = sum(len(records) for records in records_by_item.values())
    local_count = sum(
        len(item) == 2 and item[1].get("source") == "local_fast_path"
        for item in pending
    )
    print(
        f"{path}: task={task}, samples={len(samples)}, model={model}, "
        f"replicates={replicates}, cached={cached_count}, "
        f"local_fast_path={local_count}, pending_api_calls={api_call_count}"
    )
    if args.dry_run:
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as cache_handle:
        for item in pending:
            if len(item) == 2:
                cache_handle.write(json.dumps(item[1], ensure_ascii=False) + "\n")
        cache_handle.flush()

        if api_call_count:
            judge = AsyncOpenAIJudge(
                model=model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                concurrency=args.concurrency,
                timeout=args.timeout,
                max_retries=args.max_retries,
                supports_response_format=getattr(
                    args,
                    "supports_response_format",
                    True,
                ),
                default_headers=getattr(args, "default_headers", None),
            )
            requests = [
                build_judge_request(
                    task,
                    sample,
                    replicate,
                    temperature=args.temperature,
                    metadata=(key, index, replicate, sample),
                )
                for key, index, replicate, sample in (
                    item for item in pending if len(item) == 4
                )
            ]
            failures = []

            def record_result(result):
                key, index, replicate, sample = result.request.metadata
                if result.error is None:
                    record = {
                        "key": key,
                        "status": "ok",
                        "source": "judge",
                        "task": task,
                        "model": model,
                        "prompt_version": spec.prompt_version,
                        "doc_id": sample["doc_id"],
                        "sample_index": index,
                        "replicate": replicate,
                        "grade": result.value,
                        "raw_response": result.raw_response,
                        "usage": result.usage,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    records_by_item[index].append(record)
                    cache_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    cache_handle.flush()
                else:
                    failures.append((index, replicate, result.error))

            await judge.run(
                requests,
                lambda response: parse_structured_grade(task, response),
                on_result=record_result,
            )
            if failures:
                examples = "; ".join(
                    f"sample={index} replicate={replicate}: {message}"
                    for index, replicate, message in failures[:5]
                )
                raise RuntimeError(
                    f"{len(failures)} judge calls failed; rerun to resume. {examples}"
                )

    expected = len(samples) * replicates
    actual = sum(len(records) for records in records_by_item.values())
    if actual != expected:
        raise RuntimeError(f"Expected {expected} grades, found {actual}")
    for records in records_by_item.values():
        records.sort(key=lambda record: record["replicate"])

    metrics = metric_values(task, records_by_item, len(samples))
    usage = {}
    for records in records_by_item.values():
        for record in records:
            for key, value in record.get("usage", {}).items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    usage[key] = usage.get(key, 0) + value

    output_dir = args.output_dir or path.parent
    output_path = output_dir / f"results_judged_{TASK_NAMES[task]}.json"
    write_results(
        output_path,
        task,
        model,
        spec.prompt_version,
        replicates,
        metrics,
        usage,
        args.filter_name,
        path,
        source_run,
        args.base_url,
        args.temperature,
    )
    print(f"Wrote {output_path}")
    for metric, value in sorted(metrics.items()):
        print(f"  {metric}: {value:.6f}")


async def async_main(args):
    files = discover_sample_files(args.inputs)
    if not files:
        raise FileNotFoundError("No matching parametric QA sample files found")
    if args.cache_path and len(files) != 1:
        raise ValueError("--cache-path requires exactly one sample file")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")

    output_targets = {}
    for path in files:
        task = canonical_task_name(path.name)
        output_dir = (args.output_dir or path.parent).resolve()
        target = output_dir / f"results_judged_{TASK_NAMES[task]}.json"
        if target in output_targets:
            raise ValueError(
                f"{output_targets[target]} and {path} would both write {target}; "
                "pass one sample file explicitly or use separate output directories"
            )
        output_targets[target] = path

    for path in files:
        await grade_file(path, args)


def main():
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
