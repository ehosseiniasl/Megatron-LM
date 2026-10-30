#!/usr/bin/env python3
"""
Standalone script to rescore MultiPL-E samples from a previous evaluation run.

This is useful when:
1. The original evaluation was run without compilers installed
2. You want to re-evaluate with different timeout settings
3. You need to recompute metrics after fixing evaluation bugs

Usage:
    python rescore_multiple.py samples_multiple-js.jsonl
    python rescore_multiple.py samples_multiple-*.jsonl
    python rescore_multiple.py samples.jsonl --output rescored.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

# Enable code evaluation
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from lm_eval.metrics.code_eval import estimate_pass_at_k_dynamic, evaluate_candidates
from lm_eval.metrics.multiple_metrics.containerized_eval import eval_string_script
from lm_eval.tasks.multiple.multiple_utils import (
    build_program,
    check_compiler_available,
    LANGUAGE_COMPILERS,
)


def process_resp_for_rescore(repeat_idx: int, resp: str, doc: dict) -> dict:
    """Process a response for rescoring - always attempts evaluation."""
    language = doc.get("language", "py")
    program = build_program(doc["prompt"], resp, doc["tests"])

    # Check compiler availability and warn (but still try)
    if not check_compiler_available(language):
        compiler = LANGUAGE_COMPILERS.get(language, "unknown")
        print(f"Warning: Compiler '{compiler}' for '{language}' not found", file=sys.stderr)
        return {
            "program": program,
            "stdout": "",
            "stderr": f"Compiler '{compiler}' not found for language '{language}'",
            "exit_code": -1,
            "status": "CompilerNotFound",
            "passed": False,
            "task_id": doc["name"],
            "completion_id": repeat_idx,
        }

    result = eval_string_script(language, program)
    result["passed"] = result["status"] == "OK"
    result["task_id"] = doc["name"]
    result["completion_id"] = repeat_idx
    return result


def rescore_file(input_path: str, output_path: str | None = None) -> dict:
    """
    Rescore a samples JSONL file.

    Args:
        input_path: Path to the input samples JSONL file
        output_path: Optional path for output file (defaults to input_path with .rescored suffix)

    Returns:
        Dictionary with aggregated metrics
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".rescored.jsonl")
    else:
        output_path = Path(output_path)

    print(f"Rescoring: {input_path}")
    print(f"Output: {output_path}")

    # Read all samples
    docs = []
    resps = []
    original_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            original_data.append(data)
            docs.append(data["doc"])
            # Extract the raw responses (before filtering)
            # resps format: [[resp1, resp2, ...]] - outer list has one element containing all responses
            raw_resps = data["resps"][0] if data["resps"] else []
            # Each response may be a string or [string, ...] tuple
            resps.append([r[0] if isinstance(r, list) else r for r in raw_resps])

    if not docs:
        print(f"No samples found in {input_path}", file=sys.stderr)
        return {}

    # Detect language from first doc
    language = docs[0].get("language", "unknown")
    print(f"Language: {language}")
    print(f"Documents: {len(docs)}")
    print(f"Samples per doc: {len(resps[0]) if resps else 0}")

    # Check compiler availability
    if not check_compiler_available(language):
        compiler = LANGUAGE_COMPILERS.get(language, "unknown")
        print(f"ERROR: Compiler '{compiler}' for '{language}' not found!", file=sys.stderr)
        print("Install the compiler and try again.", file=sys.stderr)
        return {}

    # Re-evaluate all samples
    print("Evaluating samples...")
    filtered_resps = evaluate_candidates(resps, docs, process_resp_for_rescore)

    # Compute metrics and write output
    metrics_agg = defaultdict(list)
    status_counts = defaultdict(int)

    with open(output_path, "w", encoding="utf-8") as f:
        for orig_data, doc, filtered_resp in zip(original_data, docs, filtered_resps):
            # Compute pass@k for this document
            metrics = estimate_pass_at_k_dynamic(doc, [filtered_resp])

            # Count statuses
            for r in filtered_resp:
                status_counts[r.get("status", "Unknown")] += 1

            # Aggregate metrics
            for k, v in metrics.items():
                metrics_agg[k].append(v)

            # Update the original data with new results
            orig_data["filtered_resps"] = filtered_resp
            orig_data["metrics"] = metrics

            f.write(json.dumps(orig_data) + "\n")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nStatus Distribution:")
    total_samples = sum(status_counts.values())
    for status, count in sorted(status_counts.items()):
        pct = 100 * count / total_samples if total_samples > 0 else 0
        print(f"  {status}: {count} ({pct:.1f}%)")

    print("\nPass@k Metrics:")
    results = {}
    for k, values in sorted(metrics_agg.items()):
        mean_val = sum(values) / len(values) if values else 0
        results[k] = mean_val
        print(f"  {k}: {mean_val:.4f}")

    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Rescore MultiPL-E samples from a previous evaluation run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input samples JSONL file(s). Supports glob patterns.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (only valid with single input file)",
    )
    args = parser.parse_args()

    # Expand glob patterns
    input_files = []
    for pattern in args.input_files:
        expanded = glob(pattern)
        if expanded:
            input_files.extend(expanded)
        else:
            input_files.append(pattern)

    if len(input_files) > 1 and args.output:
        parser.error("--output can only be used with a single input file")

    all_results = {}
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}", file=sys.stderr)
            continue

        results = rescore_file(input_file, args.output if len(input_files) == 1 else None)
        if results:
            all_results[input_file] = results

    # Print combined summary if multiple files
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("COMBINED SUMMARY")
        print("=" * 60)
        for fname, results in all_results.items():
            print(f"\n{Path(fname).name}:")
            for k, v in sorted(results.items()):
                if "pass@1[" in k:
                    print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
