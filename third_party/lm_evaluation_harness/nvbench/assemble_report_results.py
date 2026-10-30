import argparse
import os
import glob
import json
from pathlib import Path
from typing import Optional

import yaml
import pandas as pd
from tabulate import tabulate


RAW_VALUE_METRICS = {"word_perplexity", "byte_perplexity", "bits_per_byte"}


def scale_metric_value(metric: str, value: float) -> float:
    if metric in RAW_VALUE_METRICS:
        return value
    return value * 100


def load_results(results_dir: str, config: Optional[dict] = None):

    # Find all results*.json files recursively
    results_files = glob.glob(
        str(Path(results_dir) / "**" / "results*.json"), recursive=True
    )

    # Initialize results dict
    all_results = {}

    # Load and process each results file
    for results_file in results_files:
        try:
            with open(results_file) as f:
                data = json.load(f)

            if "results" not in data:
                continue

            if data["config"]["limit"] is not None:
                print(f"Skipping {results_file} - limit is not None")
                continue

            # Process each task's results
            for task_name, task_results in data["results"].items():
                num_fewshot = data["n-shot"].get(task_name, "?")
                version = data["versions"].get(task_name, "?")

                # Add each metric's result to all_results
                for key, value in task_results.items():
                    if key != "alias":  # Skip the alias field
                        metric_name = key
                        entry = (task_name, metric_name, num_fewshot, version)
                        all_results[entry] = max(all_results.get(entry, value), value)

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading {results_file}: {e}")

    if config:
        assembled_results = {}
        required_tasks = {
            (metric["alias"], metric["metric"])
            for task in config["tasks"]
            for metric in task["report_metric"]
        }
        for task_name, metric in required_tasks:
            key = next(
                (k for k in all_results.keys() if k[0] == task_name and k[1] == metric),
                None,
            )
            if key:
                assembled_results[key] = all_results[key]
            else:
                print(f"Missing {task_name} {metric}")
        return assembled_results

    return all_results
def tabulate_results(results: dict, config: Optional[dict] = None):
    headers = ["Task", "Metric", "Num Fewshot", "Version"]

    # Add model names as additional headers (sorted alphabetically)
    model_names = sorted(list(results.keys()), key=lambda x: x.lower(), reverse=True)
    headers.extend(model_names)

    # Define task order and groups
    if config:
        # Dynamically build task groups from config
        task_groups = {}
        for task in config["tasks"]:
            task_group = task.get("task_group", "Other").title()
            task_name = task["task_name"]

            if task_group not in task_groups:
                task_groups[task_group] = []

            # Add aliases from report_metric to the group
            if "report_metric" in task:
                for metric in task["report_metric"]:
                    alias = metric.get("alias", task_name)
                    if alias not in task_groups[task_group]:
                        task_groups[task_group].append(alias)
            else:
                if task_name not in task_groups[task_group]:
                    task_groups[task_group].append(task_name)
    else:
        # Fallback to hardcoded groups when no config is provided
        task_groups = {
            "Aggregate": ["mmlu_pro_5_shot_base", "mmlu", "GPQA-cot"],
            "Math": ["gsm8k_cot_nemo", "minerva_math_nemo", "mmlu_stem"],
            "Code": ["humaneval_greedy", "humanevalplus_greedy", "mbpp_sanitized_3shot_greedy", "mbppplus_greedy"],
            "Base Reasoning": ["arc_challenge_llama", "hellaswag", "winogrande", "piqa", "commonsense_qa", "openbookqa", "social_iqa", "truthfulqa_mc2"],
            "LongThought Reasoning": ["minerva_math_long", "mmlu_pro_cot_long"]
        }

    # Create a flat order dictionary for sorting
    task_order = {}
    order_idx = 0
    for group, tasks in task_groups.items():
        for task in tasks:
            task_order[task] = order_idx
            order_idx += 1

    # Collect all unique task metadata across all models
    all_task_meta = set()
    for model_name, model_results in results.items():
        all_task_meta.update(model_results.keys())

    # Create a dictionary to store rows by task metadata
    rows_by_meta = {}
    for task_meta in all_task_meta:
        task_name, metric, num_fewshot, version = task_meta
        # Initialize row with task metadata
        row = [task_name, metric, num_fewshot, version]
        # Add scores for each model (or empty string if not available)
        for model_name in model_names:
            if task_meta in results[model_name]:
                row.append(scale_metric_value(metric, results[model_name][task_meta]))
            else:
                row.append("")
        rows_by_meta[task_meta] = row

    # Sort rows by task order
    def get_task_order(task_meta):
        task_name = task_meta[0]
        metric = task_meta[1]
        return task_order.get(task_name, float('inf')), metric

    sorted_task_meta = sorted(all_task_meta, key=get_task_order)
    table = [rows_by_meta[meta] for meta in sorted_task_meta]

    # Create a dictionary to track which tasks we've seen
    seen_tasks = {row[0] for row in table}

    # Add group headers to the table
    final_table = []

    # Process each group in order
    for group, tasks in task_groups.items():
        # Add group header with empty cells for model columns
        group_header = [f"{group}", "", "", ""] + [""] * len(model_names)
        final_table.append(group_header)

        # Process each task in this group
        for task in tasks:
            # Find rows for this task
            task_rows = [row for row in table if row[0] == task]

            if task_rows:
                # Add all rows for this task
                for row in task_rows:
                    final_table.append(row)
            else:
                # Add a placeholder row for missing tasks
                placeholder = [task, "", "", ""] + [""] * len(model_names)
                final_table.append(placeholder)

    # Add any remaining rows that weren't in our predefined groups
    remaining_rows = [row for row in table if row[0] not in task_order]
    if remaining_rows:
        other_header = [f"Other", "", "", ""] + [""] * len(model_names)
        final_table.append(other_header)
        final_table.extend(remaining_rows)

    # Print tabulated results
    print(tabulate(final_table, headers=headers))

    # Create DataFrame for potential CSV export
    df = pd.DataFrame(final_table, columns=headers)
    print("\nCSV Output:")
    print(df.to_csv(index=False))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config YAML file")
    parser.add_argument("-r", "--results-dir", help="Directory containing results files")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = None

    results = {}
    results_files = glob.glob(f"{args.results_dir}/**/results*.json", recursive=True)
    parent_dirs = {os.path.dirname(fname) for fname in results_files}
    for parent_dir in parent_dirs:
        model_name = os.path.basename(parent_dir)
        merged = results.setdefault(model_name, {})
        new_results = load_results(parent_dir, config)
        for key, value in new_results.items():
            merged[key] = max(merged.get(key, value), value)
    tabulate_results(results, config)


if __name__ == "__main__":
    main()
