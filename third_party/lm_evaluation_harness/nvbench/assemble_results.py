import argparse
import glob
import json
from pathlib import Path

import yaml
from tabulate import tabulate


def load_results(
    results_dir: str, config: dict | None = None, filter_to_config: bool = True
):

    configured_limits = (
        {task["task_name"]: task["limit"] for task in config.get("tasks", []) if "limit" in task}
        if config
        else {}
    )

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

            result_limit = data["config"]["limit"]
            if result_limit is not None:
                matching_limit = any(
                    task_name in data["results"] and task_limit == result_limit
                    for task_name, task_limit in configured_limits.items()
                )
                if not matching_limit:
                    print(f"Skipping {results_file} - limit is not configured")
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

        except (OSError, json.JSONDecodeError) as e:
            print(f"Error loading {results_file}: {e}")

    if config and filter_to_config:
        assembled_results = {}
        required_tasks = {
            (metric["alias"], metric["metric"])
            for task in config["tasks"]
            for metric in task.get("report_metric", [{"alias": task["task_name"], "metric": "??"}])
        }
        for task_name, metric in required_tasks:
            key = next(
                (k for k in all_results if k[0] == task_name and k[1] == metric),
                None,
            )
            if key:
                assembled_results[key] = all_results[key]
            else:
                print(f"Missing {task_name} {metric}")
        return assembled_results

    return all_results


def tabulate_results(results: dict):
    headers = ["Task", "Metric", "Num Fewshot", "Version", "Value"]
    table = sorted([[*k, v] for k, v in results.items()])
    print(tabulate(table, headers=headers))


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

    results = load_results(args.results_dir, config)
    tabulate_results(results)


if __name__ == "__main__":
    main()
