import argparse
import os

import datasets
import yaml

from lm_eval.evaluator_utils import get_task_list
from lm_eval.tasks import TaskManager, get_task_dict


datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True


def load_nvbench_tasks() -> set[str]:
    task_names = set()
    config_files = [
        fname
        for fname in os.listdir("nvbench")
        if fname.endswith(".yaml") or fname.endswith(".yml")
    ]
    for fname in config_files:
        with open(f"nvbench/{fname}", "r") as f:
            config = yaml.safe_load(f)
            for task in config["tasks"]:
                task_names.add(task["task_name"])
    return task_names


def fetch_task_datasets(task_names: set[str]):
    # Build tasks and get dataset info
    print(task_names)
    task_manager = TaskManager()
    tasks = list(task_names)
    task_dict = get_task_dict(tasks, task_manager)
    loaded_tasks = {t if isinstance(t, str) else t.group for t in task_dict.keys()}
    eval_tasks = get_task_list(task_dict)
    for task in task_names:
        if task not in loaded_tasks:
            print(f"Task {task} not found in task_dict")
    return eval_tasks


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fetch datasets for evaluation tasks")
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="List of task names to fetch. If not provided, loads from nvbench YAML configs.",
    )

    args = parser.parse_args()

    if args.tasks:
        task_names = set(args.tasks)
    else:
        task_names = load_nvbench_tasks()

    fetch_task_datasets(task_names)
