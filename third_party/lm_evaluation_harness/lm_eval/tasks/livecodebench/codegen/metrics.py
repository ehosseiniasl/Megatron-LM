# borrowed and extended from
# https://github.com/Naman-ntc/codescratch/blob/main/evaluation/bigcode-evaluation-harness/lm_eval/tasks/custom_metrics/apps_custom_metrics/utils.py
# https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/compute_code_generation_metrics.py

import os
import sys

try:
    sys.set_int_max_str_digits(50000)
except AttributeError:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any
from dataclasses import dataclass


import numpy as np
from tqdm import tqdm

from lm_eval.tasks.livecodebench.codegen.exectute import run_test, TestResult


@dataclass
class GenerationResult:
    """Result for a single code generation attempt."""

    test_results: list[TestResult]  # List of TestResult enum values per test case
    metadata: dict  # Raw metadata, not JSON stringified


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def _check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    )
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[TestResult.TIMEOUT for i in range(len(in_outs["inputs"]))]]
        if debug:
            print("global timeout")
    return result[0], metadata_list[0]


def evaluate_generations_by_problem(args):
    problem_generations: list[str] = args[0]
    sample = args[1]
    debug: bool = args[2]
    timeout: int = args[3]

    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [TestResult.WRONG_ANSWER]
        try:
            curr_res, curr_metadata = _check_correctness(
                sample, o, timeout=timeout, debug=debug
            )
            if debug:
                print(f"\nSuccessful compilation of task {o_idx}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, (np.bool_, bool)):
                    e = TestResult.PASS if e else TestResult.WRONG_ANSWER
                elif isinstance(e, int):
                    # Convert raw integer codes to TestResult enum
                    try:
                        e = TestResult(e)
                    except ValueError:
                        e = TestResult.TEST_RUNNER_ERROR
                fixed.append(e)
            curr_res = fixed
            if not np.all([r == TestResult.PASS for r in curr_res]):
                if debug:
                    print(f"Results were not PASS for all test cases {curr_res=}\n")
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            # break
            curr_metadata = {
                "error": repr(e),
                "error_code": TestResult.TEST_RUNNER_ERROR,
                "error_message": "TestRunnerError",
            }
        finally:
            assert isinstance(curr_res, list), curr_res
            assert isinstance(curr_metadata, dict), curr_metadata
            res.append(curr_res)
            metadata.append(curr_metadata)
    if debug:
        for i, r in enumerate(problem_generations):
            print("Sample\n")
            print(r)
            print("\n")
            print("Result\n")
            print(res[i])
            print("*" * 30 + "\n\n")
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    """We take the list of code generations and try to compile them
     and the run their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS dataset)
        level: difficulty level used in the generation, can be "all", "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
    """

    # generations are code generations in the same order of the dataset

    inputs = [
        [(generations_list[index], samples_list[index], debug, timeout), index]
        for index in range(len(generations_list))
    ]

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(
            max_workers=1 if debug else num_process_evaluate
        ) as executor:
            futures = {
                executor.submit(evaluate_generations_by_problem, arg): index
                for arg, index in inputs
            }

            results = {}
            metadata = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index], metadata[index] = future.result()
                pbar.update(1)

    assert len(results) == len(
        inputs
    ), f"results = {len(results)} inputs = {len(inputs)} {results=}"
    # results = {i: r for r, (_, i) in zip(results, inputs)}

    return results, metadata


def codegen_metrics(
    samples_list: list[dict[str, Any]],
    generations_list: list[list[str]],
    k_list: list[int] = [1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
    num_process_evaluate: int = 16,
    timeout: int = 6,
    debug: bool = False,
) -> list[list[GenerationResult]]:

    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample, generation_list) in enumerate(
        zip(samples_list, generations_list)
    ):
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    print(f"Evaluating {len(samples_linear)}...")

    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])

    for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])

    # Convert to simplified GenerationResult format
    generation_results = []
    for problem_idx in sorted(results.keys()):
        problem_results = []
        problem_test_results = results[problem_idx]
        problem_metadatas = metadatas.get(problem_idx, [])

        for gen_idx, test_results in enumerate(problem_test_results):
            metadata = (
                problem_metadatas[gen_idx] if gen_idx < len(problem_metadatas) else {}
            )
            # Keep metadata as dict, not JSON string
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {"raw": metadata}

            gen_result = GenerationResult(test_results=test_results, metadata=metadata)
            problem_results.append(gen_result)

        generation_results.append(problem_results)

    return generation_results
