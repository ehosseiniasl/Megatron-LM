import math
import os
from typing import Any

from lm_eval.metrics.code_eval import estimate_pass_at_k
from lm_eval.tasks.livecodebench.codegen.types import CodeGenerationProblem
from lm_eval.tasks.livecodebench.codegen.prompts import base_model_prompt
from lm_eval.tasks.livecodebench.codegen.metrics import codegen_metrics
from lm_eval.tasks.livecodebench.codegen.exectute import TestResult

_DIFFICULTIES = ("easy", "medium", "hard")


def nanmean(values):
    valid = [v for v in values if v is not None and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


def build_text(doc):
    return base_model_prompt(CodeGenerationProblem(**doc))


def score_batch(resps: list[list[str]], docs: list[dict]):
    samples = [CodeGenerationProblem(**doc).get_evaluation_sample() for doc in docs]
    n_workers = os.cpu_count() or 16
    results = codegen_metrics(
        samples, resps, k_list=[1, len(resps[0])], num_process_evaluate=n_workers
    )
    return [[r.test_results for r in problem_results] for problem_results in results]


def passthrough_score(doc: dict, preds: list[list[TestResult]]):
    assert len(preds) == 1, "passthrough_score only supports single problem"
    preds = preds[0]

    total = len(preds)
    correct = sum(
        all(pred == TestResult.PASS for pred in gen_result) for gen_result in preds
    )

    p1 = float(estimate_pass_at_k([total], [correct], 1)[0])
    result = {f"pass@1[{total}]": p1}
    if total > 1:
        pk = float(estimate_pass_at_k([total], [correct], total)[0])
        result[f"pass@{total}[{total}]"] = pk
    else:
        pk = p1

    difficulty = doc.get("difficulty", "")
    for diff in _DIFFICULTIES:
        result[f"pass@1_{diff}"] = p1 if difficulty == diff else float("nan")
        if total > 1:
            result[f"pass@k_{diff}"] = pk if difficulty == diff else float("nan")

    return result
