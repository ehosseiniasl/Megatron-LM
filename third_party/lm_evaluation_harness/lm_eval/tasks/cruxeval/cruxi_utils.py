from typing import Any
from lm_eval.tasks.cruxeval import prompts
from lm_eval.metrics.code_eval import check_correctness
from lm_eval.metrics.code_eval import estimate_pass_at_k as direct_estimate_pass_at_k
from lm_eval.metrics.code_eval import evaluate_candidates
import re


def doc_to_target(doc: dict[str, Any]) -> str:
    return f"f({doc['input']})"


def doc_to_text(doc: dict[str, Any], cot: bool = False) -> str:
    if cot:
        return prompts.make_cot_input_prompt((doc["code"], doc["output"]))
    else:
        return prompts.make_direct_input_prompt((doc["code"], doc["output"]))


def sanitize_input(generation: str) -> str:
    answer_match = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', generation, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        match = re.search(r'assert\s+f\s*\(\s*([^)]+)\s*\)\s*==', answer_content)
        if match:
            return match.group(1).strip()

    match = re.search(r'assert\s+f\s*\(\s*([^)]+)\s*\)\s*==', generation)
    if match:
        return match.group(1).strip()

    match = re.search(r'f\s*\(\s*([^)]+)\s*\)', generation)
    if match:
        return match.group(1).strip()

    return "None"


def process_resp(repeat_idx: int, resp: str, doc: dict) -> dict:
    predicted_input = sanitize_input(resp)
    code = doc["code"]
    expected_output = doc["output"]

    try:
        test_program = f"{code}\ntry:\n    result = f({predicted_input})\n    assert result == {expected_output}\nexcept:\n    assert False"
    except:
        test_program = "assert False"

    result = check_correctness(test_program, timeout=3, task_id=doc.get('task_id', repeat_idx), completion_id=repeat_idx)
    return result


def build_predictions(generations: list[list[str]], docs: list[dict[str, Any]]) -> list[list[dict]]:
    return evaluate_candidates(generations, docs, process_resp)
