from typing import Any
from lm_eval.tasks.cruxeval import prompts
from lm_eval.metrics.code_eval import check_correctness
from lm_eval.metrics.code_eval import estimate_pass_at_k as direct_estimate_pass_at_k
from lm_eval.metrics.code_eval import evaluate_candidates
import re


def doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc['output'])


def doc_to_text(doc: dict[str, Any], cot: bool = False) -> str:
    if cot:
        return prompts.make_cot_output_prompt((doc["code"], doc["input"]))
    else:
        return prompts.make_direct_output_prompt((doc["code"], doc["input"]))


def sanitize_output(generation: str) -> str:
    answer_match = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', generation, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        match = re.search(r'assert\s+f\s*\([^)]+\)\s*==\s*(.+)', answer_content)
        if match:
            return match.group(1).strip()

    match = re.search(r'assert\s+f\s*\([^)]+\)\s*==\s*(.+)', generation)
    if match:
        return match.group(1).strip()

    match = re.search(r'==\s*(.+)', generation)
    if match:
        output = match.group(1).strip()
        if '#' in output:
            output = output.split('#')[0].strip()
        return output

    return "None"


def process_resp(repeat_idx: int, resp: str, doc: dict) -> dict:
    predicted_output = sanitize_output(resp)
    code = doc["code"]
    input_val = doc["input"]

    try:
        test_program = f"{code}\ntry:\n    actual_result = f({input_val})\n    assert actual_result == {predicted_output}\nexcept:\n    assert False"
    except:
        test_program = "assert False"

    result = check_correctness(test_program, timeout=3, task_id=doc.get('task_id', repeat_idx), completion_id=repeat_idx)
    return result


def build_predictions(generations: list[list[str]], docs: list[dict[str, Any]]) -> list[list[dict]]:
    return evaluate_candidates(generations, docs, process_resp)
