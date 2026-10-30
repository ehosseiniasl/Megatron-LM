import logging
from typing import Dict

from math_verify import parse, verify

from lm_eval.tasks.minerva_math_nemo.math_grader import extract_answer, math_equal


def doc_to_text(doc: dict) -> str:
    return "Problem:\n" + doc["problem"] + "\n\nSolution:"


def doc_to_target(doc: dict) -> str:
    if "few_shot" in doc:
        return doc["solution"]
    return doc["answer"]


def safe_parse(s: str) -> list[str]:
    parsed = None
    try:
        parsed = parse(s)
    except Exception as e:
        logging.warning(f"math_verify parse error: {e}")
    return parsed or ["[invalid]"]


def process_results(doc: dict, results: list[list[str]]) -> Dict[str, int]:
    pred_answer = results
    while isinstance(pred_answer, list):
        pred_answer = pred_answer[0]

    gt_answer = doc["answer"]

    mv_gold, mv_answer, mv_is_correct = ["??"], ["??"], False
    mg_gold, mg_answer, mg_is_correct = "??", "??", False

    assert isinstance(gt_answer, str)
    assert isinstance(pred_answer, str)

    try:
        mv_gold = safe_parse(gt_answer)
        mv_answer = safe_parse(pred_answer)
        mv_is_correct = verify(mv_gold, mv_answer)
    except Exception as e:
        logging.warning(f"Error verifying math_verify: {e}")

    try:
        mg_gold = gt_answer
        mg_answer = extract_answer(pred_answer)
        if mg_answer is None:
            mg_answer = pred_answer.strip()
        mg_is_correct = math_equal(mg_answer, mg_gold)
    except Exception as e:
        logging.warning(f"Error grading math_grader: {e}")

    is_correct = mv_is_correct or mg_is_correct

    return {
        "parse_meta": {
            "math_verify": {
                "gold": mv_gold[-1],
                "parsed": mv_answer[-1] if mv_answer else "[invalid]",
                "exact_match": mv_is_correct,
            },
            "math_grader": {
                "gold": mg_gold,
                "parsed": mg_answer,
                "exact_match": mg_is_correct,
            },
        },
        "exact_match": int(is_correct),
    }
