import hashlib
import logging
from pathlib import Path
from typing import Dict, List

from math_verify import parse, verify
from lm_eval.tasks.minerva_math_nemo.math_grader import extract_answer, math_equal

MATH_500_MD5s = None


# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:"


def doc_to_target(doc: dict) -> str:
    if "few_shot" in doc:
        return doc["solution"]
    if "answer" in doc:
        return doc["answer"]
    return parse(doc["solution"])[1] or extract_answer(doc["solution"])


def md5sum(s):
    return hashlib.md5(s.encode()).hexdigest()


def m500_hashes():
    global MATH_500_MD5s
    if MATH_500_MD5s is None:
        m500_md5_path = Path(__file__).parent / "math_500_md5.csv"
        with open(m500_md5_path, "r") as f:
            MATH_500_MD5s = set(line.strip() for line in f)
    return MATH_500_MD5s


def safe_parse(s: str) -> list[str]:
    parsed = None
    try:
        parsed = parse(s)
    except Exception as e:
        print(e, s)
    return parsed or ["[invalid]"]


def exact_match(doc: dict, results: list[list[str]]) -> Dict[str, int]:
    pred_answer = results
    while isinstance(pred_answer, list):
        pred_answer = pred_answer[0]

    gt_solution = doc["solution"]

    mv_gold, mv_answer, mv_is_correct = "??", "??", False
    mg_gold, mg_answer, mg_is_correct = ("??", ), ("??", ), False

    assert isinstance(gt_solution, str)
    assert isinstance(pred_answer, str)

    try:
        mv_gold = safe_parse(gt_solution)
        mv_answer = safe_parse(pred_answer)
        mv_is_correct = verify(mv_gold, mv_answer)
    except Exception as e:
        logging.warning(f"Error verifying math_verify: {e}")

    try:
        mg_gold = extract_answer(gt_solution)
        mg_answer = extract_answer(pred_answer)
        mg_is_correct = math_equal(mg_gold, mg_answer)
    except Exception as e:
        logging.warning(f"Error grading math_grader: {e}")

    exact_match = mv_is_correct or mg_is_correct

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
        "exact_match": int(exact_match),
    }


def process_results(doc: dict, results: list[list[str]]) -> Dict[str, int]:
    results = exact_match(doc, results)
    is_exact_match = results["exact_match"]

    math500_exact_match, level_5_exact_match = -1, -1
    question_hash = md5sum(doc["problem"])

    is_math500 = question_hash in m500_hashes()
    if is_math500:
        math500_exact_match = is_exact_match

    is_level_5 = "5" in str(doc["level"])
    if is_level_5:
        level_5_exact_match = is_exact_match

    results["parse_meta"]["is_level_5"] = is_level_5
    results["parse_meta"]["is_math500"] = is_math500
    results["level_5_exact_match"] = level_5_exact_match
    results["math500_exact_match"] = math500_exact_match
    return results
