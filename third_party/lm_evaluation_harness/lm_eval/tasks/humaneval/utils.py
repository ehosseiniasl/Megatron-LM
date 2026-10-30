from lm_eval.filters.sanitize import sanitize
from lm_eval.metrics.code_eval import check_correctness
from lm_eval.metrics.code_eval import evaluate_candidates


def build_references(doc: dict):
    return doc["test"] + "\n" + f"check({doc['entry_point']})"


def build_text(doc: dict):
    return doc["prompt"]


def process_resp(repeat_idx: int, resp: str, doc: dict):
    candidate_code = sanitize(doc["prompt"] + resp, entrypoint=doc["entry_point"])
    test_program = candidate_code + "\n" + build_references(doc)
    result = check_correctness(
        test_program, timeout=3, task_id=doc["task_id"], completion_id=repeat_idx
    )
    return result


def non_sanitized_predictions(resps, docs):
    preds = []
    for resp, doc in zip(resps, docs):
        pred = [doc["prompt"] + r for r in resp]
        preds.append(pred)
    return preds


def score_predictions(candidates: list[list[str]], docs: list[dict]):
    return evaluate_candidates(candidates, docs, process_resp)
