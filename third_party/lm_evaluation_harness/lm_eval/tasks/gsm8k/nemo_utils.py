from math_verify import parse, verify


def safe_parse(s: str) -> list[str]:
    try:
        return parse(s)
    except Exception as e:
        print(e, s)
        return ["[invalid]"]


def doc_to_target(doc: dict) -> str:
    if "answer" in doc:
        return doc["answer"].split("####")[-1].strip()
    else:
        assert "target" in doc
        return doc["target"]


def process_results(doc: dict, results: list[list[str]]) -> dict[str, int]:
    pred_answer = results[0]
    while isinstance(pred_answer, list):
        pred_answer = pred_answer[0]
    assert isinstance(pred_answer, str)

    gt_answer = safe_parse(doc_to_target(doc))
    pred_answer = safe_parse(pred_answer)
    exact_match = verify(gt_answer, pred_answer)
    parsed_pred_answer_str = pred_answer[-1] if pred_answer else "[invalid]"

    return {
        "exact_match": int(exact_match),
        "parse_meta": {
            "math_verify": {
                "gold": gt_answer[-1],
                "parsed": parsed_pred_answer_str,
                "exact_match": int(exact_match),
            }
        },
    }
