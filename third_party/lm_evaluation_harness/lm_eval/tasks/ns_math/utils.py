from math_verify import parse

from lm_eval.tasks.ns_math.few_shot_examples import examples_map


def doc_to_text(doc: dict) -> str:
    return "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:"


def standard_math_fewshot_samples() -> list[dict]:
    return examples_map["math_standard_few_shot"]


def detailed_math_fewshot_samples() -> list[dict]:
    return examples_map["math_text_detailed"]


def standard_gsm8k_fewshot_samples() -> list[dict]:
    return examples_map["gsm8k_standard_few_shot"]


def detailed_gsm8k_fewshot_samples() -> list[dict]:
    return examples_map["gsm8k_detailed_few_shot"]


def doc_to_target(doc: dict) -> str:
    for k in ["solution", "answer", "reference_solution", "expected_answer"]:
        if k in doc:
            return doc[k]
    return parse(doc["solution"])[1]
