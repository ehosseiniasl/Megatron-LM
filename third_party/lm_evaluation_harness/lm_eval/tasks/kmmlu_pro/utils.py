from functools import partial


choices = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
]


# Domains follow KMMLU-Pro grouping (see https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro)
_LAW = frozenset({"법무사", "변호사", "공인노무사", "변리사"})
_TAX_AND_ACCOUNTING = frozenset({"공인회계사", "세무사", "관세사"})
_VALUE_ESTIMATION = frozenset({"손해사정사", "감정평가사"})
_MEDICINE = frozenset({"한의사", "치과의사", "약사", "한약사", "의사"})


def _filter_by_licenses(dataset, licenses: frozenset):
    return dataset.filter(lambda x: x["license_name"] in licenses)


process_law = partial(_filter_by_licenses, licenses=_LAW)
process_tax_and_accounting = partial(_filter_by_licenses, licenses=_TAX_AND_ACCOUNTING)
process_value_estimation = partial(_filter_by_licenses, licenses=_VALUE_ESTIMATION)
process_medicine = partial(_filter_by_licenses, licenses=_MEDICINE)


def format_cot_example(example, including_answer=False):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question.strip() + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt.strip())
    if including_answer:
        prompt += f"Answer: The answer is ({doc_to_target(example)})."
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


doc_to_text = partial(format_cot_example, including_answer=False)


def doc_to_target(doc: dict) -> str:
    sol = doc.get("solution", doc.get("answer"))
    if sol is None:
        raise KeyError("document must contain 'solution' or 'answer' (1-based index)")
    if isinstance(sol, str) and sol.strip().upper() in choices:
        return sol.strip().upper()
    idx = int(sol) - 1
    return choices[idx]
