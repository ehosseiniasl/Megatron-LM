import random
import re

import datasets


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        # random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "explanation": doc["Explanation"],
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)


def doc_to_text(doc):
    return (
        f"Question: {doc['Question']}\n"
        "Choices:\n"
        f"(A) {doc['choice1']}\n"
        f"(B) {doc['choice2']}\n"
        f"(C) {doc['choice3']}\n"
        f"(D) {doc['choice4']}\n"
        f"Let's think step by step: "
    )


def doc_to_target(doc):
    return f"{doc['explanation']}. The answer is {doc['answer']}"


def process_results(doc, result):
    return {"exact_match": doc["answer"] == result[0]}


def exact_match_passk(doc, results):
    from lm_eval.metrics.code_eval import estimate_pass_at_k_dynamic

    candidates = results[0]
    doc_results = [
        {"passed": doc["answer"] == pred, "gold": doc["answer"], "answer": pred}
        for pred in candidates
    ]
    return estimate_pass_at_k_dynamic(doc, [doc_results])
