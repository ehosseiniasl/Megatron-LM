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
        # Get the options list and preprocess each option
        choices = [preprocess(option) for option in doc["options"]]
        out_doc = {
            "Question": doc["question"],
            "choices": choices,
            "answer": f"({doc['answer_letter']})",
        }
        return out_doc

    return dataset.map(_process_doc)


def doc_to_text(doc):
    choices_text = "\n".join(
        [f"{chr(65+i)}) {option}" for i, option in enumerate(doc["choices"])]
    )
    num_choices = len(doc["choices"])
    # letters = ", ".join([chr(65 + i) for i in range(num_choices)])
    return (
        f"\nQuestion: {doc['Question']}\n"
        f"Choices:\n"
        f"{choices_text}\n"
        f"Answer: "
    )


def doc_to_target(doc):
    # For fewshot examples with explanation, include it (CoT reasoning)
    # For test docs without explanation, just return the answer
    if "explanation" in doc and doc["explanation"]:
        return f"{doc['explanation']} The answer is {doc['answer']}"
    else:
        return f"The answer is {doc['answer']}"


def process_results(doc, result):
    return {"exact_match": doc["answer"] == result[0]}
