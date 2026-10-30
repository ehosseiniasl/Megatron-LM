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
