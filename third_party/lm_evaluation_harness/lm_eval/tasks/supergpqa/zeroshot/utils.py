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
        # Get the options list and preprocess each option
        choices = [preprocess(option) for option in doc["options"]]

        # Get the correct answer before shuffling
        correct_answer_text = preprocess(doc["answer"])

        # Shuffle the choices
        random.shuffle(choices)

        # Find the index of the correct answer after shuffling
        correct_answer_index = choices.index(correct_answer_text)

        out_doc = {
            "Question": doc["question"],
            "choices": choices,
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)


def doc_to_choice(doc):
    # Return list of choice labels based on number of choices
    return [f"{chr(65 + i)})" for i in range(len(doc["choices"]))]
