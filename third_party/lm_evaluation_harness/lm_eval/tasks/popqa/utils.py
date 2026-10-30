import json

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        return {
            "question": doc["question"],
            "answers": json.loads(doc["possible_answers"]),
        }

    return dataset.map(_process_doc)


def doc_to_target(doc):
    return doc["answers"]
