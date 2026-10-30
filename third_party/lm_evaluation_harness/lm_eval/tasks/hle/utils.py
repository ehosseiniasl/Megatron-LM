import string

import datasets


def _normalize(text):
    """Lowercase and strip punctuation."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def exact_match_passk(doc, results):
    from lm_eval.metrics.code_eval import estimate_pass_at_k_dynamic

    candidates = results[0]
    gold = _normalize(doc["answer"])
    doc_results = [
        {"passed": _normalize(pred) == gold, "gold": doc["answer"], "answer": pred}
        for pred in candidates
    ]
    return estimate_pass_at_k_dynamic(doc, [doc_results])


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # Remove Image feature columns to avoid Pillow dependency
    cols_to_remove = [
        c for c in ["image_preview", "rationale_image"]
        if c in dataset.column_names
    ]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    # Keep only text-only questions (image is empty string or None)
    return dataset.filter(lambda doc: not doc["image"])
