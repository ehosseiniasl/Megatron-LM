"""Utilities for the WMT24++ English-to-language translation tasks."""

from __future__ import annotations

import sacrebleu

from lm_eval.api.metrics import _sacreformat


def process_docs(dataset):
    """Drop canaries and low-quality sources as recommended by the dataset."""
    return dataset.filter(
        lambda doc: doc.get("domain") != "canary"
        and not doc.get("is_bad_source", False)
    )


def process_results(doc, results):
    prediction = results[0] if isinstance(results, (list, tuple)) else results
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    return {
        "chrf++": (doc["target"].strip(), str(prediction).strip()),
    }


def chrfpp_agg(items) -> float:
    """Return chrF++ on a 0-1 scale, matching nvbench report metrics."""
    refs, preds = zip(*items)
    refs, preds = _sacreformat(list(refs), list(preds))
    return sacrebleu.corpus_chrf(preds, refs, word_order=2).score / 100.0
