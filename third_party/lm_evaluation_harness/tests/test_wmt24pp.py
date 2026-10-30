import pytest
from datasets import Dataset

from lm_eval.tasks.wmt24pp import utils


def test_process_docs_filters_canaries_and_bad_sources():
    docs = Dataset.from_list(
        [
            {
                "domain": "canary",
                "is_bad_source": False,
                "source": "canary",
                "target": "canary",
            },
            {
                "domain": "news",
                "is_bad_source": True,
                "source": "bad",
                "target": "bad",
            },
            {
                "domain": "news",
                "is_bad_source": False,
                "source": "Hello",
                "target": "Hallo",
            },
        ]
    )

    filtered = utils.process_docs(docs)

    assert len(filtered) == 1
    assert filtered[0]["source"] == "Hello"


def test_process_results_normalizes_generation():
    metrics = utils.process_results(
        {"source": " Hello ", "target": " Hallo "}, [[" Hallo \n"]]
    )

    assert metrics == {"chrf++": ("Hallo", "Hallo")}


def test_chrfpp_uses_normalized_report_scale():
    assert utils.chrfpp_agg([("hello world", "hello world")]) == pytest.approx(1.0)
