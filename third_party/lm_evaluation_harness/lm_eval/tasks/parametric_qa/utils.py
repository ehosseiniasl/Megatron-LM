import asyncio
import os
import re
import string
import unicodedata

from lm_eval.api.judge import (
    DEFAULT_JUDGE_CONCURRENCY,
    get_judge_provider,
)
from lm_eval.tasks.parametric_qa.grading import (
    DEFAULT_JUDGE_MODEL,
    FACTS_PARAMETRIC,
    judge_samples,
)


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalize_answer(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = text.lower().replace("&", " and ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = _ARTICLES_RE.sub(" ", text)
    punctuation = string.punctuation.replace(".", "").replace("%", "")
    text = text.translate(str.maketrans({char: " " for char in punctuation}))
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    return " ".join(text.split())


def process_simpleqa_verified_docs(dataset):
    return dataset.map(
        lambda doc: {
            **doc,
            "question": doc["problem"],
            "_parametric_qa_task": "simpleqa_verified",
        }
    )


def process_facts_parametric_docs(dataset):
    return dataset.map(
        lambda doc: {
            **doc,
            "question": doc["query"],
            "_parametric_qa_task": "facts_parametric",
        }
    )


def process_results(doc, results):
    prediction = str(results[0]).strip()
    return {
        "judge_accuracy": {
            "task": doc["_parametric_qa_task"],
            "question": doc["question"],
            "gold_answer": doc["answer"],
            "prediction": prediction,
        },
        "reference_match": int(
            bool(prediction)
            and normalize_answer(prediction) == normalize_answer(doc["answer"])
        ),
        "response_nonempty": int(bool(prediction)),
    }


def aggregate_judge_accuracy(items):
    if not items:
        return 0.0
    task = items[0]["task"]
    if any(item["task"] != task for item in items):
        raise ValueError("Cannot aggregate mixed parametric QA tasks")

    concurrency = int(
        os.environ.get(
            "PARAMETRIC_QA_JUDGE_CONCURRENCY",
            str(DEFAULT_JUDGE_CONCURRENCY),
        )
    )
    model = os.environ.get("PARAMETRIC_QA_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    provider_name = os.environ.get("PARAMETRIC_QA_JUDGE_PROVIDER", "openrouter")
    provider = get_judge_provider(provider_name)
    base_url = os.environ.get(
        "PARAMETRIC_QA_JUDGE_BASE_URL",
        provider.base_url,
    )
    api_key_env = (
        os.environ.get(
            "PARAMETRIC_QA_JUDGE_API_KEY_ENV",
        )
        or provider.api_key_envs
    )
    replicates = 3 if task == FACTS_PARAMETRIC else 1
    print(
        f"Grading {len(items)} {task} responses with {model} via {provider_name}: "
        f"concurrency={concurrency}, replicates={replicates}"
    )
    grades = asyncio.run(
        judge_samples(
            task,
            items,
            model=model,
            concurrency=concurrency,
            base_url=base_url,
            api_key_env=api_key_env,
            supports_response_format=provider.supports_response_format,
            default_headers=provider.default_headers,
            replicates=replicates,
        )
    )
    flat_grades = [grade for item_grades in grades for grade in item_grades]
    return sum(grade == "CORRECT" for grade in flat_grades) / len(flat_grades)
