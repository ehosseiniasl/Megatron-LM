import json
import random
import re
import string
from collections import Counter, defaultdict
from typing import Any

import datasets
import numpy as np

from lm_eval.api.samplers import ContextSampler
from lm_eval.api.task import ConfigurableTask


_GSM8K_ANSWER_RE = re.compile(
    r"Final [aA]nswer:?(?s:.)*?(\-?[\d,]+(?:\.\d*)?)\D*?\Z"
)
_SUPER_GPQA_ANSWER_RE = re.compile(r"[aA]nswer is[^A-Za-z]{1,5}([A-Za-z])")
_SUPER_GPQA_INSTRUCTION = (
    'Think step by step and then finish your answer with "The answer is (X)" '
    "where X is the correct letter choice.\n"
)


class MistralMutualInfoTask(ConfigurableTask):
    """Use Thunderdome's non-empty context for mutual-information baselines."""

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config=None):
        if config is not None:
            config = dict(config)
            config.pop("class", None)
            self._mutual_info_context = config.pop("mutual_info_context")
        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            config=config,
        )

    def construct_requests(self, doc, ctx, **kwargs):
        requests = super().construct_requests(doc, ctx, **kwargs)
        choices = self.doc_to_choice(doc)
        if not isinstance(requests, list) or len(requests) != 2 * len(choices):
            raise ValueError("Expected conditional and baseline requests per choice")
        for request, choice in zip(requests[len(choices) :], choices, strict=True):
            request.arguments = (
                self._mutual_info_context,
                f"{self.config.target_delimiter}{choice}",
            )
        return requests


class ThunderdomeSampler(ContextSampler):
    """Match Thunderdome's deterministic pretraining-eval few-shot sampler."""

    def __init__(self, docs, task, fewshot_indices=None, rnd=None) -> None:
        super().__init__(docs, task, fewshot_indices=fewshot_indices, rnd=rnd)
        self._thunderdome_rnd = random.Random(0)
        self._thunderdome_contexts = None
        eval_docs = getattr(task, "eval_docs", None)
        self._first_eval_doc = eval_docs[0] if eval_docs is not None and len(eval_docs) else None
        self._started = False

    def get_context(self, doc: dict[str, Any], num_fewshot: int) -> str:
        """Render the next persistently seeded set of few-shot examples."""
        if self._started and self._first_eval_doc is not None and doc == self._first_eval_doc:
            self._thunderdome_rnd.seed(0)
        self._started = True
        if self._thunderdome_contexts is None:
            contexts = []
            for candidate in self.docs:
                content = self.doc_to_text(candidate)
                target = self.doc_to_target(candidate)
                if self.config.doc_to_choice is not None and not isinstance(content, str):
                    content = self.doc_to_choice(candidate)[content]
                if isinstance(target, list):
                    rendered_target = target[0]
                elif self.config.doc_to_choice is not None and not isinstance(target, str):
                    rendered_target = self.doc_to_choice(candidate)[target]
                else:
                    rendered_target = target
                contexts.append(
                    f"{content}{self.target_delimiter}{rendered_target}"
                    f"{self.fewshot_delimiter}"
                )
            self._thunderdome_contexts = contexts

        contexts = self._thunderdome_contexts
        decontamination_key = self.doc_to_text(doc)
        eligible = [
            index
            for index, context in enumerate(contexts)
            if decontamination_key not in context
        ]
        if len(eligible) < num_fewshot:
            raise ValueError(
                f"Need {num_fewshot} eligible few-shot examples, got {len(eligible)}"
            )

        selected = set()
        while len(selected) < num_fewshot:
            selected.add(self._thunderdome_rnd.choice(eligible))
        return "".join(contexts[index] for index in selected)


class SameSubjectFirstNSampler(ContextSampler):
    """Select canonical few-shot examples belonging to the current subject."""

    def get_context(self, doc: dict[str, Any], num_fewshot: int) -> str:
        """Render the first examples from the evaluated document's subject."""
        selected = [
            candidate
            for candidate in self.docs
            if candidate["subject"] == doc["subject"]
            and self.doc_to_text(doc) not in self.doc_to_text(candidate)
        ][:num_fewshot]
        if len(selected) < num_fewshot:
            raise ValueError(
                f"Need {num_fewshot} examples for {doc['subject']}, got {len(selected)}"
            )

        contexts = []
        for candidate in selected:
            target = self.doc_to_target(candidate)
            if self.config.doc_to_choice is not None and not isinstance(target, str):
                target = self.doc_to_choice(candidate)[target]
            contexts.append(
                f"{self.doc_to_text(candidate)}{self.target_delimiter}{target}"
                f"{self.fewshot_delimiter}"
            )
        return "".join(contexts)


def _hellaswag_preprocess(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _last_argmax(values: list[float]) -> int:
    """Return the last index attaining the maximum, as Thunderdome does."""
    return max(range(len(values)), key=lambda index: (values[index], index))


def byte_normalized_argmax(loglikelihoods: list[float], choices: list[str]) -> int:
    """Select by UTF-8-byte-normalized score with Thunderdome tie-breaking."""
    normalized = [
        loglikelihood / len(choice.encode("utf-8")) if choice else -np.inf
        for loglikelihood, choice in zip(loglikelihoods, choices, strict=True)
    ]
    return _last_argmax(normalized)


def process_hellaswag(dataset):
    def process_doc(doc):
        ctx_b = doc["ctx_b"].strip()
        if ctx_b:
            ctx_b = ctx_b[0].upper() + ctx_b[1:]
        context = f"{doc['ctx_a'].strip()} {ctx_b}"
        raw_choices = [_hellaswag_preprocess(choice) for choice in doc["endings"]]
        return {
            "query": _hellaswag_preprocess(f"{doc['activity_label']}: {context}"),
            # Thunderdome strips each choice when constructing the continuation,
            # but its byte-normalized metric uses the pre-strip dataset value.
            "choices": [choice.strip() for choice in raw_choices],
            "raw_choices": raw_choices,
            "gold": int(doc["label"]),
        }

    return dataset.map(process_doc)


def process_hellaswag_results(doc, results):
    """Match Thunderdome's stripped prompt and raw-byte normalization."""
    loglikelihoods = [result[0] for result in results]
    prediction = byte_normalized_argmax(loglikelihoods, doc["raw_choices"])
    return {"acc": float(prediction == int(doc["gold"]))}


def process_commonsense_qa(dataset):
    def process_doc(doc):
        question = doc["question"].strip()
        concept = doc["question_concept"].strip()
        return {
            "query": (
                f"Answer the following question about {concept}.\n"
                f"Question: {question}\nAnswer:"
            ),
            "choices": [choice.strip() for choice in doc["choices"]["text"]],
            "gold": "ABCDE".index(doc["answerKey"]),
        }

    return dataset.map(process_doc)


def process_openbookqa(dataset):
    def process_doc(doc):
        labels = [label.strip() for label in doc["choices"]["label"]]
        return {
            "query": doc["question_stem"].strip(),
            "choices": [choice.strip() for choice in doc["choices"]["text"]],
            "gold": labels.index(doc["answerKey"].strip()),
        }

    return dataset.map(process_doc)


def process_piqa_results(doc, results):
    """Match Thunderdome: score stripped choices, normalize by raw bytes."""
    loglikelihoods = [result[0] for result in results]
    raw_choices = [doc["sol1"], doc["sol2"]]
    prediction = byte_normalized_argmax(loglikelihoods, raw_choices)
    return {"acc": float(prediction == int(doc["label"]))}


def process_siqa_results(doc, results):
    """Score Social IQA using Thunderdome's UTF-8 byte normalization."""
    choices = [doc[key].strip() for key in ("answerA", "answerB", "answerC")]
    prediction = byte_normalized_argmax([result[0] for result in results], choices)
    return {"acc": float(prediction == int(doc["label"]) - 1)}


def process_mistral_mc_results(doc, results):
    """Score a processed MC row with Thunderdome's last-index tie break."""
    prediction = _last_argmax([result[0] for result in results])
    gold = doc["gold"]
    correct = prediction in gold if isinstance(gold, list) else prediction == int(gold)
    return {"acc": float(correct)}


def process_mutual_info_results(doc, results):
    """Score conditional-minus-baseline requests in their two request blocks."""
    num_choices = len(doc["choices"])
    conditional = [result[0] for result in results[:num_choices]]
    baseline = [result[0] for result in results[num_choices:]]
    if len(baseline) != num_choices:
        raise ValueError(
            f"Expected {2 * num_choices} mutual-information results, got {len(results)}"
        )
    prediction = _last_argmax(
        [conditional[index] - baseline[index] for index in range(num_choices)]
    )
    return {"acc_mutual_info": float(prediction == int(doc["gold"]))}


def process_new_mmlu(dataset):
    def process_doc(doc):
        question = doc["question"].strip()
        question = question[0].upper() + question[1:]
        choices = [
            f"{letter}. {choice.strip()}\n\n"
            for letter, choice in zip("ABCD", doc["choices"], strict=True)
        ]
        display_choices = [choice.rstrip() for choice in choices]
        return {
            "query": question + "\n" + "\n".join(display_choices),
            "choices": choices,
            "gold": int(doc["answer"]),
            "subject": doc["subject"],
        }

    return dataset.map(process_doc)


def process_mmlu_redux(dataset):
    accepted_error_types = {
        "ok",
        "bad_question_clarity",
        "expert",
        "bad_options_clarity",
        "wrong_groundtruth",
    }

    def keep_doc(doc):
        if doc["error_type"] not in accepted_error_types:
            return False
        if doc["error_type"] != "wrong_groundtruth":
            return True
        corrected = doc["correct_answer"]
        if corrected is None:
            return False
        corrected = str(corrected).strip()
        if corrected.lower() in "abcd" and len(corrected) == 1:
            return True
        try:
            return int(corrected) in range(4)
        except ValueError:
            return False

    def process_doc(doc):
        question = doc["question"].strip()
        question = question[0].upper() + question[1:]
        choices = [
            f"{letter}. {choice.strip()}\n\n"
            for letter, choice in zip("ABCD", doc["choices"], strict=True)
        ]
        display_choices = [choice.rstrip() for choice in choices]
        gold = doc["answer"]
        if doc["error_type"] == "wrong_groundtruth":
            gold = str(doc["correct_answer"]).strip()
            if gold.lower() in "abcd" and len(gold) == 1:
                gold = "abcd".index(gold.lower())
        return {
            "query": question + "\n" + "\n".join(display_choices),
            "choices": choices,
            "gold": int(gold),
        }

    return dataset.filter(keep_doc).map(process_doc)


def process_naturalqs(dataset):
    """Reproduce Thunderdome's Natural Questions prompt normalization."""

    def process_doc(doc):
        question = doc["question"].strip()
        return {
            **doc,
            "question": question[0].upper() + question[1:] + "?",
            "answer": [answer.strip() for answer in doc["answer"]],
        }

    return dataset.map(process_doc)


def process_race(dataset):
    """Reproduce Thunderdome's RACE JSONL preprocessing in memory."""
    previous_questions = defaultdict(list)
    processed = []
    for doc in dataset:
        previous = previous_questions[doc["example_id"]]
        context = "\n\n".join(f"Q: {question}\n\nA: {answer}" for question, answer in previous)
        query = (
            f"Article:\n{doc['article'].strip()}\n\n{context.strip()}\n\n"
            f"Q: {doc['question'].strip()}\n\nA:"
        ).replace("\n\n\n\n", "\n\n")
        raw_choices = doc["options"]
        choices = [choice.strip() for choice in raw_choices]
        gold = "ABCDE".index(doc["answer"])
        processed.append({"query": query, "choices": choices, "gold": gold})
        previous.append((doc["question"], raw_choices[gold]))
    return datasets.Dataset.from_list(processed)


def process_gpqa_diamond(dataset):
    """Reproduce Thunderdome's seeded GPQA-main-then-diamond shuffling."""
    rng = np.random.RandomState(0)
    for _ in range(448):
        skipped_choices = [0, 1, 2, 3]
        rng.shuffle(skipped_choices)

    processed = []
    for doc in dataset:
        correct = doc["Correct Answer"].strip()
        choices = [
            correct,
            doc["Incorrect Answer 1"].strip(),
            doc["Incorrect Answer 2"].strip(),
            doc["Incorrect Answer 3"].strip(),
        ]
        rng.shuffle(choices)
        gold = choices.index(correct)
        choices = [
            f"{letter}. {choice}" for letter, choice in zip("ABCD", choices, strict=True)
        ]
        query = doc["Question"].strip() + "\n" + "\n".join(choices)
        processed.append({"query": query, "choices": choices, "gold": gold})
    return datasets.Dataset.from_list(processed)


def gpqa_fewshot_target(doc):
    """Render GPQA demonstrations with answer text, as Thunderdome does."""
    return doc["choices"][int(doc["gold"])].split(". ", maxsplit=1)[1]


def process_truthfulqa(dataset):
    """Reproduce Thunderdome's preprocessing of the canonical TruthfulQA CSV."""

    def process_doc(doc):
        correct_answers = [
            answer.strip().rstrip(".") + "."
            for answer in doc["Correct Answers"].split(";")
        ]
        incorrect_answers = [
            answer.strip().rstrip(".") + "."
            for answer in doc["Incorrect Answers"].split(";")
        ]
        return {
            "question": doc["Question"].strip(),
            "choices": correct_answers + incorrect_answers,
            "gold": list(range(len(correct_answers))),
        }

    return dataset.map(process_doc)


def process_super_gpqa(dataset):
    """Reproduce Thunderdome's public SuperGPQA preprocessing in memory."""

    def process_doc(doc):
        choices = doc["options"]
        if not all(
            choice.startswith(letter)
            for letter, choice in zip(
                string.ascii_uppercase[: len(choices)], choices, strict=True
            )
        ):
            choices = [
                f"{letter}. {choice.strip()}"
                for letter, choice in zip(
                    string.ascii_uppercase[: len(choices)], choices, strict=True
                )
            ]
        question = f"{doc['question'].strip()}\n" + "\n".join(choices)
        return {"question": question.strip(), "answer_letter": doc["answer_letter"]}

    return dataset.map(process_doc)


def super_gpqa_text(doc):
    if "question" in doc:
        question = doc["question"]
    else:
        choices = list(doc["choices"])
        # The shared harness sample has one punctuation drift from the fixed
        # demonstration used by Thunderdome.
        if doc["Question"].startswith("Where do most short-period comets"):
            choices[2] = choices[2].removesuffix(".")
        choices = [
            f"{letter}. {choice.strip()}"
            for letter, choice in zip(
                string.ascii_uppercase[: len(choices)], choices, strict=True
            )
        ]
        question = f"{doc['Question'].strip()}\n" + "\n".join(choices)
    return f"{question.strip()}\n{_SUPER_GPQA_INSTRUCTION}"


def super_gpqa_target(doc):
    if "explanation" not in doc:
        return doc["answer_letter"]

    explanation = doc["explanation"].strip()
    explanation = explanation.removeprefix("Let's think step by step. ")
    if doc["Question"].startswith("A microwave oven is connected"):
        explanation = explanation.replace(
            "\n\\]\nTherefore, the microwave oven",
            " Therefore, the microwave oven",
        )
    # Two Thunderdome demonstrations retain the leading newline of their raw
    # triple-quoted explanations.
    if doc["Question"].startswith(("Say the pupil", "A microwave oven")):
        explanation = "\n" + explanation
    return (
        f"Let's think step by step:\n{explanation}\n"
        f"The answer is {doc['answer']}.\n[DONE]"
    )


def process_super_gpqa_results(doc, results):
    matches = _SUPER_GPQA_ANSWER_RE.findall(results[0].strip())
    prediction = matches[-1] if matches else None
    return {
        "gpqa_cot_match": float(
            prediction is not None
            and prediction.lower() == doc["answer_letter"].lower()
        )
    }


def newline_choices(doc):
    """Append the GPQA answer terminator used for scoring and few-shot prompts."""
    return [f"{choice}\n" for choice in doc["choices"]]


def _extract_gsm8k_answer(text):
    match = _GSM8K_ANSWER_RE.search(text)
    if match is None:
        return None
    answer = match.group(1).strip().replace(",", "")
    return re.sub(r"\.0*$", "", answer)


def process_gsm8k(dataset):
    def process_doc(doc):
        context_answer = doc["answer"].replace("#### ", "Final answer: ")
        gold = _extract_gsm8k_answer(context_answer)
        if gold is None:
            raise ValueError(f"Unable to extract GSM8K answer from {doc['answer']!r}")
        return {
            "question": doc["question"],
            "context_answer": context_answer,
            "gold": gold,
        }

    return dataset.map(process_doc)


def process_gsm8k_results(doc, results):
    prediction = _extract_gsm8k_answer(results[0])
    return {"exact_match": float(prediction == doc["gold"])}


def process_drop(dataset):
    def process_doc(doc):
        answer = doc["answer"]
        if answer["spans"]:
            answers = [str(span) for span in answer["spans"]]
        elif answer["number"]:
            answers = [str(answer["number"])]
        else:
            date = answer["date"]
            answers = [str(date[key]) for key in ("day", "month", "year") if date[key]][:1]

        if not answers:
            raise ValueError(f"DROP row has no supported answer: {answer!r}")
        return {
            "query": f"Context: {doc['passage'].strip()}\nQ: {doc['question'].strip()}",
            "answers": answers,
        }

    return dataset.map(process_doc)


def _normalize_scrolls_answer(text):
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_f1(prediction, ground_truth):
    prediction_tokens = _normalize_scrolls_answer(prediction.strip()).split()
    ground_truth_tokens = _normalize_scrolls_answer(ground_truth.strip()).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def process_drop_results(doc, results):
    prediction = results[0]
    scores = [_token_f1(prediction, answer) for answer in doc["answers"]]
    if len(scores) == 1:
        f1 = scores[0]
    else:
        leave_one_out_maxes = [
            max(scores[:index] + scores[index + 1 :])
            for index in range(len(scores))
        ]
        f1 = sum(leave_one_out_maxes) / len(leave_one_out_maxes)
    exact_match = any(
        _normalize_scrolls_answer(prediction) == _normalize_scrolls_answer(answer)
        for answer in doc["answers"]
    )
    return {"exact_match": float(exact_match), "f1": 100.0 * f1}


def process_popqa(dataset):
    def process_doc(doc):
        return {
            "question": doc["question"],
            "answers": json.loads(doc["possible_answers"]),
        }

    return dataset.map(process_doc)


def truthfulqa_targets(doc):
    if "gold" in doc:
        return doc["gold"]
    return [
        index for index, label in enumerate(doc["mc2_targets"]["labels"]) if label
    ]


def winogrande_text(doc):
    return {"1": 0, "2": 1}[doc["answer"]]


def winogrande_target(doc):
    blank_end = doc["sentence"].index("_") + 1
    return doc["sentence"][blank_end:].strip()


def process_truthfulqa_results(doc, results):
    """Score TruthfulQA MC2 with Thunderdome's last-index tie break."""
    prediction = _last_argmax([result[0] for result in results])
    return {"acc": float(prediction in truthfulqa_targets(doc))}


def process_winogrande_results(doc, results):
    """Score Winogrande with Thunderdome's last-index tie break."""
    prediction = _last_argmax([result[0] for result in results])
    return {"acc": float(prediction == winogrande_text(doc))}


def winogrande_choices(doc):
    blank_start = doc["sentence"].index("_")
    prefix = doc["sentence"][:blank_start]
    return [prefix + doc["option1"], prefix + doc["option2"]]
