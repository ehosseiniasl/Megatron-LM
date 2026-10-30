import random
from types import SimpleNamespace

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks.mistral_pretraining.utils import (
    MistralMutualInfoTask,
    ThunderdomeSampler,
    byte_normalized_argmax,
    gpqa_fewshot_target,
    process_drop_results,
    process_gsm8k_results,
    process_hellaswag_results,
    process_mutual_info_results,
    process_naturalqs,
    process_new_mmlu,
    process_piqa_results,
    process_super_gpqa_results,
    process_truthfulqa,
    super_gpqa_target,
)


def test_byte_normalized_argmax_uses_utf8_bytes():
    # "é" is one Python character but two UTF-8 bytes. Character normalization
    # would choose it, while the Thunderdome byte normalization chooses "aa".
    assert byte_normalized_argmax([-1.5, -1.2], ["é", "aa"]) == 1


def test_byte_normalized_argmax_uses_thunderdome_tie_break():
    assert byte_normalized_argmax([-1.0, -2.0], ["a", "bb"]) == 1


def test_mutual_info_results_pair_conditional_and_unconditional_blocks():
    # Requests are constructed as all conditionals followed by all baselines.
    results = [(-2.0, False), (-3.0, False), (-10.0, False), (-20.0, False)]

    metrics = process_mutual_info_results(
        {"choices": ["A", "B"], "gold": 1}, results
    )

    assert metrics == {"acc_mutual_info": 1.0}


def test_mutual_info_task_uses_thunderdome_baseline_context(monkeypatch):
    task = object.__new__(MistralMutualInfoTask)
    task._mutual_info_context = "Answer:"
    task._config = SimpleNamespace(target_delimiter=" ")
    task.doc_to_choice = lambda doc: doc["choices"]

    requests = [
        SimpleNamespace(arguments=("Question", " one")),
        SimpleNamespace(arguments=("Question", " two")),
        SimpleNamespace(arguments=("", "one")),
        SimpleNamespace(arguments=("", "two")),
    ]
    monkeypatch.setattr(
        ConfigurableTask,
        "construct_requests",
        lambda *args, **kwargs: requests,
    )
    actual = task.construct_requests({"choices": ["one", "two"]}, "Question")

    assert [request.arguments for request in actual] == [
        ("Question", " one"),
        ("Question", " two"),
        ("Answer:", " one"),
        ("Answer:", " two"),
    ]


def test_piqa_normalizes_by_unstripped_dataset_choice_bytes():
    doc = {"sol1": "a ", "sol2": "bb", "label": 0}

    # With stripped lengths choice 1 wins (-0.75 > -1.0); with the raw two-byte
    # PIQA choice lengths, choice 0 wins (-0.5 > -0.75).
    assert process_piqa_results(doc, [(-1.0, False), (-1.5, False)]) == {
        "acc": 1.0
    }


def test_hellaswag_normalizes_by_pre_strip_choice_bytes():
    doc = {
        "raw_choices": [" a", "bb"],
        "choices": ["a", "bb"],
        "gold": 0,
    }

    # Both continuations are scored after stripping, while the leading byte in
    # the first raw choice is still part of Thunderdome's normalization length.
    assert process_hellaswag_results(doc, [(-1.0, False), (-1.5, False)]) == {
        "acc": 1.0
    }


def test_thunderdome_sampler_is_seeded_and_decontaminates():
    docs = [
        {"question": f"q{index}", "answer": [f"a{index}"]}
        for index in range(8)
    ]
    task = SimpleNamespace(
        _config=SimpleNamespace(
            target_delimiter=" ",
            fewshot_delimiter="\n",
            fewshot_config=None,
            doc_to_choice=None,
        ),
        doc_to_text=lambda doc: doc["question"],
        doc_to_target=lambda doc: doc["answer"],
        doc_to_choice=lambda doc: None,
        eval_docs=docs,
    )
    sampler = ThunderdomeSampler(docs, task, rnd=random.Random(1234))

    context = sampler.get_context(docs[0], 3)

    assert context == "q1 a1\nq4 a4\nq7 a7\n"
    assert "q0 a0" not in context
    assert sampler.get_context(docs[0], 3) == context


def test_thunderdome_sampler_renders_multiple_choice_target():
    docs = [{"question": "q0", "choices": ["A", "B"], "answer": 1}]
    task = SimpleNamespace(
        _config=SimpleNamespace(
            target_delimiter=" ",
            fewshot_delimiter="\n",
            fewshot_config=None,
            doc_to_choice="choices",
        ),
        doc_to_text=lambda doc: doc["question"],
        doc_to_target=lambda doc: doc["answer"],
        doc_to_choice=lambda doc: doc["choices"],
    )
    sampler = ThunderdomeSampler(docs, task, rnd=random.Random(1234))

    assert sampler.get_context({"question": "different"}, 1) == "q0 B\n"


def test_gpqa_fewshot_target_omits_choice_label():
    doc = {"choices": ["A. first", "B. second"], "gold": 1}

    assert gpqa_fewshot_target(doc) == "second"


def test_truthfulqa_reproduces_raw_csv_answer_preprocessing():
    doc = {
        "Question": "  A question? ",
        "Correct Answers": "first; second.",
        "Incorrect Answers": "wrong; also wrong.",
    }
    dataset = SimpleNamespace(map=lambda function: function(doc))

    assert process_truthfulqa(dataset) == {
        "question": "A question?",
        "choices": ["first.", "second.", "wrong.", "also wrong."],
        "gold": [0, 1],
    }


def test_new_mmlu_keeps_suffix_out_of_display_prompt():
    doc = {
        "question": "which answer?",
        "choices": ["one", "two", "three", "four"],
        "answer": 1,
        "subject": "test",
    }
    dataset = SimpleNamespace(map=lambda function: function(doc))

    processed = process_new_mmlu(dataset)

    assert processed["query"] == (
        "Which answer?\nA. one\nB. two\nC. three\nD. four"
    )
    assert processed["choices"] == [
        "A. one\n\n",
        "B. two\n\n",
        "C. three\n\n",
        "D. four\n\n",
    ]


def test_naturalqs_reproduces_thunderdome_question_normalization():
    doc = {"question": "  where is it  ", "answer": [" answer "]}
    dataset = SimpleNamespace(map=lambda function: function(doc))

    assert process_naturalqs(dataset) == {
        "question": "Where is it?",
        "answer": ["answer"],
    }


def test_gsm8k_v3_requires_final_answer_at_end():
    doc = {"gold": "1200"}

    assert process_gsm8k_results(doc, ["work\nFinal answer: 1,200.0"])[
        "exact_match"
    ] == 1.0
    assert process_gsm8k_results(doc, ["The answer is 1,200"])["exact_match"] == 0.0


def test_drop_scoring_matches_thunderdome_scale_and_normalization():
    result = process_drop_results(
        {"answers": ["the blue bicycle"]}, ["Blue bicycle!"]
    )

    assert result == {"exact_match": 1.0, "f1": 100.0}


def test_super_gpqa_uses_last_cot_answer_and_thunderdome_target():
    doc = {
        "Question": "Example",
        "explanation": "Let's think step by step. Reasoning.",
        "answer": "(B)",
        "answer_letter": "B",
    }

    assert super_gpqa_target(doc) == (
        "Let's think step by step:\nReasoning.\nThe answer is (B).\n[DONE]"
    )
    assert process_super_gpqa_results(
        doc, ["The answer is (A), but after checking, the answer is: B."]
    ) == {"gpqa_cot_match": 1.0}
