import asyncio
import json
from types import SimpleNamespace

import openai

import lm_eval.tasks.parametric_qa.utils as task_utils
from lm_eval.api.judge import (
    DEFAULT_JUDGE_CONCURRENCY,
    NVIDIA_INFERENCE_BASE_URL,
    AsyncOpenAIJudge,
    JudgeRequest,
    get_judge_provider,
)
from lm_eval.tasks.parametric_qa.grading import (
    FACTS_PARAMETRIC,
    SIMPLEQA_VERIFIED,
    canonical_task_name,
    format_grader_prompt,
    parse_grade,
    parse_structured_grade,
)
from lm_eval.tasks.parametric_qa.utils import normalize_answer, process_results
from nvbench.assemble_report_results import load_results
from scripts.grade_parametric_qa import grade_file, local_fast_grade, metric_values


def test_task_name_detection():
    assert canonical_task_name("simpleqa_verified_25shot") == SIMPLEQA_VERIFIED
    assert canonical_task_name("samples_facts_parametric_25shot_run.jsonl") == (
        FACTS_PARAMETRIC
    )


def test_simpleqa_grade_parsing():
    assert parse_grade(SIMPLEQA_VERIFIED, "A") == "CORRECT"
    assert parse_grade(SIMPLEQA_VERIFIED, "INCORRECT") == "INCORRECT"
    assert parse_grade(SIMPLEQA_VERIFIED, "NOT ATTEMPTED") == "NOT_ATTEMPTED"
    assert parse_grade(SIMPLEQA_VERIFIED, "unparsable") is None


def test_facts_grade_parsing_requires_final_output_or_bare_letter():
    response = "The prediction is plausible.\nOutput: [UNKNOWN]"
    assert parse_grade(FACTS_PARAMETRIC, response) == "UNKNOWN"
    assert parse_grade(FACTS_PARAMETRIC, "**Output:** MISTAKE") == "MISTAKE"
    assert parse_grade(FACTS_PARAMETRIC, "Output: **NOT_ATTEMPTED**") == "NOT_ATTEMPTED"
    assert parse_grade(FACTS_PARAMETRIC, "D") == "NOT_ATTEMPTED"
    assert parse_grade(FACTS_PARAMETRIC, "CORRECT") is None
    assert parse_structured_grade(FACTS_PARAMETRIC, '{"grade":"CORRECT"}') == "CORRECT"
    assert parse_structured_grade(SIMPLEQA_VERIFIED, '{"grade":"B"}') == "INCORRECT"


def test_grader_prompts_include_unmodified_example_values():
    prompt = format_grader_prompt(
        SIMPLEQA_VERIFIED,
        "A question with {braces}?",
        "A gold answer",
        "A prediction",
    )
    assert "Question: A question with {braces}?" in prompt
    assert "Gold target: A gold answer" in prompt
    assert "Predicted answer: A prediction" in prompt


def test_local_diagnostic_normalization_and_metrics():
    assert normalize_answer("The Blue Whale.") == "blue whale"
    assert normalize_answer("120,000 euros") == "120000 euros"
    result = process_results(
        {
            "answer": "The Blue Whale",
            "question": "Largest mammal?",
            "_parametric_qa_task": SIMPLEQA_VERIFIED,
        },
        ["blue whale."],
    )
    assert result["reference_match"] == 1
    assert result["response_nonempty"] == 1
    assert result["judge_accuracy"]["prediction"] == "blue whale."

    result = process_results(
        {
            "answer": "Paris",
            "question": "Capital of France?",
            "_parametric_qa_task": SIMPLEQA_VERIFIED,
        },
        [""],
    )
    assert result["reference_match"] == 0
    assert result["response_nonempty"] == 0
    assert (
        local_fast_grade(
            SIMPLEQA_VERIFIED,
            {"prediction": "Paris.", "gold_answer": "PARIS"},
        )
        == "CORRECT"
    )
    assert (
        local_fast_grade(
            SIMPLEQA_VERIFIED,
            {"prediction": "Who", "gold_answer": "The Who"},
        )
        is None
    )


def test_in_task_aggregation_defaults_to_128_concurrent_judges(monkeypatch):
    captured = {}

    async def fake_judge_samples(task, samples, **kwargs):
        captured["task"] = task
        captured["samples"] = samples
        captured.update(kwargs)
        return [["CORRECT", "MISTAKE", "CORRECT"]]

    monkeypatch.setattr(task_utils, "judge_samples", fake_judge_samples)
    monkeypatch.delenv("PARAMETRIC_QA_JUDGE_CONCURRENCY", raising=False)
    score = task_utils.aggregate_judge_accuracy(
        [
            {
                "task": FACTS_PARAMETRIC,
                "question": "Question",
                "gold_answer": "Gold",
                "prediction": "Prediction",
            }
        ]
    )
    assert score == 2 / 3
    assert captured["concurrency"] == DEFAULT_JUDGE_CONCURRENCY
    assert captured["replicates"] == 3


def test_nvidia_provider_is_used_by_in_task_aggregation(monkeypatch):
    captured = {}

    async def fake_judge_samples(task, samples, **kwargs):
        captured["task"] = task
        captured.update(kwargs)
        return [["CORRECT"]]

    monkeypatch.setattr(task_utils, "judge_samples", fake_judge_samples)
    monkeypatch.setenv("PARAMETRIC_QA_JUDGE_PROVIDER", "nvidia")
    score = task_utils.aggregate_judge_accuracy(
        [
            {
                "task": SIMPLEQA_VERIFIED,
                "question": "Question",
                "gold_answer": "Gold",
                "prediction": "Prediction",
            }
        ]
    )
    assert score == 1.0
    assert captured["base_url"] == NVIDIA_INFERENCE_BASE_URL
    assert captured["api_key_env"] == ("NVIDIA_API_KEY", "NVIDIA_INFERENCE_API_KEY")
    assert captured["supports_response_format"] is False
    assert captured["default_headers"] == {}


def test_nvidia_provider_configuration():
    provider = get_judge_provider("nvidia")
    assert provider.base_url == NVIDIA_INFERENCE_BASE_URL
    assert provider.api_key_envs == ("NVIDIA_API_KEY", "NVIDIA_INFERENCE_API_KEY")
    assert provider.supports_response_format is False
    assert provider.default_headers == {}


def test_missing_judge_key_explains_provider_switches(monkeypatch):
    monkeypatch.delenv("MISSING_JUDGE_KEY", raising=False)
    judge = AsyncOpenAIJudge(model="test-model", api_key_env="MISSING_JUDGE_KEY")
    try:
        asyncio.run(
            judge.run(
                [JudgeRequest(messages=[{"role": "user", "content": "grade"}])],
                lambda response: response,
            )
        )
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("Expected missing API key to raise RuntimeError")

    assert "JUDGE_API_KEY_MISSING" in message
    assert "OPENROUTER_API_KEY" in message
    assert "NVIDIA_API_KEY or NVIDIA_INFERENCE_API_KEY" in message
    assert "PARAMETRIC_QA_JUDGE_PROVIDER=openrouter|nvidia" in message


def test_library_judge_bounds_concurrency_and_preserves_order(monkeypatch):
    active = 0
    max_active = 0

    class FakeCompletions:
        async def create(self, **kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=kwargs["messages"][0]["content"]
                        )
                    )
                ],
                usage=SimpleNamespace(total_tokens=7),
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            pass

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    judge = AsyncOpenAIJudge(
        model="test-model",
        api_key="test",
        concurrency=2,
        response_retries=0,
    )
    requests = [
        JudgeRequest(messages=[{"role": "user", "content": str(index)}], metadata=index)
        for index in range(5)
    ]
    results = asyncio.run(judge.run(requests, lambda response: int(response)))
    assert max_active == 2
    assert [result.value for result in results] == list(range(5))
    assert [result.request.metadata for result in results] == list(range(5))


def test_library_judge_retries_unparsable_responses(monkeypatch):
    seeds = []

    class FakeCompletions:
        async def create(self, **kwargs):
            seeds.append(kwargs["seed"])
            content = "unparsable" if len(seeds) == 1 else "7"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(total_tokens=len(seeds)),
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            pass

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    judge = AsyncOpenAIJudge(
        model="test-model",
        api_key="test",
        concurrency=1,
        response_retries=1,
    )
    request = JudgeRequest(
        messages=[{"role": "user", "content": "grade"}],
        seed=5,
    )
    result = asyncio.run(
        judge.run(
            [request],
            lambda response: int(response) if response.isdigit() else None,
        )
    )[0]
    assert result.value == 7
    assert result.usage["total_tokens"] == 3
    assert seeds == [5, 1_000_008]


def test_library_judge_can_skip_structured_response_format(monkeypatch):
    create_kwargs = []

    class FakeCompletions:
        async def create(self, **kwargs):
            create_kwargs.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="A"))],
                usage=None,
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            pass

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    judge = AsyncOpenAIJudge(
        model="openai/gpt-oss-120b",
        api_key="test",
        supports_response_format=False,
        response_retries=0,
    )
    result = asyncio.run(
        judge.run(
            [
                JudgeRequest(
                    messages=[{"role": "user", "content": "grade"}],
                    response_format={"type": "json_schema"},
                )
            ],
            lambda response: response if response == "A" else None,
        )
    )[0]
    assert result.value == "A"
    assert "response_format" not in create_kwargs[0]


def test_facts_metrics_average_votes_and_report_strict_accuracy():
    records = {
        0: [{"grade": "CORRECT"}] * 3,
        1: [
            {"grade": "CORRECT"},
            {"grade": "UNKNOWN"},
            {"grade": "MISTAKE"},
        ],
    }
    metrics = metric_values(FACTS_PARAMETRIC, records, num_samples=2)
    assert metrics["judge_accuracy"] == 4 / 6
    assert metrics["judge_unknown_rate"] == 1 / 6
    assert metrics["judge_strict_accuracy"] == 1 / 2


def test_local_fast_path_writes_resumable_cache_and_lm_eval_results(tmp_path):
    samples_path = tmp_path / "samples_simpleqa_verified_25shot_test.jsonl"
    samples = [
        {
            "doc_id": 0,
            "doc": {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "_parametric_qa_task": "simpleqa_verified",
            },
            "filtered_resps": [["Paris."]],
        },
        {
            "doc_id": 1,
            "doc": {
                "question": "What is the capital of Japan?",
                "answer": "Tokyo",
                "_parametric_qa_task": "simpleqa_verified",
            },
            "filtered_resps": [[""]],
        },
    ]
    samples_path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )
    (tmp_path / "results_test.json").write_text(
        json.dumps(
            {
                "config": {"limit": None},
                "n-samples": {
                    "simpleqa_verified_25shot": {
                        "original": 2,
                        "effective": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        judge_model="auto",
        replicates=None,
        cache_path=None,
        no_local_fast_path=False,
        dry_run=False,
        api_key_env="THIS_KEY_MUST_NOT_BE_NEEDED",
        base_url="https://example.invalid/v1",
        timeout=1,
        max_retries=0,
        concurrency=1,
        temperature=None,
        output_dir=tmp_path,
        filter_name="openrouter",
    )

    asyncio.run(grade_file(samples_path, args))

    cache_path = samples_path.with_suffix(".openrouter-cache.jsonl")
    cache_records = [json.loads(line) for line in cache_path.read_text().splitlines()]
    assert [record["source"] for record in cache_records] == [
        "local_fast_path",
        "local_fast_path",
    ]
    result = json.loads(
        (tmp_path / "results_judged_simpleqa_verified_25shot.json").read_text()
    )
    assert result["config"]["limit"] is None
    assert result["config"]["deferred_judge"]["source_results"].endswith(
        "results_test.json"
    )
    task_results = result["results"]["simpleqa_verified_25shot"]
    assert task_results["judge_accuracy,openrouter"] == 0.5
    assert task_results["judge_attempted_accuracy,openrouter"] == 1.0
    assert task_results["judge_hedging_rate,openrouter"] == 0.5
    assembled = load_results(
        tmp_path,
        {
            "tasks": [
                {
                    "task_name": "simpleqa_verified_25shot",
                    "report_metric": [
                        {
                            "alias": "simpleqa_verified_25shot",
                            "metric": "judge_accuracy,openrouter",
                        }
                    ],
                }
            ]
        },
    )
    assert (
        assembled[
            (
                "simpleqa_verified_25shot",
                "judge_accuracy,openrouter",
                25,
                1.0,
            )
        ]
        == 0.5
    )


def test_judge_path_caches_completed_calls(tmp_path, monkeypatch):
    samples_path = tmp_path / "samples_simpleqa_verified_25shot_test.jsonl"
    sample = {
        "doc_id": 0,
        "doc": {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "_parametric_qa_task": "simpleqa_verified",
        },
        "filtered_resps": [["London"]],
    }
    samples_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"grade":"B"}'))
                ],
                usage=SimpleNamespace(total_tokens=7),
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            pass

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "test")
    args = SimpleNamespace(
        judge_model="auto",
        replicates=None,
        cache_path=None,
        no_local_fast_path=False,
        dry_run=False,
        api_key_env="TEST_OPENROUTER_KEY",
        base_url="https://example.invalid/v1",
        timeout=1,
        max_retries=0,
        concurrency=1,
        temperature=None,
        output_dir=tmp_path,
        filter_name="openrouter",
    )

    asyncio.run(grade_file(samples_path, args))

    cache_record = json.loads(
        samples_path.with_suffix(".openrouter-cache.jsonl").read_text()
    )
    assert cache_record["source"] == "judge"
    assert cache_record["grade"] == "INCORRECT"
    result = json.loads(
        (tmp_path / "results_judged_simpleqa_verified_25shot.json").read_text()
    )
    assert result["config"]["deferred_judge"]["usage"]["total_tokens"] == 7
