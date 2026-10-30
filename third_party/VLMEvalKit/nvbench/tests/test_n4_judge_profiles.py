"""Contract tests for judge-backed benchmarks in the N4 suite."""

from pathlib import Path

from orchestration.eval_component_config_loading import load_benchmark_suite
from orchestration.judge_runtime import judge_artifact_label
from orchestration.vlmeval_result_contract import validate_benchmark_result_contract


NVBENCH = Path(__file__).resolve().parents[1]
SUITE = NVBENCH / "configs" / "suites" / "n4_v6.yaml"
JUDGE_DATASETS = (
    "WorldVQA",
    "ChartQAPro",
    "MathVista_MINI",
    "MMMU_Pro_10c",
    "MMMU_Pro_10c_COT",
)


def _judge_benchmarks():
    suite = load_benchmark_suite(SUITE)
    by_dataset = {item.dataset: item for item in suite.benchmarks}
    return tuple(by_dataset[dataset] for dataset in JUDGE_DATASETS)


def test_judge_models_and_result_artifacts_are_bound() -> None:
    for benchmark in _judge_benchmarks():
        validate_benchmark_result_contract(benchmark)
        if benchmark.judge == "exact_matching":
            assert benchmark.expected_result_suffix == "_acc.csv"
            continue
        label = judge_artifact_label(benchmark.judge)
        assert benchmark.expected_result_suffix == f"_{label}_score.csv"


def test_provider_model_is_explicit_for_every_judged_benchmark() -> None:
    judged = tuple(
        item.judge
        for item in _judge_benchmarks()
        if item.judge != "exact_matching"
    )
    assert judged == (
        "us/azure/openai/eccn-gpt-4.1-mini",
        "us/azure/openai/eccn-gpt-4.1-mini",
    )
