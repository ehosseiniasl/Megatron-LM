from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from orchestration.configured_eval_run import (
    BenchmarkSuiteConfig,
    ResolvedEvalRun,
)
from orchestration.fewshot_dataset_adaptation.demonstration_selection import (
    FewshotRowSelection,
)
from orchestration.eval_run_config_loading import (
    load_configured_eval_run,
)
from orchestration.eval_run_path_resolution import (
    prepare_vlmeval_run_directory,
    resolve_eval_run,
)
from orchestration.vlmeval_generation_workflow import (
    VLMEvalGenerationResult,
)
from orchestration.utils import write_json
from orchestration.vlmeval_result_contract import NormalizedMetric
from phase1_resolution_test_support import ResolutionFixture


NVBENCH = Path(__file__).resolve().parents[1]
SUITE_RUN = (
    NVBENCH
    / "configs"
    / "run_configs"
    / "n4_v6_example.yaml"
)


def three_dataset_suite() -> BenchmarkSuiteConfig:
    configured = load_configured_eval_run(SUITE_RUN)
    assert configured.benchmark_suite is not None
    return replace(
        configured.benchmark_suite,
        name="phase8_unit_suite",
        benchmarks=configured.benchmark_suite.benchmarks[:3],
    )


def resolved_suite_fixture() -> tuple[ResolutionFixture, ResolvedEvalRun]:
    fixture = ResolutionFixture()
    try:
        configured = replace(
            fixture.config(),
            benchmark=None,
            benchmark_suite=three_dataset_suite(),
        )
        resolved = resolve_eval_run(
            configured,
            fixture.workspace,
            validate_destination=False,
        )
        prepare_vlmeval_run_directory(resolved)
    except Exception:
        fixture.close()
        raise
    return fixture, resolved


def _shared(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    path.chmod(0o640)
    return path


def completed_generation_result(directory: Path, dataset: str) -> VLMEvalGenerationResult:
    selection = FewshotRowSelection((), (), (0,), (f"{dataset}-target",))
    partitions = _shared(directory / "demonstration_and_scored_row_partitions.json", b"rows")
    prediction = _shared(directory / f"model_{dataset}.xlsx", b"prediction")
    score = _shared(directory / f"model_{dataset}_score.bin", b"score")
    metric = NormalizedMetric(
        dataset,
        "overall",
        0.5,
        "fraction",
        score.name,
        "Overall",
    )
    metrics = directory / "vlmeval_metrics.v1.json"
    write_json(
        metrics,
        {
            "schema_version": 1,
            "metrics": [
                {
                    "dataset": metric.dataset,
                    "metric": metric.metric,
                    "value": metric.value,
                    "unit": metric.unit,
                    "source_file": metric.source_file,
                    "source_field": metric.source_field,
                }
            ],
        },
    )
    status = directory / "vlmeval_dataset_status.json"
    record = {
        "dataset": dataset,
        "status": "completed",
        "expected_scored_indices": list(selection.scored_indices),
        "observed_prediction_count": 1,
        "demonstration_and_scored_row_partitions_path": partitions.name,
        "prediction_path": prediction.name,
        "score_path": score.name,
        "metrics_path": metrics.name,
        "failure": None,
    }
    write_json(status, {"schema_version": 3, "datasets": [record]})
    return VLMEvalGenerationResult(
        selection,
        partitions,
        prediction,
        score,
        metrics,
        status,
        (metric,),
    )


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value
