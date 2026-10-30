"""Result-profile validation, prediction completeness, and metric normalization."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path, PurePosixPath

from .configured_eval_run import BenchmarkConfig, EvalConfigError
from .fewshot_dataset_adaptation.demonstration_selection import (
    DemonstrationAndScoredRowPartitionsArtifact,
    FewshotRowSelection,
)
from .judge_runtime import evaluate_with_judge_model
from .utils import (
    VLMEvalRowIdentityError,
    VLMEvalRowIndex,
    VLMEVAL_FAILURE_SENTINEL,
    is_safe_relative_path,
    relative_path,
    require_shared_file,
    require_unique_vlmeval_row_indices,
    resolve_directory,
    sanitized_error_text,
    write_json,
)


class VLMEvalResultContractError(RuntimeError):
    """Raised when a prediction or score artifact violates this result contract."""


class GeneratedAnswerResultFinalizationError(RuntimeError):
    """Raised after a generated-answer result cannot be safely finalized."""


@dataclass(frozen=True)
class NormalizedMetric:
    """One immutable validated metric with exact source-field provenance."""

    dataset: str
    metric: str
    value: int | float
    unit: str
    source_file: str
    source_field: str


@dataclass(frozen=True)
class PredictionCompleteness:
    """Immutable prediction cardinality and canonical indices in scored-row order."""

    observed_row_count: int
    observed_indices: tuple[VLMEvalRowIndex, ...]


@dataclass(frozen=True)
class GeneratedAnswerResultArtifacts:
    """Immutable paths and normalized metrics produced by one finalization.

    Paths identify raw prediction and score inputs plus owned metrics and
    status artifacts. Construction mutates nothing; the finalizer returns this
    record only after all path, completeness, scoring, and write checks pass.
    """

    row_partitions_artifact: Path
    prediction_artifact: Path
    score_artifact: Path
    metrics_artifact: Path
    status_artifact: Path
    normalized_metrics: tuple[NormalizedMetric, ...]
    observed_prediction_count: int


_SAFE_SCORE_SUFFIX = re.compile(r"_[a-z0-9][a-z0-9_-]*\.(json|csv)")
_SAFE_METRIC = re.compile(r"[a-z][a-z0-9_]*")
_OCRBENCH_RESULT = (
    "OCRBench",
    "OCRBench",
    "ocrbench_score_json",
    "_score.json",
    "reader_defined",
)
_SCREENSPOT_DATASETS = frozenset(
    {"ScreenSpot_Mobile", "ScreenSpot_Desktop", "ScreenSpot_Web"}
)
_SCREENSPOT_RESULT = (
    "ScreenSpot",
    "screenspot_score_json",
    "_score.json",
    "reader_defined",
)
_ACCURACY_CSV_RESULTS = frozenset(
    {
        ("vlmeval_accuracy_csv", "_acc.csv", "fraction"),
        ("vlmeval_accuracy_csv", "_acc.csv", "percent"),
    }
)
_PRIMARY_METRIC_CSV_RESULTS = frozenset(
    {("chartqapro_primary_csv", "_acc.csv", "fraction")}
)
_JUDGED_PRIMARY_METRIC_CSV_RESULTS = frozenset(
    {
        ("mathvista_primary_csv", "percent"),
        ("worldvqa_primary_csv", "fraction"),
    }
)
METRICS_FILENAME = "vlmeval_metrics.v1.json"
STATUS_FILENAME = "vlmeval_dataset_status.json"
DATASET_STATUS_RECORD_FIELDS = frozenset(
    {
        "dataset",
        "status",
        "expected_scored_indices",
        "observed_prediction_count",
        "demonstration_and_scored_row_partitions_path",
        "prediction_path",
        "score_path",
        "metrics_path",
        "failure",
    }
)


def _is_supported_primary_metric_csv(
    reader: str, suffix: str, unit: str
) -> bool:
    profile = (reader, suffix, unit)
    judged_profile = (reader, unit)
    judged_suffix = (
        _SAFE_SCORE_SUFFIX.fullmatch(suffix) is not None
        and suffix.endswith("_score.csv")
    )
    return profile in _PRIMARY_METRIC_CSV_RESULTS or (
        judged_profile in _JUDGED_PRIMARY_METRIC_CSV_RESULTS and judged_suffix
    )


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise VLMEvalResultContractError(message)


def validate_benchmark_result_contract(benchmark: BenchmarkConfig) -> None:
    """Validate one benchmark's implemented result-reader profile without mutation.

    ``benchmark`` must name the pinned OCRBench contract or a supported
    accuracy-CSV profile. Unsupported combinations raise ``EvalConfigError``.
    """
    configured = (
        benchmark.dataset,
        benchmark.upstream_class,
        benchmark.result_reader,
        benchmark.expected_result_suffix,
        benchmark.result_metric_unit,
    )
    if configured == _OCRBENCH_RESULT:
        return
    if configured[0] in _SCREENSPOT_DATASETS and configured[1:] == _SCREENSPOT_RESULT:
        return
    csv_profile = (
        benchmark.result_reader,
        benchmark.expected_result_suffix,
        benchmark.result_metric_unit,
    )
    if (
        csv_profile not in _ACCURACY_CSV_RESULTS
        and not _is_supported_primary_metric_csv(*csv_profile)
    ):
        raise EvalConfigError("unsupported VLMEvalKit benchmark result contract")


def _prediction_records(prediction_data: object) -> list[Mapping[object, object]]:
    if type(prediction_data) in (list, tuple):
        raw_records = prediction_data
    else:
        converter = getattr(prediction_data, "to_dict", None)
        _fail(
            callable(converter),
            "prediction data must be records or a pandas-like object",
        )
        try:
            raw_records = converter(orient="records")
        except Exception as error:
            raise VLMEvalResultContractError(
                "prediction record conversion failed"
            ) from error
    _fail(
        type(raw_records) in (list, tuple),
        "prediction conversion did not return records",
    )
    _fail(
        all(isinstance(record, Mapping) for record in raw_records),
        "malformed prediction row",
    )
    return list(raw_records)


def _unique_indices(
    values: tuple[object, ...], label: str
) -> tuple[VLMEvalRowIndex, ...]:
    try:
        return require_unique_vlmeval_row_indices(values, label=f"{label} indices")
    except VLMEvalRowIdentityError as error:
        raise VLMEvalResultContractError(str(error)) from error


def validate_prediction_completeness(
    prediction_data: object,
    expected_indices: tuple[VLMEvalRowIndex, ...],
) -> PredictionCompleteness:
    """Return a complete ordered prediction summary without mutating its inputs.

    ``prediction_data`` must be records or expose ``to_dict(orient="records")``.
    Every row requires a canonical index and nonempty text or finite numeric
    prediction. The exact failure sentinel is retained as an auditable wrong
    answer after bounded empty-response retries; decorated sentinel strings
    remain invalid because they may represent transport failures. Malformed,
    partial, duplicate, or reordered rows raise
    :class:`VLMEvalResultContractError`.
    """
    _fail(type(expected_indices) is tuple, "expected indices must be a tuple")
    canonical_expected = _unique_indices(expected_indices, "expected")
    records = _prediction_records(prediction_data)
    _fail(
        len(records) == len(canonical_expected),
        "prediction row count differs from expected",
    )
    observed: list[object] = []
    for position, record in enumerate(records):
        _fail(
            "index" in record and "prediction" in record,
            f"row {position} lacks required fields",
        )
        prediction = record["prediction"]
        if type(prediction) is str:
            _fail(bool(prediction.strip()), f"row {position} prediction is empty")
            _fail(
                prediction == VLMEVAL_FAILURE_SENTINEL
                or VLMEVAL_FAILURE_SENTINEL not in prediction,
                f"row {position} contains a failure sentinel",
            )
        else:
            numeric = not isinstance(prediction, bool) and isinstance(prediction, Real)
            _fail(
                numeric and math.isfinite(prediction),
                f"row {position} prediction must be text or a finite real number",
            )
        observed.append(record["index"])
    observed_indices = _unique_indices(tuple(observed), "observed")
    _fail(
        observed_indices == canonical_expected,
        "prediction indices differ from expected order",
    )
    return PredictionCompleteness(len(records), observed_indices)


def _contained_regular_file(path: Path, run_directory: Path, label: str) -> Path:
    _fail(isinstance(path, Path), f"{label} path must be a Path")
    _fail(path.is_absolute(), f"{label} path must be absolute")
    try:
        return require_shared_file(path, resolve_directory(run_directory), label)
    except ValueError as error:
        raise VLMEvalResultContractError(str(error)) from error


def resolve_expected_score_artifact(
    prediction_artifact: Path,
    expected_result_suffix: str,
    run_directory: Path,
) -> Path:
    """Return the contained group-readable score paired with ``prediction_artifact``.

    Both paths must be absolute and the suffix must be a safe ``_name.json``
    or ``_name.csv`` value. No path is created or mutated; invalid paths,
    permissions, or suffixes raise :class:`VLMEvalResultContractError`.
    """
    valid_suffix = (
        type(expected_result_suffix) is str
        and _SAFE_SCORE_SUFFIX.fullmatch(expected_result_suffix) is not None
    )
    _fail(valid_suffix, "expected result suffix is unsafe")
    _fail(isinstance(run_directory, Path), "run directory must be a Path")
    _fail(run_directory.is_absolute(), "run directory must be absolute")
    root = resolve_directory(run_directory)
    _contained_regular_file(prediction_artifact, root, "prediction artifact")
    score = prediction_artifact.with_name(
        f"{prediction_artifact.stem}{expected_result_suffix}"
    )
    return _contained_regular_file(score, root, "score artifact")


def _validate_metrics(metrics: tuple[NormalizedMetric, ...]) -> None:
    _fail(
        type(metrics) is tuple and bool(metrics),
        "normalized metrics must be a nonempty tuple",
    )
    identities: list[tuple[str, str]] = []
    for position, metric in enumerate(metrics):
        _fail(
            type(metric) is NormalizedMetric,
            f"metric {position} has the wrong record type",
        )
        valid_dataset = type(metric.dataset) is str and bool(metric.dataset)
        _fail(
            valid_dataset and metric.dataset == metric.dataset.strip(),
            "invalid dataset",
        )
        valid_metric = type(metric.metric) is str and _SAFE_METRIC.fullmatch(
            metric.metric
        )
        _fail(bool(valid_metric), "invalid metric identifier")
        numeric = type(metric.value) is int or (
            type(metric.value) is float and math.isfinite(metric.value)
        )
        _fail(numeric and metric.value >= 0, "invalid metric value")
        valid_unit = type(metric.unit) is str and metric.unit in {
            "count",
            "fraction",
            "percent",
        }
        _fail(valid_unit, "invalid metric unit")
        if metric.unit == "fraction":
            _fail(metric.value <= 1, "fraction metric exceeds one")
        if metric.unit == "percent":
            _fail(metric.value <= 100, "percent metric exceeds 100")
        valid_source = is_safe_relative_path(metric.source_file)
        valid_source = valid_source and bool(PurePosixPath(metric.source_file).parts)
        _fail(valid_source, "invalid metric source path")
        valid_field = type(metric.source_field) is str and bool(metric.source_field)
        _fail(
            valid_field and metric.source_field == metric.source_field.strip(),
            "invalid metric source field",
        )
        identities.append((metric.dataset, metric.metric))
    _fail(len(identities) == len(set(identities)), "duplicate normalized metric")


def normalized_metrics_payload(
    metrics: tuple[NormalizedMetric, ...],
) -> dict[str, object]:
    """Return a fresh schema-v1 JSON payload from validated metrics without mutation.

    ``metrics`` must be a nonempty exact tuple of valid records. Invalid records
    raise :class:`VLMEvalResultContractError`.
    """
    _validate_metrics(metrics)
    return {"schema_version": 1, "metrics": [asdict(metric) for metric in metrics]}


def normalized_metrics_from_payload(payload: object) -> tuple[NormalizedMetric, ...]:
    """Decode one schema-v1 metrics payload without mutating ``payload``.

    The returned records are immutable. Malformed schemas, records, or metric
    values raise :class:`VLMEvalResultContractError`.
    """
    if (
        type(payload) is not dict
        or set(payload) != {"schema_version", "metrics"}
        or payload.get("schema_version") != 1
        or type(payload.get("metrics")) is not list
    ):
        raise VLMEvalResultContractError(
            "normalized metrics payload has the wrong schema"
        )
    try:
        metrics = tuple(NormalizedMetric(**item) for item in payload["metrics"])
    except (TypeError, AttributeError) as error:
        raise VLMEvalResultContractError(
            "normalized metric records are malformed"
        ) from error
    _validate_metrics(metrics)
    return metrics


def read_normalized_metrics(
    benchmark: BenchmarkConfig,
    score_artifact: Path,
    run_directory: Path,
) -> tuple[NormalizedMetric, ...]:
    """Read immutable normalized metrics through ``benchmark``'s pinned reader.

    The score must use the benchmark's suffix and be a contained shared file.
    Raw artifacts are not mutated. Unsupported profiles or malformed score
    content raise :class:`VLMEvalResultContractError`.
    """
    _fail(type(benchmark) is BenchmarkConfig, "benchmark must be a BenchmarkConfig")
    _contained_regular_file(score_artifact, run_directory, "score artifact")
    valid_name = score_artifact.name.endswith(benchmark.expected_result_suffix)
    _fail(
        valid_name and score_artifact.name != benchmark.expected_result_suffix,
        "wrong score path",
    )
    contract = (
        benchmark.dataset,
        benchmark.upstream_class,
        benchmark.result_reader,
        benchmark.expected_result_suffix,
        benchmark.result_metric_unit,
    )
    if contract == _OCRBENCH_RESULT:
        from .vlmeval_result_readers import (
            OCRBenchScoreJSONError,
            read_ocrbench_score_json,
        )

        try:
            metrics = read_ocrbench_score_json(score_artifact, run_directory)
        except OCRBenchScoreJSONError as error:
            raise VLMEvalResultContractError(
                "OCRBench score artifact is invalid"
            ) from error
    elif contract[0] in _SCREENSPOT_DATASETS and contract[1:] == _SCREENSPOT_RESULT:
        from .vlmeval_result_readers import (
            ScreenSpotScoreJSONError,
            read_screenspot_score_json,
        )

        try:
            metrics = read_screenspot_score_json(
                score_artifact, run_directory, dataset=benchmark.dataset
            )
        except ScreenSpotScoreJSONError as error:
            raise VLMEvalResultContractError(
                "ScreenSpot score artifact is invalid"
            ) from error
    elif contract[2:] in _ACCURACY_CSV_RESULTS:
        from .vlmeval_result_readers import (
            VLMEvalAccuracyCSVError,
            read_vlmeval_accuracy_csv,
        )

        try:
            metrics = read_vlmeval_accuracy_csv(
                score_artifact,
                run_directory,
                dataset=benchmark.dataset,
                unit=benchmark.result_metric_unit,
            )
        except VLMEvalAccuracyCSVError as error:
            raise VLMEvalResultContractError(
                "accuracy CSV artifact is invalid"
            ) from error
    elif _is_supported_primary_metric_csv(*contract[2:]):
        from .vlmeval_result_readers import (
            VLMEvalPrimaryMetricCSVError,
            read_vlmeval_primary_metric_csv,
        )

        try:
            metrics = read_vlmeval_primary_metric_csv(
                score_artifact,
                run_directory,
                dataset=benchmark.dataset,
                unit=benchmark.result_metric_unit,
                profile=benchmark.result_reader,
            )
        except VLMEvalPrimaryMetricCSVError as error:
            raise VLMEvalResultContractError(
                "primary-metric CSV artifact is invalid"
            ) from error
    else:
        raise VLMEvalResultContractError("unsupported benchmark result contract")
    _validate_metrics(metrics)
    return metrics


# Native finalization and owned terminal artifacts.


def _load_upstream_artifact(path: str) -> object:
    """Load one native artifact through VLMEvalKit's public helper.

    The import remains local so ordinary result-contract imports do not require
    upstream VLMEvalKit. The native loader's failures propagate to finalization.
    """
    from vlmeval.smp import file

    if Path(path).suffix.lower() == ".xlsx":
        return file.pd.read_excel(path, keep_default_na=False)
    return file.load(path)


@contextmanager
def _preserve_prediction_text_during_evaluation(
    prediction: Path,
) -> Iterator[None]:
    """Keep literal Excel answers such as ``N/A`` intact for native scoring."""
    if prediction.suffix.lower() != ".xlsx":
        yield
        return

    from vlmeval.smp import file

    original = file.load

    def load(candidate: str, fmt: str | None = None) -> object:
        try:
            is_prediction = Path(candidate).resolve() == prediction
        except OSError:
            is_prediction = False
        if is_prediction and fmt in {None, "xlsx"}:
            return file.pd.read_excel(candidate, keep_default_na=False)
        return original(candidate, fmt=fmt)

    file.load = load
    try:
        yield
    finally:
        file.load = original


def _known_count(value: object) -> int | None:
    try:
        count = len(value)  # type: ignore[arg-type]
    except Exception:
        return None
    return count if type(count) is int and count >= 0 else None


def _validated_row_partitions(
    value: DemonstrationAndScoredRowPartitionsArtifact,
    root: Path,
) -> DemonstrationAndScoredRowPartitionsArtifact:
    if type(value) is not DemonstrationAndScoredRowPartitionsArtifact:
        raise TypeError("row partitions have the wrong record type")
    artifact = require_shared_file(
        value.artifact, root, "demonstration and scored row partitions"
    )
    return DemonstrationAndScoredRowPartitionsArtifact(artifact)


def _status_payload(
    benchmark: BenchmarkConfig,
    selection: FewshotRowSelection,
    status: str,
    root: Path,
    *,
    row_partitions: DemonstrationAndScoredRowPartitionsArtifact,
    prediction: Path | None,
    score: Path | None,
    metrics: Path | None,
    observed_count: int | None,
    failure: dict[str, str] | None,
) -> dict[str, object]:
    record = {
        "dataset": benchmark.dataset,
        "status": status,
        "expected_scored_indices": list(selection.scored_indices),
        "observed_prediction_count": observed_count,
        "demonstration_and_scored_row_partitions_path": relative_path(
            row_partitions.artifact, root
        ),
        "prediction_path": relative_path(prediction, root),
        "score_path": relative_path(score, root),
        "metrics_path": relative_path(metrics, root),
        "failure": failure,
    }
    return {"schema_version": 3, "datasets": [record]}


def finalize_vlmeval_generated_answer_result(
    dataset_proxy: object,
    benchmark: BenchmarkConfig,
    selection: FewshotRowSelection,
    prediction_artifact: Path,
    run_directory: Path,
    *,
    row_partitions: DemonstrationAndScoredRowPartitionsArtifact,
    evaluation_judge: str = "exact_matching",
    prediction_loader: Callable[[str], object] | None = None,
) -> GeneratedAnswerResultArtifacts:
    """Validate, score, normalize, and record one generated-answer result.

    Inputs supply the native proxy, benchmark, selected rows, raw prediction,
    and existing row-partition artifact. Raw prediction and score files remain
    unchanged; owned metrics and status JSON are atomically written. The
    optional loader replaces only the function-local upstream loader. Any
    failure writes truthful failed status when possible, then raises
    :class:`GeneratedAnswerResultFinalizationError` with the original cause.
    """
    root = resolve_directory(run_directory)
    try:
        row_partitions = _validated_row_partitions(row_partitions, root)
    except Exception as error:
        raise GeneratedAnswerResultFinalizationError(
            f"generated-answer result finalization failed for {benchmark.dataset}"
        ) from error
    status_artifact = root / STATUS_FILENAME
    metrics_artifact = root / METRICS_FILENAME
    prediction: Path | None = None
    score: Path | None = None
    written_metrics: Path | None = None
    observed_count: int | None = None
    try:
        if (
            type(evaluation_judge) is not str
            or not evaluation_judge
            or evaluation_judge != evaluation_judge.strip()
        ):
            raise ValueError("evaluation judge must be a nonempty stripped string")
        prediction = require_shared_file(
            prediction_artifact, root, "prediction artifact"
        )
        loader = (
            _load_upstream_artifact if prediction_loader is None else prediction_loader
        )
        prediction_data = loader(str(prediction))
        observed_count = _known_count(prediction_data)
        completeness = validate_prediction_completeness(
            prediction_data, selection.scored_indices
        )
        observed_count = completeness.observed_row_count
        evaluator = getattr(dataset_proxy, "evaluate", None)
        if not callable(evaluator):
            raise TypeError("dataset proxy has no callable evaluate method")
        with _preserve_prediction_text_during_evaluation(prediction):
            evaluate_with_judge_model(
                evaluator,
                str(prediction),
                evaluation_judge,
                dataset=benchmark.dataset,
            )
        score = resolve_expected_score_artifact(
            prediction, benchmark.expected_result_suffix, root
        )
        score = require_shared_file(score, root, "score artifact")
        metrics = read_normalized_metrics(benchmark, score, root)
        write_json(metrics_artifact, normalized_metrics_payload(metrics))
        written_metrics = metrics_artifact
        row_partitions = _validated_row_partitions(row_partitions, root)
        completed = _status_payload(
            benchmark,
            selection,
            "completed",
            root,
            row_partitions=row_partitions,
            prediction=prediction,
            score=score,
            metrics=written_metrics,
            observed_count=observed_count,
            failure=None,
        )
        write_json(status_artifact, completed)
        return GeneratedAnswerResultArtifacts(
            row_partitions.artifact,
            prediction,
            score,
            metrics_artifact,
            status_artifact,
            metrics,
            observed_count,
        )
    except Exception as error:
        try:
            row_partitions = _validated_row_partitions(row_partitions, root)
            failed = _status_payload(
                benchmark,
                selection,
                "failed",
                root,
                row_partitions=row_partitions,
                prediction=prediction,
                score=score,
                metrics=written_metrics,
                observed_count=observed_count,
                failure={
                    "type": type(error).__name__,
                    "message": sanitized_error_text(error, root),
                },
            )
            write_json(status_artifact, failed)
        except Exception:
            pass
        raise GeneratedAnswerResultFinalizationError(
            f"generated-answer result finalization failed for {benchmark.dataset}"
        ) from error
