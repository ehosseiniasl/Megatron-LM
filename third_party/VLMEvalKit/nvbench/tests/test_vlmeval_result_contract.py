"""Focused prediction-completeness and OCRBench normalization contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest import mock

from orchestration import vlmeval_result_contract as result_contract
from orchestration.configured_eval_run import EvalConfigError
from orchestration.eval_run_config_loading import load_configured_eval_run
from orchestration.fewshot_dataset_adaptation.demonstration_selection import (
    FewshotRowSelection,
)
from orchestration.utils import JsonArtifactError, write_json
from orchestration.vlmeval_orchestration_failure_status import (
    VLMEvalOrchestrationFailureStatusError,
    write_vlmeval_orchestration_failure_status,
)
from orchestration.vlmeval_result_contract import (
    GeneratedAnswerResultFinalizationError,
    NormalizedMetric,
    PredictionCompleteness,
    VLMEvalResultContractError,
    finalize_vlmeval_generated_answer_result,
    normalized_metrics_payload,
    read_normalized_metrics,
    resolve_expected_score_artifact,
    validate_benchmark_result_contract,
    validate_prediction_completeness,
)

from phase3_generation_test_support import (
    UnitDataset,
    benchmark,
    load_unit_predictions,
)
from phase5_orchestration_test_support import (
    row_partitions as _row_partitions,
    shared_file,
)

NVBENCH = Path(__file__).resolve().parents[1]
FIXTURES = NVBENCH / "tests" / "fixtures" / "vlmeval_result_contract"
RUN_CONFIG = NVBENCH / "configs" / "run_configs" / "n4_v6_example.yaml"


class VLMEvalResultContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        suite = load_configured_eval_run(RUN_CONFIG).benchmark_suite
        assert suite is not None
        cls.benchmark = suite.benchmarks[0]
        cls.invalid_score_cases = json.loads(
            (FIXTURES / "ocrbench_score_invalid_cases.json").read_text()
        )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_directory = Path(self.temporary.name).resolve()
        self.prediction = self.run_directory / "model_OCRBench.xlsx"
        self.score = self.run_directory / "model_OCRBench_score.json"
        self.prediction.write_bytes(b"raw-prediction-artifact")
        self.prediction.chmod(0o640)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def install_score_fixture(self, name: str) -> None:
        shutil.copyfile(FIXTURES / name, self.score)
        self.score.chmod(0o640)

    def write_score(self, value: object) -> None:
        self.score.write_text(json.dumps(value), encoding="utf-8")
        self.score.chmod(0o640)

    def test_exact_score_path_resolution_and_containment(self) -> None:
        self.install_score_fixture("ocrbench_score_perfect_10.json")
        resolved = resolve_expected_score_artifact(
            self.prediction,
            "_score.json",
            self.run_directory,
        )
        self.assertEqual(resolved, self.score.resolve())
        for suffix in ("score.json", "_score", "_../score.json", "_score.txt", ""):
            with (
                self.subTest(suffix=suffix),
                self.assertRaises(VLMEvalResultContractError),
            ):
                resolve_expected_score_artifact(
                    self.prediction, suffix, self.run_directory
                )
        self.score.unlink()
        with self.assertRaises(VLMEvalResultContractError):
            resolve_expected_score_artifact(
                self.prediction, "_score.json", self.run_directory
            )

    def test_xlsx_prediction_loader_preserves_literal_na_text(self) -> None:
        fake_file = types.ModuleType("vlmeval.smp.file")
        fake_file.pd = mock.Mock()
        fake_file.pd.read_excel.return_value = "preserved"
        original_load = mock.Mock(return_value="ordinary")
        fake_file.load = original_load
        fake_smp = types.ModuleType("vlmeval.smp")
        fake_smp.file = fake_file
        fake_vlmeval = types.ModuleType("vlmeval")
        fake_vlmeval.smp = fake_smp

        modules = {
            "vlmeval": fake_vlmeval,
            "vlmeval.smp": fake_smp,
            "vlmeval.smp.file": fake_file,
        }
        with mock.patch.dict(sys.modules, modules):
            self.assertEqual(
                result_contract._load_upstream_artifact(str(self.prediction)),
                "preserved",
            )
            with result_contract._preserve_prediction_text_during_evaluation(
                self.prediction
            ):
                self.assertEqual(fake_file.load(str(self.prediction)), "preserved")
                self.assertEqual(fake_file.load("other.pkl"), "ordinary")

        self.assertIs(fake_file.load, original_load)
        fake_file.pd.read_excel.assert_called_with(
            str(self.prediction), keep_default_na=False
        )

    def test_score_path_rejects_nonfiles_and_realpath_escape(self) -> None:
        outside = self.run_directory.parent / f"{self.run_directory.name}-outside.xlsx"
        outside_score = outside.with_name(f"{outside.stem}_score.json")
        outside.write_text("prediction", encoding="utf-8")
        outside_score.write_text("{}", encoding="utf-8")
        try:
            with self.assertRaises(VLMEvalResultContractError):
                resolve_expected_score_artifact(
                    outside, "_score.json", self.run_directory
                )
            self.score.symlink_to(outside_score)
            with self.assertRaises(VLMEvalResultContractError):
                resolve_expected_score_artifact(
                    self.prediction,
                    "_score.json",
                    self.run_directory,
                )
        finally:
            outside.unlink(missing_ok=True)
            outside_score.unlink(missing_ok=True)

    def test_perfect_score_normalizes_in_source_order_without_mutation(self) -> None:
        self.install_score_fixture("ocrbench_score_perfect_10.json")
        before = self.score.read_bytes()
        before_hash = hashlib.sha256(before).hexdigest()
        metrics = read_normalized_metrics(
            self.benchmark, self.score, self.run_directory
        )
        self.assertEqual(
            tuple(metric.metric for metric in metrics),
            (
                "text_recognition",
                "scene_text_centric_vqa",
                "doc_oriented_vqa",
                "key_information_extraction",
                "handwritten_mathematical_expression_recognition",
                "final_score",
                "final_score_norm",
            ),
        )
        self.assertEqual(
            tuple(metric.value for metric in metrics), (6, 1, 1, 1, 1, 10, 1.0)
        )
        self.assertEqual(
            tuple(metric.unit for metric in metrics), ("count",) * 6 + ("percent",)
        )
        self.assertTrue(all(metric.dataset == "OCRBench" for metric in metrics))
        self.assertTrue(
            all(metric.source_file == self.score.name for metric in metrics)
        )
        self.assertEqual(self.score.read_bytes(), before)
        self.assertEqual(
            hashlib.sha256(self.score.read_bytes()).hexdigest(), before_hash
        )

    def test_mixed_score_and_exact_normalized_payload(self) -> None:
        self.install_score_fixture("ocrbench_score_mixed.json")
        metrics = read_normalized_metrics(
            self.benchmark, self.score, self.run_directory
        )
        payload = normalized_metrics_payload(metrics)
        expected_fields = (
            "Text Recognition",
            "Scene Text-centric VQA",
            "Doc-oriented VQA",
            "Key Information Extraction",
            "Handwritten Mathematical Expression Recognition",
            "Final Score",
            "Final Score Norm",
        )
        self.assertEqual(list(payload), ["schema_version", "metrics"])
        self.assertEqual(payload["schema_version"], 1)
        expected_keys = [
            "dataset",
            "metric",
            "value",
            "unit",
            "source_file",
            "source_field",
        ]
        self.assertTrue(
            all(list(record) == expected_keys for record in payload["metrics"])
        )
        self.assertEqual(
            [record["source_field"] for record in payload["metrics"]],
            list(expected_fields),
        )
        self.assertEqual(
            [record["value"] for record in payload["metrics"]],
            [221, 55, 47, 32, 12, 367, 36.7],
        )
        self.assertEqual(payload, normalized_metrics_payload(metrics))
        json.dumps(payload, allow_nan=False)

    def test_current_ocrbench_category_layout_is_accepted(self) -> None:
        self.write_score(
            {
                "Text Recognition": 258,
                "Scene Text-centric VQA": 168,
                "Doc-oriented VQA": 132,
                "Key Information Extraction": 132,
                "Handwritten Mathematical Expression Recognition": 34,
                "Final Score": 724,
                "Final Score Norm": 72.4,
            }
        )
        metrics = read_normalized_metrics(
            self.benchmark, self.score, self.run_directory
        )
        self.assertEqual(
            tuple(metric.value for metric in metrics),
            (258, 168, 132, 132, 34, 724, 72.4),
        )

    def test_invalid_ocrbench_scores_fail_closed(self) -> None:
        for name, payload in self.invalid_score_cases.items():
            with self.subTest(name=name):
                self.write_score(payload)
                with self.assertRaises(VLMEvalResultContractError):
                    read_normalized_metrics(
                        self.benchmark, self.score, self.run_directory
                    )
        for name in (
            "ocrbench_score_malformed.json",
            "ocrbench_score_nan.json",
            "ocrbench_score_duplicate.json",
        ):
            with self.subTest(name=name):
                self.install_score_fixture(name)
                with self.assertRaises(VLMEvalResultContractError):
                    read_normalized_metrics(
                        self.benchmark, self.score, self.run_directory
                    )
        self.write_score([])
        with self.assertRaises(VLMEvalResultContractError):
            read_normalized_metrics(self.benchmark, self.score, self.run_directory)

    def test_score_reader_rejects_permissions_paths_and_dispatch_mismatch(self) -> None:
        self.install_score_fixture("ocrbench_score_perfect_10.json")
        self.score.chmod(0o600)
        with self.assertRaises(VLMEvalResultContractError):
            read_normalized_metrics(self.benchmark, self.score, self.run_directory)
        self.score.chmod(0o640)
        wrong_name = self.run_directory / "score.json"
        shutil.copyfile(self.score, wrong_name)
        wrong_name.chmod(0o640)
        with self.assertRaises(VLMEvalResultContractError):
            read_normalized_metrics(self.benchmark, wrong_name, self.run_directory)
        changed = replace(self.benchmark, result_reader="future_reader")
        with self.assertRaises(VLMEvalResultContractError):
            read_normalized_metrics(changed, self.score, self.run_directory)

    def test_only_the_implemented_result_profile_is_configurable(self) -> None:
        for changes in (
            {"result_reader": "future_reader"},
            {"expected_result_suffix": "_acc.csv"},
            {"dataset": "OtherDataset"},
            {"upstream_class": "OtherDataset"},
        ):
            with self.subTest(changes=changes), self.assertRaises(EvalConfigError):
                validate_benchmark_result_contract(replace(self.benchmark, **changes))

    def test_manual_normalized_metric_validation_is_strict(self) -> None:
        valid = NormalizedMetric(
            dataset="OCRBench",
            metric="accuracy",
            value=0.5,
            unit="fraction",
            source_file="scores/result_score.json",
            source_field="Accuracy",
        )
        self.assertEqual(
            normalized_metrics_payload((valid,))["metrics"][0]["value"], 0.5
        )
        invalid_changes = (
            {"dataset": " OCRBench"},
            {"metric": 1},
            {"metric": "Final Score"},
            {"value": True},
            {"value": float("nan")},
            {"value": -1},
            {"value": 1.1, "unit": "fraction"},
            {"value": 100.1, "unit": "percent"},
            {"unit": "ratio"},
            {"source_file": "/absolute/result.json"},
            {"source_file": "../outside.json"},
            {"source_file": "scores\\result.json"},
            {"source_file": "."},
            {"source_field": ""},
        )
        for changes in invalid_changes:
            with (
                self.subTest(changes=changes),
                self.assertRaises(VLMEvalResultContractError),
            ):
                normalized_metrics_payload((replace(valid, **changes),))
        with self.assertRaises(VLMEvalResultContractError):
            normalized_metrics_payload(())
        with self.assertRaises(VLMEvalResultContractError):
            normalized_metrics_payload((valid, valid))
        with self.assertRaises(VLMEvalResultContractError):
            normalized_metrics_payload([valid])
        with self.assertRaises(FrozenInstanceError):
            valid.value = 1.0


class _PandasLikeRecords:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records
        self.orients: list[str] = []

    def to_dict(self, *, orient: str):
        self.orients.append(orient)
        return self.records


class VLMEvalPredictionCompletenessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prediction_cases = json.loads(
            (FIXTURES / "prediction_cases.json").read_text()
        )

    def test_complete_records_and_pandas_like_data_are_not_mutated(self) -> None:
        records = self.prediction_cases["complete"]
        original = copy.deepcopy(records)
        expected = tuple(range(10))
        result = validate_prediction_completeness(records, expected)
        self.assertEqual(result, PredictionCompleteness(10, expected))
        self.assertEqual(records, original)
        self.assertEqual(
            validate_prediction_completeness(tuple(records), expected), result
        )
        frame = _PandasLikeRecords(records)
        self.assertEqual(validate_prediction_completeness(frame, expected), result)
        self.assertEqual(frame.orients, ["records"])
        self.assertEqual(frame.records, original)
        with self.assertRaises(FrozenInstanceError):
            result.observed_row_count = 0

    def test_exact_failure_sentinel_is_an_auditable_wrong_answer(self) -> None:
        records = [{"index": 0, "prediction": "Failed to obtain answer"}]
        self.assertEqual(
            validate_prediction_completeness(records, (0,)),
            PredictionCompleteness(1, (0,)),
        )

    def test_count_duplicate_order_and_content_fail_closed(self) -> None:
        cases = (
            ("partial", tuple(range(10))),
            ("extra", (0, 1)),
            ("duplicate", (0, 1)),
            ("reordered", (0, 1)),
            ("failure_sentinel", (0,)),
            ("nonstring", (0,)),
            ("missing_field", (0,)),
        )
        for name, expected in cases:
            with self.subTest(name=name), self.assertRaises(VLMEvalResultContractError):
                validate_prediction_completeness(self.prediction_cases[name], expected)
        malformed = (
            None,
            {"index": 0, "prediction": "answer"},
            [1],
            [{"index": 0, "prediction": "  "}],
            [{"index": 0, "prediction": float("nan")}],
            _PandasLikeRecords([]),
        )
        for value in malformed:
            with (
                self.subTest(value=value),
                self.assertRaises(VLMEvalResultContractError),
            ):
                validate_prediction_completeness(value, (0,))

    def test_indices_are_deterministic_json_scalars(self) -> None:
        for index in (True, 1.5, "", " spaced "):
            with (
                self.subTest(index=index),
                self.assertRaises(VLMEvalResultContractError),
            ):
                validate_prediction_completeness(
                    [{"index": index, "prediction": "answer"}],
                    (index,),
                )
        with self.assertRaises(VLMEvalResultContractError):
            validate_prediction_completeness([], (0, 0))
        with self.assertRaises(VLMEvalResultContractError):
            validate_prediction_completeness([], (1, "1"))
        result = validate_prediction_completeness(
            [{"index": "target-1", "prediction": "answer"}],
            ("target-1",),
        )
        self.assertEqual(result.observed_indices, ("target-1",))

    def test_finite_numeric_answers_survive_spreadsheet_type_inference(self) -> None:
        records = [
            {"index": 5, "prediction": 0.62},
            {"index": 6, "prediction": 23},
        ]
        original = copy.deepcopy(records)
        result = validate_prediction_completeness(records, (5, 6))
        self.assertEqual(result, PredictionCompleteness(2, (5, 6)))
        self.assertEqual(records, original)
        for prediction in (True, False, float("nan"), float("inf"), float("-inf")):
            with (
                self.subTest(prediction=prediction),
                self.assertRaises(VLMEvalResultContractError),
            ):
                validate_prediction_completeness(
                    [{"index": 5, "prediction": prediction}],
                    (5,),
                )


def _selection() -> FewshotRowSelection:
    return FewshotRowSelection((), (), (1, 2), ("row-1", "row-2"))


def _prediction(root: Path) -> Path:
    path = root / "prediction.json"
    path.write_bytes(b'{"opaque":true}\n')
    path.chmod(0o640)
    return path


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class GeneratedAnswerResultFinalizationTest(unittest.TestCase):
    def test_success_preserves_raw_artifacts_and_writes_owned_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = _prediction(root)
            dataset = UnitDataset()
            prediction_before = prediction.read_bytes()
            partitions = _row_partitions(root)
            partitions_before = partitions.artifact.read_bytes()
            result = finalize_vlmeval_generated_answer_result(
                dataset,
                benchmark(),
                _selection(),
                prediction,
                root,
                row_partitions=partitions,
                evaluation_judge="unit-llm-judge",
                prediction_loader=load_unit_predictions,
            )
            score_before = result.score_artifact.read_bytes()
            status_payload = _load_json(result.status_artifact)
            status = status_payload["datasets"][0]

            self.assertEqual(dataset.evaluate_calls, 1)
            self.assertEqual(dataset.evaluate_arguments, [str(prediction)])
            self.assertEqual(dataset.evaluate_keywords, [{"model": "unit-llm-judge"}])
            self.assertEqual(prediction.read_bytes(), prediction_before)
            self.assertEqual(partitions.artifact.read_bytes(), partitions_before)
            self.assertEqual(result.score_artifact.read_bytes(), score_before)
            self.assertEqual(status_payload["schema_version"], 3)
            self.assertEqual(result.row_partitions_artifact, partitions.artifact)
            self.assertEqual(status["prediction_path"], "prediction.json")
            self.assertEqual(status["score_path"], "prediction_score.json")
            self.assertEqual(status["metrics_path"], "vlmeval_metrics.v1.json")
            self.assertEqual(status["observed_prediction_count"], 2)
            self.assertIsNone(status["failure"])
            self.assertEqual(len(result.normalized_metrics), 7)
            self.assertEqual(len(_load_json(result.metrics_artifact)["metrics"]), 7)
            self.assertEqual(result.metrics_artifact.stat().st_mode & 0o777, 0o640)
            self.assertEqual(result.status_artifact.stat().st_mode & 0o777, 0o640)

    def test_finite_numeric_predictions_reach_native_scoring_unchanged(self) -> None:
        def numeric_predictions(_path: str) -> list[dict[str, object]]:
            return [
                {"index": "row-1", "prediction": 0.62},
                {"index": "row-2", "prediction": 23},
            ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = _prediction(root)
            before = prediction.read_bytes()
            dataset = UnitDataset()
            result = finalize_vlmeval_generated_answer_result(
                dataset,
                benchmark(),
                _selection(),
                prediction,
                root,
                row_partitions=_row_partitions(root),
                prediction_loader=numeric_predictions,
            )
            self.assertEqual(result.observed_prediction_count, 2)
            self.assertEqual(dataset.evaluate_calls, 1)
            self.assertEqual(prediction.read_bytes(), before)
            status = _load_json(result.status_artifact)["datasets"][0]
            self.assertEqual(status["status"], "completed")
            self.assertIsNone(status["failure"])

    def test_loader_and_completeness_failures_write_failed_status(self) -> None:
        def failed_loader(path: str) -> object:
            raise LookupError(Path(path).name)

        loaders = (
            (failed_loader, LookupError),
            (
                lambda path: [{"index": "row-1", "prediction": "one"}],
                VLMEvalResultContractError,
            ),
        )
        for loader, cause_type in loaders:
            with self.subTest(cause_type=cause_type.__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    prediction = _prediction(root)
                    dataset = UnitDataset()
                    with self.assertRaises(
                        GeneratedAnswerResultFinalizationError
                    ) as context:
                        finalize_vlmeval_generated_answer_result(
                            dataset,
                            benchmark(),
                            _selection(),
                            prediction,
                            root,
                            row_partitions=_row_partitions(root),
                            prediction_loader=loader,
                        )
                    self.assertIsInstance(context.exception.__cause__, cause_type)
                    payload = _load_json(root / "vlmeval_dataset_status.json")
                    record = payload["datasets"][0]
                    self.assertEqual(payload["schema_version"], 3)
                    self.assertEqual(
                        record["demonstration_and_scored_row_partitions_path"],
                        "demonstration_and_scored_row_partitions.json",
                    )
                    self.assertEqual(record["status"], "failed")
                    self.assertEqual(record["failure"]["type"], cause_type.__name__)
                    self.assertIsNone(record["metrics_path"])
                    if cause_type is VLMEvalResultContractError:
                        self.assertEqual(record["observed_prediction_count"], 1)
                    self.assertEqual(dataset.evaluate_calls, 0)

    def test_evaluator_failure_is_wrapped_and_recorded(self) -> None:
        class EvaluationError(RuntimeError):
            pass

        class FailedDataset(UnitDataset):
            def evaluate(self, eval_file: object, **judge_kwargs: object) -> object:
                raise EvaluationError(Path(str(eval_file)).name)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(GeneratedAnswerResultFinalizationError) as context:
                finalize_vlmeval_generated_answer_result(
                    FailedDataset(),
                    benchmark(),
                    _selection(),
                    _prediction(root),
                    root,
                    row_partitions=_row_partitions(root),
                    prediction_loader=load_unit_predictions,
                )
            self.assertIsInstance(context.exception.__cause__, EvaluationError)
            status = _load_json(root / "vlmeval_dataset_status.json")["datasets"][0]
            self.assertEqual(status["failure"]["type"], "EvaluationError")

    def test_missing_and_malformed_scores_fail_closed(self) -> None:
        class MissingScoreDataset(UnitDataset):
            def evaluate(self, eval_file: object, **judge_kwargs: object) -> object:
                return None

        class MalformedScoreDataset(UnitDataset):
            def evaluate(self, eval_file: object, **judge_kwargs: object) -> object:
                path = Path(str(eval_file)).with_name("prediction_score.json")
                path.write_text("{}\n", encoding="utf-8")
                path.chmod(0o640)
                return None

        class PrivateScoreDataset(UnitDataset):
            def evaluate(self, eval_file: object, **judge_kwargs: object) -> object:
                result = super().evaluate(eval_file, **judge_kwargs)
                Path(str(eval_file)).with_name("prediction_score.json").chmod(0o600)
                return result

        cases = (
            (MissingScoreDataset(), VLMEvalResultContractError),
            (MalformedScoreDataset(), VLMEvalResultContractError),
            (PrivateScoreDataset(), VLMEvalResultContractError),
        )
        for dataset, cause_type in cases:
            with self.subTest(dataset=type(dataset).__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    with self.assertRaises(
                        GeneratedAnswerResultFinalizationError
                    ) as context:
                        finalize_vlmeval_generated_answer_result(
                            dataset,
                            benchmark(),
                            _selection(),
                            _prediction(root),
                            root,
                            row_partitions=_row_partitions(root),
                            prediction_loader=load_unit_predictions,
                        )
                    self.assertIsInstance(context.exception.__cause__, cause_type)

    def test_private_prediction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = _prediction(root)
            prediction.chmod(0o600)
            with self.assertRaises(GeneratedAnswerResultFinalizationError) as context:
                finalize_vlmeval_generated_answer_result(
                    UnitDataset(),
                    benchmark(),
                    _selection(),
                    prediction,
                    root,
                    row_partitions=_row_partitions(root),
                    prediction_loader=load_unit_predictions,
                )
            self.assertRegex(str(context.exception.__cause__), "group-readable")

    def test_atomic_writer_cleans_temporary_files_on_write_and_replace_errors(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "owned.json"
            with self.assertRaises(JsonArtifactError):
                write_json(destination, {"value": object()})
            self.assertFalse(list(root.glob(".owned.json.*.tmp")))

            (root / "vlmeval_metrics.v1.json").mkdir()
            with self.assertRaises(GeneratedAnswerResultFinalizationError):
                finalize_vlmeval_generated_answer_result(
                    UnitDataset(),
                    benchmark(),
                    _selection(),
                    _prediction(root),
                    root,
                    row_partitions=_row_partitions(root),
                    prediction_loader=load_unit_predictions,
                )
            self.assertFalse(list(root.glob(".*.tmp")))
            status = _load_json(root / "vlmeval_dataset_status.json")["datasets"][0]
            self.assertEqual(status["status"], "failed")
            self.assertIsNone(status["metrics_path"])


class VLMEvalRowPartitionStatusIntegrityTest(unittest.TestCase):
    def test_failure_status_rejects_wrong_private_and_outside_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "run"
            root.mkdir()
            outside = shared_file(parent / "outside.json", b"outside")
            private = _row_partitions(root)
            private.artifact.chmod(0o600)
            cases = (
                (object(), "wrong record type"),
                (type(private)(outside), "outside the run"),
                (type(private)(None), "path must be a Path"),
                (private, "group-readable"),
            )
            for value, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(
                        VLMEvalOrchestrationFailureStatusError,
                        message,
                    ):
                        write_vlmeval_orchestration_failure_status(
                            benchmark(),
                            root,
                            RuntimeError("failed"),
                            stage="vlmeval_generation",
                            row_partitions=value,  # type: ignore[arg-type]
                        )


if __name__ == "__main__":
    unittest.main()
