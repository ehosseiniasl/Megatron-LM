from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from orchestration.fewshot_dataset_adaptation import (
    fewshot_conversation_builder as dataset_proxy,
)
from orchestration.vlmeval_generation_workflow import (
    PREDICTIONS_READY_FILENAME,
    VLMEvalGenerationResult,
    VLMEvalGenerationWorkflowError,
    adopt_existing_vlmeval_predictions,
    run_vlmeval_generation,
    run_vlmeval_predictions,
    score_vlmeval_predictions,
)
from phase3_generation_test_support import (
    RecordingPathResolver,
    RecordingRunner,
    UnitClient,
    UnitDataset,
    benchmark,
    load_unit_predictions,
    run_unit_workflow,
)

VLMEvalFewshotDatasetProxy = dataset_proxy.VLMEvalFewshotDatasetProxy


class VLMEvalGenerationWorkflowTest(unittest.TestCase):
    def test_adopts_complete_predictions_from_a_pre_split_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = run_vlmeval_predictions(
                UnitClient(),
                benchmark(),
                root,
                "unit-model",
                api_nproc=1,
                dataset_factory=lambda _name: UnitDataset(),
                inference_runner=RecordingRunner(),
                prediction_path_resolver=RecordingPathResolver(),
                prediction_loader=load_unit_predictions,
            )
            generated.readiness_artifact.unlink()
            adopted = adopt_existing_vlmeval_predictions(
                benchmark(),
                root,
                "unit-model",
                dataset_factory=lambda _name: UnitDataset(),
                prediction_path_resolver=RecordingPathResolver(),
                prediction_loader=load_unit_predictions,
            )
            self.assertEqual(adopted.prediction_artifact, generated.prediction_artifact)
            self.assertTrue(adopted.readiness_artifact.is_file())

    def test_adoption_rejects_a_different_demonstration_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = run_vlmeval_predictions(
                UnitClient(),
                benchmark(),
                root,
                "unit-model",
                api_nproc=1,
                dataset_factory=lambda _name: UnitDataset(),
                inference_runner=RecordingRunner(),
                prediction_path_resolver=RecordingPathResolver(),
                prediction_loader=load_unit_predictions,
            )
            generated.readiness_artifact.unlink()
            partitions = json.loads(generated.row_partitions_artifact.read_text())
            partitions["demonstrations"][0]["source_index"] = "different"
            generated.row_partitions_artifact.write_text(json.dumps(partitions) + "\n")
            with self.assertRaisesRegex(
                VLMEvalGenerationWorkflowError,
                "row partitions do not match",
            ):
                adopt_existing_vlmeval_predictions(
                    benchmark(),
                    root,
                    "unit-model",
                    dataset_factory=lambda _name: UnitDataset(),
                    prediction_path_resolver=RecordingPathResolver(),
                    prediction_loader=load_unit_predictions,
                )

    def test_prediction_handoff_scores_later_without_rerunning_inference(self) -> None:
        generation_dataset = UnitDataset()
        scoring_dataset = UnitDataset()
        runner = RecordingRunner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = run_vlmeval_predictions(
                UnitClient(),
                benchmark(),
                root,
                "unit-model",
                api_nproc=2,
                dataset_factory=lambda _name: generation_dataset,
                inference_runner=runner,
                prediction_path_resolver=RecordingPathResolver(),
                prediction_loader=load_unit_predictions,
            )
            self.assertEqual(generation_dataset.evaluate_calls, 0)
            self.assertEqual(
                predictions.readiness_artifact, root / PREDICTIONS_READY_FILENAME
            )
            result = score_vlmeval_predictions(
                benchmark(),
                root,
                evaluation_judge="exact_matching",
                dataset_factory=lambda _name: scoring_dataset,
                prediction_loader=load_unit_predictions,
            )
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(scoring_dataset.evaluate_calls, 1)
        self.assertEqual(result.normalized_metrics[-1].value, 2.0)

    def test_scoring_rejects_a_tampered_prediction_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = run_vlmeval_predictions(
                UnitClient(),
                benchmark(),
                root,
                "unit-model",
                api_nproc=1,
                dataset_factory=lambda _name: UnitDataset(),
                inference_runner=RecordingRunner(),
                prediction_path_resolver=RecordingPathResolver(),
                prediction_loader=load_unit_predictions,
            )
            payload = json.loads(predictions.readiness_artifact.read_text())
            payload["observed_prediction_count"] = 1
            predictions.readiness_artifact.write_text(json.dumps(payload) + "\n")
            predictions.readiness_artifact.chmod(0o640)
            with self.assertRaisesRegex(
                VLMEvalGenerationWorkflowError, "readiness count"
            ):
                score_vlmeval_predictions(
                    benchmark(),
                    root,
                    evaluation_judge="exact_matching",
                    dataset_factory=lambda _name: UnitDataset(),
                    prediction_loader=load_unit_predictions,
                )

    def test_composes_factory_selection_proxy_runner_and_artifact(self) -> None:
        dataset = UnitDataset()
        original_rows = copy.deepcopy(dataset.data.rows)
        original_class = type(dataset)
        factory_calls: list[str] = []

        def factory(dataset_name: str) -> object:
            factory_calls.append(dataset_name)
            return dataset

        runner = RecordingRunner()
        resolver = RecordingPathResolver()
        client = UnitClient()
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            result = run_vlmeval_generation(
                client,
                benchmark(),
                work_dir,
                "unit-model",
                api_nproc=3,
                retry_failed=False,
                dataset_factory=factory,
                inference_runner=runner,
                prediction_path_resolver=resolver,
                prediction_loader=load_unit_predictions,
            )
            prediction_bytes = result.prediction_artifact.read_bytes()
            status_payload = json.loads(
                result.status_artifact.read_text(encoding="utf-8")
            )

        self.assertIsInstance(result, VLMEvalGenerationResult)
        self.assertEqual(factory_calls, ["OCRBench"])
        self.assertEqual(
            resolver.calls,
            [(str(work_dir.resolve()), "unit-model", "OCRBench", True)],
        )
        self.assertEqual(result.selection.demonstration_positions, (0,))
        self.assertEqual(result.selection.scored_indices, ("row-1", "row-2"))
        self.assertEqual(result.prediction_artifact, work_dir / "prediction.json")
        self.assertEqual(result.score_artifact, work_dir / "prediction_score.json")
        self.assertEqual(result.metrics_artifact, work_dir / "vlmeval_metrics.v1.json")
        self.assertEqual(
            result.status_artifact, work_dir / "vlmeval_dataset_status.json"
        )
        self.assertEqual(len(result.normalized_metrics), 7)
        self.assertEqual(prediction_bytes, b'{"opaque":true}\n')
        self.assertEqual(status_payload["datasets"][0]["status"], "completed")
        self.assertEqual(len(runner.calls), 1)
        invocation = runner.calls[0]
        self.assertEqual(
            set(invocation),
            {
                "model",
                "work_dir",
                "model_name",
                "dataset",
                "api_nproc",
                "retry_failed",
            },
        )
        self.assertIs(invocation["model"], client)
        self.assertEqual(invocation["work_dir"], str(work_dir.resolve()))
        self.assertEqual(invocation["model_name"], "unit-model")
        self.assertEqual(invocation["api_nproc"], 3)
        self.assertIs(invocation["retry_failed"], False)
        self.assertIsInstance(invocation["dataset"], VLMEvalFewshotDatasetProxy)
        self.assertIs(invocation["dataset"].wrapped_dataset, dataset)
        self.assertIs(type(dataset), original_class)
        self.assertEqual(dataset.data.rows, original_rows)
        self.assertEqual(dataset.evaluate_calls, 1)
        self.assertEqual(dataset.evaluate_arguments, [str(result.prediction_artifact)])
        self.assertEqual(dataset.evaluate_keywords, [{"model": "exact_matching"}])
        with self.assertRaises(FrozenInstanceError):
            result.prediction_artifact = Path("changed")  # type: ignore[misc]

    def test_rejects_invalid_work_directory_name_process_count_and_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file_path = root / "file"
            file_path.write_text("not a directory", encoding="utf-8")
            missing = root / "missing"
            read_only = root / "read-only"
            read_only.mkdir()
            read_only.chmod(0o555)

            cases = (
                {"work_dir": missing},
                {"work_dir": file_path},
                {"work_dir": read_only},
                {"model_name": ""},
                {"model_name": "../escape"},
                {"model_name": "bad/name"},
                {"model_name": "bad name"},
                {"api_nproc": 0},
                {"api_nproc": -1},
                {"api_nproc": True},
                {"api_nproc": 1.5},
                {"evaluation_judge": ""},
                {"evaluation_judge": " external"},
                {"evaluation_judge": 1},
                {"retry_failed": 1},
                {"client": object()},
            )
            try:
                for changes in cases:
                    with self.subTest(changes=changes):
                        with self.assertRaises(VLMEvalGenerationWorkflowError):
                            run_unit_workflow(root, **changes)
            finally:
                read_only.chmod(0o755)

    def test_dataset_factory_and_result_contracts_fail_closed(self) -> None:
        class WrongClass:
            pass

        candidates = (
            (None, benchmark()),
            (WrongClass(), benchmark()),
            (UnitDataset("wrong-name"), benchmark()),
        )
        incomplete = type("IncompleteDataset", (), {})()
        incomplete.dataset_name = "OCRBench"
        incomplete.data = None
        candidates += ((incomplete, benchmark(upstream_class="IncompleteDataset")),)

        no_prompt = type("NoPromptDataset", (), {})()
        no_prompt.dataset_name = "OCRBench"
        no_prompt.data = UnitDataset().data
        no_prompt.evaluate = lambda *args, **kwargs: None
        candidates += ((no_prompt, benchmark(upstream_class="NoPromptDataset")),)

        no_evaluate = type("NoEvaluateDataset", (), {})()
        no_evaluate.dataset_name = "OCRBench"
        no_evaluate.data = UnitDataset().data
        no_evaluate.build_prompt = lambda record: []
        candidates += ((no_evaluate, benchmark(upstream_class="NoEvaluateDataset")),)

        with tempfile.TemporaryDirectory() as temporary:
            for candidate, contract in candidates:
                with self.subTest(candidate=type(candidate).__name__):
                    with self.assertRaises(VLMEvalGenerationWorkflowError):
                        run_unit_workflow(
                            temporary,
                            benchmark=contract,
                            dataset_factory=lambda name, value=candidate: value,
                        )

    def test_native_specialized_dataset_subclass_is_accepted(self) -> None:
        class SpecializedUnitDataset(UnitDataset):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            result = run_unit_workflow(
                temporary,
                dataset_factory=lambda name: SpecializedUnitDataset(name),
            )
        self.assertEqual(result.selection.scored_indices, ("row-1", "row-2"))

    def test_factory_and_runner_errors_retain_their_causes(self) -> None:
        class SentinelError(Exception):
            pass

        def failed_factory(dataset_name: str) -> object:
            raise SentinelError(dataset_name)

        def failed_runner(**kwargs: object) -> object:
            raise SentinelError(str(kwargs["model_name"]))

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(VLMEvalGenerationWorkflowError) as factory_context:
                run_unit_workflow(
                    temporary,
                    dataset_factory=failed_factory,
                )
            self.assertIsInstance(factory_context.exception.__cause__, SentinelError)

            with self.assertRaises(VLMEvalGenerationWorkflowError) as runner_context:
                run_unit_workflow(
                    temporary,
                    inference_runner=failed_runner,
                )
            self.assertIsInstance(runner_context.exception.__cause__, SentinelError)

    def test_missing_unreadable_or_outside_artifact_fails(self) -> None:
        def no_artifact_runner(**kwargs: object) -> object:
            return kwargs["model"]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(VLMEvalGenerationWorkflowError):
                run_unit_workflow(
                    root,
                    inference_runner=no_artifact_runner,
                )

            outside = root.parent / "outside-prediction.json"

            def outside_resolver(*args: object, **kwargs: object) -> str:
                return str(outside)

            with self.assertRaises(VLMEvalGenerationWorkflowError):
                run_unit_workflow(
                    root,
                    inference_runner=no_artifact_runner,
                    prediction_path_resolver=outside_resolver,
                )

            def directory_artifact_runner(**kwargs: object) -> object:
                (root / "artifact-directory").mkdir()
                return kwargs["model"]

            with self.assertRaises(VLMEvalGenerationWorkflowError):
                run_unit_workflow(
                    root,
                    inference_runner=directory_artifact_runner,
                    prediction_path_resolver=RecordingPathResolver(
                        "artifact-directory"
                    ),
                )

            unreadable_runner = RecordingRunner("unreadable.json")

            def write_unreadable(**kwargs: object) -> object:
                result = unreadable_runner(**kwargs)
                (root / "unreadable.json").chmod(0o000)
                return result

            try:
                with self.assertRaises(VLMEvalGenerationWorkflowError):
                    run_unit_workflow(
                        root,
                        inference_runner=write_unreadable,
                        prediction_path_resolver=RecordingPathResolver(
                            "unreadable.json"
                        ),
                    )
            finally:
                (root / "unreadable.json").chmod(0o600)

    def test_workflow_owners_use_exact_upstream_apis_without_registry_edits(
        self,
    ) -> None:
        source_root = Path(__file__).resolve().parents[1] / "orchestration"
        workflow_source = (source_root / "vlmeval_generation_workflow.py").read_text(
            encoding="utf-8"
        )
        native_source = (source_root / "vlmeval_native_dataset_preflight.py").read_text(
            encoding="utf-8"
        )
        result_source = (source_root / "vlmeval_result_contract.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from vlmeval.dataset import build_dataset", native_source)
        self.assertIn("from vlmeval.inference import infer_data_job", workflow_source)
        self.assertIn("from vlmeval.smp import get_pred_file_path", workflow_source)
        self.assertIn("from vlmeval.smp import file", result_source)
        self.assertIn("keep_default_na=False", result_source)
        self.assertIn("use_env_format=True", workflow_source)
        for forbidden in (
            "supported_VLM",
            "sys.modules",
            "monkeypatch",
            "run.py",
            ".__class__ =",
        ):
            self.assertNotIn(forbidden, workflow_source)
            self.assertNotIn(forbidden, native_source)
            self.assertNotIn(forbidden, result_source)


if __name__ == "__main__":
    unittest.main()
