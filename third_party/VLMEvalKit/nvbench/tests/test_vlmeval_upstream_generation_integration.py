from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from orchestration.vlmeval_generation_workflow import (
    run_vlmeval_generation,
)
from phase3_generation_test_support import benchmark, valid_score_payload


def _missing_upstream_dependencies() -> tuple[str, ...]:
    required = ("numpy", "pandas", "torch", "tqdm", "validators", "PIL")
    return tuple(name for name in required if importlib.util.find_spec(name) is None)


class VLMEvalGenerationUpstreamIntegrationTest(unittest.TestCase):
    @unittest.skipIf(
        bool(_missing_upstream_dependencies()),
        "missing upstream dependencies: " + ", ".join(_missing_upstream_dependencies()),
    )
    def test_real_client_and_default_upstream_inference_produce_artifact(self) -> None:
        import pandas as pd
        from PIL import Image

        from fake_dynamic_server import (
            FakeDynamicServer,
            QueuedResponse,
            chat_completion_response,
        )
        from orchestration.dynamic_server_http_client import (
            megatron_dynamic_server_chat_client as chat_client,
        )
        from vlmeval.smp import load

        MegatronDynamicServerChatClient = chat_client.MegatronDynamicServerChatClient

        class OCRBench:
            def __init__(self, rows: list[dict[str, object]]) -> None:
                self.dataset_name = "OCRBench"
                self.data = pd.DataFrame(rows)
                self.evaluate_calls = 0
                self.evaluate_arguments: list[str] = []
                self.evaluate_judges: list[str] = []
                self.prediction_bytes_before_evaluation: bytes | None = None

            def build_prompt(self, record: object) -> list[dict[str, str]]:
                return [
                    {"type": "image", "value": str(record["image_path"])},
                    {"type": "text", "value": str(record["question"])},
                ]

            def evaluate(self, eval_file: object, **judge_kwargs: object) -> object:
                self.evaluate_calls += 1
                prediction = Path(str(eval_file))
                self.evaluate_arguments.append(str(prediction))
                if set(judge_kwargs) != {"model"}:
                    raise ValueError("the configured evaluator judge is required")
                self.evaluate_judges.append(str(judge_kwargs["model"]))
                self.prediction_bytes_before_evaluation = prediction.read_bytes()
                score = prediction.with_name(f"{prediction.stem}_score.json")
                score.write_text(
                    json.dumps(valid_score_payload()) + "\n",
                    encoding="utf-8",
                )
                score.chmod(0o640)
                return {"ignored": True}

        contract = benchmark()
        prediction_format = os.environ.get("PRED_FORMAT")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_paths = []
            for position in range(2):
                image_path = root / f"image-{position}.png"
                Image.new("RGB", (3, 2), (position * 40, 10, 20)).save(image_path)
                image_paths.append(image_path)
            dataset = OCRBench(
                [
                    {
                        "index": f"row-{position}",
                        "question": f"question {position}",
                        "answer": f"answer {position}",
                        "image_path": str(image_path),
                    }
                    for position, image_path in enumerate(image_paths)
                ]
            )
            original = dataset.data.copy(deep=True)
            factory_calls: list[str] = []

            def factory(dataset_name: str) -> object:
                factory_calls.append(dataset_name)
                return dataset

            with FakeDynamicServer(
                [QueuedResponse(body=chat_completion_response("server answer"))]
            ) as server:
                client = MegatronDynamicServerChatClient(
                    server.endpoint,
                    "synthetic-model",
                    contract.generation,
                    3.0,
                )
                result = run_vlmeval_generation(
                    client,
                    contract,
                    root,
                    "synthetic-client",
                    api_nproc=1,
                    dataset_factory=factory,
                )
                artifact = load(str(result.prediction_artifact))
                self.assertEqual(list(artifact["index"]), ["row-1"])
                self.assertEqual(list(artifact["prediction"]), ["server answer"])
                self.assertEqual(
                    result.prediction_artifact.read_bytes(),
                    dataset.prediction_bytes_before_evaluation,
                )
                self.assertTrue(result.score_artifact.is_file())
                self.assertTrue(result.metrics_artifact.is_file())
                self.assertTrue(result.status_artifact.is_file())
                self.assertEqual(len(result.normalized_metrics), 7)
                status = json.loads(result.status_artifact.read_text(encoding="utf-8"))
                self.assertEqual(status["datasets"][0]["status"], "completed")
                self.assertEqual(len(server.requests), 1)
                self.assertFalse(list(root.glob("*.pkl")))

            self.assertFalse(server.thread_alive)
            self.assertEqual(factory_calls, ["OCRBench"])
            pd.testing.assert_frame_equal(dataset.data, original)
            self.assertEqual(dataset.evaluate_calls, 1)
            self.assertEqual(
                dataset.evaluate_arguments,
                [str(result.prediction_artifact)],
            )
            self.assertEqual(dataset.evaluate_judges, ["exact_matching"])
            self.assertEqual(os.environ.get("PRED_FORMAT"), prediction_format)


if __name__ == "__main__":
    unittest.main()
