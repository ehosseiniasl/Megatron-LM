"""Small in-memory test doubles for the Phase 3 generation workflow."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from orchestration.configured_eval_run import (
    BenchmarkConfig,
    GenerationConfig,
)
from orchestration.fewshot_dataset_adaptation import (
    fewshot_conversation_builder as dataset_proxy,
)
from orchestration.fewshot_dataset_adaptation import (
    demonstration_selection as partition_artifact,
)
from orchestration.vlmeval_generation_workflow import (
    VLMEvalGenerationResult,
    run_vlmeval_generation,
)


PARTITIONS_FILENAME = partition_artifact.PARTITIONS_FILENAME
VLMEvalFewshotDatasetProxy = dataset_proxy.VLMEvalFewshotDatasetProxy


class FakeRow(dict[str, object]):
    def to_dict(self) -> dict[str, object]:
        return dict(self)


class FakeILoc:
    def __init__(self, frame: "FakeFrame") -> None:
        self.frame = frame

    def __getitem__(self, key: object) -> object:
        if type(key) is int:
            return FakeRow(copy.deepcopy(self.frame.rows[key]))
        if isinstance(key, list) and all(type(value) is int for value in key):
            return FakeFrame([self.frame.rows[value] for value in key])
        raise TypeError("unsupported fake iloc key")


class FakeFrame:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = copy.deepcopy(rows)
        self.iloc = FakeILoc(self)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: str) -> list[object]:
        return [row[key] for row in self.rows]

    def __setitem__(self, key: str, values: list[object]) -> None:
        for row, value in zip(self.rows, values, strict=True):
            row[key] = value

    def copy(self, deep: bool = True) -> "FakeFrame":
        if not deep:
            raise AssertionError("workflow proxy must request a deep copy")
        return FakeFrame(self.rows)


class OCRBench:
    def __init__(self, dataset_name: str = "OCRBench") -> None:
        self.dataset_name = dataset_name
        self.data = FakeFrame(
            [
                {
                    "index": f"row-{position}",
                    "question": f"question {position}",
                    "answer": f"answer {position}",
                    "image_path": f"image-{position}.png",
                }
                for position in range(3)
            ]
        )
        self.evaluate_calls = 0
        self.evaluate_arguments: list[str] = []
        self.evaluate_keywords: list[dict[str, object]] = []

    def build_prompt(self, record: object) -> list[dict[str, str]]:
        if not isinstance(record, dict):
            record = record.to_dict()  # type: ignore[union-attr]
        return [
            {"type": "image", "value": str(record["image_path"])},
            {"type": "text", "value": str(record["question"])},
        ]

    def evaluate(self, eval_file: object, **judge_kwargs: object) -> object:
        self.evaluate_calls += 1
        self.evaluate_arguments.append(str(eval_file))
        self.evaluate_keywords.append(judge_kwargs)
        prediction = Path(str(eval_file))
        score = prediction.with_name(f"{prediction.stem}_score.json")
        score.write_text(json.dumps(valid_score_payload()) + "\n", encoding="utf-8")
        score.chmod(0o640)
        return {"ignored": True}


UnitDataset = OCRBench


class UnitClient:
    is_api = True

    def generate(self, message: object, dataset: str | None = None) -> str:
        return f"unused: {message!r} {dataset!r}"


def benchmark(
    *,
    upstream_class: str = "OCRBench",
    dataset: str = "OCRBench",
) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="unit_1shot",
        dataset=dataset,
        upstream_class=upstream_class,
        modality="multimodal",
        num_shot=1,
        demonstration_type="benchmark",
        demonstration_answer_format="raw",
        include_demonstration_images=True,
        prepend_reasoning_cue=False,
        scored_limit=2,
        result_reader="ocrbench_score_json",
        expected_result_suffix="_score.json",
        result_metric_unit="reader_defined",
        judge="exact_matching",
        resource="n1",
        generation=GenerationConfig(
            temperature=0.0,
            retry=1,
            image_size=-1,
            image_detail="low",
            max_new_tokens=8,
            stop=("\n\n",),
            enable_thinking=False,
            chat_template="minimal",
        ),
    )


class RecordingPathResolver:
    def __init__(self, artifact_name: str = "prediction.json") -> None:
        self.artifact_name = artifact_name
        self.calls: list[tuple[str, str, str, bool]] = []
        self.row_partitions_existed: list[bool] = []

    def __call__(
        self,
        work_dir: str,
        model_name: str,
        dataset_name: str,
        *,
        use_env_format: bool,
    ) -> str:
        self.calls.append((work_dir, model_name, dataset_name, use_env_format))
        self.row_partitions_existed.append(
            (Path(work_dir) / PARTITIONS_FILENAME).is_file()
        )
        return str(Path(work_dir) / self.artifact_name)


class RecordingRunner:
    def __init__(self, artifact_name: str = "prediction.json") -> None:
        self.artifact_name = artifact_name
        self.calls: list[dict[str, object]] = []
        self.row_partitions_existed: list[bool] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        self.row_partitions_existed.append(
            (Path(str(kwargs["work_dir"])) / PARTITIONS_FILENAME).is_file()
        )
        proxy = kwargs["dataset"]
        if not isinstance(proxy, VLMEvalFewshotDatasetProxy):
            raise AssertionError("runner did not receive the Phase 2 proxy")
        proxy.data["prediction"] = ["one", "two"]
        artifact = Path(str(kwargs["work_dir"])) / self.artifact_name
        artifact.write_bytes(b'{"opaque":true}\n')
        artifact.chmod(0o640)
        return kwargs["model"]


def valid_score_payload() -> dict[str, int | float]:
    return {
        "Text Recognition": 10,
        "Scene Text-centric VQA": 2,
        "Doc-oriented VQA": 3,
        "Key Information Extraction": 4,
        "Handwritten Mathematical Expression Recognition": 1,
        "Final Score": 20,
        "Final Score Norm": 2.0,
    }


def load_unit_predictions(path: str) -> object:
    if not Path(path).is_file():
        raise AssertionError("prediction loader received a missing artifact")
    return [
        {"index": "row-1", "prediction": "one"},
        {"index": "row-2", "prediction": "two"},
    ]


def run_unit_workflow(
    default_work_dir: str | os.PathLike[str], **changes: object
) -> VLMEvalGenerationResult:
    arguments: dict[str, object] = {
        "client": UnitClient(),
        "benchmark": benchmark(),
        "work_dir": default_work_dir,
        "model_name": "unit-model",
        "api_nproc": 1,
        "dataset_factory": lambda name: UnitDataset(name),
        "inference_runner": RecordingRunner(),
        "prediction_path_resolver": RecordingPathResolver(),
        "prediction_loader": load_unit_predictions,
    }
    arguments.update(changes)
    return run_vlmeval_generation(**arguments)  # type: ignore[arg-type]
