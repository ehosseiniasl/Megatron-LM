"""Focused tests for secure judge credential loading."""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from orchestration.judge_runtime import (
    JUDGE_CREDENTIALS_FILE_ENV,
    JudgeRuntimeError,
    evaluate_with_judge_model,
    judge_artifact_label,
    normalize_openai_api_base,
    prepare_judge_runtime,
)


class _FakeOpenAIWrapper:
    def __init__(self, model: str, *args: object, **kwargs: object) -> None:
        self.model = model

    def working(self) -> bool:
        return True

    def generate(self) -> str:
        return self.model


class JudgeRuntimeTest(unittest.TestCase):
    def _credential_file(self, root: Path, mode: int = 0o600) -> Path:
        path = root / "judge.json"
        path.write_text(
            json.dumps(
                {
                    "OPENAI_API_BASE": "https://judge.example/v1/chat/completions",
                    "OPENAI_API_KEY": "unit-secret",
                }
            ),
            encoding="utf-8",
        )
        path.chmod(mode)
        return path

    def test_exact_matching_needs_no_home_or_credentials(self) -> None:
        self.assertIsNone(prepare_judge_runtime("exact_matching", {}))

    def test_secure_file_loads_without_returning_the_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._credential_file(Path(temporary))
            environment = {"HOME": temporary, JUDGE_CREDENTIALS_FILE_ENV: str(path)}
            self.assertEqual(prepare_judge_runtime("gpt-4o-mini", environment), path)
            self.assertEqual(environment["OPENAI_API_KEY"], "unit-secret")
            self.assertEqual(
                environment["OPENAI_API_BASE"],
                "https://judge.example/v1/chat/completions",
            )

    def test_direct_runtime_injection_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                "HOME": temporary,
                "OPENAI_API_KEY": "direct-secret",
                "OPENAI_API_BASE": "https://judge.example/v1/chat/completions",
            }
            self.assertIsNone(prepare_judge_runtime("gpt-4o-mini", environment))
            self.assertEqual(
                logging.getLogger("vlmeval.api.gpt").level, logging.WARNING
            )

    def test_insecure_file_and_conflicting_injection_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._credential_file(root, 0o640)
            environment = {"HOME": temporary, JUDGE_CREDENTIALS_FILE_ENV: str(path)}
            with self.assertRaisesRegex(JudgeRuntimeError, "mode must be 0600"):
                prepare_judge_runtime("gpt-4o-mini", environment)
            path.chmod(0o600)
            environment["OPENAI_API_KEY"] = "different-secret"
            with self.assertRaisesRegex(JudgeRuntimeError, "conflicts"):
                prepare_judge_runtime("gpt-4o-mini", environment)

    def test_endpoint_normalization_accepts_sdk_base_or_direct_url(self) -> None:
        self.assertEqual(
            normalize_openai_api_base("https://judge.example/v1/chat/completions"),
            "https://judge.example/v1/chat/completions",
        )
        self.assertEqual(
            normalize_openai_api_base("https://judge.example/v1/"),
            "https://judge.example/v1/chat/completions",
        )
        for endpoint in (
            "judge.example/v1/chat/completions",
            "https://user:secret@judge.example/v1/chat/completions",
            "https://judge.example/v1/chat/completions?key=secret",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(JudgeRuntimeError):
                    normalize_openai_api_base(endpoint)

    def test_provider_model_id_maps_to_readable_artifact_label(self) -> None:
        self.assertEqual(
            judge_artifact_label("us/azure/openai/eccn-gpt-4.1-mini"),
            "us_azure_openai_eccn-gpt-4_1-mini",
        )
        self.assertEqual(judge_artifact_label("gpt-4o-mini"), "gpt-4o-mini")

    def test_unsafe_provider_model_is_sent_exactly_but_scored_under_label(self) -> None:
        observed: dict[str, str] = {}
        vlmeval = types.ModuleType("vlmeval")
        vlmeval_api = types.ModuleType("vlmeval.api")
        vlmeval_api.OpenAIWrapper = _FakeOpenAIWrapper
        vlmeval.api = vlmeval_api

        def evaluator(prediction: str, *, model: str) -> None:
            from vlmeval.api import OpenAIWrapper

            wrapper = OpenAIWrapper(model)
            restored = pickle.loads(pickle.dumps(wrapper))
            observed.update(
                prediction=prediction,
                artifact_model=restored.model,
                request_model=restored.generate(),
            )

        provider_model = "us/azure/openai/eccn-gpt-4.1-mini"
        with mock.patch.dict(
            sys.modules, {"vlmeval": vlmeval, "vlmeval.api": vlmeval_api}
        ):
            label = evaluate_with_judge_model(
                evaluator, "/run/predictions.xlsx", provider_model
            )
            self.assertIs(vlmeval_api.OpenAIWrapper, _FakeOpenAIWrapper)

        self.assertEqual(label, "us_azure_openai_eccn-gpt-4_1-mini")
        self.assertEqual(
            observed,
            {
                "prediction": "/run/predictions.xlsx",
                "artifact_model": label,
                "request_model": provider_model,
            },
        )

    def test_worldvqa_partial_cache_resume_aligns_tasks_to_missing_keys(self) -> None:
        observed: dict[str, object] = {}
        worldvqa_module = types.ModuleType("vlmeval.dataset.worldvqa")

        def original_progress(
            func: object,
            tasks: object,
            *,
            nproc: int | None,
            save: str | None,
            keys: object,
            use_process: bool,
            **kwargs: object,
        ) -> list[object]:
            observed.update(tasks=list(tasks), keys=list(keys), save=save)
            return []

        worldvqa_module.utils = types.SimpleNamespace(
            track_progress_rich=original_progress
        )

        def evaluate(_self: object, _prediction: str, *, model: str) -> object:
            self.assertEqual(model, "gpt-4o-mini")
            return worldvqa_module.utils.track_progress_rich(
                object(),
                ["row-0", "row-1", "row-2"],
                nproc=4,
                save="partial.pkl",
                keys=[1, 2],
            )

        worldvqa_type = type(
            "WorldVQA",
            (),
            {
                "__module__": "vlmeval.dataset.worldvqa",
                "evaluate": evaluate,
            },
        )
        vlmeval = types.ModuleType("vlmeval")
        vlmeval_dataset = types.ModuleType("vlmeval.dataset")
        vlmeval_dataset.worldvqa = worldvqa_module

        with mock.patch.dict(
            sys.modules,
            {
                "vlmeval": vlmeval,
                "vlmeval.dataset": vlmeval_dataset,
                "vlmeval.dataset.worldvqa": worldvqa_module,
            },
        ):
            evaluate_with_judge_model(
                worldvqa_type().evaluate,
                "/run/predictions.xlsx",
                "gpt-4o-mini",
                dataset="WorldVQA",
            )

        self.assertEqual(
            observed,
            {"tasks": ["row-1", "row-2"], "keys": [1, 2], "save": "partial.pkl"},
        )
        self.assertIs(worldvqa_module.utils.track_progress_rich, original_progress)


if __name__ == "__main__":
    unittest.main()
