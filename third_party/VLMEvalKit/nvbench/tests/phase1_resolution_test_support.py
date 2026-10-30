from __future__ import annotations

import tempfile
from pathlib import Path

from orchestration.configured_eval_run import (
    BenchmarkConfig,
    CheckpointConfig,
    ClusterConfig,
    ConfiguredEvalRun,
    ConfiguredPaths,
    GenerationConfig,
    JudgeScoringResourceConfig,
    ModelConfig,
    OutputConfig,
    ResourceConfig,
    RunConfig,
    ServerConfig,
    TokenizerConfig,
)


class ResolutionFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name).resolve()
        self.workspace.chmod(0o750)
        self.runtime = self.workspace / "runtime"
        self.runtime.mkdir()
        (self.runtime / "tools").mkdir()
        assets = self.runtime / "third_party/VLMEvalKit/nvbench/assets"
        assets.mkdir(parents=True)
        self.entrypoint = self.runtime / "tools/run_dynamic_text_generation_server.py"
        self.template = assets / "minimal_chat_template.jinja"
        self.entrypoint.write_text("print('server')\n", encoding="utf-8")
        self.template.write_text("{{ messages }}\n", encoding="utf-8")
        self.checkpoint = self.workspace / "checkpoints"
        self.iteration = self.checkpoint / "iter_0049250"
        self.iteration.mkdir(parents=True)
        self.tracker = self.checkpoint / "latest_checkpointed_iteration.txt"
        self.metadata = self.iteration / ".metadata"
        self.tracker.write_text("49250", encoding="utf-8")
        self.metadata.write_bytes(b"checkpoint-metadata")
        self.cache = self.workspace / "cache"
        self.cache.mkdir()
        self.output = self.workspace / "outputs"

    def close(self) -> None:
        self.temporary.cleanup()

    def config(self, *, resume: bool = False) -> ConfiguredEvalRun:
        tokenizer = TokenizerConfig(
            repository="nvidia/n4-v6-262k",
            revision="8f5dd326777cbdd5c018d92ba35f473d971e121d",
        )
        server = ServerConfig(
            context_length=8192,
            health_timeout_seconds=60,
            request_timeout_seconds=30,
            cli_overrides=(
                ("tokenizer-prompt-format", "mistral"),
            ),
        )
        generation = GenerationConfig(
            0.0,
            10,
            -1,
            "low",
            64,
            ("\n\n",),
            False,
            "minimal",
        )
        resources = (
            ResourceConfig(
                "n1", 1, 8, 8, 8, 56, "800G", 8, "64G", "900G", "01:00:00", 2
            ),
            ResourceConfig(
                "n2", 2, 8, 8, 8, 56, "800G", 8, "64G", "900G", "01:00:00", 1
            ),
            ResourceConfig(
                "n4", 4, 8, 8, 32, 56, "800G", 8, "64G", "900G", "01:00:00", 1
            ),
            ResourceConfig(
                "n8", 8, 8, 8, 64, 56, "800G", 8, "64G", "900G", "01:00:00", 1
            ),
        )
        return ConfiguredEvalRun(
            run=RunConfig("fixture-run", "local_torchrun", 1, 8, 8, 8, "n1"),
            cluster=ClusterConfig(
                "fixture-cluster",
                str(self.workspace / "cluster.yaml"),
                "fixture-account",
                "fixture-partition",
                resources,
                JudgeScoringResourceConfig("cpu", 8, "32G", "01:00:00", 2),
            ),
            model=ModelConfig(
                "fixture-model",
                tokenizer,
                server,
            ),
            benchmark=BenchmarkConfig(
                "ocr-smoke",
                "OCRBench",
                "OCRBench",
                "multimodal",
                2,
                "benchmark",
                "raw",
                True,
                False,
                10,
                "ocrbench_score_json",
                "_score.json",
                "reader_defined",
                "exact_matching",
                "n1",
                generation,
            ),
            checkpoint=CheckpointConfig(
                "${workspace}/checkpoints", 49250
            ),
            paths=ConfiguredPaths("${workspace}/cache"),
            output=OutputConfig("${workspace}/outputs", resume),
        )
