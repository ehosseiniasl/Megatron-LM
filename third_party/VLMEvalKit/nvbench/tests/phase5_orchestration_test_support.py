"""Small dependency fakes for split server/client orchestration tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from orchestration.fewshot_dataset_adaptation import (
    demonstration_selection as row_partition_artifact,
)
from orchestration.fewshot_dataset_adaptation.demonstration_selection import (
    FewshotRowSelection,
)
from orchestration.eval_run_path_resolution import (
    PreparedMegatronDynamicServerVlmFewshotRun,
    prepare_vlmeval_run_directory,
    resolve_eval_run,
)
from orchestration.megatron_server_process_management import (
    megatron_dynamic_server_launch as server_launch,
    split_megatron_server_step as server_step,
)
from orchestration.megatron_dynamic_server_vlm_fewshot_run import (
    MegatronDynamicServerVlmFewshotDependencies,
)
from orchestration.resolved_run_manifest import (
    canonical_config_fingerprint,
    write_resolved_run_manifest,
)
from orchestration.vlmeval_generation_workflow import (
    VLMEvalGenerationResult,
)

PreparedServerStep = server_step.PreparedServerStep
EnclosingAllocation = server_step.EnclosingAllocation
SplitMegatronServerStepDependencies = server_step.SplitMegatronServerStepDependencies
TerminalServerStep = server_step.TerminalServerStep
build_megatron_dynamic_server_launch = server_launch.build_megatron_dynamic_server_launch
create_split_server_step_control_paths = server_step.create_split_server_step_control_paths
write_prepared_server_step = server_step.write_prepared_server_step
from phase1_resolution_test_support import ResolutionFixture


class RecordingProcess:
    def __init__(self, events: list[str], *, cleanup_error: bool = False) -> None:
        self.events = events
        self.cleanup_error = cleanup_error

    def start(self) -> None:
        self.events.append("start")

    def poll(self) -> None:
        self.events.append("poll")
        return None

    def forward_signal(self, signum: int) -> None:
        self.events.append(f"signal:{signum}")

    def terminate(self) -> int:
        self.events.append("terminate")
        if self.cleanup_error:
            raise RuntimeError("synthetic cleanup failure")
        return -15


def prepared_run_fixture(
) -> tuple[ResolutionFixture, PreparedMegatronDynamicServerVlmFewshotRun]:
    fixture = ResolutionFixture()
    try:
        resolved = resolve_eval_run(
            fixture.config(),
            fixture.workspace,
            validate_destination=False,
        )
        compiler_cache = fixture.workspace / "compiler-cache"
        compiler_cache.mkdir(mode=0o700)
        launch = build_megatron_dynamic_server_launch(
            resolved,
            resolved.paths.run_directory,
            host="127.0.0.1",
            port=8000,
            coordinator_port=8001,
            python_executable=str(Path(sys.executable).resolve()),
            compiler_cache_directory=compiler_cache,
        )
    except Exception:
        fixture.close()
        raise
    return fixture, PreparedMegatronDynamicServerVlmFewshotRun(resolved, launch)


def prepare_client_side_state(
    prepared: PreparedMegatronDynamicServerVlmFewshotRun,
    job_id: str = "12345",
) -> None:
    resolved = prepared.resolved
    directory = prepare_vlmeval_run_directory(resolved)
    write_resolved_run_manifest(resolved, slurm_job_id=job_id)
    paths = create_split_server_step_control_paths(directory.path, job_id)
    write_prepared_server_step(
        paths,
        PreparedServerStep(
            canonical_config_fingerprint(resolved),
            prepared.launch.endpoint,
            prepared.launch.health_endpoint,
            str(paths.attempt_directory / "megatron_dynamic_server.log"),
        ),
    )


def shared_file(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    path.chmod(0o640)
    return path


def row_partitions(
    root: Path,
) -> row_partition_artifact.DemonstrationAndScoredRowPartitionsArtifact:
    artifact = shared_file(
        root / "demonstration_and_scored_row_partitions.json", b"partitions"
    )
    return row_partition_artifact.DemonstrationAndScoredRowPartitionsArtifact(
        artifact
    )


def completed_result(root: Path) -> VLMEvalGenerationResult:
    selection = FewshotRowSelection((), (), (0,), (17,))
    partitions = row_partitions(root)
    prediction = shared_file(root / "prediction.xlsx", b"prediction")
    score = shared_file(root / "prediction_score.json", b"score")
    metrics = shared_file(root / "vlmeval_metrics.v1.json", b"metrics")
    status_payload = {
        "schema_version": 3,
        "datasets": [
            {
                "dataset": "OCRBench",
                "status": "completed",
                "expected_scored_indices": [17],
                "observed_prediction_count": 1,
                "demonstration_and_scored_row_partitions_path": partitions.artifact.name,
                "prediction_path": prediction.name,
                "score_path": score.name,
                "metrics_path": metrics.name,
                "failure": None,
            }
        ],
    }
    status = shared_file(
        root / "vlmeval_dataset_status.json",
        json.dumps(status_payload).encode("utf-8"),
    )
    return VLMEvalGenerationResult(
        selection,
        partitions.artifact,
        prediction,
        score,
        metrics,
        status,
        (),
    )


def server_dependencies(
    events: list[str],
    process: RecordingProcess,
    *,
    shutdown_waiter=None,
) -> SplitMegatronServerStepDependencies:
    def allocation(_resolved):
        events.append("allocation")
        return EnclosingAllocation("12345", 1, 1, 8)

    def process_factory(_launch):
        events.append("process")
        return process

    def shutdown(*args):
        events.append("shutdown")
        if shutdown_waiter is not None:
            return shutdown_waiter(*args)
        return None

    return SplitMegatronServerStepDependencies(
        allocation, process_factory, shutdown
    )


def client_dependencies(
    events: list[str],
    generation_runner,
    *,
    terminal_status: str = "clean",
    suite_generation_runner=None,
    health_waiter=None,
) -> MegatronDynamicServerVlmFewshotDependencies:
    def prepared_waiter(*args):
        events.append("prepared")
        return server_step.wait_for_prepared_server_step(*args)

    def manifest(resolved, job_id):
        events.append("manifest")
        from orchestration.resolved_run_manifest import (
            validate_prepared_run_manifest,
        )

        return validate_prepared_run_manifest(resolved, job_id)

    def health(*args, **kwargs):
        events.append("health")
        if health_waiter is not None:
            return health_waiter(*args, **kwargs)
        return None

    def client(*_args, **_kwargs):
        events.append("client")
        return object()

    def preflight(_client, image):
        assert image.is_file()
        events.append("preflight")

    def generation(*args, **kwargs):
        events.append("generation")
        return generation_runner(*args, **kwargs)

    def suite_generation(*args, **kwargs):
        if suite_generation_runner is None:
            raise AssertionError("single-benchmark test invoked suite generation")
        events.append("suite_generation")
        return suite_generation_runner(*args, **kwargs)

    def shutdown(paths, fingerprint):
        events.append("shutdown")
        return server_step.write_server_step_shutdown_request(paths, fingerprint)

    def terminal(paths, fingerprint, _timeout):
        events.append("terminal")
        return TerminalServerStep(fingerprint, terminal_status, -15)

    return MegatronDynamicServerVlmFewshotDependencies(
        prepared_waiter,
        manifest,
        health,
        client,
        preflight,
        generation,
        generation,
        suite_generation,
        shutdown,
        terminal,
    )
