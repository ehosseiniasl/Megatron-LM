from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from orchestration.configured_eval_run import EvalConfigError
from orchestration.eval_run_path_resolution import (
    prepare_vlmeval_run_directory,
    resolve_eval_run,
)
from orchestration.megatron_server_process_management import (
    split_megatron_server_step as server_step,
)
from orchestration.resolved_run_manifest import (
    validate_prepared_run_manifest,
    write_resolved_run_manifest,
)
from phase1_resolution_test_support import ResolutionFixture
from phase5_orchestration_test_support import (
    RecordingProcess,
    prepared_run_fixture,
    server_dependencies,
)

SplitMegatronServerStepError = server_step.SplitMegatronServerStepError
EnclosingAllocation = server_step.EnclosingAllocation
EnclosingAllocationError = server_step.EnclosingAllocationError
PreparedServerStep = server_step.PreparedServerStep
SplitServerStepControlError = server_step.SplitServerStepControlError
SplitServerStepWaitError = server_step.SplitServerStepWaitError
TerminalServerStep = server_step.TerminalServerStep
CONTROL_DIRECTORY_NAME = server_step.CONTROL_DIRECTORY_NAME
PREPARED_FILENAME = server_step.PREPARED_FILENAME
SHUTDOWN_REQUEST_FILENAME = server_step.SHUTDOWN_REQUEST_FILENAME
TERMINAL_FILENAME = server_step.TERMINAL_FILENAME
create_split_server_step_control_paths = server_step.create_split_server_step_control_paths
execute_split_megatron_server_step = server_step.execute_split_megatron_server_step
external_server_process_poll = server_step.external_server_process_poll
read_prepared_server_step = server_step.read_prepared_server_step
read_server_step_shutdown_request = server_step.read_server_step_shutdown_request
read_terminal_server_step = server_step.read_terminal_server_step
resolve_split_server_step_control_paths = server_step.resolve_split_server_step_control_paths
validate_enclosing_allocation = server_step.validate_enclosing_allocation
wait_for_prepared_server_step = server_step.wait_for_prepared_server_step
wait_for_server_step_shutdown_request = server_step.wait_for_server_step_shutdown_request
wait_for_terminal_server_step = server_step.wait_for_terminal_server_step
write_prepared_server_step = server_step.write_prepared_server_step
write_server_step_shutdown_request = server_step.write_server_step_shutdown_request
write_terminal_server_step = server_step.write_terminal_server_step


FINGERPRINT = "a" * 64


class SplitMegatronServerStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture, self.prepared = prepared_run_fixture()

    def tearDown(self) -> None:
        self.fixture.close()

    @property
    def run_directory(self) -> Path:
        return Path(self.prepared.resolved.paths.run_directory)

    def test_success_prepares_then_reaps_and_records_clean_terminal(self) -> None:
        events: list[str] = []
        process = RecordingProcess(events)
        result = execute_split_megatron_server_step(
            self.prepared,
            dependencies=server_dependencies(events, process),
        )
        self.assertEqual(
            events,
            ["allocation", "process", "start", "shutdown", "terminate"],
        )
        paths = resolve_split_server_step_control_paths(self.run_directory, "12345")
        prepared = read_prepared_server_step(paths)
        terminal = read_terminal_server_step(paths)
        self.assertEqual(terminal.status, "clean")
        self.assertEqual(terminal.process_returncode, -15)
        self.assertEqual(prepared.server_log_path, str(result.server_log_artifact))
        self.assertEqual(result.server_log_artifact.parent, paths.attempt_directory)
        manifest = json.loads(result.prepared_manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_status"], "prepared")
        self.assertEqual(manifest["slurm_job_id"], "12345")

    def test_early_process_exit_records_failed_terminal(self) -> None:
        events: list[str] = []
        process = RecordingProcess(events)

        def early_exit(_paths, _fingerprint, _poll):
            raise RuntimeError("process exited before shutdown")

        with self.assertRaisesRegex(SplitMegatronServerStepError, "server step failed"):
            execute_split_megatron_server_step(
                self.prepared,
                dependencies=server_dependencies(
                    events, process, shutdown_waiter=early_exit
                ),
            )
        paths = resolve_split_server_step_control_paths(self.run_directory, "12345")
        self.assertEqual(read_terminal_server_step(paths).status, "failed")
        self.assertEqual(events[-1], "terminate")

    def test_cleanup_error_records_failed_terminal(self) -> None:
        events: list[str] = []
        process = RecordingProcess(events, cleanup_error=True)
        with self.assertRaisesRegex(SplitMegatronServerStepError, "server step failed"):
            execute_split_megatron_server_step(
                self.prepared,
                dependencies=server_dependencies(events, process),
            )
        paths = resolve_split_server_step_control_paths(self.run_directory, "12345")
        self.assertEqual(read_terminal_server_step(paths).status, "failed")
        self.assertEqual(events[-1], "terminate")

    def test_allocation_failure_prevents_output_and_process_construction(self) -> None:
        events: list[str] = []
        process = RecordingProcess(events)
        deps = server_dependencies(events, process)

        def reject_allocation(_resolved):
            events.append("allocation")
            raise RuntimeError("synthetic allocation mismatch")

        deps = replace(deps, allocation_validator=reject_allocation)
        with self.assertRaisesRegex(RuntimeError, "synthetic allocation mismatch"):
            execute_split_megatron_server_step(self.prepared, dependencies=deps)
        self.assertEqual(events, ["allocation"])
        self.assertFalse(self.run_directory.exists())


class EnclosingAllocationValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ResolutionFixture()
        self.resolved = resolve_eval_run(
            self.fixture.config(),
            self.fixture.workspace,
            validate_destination=False,
        )
        self.environment = {
            "SLURM_JOB_ID": "12345",
            "SLURM_NNODES": "1",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NTASKS": "1",
            "SLURM_GPUS_ON_NODE": "8",
        }

    def tearDown(self) -> None:
        self.fixture.close()

    def test_exact_allocation_is_accepted_without_mutating_environment(self) -> None:
        original = self.environment.copy()
        self.assertEqual(
            validate_enclosing_allocation(self.resolved, self.environment),
            EnclosingAllocation("12345", 1, 1, 8),
        )
        self.assertEqual(self.environment, original)

    def test_missing_malformed_and_mismatched_fields_fail_closed(self) -> None:
        cases = (
            ("SLURM_JOB_ID", "123_4"),
            ("SLURM_NNODES", "2"),
            ("SLURM_JOB_NUM_NODES", "2"),
            ("SLURM_NTASKS", "8"),
            ("SLURM_GPUS_ON_NODE", "4"),
            ("SLURM_GPUS_ON_NODE", "gpu:8"),
        )
        for field, value in cases:
            environment = self.environment | {field: value}
            with self.subTest(field=field, value=value), self.assertRaises(
                EnclosingAllocationError
            ):
                validate_enclosing_allocation(self.resolved, environment)
        for field in self.environment:
            environment = self.environment.copy()
            del environment[field]
            with self.subTest(missing=field), self.assertRaises(EnclosingAllocationError):
                validate_enclosing_allocation(self.resolved, environment)

    def test_profile_shape_must_match_the_allocation(self) -> None:
        configured = replace(
            self.resolved.configured,
            run=replace(self.resolved.configured.run, gpus_per_node=4),
        )
        mismatched = replace(self.resolved, configured=configured)
        with self.assertRaises(EnclosingAllocationError):
            validate_enclosing_allocation(mismatched, self.environment)


class SplitPreparedRunManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ResolutionFixture()
        self.resolved = resolve_eval_run(
            self.fixture.config(),
            self.fixture.workspace,
            validate_destination=False,
        )
        prepare_vlmeval_run_directory(self.resolved)

    def tearDown(self) -> None:
        self.fixture.close()

    def test_accepts_exact_prepared_manifest_for_same_job(self) -> None:
        path = write_resolved_run_manifest(self.resolved, slurm_job_id="123")
        self.assertEqual(validate_prepared_run_manifest(self.resolved, "123"), path)

    def test_rejects_job_or_status_mismatch(self) -> None:
        write_resolved_run_manifest(self.resolved, slurm_job_id="123")
        with self.assertRaisesRegex(EvalConfigError, "different Slurm job"):
            validate_prepared_run_manifest(self.resolved, "124")
        write_resolved_run_manifest(
            self.resolved, run_status="partial", slurm_job_id="123"
        )
        with self.assertRaisesRegex(EvalConfigError, "prepared manifest"):
            validate_prepared_run_manifest(self.resolved, "123")

    def test_rejects_fingerprint_or_payload_mutation(self) -> None:
        path = write_resolved_run_manifest(self.resolved, slurm_job_id="123")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["resolved_paths"]["cache_root"] = "/different/cache"
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o640)
        with self.assertRaisesRegex(EvalConfigError, "fingerprint"):
            validate_prepared_run_manifest(self.resolved, "123")

    def test_rejects_invalid_job_type_and_private_manifest(self) -> None:
        path = write_resolved_run_manifest(self.resolved, slurm_job_id="123")
        with self.assertRaisesRegex(EvalConfigError, "decimal string"):
            validate_prepared_run_manifest(
                self.resolved, 123  # type: ignore[arg-type]
            )
        path.chmod(0o600)
        with self.assertRaisesRegex(EvalConfigError, "group-readable"):
            validate_prepared_run_manifest(self.resolved, "123")


class SplitServerStepControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.root.chmod(0o750)
        self.run_directory = self.root / "run"
        self.run_directory.mkdir(mode=0o750)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def prepared() -> PreparedServerStep:
        return PreparedServerStep(
            FINGERPRINT,
            "http://127.0.0.1:8000/v1/chat/completions",
            "http://127.0.0.1:8000/v1/health",
            "/shared/run/megatron_dynamic_server.log",
        )

    def test_creates_fresh_attempt_with_fixed_group_accessible_paths(self) -> None:
        paths = create_split_server_step_control_paths(self.run_directory, "123")
        self.assertEqual(
            paths.attempt_directory,
            self.run_directory / CONTROL_DIRECTORY_NAME / "123",
        )
        self.assertEqual(paths.prepared.name, PREPARED_FILENAME)
        self.assertEqual(paths.shutdown_request.name, SHUTDOWN_REQUEST_FILENAME)
        self.assertEqual(paths.terminal.name, TERMINAL_FILENAME)
        self.assertEqual(paths.attempt_directory.stat().st_mode & 0o777, 0o750)
        self.assertEqual(
            resolve_split_server_step_control_paths(self.run_directory, "123"),
            paths,
        )
        with self.assertRaisesRegex(SplitServerStepControlError, "already exists"):
            create_split_server_step_control_paths(self.run_directory, "123")

    def test_rejects_invalid_job_id_private_run_and_symlink_control_root(self) -> None:
        for job_id in ("", "0", "01", "-1", "abc"):
            with self.subTest(job_id=job_id):
                with self.assertRaisesRegex(SplitServerStepControlError, "job ID"):
                    create_split_server_step_control_paths(self.run_directory, job_id)

        self.run_directory.chmod(0o700)
        with self.assertRaisesRegex(SplitServerStepControlError, "shared"):
            create_split_server_step_control_paths(self.run_directory, "1")
        self.run_directory.chmod(0o750)

        outside = self.root / "outside"
        outside.mkdir(mode=0o750)
        control_root = self.run_directory / CONTROL_DIRECTORY_NAME
        control_root.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(SplitServerStepControlError, "real shared"):
            create_split_server_step_control_paths(self.run_directory, "2")

    def test_prepared_artifact_is_exact_frozen_and_write_once(self) -> None:
        paths = create_split_server_step_control_paths(self.run_directory, "10")
        record = self.prepared()
        artifact = write_prepared_server_step(paths, record)
        self.assertEqual(artifact.stat().st_mode & 0o777, 0o640)
        self.assertEqual(read_prepared_server_step(paths), record)
        with self.assertRaises(FrozenInstanceError):
            record.endpoint = "changed"  # type: ignore[misc]
        with self.assertRaisesRegex(SplitServerStepControlError, "immutable"):
            write_prepared_server_step(paths, record)

    def test_prepared_artifact_rejects_schema_type_and_path_forgery(self) -> None:
        cases = (
            {"schema_version": 2},
            {
                "schema_version": 1,
                "config_fingerprint": "bad",
                "endpoint": "endpoint",
                "health_endpoint": "health",
                "server_log_path": "log",
            },
        )
        for offset, payload in enumerate(cases, start=20):
            paths = create_split_server_step_control_paths(
                self.run_directory, str(offset)
            )
            paths.prepared.write_text(json.dumps(payload), encoding="utf-8")
            paths.prepared.chmod(0o640)
            with self.subTest(payload=payload):
                with self.assertRaises(SplitServerStepControlError):
                    read_prepared_server_step(paths)

        paths = create_split_server_step_control_paths(self.run_directory, "30")
        forged = replace(paths, prepared=self.root / PREPARED_FILENAME)
        with self.assertRaisesRegex(SplitServerStepControlError, "fixed filenames"):
            write_prepared_server_step(forged, self.prepared())

    def test_shutdown_request_is_fingerprint_bound_and_write_once(self) -> None:
        paths = create_split_server_step_control_paths(self.run_directory, "40")
        artifact = write_server_step_shutdown_request(paths, FINGERPRINT)
        self.assertEqual(artifact.stat().st_mode & 0o777, 0o640)
        self.assertEqual(read_server_step_shutdown_request(paths), FINGERPRINT)
        with self.assertRaisesRegex(SplitServerStepControlError, "immutable"):
            write_server_step_shutdown_request(paths, FINGERPRINT)
        with self.assertRaisesRegex(SplitServerStepControlError, "lowercase"):
            write_server_step_shutdown_request(
                create_split_server_step_control_paths(self.run_directory, "41"),
                "invalid",
            )

    def test_terminal_artifact_validates_status_returncode_and_write_once(self) -> None:
        for job_id, terminal in (
            ("50", TerminalServerStep(FINGERPRINT, "clean", 0)),
            ("51", TerminalServerStep(FINGERPRINT, "failed", -15)),
            ("52", TerminalServerStep(FINGERPRINT, "failed", None)),
        ):
            paths = create_split_server_step_control_paths(self.run_directory, job_id)
            artifact = write_terminal_server_step(paths, terminal)
            self.assertEqual(artifact.stat().st_mode & 0o777, 0o640)
            self.assertEqual(read_terminal_server_step(paths), terminal)
            with self.assertRaisesRegex(SplitServerStepControlError, "immutable"):
                write_terminal_server_step(paths, terminal)

        for job_id, terminal in (
            ("60", TerminalServerStep(FINGERPRINT, "unknown", 0)),
            ("61", TerminalServerStep(FINGERPRINT, "clean", True)),
        ):
            paths = create_split_server_step_control_paths(
                self.run_directory, job_id
            )
            with self.assertRaises(SplitServerStepControlError):
                write_terminal_server_step(paths, terminal)


class FakeClock:
    def __init__(self, callback=None) -> None:
        self.value = 0.0
        self.callback = callback
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds
        if self.callback is not None:
            self.callback(len(self.sleeps))


class SplitServerStepWaitingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_directory = Path(self.temporary.name).resolve() / "run"
        self.run_directory.parent.chmod(0o750)
        self.run_directory.mkdir(mode=0o750)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def prepared(fingerprint: str = FINGERPRINT) -> PreparedServerStep:
        return PreparedServerStep(fingerprint, "endpoint", "health", "/run/server.log")

    def test_prepared_wait_observes_late_artifact_and_exact_identity(self) -> None:
        holder = {}

        def create(_count: int) -> None:
            if holder:
                return
            paths = create_split_server_step_control_paths(self.run_directory, "100")
            write_prepared_server_step(paths, self.prepared())
            holder["paths"] = paths

        clock = FakeClock(create)
        paths, prepared = wait_for_prepared_server_step(
            self.run_directory,
            "100",
            FINGERPRINT,
            3,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
        )
        self.assertEqual(paths, holder["paths"])
        self.assertEqual(prepared, self.prepared())
        self.assertEqual(clock.sleeps, [1.0])

    def test_prepared_wait_times_out_and_rejects_identity_mismatch(self) -> None:
        clock = FakeClock()
        with self.assertRaisesRegex(SplitServerStepWaitError, "timed out"):
            wait_for_prepared_server_step(
                self.run_directory,
                "101",
                FINGERPRINT,
                2,
                monotonic=clock.monotonic,
                sleeper=clock.sleep,
            )
        paths = create_split_server_step_control_paths(self.run_directory, "102")
        write_prepared_server_step(paths, self.prepared("b" * 64))
        with self.assertRaisesRegex(SplitServerStepWaitError, "fingerprint mismatch"):
            wait_for_prepared_server_step(
                self.run_directory, "102", FINGERPRINT, 1
            )

    def test_shutdown_wait_handles_request_early_exit_and_mismatch(self) -> None:
        paths = create_split_server_step_control_paths(self.run_directory, "200")

        def request(_seconds: float) -> None:
            write_server_step_shutdown_request(paths, FINGERPRINT)

        wait_for_server_step_shutdown_request(
            paths, FINGERPRINT, lambda: None, poll_seconds=5, sleeper=request
        )

        exited = create_split_server_step_control_paths(self.run_directory, "201")
        with self.assertRaisesRegex(SplitServerStepWaitError, "exited.*7"):
            wait_for_server_step_shutdown_request(
                exited, FINGERPRINT, lambda: 7, sleeper=lambda _seconds: None
            )

        mismatch = create_split_server_step_control_paths(self.run_directory, "202")
        write_server_step_shutdown_request(mismatch, "b" * 64)
        with self.assertRaisesRegex(SplitServerStepWaitError, "fingerprint mismatch"):
            wait_for_server_step_shutdown_request(
                mismatch, FINGERPRINT, lambda: None
            )

    def test_terminal_wait_observes_cleanup_timeout_and_mismatch(self) -> None:
        paths = create_split_server_step_control_paths(self.run_directory, "300")

        def terminate(_count: int) -> None:
            if not paths.terminal.exists():
                write_terminal_server_step(
                    paths, TerminalServerStep(FINGERPRINT, "clean", -15)
                )

        clock = FakeClock(terminate)
        terminal = wait_for_terminal_server_step(
            paths,
            FINGERPRINT,
            2,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
        )
        self.assertEqual(terminal.status, "clean")

        absent = create_split_server_step_control_paths(self.run_directory, "301")
        clock = FakeClock()
        with self.assertRaisesRegex(SplitServerStepWaitError, "timed out"):
            wait_for_terminal_server_step(
                absent,
                FINGERPRINT,
                1,
                monotonic=clock.monotonic,
                sleeper=clock.sleep,
            )

        mismatch = create_split_server_step_control_paths(self.run_directory, "302")
        write_terminal_server_step(
            mismatch, TerminalServerStep("b" * 64, "failed", 1)
        )
        with self.assertRaisesRegex(SplitServerStepWaitError, "fingerprint mismatch"):
            wait_for_terminal_server_step(mismatch, FINGERPRINT, 1)

    def test_external_poll_adapts_terminal_status(self) -> None:
        absent = create_split_server_step_control_paths(self.run_directory, "400")
        self.assertIsNone(external_server_process_poll(absent, FINGERPRINT))
        failed = create_split_server_step_control_paths(self.run_directory, "401")
        write_terminal_server_step(
            failed, TerminalServerStep(FINGERPRINT, "failed", None)
        )
        self.assertEqual(external_server_process_poll(failed, FINGERPRINT), 1)
        clean = create_split_server_step_control_paths(self.run_directory, "402")
        write_terminal_server_step(
            clean, TerminalServerStep(FINGERPRINT, "clean", -15)
        )
        self.assertEqual(external_server_process_poll(clean, FINGERPRINT), -15)

    def test_rejects_invalid_wait_intervals(self) -> None:
        paths = create_split_server_step_control_paths(self.run_directory, "500")
        for timeout, poll in ((0, 1), (1, 0), (1, 2), (float("nan"), 1)):
            with self.subTest(timeout=timeout, poll=poll):
                with self.assertRaises(SplitServerStepWaitError):
                    wait_for_terminal_server_step(
                        paths, FINGERPRINT, timeout, poll_seconds=poll
                    )


if __name__ == "__main__":
    unittest.main()
