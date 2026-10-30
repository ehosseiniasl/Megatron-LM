from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from orchestration.configured_eval_run import EvalConfigError
from orchestration.eval_run_path_resolution import resolve_eval_run
from orchestration.resolved_run_manifest import (
    DatasetTerminalStatus,
    MANIFEST_FILENAME,
    build_resolved_run_manifest,
    canonical_config_fingerprint,
    stable_manifest_payload,
    validate_dataset_terminal_statuses,
    validate_output_destination,
    write_resolved_run_manifest,
)

from phase1_resolution_test_support import ResolutionFixture


class ResolvedRunManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ResolutionFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def _resolved(self, resume: bool = False):
        return resolve_eval_run(
            self.fixture.config(resume=resume),
            self.fixture.workspace,
            validate_destination=False,
        )

    @staticmethod
    def _make_run_directory(resolved) -> Path:
        run_directory = Path(resolved.paths.run_directory)
        run_directory.mkdir(parents=True)
        return run_directory

    @staticmethod
    def _terminal(
        dataset: str = "OCRBench",
        status: str = "failed",
        path: str = "vlmeval_dataset_status.json",
    ) -> DatasetTerminalStatus:
        return DatasetTerminalStatus(dataset, status, path)

    def test_stable_payload_preserves_configured_and_resolved_overrides(self) -> None:
        resolved = self._resolved()
        payload = stable_manifest_payload(resolved)
        server = payload["configured_eval_run"]["model"]["server"]
        self.assertIn("cli_overrides", server)
        self.assertEqual(server["cli_overrides"], (("tokenizer-prompt-format", "mistral"),))
        self.assertIn("resolved_cli_overrides", payload)
        self.assertNotEqual(server["cli_overrides"], ())
        self.assertEqual(payload["schema_version"], 2)
        self.assertNotIn("runtime_revision", payload)
        self.assertNotIn("nvbench_source_provenance", payload)
        self.assertNotIn("execution_environment_provenance", payload)
        self.assertNotIn("generated_at", payload)
        self.assertNotIn("slurm_job_id", payload)

    def test_volatile_metadata_does_not_change_config_fingerprint(self) -> None:
        resolved = self._resolved()
        first = build_resolved_run_manifest(
            resolved, generated_at="2026-01-01T00:00:00Z", slurm_job_id="1"
        )
        second = build_resolved_run_manifest(
            resolved, generated_at="2027-01-01T00:00:00Z", slurm_job_id="2"
        )
        self.assertEqual(first["config_fingerprint"], second["config_fingerprint"])
        self.assertEqual(first["config_fingerprint"], canonical_config_fingerprint(resolved))

    def test_demonstration_contract_changes_config_fingerprint(self) -> None:
        resolved = self._resolved()
        benchmark = resolved.configured.benchmark
        assert benchmark is not None
        baseline = canonical_config_fingerprint(resolved)
        variants = (
            replace(benchmark, num_shot=4),
            replace(benchmark, demonstration_type="trivial"),
            replace(benchmark, include_demonstration_images=False),
            replace(benchmark, demonstration_indices=("row-1",)),
            replace(benchmark, prompt_adapter="cvbench_explicit_options"),
        )
        for variant in variants:
            changed = replace(
                resolved,
                configured=replace(resolved.configured, benchmark=variant),
            )
            self.assertNotEqual(baseline, canonical_config_fingerprint(changed))

    def test_only_active_prompt_adapter_is_recorded(self) -> None:
        resolved = self._resolved()
        benchmark = resolved.configured.benchmark
        assert benchmark is not None
        baseline_payload = stable_manifest_payload(resolved)
        self.assertNotIn(
            "prompt_adapter", baseline_payload["configured_eval_run"]["benchmark"]
        )

        adapted = replace(
            resolved,
            configured=replace(
                resolved.configured,
                benchmark=replace(
                    benchmark, prompt_adapter="cvbench_explicit_options"
                ),
            ),
        )
        adapted_payload = stable_manifest_payload(adapted)
        self.assertEqual(
            adapted_payload["configured_eval_run"]["benchmark"]["prompt_adapter"],
            "cvbench_explicit_options",
        )

    def test_atomic_writer_emits_sorted_group_readable_json(self) -> None:
        resolved = self._resolved()
        run_directory = self._make_run_directory(resolved)
        path = write_resolved_run_manifest(
            resolved, generated_at="2026-09-03T00:00:00Z", slurm_job_id="123"
        )
        self.assertEqual(path, run_directory / MANIFEST_FILENAME)
        self.assertTrue(path.read_bytes().endswith(b"\n"))
        parsed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["slurm_job_id"], "123")
        self.assertTrue(path.stat().st_mode & 0o040)
        self.assertFalse(list(run_directory.glob(f".{MANIFEST_FILENAME}.*.tmp")))

    def test_atomic_writer_cleans_temporary_file_on_replace_error(self) -> None:
        resolved = self._resolved()
        run_directory = self._make_run_directory(resolved)
        with mock.patch("os.replace", side_effect=OSError("synthetic")):
            with self.assertRaisesRegex(EvalConfigError, "cannot write"):
                write_resolved_run_manifest(resolved)
        self.assertFalse(list(run_directory.glob(f".{MANIFEST_FILENAME}.*.tmp")))

    def test_new_run_directory_is_accepted_but_never_created_by_validation(self) -> None:
        resolved = self._resolved()
        validate_output_destination(resolved)
        self.assertFalse(Path(resolved.paths.run_directory).exists())

    def test_any_existing_destination_fails_when_resume_is_false(self) -> None:
        resolved = self._resolved()
        self._make_run_directory(resolved)
        with self.assertRaisesRegex(EvalConfigError, "resume is false"):
            validate_output_destination(resolved)

    def test_existing_private_run_directory_fails(self) -> None:
        resolved = self._resolved(resume=True)
        run_directory = self._make_run_directory(resolved)
        run_directory.chmod(0o700)
        with self.assertRaisesRegex(EvalConfigError, "group-readable and traversable"):
            validate_output_destination(resolved)

    def test_resume_requires_a_readable_group_readable_manifest(self) -> None:
        resolved = self._resolved(resume=True)
        run_directory = self._make_run_directory(resolved)
        with self.assertRaisesRegex(EvalConfigError, "no readable"):
            validate_output_destination(resolved)
        manifest = run_directory / MANIFEST_FILENAME
        manifest.write_text("{}\n", encoding="utf-8")
        manifest.chmod(0o600)
        with self.assertRaisesRegex(EvalConfigError, "not group-readable"):
            validate_output_destination(resolved)

    def test_resume_rejects_malformed_manifest(self) -> None:
        resolved = self._resolved(resume=True)
        run_directory = self._make_run_directory(resolved)
        manifest = run_directory / MANIFEST_FILENAME
        manifest.write_text("not-json\n", encoding="utf-8")
        manifest.chmod(0o640)
        with self.assertRaisesRegex(EvalConfigError, "cannot read existing"):
            validate_output_destination(resolved)

    def test_matching_partial_and_failed_runs_are_resumable(self) -> None:
        for status in ("prepared", "partial", "failed"):
            with self.subTest(status=status):
                fixture = ResolutionFixture()
                try:
                    resolved = resolve_eval_run(
                        fixture.config(resume=True),
                        fixture.workspace,
                        validate_destination=False,
                    )
                    Path(resolved.paths.run_directory).mkdir(parents=True)
                    records = (
                        (self._terminal(),) if status == "failed" else ()
                    )
                    write_resolved_run_manifest(
                        resolved,
                        run_status=status,
                        dataset_terminal_statuses=records,
                    )
                    validate_output_destination(resolved)
                finally:
                    fixture.close()

    def test_completed_run_fails_even_when_resume_is_true(self) -> None:
        resolved = self._resolved(resume=True)
        self._make_run_directory(resolved)
        write_resolved_run_manifest(
            resolved,
            run_status="completed",
            dataset_terminal_statuses=(self._terminal(status="completed"),),
        )
        with self.assertRaisesRegex(EvalConfigError, "cannot be resumed"):
            validate_output_destination(resolved)

    def test_fingerprint_mismatch_and_unknown_status_fail(self) -> None:
        resolved = self._resolved(resume=True)
        run_directory = self._make_run_directory(resolved)
        path = write_resolved_run_manifest(resolved, run_status="partial")
        value = json.loads(path.read_text(encoding="utf-8"))
        value["config_fingerprint"] = "0" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
        path.chmod(0o640)
        with self.assertRaisesRegex(EvalConfigError, "does not match its payload"):
            validate_output_destination(resolved)

        different = replace(
            resolved,
            paths=replace(resolved.paths, cache_root=str(self.fixture.workspace / "different")),
        )
        value = build_resolved_run_manifest(different, run_status="partial")
        path.write_text(json.dumps(value), encoding="utf-8")
        path.chmod(0o640)
        with self.assertRaisesRegex(EvalConfigError, "different config fingerprint"):
            validate_output_destination(resolved)

        value = build_resolved_run_manifest(resolved, run_status="partial")
        value["run_status"] = "mystery"
        path.write_text(json.dumps(value), encoding="utf-8")
        path.chmod(0o640)
        with self.assertRaisesRegex(EvalConfigError, "non-resumable status"):
            validate_output_destination(resolved)

    def test_existing_non_directory_destination_fails(self) -> None:
        resolved = self._resolved(resume=True)
        run_directory = Path(resolved.paths.run_directory)
        run_directory.parent.mkdir(parents=True)
        run_directory.write_text("collision", encoding="utf-8")
        with self.assertRaisesRegex(EvalConfigError, "not a directory"):
            validate_output_destination(resolved)

    def test_resume_authorization_is_excluded_from_compatibility_fingerprint(self) -> None:
        nonresume = self._resolved(False)
        resume_config = replace(nonresume.configured.output, resume=True)
        resume = replace(nonresume, configured=replace(nonresume.configured, output=resume_config))
        self.assertEqual(
            canonical_config_fingerprint(nonresume), canonical_config_fingerprint(resume)
        )
        self.assertFalse(
            stable_manifest_payload(nonresume)["configured_eval_run"]["output"]["resume"]
        )
        self.assertTrue(stable_manifest_payload(resume)["configured_eval_run"]["output"]["resume"])

    def test_false_initial_run_can_resume_with_true_but_other_changes_cannot(self) -> None:
        initial = self._resolved(False)
        self._make_run_directory(initial)
        write_resolved_run_manifest(initial, run_status="partial")
        resume_output = replace(initial.configured.output, resume=True)
        resumable = replace(
            initial, configured=replace(initial.configured, output=resume_output)
        )
        validate_output_destination(resumable)

        scientific = replace(
            resumable,
            configured=replace(
                resumable.configured,
                benchmark=replace(
                    resumable.configured.benchmark,
                    generation=replace(
                        resumable.configured.benchmark.generation, max_new_tokens=65
                    ),
                ),
            ),
        )
        path_change = replace(
            resumable,
            paths=replace(resumable.paths, cache_root=str(self.fixture.workspace / "other-cache")),
        )
        profile_change = replace(
            resumable,
            configured=replace(
                resumable.configured,
                model=replace(resumable.configured.model, name="different-model"),
            ),
        )
        baseline = canonical_config_fingerprint(resumable)
        for changed in (scientific, path_change, profile_change):
            self.assertNotEqual(baseline, canonical_config_fingerprint(changed))

    def test_terminal_status_serialization_and_fingerprint_invariance(self) -> None:
        resolved = self._resolved()
        records = (
            self._terminal("ZBench", "failed", "z/status.json"),
            self._terminal("ABench", "completed", "a/status.json"),
        )
        manifest = build_resolved_run_manifest(
            resolved,
            run_status="failed",
            dataset_terminal_statuses=records,
        )
        self.assertEqual(
            [record["dataset"] for record in manifest["dataset_terminal_statuses"]],
            ["ABench", "ZBench"],
        )
        self.assertNotIn("dataset_terminal_statuses", stable_manifest_payload(resolved))
        self.assertEqual(manifest["config_fingerprint"], canonical_config_fingerprint(resolved))

    def test_terminal_status_run_consistency(self) -> None:
        completed = self._terminal(status="completed")
        failed = self._terminal(status="failed")
        invalid = (
            ("unknown", ()),
            ("prepared", (failed,)),
            ("completed", ()),
            ("completed", (failed,)),
            ("failed", ()),
            ("failed", (completed,)),
            ("partial", (completed,)),
        )
        for run_status, records in invalid:
            with self.subTest(run_status=run_status, records=records):
                with self.assertRaises(EvalConfigError):
                    validate_dataset_terminal_statuses(run_status, records)

        self.assertEqual(validate_dataset_terminal_statuses("prepared", ()), ())
        self.assertEqual(
            validate_dataset_terminal_statuses("completed", (completed,)),
            (completed,),
        )
        self.assertEqual(validate_dataset_terminal_statuses("failed", (failed,)), (failed,))
        self.assertEqual(validate_dataset_terminal_statuses("partial", (failed,)), (failed,))

    def test_terminal_status_identity_path_and_order_validation(self) -> None:
        invalid = (
            (self._terminal(dataset=""),),
            (self._terminal(dataset=" OCRBench"),),
            (self._terminal(status="partial"),),
            (self._terminal(path="../status.json"),),
            (self._terminal(path="/status.json"),),
            (self._terminal(path="a/../status.json"),),
            (self._terminal(path="a\\status.json"),),
            (self._terminal(), self._terminal()),
        )
        for records in invalid:
            with self.subTest(records=records):
                with self.assertRaises(EvalConfigError):
                    validate_dataset_terminal_statuses("failed", records)

        records = (
            self._terminal("ZBench", "failed", "z/status.json"),
            self._terminal("ABench", "completed", "a/status.json"),
        )
        validated = validate_dataset_terminal_statuses("failed", records)
        self.assertEqual(tuple(record.dataset for record in validated), ("ABench", "ZBench"))


if __name__ == "__main__":
    unittest.main()
