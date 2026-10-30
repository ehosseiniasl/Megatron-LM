"""Essential contracts for the multi-node server supervisor."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

from orchestration.eval_run_path_resolution import (
    PreparedMegatronDynamicServerVlmFewshotRun,
    prepare_vlmeval_run_directory,
    resolve_eval_run,
)
from orchestration.configured_eval_run import configured_resource, run_for_resource
from orchestration.megatron_server_process_management.megatron_dynamic_server_launch import (
    build_megatron_dynamic_server_launch,
)
from orchestration.megatron_server_process_management.multinode_megatron_server_step import (
    MultinodeAllocation,
    MultinodeServerStepDependencies,
    _write_node_terminal,
    execute_multinode_megatron_server_step,
    validate_multinode_allocation,
)
from orchestration.megatron_server_process_management.split_megatron_server_step import (
    EnclosingAllocationError,
    PreparedServerStep,
    SplitMegatronServerStepError,
    TerminalServerStep,
    create_split_server_step_control_paths,
    write_prepared_server_step,
    write_server_step_shutdown_request,
    write_terminal_server_step,
)
from orchestration.resolved_run_manifest import (
    canonical_config_fingerprint,
    write_resolved_run_manifest,
)
from phase1_resolution_test_support import ResolutionFixture
from phase5_orchestration_test_support import RecordingProcess


class MultinodeServerStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ResolutionFixture()
        configured = self.fixture.config()
        configured = replace(
            configured,
            run=run_for_resource(
                configured.run.name,
                configured_resource(configured, "n2"),
            ),
        )
        self.resolved = resolve_eval_run(
            configured, self.fixture.workspace, validate_destination=False
        )
        self.cache = self.fixture.workspace / "compiler-cache"
        self.cache.mkdir(mode=0o700)
        self.fingerprint = canonical_config_fingerprint(self.resolved)

    def tearDown(self) -> None:
        self.fixture.close()

    def prepared(self, rank: int) -> PreparedMegatronDynamicServerVlmFewshotRun:
        launch = build_megatron_dynamic_server_launch(
            self.resolved,
            self.resolved.paths.run_directory,
            host="server-node-0",
            port=8000,
            coordinator_port=8001,
            python_executable=str(Path(sys.executable).resolve()),
            compiler_cache_directory=self.cache,
            node_rank=rank,
            master_addr="server-node-0",
            rendezvous_port=29500,
        )
        return PreparedMegatronDynamicServerVlmFewshotRun(self.resolved, launch)

    def dependencies(self, rank: int, *, peer_status: str = "clean"):
        process = RecordingProcess([])

        def shutdown(paths, fingerprint, _poll):
            write_server_step_shutdown_request(paths, fingerprint)
            if rank == 0:
                _write_node_terminal(
                    paths, 1, TerminalServerStep(fingerprint, peer_status, -15)
                )
            else:
                write_terminal_server_step(
                    paths, TerminalServerStep(fingerprint, "clean", -15)
                )

        return MultinodeServerStepDependencies(
            lambda _resolved: MultinodeAllocation("12345", 2, rank),
            lambda _launch: process,
            shutdown,
        )

    def prepare_rank_zero(self) -> None:
        prepared = self.prepared(0)
        directory = prepare_vlmeval_run_directory(self.resolved)
        write_resolved_run_manifest(self.resolved, slurm_job_id="12345")
        paths = create_split_server_step_control_paths(directory.path, "12345")
        write_prepared_server_step(
            paths,
            PreparedServerStep(
                self.fingerprint,
                prepared.launch.endpoint,
                prepared.launch.health_endpoint,
                str(paths.attempt_directory / "megatron_dynamic_server_node_000.log"),
            ),
        )

    def test_rank_zero_aggregates_clean_node_terminals(self) -> None:
        result = execute_multinode_megatron_server_step(
            self.prepared(0), dependencies=self.dependencies(0)
        )
        self.assertEqual(result.server_log_artifact.name, "megatron_dynamic_server_node_000.log")
        self.assertTrue(result.prepared_manifest.is_file())
        self.assertTrue(result.terminal_control_artifact.is_file())

    def test_secondary_joins_rank_zero_artifacts(self) -> None:
        self.prepare_rank_zero()
        result = execute_multinode_megatron_server_step(
            self.prepared(1), dependencies=self.dependencies(1)
        )
        self.assertEqual(result.server_log_artifact.name, "megatron_dynamic_server_node_001.log")
        self.assertTrue(result.terminal_control_artifact.is_file())

    def test_rank_zero_rejects_failed_peer(self) -> None:
        with self.assertRaisesRegex(SplitMegatronServerStepError, "multi-node server step failed"):
            execute_multinode_megatron_server_step(
                self.prepared(0), dependencies=self.dependencies(0, peer_status="failed")
            )

    def test_allocation_requires_one_supervisor_per_node(self) -> None:
        environment = {
            "SLURM_JOB_ID": "12345",
            "SLURM_NNODES": "2",
            "SLURM_JOB_NUM_NODES": "2",
            "SLURM_NTASKS": "2",
            "SLURM_GPUS_ON_NODE": "8",
            "SLURM_NODEID": "1",
            "SLURM_LOCALID": "0",
            "SLURM_PROCID": "1",
        }
        self.assertEqual(
            validate_multinode_allocation(self.resolved, environment),
            MultinodeAllocation("12345", 2, 1),
        )
        with self.assertRaises(EnclosingAllocationError):
            validate_multinode_allocation(
                self.resolved, environment | {"SLURM_NTASKS": "1"}
            )


if __name__ == "__main__":
    unittest.main()
