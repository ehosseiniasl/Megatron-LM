"""Focused deterministic Megatron dynamic-server launch tests."""

from __future__ import annotations

import copy
import json
import signal
import stat
import sys
import tempfile
import time
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from orchestration.configured_eval_run import ResolvedEvalRun
from orchestration.eval_run_config_loading import load_configured_eval_run
from orchestration.eval_run_path_resolution import resolve_eval_run
from orchestration.megatron_server_process_management import (
    megatron_dynamic_server_launch as server_launch,
)
from phase1_resolution_test_support import ResolutionFixture

MegatronDynamicServerLaunchError = server_launch.MegatronDynamicServerLaunchError
build_megatron_dynamic_server_launch = server_launch.build_megatron_dynamic_server_launch
render_megatron_dynamic_server_launch = server_launch.render_megatron_dynamic_server_launch
LocalTorchrunServerProcess = server_launch.LocalTorchrunServerProcess
LocalTorchrunServerProcessError = server_launch.LocalTorchrunServerProcessError
MegatronDynamicServerLaunch = server_launch.MegatronDynamicServerLaunch


NVBENCH = Path(__file__).resolve().parents[1]
MATCHED1540_RUN_CONFIG = (
    NVBENCH
    / "configs"
    / "run_configs"
    / "n4_v6_example.yaml"
)
_CHILD = Path(__file__).with_name("local_process_supervisor_child.py")


class MegatronDynamicServerLaunchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ResolutionFixture()
        resolved = resolve_eval_run(
            self.fixture.config(),
            self.fixture.workspace,
            validate_destination=False,
        )
        self.compiler_cache = self.fixture.workspace / "compiler-cache"
        self.compiler_cache.mkdir(mode=0o700)
        self.overrides = (
            ("deterministic-mode", True),
            ("inference-max-requests", 1),
            ("tokenizer-type", "huggingface"),
            ("vision-model-type", "mistral-vit-1p6b"),
            ("transformer-impl", "transformer_engine"),
            ("img-h", 1540),
            ("img-w", 1540),
            ("patch-dim", 14),
            ("vision-dynamic-resolution-max-patches", 13312),
            ("vision-projection-expansion-factor", 2.5),
            ("num-speculative-tokens", 0),
        )
        self.resolved = replace(resolved, resolved_cli_overrides=self.overrides)

    def tearDown(self) -> None:
        self.fixture.close()

    def build(self, **changes):
        values = {
            "resolved": self.resolved,
            "run_directory": self.resolved.paths.run_directory,
            "host": "127.0.0.1",
            "port": 8000,
            "coordinator_port": 8001,
            "python_executable": sys.executable,
            "compiler_cache_directory": self.compiler_cache,
        }
        values.update(changes)
        return build_megatron_dynamic_server_launch(**values)

    def test_exact_argv_order_caps_overrides_and_no_expectations(self) -> None:
        launch = self.build()
        server = self.resolved.configured.model.server
        expected_prefix = (
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc-per-node",
            str(self.resolved.configured.run.gpus_per_node),
            "--max-restarts",
            "0",
            self.resolved.paths.runtime_entrypoint,
            "--load",
            self.resolved.paths.checkpoint_root,
            "--ckpt-step",
            str(self.resolved.configured.checkpoint.iteration),
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--inference-coordinator-port",
            "8001",
        )
        context = str(server.context_length)
        expected_caps = (
            "--inference-max-seq-length",
            context,
            "--inference-dynamic-batching-max-tokens",
            "16384",
        )
        expected_overrides = (
            "--deterministic-mode",
            "--inference-max-requests",
            "1",
            "--tokenizer-type",
            "huggingface",
            "--vision-model-type",
            "mistral-vit-1p6b",
            "--transformer-impl",
            "transformer_engine",
            "--img-h",
            "1540",
            "--img-w",
            "1540",
            "--patch-dim",
            "14",
            "--vision-dynamic-resolution-max-patches",
            "13312",
            "--vision-projection-expansion-factor",
            "2.5",
            "--num-speculative-tokens",
            "0",
        )
        self.assertEqual(launch.argv, expected_prefix + expected_caps + expected_overrides)

    def test_multinode_argv_changes_only_torchrun_topology(self) -> None:
        configured = replace(
            self.resolved.configured,
            run=replace(
                self.resolved.configured.run,
                launch_mode="slurm_multinode_torchrun",
                nodes=2,
            ),
        )
        resolved = replace(self.resolved, configured=configured)
        local = self.build()
        distributed = self.build(
            resolved=resolved,
            host="server-node-0",
            node_rank=1,
            master_addr="server-node-0",
            rendezvous_port=29500,
        )
        local_entrypoint = local.argv.index(self.resolved.paths.runtime_entrypoint)
        distributed_entrypoint = distributed.argv.index(
            self.resolved.paths.runtime_entrypoint
        )
        local_server_arguments = list(local.argv[local_entrypoint:])
        host_index = local_server_arguments.index("--host")
        del local_server_arguments[host_index : host_index + 2]
        self.assertEqual(tuple(local_server_arguments), distributed.argv[distributed_entrypoint:])
        self.assertEqual(
            distributed.argv[4:distributed_entrypoint],
            (
                "--nnodes",
                "2",
                "--nproc-per-node",
                "8",
                "--node-rank",
                "1",
                "--master-addr",
                "server-node-0",
                "--master-port",
                "29500",
                "--max-restarts",
                "0",
            ),
        )
        self.assertNotIn("--standalone", distributed.argv)
        self.assertEqual(
            distributed.endpoint,
            "http://server-node-0:8000/v1/chat/completions",
        )
        self.assertEqual(
            distributed.health_endpoint,
            "http://server-node-0:8000/v1/health",
        )
        self.assertEqual(distributed.environment_overrides, local.environment_overrides)

    def test_checkpoint_owned_context_adds_no_sequence_overrides(self) -> None:
        configured = replace(
            self.resolved.configured,
            model=replace(
                self.resolved.configured.model,
                server=replace(
                    self.resolved.configured.model.server,
                    context_length=None,
                ),
            ),
        )
        overrides = self.overrides + (("inference-context-from-checkpoint", True),)
        launch = self.build(
            resolved=replace(
                self.resolved,
                configured=configured,
                resolved_cli_overrides=overrides,
            )
        )
        self.assertIn("--inference-context-from-checkpoint", launch.argv)
        for argument in (
            "--decoder-seq-length",
            "--max-position-embeddings",
            "--inference-max-seq-length",
            "--inference-dynamic-batching-max-tokens",
        ):
            self.assertNotIn(argument, launch.argv)

    def test_explicit_context_preserves_a_safe_batch_token_floor(self) -> None:
        for context_length, expected_budget in ((8192, 16384), (12000, 16384), (32768, 32768)):
            with self.subTest(context_length=context_length):
                configured = replace(
                    self.resolved.configured,
                    model=replace(
                        self.resolved.configured.model,
                        server=replace(
                            self.resolved.configured.model.server,
                            context_length=context_length,
                        ),
                    ),
                )
                launch = self.build(resolved=replace(self.resolved, configured=configured))
                budget_index = launch.argv.index("--inference-dynamic-batching-max-tokens")
                self.assertEqual(launch.argv[budget_index + 1], str(expected_budget))

    def test_exact_environment_endpoints_paths_and_render_isolation(self) -> None:
        launch = self.build()
        run_directory = Path(self.resolved.paths.run_directory)
        self.assertFalse(run_directory.exists())
        self.assertEqual(launch.working_directory, self.resolved.paths.runtime_root)
        self.assertEqual(launch.endpoint, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(launch.health_endpoint, "http://127.0.0.1:8000/v1/health")
        self.assertEqual(
            launch.server_log_path,
            str(run_directory / "megatron_dynamic_server.log"),
        )
        self.assertEqual(
            launch.environment_overrides,
            (
                ("PYTHONPATH", self.resolved.paths.runtime_root),
                ("PYTHONNOUSERSITE", "1"),
                ("PYTHONDONTWRITEBYTECODE", "1"),
                ("PYTHONUNBUFFERED", "1"),
                ("WANDB_MODE", "disabled"),
                ("TRITON_CACHE_DIR", str(self.compiler_cache / "triton")),
                (
                    "TORCHINDUCTOR_CACHE_DIR",
                    str(self.compiler_cache / "torchinductor"),
                ),
                ("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "0"),
                ("NCCL_ALGO", "Ring"),
                ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
                ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0"),
            ),
        )
        self.assertNotIn(
            "CUDA_DEVICE_MAX_CONNECTIONS", dict(launch.environment_overrides)
        )
        rendered = render_megatron_dynamic_server_launch(launch)
        frozen_snapshot = copy.deepcopy(rendered)
        json.dumps(rendered, allow_nan=False)
        rendered["argv"].append("--mutated")
        rendered["environment_overrides"][0]["value"] = "mutated"
        self.assertEqual(render_megatron_dynamic_server_launch(launch), frozen_snapshot)
        self.assertEqual(launch.argv, tuple(frozen_snapshot["argv"]))
        with self.assertRaises(FrozenInstanceError):
            launch.endpoint = "mutated"

    def test_nondeterministic_profile_omits_determinism_environment(self) -> None:
        overrides = tuple(item for item in self.overrides if item[0] != "deterministic-mode")
        launch = self.build(resolved=replace(self.resolved, resolved_cli_overrides=overrides))
        environment = dict(launch.environment_overrides)
        for name in (
            "NCCL_ALGO",
            "CUBLAS_WORKSPACE_CONFIG",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
        ):
            self.assertNotIn(name, environment)
        self.assertNotIn("LD_LIBRARY_PATH", environment)

    def test_trusted_standalone_common_state_is_explicit_and_requires_its_file(self) -> None:
        checkpoint = replace(
            self.resolved.configured.checkpoint,
            trusted_standalone_common_state=True,
        )
        configured = replace(self.resolved.configured, checkpoint=checkpoint)
        resolved = replace(self.resolved, configured=configured)
        with self.assertRaisesRegex(
            MegatronDynamicServerLaunchError, "readable standalone common.pt"
        ):
            self.build(resolved=resolved)

        (self.fixture.iteration / "common.pt").write_bytes(b"trusted-test-fixture")
        environment = dict(self.build(resolved=resolved).environment_overrides)
        self.assertEqual(environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"], "1")
        self.assertEqual(environment["TORCH_FORCE_WEIGHTS_ONLY_LOAD"], "0")

    def test_real_profile_contains_only_intentional_overrides(self) -> None:
        configured = load_configured_eval_run(MATCHED1540_RUN_CONFIG)
        resolved = replace(
            self.resolved,
            resolved_cli_overrides=configured.model.server.cli_overrides,
        )
        argv = self.build(resolved=resolved).argv
        expected = {
            "--tokenizer-prompt-format": "mistral",
            "--vlm-language-transformer-impl": "inference_optimized",
            "--vlm-image-preprocess-preset": "mistral-crawl-eval",
        }
        for flag, value in expected.items():
            self.assertEqual(argv.count(flag), 1)
            self.assertEqual(argv[argv.index(flag) + 1], value)

    def test_invalid_type_topology_host_and_ports_fail_closed(self) -> None:
        with self.assertRaises(MegatronDynamicServerLaunchError):
            self.build(resolved=object())
        run = self.resolved.configured.run
        cases = (
            replace(self.resolved.configured, run=replace(run, launch_mode="slurm")),
            replace(self.resolved.configured, run=replace(run, nodes=2)),
            replace(self.resolved.configured, run=replace(run, gpus_per_node=7)),
        )
        for configured in cases:
            with self.subTest(configured=configured), self.assertRaises(
                MegatronDynamicServerLaunchError
            ):
                self.build(resolved=replace(self.resolved, configured=configured))
        for changes in (
            {"host": "localhost"},
            {"port": True},
            {"port": 1023},
            {"port": 65536},
            {"coordinator_port": 8000},
        ):
            with self.subTest(changes=changes), self.assertRaises(
                MegatronDynamicServerLaunchError
            ):
                self.build(**changes)

    def test_invalid_paths_fail_closed(self) -> None:
        for changes in (
            {"run_directory": "relative/run"},
            {"run_directory": self.fixture.workspace / "elsewhere"},
            {"python_executable": "python3"},
            {"compiler_cache_directory": "relative/cache"},
            {"compiler_cache_directory": "/tmp"},
            {"compiler_cache_directory": self.fixture.workspace / "missing-cache"},
        ):
            with self.subTest(changes=changes), self.assertRaises(
                MegatronDynamicServerLaunchError
            ):
                self.build(**changes)
        paths = replace(self.resolved.paths, runtime_entrypoint="relative/server.py")
        with self.assertRaises(MegatronDynamicServerLaunchError):
            self.build(resolved=replace(self.resolved, paths=paths))

    def test_duplicate_owned_false_and_malformed_overrides_fail_closed(self) -> None:
        bad_overrides = (
            self.overrides + (("tokenizer-type", "duplicate"),),
            (("load", "/other/checkpoint"),),
            (("deterministic-mode", False),),
            (("invalid_name", 1),),
            (("temperature", float("nan")),),
            (("empty", ""),),
            (("malformed", 1, 2),),
        )
        for overrides in bad_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(
                MegatronDynamicServerLaunchError
            ):
                self.build(resolved=replace(self.resolved, resolved_cli_overrides=overrides))


class LocalTorchrunServerProcessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def launch(
        self,
        mode: str,
        overrides: tuple[tuple[str, str], ...] = (),
    ) -> MegatronDynamicServerLaunch:
        return MegatronDynamicServerLaunch(
            argv=(sys.executable, "-B", str(_CHILD), mode),
            working_directory=str(self.root),
            environment_overrides=overrides,
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            health_endpoint="http://127.0.0.1:8000/v1/health",
            server_log_path=str(self.root / "server.log"),
        )

    def supervisor(
        self,
        mode: str,
        *,
        overrides: tuple[tuple[str, str], ...] = (),
        environment: dict[str, str] | None = None,
    ) -> LocalTorchrunServerProcess:
        return LocalTorchrunServerProcess(
            self.launch(mode, overrides),
            terminate_timeout_seconds=0.03,
            kill_timeout_seconds=0.2,
            base_environment={} if environment is None else environment,
        )

    def wait_for_log(self, text: str, timeout: float = 0.5) -> None:
        path = self.root / "server.log"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists() and text in path.read_text(encoding="utf-8"):
                return
            time.sleep(0.005)
        self.fail(f"log did not contain {text!r}")

    def wait_for_exit(
        self,
        process: LocalTorchrunServerProcess,
        timeout: float = 0.5,
    ) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = process.poll()
            if result is not None:
                return result
            time.sleep(0.005)
        self.fail("subprocess did not exit")

    def test_only_python_roots_are_sanitized_and_container_metadata_is_preserved(
        self,
    ) -> None:
        log_path = self.root / "server.log"
        log_path.write_text("stale content", encoding="utf-8")
        inherited = {
            "KEEP": "yes",
            "PYTHONHOME": "/bad/home",
            "PYTHONPATH": "/bad/pythonpath",
            "CONDA_PREFIX": "/container/conda",
            "_CONDA_EXE": "/container/conda/bin/conda",
            "VIRTUAL_ENV": "/container/venv",
            "PIXI_PROJECT_MANIFEST": "/container/pixi.toml",
            "LD_LIBRARY_PATH": "/container/native/lib",
            "OVERRIDE": "old",
        }
        overrides = (("PYTHONPATH", "/clean/runtime"), ("OVERRIDE", "new"))
        process = self.supervisor(
            "environment", overrides=overrides, environment=inherited
        )
        process.start()
        self.assertEqual(self.wait_for_exit(process), 0)
        log_lines = log_path.read_text(encoding="utf-8").splitlines()
        payload = json.loads(log_lines[0])
        self.assertEqual(payload["KEEP"], "yes")
        self.assertEqual(payload["PYTHONPATH"], "/clean/runtime")
        self.assertEqual(payload["LD_LIBRARY_PATH"], "/container/native/lib")
        self.assertEqual(payload["OVERRIDE"], "new")
        self.assertEqual(payload["__cwd__"], str(self.root))
        self.assertIn("STDERR_CAPTURED", log_lines)
        self.assertIsNone(payload["PYTHONHOME"])
        self.assertEqual(payload["CONDA_PREFIX"], "/container/conda")
        self.assertEqual(payload["_CONDA_EXE"], "/container/conda/bin/conda")
        self.assertEqual(payload["VIRTUAL_ENV"], "/container/venv")
        self.assertEqual(payload["PIXI_PROJECT_MANIFEST"], "/container/pixi.toml")
        self.assertEqual(stat.S_IMODE(log_path.stat().st_mode), 0o640)
        self.assertIsNone(process._log_stream)

    def test_start_once_and_forwarded_sigterm_targets_launcher(self) -> None:
        process = self.supervisor("exit_on_term")
        self.addCleanup(process.terminate)
        process.start()
        self.wait_for_log("READY")
        with self.assertRaises(LocalTorchrunServerProcessError):
            process.start()
        process.forward_signal(signal.SIGTERM)
        self.assertEqual(self.wait_for_exit(process), 23)
        self.wait_for_log("FORWARDED_TERM")
        self.assertIsNone(process._log_stream)

    def test_forced_launcher_cleanup_is_bounded_and_reported_as_failure(self) -> None:
        process = self.supervisor("ignore_term")
        process.start()
        self.wait_for_log("READY")
        started = time.monotonic()
        with self.assertRaisesRegex(
            LocalTorchrunServerProcessError, "required forced cleanup"
        ):
            process.terminate()
        self.assertLess(time.monotonic() - started, 0.3)
        self.assertEqual(process.returncode, -signal.SIGKILL)
        self.assertEqual(process.terminate(), -signal.SIGKILL)
        self.assertIsNone(process._log_stream)

    def test_supervisor_does_not_reimplement_worker_process_groups(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "orchestration"
            / "megatron_server_process_management"
            / "megatron_dynamic_server_launch.py"
        )
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("start_new_session", text)
        self.assertNotIn("killpg", text)

    def test_context_manager_cleans_up_and_invalid_inputs_fail_closed(self) -> None:
        process = self.supervisor("exit_on_term")
        with process:
            self.wait_for_log("READY")
            self.assertIsNone(process.returncode)
        self.assertEqual(process.returncode, 23)
        self.assertIsNone(process._log_stream)
        with self.assertRaises(LocalTorchrunServerProcessError):
            LocalTorchrunServerProcess(self.launch("environment"), kill_timeout_seconds=0)
        with self.assertRaises(LocalTorchrunServerProcessError):
            LocalTorchrunServerProcess(self.launch("environment"), base_environment=[])
        with self.assertRaises(LocalTorchrunServerProcessError):
            self.supervisor("environment").forward_signal(signal.SIGTERM)
        with self.assertRaises(LocalTorchrunServerProcessError):
            process.forward_signal(float(signal.SIGTERM))

    def test_invalid_environment_override_fails_before_process_start(self) -> None:
        cases = (
            (("DUPLICATE", "one"), ("DUPLICATE", "two")),
            (("BAD\0NAME", "value"),),
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(
                LocalTorchrunServerProcessError
            ):
                self.supervisor("environment", overrides=overrides)


if __name__ == "__main__":
    unittest.main()
