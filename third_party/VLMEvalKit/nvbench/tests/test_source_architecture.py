from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
import warnings
from pathlib import Path


NVBENCH_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = NVBENCH_ROOT / "tests"
CLIENT_ROOT = NVBENCH_ROOT / "orchestration"
CLIENT_SUBPACKAGE_ROOTS = tuple(
    CLIENT_ROOT / name
    for name in (
        "dynamic_server_http_client",
        "fewshot_dataset_adaptation",
        "megatron_server_process_management",
        "vlmeval_result_readers",
    )
)
LEAF_MODULE_PACKAGES = {
    module: package
    for package, modules in {
        "fewshot_dataset_adaptation": """demonstration_selection
            fewshot_conversation_builder""",
        "dynamic_server_http_client": """megatron_dynamic_server_chat_client
            server_preflight""",
        "megatron_server_process_management": """megatron_dynamic_server_launch
            multinode_megatron_server_step split_megatron_server_step""",
        "vlmeval_result_readers": """ocrbench_score_json vlmeval_accuracy_csv""",
    }.items()
    for module in modules.split()
}
OBSOLETE_LEAF_MODULE_PACKAGES = {
    module: package
    for package, modules in {
        "fewshot_dataset_adaptation": """demonstration_answer_normalization
            fewshot_selection_records packaged_demonstration_loading vlmeval_message_conversion
            vlmeval_fewshot_dataset_proxy
            vlmeval_demonstration_scored_row_partition_artifact""",
        "dynamic_server_http_client": """megatron_dynamic_server_chat_protocol
            server_health_preflight server_generation_preflight""",
        "megatron_server_process_management": """local_torchrun_server_process
            split_server_step_control split_server_step_waiting termination_signal_handling""",
    }.items()
    for module in modules.split()
}
OBSOLETE_CLIENT_ROOT_MODULES = frozenset(
    """configured_benchmark_suite enclosing_allocation_validation
    eval_benchmark_plan_contract_validation eval_benchmark_suite_profile_loading
    eval_run_contract_validation immutable_run_json_artifact
    megatron_dynamic_server_vlm_fewshot_contracts
    megatron_dynamic_server_vlm_fewshot_dependencies
    megatron_dynamic_server_vlm_fewshot_preparation model_profile_config_loading
    resolved_run_terminal_status runtime_repository_resolution strict_json_serialization
    vlmeval_benchmark_result_contract_validation vlmeval_configured_run_mode
    vlmeval_eval_suite_loading vlmeval_external_server_result_recording
    vlmeval_generated_answer_result_finalization vlmeval_native_dataset_contract
    vlmeval_prediction_completeness vlmeval_result_artifact_writing
    vlmeval_result_contract_types vlmeval_row_identity vlmeval_run_directory_preparation
    vlmeval_suite_result_recording vlmeval_suite_status_validation
    vlmeval_upstream_generation_bridge""".split()
)
OBSOLETE_NVBENCH_MODULES = frozenset({"vendored_vlmeval_tree_digest"})
OBSOLETE_MODULES = frozenset(
    {
        *OBSOLETE_LEAF_MODULE_PACKAGES,
        *OBSOLETE_CLIENT_ROOT_MODULES,
        *OBSOLETE_NVBENCH_MODULES,
    }
)
ENTRYPOINTS = (
    NVBENCH_ROOT / "submit_suite.py",
    NVBENCH_ROOT / "run_server.py",
    NVBENCH_ROOT / "run_client.py",
    NVBENCH_ROOT / "run_judge.py",
    NVBENCH_ROOT / "preflight_datasets.py",
    NVBENCH_ROOT / "verify_vendored_tree.py",
)
MODULE_ENTRYPOINTS = {
    "submit_suite.py",
    "run_server.py",
    "run_client.py",
    "run_judge.py",
    "preflight_datasets.py",
    "verify_vendored_tree.py",
}
HELP_OPTION = {
    "verify_vendored_tree.py": "--root",
}


def production_python_files() -> tuple[Path, ...]:
    files = list(NVBENCH_ROOT.glob("*.py"))
    files.extend(CLIENT_ROOT.glob("*.py"))
    for directory in CLIENT_SUBPACKAGE_ROOTS:
        files.extend(directory.glob("*.py"))
    return tuple(sorted(files))


class SourceArchitectureTest(unittest.TestCase):
    def test_leaf_packages_replace_the_moved_flat_modules(self) -> None:
        for module, package in LEAF_MODULE_PACKAGES.items():
            self.assertTrue((CLIENT_ROOT / package / f"{module}.py").is_file(), module)
            self.assertFalse((CLIENT_ROOT / f"{module}.py").exists(), module)
        for module, package in OBSOLETE_LEAF_MODULE_PACKAGES.items():
            self.assertFalse((CLIENT_ROOT / package / f"{module}.py").exists(), module)
        for module in OBSOLETE_CLIENT_ROOT_MODULES:
            self.assertFalse((CLIENT_ROOT / f"{module}.py").exists(), module)
        for module in OBSOLETE_NVBENCH_MODULES:
            self.assertFalse((NVBENCH_ROOT / f"{module}.py").exists(), module)
        for directory in CLIENT_SUBPACKAGE_ROOTS:
            self.assertTrue((directory / "__init__.py").is_file(), directory.name)

    def test_public_production_contracts_are_documented(self) -> None:
        for path in production_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            definitions = [
                node
                for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
            ]
            methods = [
                method
                for definition in definitions
                if isinstance(definition, ast.ClassDef)
                for method in definition.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not method.name.startswith("_")
            ]
            with self.subTest(path=path.name):
                self.assertFalse(
                    [node.name for node in (*definitions, *methods) if not ast.get_docstring(node)]
                )

    def test_source_imports_use_the_new_leaf_package_boundaries(self) -> None:
        paths = production_python_files() + tuple(sorted(TEST_ROOT.glob("*.py")))
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            modules.extend(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            for module in filter(None, modules):
                basename = module.rsplit(".", 1)[-1]
                with self.subTest(path=path.name, imported_module=module):
                    self.assertNotIn(basename, OBSOLETE_MODULES)
                expected_package = LEAF_MODULE_PACKAGES.get(basename)
                if expected_package is None or path.parent.name == expected_package:
                    continue
                with self.subTest(path=path.name, imported_module=module):
                    self.assertIn(f".{expected_package}.{basename}", f".{module}")

    def test_production_python_has_no_slurm_or_shell_launch(self) -> None:
        for path in production_python_files():
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("shell=True", text)
                self.assertNotIn("subprocess.run(['srun'", text)
                self.assertNotIn('subprocess.run(["srun"', text)
                self.assertNotIn("subprocess.Popen(['srun'", text)
                self.assertNotIn('subprocess.Popen(["srun"', text)

    def test_production_does_not_mutate_upstream_registries(self) -> None:
        for path in production_python_files():
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("supported_VLM", text)
                self.assertNotIn("sys.modules[", text)
                self.assertNotIn("mock.patch", text)

    def test_entrypoint_is_thin_and_help_imports_without_upstream_runtime(self) -> None:
        for entrypoint in ENTRYPOINTS:
            with self.subTest(entrypoint=entrypoint.name):
                maximum_lines = {
                    "run_judge.py": 120,
                    "verify_vendored_tree.py": 150,
                }.get(entrypoint.name, 100)
                self.assertLessEqual(
                    len(entrypoint.read_text(encoding="utf-8").splitlines()),
                    maximum_lines,
                )
                if entrypoint.name in MODULE_ENTRYPOINTS:
                    command = (
                        sys.executable,
                        "-B",
                        "-m",
                        f"nvbench.{entrypoint.stem}",
                        "--help",
                    )
                    environment = {
                        **os.environ,
                        "PYTHONPATH": str(NVBENCH_ROOT.parent),
                    }
                else:
                    command = (sys.executable, "-B", str(entrypoint), "--help")
                    environment = os.environ.copy()
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2,
                    env=environment,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                expected = HELP_OPTION.get(entrypoint.name, "--run-config")
                self.assertIn(expected, completed.stdout)

    def test_production_entrypoints_do_not_bootstrap_import_paths(self) -> None:
        for entrypoint in ENTRYPOINTS:
            with self.subTest(entrypoint=entrypoint.name):
                text = entrypoint.read_text(encoding="utf-8")
                self.assertNotIn("sys.path", text)
                self.assertNotIn("# noqa: E402", text)
                if entrypoint.name in MODULE_ENTRYPOINTS:
                    self.assertFalse(text.startswith("#!"))

    def test_production_modules_remain_reviewable_and_lines_are_bounded(self) -> None:
        for path in production_python_files():
            lines = path.read_text(encoding="utf-8").splitlines()
            with self.subTest(path=path.name):
                if path.parent in (
                    CLIENT_ROOT,
                    *CLIENT_SUBPACKAGE_ROOTS,
                ) and path.name != "utils.py":
                    if len(lines) > 550:
                        warnings.warn(
                            f"review module cohesion: {path} has {len(lines)} lines",
                            stacklevel=2,
                        )
                self.assertFalse(
                    [number for number, line in enumerate(lines, 1) if len(line) > 100]
                )


if __name__ == "__main__":
    unittest.main()
