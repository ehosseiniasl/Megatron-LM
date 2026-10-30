# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Exercise installer options without compiling the training dependency set."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

INSTALLER = (Path(__file__).resolve().parents[1] / "install.sh").read_text()
PREAMBLE = INSTALLER.split("# ==== cpython ====")[0]
INTERNAL_INSTALLS = INSTALLER.split("# ==== cutlass-dsl-internal ====")[1].split(
    "# ==== mistral-dataloader ===="
)[0]
ONE_LOGGER_INSTALL = INSTALLER.split("# ==== one-logger ====")[1].split(
    "# ==== debugging ===="
)[0]
CUDNN_CHECK = (
    "# "
    + INSTALLER.split("# PyPI's 1.28.0 release")[1].split(
        "# Two distributions ship a cutlass package"
    )[0]
)
INTERNAL_CHECK = INSTALLER.split("# Two distributions ship a cutlass package")[1].split(
    "# Ask the loaded library its own version."
)[0]
# The split above starts in a comment line; restore the comment prefix.
INTERNAL_CHECK = "# " + INTERNAL_CHECK
VERSION_STAMP = INSTALLER.split("# ==== version-stamp ====")[1]


class InstallOptionsTest(unittest.TestCase):
    def run_installer(self, body, **options):
        env = os.environ.copy()
        for name in (
            "INSTALL_INTERNAL_DEPS",
            "WHEEL_DIR",
            "SOURCE_DIR",
            "NCCL_REPO_URL",
            "NCCL_REPO_SHA",
            "CUDNN_FRONTEND_REPO_URL",
            "CUDNN_FRONTEND_REPO_SHA",
            "TRANSFORMER_ENGINE_REPO_URL",
            "TRANSFORMER_ENGINE_REPO_SHA",
        ):
            env.pop(name, None)
        env.update(options)
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "install.sh"
            script.write_text(PREAMBLE + body)
            return subprocess.run(
                ["bash", str(script)], env=env, text=True, capture_output=True
            )

    def test_default_requires_internal_wheels(self):
        with tempfile.TemporaryDirectory() as wheels:
            result = self.run_installer(INTERNAL_INSTALLS, WHEEL_DIR=wheels)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing wheel:", result.stderr)

    def test_plain_bash_preserves_training_sources_and_paths(self):
        result = self.run_installer("""
test "$WHEEL_DIR" = "$(dirname "${BASH_SOURCE[0]}")"
test "$SOURCE_DIR" = /opt
printf '%s\\n' "$NCCL_REPO" "$NCCL_REF" "$CUDNN_FRONTEND_REPO" \\
    "$CUDNN_FRONTEND_REF" "$TRANSFORMER_ENGINE_REPO" "$TRANSFORMER_ENGINE_REF"
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "https://gitlab-master.nvidia.com/nccl/nccl.git",
                "78b39cdb6bcd1568f7cebde80878606655cf06b1",
                "ssh://git@gitlab-master.nvidia.com:12051/ADLR/cudnn_frontend.git",
                "d844f3053f924fccf9f7ada0084c012eba1d20a9",
                "ssh://git@gitlab-master.nvidia.com:12051/ADLR/transformer_engine.git",
                "1ac66e5a2e9666b36dc5e0b6e8ce1a4ce3a5a86f",
            ],
        )

    def test_original_version_stamp_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_installer(VERSION_STAMP.replace("/etc/", directory + "/"))
            self.assertEqual(result.returncode, 0, result.stderr)
            stamp = Path(directory) / "megatron_dependency_set_version"
            self.assertEqual(stamp.read_text(), "14\n")
            self.assertEqual(list(Path(directory).iterdir()), [stamp])

    def test_legacy_directory_overrides_still_work(self):
        result = self.run_installer(
            '\nprintf "%s %s\\n" "$WHEEL_DIR" "$SOURCE_DIR"\n',
            WHEEL_DIR="/custom/wheels",
            SOURCE_DIR="/custom/sources",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "/custom/wheels /custom/sources")

    def test_default_installs_internal_deps_and_checks_both_wheels(self):
        body = """
pip() { printf 'pip:%s\\n' "$*"; }
python() { printf 'python:%s\\n' "$*"; }
touch "$WHEEL_DIR/$CUTLASS_DSL_INTERNAL_WHEEL" "$WHEEL_DIR/$GDP_ATTN_FUSED_WHEEL"
"""
        with tempfile.TemporaryDirectory() as wheels:
            result = self.run_installer(
                body
                + INTERNAL_INSTALLS
                + ONE_LOGGER_INSTALL
                + CUDNN_CHECK
                + INTERNAL_CHECK,
                WHEEL_DIR=wheels,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("pip:install"), 3)
        self.assertIn(
            "pip:install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/"
            "hwinf-mlwfo-pypi/simple one-logger==1.5.5",
            result.stdout,
        )
        self.assertIn("import gdp_attn", result.stdout)
        self.assertIn("tanh_clamp_scale", result.stdout)
        self.assertIn("m.version('nvidia-cutlass-dsl-internal')", result.stdout)

    def test_public_ci_skips_internal_installs_and_checks(self):
        body = """
pip() { echo 'unexpected pip invocation' >&2; return 1; }
python() { printf 'python:%s\\n' "$*"; }
"""
        with tempfile.TemporaryDirectory() as wheels:
            result = self.run_installer(
                body
                + INTERNAL_INSTALLS
                + ONE_LOGGER_INSTALL
                + CUDNN_CHECK
                + INTERNAL_CHECK,
                WHEEL_DIR=wheels,
                INSTALL_INTERNAL_DEPS="0",
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("import cutlass.cute; import flash_attn.cute", result.stdout)
        self.assertNotIn("gdp_attn", result.stdout)
        self.assertNotIn("cutlass.tensor_utils", result.stdout)
        self.assertNotIn("tanh_clamp_scale", result.stdout)
        self.assertIn("import cudnn;", result.stdout)

    def test_only_explicit_zero_skips_wheels(self):
        with tempfile.TemporaryDirectory() as wheels:
            for value in ("", "1", "false"):
                with self.subTest(value=value):
                    result = self.run_installer(
                        INTERNAL_INSTALLS,
                        WHEEL_DIR=wheels,
                        INSTALL_INTERNAL_DEPS=value,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("missing wheel:", result.stderr)

    def test_paired_source_overrides(self):
        for dependency in ("NCCL", "CUDNN_FRONTEND", "TRANSFORMER_ENGINE"):
            with self.subTest(dependency=dependency):
                options = {
                    f"{dependency}_REPO_URL": "https://github.com/example/source.git",
                    f"{dependency}_REPO_SHA": "a" * 40,
                }
                body = f'\nprintf "%s %s\\n" "${dependency}_REPO" "${dependency}_REF"\n'
                result = self.run_installer(body, **options)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    result.stdout.strip(),
                    "https://github.com/example/source.git " + "a" * 40,
                )


if __name__ == "__main__":
    unittest.main()
