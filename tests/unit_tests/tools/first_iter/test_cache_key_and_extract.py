# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Regression tests for the two cache defects that silently disabled reuse.

Both were invisible in logs: the AOT key drifted with call order so nothing
ever hit, and extraction failed whenever the container lacked the zstd CLI.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

_root_override = os.environ.get("MCORE_REPO_ROOT")
_REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
_TOOLS = _REPO_ROOT / "tools" / "persistent_cache"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def aot():
    return _load("cute_aot_cache_under_test", "cute_aot_cache.py")


@pytest.fixture(scope="module")
def node_setup():
    return _load(
        "mcore_cache_node_setup_under_test", "_triton_patch_shim/mcore_cache_node_setup.py"
    )


def _sample(value):
    return value


class TestAotCacheKey:
    """The key must identify WHAT is compiled, never WHEN."""

    def test_key_ignores_call_ordinal(self, aot):
        # A seed compiles with the MLIR file cache cold and a consuming run does
        # not, so the same kernel arrives at a different ordinal. Keying on it
        # meant a warm run reused nothing.
        first = aot._call_key("ns", 0, _sample, (1, "a"), {"opt": 2})
        later = aot._call_key("ns", 37, _sample, (1, "a"), {"opt": 2})
        assert first == later

    @pytest.mark.parametrize(
        "namespace,args,kwargs",
        [
            ("other-ns", (1, "a"), {"opt": 2}),
            ("ns", (2, "a"), {"opt": 2}),
            ("ns", (1, "b"), {"opt": 2}),
            ("ns", (1, "a"), {"opt": 3}),
        ],
    )
    def test_key_still_separates_distinct_compiles(self, aot, namespace, args, kwargs):
        baseline = aot._call_key("ns", 0, _sample, (1, "a"), {"opt": 2})
        assert aot._call_key(namespace, 0, _sample, args, kwargs) != baseline

    def test_key_is_stable_across_processes(self, aot):
        # Ranks in one job are separate processes; a per-process seed in the
        # hash would give every rank a different key for the same kernel.
        code = (
            "import importlib.util;"
            f"spec=importlib.util.spec_from_file_location('m', r'{_TOOLS / 'cute_aot_cache.py'}');"
            "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
            "print(m._call_key('ns', 0, len, (1,), {}))"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            ).stdout.strip()
            for _ in range(2)
        }
        assert len(runs) == 1


@pytest.mark.skipif(shutil.which("zstd") is None, reason="needs the zstd CLI to write the fixture")
class TestExtractWithoutZstdCli:
    """Containers are not guaranteed to ship the zstd helper tar shells out to."""

    @staticmethod
    def _write_tarball(tmp_path: Path) -> Path:
        payload = tmp_path / "src"
        (payload / "nested").mkdir(parents=True)
        (payload / "kernel.bin").write_bytes(b"kernel-payload")
        (payload / "nested" / "other.bin").write_bytes(b"nested-payload")
        plain = tmp_path / "scope.tar"
        with tarfile.open(plain, "w") as archive:
            archive.add(payload, arcname=".")
        subprocess.run(["zstd", "-q", "-f", str(plain)], check=True)
        return tmp_path / "scope.tar.zst"

    def test_round_trips_without_the_cli(self, node_setup, tmp_path):
        pytest.importorskip("zstandard")
        tarball = self._write_tarball(tmp_path)
        dest = tmp_path / "dest"
        dest.mkdir()

        assert node_setup._extract_without_zstd_cli(str(tarball), str(dest)) is None
        assert (dest / "kernel.bin").read_bytes() == b"kernel-payload"
        assert (dest / "nested" / "other.bin").read_bytes() == b"nested-payload"

    def test_fallback_runs_in_a_fresh_interpreter(self, node_setup, tmp_path, monkeypatch):
        # The fallback must not import zstandard in-process: reached from
        # sitecustomize (interpreter still starting) in a worker thread, that
        # import raised ModuleNotFoundError in the 26.07 image even though the
        # module is installed (job 3633302 ran cold). A fresh interpreter sees
        # the normal sys.path. Prove the subprocess route by pointing the
        # interpreter at a stub that records the call and extracts a plain tar.
        tarball = self._write_tarball(tmp_path)
        dest = tmp_path / "dest"
        dest.mkdir()
        stub = tmp_path / "python_stub.sh"
        stub.write_text(
            "#!/bin/sh\n"
            f"echo \"$*\" > {tmp_path}/stub_args\n"
            # $3 = tarball, $4 = dest ("-c", script, tarball, dest)
            'zstd -dc "$3" | tar -xf - -C "$4"\n'
        )
        stub.chmod(0o755)
        monkeypatch.setattr(node_setup.sys, "executable", str(stub))

        assert node_setup._extract_without_zstd_cli(str(tarball), str(dest)) is None
        assert "-c" in (tmp_path / "stub_args").read_text()
        assert (dest / "kernel.bin").read_bytes() == b"kernel-payload"

    def test_reports_a_reason_instead_of_raising(self, node_setup, tmp_path):
        # Callers treat a returned reason as "fall back to a cold start"; an
        # exception here would take the whole run down instead.
        corrupt = tmp_path / "scope.tar.zst"
        corrupt.write_bytes(b"not a zstd frame")
        dest = tmp_path / "dest"
        dest.mkdir()

        reason = node_setup._extract_without_zstd_cli(str(corrupt), str(dest))
        assert reason is not None and reason.startswith("python zstandard: ")
