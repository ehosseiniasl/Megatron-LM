# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Optional topology installation contract; production PCI validation is not bypassed."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
TOOLS = REPO_ROOT / "tools/persistent_cache"


@pytest.fixture
def installation(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "topology_test_installer", TOOLS / "install_image_cache.py"
    )
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    namespace = tmp_path / "kernel-namespace"
    (namespace / "consolidated").mkdir(parents=True)
    source = tmp_path / "old-cmh-topology.xml"
    source.write_text('<system version="1"/>')
    dest = tmp_path / "image-cache"
    calls = []
    checks = []
    monkeypatch.setattr(installer, "_git_commit", lambda _path: "matching-tools-commit")

    def no_extract(*_args):
        raise AssertionError("optional topology must not be extracted from a kernel archive")

    monkeypatch.setattr(
        installer, "_load_node_setup", lambda: SimpleNamespace(_extract_one=no_extract)
    )
    monkeypatch.setattr(
        installer, "_self_check", lambda *args, **kwargs: checks.append((args, kwargs))
    )

    def run_helper(command, *, check):
        assert check is True
        assert command[:2] == [sys.executable, str(TOOLS / "prepare_nccl_topology_template.py")]
        assert "--skip-pci-validation" not in command
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        manifest = Path(command[command.index("--manifest") + 1])
        output.write_bytes(source.read_bytes())
        manifest.write_text(json.dumps({"namespace": command[command.index("--namespace") + 1]}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(installer.subprocess, "run", run_helper)
    argv = ["installer", "--namespace", str(namespace), "--dest", str(dest), "--tools", str(TOOLS)]
    return installer, argv, source, dest, namespace, calls, checks


def test_optional_topology_is_separate_payload_and_enables_only_its_own_default(
    installation, monkeypatch
):
    installer, argv, source, dest, namespace, calls, checks = installation
    # An identically named archive is deliberately ignored: the paired input is
    # independent of where the seven kernel-cache scopes were seeded.
    (namespace / "consolidated/nccl_topo.tar.zst").write_bytes(b"not the selected source")
    monkeypatch.setattr(
        sys, "argv", argv + ["--nccl-topo-file", str(source), "--nccl-topo-namespace", "cmh-v1"]
    )
    assert installer.main() == 0
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("--source") + 1] == str(source)
    assert command[command.index("--output") + 1] == str(dest / "nccl_topo/topo_template.xml")
    assert command[command.index("--manifest") + 1] == str(
        dest / "nccl_topo/topo_template.manifest.json"
    )
    manifest = (dest / "MANIFEST").read_text()
    assert f"scopes={installer.DEFAULT_SCOPES} nccl_topo\n" in manifest
    assert "runtime_nccl_topo_cache=1\n" in manifest
    assert "nccl_topo_namespace=cmh-v1\n" in manifest
    assert "runtime_prewarm_weighted_relu=0\n" in manifest
    assert "runtime_equivalent_autotune_cache=0\n" in manifest
    assert "scope_nccl_topo=files:2 sha256:" in manifest
    assert not (dest / "tools").exists()
    assert source.read_text() == '<system version="1"/>'
    assert checks == [((dest, TOOLS / "_triton_patch_shim"), {"expect_nccl_topology": True})]


def test_no_topology_keeps_seven_scope_default_and_does_not_call_helper(installation, monkeypatch):
    installer, argv, _source, dest, _namespace, calls, checks = installation
    monkeypatch.setattr(sys, "argv", argv)
    assert installer.main() == 0
    manifest = (dest / "MANIFEST").read_text()
    assert f"scopes={installer.DEFAULT_SCOPES}\n" in manifest
    assert "nccl_topo" not in manifest
    assert not (dest / "nccl_topo").exists()
    assert not calls
    assert checks[0][1] == {"expect_nccl_topology": False}


@pytest.mark.parametrize("missing", ["file", "namespace"])
def test_incomplete_pair_fails_before_creating_install(installation, monkeypatch, missing):
    installer, argv, source, dest, _namespace, calls, _checks = installation
    extra = (
        ["--nccl-topo-namespace", "cmh-v1"]
        if missing == "file"
        else ["--nccl-topo-file", str(source)]
    )
    monkeypatch.setattr(sys, "argv", argv + extra)
    with pytest.raises(SystemExit) as error:
        installer.main()
    assert error.value.code == 2
    assert not dest.exists()
    assert not calls


def test_scope_without_paired_input_is_rejected(installation, monkeypatch):
    installer, argv, _source, dest, _namespace, calls, _checks = installation
    monkeypatch.setattr(sys, "argv", argv + ["--scopes", "triton nccl_topo"])
    with pytest.raises(SystemExit) as error:
        installer.main()
    assert error.value.code == 2
    assert not dest.exists()
    assert not calls


def test_host_validation_failure_never_publishes_manifest(installation, monkeypatch):
    installer, argv, source, dest, _namespace, _calls, checks = installation
    monkeypatch.setattr(
        sys, "argv", argv + ["--nccl-topo-file", str(source), "--nccl-topo-namespace", "cmh-v1"]
    )

    def fail_validation(command, **_kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(installer.subprocess, "run", fail_validation)
    with pytest.raises(subprocess.CalledProcessError):
        installer.main()
    assert not (dest / "MANIFEST").exists()
    assert not checks


def test_self_check_clears_inherited_topology_and_requires_new_payload(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "self_check_installer", TOOLS / "install_image_cache.py"
    )
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    for name in (
        "MCORE_NCCL_TOPO_CACHE",
        "NCCL_TOPO_FILE",
        "NCCL_TOPO_DUMP_FILE",
        "NCCL_GRAPH_FILE",
        "NCCL_GRAPH_DUMP_FILE",
    ):
        monkeypatch.setenv(name, "unrelated-old-setting")
    dest = tmp_path / "image-cache"
    dest.mkdir()
    (dest / ".mcore_ready").touch()
    (dest / ".setup.lock").touch()

    def run_python(command, *, env, **_kwargs):
        assert all(
            name not in env
            for name in (
                "MCORE_NCCL_TOPO_CACHE",
                "NCCL_TOPO_FILE",
                "NCCL_TOPO_DUMP_FILE",
                "NCCL_GRAPH_FILE",
                "NCCL_GRAPH_DUMP_FILE",
            )
        )
        assert env["PYTHONPATH"] == str(TOOLS / "_triton_patch_shim")
        assert "NCCL_TOPO_FILE" in command[-1]
        assert "torch" not in command[-1] and "nccl" not in command[-1]
        return SimpleNamespace(
            returncode=0, stdout=f"{dest}/triton\n{dest}/nccl_topo/topo_template.xml\n", stderr=""
        )

    monkeypatch.setattr(installer.subprocess, "run", run_python)
    installer._self_check(dest, TOOLS / "_triton_patch_shim", expect_nccl_topology=True)
    assert not (dest / ".mcore_ready").exists()
    assert not (dest / ".setup.lock").exists()
