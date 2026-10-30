# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Topology activation orchestration; hardware validation is tested separately."""

import importlib.util
import os
from pathlib import Path

import pytest

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
PC_DIR = REPO_ROOT / "tools" / "persistent_cache"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    original_environment = dict(os.environ)
    for key in list(os.environ):
        if key.startswith(("MCORE_", "NCCL_")):
            monkeypatch.delenv(key)
    spec = importlib.util.spec_from_file_location(
        "topology_runtime", PC_DIR / "_triton_patch_shim/mcore_cache_node_setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = tmp_path / "image"
    (image / "nccl_topo").mkdir(parents=True)
    (image / "nccl_topo/topo_template.xml").write_text("test topology")
    (image / "nccl_topo/topo_template.manifest.json").write_text("test manifest")
    (image / "MANIFEST").write_text(
        "namespace=kernels\nruntime_nccl_topo_cache=1\nnccl_topo_namespace=cmh-test\n"
    )
    # A small validator double isolates activation from the actual host's PCI
    # layout. Real XML/hash/device mismatch behavior has dedicated helper tests.
    validator = tmp_path / "validator.py"
    validator.write_text(
        "def validate_template(template, manifest, namespace=None):\n"
        "    assert namespace == 'cmh-test'\n"
        "    assert template.read_text() == 'test topology'\n"
        "    assert manifest.read_text() == 'test manifest'\n"
        "    return {'gpu_count': 4}\n"
    )
    monkeypatch.setattr(module, "_shim_tool", lambda _name: str(validator))
    monkeypatch.setattr(module, "_rebase_triton_groups", lambda _base: None)
    monkeypatch.setenv("MCORE_CACHE_LOCAL", str(tmp_path / "local"))
    monkeypatch.setenv("MCORE_CACHE_SCOPES", "triton")
    monkeypatch.setenv("SLURM_JOB_ID", "topology-test")
    try:
        yield module, image
    finally:
        # setup() writes cache variables directly, outside monkeypatch's undo
        # list. Do not leak those pinned paths into other standalone tests.
        os.environ.clear()
        os.environ.update(original_environment)


@pytest.mark.parametrize("copy_image", [False, True])
def test_topology_is_validated_at_final_path_and_native_nccl_env_is_set(
    runtime, monkeypatch, copy_image
):
    module, image = runtime
    monkeypatch.setenv("MCORE_CACHE_IMAGE_COPY", str(int(copy_image)))
    for name in (
        "NCCL_TOPO_FILE",
        "NCCL_TOPO_DUMP_FILE",
        "NCCL_GRAPH_FILE",
        "NCCL_GRAPH_DUMP_FILE",
    ):
        monkeypatch.setenv(name, "/old/file")
    module._setup_image(str(image))
    base = image.parent / "local" if copy_image else image
    assert os.environ["NCCL_TOPO_FILE"] == str(base / "nccl_topo/topo_template.xml")
    assert os.environ["TRITON_CACHE_DIR"] == str(base / "triton")
    assert not any(
        name in os.environ
        for name in ("NCCL_TOPO_DUMP_FILE", "NCCL_GRAPH_FILE", "NCCL_GRAPH_DUMP_FILE")
    )
    assert (image / "nccl_topo/topo_template.xml").read_text() == "test topology"


@pytest.mark.parametrize("reason", ["missing", "invalid", "namespace"])
def test_bad_topology_falls_back_without_disabling_kernel_caches(runtime, monkeypatch, reason):
    module, image = runtime
    if reason == "missing":
        (image / "nccl_topo/topo_template.manifest.json").unlink()
    elif reason == "invalid":
        (image / "nccl_topo/topo_template.xml").write_text("changed")
    else:
        with (image / "MANIFEST").open("a") as stream:
            stream.write("nccl_topo_namespace=another-cluster\n")
    monkeypatch.setenv("NCCL_TOPO_FILE", "/stale/topology.xml")
    module._setup_image(str(image))
    assert "NCCL_TOPO_FILE" not in os.environ
    assert os.environ["TRITON_CACHE_DIR"] == str(image / "triton")


@pytest.mark.parametrize("kind", ["old", "disabled", "duplicate", "invalid"])
def test_unselected_topology_leaves_external_nccl_configuration_alone(runtime, monkeypatch, kind):
    module, image = runtime
    if kind == "old":
        (image / "MANIFEST").write_text("namespace=old-image\n")
    elif kind == "disabled":
        monkeypatch.setenv("MCORE_NCCL_TOPO_CACHE", "0")
    elif kind == "duplicate":
        with (image / "MANIFEST").open("a") as stream:
            stream.write("runtime_nccl_topo_cache=1\n")
    else:
        monkeypatch.setenv("MCORE_NCCL_TOPO_CACHE", "yes")
    monkeypatch.setenv("NCCL_TOPO_FILE", "/cluster/managed.xml")
    module._setup_image(str(image))
    assert os.environ["NCCL_TOPO_FILE"] == "/cluster/managed.xml"


def test_topology_metadata_does_not_invalidate_older_inductor_defaults(runtime, monkeypatch):
    module, image = runtime
    with (image / "MANIFEST").open("a") as stream:
        stream.write("runtime_prewarm_weighted_relu=1\nruntime_equivalent_autotune_cache=0\n")
    module._setup_image(str(image))
    assert os.environ["MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU"] == "1"
    assert os.environ["MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE"] == "0"
    assert "NCCL_TOPO_FILE" in os.environ


def test_image_setup_failure_cannot_leave_unvalidated_topology_active(runtime, monkeypatch):
    module, image = runtime
    for variable in module._TOPO_ENV:
        monkeypatch.setenv(variable, "/stale/file")
    monkeypatch.setattr(module, "_dir_write_exec", lambda _path: False)
    monkeypatch.setattr(module, "resolve_base", lambda _mode: None)
    module._setup_image(str(image))
    assert not any(variable in os.environ for variable in module._TOPO_ENV)


def test_explicit_enable_copies_optional_payload_even_when_image_default_is_off(
    runtime, monkeypatch
):
    module, image = runtime
    (image / "MANIFEST").write_text("namespace=kernels\nnccl_topo_namespace=cmh-test\n")
    monkeypatch.setenv("MCORE_NCCL_TOPO_CACHE", "1")
    monkeypatch.setenv("MCORE_CACHE_IMAGE_COPY", "1")
    module._setup_image(str(image))
    assert os.environ["NCCL_TOPO_FILE"] == str(image.parent / "local/nccl_topo/topo_template.xml")
