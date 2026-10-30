# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU coverage for the image bake's fresh-process reuse gate."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location(
    "finalize_image_extensions", REPO_ROOT / "tools/persistent_cache/finalize_image_extensions.py"
)
finalizer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(finalizer)


@pytest.fixture
def training_repo(tmp_path):
    repo = tmp_path / "training"
    package = repo / "megatron/core/tensor_parallel"
    package.mkdir(parents=True)
    (package / "csrc").mkdir()
    (package / "csrc/gtp_vmm_allocator.cpp").write_text("// allocator source\n")
    for directory in (repo / "megatron", repo / "megatron/core", package):
        (directory / "__init__.py").write_text("")
    (repo / "torch.py").write_text("__version__='test'\nclass version:\n    cuda='test'\n")
    (repo / "build_stub.py").write_text('''
import os
from pathlib import Path
from types import SimpleNamespace
def load(name):
    root = Path(os.environ['TORCH_EXTENSIONS_DIR'])
    if os.environ.get('FIC_TEST_WRONG_PATH'):
        root = root.parent / 'outside'
    path = root / name / (name + '.so')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or os.environ.get('FIC_TEST_REBUILD'):
        path.write_bytes(b'library')
    return SimpleNamespace(__file__=str(path))
''')
    (package / "random.py").write_text('''
from build_stub import load
_share_storage_ext = None
def _get_share_storage():
    global _share_storage_ext
    _share_storage_ext = load('share_storage_ext')
''')
    (package / "gtp_vmm_allocator.py").write_text('''
from build_stub import load
def _load_vmm_allocator_extension():
    return load('gtp_vmm_allocator')
''')
    return repo


def test_finalization_checks_fresh_process_and_records_provenance(tmp_path, training_repo):
    dest = tmp_path / "image"
    report = finalizer.finalize(dest, training_repo)
    assert report["fresh_process_reused"]
    assert len(report["artifacts"]) == 2
    assert report["source_sha256"].keys() == {
        "megatron/core/tensor_parallel/random.py",
        "megatron/core/tensor_parallel/gtp_vmm_allocator.py",
        "megatron/core/tensor_parallel/csrc/gtp_vmm_allocator.cpp",
    }
    assert json.loads((dest / "extensions-finalization.json").read_text()) == report


def test_rebuild_on_fresh_load_rejects_bake(tmp_path, training_repo, monkeypatch):
    monkeypatch.setenv("FIC_TEST_REBUILD", "1")
    dest = tmp_path / "image"
    with pytest.raises(RuntimeError, match="changed build artifacts"):
        finalizer.finalize(dest, training_repo)
    assert not (dest / "extensions-finalization.json").exists()


def test_source_tree_library_is_not_accepted_as_baked(tmp_path, training_repo, monkeypatch):
    monkeypatch.setenv("FIC_TEST_WRONG_PATH", "1")
    dest = tmp_path / "image"
    with pytest.raises(RuntimeError, match="outside final image path"):
        finalizer.finalize(dest, training_repo)
    assert not (dest / "extensions-finalization.json").exists()
