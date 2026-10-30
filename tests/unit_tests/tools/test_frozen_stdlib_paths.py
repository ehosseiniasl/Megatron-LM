# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Test the frozen-stdlib repair's ordering, complete module set, and fail-closed guard."""

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

_override = os.environ.get("MCORE_REPO_ROOT")
ROOT = Path(_override) if _override else Path(__file__).resolve().parents[3]
MODULES = ("genericpath", "posixpath", "ntpath")


def patch_block(relative: str) -> str:
    """Extract the exact four-line repair from either source installation route."""
    lines = (ROOT / relative).read_text().splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.strip().startswith("for _stdlib_module in ")
    )
    return textwrap.dedent("\n".join(lines[start : start + 4]))


def test_installer_patches_all_path_modules_before_configure():
    """Retain the performance patch and freeze the complete Ubuntu path module set."""
    script = patch_block("dependencies/install.sh")
    assert "for _stdlib_module in genericpath posixpath ntpath; do" in script
    assert 'grep -q "class ALLOW_MISSING" Lib/genericpath.py' in script
    source = (ROOT / "dependencies/install.sh").read_text()
    assert (
        source.index('git show --format= --binary "${CPYTHON_PATCH_COMMIT}"')
        < source.index("for _stdlib_module in ")
        < source.index("./configure --prefix=/usr")
    )
    dockerfile = (ROOT / "dependencies/Dockerfile.training").read_text()
    assert "bash /opt/dependencies/install.sh" in dockerfile
    assert "./configure" not in dockerfile
    assert "pip install" not in dockerfile


@pytest.mark.parametrize("missing", [None, "genericpath", "posixpath", "ntpath", "backport"])
def test_patch_copies_all_modules_and_rejects_incomplete_base(tmp_path, missing):
    """Exercise the actual shell insert against a synthetic base without touching host Python."""
    base = tmp_path / "base"
    library = tmp_path / "source/Lib"
    base.mkdir()
    library.mkdir(parents=True)
    for module in MODULES:
        content = (
            "class ALLOW_MISSING: pass\n" if module == "genericpath" else f"name = {module!r}\n"
        )
        if missing == "backport" and module == "genericpath":
            content = "name = 'vanilla'\n"
        if missing != module:
            (base / f"{module}.py").write_text(content)
        (library / f"{module}.py").write_text("vanilla source\n")
    command = patch_block("dependencies/install.sh").replace("/usr/lib/python3.12", str(base))
    result = subprocess.run(
        ["bash", "-euc", command], cwd=library.parent, capture_output=True, text=True, timeout=10
    )
    assert (result.returncode == 0) == (missing is None)
    if missing is None:
        for module in MODULES:
            assert (library / f"{module}.py").read_bytes() == (base / f"{module}.py").read_bytes()
