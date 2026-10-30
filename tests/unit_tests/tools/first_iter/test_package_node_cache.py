"""No-GPU tests for node-local cache packaging and atomic publication."""

import os
import subprocess
from pathlib import Path

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
PC_DIR = REPO_ROOT / "tools" / "persistent_cache"
PACKAGE = PC_DIR / "package_node_cache.sh"
CONSOLIDATE = PC_DIR / "consolidate_cache.sh"


def _local_cache(tmp_path: Path, job_id: str = "123") -> Path:
    local = tmp_path / "local"
    for scope in ("triton", "inductor"):
        scope_dir = local / scope
        scope_dir.mkdir(parents=True)
        (scope_dir / f"{scope}.txt").write_text(f"cached-{scope}\n")
    (local / ".mcore_ready").write_text(f"write {job_id}")
    return local


def _env(tmp_path: Path, local: Path, job_id: str = "123") -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SLURM_JOB_ID": job_id,
            "SLURM_NODEID": "7",
            "MCORE_CACHE_DIR": str(tmp_path / "namespace"),
            "MCORE_CACHE_PACKAGE_LOCAL": str(local),
            "MCORE_CACHE_PACKAGE_TMP": str(tmp_path / "package_tmp"),
            "MCORE_CACHE_SCOPES": "triton inductor",
        }
    )
    return env


def test_package_then_consolidate(tmp_path):
    local = _local_cache(tmp_path)
    env = _env(tmp_path, local)

    result = subprocess.run(["bash", str(PACKAGE)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    shard = Path(env["MCORE_CACHE_DIR"]) / "cache_write" / "_node" / "node-7"
    assert sorted(path.name for path in shard.iterdir()) == [
        "MANIFEST",
        "inductor.tar.zst",
        "triton.tar.zst",
    ]
    assert not local.exists()
    assert not any(path.name.startswith(".node-7.part-") for path in shard.parent.iterdir())
    listing = subprocess.run(
        ["tar", "-tf", str(shard / "triton.tar.zst")], capture_output=True, text=True, check=True
    )
    assert "./triton.txt" in listing.stdout.splitlines()

    result = subprocess.run(
        ["bash", str(CONSOLIDATE)], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    consolidated = Path(env["MCORE_CACHE_DIR"]) / "consolidated"
    assert (consolidated / "triton.tar.zst").is_file()
    assert (consolidated / "inductor.tar.zst").is_file()


def test_package_rejects_wrong_sentinel_without_publication(tmp_path):
    local = _local_cache(tmp_path)
    (local / ".mcore_ready").write_text("read 123")
    env = _env(tmp_path, local)

    result = subprocess.run(["bash", str(PACKAGE)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "is not 'write 123'" in result.stderr
    assert not (Path(env["MCORE_CACHE_DIR"]) / "cache_write").exists()
    assert local.exists()


def test_package_preserves_an_empty_scope(tmp_path):
    local = _local_cache(tmp_path)
    (local / "cudnn_fe").mkdir()
    env = _env(tmp_path, local)
    env["MCORE_CACHE_SCOPES"] = "triton inductor cudnn_fe"

    result = subprocess.run(["bash", str(PACKAGE)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "empty scope:" in result.stdout

    shard = Path(env["MCORE_CACHE_DIR"]) / "cache_write" / "_node" / "node-7"
    tarball = shard / "cudnn_fe.tar.zst"
    assert tarball.is_file() and tarball.stat().st_size > 0
    extracted = tmp_path / "empty-scope-extracted"
    extracted.mkdir()
    subprocess.run(["tar", "-xf", str(tarball), "-C", str(extracted)], check=True)
    assert not list(extracted.rglob("*"))


def test_package_rejects_a_missing_scope(tmp_path):
    local = _local_cache(tmp_path)
    env = _env(tmp_path, local)
    env["MCORE_CACHE_SCOPES"] = "triton inductor cudnn_fe"

    result = subprocess.run(["bash", str(PACKAGE)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "missing scope:" in result.stderr
    assert not (Path(env["MCORE_CACHE_DIR"]) / "cache_write").exists()
