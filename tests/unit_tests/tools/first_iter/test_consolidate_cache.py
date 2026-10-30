# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for tools/persistent_cache/consolidate_cache.sh."""

import os
import subprocess
import tarfile
from pathlib import Path

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]


def _consolidate() -> Path:
    return REPO_ROOT / "tools" / "persistent_cache" / "consolidate_cache.sh"


def _write_shard(shard_dir: Path, scope: str, files: dict[str, bytes]) -> None:
    """Create shard_dir/<scope>.tar.zst (gzip; tar -xf auto-detects) from files."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    src = shard_dir.parent / f"_src_{shard_dir.name}_{scope}"
    for rel, content in files.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    # Name it .tar.zst but write gzip so the assertions can read it back with the
    # Python stdlib (which has no zstd reader); consolidate_cache.sh extracts with
    # `tar -xf`, which auto-detects the format either way.
    with tarfile.open(shard_dir / f"{scope}.tar.zst", "w:gz") as tar:
        for rel in files:
            tar.add(src / rel, arcname=rel)


def _run(namespace: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MCORE_CACHE_DIR"] = str(namespace)
    env["MCORE_CACHE_SCOPES"] = "triton"
    # Force gzip output so the test can read the consolidated tarball with tarfile.
    env["CACHE_COMPRESSOR"] = "gzip"
    # macOS bsdtar otherwise adds AppleDouble ._* entries (Linux unaffected).
    env["COPYFILE_DISABLE"] = "1"
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(_consolidate())], check=True, capture_output=True, env=env, text=True
    )


def _members(tarball: Path) -> set[str]:
    with tarfile.open(tarball, "r:gz") as tar:
        return {member.name.lstrip("./") for member in tar.getmembers() if member.isfile()}


def _seed_three_nodes(namespace: Path) -> None:
    shards = namespace / "cache_write" / "_node"
    # One shared file on all three nodes + one unique file per node.
    _write_shard(shards / "node-0", "triton", {"AAAA/k.cubin": b"shared", "BB/only0.cubin": b"n0"})
    _write_shard(shards / "node-1", "triton", {"AAAA/k.cubin": b"shared", "CC/only1.cubin": b"n1"})
    _write_shard(shards / "node-2", "triton", {"AAAA/k.cubin": b"shared", "DD/only2.cubin": b"n2"})


def test_consolidate_unions_all_shards_by_default(tmp_path: Path) -> None:
    _seed_three_nodes(tmp_path)
    _run(tmp_path)
    out = tmp_path / "consolidated" / "triton.tar.zst"
    assert out.is_file()
    # Default min_fraction=0 keeps the full union: the shared file plus all uniques.
    assert _members(out) == {"AAAA/k.cubin", "BB/only0.cubin", "CC/only1.cubin", "DD/only2.cubin"}


def test_consolidate_min_fraction_keeps_only_shared_base(tmp_path: Path) -> None:
    _seed_three_nodes(tmp_path)
    # Require presence on 100% of nodes -> only the file on all three survives.
    _run(tmp_path, MCORE_CACHE_CONSOLIDATE_MIN_FRACTION="1.0")
    out = tmp_path / "consolidated" / "triton.tar.zst"
    assert _members(out) == {"AAAA/k.cubin"}


def test_consolidate_dedupes_shared_content(tmp_path: Path) -> None:
    _seed_three_nodes(tmp_path)
    _run(tmp_path)
    out = tmp_path / "consolidated" / "triton.tar.zst"
    # The shared path appears once, not once per source node.
    with tarfile.open(out, "r:gz") as tar:
        shared = [m for m in tar.getmembers() if m.name.lstrip("./") == "AAAA/k.cubin"]
    assert len(shared) == 1


def _seed_durable_dirs(namespace: Path) -> None:
    """Write-mode layout: plain per-node DIRS under
    <ns>/cache_write/_node/node-<rank>/<scope>/. Deliberately uses non-standard
    dir names to prove discovery is name-agnostic."""
    for host, unique in (("hostA", "BB/only0.cubin"), ("hostB", "CC/only1.cubin")):
        scope_dir = namespace / "cache_write" / "_node" / host / "triton"
        for rel, content in {"AAAA/k.cubin": b"shared", unique: host.encode()}.items():
            path = scope_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        # sentinel sits at the node-dir root, outside scope dirs
        (namespace / "cache_write" / "_node" / host / ".mcore_ready").write_text("write 123")


def test_consolidate_consumes_durable_seed_dirs(tmp_path: Path) -> None:
    _seed_durable_dirs(tmp_path)
    env = os.environ.copy()
    env["MCORE_CACHE_DIR"] = str(tmp_path)
    env["MCORE_CACHE_SCOPES"] = "triton"
    env["CACHE_COMPRESSOR"] = "gzip"
    env["COPYFILE_DISABLE"] = "1"
    env.pop("MCORE_CACHE_CONSOLIDATE_SRC", None)
    subprocess.run(
        ["bash", str(_consolidate())], check=True, capture_output=True, env=env, text=True
    )
    out = tmp_path / "consolidated" / "triton.tar.zst"
    assert out.is_file()
    members = _members(out)
    assert members == {"AAAA/k.cubin", "BB/only0.cubin", "CC/only1.cubin"}
    # the per-node sentinel must not leak into the sealed base
    assert not any("mcore_ready" in m for m in members)


def test_consolidate_preserves_an_empty_scope_archive(tmp_path: Path) -> None:
    shards = tmp_path / "cache_write" / "_node"
    _write_shard(shards / "node-0", "cudnn_fe", {})
    _write_shard(shards / "node-1", "cudnn_fe", {})

    result = _run(tmp_path, MCORE_CACHE_SCOPES="cudnn_fe")
    assert "empty_scope=1" in result.stdout
    out = tmp_path / "consolidated" / "cudnn_fe.tar.zst"
    assert out.is_file() and out.stat().st_size > 0
    assert _members(out) == set()


def test_consolidate_drops_a_partially_extracted_shard(tmp_path: Path) -> None:
    """A truncated shard must not contribute files: its partial extraction used
    to stay in the staging union and could be sealed as the representative copy."""
    shards = tmp_path / "cache_write" / "_node"
    healthy = b"H" * 1024
    _write_shard(shards / "node-0", "triton", {"HASH/kernel.cubin": healthy})
    _write_shard(shards / "node-1", "triton", {"HASH/kernel.cubin": healthy})
    # Truncate node-0's archive so tar extracts nothing usable and fails.
    broken = shards / "node-0" / "triton.tar.zst"
    broken.write_bytes(broken.read_bytes()[:40])

    result = _run(tmp_path)
    assert "extract failed" in result.stderr
    assert "shards=1" in result.stdout
    out = tmp_path / "consolidated" / "triton.tar.zst"
    with tarfile.open(out, "r:gz") as tar:
        member = tar.extractfile("./HASH/kernel.cubin") or tar.extractfile("HASH/kernel.cubin")
        assert member is not None and member.read() == healthy
