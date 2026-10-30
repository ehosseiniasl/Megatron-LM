"""No-GPU tests for the standalone env-only cache path (enable.sh +
mcore_cache_node_setup via the sitecustomize shim).

Everything runs in subprocesses with a scratch namespace: no torch, no SLURM,
no cluster. Covers the design's validation smoke: N concurrent Python
processes race the per-node setup; exactly one extracts, all see identical
pinned env, and the extracted content is present.
"""

import os
import subprocess
import tarfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# MCORE_REPO_ROOT lets the file run from a copy outside the package tree
# (the tests/unit_tests package __init__ imports torch; this file needs none).
_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
PC_DIR = REPO_ROOT / "tools" / "persistent_cache"
SHIM_DIR = PC_DIR / "_triton_patch_shim"


def _make_namespace(tmp_path: Path, scopes=("triton", "inductor")) -> Path:
    """Build a fake namespace with one gzip 'consolidated' tarball per scope."""
    ns = tmp_path / "ns"
    cons = ns / "consolidated"
    cons.mkdir(parents=True)
    for scope in scopes:
        payload = tmp_path / f"payload_{scope}"
        payload.mkdir()
        (payload / f"{scope}_artifact.txt").write_text(f"cached-{scope}\n")
        # gzip-compressed but named .tar.zst, matching the write side's
        # documented cosmetic naming (tar -xf auto-detects).
        with tarfile.open(cons / f"{scope}.tar.zst", "w:gz") as tar:
            tar.add(payload, arcname=".")
    return ns


def _env(ns: Path, local: Path, mode="read", scopes="triton inductor"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MCORE_CACHE")}
    env.pop("NODE_CACHE_BASE", None)
    env["PYTHONPATH"] = str(SHIM_DIR)
    env["MCORE_CACHE_DIR"] = str(ns)
    env["MCORE_CACHE_MODE"] = mode
    env["MCORE_CACHE_SCOPES"] = scopes
    env["MCORE_CACHE_LOCAL"] = str(local)
    return env


def _probe(env, expression="os.environ.get('TORCHINDUCTOR_CACHE_DIR', 'MISSING')"):
    return subprocess.run(
        ["python3", "-c", f"import os, sys; sys.stdout.write({expression})"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_concurrent_processes_extract_once(tmp_path):
    ns = _make_namespace(tmp_path)
    local = tmp_path / f"local_{uuid.uuid4().hex[:8]}"
    env = _env(ns, local)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _i: _probe(env), range(8)))

    pinned = {result.stdout for result in results}
    assert all(result.returncode == 0 for result in results)
    assert pinned == {str(local / "inductor")}
    extract_lines = [
        line
        for result in results
        for line in result.stderr.splitlines()
        if "[CACHE NODE SETUP] extracted" in line
    ]
    assert len(extract_lines) == 1, extract_lines
    assert (local / "inductor" / "inductor_artifact.txt").read_text() == "cached-inductor\n"
    assert (local / "triton" / "triton_artifact.txt").exists()
    assert (local / ".mcore_ready").is_file()


def test_missing_scope_is_cold_not_fatal(tmp_path):
    ns = _make_namespace(tmp_path, scopes=("inductor",))
    local = tmp_path / "local"
    env = _env(ns, local, scopes="triton inductor")
    result = _probe(env, "os.environ['TRITON_CACHE_DIR']")
    assert result.returncode == 0
    # triton had no tarball: pinned to an empty dir (cold), not an error.
    assert result.stdout == str(local / "triton")
    assert "triton=missing" in result.stderr


def test_write_mode_pins_empty_dirs(tmp_path):
    ns = tmp_path / "ns_write"
    ns.mkdir()
    local = tmp_path / "local_w"
    env = _env(ns, local, mode="write")
    result = _probe(env, "os.environ['TRITON_CACHE_DIR']")
    assert result.returncode == 0
    assert result.stdout == str(local / "triton")
    assert list((local / "triton").iterdir()) == []


@pytest.mark.parametrize("existing_cache", [False, True])
def test_inert_without_cache_dir(tmp_path, existing_cache):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MCORE_CACHE")}
    env.pop("NODE_CACHE_BASE", None)
    env.pop("TORCHINDUCTOR_CACHE_DIR", None)
    expected = "MISSING"
    if existing_cache:
        expected = str(tmp_path / "existing_inductor")
        env["TORCHINDUCTOR_CACHE_DIR"] = expected
    env["PYTHONPATH"] = str(SHIM_DIR)
    result = _probe(env)
    assert result.returncode == 0
    assert result.stdout == expected


def test_enable_sh_exports_interface(tmp_path):
    ns = _make_namespace(tmp_path)
    script = (
        f"export MCORE_CACHE_DIR={ns} && "
        f"source {PC_DIR}/enable.sh >/dev/null && "
        'echo "$MCORE_CACHE_MODE|$MCORE_CUTE_DSL_AOT_NAMESPACE|$MCORE_CACHE_ENV_FORWARD" && '
        'case ":$PYTHONPATH:" in *"_triton_patch_shim:"*) echo shim-on-path;; esac'
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    line, shim = result.stdout.strip().splitlines()
    mode, ns_default, forward = line.split("|")
    assert mode == "read"
    assert ns_default == "ns"  # AOT namespace defaults to basename(MCORE_CACHE_DIR)
    assert "MCORE_CACHE_DIR" in forward and "PYTHONPATH" in forward
    assert shim == "shim-on-path"


def test_namespace_layout_resolves(tmp_path):
    layout = "consolidated"
    ns = tmp_path / "ns_layout"
    cons = ns / layout
    cons.mkdir(parents=True)
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "artifact").write_text("x")
    with tarfile.open(cons / "inductor.tar.zst", "w:gz") as tar:
        tar.add(payload, arcname=".")
    local = tmp_path / f"local_{layout.replace('/', '_')}"
    env = _env(ns, local, scopes="inductor")
    result = _probe(env)
    assert result.returncode == 0
    assert (local / "inductor" / "artifact").exists()


def _bare_env(ns: Path, mode: str, jid=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MCORE_CACHE")}
    env.pop("NODE_CACHE_BASE", None)
    env["PYTHONPATH"] = str(SHIM_DIR)
    env["MCORE_CACHE_DIR"] = str(ns)
    env["MCORE_CACHE_MODE"] = mode
    if jid is not None:
        env["SLURM_JOB_ID"] = jid
    return env


def test_write_mode_defaults_to_durable_namespace_dir(tmp_path):
    """No MCORE_CACHE_LOCAL: write mode lands under
    <ns>/cache_write/_node/node-<nodeid> (logical rank, no job id) so shards
    stay rank-addressable and consolidate_cache.sh consumes them offline."""
    ns = tmp_path / "ns_durable"
    ns.mkdir()
    env = _bare_env(ns, "write")
    env["SLURM_NODEID"] = "7"
    result = _probe(env, "os.environ['TRITON_CACHE_DIR']")
    assert result.returncode == 0
    assert result.stdout == str(ns / "cache_write" / "_node" / "node-7" / "triton"), result.stdout
    # outside SLURM the fallback is the hostname, still node- prefixed
    result = _probe(_bare_env(ns, "write"), "os.environ['TRITON_CACHE_DIR']")
    node_dir = Path(result.stdout).parent
    assert node_dir.parent == ns / "cache_write" / "_node" and node_dir.name.startswith("node-")
    assert "_manual" not in node_dir.name  # no job id in the dir name


def test_new_seed_job_wipes_previous_jobs_leftovers(tmp_path):
    ns = tmp_path / "ns_wipe"
    ns.mkdir()
    r1 = _probe(_bare_env(ns, "write", jid="111"), "os.environ['TRITON_CACHE_DIR']")
    assert r1.returncode == 0
    stale = Path(r1.stdout) / "stale_from_job_111.txt"
    stale.write_text("old")
    # same job id -> reused, file survives
    r2 = _probe(_bare_env(ns, "write", jid="111"), "os.environ['TRITON_CACHE_DIR']")
    assert r2.returncode == 0 and stale.exists()
    # new job id -> wiped
    r3 = _probe(_bare_env(ns, "write", jid="222"), "os.environ['TRITON_CACHE_DIR']")
    assert r3.returncode == 0 and r3.stdout == r1.stdout
    assert not stale.exists()


def test_enable_sh_write_mode_requires_cache_dir():
    """Seeding has nowhere to land without a namespace; image caches are read-only."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"unset MCORE_CACHE_DIR; export MCORE_CACHE_MODE=write; source {PC_DIR}/enable.sh",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "MCORE_CACHE_DIR" in result.stderr


def test_enable_sh_without_cache_dir_selects_image_source():
    script = (
        f"unset MCORE_CACHE_DIR; source {PC_DIR}/enable.sh && "
        'echo "$MCORE_CACHE_MODE|$MCORE_CACHE_IMAGE_DIR|$MCORE_CUTE_DSL_AOT_NS_FROM"'
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "source=image" in result.stdout
    mode, image_dir, ns_from = result.stdout.strip().splitlines()[-1].split("|")
    assert mode == "read" and image_dir == "/opt/mcore_cache"
    # the shim recovers the AOT namespace from the cache's marker, so the default is a placeholder
    assert ns_from == "default"


def test_enable_sh_write_mode_exports_aot(tmp_path):
    """Write mode must arm CuTe AOT export with the namespace-derived key,
    so seeds capture the .o objects into the durable cutedsl scope."""
    ns = tmp_path / "ns_w"
    ns.mkdir()
    script = (
        f"export MCORE_CACHE_DIR={ns} MCORE_CACHE_MODE=write && "
        f"source {PC_DIR}/enable.sh >/dev/null && "
        'echo "$MCORE_CUTE_DSL_AOT_CACHE|$MCORE_CUTE_DSL_AOT_NAMESPACE"'
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1|ns_w"


@pytest.mark.parametrize("mode", ["read", "write"])
@pytest.mark.parametrize("override", [None, "0"])
def test_enable_sh_arms_inductor_recording_and_replay(tmp_path, mode, override):
    env = {key: value for key, value in os.environ.items() if not key.startswith("MCORE_")}
    env.update(MCORE_CACHE_DIR=str(tmp_path), MCORE_CACHE_MODE=mode)
    if override is not None:
        env["MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE"] = override
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" >/dev/null; printf "%s" "$MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE"',
            "cache-test",
            str(PC_DIR / "enable.sh"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == (override if override is not None else "1")


def _make_ns_with_marker(tmp_path: Path, marker_ns: str) -> Path:
    """Namespace whose consolidated cutedsl tarball embeds an AOT marker."""
    ns = tmp_path / "ns_marker"
    (ns / "consolidated").mkdir(parents=True)
    payload = tmp_path / "cutedsl_payload"
    (payload / "aot_objects").mkdir(parents=True)
    (payload / "aot_objects" / "deadbeef.o").write_bytes(b"obj")
    (payload / ".aot_namespace").write_text(marker_ns)
    with tarfile.open(ns / "consolidated" / "cutedsl.tar.zst", "w:gz") as tar:
        tar.add(payload, arcname=".")
    return ns


def test_seed_write_mode_embeds_aot_marker(tmp_path):
    ns = tmp_path / "ns_seed"
    ns.mkdir()
    env = _bare_env(ns, "write", jid="77")
    env["SLURM_NODEID"] = "0"
    env["MCORE_CUTE_DSL_AOT_NAMESPACE"] = "my-seed-key"
    result = _probe(env, "os.environ['CUTE_DSL_CACHE_DIR']")
    assert result.returncode == 0
    marker = ns / "cache_write" / "_node" / "node-0" / "cutedsl" / ".aot_namespace"
    assert marker.read_text() == "my-seed-key"


def test_reader_prefers_embedded_marker_over_default(tmp_path):
    """A cutedsl tarball moved between namespaces must keep hitting: the
    marker overrides the reader's basename-derived default."""
    ns = _make_ns_with_marker(tmp_path, "original-seed-key")
    env = _bare_env(ns, "read")
    env["MCORE_CACHE_SCOPES"] = "cutedsl"
    env["MCORE_CACHE_LOCAL"] = str(tmp_path / "local_m")
    env["MCORE_CUTE_DSL_AOT_NAMESPACE"] = "ns_marker"  # what enable.sh would default
    env["MCORE_CUTE_DSL_AOT_NS_FROM"] = "default"
    result = _probe(env, "os.environ['MCORE_CUTE_DSL_AOT_NAMESPACE']")
    assert result.returncode == 0
    assert result.stdout == "original-seed-key"


def test_reader_explicit_namespace_beats_marker(tmp_path):
    ns = _make_ns_with_marker(tmp_path, "original-seed-key")
    env = _bare_env(ns, "read")
    env["MCORE_CACHE_SCOPES"] = "cutedsl"
    env["MCORE_CACHE_LOCAL"] = str(tmp_path / "local_e")
    env["MCORE_CUTE_DSL_AOT_NAMESPACE"] = "user-pinned-key"  # no NS_FROM=default
    result = _probe(env, "os.environ['MCORE_CUTE_DSL_AOT_NAMESPACE']")
    assert result.returncode == 0
    assert result.stdout == "user-pinned-key"


def test_container_sitecustomize_runs_before_setup_and_its_site_dirs_serve_the_fallback(tmp_path):
    """Model the 26.07 deps+v8 image: the container's own sitecustomize is what
    adds the site dir holding third-party modules. Our shim must chain to it
    BEFORE its setup, so the zstd fallback (a fresh interpreter that also goes
    through our sitecustomize) can import ``zstandard`` from that dir. The fake
    module also raises the v8 stdlib's AttributeError for ``filter=``, which the
    fallback must survive."""
    site_dir = tmp_path / "container_site"
    (site_dir / "zstandard").mkdir(parents=True)
    (site_dir / "zstandard" / "__init__.py").write_text(
        "import io, tarfile\n"
        "MAGIC = b'FAKEZSTD'\n"
        "class ZstdDecompressor:\n"
        "    def stream_reader(self, raw):\n"
        "        data = raw.read()\n"
        "        assert data.startswith(MAGIC)\n"
        "        return io.BytesIO(data[len(MAGIC):])\n"
        "_orig = tarfile.TarFile.extractall\n"
        "def _v8_like(self, *args, **kwargs):\n"
        "    if 'filter' in kwargs:\n"
        "        raise AttributeError(\"module 'posixpath' has no attribute 'ALLOW_MISSING'\")\n"
        "    return _orig(self, *args, **kwargs)\n"
        "tarfile.TarFile.extractall = _v8_like\n"
    )
    container_sc = tmp_path / "container_sitecustomize.py"
    # The fake package must win over an installed zstandard, but only after
    # the shim chains to the container customization, including in the fallback.
    container_sc.write_text(
        "import site, sys\n"
        f"site_dir = {str(site_dir)!r}\n"
        "site.addsitedir(site_dir)\n"
        "sys.path.remove(site_dir)\n"
        "sys.path.insert(0, site_dir)\n"
    )

    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "kernel.bin").write_bytes(b"payload")
    plain = tmp_path / "plain.tar"
    with tarfile.open(plain, "w") as tar:
        tar.add(payload, arcname=".")
    ns = tmp_path / "ns_v8"
    (ns / "consolidated").mkdir(parents=True)
    # tar -xf cannot read this (not a real compression format) -> fallback path.
    (ns / "consolidated" / "triton.tar.zst").write_bytes(b"FAKEZSTD" + plain.read_bytes())

    local = tmp_path / "local_v8"
    env = _env(ns, local, scopes="triton")
    env["MCORE_CONTAINER_SITECUSTOMIZE"] = str(container_sc)
    env["MCORE_TRITON_GROUP_REBASE"] = "0"
    result = _probe(env, "os.environ['TRITON_CACHE_DIR']")
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(local / "triton")
    assert "triton=ok-python-zstd" in result.stderr, result.stderr
    assert (local / "triton" / "kernel.bin").read_bytes() == b"payload"
