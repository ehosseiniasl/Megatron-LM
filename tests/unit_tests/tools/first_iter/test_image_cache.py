# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""No-GPU tests for the image-resident cache: install_image_cache.py bakes a
namespace into a directory tree, and the sitecustomize shim uses that tree in
place (writable rootfs) or copies it node-locally (read-only rootfs), with no
tarball extraction and no MCORE_CACHE_DIR."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

import pytest

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
PC_DIR = REPO_ROOT / "tools" / "persistent_cache"
INSTALLER = PC_DIR / "install_image_cache.py"


def _make_namespace(tmp_path: Path) -> Path:
    """Namespace with consolidated triton (incl. a group manifest with stale
    absolute paths), inductor, and a cutedsl scope carrying an AOT marker."""
    ns = tmp_path / "ns-nano-v1"
    cons = ns / "consolidated"
    cons.mkdir(parents=True)
    triton = tmp_path / "payload_triton" / "ABCDEF"
    triton.mkdir(parents=True)
    (triton / "kernel.cubin").write_bytes(b"cubin")
    (triton / "__grp__kernel.json").write_text(
        json.dumps(
            {"child_paths": {"kernel.cubin": "/tmp/mcore_cache_999/triton/ABCDEF/kernel.cubin"}}
        )
    )
    with tarfile.open(cons / "triton.tar.zst", "w:gz") as tar:
        tar.add(triton.parent, arcname=".")
    inductor = tmp_path / "payload_inductor"
    inductor.mkdir()
    (inductor / "fx_graph.txt").write_text("graph")
    with tarfile.open(cons / "inductor.tar.zst", "w:gz") as tar:
        tar.add(inductor, arcname=".")
    cutedsl = tmp_path / "payload_cutedsl"
    (cutedsl / "aot_objects").mkdir(parents=True)
    (cutedsl / "aot_objects" / "obj.o").write_bytes(b"o")
    (cutedsl / ".aot_namespace").write_text("seed-aot-key")
    with tarfile.open(cons / "cutedsl.tar.zst", "w:gz") as tar:
        tar.add(cutedsl, arcname=".")
    (ns / "SEED_INFO").write_text("job_id=4242\nimage=base.sqsh\n")
    return ns


def _install(
    tmp_path: Path, ns: Path, scopes="triton inductor cutedsl cudnn_fe", options=()
) -> Path:
    dest = tmp_path / "image_root" / "opt" / "mcore_cache"
    result = subprocess.run(
        [
            sys.executable,
            str(INSTALLER),
            "--namespace",
            str(ns),
            "--dest",
            str(dest),
            "--tools",
            str(PC_DIR),
            "--scopes",
            scopes,
            "--source-image",
            "base.sqsh",
            *options,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"export MCORE_CACHE_IMAGE_DIR={dest}; source " in result.stdout
    assert f"source {dest}/tools/persistent_cache/enable.sh" not in result.stdout
    return dest


def _shim_env(dest: Path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MCORE_")}
    env.pop("NODE_CACHE_BASE", None)
    env["PYTHONPATH"] = str(PC_DIR / "_triton_patch_shim")
    env["MCORE_CACHE_IMAGE_DIR"] = str(dest)
    env["MCORE_CACHE_SCOPES"] = "triton inductor cutedsl cudnn_fe"
    env["MCORE_CUTE_DSL_AOT_NAMESPACE"] = "image"
    env["MCORE_CUTE_DSL_AOT_NS_FROM"] = "default"
    env.update(extra)
    return env


def _probe(env, expression="os.environ.get('TRITON_CACHE_DIR', 'MISSING')"):
    return subprocess.run(
        [sys.executable, "-c", f"import os, sys; sys.stdout.write({expression})"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _runtime_defaults(env):
    result = _probe(
        env,
        "__import__('json').dumps([os.environ.get('MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU'), "
        "os.environ.get('MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE'), "
        "any(getattr(finder, 'target', None) == 'megatron.training.initialize' for finder in sys.meta_path)])",
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_installer_lays_out_scopes_and_manifest_without_tools(tmp_path):
    ns = _make_namespace(tmp_path)
    dest = _install(tmp_path, ns)
    assert (dest / "triton" / "ABCDEF" / "kernel.cubin").read_bytes() == b"cubin"
    assert (dest / "inductor" / "fx_graph.txt").exists()
    assert (dest / "cutedsl" / ".aot_namespace").read_text() == "seed-aot-key"
    assert (dest / "cudnn_fe").is_dir()  # missing scope -> empty dir, recorded as missing
    # Triton group manifests point at the INSTALL location, not the seed job's /tmp.
    manifest = json.loads((dest / "triton" / "ABCDEF" / "__grp__kernel.json").read_text())
    assert manifest["child_paths"]["kernel.cubin"] == str(
        dest / "triton" / "ABCDEF" / "kernel.cubin"
    )
    # Runtime code comes from the checkout; the image contains cache data only.
    assert not (dest / "tools").exists()
    assert not (dest / ".mcore_ready").exists()
    assert not (dest / ".setup.lock").exists()
    assert not Path(f"{dest}.setup.lock").exists()
    text = (dest / "MANIFEST").read_text()
    assert "namespace=ns-nano-v1" in text
    assert "source_image=base.sqsh" in text
    assert "scope_cudnn_fe=missing" in text
    assert "scope_triton=files:2" in text
    assert "seed_job_id=4242" in text  # SEED_INFO travels into the image
    assert "tools_commit=" in text
    assert "tools_dir=" not in text
    assert "runtime_prewarm_weighted_relu=0\n" in text
    assert "runtime_equivalent_autotune_cache=0\n" in text


@pytest.mark.parametrize("copy_image", [False, True])
@pytest.mark.parametrize(
    "options,expected",
    [
        ((), ["0", "0", False]),
        (("--prewarm-weighted-relu",), ["1", "0", True]),
        (("--equivalent-autotune-cache",), ["0", "1", False]),
        (("--prewarm-weighted-relu", "--equivalent-autotune-cache"), ["1", "1", True]),
    ],
)
def test_image_runtime_defaults_are_independent_and_arm_before_import(
    tmp_path, copy_image, options, expected
):
    ns = _make_namespace(tmp_path)
    dest = _install(tmp_path, ns, options=options)
    env = _shim_env(
        dest, MCORE_CACHE_IMAGE_COPY=str(int(copy_image)), MCORE_CACHE_LOCAL=str(tmp_path / "local")
    )
    assert _runtime_defaults(env) == expected


@pytest.mark.parametrize(
    "override", ["MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU", "MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE"]
)
def test_explicit_zero_overrides_only_its_image_default(tmp_path, override):
    ns = _make_namespace(tmp_path)
    dest = _install(
        tmp_path, ns, options=("--prewarm-weighted-relu", "--equivalent-autotune-cache")
    )
    expected = ["1", "1", True]
    if override == "MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU":
        expected[0], expected[2] = "0", False
    else:
        expected[1] = "0"
    assert _runtime_defaults(_shim_env(dest, **{override: "0"})) == expected


@pytest.mark.parametrize(
    "runtime_lines",
    [
        "runtime_prewarm_weighted_relu=1\n",  # incomplete defaults
        "runtime_prewarm_weighted_relu=1\nruntime_equivalent_autotune_cache=true\n",
        "runtime_prewarm_weighted_relu=\nruntime_equivalent_autotune_cache=1\n",
        "runtime_prewarm_weighted_relu=2\nruntime_equivalent_autotune_cache=1\n",
        "runtime_prewarm_weighted_relu=0\nruntime_prewarm_weighted_relu=1\nruntime_equivalent_autotune_cache=1\n",
    ],
)
def test_invalid_image_defaults_do_not_enable_either_feature(tmp_path, runtime_lines):
    ns = _make_namespace(tmp_path)
    dest = _install(tmp_path, ns)
    (dest / "MANIFEST").write_text("namespace=test\n" + runtime_lines)
    assert _runtime_defaults(_shim_env(dest)) == [None, None, False]
    # Malformed baked metadata must not erase an intentional caller setting.
    env = _shim_env(dest, MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU="1")
    assert _runtime_defaults(env) == ["1", None, True]


def test_old_manifest_and_unknown_metadata_do_not_enable_features(tmp_path):
    ns = _make_namespace(tmp_path)
    dest = _install(tmp_path, ns)
    (dest / "MANIFEST").write_text(
        "namespace=old\nMCORE_INDUCTOR_PREWARM_WEIGHTED_RELU=1\nruntime_arbitrary=1\n"
    )
    assert _runtime_defaults(_shim_env(dest)) == [None, None, False]


def test_defaults_are_not_applied_when_image_setup_fails(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "failed_image_setup", PC_DIR / "_triton_patch_shim" / "mcore_cache_node_setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "read_manifest", lambda _path: dict.fromkeys(module._IMAGE_RUNTIME_DEFAULTS, "1")
    )
    monkeypatch.setattr(module, "_dir_write_exec", lambda _path: False)
    monkeypatch.setattr(module, "resolve_base", lambda _mode: None)
    for variable in module._IMAGE_RUNTIME_DEFAULTS.values():
        monkeypatch.delenv(variable, raising=False)
    module._setup_image(str(tmp_path))
    assert all(variable not in os.environ for variable in module._IMAGE_RUNTIME_DEFAULTS.values())


def test_image_defaults_are_ignored_for_lustre_and_seed_writes(tmp_path):
    ns = _make_namespace(tmp_path)
    dest = _install(
        tmp_path, ns, options=("--prewarm-weighted-relu", "--equivalent-autotune-cache")
    )
    for mode in ("read", "write"):
        env = _shim_env(
            dest,
            MCORE_CACHE_DIR=str(ns),
            MCORE_CACHE_MODE=mode,
            MCORE_CACHE_LOCAL=str(tmp_path / mode),
        )
        assert _runtime_defaults(env) == [None, None, False]
    env["MCORE_CACHE_MODE"] = "read"
    env["MCORE_CACHE_PREFER_IMAGE"] = "1"
    assert _runtime_defaults(env) == ["1", "1", True]


@pytest.mark.parametrize(
    "inherited_shims", ["none", "legacy", "legacy_then_repo", "repo_then_legacy"]
)
def test_repository_enable_consumes_image_and_ignores_legacy_loader(tmp_path, inherited_shims):
    ns = _make_namespace(tmp_path)
    dest = _install(
        tmp_path, ns, options=("--prewarm-weighted-relu", "--equivalent-autotune-cache")
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("MCORE_")}
    env.pop("PYTHONPATH", None)
    env.pop("NODE_CACHE_BASE", None)
    env["MCORE_CACHE_IMAGE_DIR"] = str(dest)
    legacy_marker = tmp_path / "legacy-loader-ran"
    repo_shim = str(PC_DIR / "_triton_patch_shim")
    expected_pythonpath = repo_shim
    if inherited_shims != "none":
        legacy_tools = dest / "tools" / "persistent_cache"
        legacy_shim = legacy_tools / "_triton_patch_shim"
        legacy_shim.mkdir(parents=True)
        (legacy_tools / "enable.sh").write_text(
            'echo "legacy loader must not be sourced" >&2; return 91\n'
        )
        (legacy_shim / "sitecustomize.py").write_text(
            f"from pathlib import Path\nPath({str(legacy_marker)!r}).touch()\n"
            'raise RuntimeError("legacy loader must not run")\n'
        )
        with (dest / "MANIFEST").open("a") as stream:
            stream.write(f"tools_dir={legacy_tools}\n")
        # An old shim must lose even when the selected repository already appears later.
        env["PYTHONPATH"] = {
            "legacy": str(legacy_shim),
            "legacy_then_repo": f"{legacy_shim}:{repo_shim}",
            "repo_then_legacy": f"{repo_shim}:{legacy_shim}",
        }[inherited_shims]
        expected_pythonpath = (
            env["PYTHONPATH"]
            if inherited_shims == "repo_then_legacy"
            else f"{repo_shim}:{env['PYTHONPATH']}"
        )
    program = (
        "import json, os, sitecustomize, mcore_cache_node_setup; print(json.dumps(["
        "os.environ['MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU'], "
        "os.environ['MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE'], "
        "os.environ['TRITON_CACHE_DIR'], sitecustomize.__file__, mcore_cache_node_setup.__file__, "
        "os.environ['PYTHONPATH']]))"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" >/dev/null && source "$1" >/dev/null && exec "$2" -c "$3"',
            "image-consumer",
            str(PC_DIR / "enable.sh"),
            sys.executable,
            program,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "1",
        "1",
        str(dest / "triton"),
        str(PC_DIR / "_triton_patch_shim" / "sitecustomize.py"),
        str(PC_DIR / "_triton_patch_shim" / "mcore_cache_node_setup.py"),
        expected_pythonpath,  # The second source must leave the selected-first path unchanged.
    ]
    assert not legacy_marker.exists()
    assert "legacy loader must not" not in result.stderr
    assert (dest / "triton" / "ABCDEF" / "kernel.cubin").read_bytes() == b"cubin"


def test_installer_refuses_to_overwrite_without_force(tmp_path):
    ns = _make_namespace(tmp_path)
    dest = _install(tmp_path, ns)
    result = subprocess.run(
        [
            sys.executable,
            str(INSTALLER),
            "--namespace",
            str(ns),
            "--dest",
            str(dest),
            "--tools",
            str(PC_DIR),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0 and "already holds" in result.stderr


def test_shim_uses_writable_image_cache_in_place(tmp_path):
    """Writable rootfs (pyxis default here): pin straight at the baked dirs,
    never copy, never wipe, and recover the AOT namespace from the marker."""
    ns = _make_namespace(tmp_path)
    dest = _install(tmp_path, ns)
    env = _shim_env(dest, SLURM_JOB_ID="1001")
    result = _probe(
        env,
        "','.join([os.environ['TRITON_CACHE_DIR'], os.environ['MCORE_CUTE_DSL_AOT_NAMESPACE']])",
    )
    assert result.returncode == 0, result.stderr
    triton_dir, aot_ns = result.stdout.split(",")
    assert triton_dir == str(dest / "triton")
    assert aot_ns == "seed-aot-key"
    assert "image cache in place" in result.stderr
    assert (dest / "triton" / "ABCDEF" / "kernel.cubin").exists()
    assert (dest / ".mcore_ready").read_text() == "image 1001"
    # A later job (fresh container in practice, stale sentinel here) must NOT wipe the cache.
    second = _probe(_shim_env(dest, SLURM_JOB_ID="1002"))
    assert second.returncode == 0 and second.stdout == str(dest / "triton")
    assert (dest / "triton" / "ABCDEF" / "kernel.cubin").exists()
    assert (dest / ".mcore_ready").read_text() == "image 1002"


def test_shim_copies_image_cache_when_in_place_use_is_disabled(tmp_path):
    """MCORE_CACHE_IMAGE_COPY=1 forces the read-only-rootfs path deterministically
    (a chmod-based check would pass vacuously as root)."""
    ns = _make_namespace(tmp_path)
    dest = _install(tmp_path, ns)
    local = tmp_path / f"local_{uuid.uuid4().hex[:8]}"
    env = _shim_env(
        dest, SLURM_JOB_ID="2001", MCORE_CACHE_IMAGE_COPY="1", MCORE_CACHE_LOCAL=str(local)
    )
    result = _probe(env, "os.environ['TRITON_CACHE_DIR']")
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(local / "triton")
    assert "image cache copied" in result.stderr and "triton=ok" in result.stderr
    assert (local / "triton" / "ABCDEF" / "kernel.cubin").read_bytes() == b"cubin"
    assert (local / "inductor" / "fx_graph.txt").exists()
    # the copy was rebased to ITS location
    manifest = json.loads((local / "triton" / "ABCDEF" / "__grp__kernel.json").read_text())
    assert manifest["child_paths"]["kernel.cubin"] == str(
        local / "triton" / "ABCDEF" / "kernel.cubin"
    )
    assert (local / ".mcore_ready").read_text() == "image-copy 2001"
    # the image tree is untouched by the copy path
    assert not (dest / ".mcore_ready").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_shim_falls_back_to_copy_on_read_only_image_dir(tmp_path):
    ns = _make_namespace(tmp_path)
    dest = _install(tmp_path, ns)
    local = tmp_path / "local_ro"
    dest.chmod(0o555)
    try:
        env = _shim_env(dest, SLURM_JOB_ID="3001", MCORE_CACHE_LOCAL=str(local))
        result = _probe(env, "os.environ['TRITON_CACHE_DIR']")
    finally:
        dest.chmod(0o755)
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(local / "triton")
    assert "image cache copied" in result.stderr


def test_lustre_namespace_wins_unless_image_is_preferred(tmp_path):
    ns = _make_namespace(tmp_path)
    dest = _install(tmp_path, ns)
    local = tmp_path / "local_lustre"
    env = _shim_env(
        dest, SLURM_JOB_ID="4001", MCORE_CACHE_DIR=str(ns), MCORE_CACHE_LOCAL=str(local)
    )
    result = _probe(env, "os.environ['TRITON_CACHE_DIR']")
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(local / "triton")  # extracted from the namespace tarballs
    assert "extracted mode=read" in result.stderr
    env["MCORE_CACHE_PREFER_IMAGE"] = "1"
    result = _probe(env, "os.environ['TRITON_CACHE_DIR']")
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(dest / "triton")


def test_write_mode_ignores_image_cache(tmp_path):
    """Seeding must never land in the baked tree even when one is present."""
    ns = _make_namespace(tmp_path)
    dest = _install(tmp_path, ns)
    seed_ns = tmp_path / "seed_ns"
    seed_ns.mkdir()
    env = _shim_env(
        dest,
        SLURM_JOB_ID="5001",
        SLURM_NODEID="3",
        MCORE_CACHE_DIR=str(seed_ns),
        MCORE_CACHE_MODE="write",
    )
    result = _probe(env, "os.environ['TRITON_CACHE_DIR']")
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(seed_ns / "cache_write" / "_node" / "node-3" / "triton")


def test_inert_without_any_source(tmp_path):
    env = _shim_env(tmp_path / "no_such_dir")
    result = _probe(env)
    assert result.returncode == 0 and result.stdout == "MISSING"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_in_place_use_needs_no_writable_parent(tmp_path):
    """The in-place lock must live inside the cache dir: /opt need not be writable
    even when /opt/mcore_cache is (review finding: setup fell back to a cold start)."""
    ns = _make_namespace(tmp_path)
    parent = tmp_path / "image_root" / "opt"
    dest = _install(tmp_path, ns)  # installs under parent/mcore_cache
    assert dest.parent == parent
    parent.chmod(0o555)
    try:
        result = _probe(_shim_env(dest, SLURM_JOB_ID="6001"), "os.environ['TRITON_CACHE_DIR']")
    finally:
        parent.chmod(0o755)
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(dest / "triton")
    assert "image cache in place" in result.stderr
    assert (dest / ".setup.lock").exists() and not Path(f"{dest}.setup.lock").exists()


def test_legacy_image_enable_sh_still_derives_image_dir_from_its_location(tmp_path):
    """Keep compatibility for existing images that already contain a loader."""
    ns = _make_namespace(tmp_path)
    dest = _install(tmp_path, ns)
    legacy_tools = dest / "tools" / "persistent_cache"
    legacy_tools.mkdir(parents=True)
    shutil.copy2(PC_DIR / "enable.sh", legacy_tools / "enable.sh")
    script = f"unset MCORE_CACHE_DIR MCORE_CACHE_IMAGE_DIR; source {dest}/tools/persistent_cache/enable.sh >/dev/null && echo \"$MCORE_CACHE_IMAGE_DIR\""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(dest)
    # the checkout's own enable.sh (no MANIFEST beside it) keeps the /opt default
    script = f"unset MCORE_CACHE_DIR MCORE_CACHE_IMAGE_DIR; source {PC_DIR}/enable.sh >/dev/null && echo \"$MCORE_CACHE_IMAGE_DIR\""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.stdout.strip() == "/opt/mcore_cache"


@pytest.mark.parametrize("symlink", [False, True])
def test_installer_removes_only_stale_bundled_tools(tmp_path, symlink):
    ns = _make_namespace(tmp_path)
    dest = tmp_path / "image_root" / "opt" / "mcore_cache"
    tools_parent = dest / "tools"
    tools_parent.mkdir(parents=True)
    sibling = tools_parent / "keep.txt"
    sibling.write_text("unrelated image tool")
    external = tmp_path / "external-tools"
    external.mkdir()
    (external / "enable.sh").write_text("keep external source")
    stale = tools_parent / "persistent_cache"
    if symlink:
        stale.symlink_to(external, target_is_directory=True)
    else:
        stale.mkdir()
        (stale / "enable.sh").write_text("obsolete loader")
    assert _install(tmp_path, ns) == dest
    assert not stale.exists() and not stale.is_symlink()
    assert sibling.read_text() == "unrelated image tool"
    assert (external / "enable.sh").read_text() == "keep external source"


def test_installer_self_check_uses_supplied_repository_tools(tmp_path):
    ns = _make_namespace(tmp_path)
    tools = tmp_path / "selected-checkout" / "tools" / "persistent_cache"
    shutil.copytree(PC_DIR, tools, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    marker = tmp_path / "selected-shim-used"
    shim = tools / "_triton_patch_shim" / "sitecustomize.py"
    with shim.open("a") as stream:
        stream.write(f"\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('selected')\n")
    dest = _install(tmp_path, ns, options=("--tools", str(tools)))
    assert marker.read_text() == "selected"
    assert not (dest / "tools").exists()


def test_installer_does_not_remove_loaders_through_a_symlinked_parent(tmp_path):
    ns = _make_namespace(tmp_path)
    dest = tmp_path / "image_root" / "opt" / "mcore_cache"
    dest.mkdir(parents=True)
    external = tmp_path / "external-tools"
    (external / "persistent_cache").mkdir(parents=True)
    loader = external / "persistent_cache" / "enable.sh"
    loader.write_text("external loader must remain")
    (dest / "tools").symlink_to(external, target_is_directory=True)
    result = subprocess.run(
        [
            sys.executable,
            str(INSTALLER),
            "--namespace",
            str(ns),
            "--dest",
            str(dest),
            "--tools",
            str(PC_DIR),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "symlinked tools directory" in result.stderr
    assert loader.read_text() == "external loader must remain"
