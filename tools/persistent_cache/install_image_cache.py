#!/usr/bin/env python3
"""Install a sealed cache namespace into a container filesystem (image bake).

Turns ``<namespace>/consolidated/<scope>.tar.zst`` into pre-extracted scope
directories under ``--dest`` (default /opt/mcore_cache), rebases Triton's
absolute group manifests to that final location, and writes a MANIFEST the
repository runtime shim uses to recognise and key the cache. Runtime tools
are not bundled: source the matching checkout's
``tools/persistent_cache/enable.sh`` before training. For a custom ``--dest``,
also export ``MCORE_CACHE_IMAGE_DIR`` to that location.

Run inside the container whose filesystem is being saved (see
bake_cache_image.sh), or anywhere for tests. Stdlib only.

    python3 install_image_cache.py --namespace /path/to/cache-run/cache \
        --dest /opt/mcore_cache [--tools tools/persistent_cache] \
        [--source-image base.sqsh]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SCOPES = "triton inductor cuda_ptx cutedsl hybrid_ep cudnn_fe megatron_fused_kernels"


def _load_node_setup():
    """Import the shim module by path (it is not a package)."""
    path = HERE / "_triton_patch_shim" / "mcore_cache_node_setup.py"
    spec = importlib.util.spec_from_file_location("mcore_cache_node_setup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_files(root: Path) -> int:
    return sum(len(files) for _dir, _sub, files in os.walk(root))


def _git_commit(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _self_check(dest: Path, shim_dir: Path, expect_nccl_topology: bool = False) -> None:
    """Start a bare interpreter with the shim on PYTHONPATH and verify it pins
    the caches at ``dest`` (the exact runtime path, minus the training stack)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MCORE_CACHE")}
    env.pop("NODE_CACHE_BASE", None)
    for name in (
        "MCORE_NCCL_TOPO_CACHE",
        "NCCL_TOPO_FILE",
        "NCCL_TOPO_DUMP_FILE",
        "NCCL_GRAPH_FILE",
        "NCCL_GRAPH_DUMP_FILE",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(shim_dir)
    env["MCORE_CACHE_IMAGE_DIR"] = str(dest)
    env["MCORE_CACHE_MODE"] = "read"
    env["MCORE_TRITON_GROUP_REBASE"] = "0"  # already rebased at install; keep the check pure
    env["SLURM_JOB_ID"] = "install-self-check"
    program = "import os; print(os.environ['TRITON_CACHE_DIR'])"
    expected = str(dest / "triton")
    if expect_nccl_topology:
        program += "; print(os.environ['NCCL_TOPO_FILE'])"
        expected += "\n" + str(dest / "nccl_topo" / "topo_template.xml")
    result = subprocess.run(
        [sys.executable, "-c", program], env=env, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0 or result.stdout.strip() != expected:
        raise SystemExit(
            f"self-check failed: rc={result.returncode} stdout={result.stdout!r} "
            f"stderr={result.stderr.strip()}"
        )
    # The check leaves a sentinel/lock behind; the image must not ship them.
    for stale in (dest / ".mcore_ready", dest / ".setup.lock", Path(f"{dest}.setup.lock")):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
    print(f"[IMAGE CACHE] self-check ok: shim pinned TRITON_CACHE_DIR={dest / 'triton'}")
    if expect_nccl_topology:
        print(
            f"[IMAGE CACHE] self-check ok: shim pinned NCCL_TOPO_FILE={dest / 'nccl_topo' / 'topo_template.xml'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--namespace", required=True, help="cache namespace dir holding consolidated/"
    )
    parser.add_argument(
        "--dest", default="/opt/mcore_cache", help="install location inside the image"
    )
    parser.add_argument(
        "--tools",
        default=str(HERE),
        help="repository tools/persistent_cache dir for install helpers and the runtime self-check; not bundled",
    )
    parser.add_argument("--scopes", default=os.environ.get("MCORE_CACHE_SCOPES") or DEFAULT_SCOPES)
    parser.add_argument("--source-image", default="", help="base image name, recorded in MANIFEST")
    parser.add_argument(
        "--nccl-topo-file", help="optional NCCL topology XML to validate on the bake node"
    )
    parser.add_argument(
        "--nccl-topo-namespace", help="matching producer label for the optional topology"
    )
    parser.add_argument(
        "--prewarm-weighted-relu",
        action="store_true",
        help="enable weighted-ReLU prewarm by default for this image",
    )
    parser.add_argument(
        "--equivalent-autotune-cache",
        action="store_true",
        help="enable score-equivalent Inductor autotune reuse by default for this image",
    )
    parser.add_argument(
        "--training-repo", help="matching checkout used to finalize C++ extensions at --dest"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing install at --dest"
    )
    parser.add_argument("--no-self-check", action="store_true")
    args = parser.parse_args()

    if (args.nccl_topo_file is None) != (args.nccl_topo_namespace is None):
        parser.error("--nccl-topo-file and --nccl-topo-namespace must be supplied together")
    topology_source = None
    if args.nccl_topo_file is not None:
        if not args.nccl_topo_file or not Path(args.nccl_topo_file).is_file():
            parser.error("--nccl-topo-file must name a readable topology XML file")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.nccl_topo_namespace):
            parser.error(
                "--nccl-topo-namespace must be an alphanumeric label with optional '.', '_' or '-'"
            )
        topology_source = Path(args.nccl_topo_file).resolve()

    namespace = Path(args.namespace).resolve()
    dest = Path(args.dest)
    tools = Path(args.tools).resolve()
    scopes = args.scopes.split()
    if "nccl_topo" in scopes and topology_source is None:
        parser.error("the nccl_topo scope requires --nccl-topo-file and --nccl-topo-namespace")
    kernel_scopes = [scope for scope in scopes if scope != "nccl_topo"]
    scopes = kernel_scopes + (["nccl_topo"] if topology_source is not None else [])
    node_setup = _load_node_setup()

    consolidated = namespace / "consolidated"
    if not consolidated.is_dir():
        raise SystemExit(f"{namespace} has no consolidated/ (run consolidate_cache.sh first)")
    if (dest / "MANIFEST").exists():
        if not args.force:
            raise SystemExit(f"{dest} already holds an image cache (use --force to replace)")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    lines = [
        "# first-iteration kernel cache installed by tools/persistent_cache/install_image_cache.py",
        f"namespace={namespace.name}",
        f"source_namespace={namespace}",
        f"source_image={args.source_image}",
        f"created_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"tools_commit={_git_commit(tools)}",
        f"scopes={' '.join(scopes)}",
        f"runtime_prewarm_weighted_relu={int(args.prewarm_weighted_relu)}",
        f"runtime_equivalent_autotune_cache={int(args.equivalent_autotune_cache)}",
    ]
    if topology_source is not None:
        topology_dir = dest / "nccl_topo"
        topology_dir.mkdir(parents=True, exist_ok=True)
        topology_xml = topology_dir / "topo_template.xml"
        # This helper validates the XML against the current compute node's PCI
        # devices. Never use its offline/skip-validation option for an image bake.
        subprocess.run(
            [
                sys.executable,
                str(tools / "prepare_nccl_topology_template.py"),
                "--source",
                str(topology_source),
                "--output",
                str(topology_xml),
                "--manifest",
                str(topology_dir / "topo_template.manifest.json"),
                "--namespace",
                args.nccl_topo_namespace,
            ],
            check=True,
        )
        lines.extend(
            [
                "runtime_nccl_topo_cache=1",
                f"nccl_topo_namespace={args.nccl_topo_namespace}",
                f"scope_nccl_topo=files:{_count_files(topology_dir)} sha256:{_sha256(topology_xml)}",
            ]
        )
    seed_info = namespace / "SEED_INFO"
    if seed_info.is_file():
        for raw in seed_info.read_text().splitlines():
            raw = raw.strip()
            if raw and not raw.startswith("#") and "=" in raw:
                lines.append(f"seed_{raw}")

    for scope in kernel_scopes:
        scope_dir = dest / scope
        scope_dir.mkdir(parents=True, exist_ok=True)
        tarball = consolidated / f"{scope}.tar.zst"
        if not tarball.is_file():
            print(f"[IMAGE CACHE] {scope}: no tarball (scope will run cold)")
            lines.append(f"scope_{scope}=missing")
            continue
        outcome = node_setup._extract_one(str(tarball), str(scope_dir))
        if outcome.startswith("failed"):
            raise SystemExit(f"{scope}: extraction failed: {outcome}")
        files = _count_files(scope_dir)
        print(f"[IMAGE CACHE] {scope}: {outcome} files={files} from {tarball.name}")
        lines.append(f"scope_{scope}=files:{files} sha256:{_sha256(tarball)}")

    # Triton's __grp__*.json manifests carry absolute child paths from the seed
    # job; point them at the install location so the warm run sees full groups.
    rebase = tools / "rebase_triton_groups.sh"
    triton_dir = dest / "triton"
    if rebase.is_file() and triton_dir.is_dir() and _count_files(triton_dir):
        result = subprocess.run(
            ["bash", str(rebase), str(triton_dir)], capture_output=True, text=True
        )
        print(result.stdout.strip() or result.stderr.strip())
        if result.returncode != 0:
            raise SystemExit("triton group rebase failed")

    if args.training_repo and "megatron_fused_kernels" in scopes:
        # Copying Ninja outputs changes their absolute source paths. Finish the
        # build here, before saving the image, and verify a fresh process reuses it.
        training_repo = Path(args.training_repo).resolve()
        subprocess.run(
            [
                sys.executable,
                str(tools / "finalize_image_extensions.py"),
                "--dest",
                str(dest),
                "--training-repo",
                str(training_repo),
            ],
            check=True,
        )
        lines.extend(
            [
                f"extensions_training_commit={_git_commit(training_repo)}",
                "extensions_finalization=extensions-finalization.json",
                f"extensions_final_files={_count_files(dest / 'megatron_fused_kernels')}",
            ]
        )

    # A retry of an older partial install may still contain bundled loaders.
    # Remove only that directory; keep neighboring image tools untouched.
    tools_dest = dest / "tools" / "persistent_cache"
    if tools_dest.exists() or tools_dest.is_symlink():
        if tools_dest.parent.is_symlink():
            raise SystemExit(
                "refusing to remove bundled loaders through a symlinked tools directory"
            )
        if tools_dest.is_symlink() or not tools_dest.is_dir():
            tools_dest.unlink()
        else:
            shutil.rmtree(tools_dest)

    (dest / "MANIFEST").write_text("\n".join(lines) + "\n")
    if not args.no_self_check:
        _self_check(
            dest, tools / "_triton_patch_shim", expect_nccl_topology=topology_source is not None
        )

    total = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _dirs, files in os.walk(dest)
        for name in files
        if not os.path.islink(os.path.join(root, name))
    )
    print(
        f"[IMAGE CACHE] installed {namespace.name} at {dest} "
        f"({total / (1 << 20):.0f} MiB, {time.monotonic() - started:.1f}s)"
    )
    image_env = (
        f"export MCORE_CACHE_IMAGE_DIR={shlex.quote(str(dest))}; "
        if str(dest) != "/opt/mcore_cache"
        else ""
    )
    print(
        f"[IMAGE CACHE] consume with: {image_env}source {shlex.quote(str(tools / 'enable.sh'))} (before training)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
