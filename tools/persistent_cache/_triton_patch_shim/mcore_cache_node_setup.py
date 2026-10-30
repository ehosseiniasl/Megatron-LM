"""Standalone per-node cache setup, run from sitecustomize before any import.

The launcher sources tools/persistent_cache/enable.sh (host side or inside the
container) and every Python worker lands here through the PYTHONPATH shim.
Before torch/triton/cute/deep_ep are importable, this module points the compile
caches (Triton, Inductor, CUDA PTX, CuTe DSL, HybridEP JIT, cuDNN frontend,
fused-kernel extensions) at a warm cache. Two cache sources:

  image   Pre-extracted scope directories baked into the container image by
          tools/persistent_cache/install_image_cache.py (MCORE_CACHE_IMAGE_DIR,
          default /opt/mcore_cache, identified by its MANIFEST). The container
          rootfs is already unpacked on node-local storage, so when it is
          writable the cache is used IN PLACE: nothing is copied or extracted.
          On a read-only rootfs the scope dirs are copied once per node to
          node-local scratch instead (still no Lustre traffic, no tarballs).

  lustre  A namespace directory MCORE_CACHE_DIR holding
          consolidated/<scope>.tar.zst (sealed by consolidate_cache.sh). The
          first process on each node extracts the tarballs to node-local
          scratch under an exclusive lock. This is also the seeding path:
          MCORE_CACHE_MODE=write pins the caches to the durable per-node dir
          <ns>/cache_write/_node/node-<SLURM_NODEID>/ so the compilers populate
          it as a side effect of a bounded run.

Source selection in read mode: MCORE_CACHE_DIR set -> lustre, unless
MCORE_CACHE_PREFER_IMAGE=1 and an image cache is present; MCORE_CACHE_DIR
unset -> image when present, else inert. Write mode always needs
MCORE_CACHE_DIR.

Everything is fail-open: any error leaves the process on the native
(cold-compile) path with one stderr warning. Stdlib only; must never import
torch/triton/cute/deep_ep.
"""

import errno
import fcntl
import importlib.util
import os
import shutil
import subprocess
import sys
import time

_SCOPE_ENV = {
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "CUDA_CACHE_PATH": "cuda_ptx",
    "CUTE_DSL_CACHE_DIR": "cutedsl",
    "HYBRID_EP_CACHE_DIR": "hybrid_ep",
    "HYBRID_EP_JIT_DIR": "hybrid_ep",
    "CUDNN_FRONTEND_CACHE_DIR": "cudnn_fe",
    "TORCH_EXTENSIONS_DIR": "megatron_fused_kernels",
}
_DEFAULT_SCOPES = "triton inductor cuda_ptx cutedsl hybrid_ep cudnn_fe megatron_fused_kernels"
_DEFAULT_IMAGE_DIR = "/opt/mcore_cache"
_MANIFEST = "MANIFEST"
_IMAGE_RUNTIME_DEFAULTS = {
    "runtime_prewarm_weighted_relu": "MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU",
    "runtime_equivalent_autotune_cache": "MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE",
}
_TOPO_RUNTIME_KEY = "runtime_nccl_topo_cache"
_TOPO_ENV = ("NCCL_TOPO_FILE", "NCCL_TOPO_DUMP_FILE", "NCCL_GRAPH_FILE", "NCCL_GRAPH_DUMP_FILE")
_READY = ".mcore_ready"
_LOCK_TIMEOUT_S = 600


def _warn(msg):
    print(f"[CACHE NODE SETUP] {msg}", file=sys.stderr, flush=True)


def _probe_write_exec(directory, probe_dir):
    """True when `directory` exists (creating it) and `probe_dir` takes a runnable file."""
    try:
        os.makedirs(directory, exist_ok=True)
        probe = os.path.join(
            probe_dir, f".pc_probe.{os.environ.get('SLURM_JOB_ID', 'manual')}.{os.getpid()}"
        )
        with open(probe, "w") as fh:
            fh.write("#!/bin/sh\nexit 0\n")
        os.chmod(probe, 0o700)
        result = subprocess.run([probe], capture_output=True)
        os.unlink(probe)
        return result.returncode == 0
    except OSError:
        return False


def _can_write_exec(directory):
    """True when `directory` is creatable, writable, and exec-mountable."""
    return _probe_write_exec(directory, os.path.dirname(directory))


def _dir_write_exec(directory):
    """True when an EXISTING directory itself takes writes and runs a file."""
    return os.path.isdir(directory) and _probe_write_exec(directory, directory)


def resolve_base(mode="read"):
    """Pick the cache base: override, else mode-dependent default.

    read: ephemeral node-local scratch (/dev/shm -> /tmp -> Lustre), keyed by
    job id — extraction is per job by design.
    write: the DURABLE per-node dir <ns>/cache_write/_node/node-<SLURM_NODEID>
    (no job id), so the seed survives the job and consolidate_cache.sh runs
    offline from any host later. The LOGICAL node id (not the hostname) keeps
    each shard rank-addressable: seeds run without spares, so SLURM_NODEID
    equals the worker group rank and a shard can also be consumed per-rank
    (node-sharded style) without consolidation. Hostname is the fallback
    outside SLURM. One seed job per namespace at a time; stale content from a
    PREVIOUS seed job is wiped by the sentinel's job-id check in _once_per_node.
    """
    override = os.environ.get("MCORE_CACHE_LOCAL")
    if override:
        if _can_write_exec(override):
            return override
        _warn(f"MCORE_CACHE_LOCAL={override} is not write+exec; probing defaults")
    jid = os.environ.get("SLURM_JOB_ID", "manual")
    host = os.uname().nodename.split(".")[0]
    if mode == "write":
        node_id = os.environ.get("SLURM_NODEID", host)
        durable = os.path.join(
            os.environ.get("MCORE_CACHE_DIR", "/tmp"), "cache_write", "_node", f"node-{node_id}"
        )
        return durable if _can_write_exec(durable) else None
    # Read-mode Lustre fallback lives under _scratch/, NOT _node/: _node/ is
    # reserved for durable seed shards, and consolidate_cache.sh treats any
    # scope-bearing dir there as a seed source.
    candidates = [
        f"/dev/shm/mcore_cache_{jid}",
        f"/tmp/mcore_cache_{jid}",
        os.path.join(os.environ.get("MCORE_CACHE_DIR", "/tmp"), "_scratch", f"{host}_{jid}"),
    ]
    for candidate in candidates:
        if _can_write_exec(candidate):
            return candidate
    return None


# --------------------------------------------------------------------------- lustre
def _consolidated_tar(cache_dir, scope):
    """Resolve the sealed tarball consolidate_cache.sh wrote for `scope`."""
    path = os.path.join(cache_dir, "consolidated", f"{scope}.tar.zst")
    return path if os.path.isfile(path) else None


_PY_ZSTD_EXTRACT = r"""
import sys, tarfile
import zstandard
tarball, dest = sys.argv[1], sys.argv[2]
with open(tarball, "rb") as raw:
    with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
        # "r|" streams from a non-seekable object.
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            try:
                archive.extractall(dest, filter="tar")
            except Exception:
                # filter= predates 3.12 on some builds (TypeError), and the 26.07
                # deps+v8 stdlib raises AttributeError (posixpath.ALLOW_MISSING)
                # from inside the filter. Re-open: a "r|" stream cannot rewind.
                with open(tarball, "rb") as raw2:
                    with zstandard.ZstdDecompressor().stream_reader(raw2) as stream2:
                        with tarfile.open(fileobj=stream2, mode="r|") as archive2:
                            archive2.extractall(dest)
"""


def _extract_without_zstd_cli(tarball, dest):
    """Extract a .tar.zst with Python's zstandard. Returns None on success, else the reason.

    Runs in a FRESH interpreter, not in-process: this function is reached from
    sitecustomize, i.e., while the calling interpreter is still starting up, and
    from a worker thread. In that state `import zstandard` raised
    ModuleNotFoundError inside the 26.07 training image although the module is
    installed and imports fine from a started interpreter (aws-cmh job 3633302:
    every scope reported "zstandard unavailable" and the run went cold). A
    subprocess sees the normal sys.path; the shim's re-entrancy guard keeps its
    own sitecustomize inert there.
    """
    result = subprocess.run(
        [sys.executable, "-c", _PY_ZSTD_EXTRACT, tarball, dest], capture_output=True, text=True
    )
    if result.returncode == 0:
        return None
    reason = (result.stderr.strip().splitlines() or ["unknown error"])[-1]
    return f"python zstandard: {reason}"


def _extract_one(tarball, dest):
    """Extract one scope tarball into dest. Returns 'ok', 'ok-python-zstd' or 'failed: ...'."""
    os.makedirs(dest, exist_ok=True)
    # tar -xf auto-detects zstd or gzip, matching the write side (lib.sh).
    result = subprocess.run(["tar", "-xf", tarball, "-C", dest], capture_output=True, text=True)
    if result.returncode == 0:
        return "ok"
    # GNU tar shells out to a `zstd` binary that some containers omit (the
    # 26.07 training images do). Python's zstandard is present there, so
    # decompress in-process rather than fail open to a cold start.
    reason = _extract_without_zstd_cli(tarball, dest)
    if reason is None:
        return "ok-python-zstd"
    return f"failed: {result.stderr.strip()} || fallback: {reason}"


def _extract_scopes(cache_dir, base, scopes):
    """Extract every scope tarball in parallel; return {scope: outcome}."""
    from concurrent.futures import ThreadPoolExecutor

    def one(scope):
        dest = os.path.join(base, scope)
        os.makedirs(dest, exist_ok=True)
        tarball = _consolidated_tar(cache_dir, scope)
        if tarball is None:
            return (scope, "missing")
        return (scope, _extract_one(tarball, dest))

    outcomes = {}
    with ThreadPoolExecutor(max_workers=max(1, len(scopes))) as pool:
        for scope, outcome in pool.map(one, scopes):
            outcomes[scope] = outcome
    return outcomes


# --------------------------------------------------------------------------- image
def image_cache_dir():
    """The configured image-cache location (existence not implied)."""
    return os.environ.get("MCORE_CACHE_IMAGE_DIR") or _DEFAULT_IMAGE_DIR


def read_manifest(image_dir):
    """Parse the installer's key=value MANIFEST; {} when absent/unreadable."""
    manifest = {}
    try:
        with open(os.path.join(image_dir, _MANIFEST)) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                # Duplicate feature defaults are ambiguous, even if the last
                # value is valid. Preserve that invalidity through later lines.
                guarded = key in _IMAGE_RUNTIME_DEFAULTS or key in (
                    _TOPO_RUNTIME_KEY,
                    "nccl_topo_namespace",
                )
                manifest[key] = None if guarded and key in manifest else value.strip()
    except (OSError, UnicodeError):
        return {}
    return manifest


def _apply_image_runtime_defaults(manifest):
    """Apply only the installer's explicit feature defaults; caller settings win."""
    if not any(key in manifest for key in _IMAGE_RUNTIME_DEFAULTS):
        return  # Older images and generic cache namespaces opt into nothing.
    if any(manifest.get(key) not in ("0", "1") for key in _IMAGE_RUNTIME_DEFAULTS):
        _warn("invalid image runtime defaults; ignoring baked feature defaults")
        return
    for key, variable in _IMAGE_RUNTIME_DEFAULTS.items():
        os.environ.setdefault(variable, manifest[key])


def image_cache_present(image_dir):
    """An image cache is a directory carrying the installer's MANIFEST."""
    return bool(image_dir) and os.path.isfile(os.path.join(image_dir, _MANIFEST))


def _copy_scopes(image_dir, base, scopes):
    """Copy every image scope dir into base in parallel; return {scope: outcome}."""
    from concurrent.futures import ThreadPoolExecutor

    def one(scope):
        src = os.path.join(image_dir, scope)
        dest = os.path.join(base, scope)
        os.makedirs(dest, exist_ok=True)
        if not os.path.isdir(src):
            return (scope, "missing")
        # cp -a keeps symlinks/modes and is markedly faster than a Python walk
        # for the tens of thousands of small Inductor/Triton files.
        result = subprocess.run(["cp", "-a", f"{src}/.", dest], capture_output=True, text=True)
        if result.returncode == 0:
            return (scope, "ok")
        try:
            shutil.copytree(src, dest, symlinks=True, dirs_exist_ok=True)
        except Exception as exc:
            return (scope, f"failed: {result.stderr.strip()} || fallback: {exc}")
        return (scope, "ok-copytree")

    outcomes = {}
    with ThreadPoolExecutor(max_workers=max(1, len(scopes))) as pool:
        for scope, outcome in pool.map(one, scopes):
            outcomes[scope] = outcome
    return outcomes


# --------------------------------------------------------------------------- common
def _shim_tool(name):
    """Path of a sibling tool in tools/persistent_cache/."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), name)


def _rebase_triton_groups(base):
    triton_dir = os.path.join(base, "triton")
    if os.environ.get("MCORE_TRITON_GROUP_REBASE", "1") != "1":
        return
    script = _shim_tool("rebase_triton_groups.sh")
    if not os.path.isfile(script) or not os.path.isdir(triton_dir):
        return
    result = subprocess.run(["bash", script, triton_dir], capture_output=True, text=True)
    if result.returncode != 0:
        # Nonzero also covers "no manifests"; misses fall back to a safe compile.
        _warn("triton group-path rebase incomplete (cache misses compile natively)")


def _finish_read_base(base):
    """Post-populate step shared by every read source: Triton group rebase
    (absolute child paths must point at THIS base)."""
    _rebase_triton_groups(base)


def _configure_image_topology(base, manifest):
    """Validate optional image topology on this node before NCCL reads it.

    Tools stay in the checkout. Only XML and its hardware fingerprint travel
    with the image; no NCCL library replacement or graph replay is involved.
    An explicit 0 disables this feature. Images without this feature leave
    the caller's NCCL configuration alone.
    """
    enabled = os.environ.get("MCORE_NCCL_TOPO_CACHE", manifest.get(_TOPO_RUNTIME_KEY, "0"))
    if enabled == "0":
        return
    if enabled != "1":
        _warn("invalid NCCL topology setting; image topology disabled")
        return
    # This opt-in path owns these variables. In particular, a previous dump
    # must not cause an extra discovery and an inherited graph is not replayed.
    for variable in _TOPO_ENV:
        os.environ.pop(variable, None)
    try:
        from pathlib import Path

        namespace = manifest.get("nccl_topo_namespace")
        if not namespace:
            raise ValueError("image has no unambiguous nccl_topo_namespace")
        spec = importlib.util.spec_from_file_location(
            "_mcore_nccl_topology", _shim_tool("prepare_nccl_topology_template.py")
        )
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        template = Path(base) / "nccl_topo" / "topo_template.xml"
        metadata = helper.validate_template(
            template, template.with_name("topo_template.manifest.json"), namespace=namespace
        )
    except Exception as exc:
        _warn(f"NCCL topology rejected; native discovery: {exc}")
        return
    os.environ["NCCL_TOPO_FILE"] = str(template)
    _warn(
        f"NCCL topology enabled namespace={namespace} file={template} gpus={metadata.get('gpu_count', '?')}"
    )


class _NodeLock:
    """Exclusive advisory lock with a bounded wait (bash flock -w equivalent)."""

    def __init__(self, path):
        self.path = path
        self.fd = None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as error:
                if error.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.monotonic() > deadline:
                    raise TimeoutError(f"lock {self.path} not acquired in {_LOCK_TIMEOUT_S}s")
                time.sleep(0.25)

    def __exit__(self, *_exc):
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)


_AOT_MARKER = os.path.join("cutedsl", ".aot_namespace")


def _write_aot_namespace_marker(base, scopes):
    """Embed the seed's AOT namespace inside the cutedsl scope.

    CuTe AOT objects are keyed by MCORE_CUTE_DSL_AOT_NAMESPACE, which readers
    default from basename(MCORE_CACHE_DIR) — so a cutedsl tarball copied into
    another namespace (or baked into an image) would silently never hit. The
    marker travels inside the scope through consolidation and image install
    and lets readers recover the seed's key.
    """
    namespace = os.environ.get("MCORE_CUTE_DSL_AOT_NAMESPACE")
    if not namespace or "cutedsl" not in scopes:
        return
    marker = os.path.join(base, _AOT_MARKER)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as fh:
        fh.write(namespace)


def _apply_aot_namespace_marker(base):
    """Prefer the seed-embedded AOT namespace over the reader's default.

    Only when the reader's value came from enable.sh's default
    (MCORE_CUTE_DSL_AOT_NS_FROM=default) — an explicit user export wins.
    """
    if os.environ.get("MCORE_CUTE_DSL_AOT_NS_FROM") != "default":
        return
    marker = os.path.join(base, _AOT_MARKER)
    try:
        with open(marker) as fh:
            namespace = fh.read().strip()
    except OSError:
        return
    if namespace:
        os.environ["MCORE_CUTE_DSL_AOT_NAMESPACE"] = namespace


def _pin_environment(base, mode):
    """Point every cache consumer at the scope dirs under base (all processes)."""
    for env_name, scope in _SCOPE_ENV.items():
        scope_dir = os.path.join(base, scope)
        os.makedirs(scope_dir, exist_ok=True)
        os.environ[env_name] = scope_dir

    hybrid_root = os.path.join(base, "hybrid_ep")
    has_so = any(
        name.endswith(".so") for _dir, _subdirs, files in os.walk(hybrid_root) for name in files
    )
    os.environ["HYBRID_EP_LOAD_CACHED_KERNELS"] = "1" if has_so else "0"


def _summary(outcomes):
    return " ".join(
        f"{scope}={outcome.split(':')[0]}" for scope, outcome in sorted(outcomes.items())
    )


def _warn_failures(outcomes):
    for scope, outcome in sorted(outcomes.items()):
        if outcome.startswith("failed"):
            _warn(f"scope {scope} NOT restored -> {outcome}")


def _once_per_node(base, stamp, wipe, populate):
    """Run `populate()` exactly once per node for `base`.

    The sentinel records `stamp` ("<source> <jobid>"). Same stamp -> reuse
    (ft_launcher restart cycles within a job); a different job's leftovers ->
    `wipe` decides whether base is cleared first (scratch dirs: yes; the image
    cache: never). Race losers block on the lock, then see the sentinel.
    Returns the sentinel's previous content when populate was skipped, else None.

    The lock lives beside a wiped base (wiping would remove a lock inside it)
    and inside a never-wiped base (its parent, e.g., /opt, need not be writable).
    """

    def _current(path):
        try:
            with open(path) as fh:
                content = fh.read().strip()
        except OSError:
            return None
        return content

    os.makedirs(base, exist_ok=True)
    sentinel = os.path.join(base, _READY)
    content = _current(sentinel)
    if content == stamp:
        return content
    if content is not None and content.split()[-1:] == [stamp.split()[-1]]:
        # Same job, other mode/source: stale MCORE_CACHE_LOCAL reuse (e.g., a
        # read leg on a write leg's dirs). Fail-open with the fix spelled out.
        _warn(
            f"base {base} was set up as '{content}' but this run is "
            f"'{stamp}' — point MCORE_CACHE_LOCAL at a fresh directory"
        )
        return content
    lock_path = f"{base}.setup.lock" if wipe else os.path.join(base, ".setup.lock")
    with _NodeLock(lock_path):
        content = _current(sentinel)
        if content == stamp:
            return content
        if wipe:
            shutil.rmtree(base, ignore_errors=True)
            os.makedirs(base, exist_ok=True)
        populate()
        with open(sentinel, "w") as fh:
            fh.write(stamp)
    return None


# --------------------------------------------------------------------------- entry
def setup():
    if os.environ.get("_MCORE_CACHE_NODE_SETUP_ACTIVE") == "1":
        # Child python spawned by setup itself (triton rebase, topo template
        # prep) — re-entering would deadlock on the node lock its parent holds.
        return
    cache_dir = os.environ.get("MCORE_CACHE_DIR")
    mode = os.environ.get("MCORE_CACHE_MODE", "read")
    image_dir = image_cache_dir()
    has_image = image_cache_present(image_dir)
    prefer_image = os.environ.get("MCORE_CACHE_PREFER_IMAGE", "0") == "1"
    if mode == "read" and has_image and (not cache_dir or prefer_image):
        source = "image"
    elif cache_dir:
        source = "lustre"
    else:
        return  # nothing configured: inert
    os.environ["_MCORE_CACHE_NODE_SETUP_ACTIVE"] = "1"
    try:
        if source == "image":
            _setup_image(image_dir)
        else:
            _setup_lustre(cache_dir, mode)
    finally:
        os.environ.pop("_MCORE_CACHE_NODE_SETUP_ACTIVE", None)


def _scopes():
    return (os.environ.get("MCORE_CACHE_SCOPES") or _DEFAULT_SCOPES).split()


def _setup_lustre(cache_dir, mode):
    if mode not in ("read", "write"):
        _warn(f"invalid MCORE_CACHE_MODE={mode}; expected read|write — cache disabled")
        return
    scopes = _scopes()
    base = resolve_base(mode)
    if base is None:
        _warn("no write+exec cache base (/dev/shm, /tmp, lustre); cache disabled")
        return
    jid = os.environ.get("SLURM_JOB_ID", "manual")
    started = time.monotonic()
    report = {}

    def populate():
        if mode == "write":
            _write_aot_namespace_marker(base, scopes)
            return
        outcomes = _extract_scopes(cache_dir, base, scopes)
        _finish_read_base(base)
        _warn_failures(outcomes)
        report["summary"] = _summary(outcomes)

    if _once_per_node(base, f"{mode} {jid}", wipe=True, populate=populate) is None:
        _warn(
            f"extracted mode={mode} base={base} "
            f"in {time.monotonic() - started:.1f}s {report.get('summary', '')}".rstrip()
        )
    if mode == "read":
        _apply_aot_namespace_marker(base)
    _pin_environment(base, mode)


def _setup_image(image_dir):
    scopes = _scopes()
    manifest = read_manifest(image_dir)
    if os.environ.get("MCORE_NCCL_TOPO_CACHE", manifest.get(_TOPO_RUNTIME_KEY, "0")) == "1":
        # Clear before any setup can fail; never leave an inherited topology
        # active when the selected image's template was not validated.
        for variable in _TOPO_ENV:
            os.environ.pop(variable, None)
        if "nccl_topo" not in scopes:
            scopes.append("nccl_topo")
    jid = os.environ.get("SLURM_JOB_ID", "manual")
    started = time.monotonic()

    in_place = _dir_write_exec(image_dir) and os.environ.get("MCORE_CACHE_IMAGE_COPY", "0") != "1"
    if in_place:
        # Direct use: the rootfs is node-local and writable, so the compilers
        # read AND write the baked dirs in place. Never wipe here — that would
        # delete the cache the image ships. Any failure (lock, rebase) falls
        # back to the node-local copy below instead of a cold start.
        base = image_dir
        missing = [scope for scope in scopes if not os.path.isdir(os.path.join(base, scope))]

        def populate():
            _finish_read_base(base)

        try:
            if _once_per_node(base, f"image {jid}", wipe=False, populate=populate) is None:
                _warn(
                    f"image cache in place base={base} namespace={manifest.get('namespace', '?')} "
                    f"in {time.monotonic() - started:.1f}s"
                    + (f" missing={','.join(missing)}" if missing else "")
                )
        except (OSError, TimeoutError) as exc:
            _warn(f"image cache in-place setup failed ({exc}); copying node-locally instead")
            in_place = False
    if not in_place:
        # Read-only rootfs: one node-local copy per job (no tarballs, no Lustre).
        base = resolve_base("read")
        if base is None:
            _warn("no write+exec cache base for the image-cache copy; cache disabled")
            return
        report = {}

        def populate():
            outcomes = _copy_scopes(image_dir, base, scopes)
            _finish_read_base(base)
            _warn_failures(outcomes)
            report["summary"] = _summary(outcomes)

        if _once_per_node(base, f"image-copy {jid}", wipe=True, populate=populate) is None:
            _warn(
                f"image cache copied from {image_dir} base={base} "
                f"namespace={manifest.get('namespace', '?')} "
                f"in {time.monotonic() - started:.1f}s {report.get('summary', '')}".rstrip()
            )
    _apply_aot_namespace_marker(base)
    _pin_environment(base, "read")
    _apply_image_runtime_defaults(manifest)
    _configure_image_topology(base, manifest)
