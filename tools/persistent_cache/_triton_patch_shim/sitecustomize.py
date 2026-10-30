"""sitecustomize.py loaded via PYTHONPATH prepend by tools/persistent_cache/enable.sh.

Loaded automatically by Python's `site` module before user code runs. Jobs:

0. Run the container's own /usr/lib/python3.12/sitecustomize.py first (it may
   add site directories, as the 26.07 deps+v8 image does, besides installing
   apport); override the path with MCORE_CONTAINER_SITECUSTOMIZE.
1. Standalone cache path: run mcore_cache_node_setup — the first process per
   node makes the warm cache available node-locally (image cache used in
   place, or consolidated tarballs extracted), every process pins the cache
   env vars. See that module for source selection.
2. Arm lazy patches so helper processes never import Torch/CuTe/HybridEP just
   by starting:
   - HybridEP: link persisted JIT `.so` files into this process's PID-specific
     cache directory and enable `load_cached_kernels`.
   - CuTe DSL: let explicit `cute.compile` calls use the built-in IR file cache
     (the installed API forces `no_cache=True`), with a cache-dir-independent
     module hash; optionally reuse exported TVM-FFI objects across processes.
   - Inductor: recover autotune results after an FX-graph cache reload.
   - Triton: disk-cache autotune results for configs carrying a pre_hook, which
     upstream refuses to cache.
"""

import importlib.abc
import inspect
import io
import os
import sys


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Apply an optional cache patch after its dependency is imported."""

    def __init__(self, target, callback, failure_label):
        self.target = target
        self.callback = callback
        self.failure_label = failure_label

    def find_spec(self, fullname, path, target=None):
        if fullname != self.target:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            try:
                spec = find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is None or spec.loader is None:
                continue
            original_loader = spec.loader
            callback = self.callback
            failure_label = self.failure_label

            class _Loader(importlib.abc.Loader):
                def create_module(self, inner_spec):
                    create_module = getattr(original_loader, "create_module", None)
                    return create_module(inner_spec) if create_module else None

                def exec_module(self, module):
                    original_loader.exec_module(module)
                    try:
                        callback(module)
                    except Exception as exc:
                        print(
                            f"[{failure_label}] setup failed; using native path: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )

            spec.loader = _Loader()
            return spec
        return None


def _arm_post_import(target, callback, failure_label):
    module = sys.modules.get(target)
    if module is not None:
        callback(module)
    else:
        sys.meta_path.insert(0, _PostImportFinder(target, callback, failure_label))


# (0) Chain to the container's own sitecustomize FIRST. Being first on
# PYTHONPATH, this file replaces it, and it may do more than install apport: the
# 26.07 deps+v8 image's /etc/python3.12/sitecustomize.py is what puts
# /usr/local/lib/python3.12/dist-packages on sys.path (its CPython build has no
# dist-packages default). Chaining at the end left every third-party import
# during our setup failing with "No module named ..." (job 3633302 ran cold).
# runpy executes the file by path, so a different Python version still works.
try:
    import runpy

    _container_sitecustomize = os.environ.get(
        "MCORE_CONTAINER_SITECUSTOMIZE", "/usr/lib/python3.12/sitecustomize.py"
    )
    if os.path.exists(_container_sitecustomize):
        runpy.run_path(_container_sitecustomize, run_name="__sitecustomize_chain__")
except Exception as exc:
    print(
        f"[CACHE SITECUSTOMIZE] container sitecustomize chain failed: {exc}",
        file=sys.stderr,
        flush=True,
    )

# Standalone env-only path (enable.sh, no rank-entrypoint wrapper): the first
# process per node makes the warm cache available node-locally, every process
# pins the cache env vars. No-op when neither MCORE_CACHE_DIR nor an image cache
# is configured, or when the legacy bash bootstrap owns the run (NODE_CACHE_BASE
# set). Must run before any hook can arm a training-stack cache.
try:
    import mcore_cache_node_setup  # noqa: E402

    mcore_cache_node_setup.setup()
except Exception as exc:
    print(f"[CACHE NODE SETUP] failed; using native path: {exc}", file=sys.stderr, flush=True)


def _install_cute_dsl_file_cache():
    """Let explicit CuTe DSL compiles use the built-in IR file cache."""
    if os.environ.get("MCORE_CUTE_DSL_FILE_CACHE") != "1":
        return
    target = "cutlass.base_dsl.dsl"
    if target not in sys.modules:
        _arm_post_import(target, lambda _module: _install_cute_dsl_file_cache(), "CUTE-DSL CACHE")
        return

    from cutlass.base_dsl.dsl import BaseDSL

    original_func = BaseDSL._func
    original_generate_mlir = BaseDSL.generate_mlir
    if getattr(original_func, "_mcore_persistent_cache", False):
        return

    activated = False

    def stable_get_module_hash(self, module, function_name):
        # The installed implementation hashes every EnvironmentVarManager field,
        # including cache_dir. Our node-local path contains SLURM_JOB_ID, so the
        # same IR otherwise gets a different key every restart. Cache location is
        # not a code-generation input. Reproduce the installed hash routine with
        # a local substitution instead of mutating shared EnvironmentVarManager.
        data = io.BytesIO()
        module.operation.write_bytecode(data)
        for attr, value in self.envar.__dict__.items():
            if value is not None:
                if attr == "cache_dir":
                    value = "<MCORE_CUTE_DSL_CACHE_DIR>"
                data.write(str(value).encode())
        data.write(self.compile_options.to_str().encode())
        hash_obj = self.get_version().copy()
        hash_obj.update(data.getvalue())
        return hash_obj.hexdigest()

    _mcore_generate_mlir_signature = inspect.signature(original_generate_mlir)

    def cached_generate_mlir(self, *call_args, **call_kwargs):
        # Bind against the INSTALLED signature rather than pinning parameter
        # positions. cuteDSL's generate_mlir gained compile_to_precompiled_mlir
        # in the Run-2 cutedslgdp image, and a fixed parameter list raised
        #   TypeError: cached_generate_mlir() got an unexpected keyword
        #              argument 'compile_to_precompiled_mlir'
        # on every rank. Binding keeps this wrapper valid across cuteDSL
        # versions; if binding fails we defer to the original untouched.
        try:
            bound = _mcore_generate_mlir_signature.bind(self, *call_args, **call_kwargs)
        except TypeError:
            return original_generate_mlir(self, *call_args, **call_kwargs)
        # CompileCallable._compile sets both compile_only=True and no_cache=True,
        # and BaseDSL._func reinforces no_cache for compile-only calls. Override
        # the final value only at the boundary that consumes it.
        if bound.arguments.get("compile_only") and bound.arguments.get("no_cache"):
            bound.arguments["no_cache"] = False
        return original_generate_mlir(*bound.args, **bound.kwargs)

    def cached_func(self, func, *args, **kwargs):
        nonlocal activated
        use_file_cache = kwargs.get("compile_only") is True and kwargs.get("no_cache") is True
        if use_file_cache and not activated:
            activated = True
            _cache_detail(f"[CUTE-DSL CACHE] explicit compile intercepted pid={os.getpid()}")
        try:
            return original_func(self, func, *args, **kwargs)
        finally:
            # `cute.compile` callers retain the returned executable in their own
            # shape cache. Drop only BaseDSL's extra reference after file
            # write-through/load so explicit compiles keep original lifetime.
            if use_file_cache:
                self.jit_cache.clear()

    cached_func._mcore_persistent_cache = True
    cached_generate_mlir._mcore_persistent_cache = True
    stable_get_module_hash._mcore_persistent_cache = True
    BaseDSL._func = cached_func
    BaseDSL.generate_mlir = cached_generate_mlir
    BaseDSL.get_module_hash = stable_get_module_hash
    _cache_status(f"[CUTE-DSL CACHE] enabled pid={os.getpid()}")


def _install_hybrid_ep_cache():
    """Enable persisted HybridEP JIT libraries for this Python process."""
    if os.environ.get("HYBRID_EP_LOAD_CACHED_KERNELS") != "1":
        return
    cache_root = os.environ.get("HYBRID_EP_CACHE_DIR")
    if not cache_root:
        return
    target = "deep_ep"
    if target not in sys.modules:
        _arm_post_import(target, lambda _module: _install_hybrid_ep_cache(), "HYBRID-EP CACHE")
        return

    import atexit
    import glob

    jit_root = os.path.join(cache_root, ".deepep", "hybrid_ep", "jit")
    current_dir = os.path.join(jit_root, f"proc-{os.getpid()}")
    os.makedirs(current_dir, exist_ok=True)
    sources = {}
    for source in sorted(glob.glob(os.path.join(jit_root, "proc-*", "*.so"))):
        if os.path.islink(source) and not os.path.exists(source):
            # Warm ranks share this node-local tree. Ignore stale links instead
            # of racing to delete the same path from every process.
            continue
        resolved = os.path.realpath(source)
        if os.path.isfile(resolved):
            sources.setdefault(os.path.basename(source), resolved)

    seeded = 0
    created_targets = []
    for basename, source in sources.items():
        target = os.path.join(current_dir, basename)
        if os.path.lexists(target):
            continue
        os.symlink(source, target)
        created_targets.append(target)
        seeded += 1

    def cleanup_links():
        for target in created_targets:
            if os.path.islink(target):
                os.unlink(target)
        try:
            os.rmdir(current_dir)
        except OSError:
            pass

    atexit.register(cleanup_links)

    import deep_ep

    cls = deep_ep.HybridEPBuffer
    original_init = cls.__init__
    if getattr(original_init, "_mcore_persistent_cache", False):
        return

    def cached_init(self, *args, **kwargs):
        kwargs.setdefault("load_cached_kernels", True)
        return original_init(self, *args, **kwargs)

    cached_init._mcore_persistent_cache = True
    cls.__init__ = cached_init
    _cache_status(
        f"[HYBRID-EP CACHE] enabled pid={os.getpid()} seeded_symlinks={seeded} dir={current_dir}"
    )


# Make the helper modules in tools/persistent_cache importable.
_shim_dir = os.path.dirname(os.path.abspath(__file__))
_pc_dir = os.path.dirname(_shim_dir)
if _pc_dir not in sys.path:
    sys.path.insert(0, _pc_dir)

# (1) Arm the Triton autotune disk-cache patch. Upstream Triton skips disk-cache
# lookup for autotune configs that carry a pre_hook (the Mamba SSD kernels do),
# so those kernels re-autotune on every restart even with a warm cache. The
# finder patches Autotuner on the first `import triton.runtime.autotuner` and
# never imports Triton itself. Kill switch: TRITON_AUTOTUNE_PREHOOK_PATCH_DISABLE=1.
try:
    import triton_autotune_disk_cache  # noqa: E402

    triton_autotune_disk_cache.arm()
except Exception as exc:
    print(
        f"[TRITON_PATCH] setup failed; using native autotune cache: {exc}",
        file=sys.stderr,
        flush=True,
    )

# Rank-gated cache logging: quiet by default (rank-0 status lines only),
# MCORE_CACHE_VERBOSE=1 restores the full per-rank output.
try:
    from cache_verbosity import detail as _cache_detail
    from cache_verbosity import status as _cache_status
except Exception:

    def _cache_status(msg):
        print(msg, file=sys.stderr, flush=True)

    def _cache_detail(msg):
        pass


# (2) Reuse HybridEP JIT libraries. Opt-in: mcore_cache_node_setup sets
# HYBRID_EP_LOAD_CACHED_KERNELS=1 only when the warm cache holds at least one .so.
try:
    _install_hybrid_ep_cache()
except Exception as exc:
    print(
        f"[HYBRID-EP CACHE] setup failed; falling back to JIT: {exc}", file=sys.stderr, flush=True
    )

# (3) Use CuTe DSL's existing file-cache path for explicit `cute.compile` calls.
# Patching generate_mlir is intentional: BaseDSL._func itself forces no_cache
# back to true for compile-only calls after consuming the caller's kwargs.
try:
    _install_cute_dsl_file_cache()
except Exception as exc:
    print(f"[CUTE-DSL CACHE] setup failed; falling back to JIT: {exc}", file=sys.stderr, flush=True)

# (4) Optionally bypass process-variant CuTe IR generation by loading an AOT
# object exported by an earlier process. The helper is gated by its own env var.
try:
    import cute_aot_cache  # noqa: E402

    cute_aot_cache.arm()
except Exception as exc:
    print(f"[CUTE-AOT CACHE] setup failed; falling back to JIT: {exc}", file=sys.stderr, flush=True)

# (5) Arm the Inductor restart-autotune fallback before generated FX graph
# modules import torch._inductor.runtime.autotune_cache.
try:
    import inductor_autotune_restart_cache  # noqa: E402

    inductor_autotune_restart_cache.arm()
    inductor_autotune_restart_cache.install()
except Exception as exc:
    print(
        f"[INDUCTOR-AUTOTUNE RESTART] setup failed; using upstream cache: {exc}",
        file=sys.stderr,
        flush=True,
    )

# Optional startup warmup uses ordinary compiled functions and retains guards.
# The helper imports no Torch until the existing training warmup has completed.
if os.environ.get("MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU") == "1":
    try:
        import inductor_prewarm  # noqa: E402

        _arm_post_import(
            "megatron.training.initialize", inductor_prewarm.install, "INDUCTOR PREWARM"
        )
    except Exception as exc:
        print(f"[INDUCTOR PREWARM] setup failed: {exc}", file=sys.stderr, flush=True)
