"""Opt-in cross-process object cache for explicit CuTe DSL TVM-FFI compiles."""

from __future__ import annotations

import atexit
import hashlib
import importlib.abc
import inspect
import itertools
import json
import marshal
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

try:
    from cache_verbosity import detail as _log_detail
    from cache_verbosity import status as _log_status
except Exception:  # fail-open: never let logging gating break the cache

    def _log_status(msg):
        print(msg, file=sys.stderr, flush=True)

    def _log_detail(msg):
        pass


_ARMED = False

# Reuse/recompile accounting. This wrapper is the one place where the outcome is
# observable: an explicit TVM-FFI compile either loads a cached object or falls
# through to a real compile. Without it a warm run and a cold one look identical
# in the log. Per-call lines are already verbosity-gated above; this is the
# always-on aggregate, one line per reporting rank at exit.
_STATS = {
    "reused": 0,  # cached object found and loaded
    "recompiled": 0,  # no cached object -> real compile, then exported
    "load_failed": 0,  # cached object present but unusable -> recompiled
    "export_failed": 0,  # compiled, but could not be published for the next run
}
_REPORT_REGISTERED = False


def _report_stats():
    """One line summarising reuse vs recompile for this process."""
    if not sum(_STATS.values()):
        return
    rank = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))
    if rank != "0" and os.environ.get("MCORE_CUTE_DSL_AOT_DEBUG") != "1":
        return
    summary = " ".join(f"{name}={count}" for name, count in sorted(_STATS.items()))
    print(f"[CUTE-AOT CACHE] summary rank={rank} {summary}", file=sys.stderr, flush=True)


_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
_C_VOID_P = re.compile(r"c_void_p\(\d+\)")
_CUDA_ORDINAL = re.compile(r"@cuda:\d+")
_SKIP_ATTRS = {
    "_c_pointer",
    "_c_pointers_cache",
    "_compiled_kernel",
    "_dsl_object",
    "_logger",
    "_pointer",
    "_workspace",
    "jit_cache",
}


def _stable(value: Any, depth: int = 0, seen: set[int] | None = None) -> Any:
    """Return a deterministic, JSON-safe compile descriptor."""
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if depth >= 7:
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}

    value_id = id(value)
    if value_id in seen:
        return {"cycle": f"{type(value).__module__}.{type(value).__qualname__}"}
    seen.add(value_id)
    try:
        if isinstance(value, list):
            return {"list": [_stable(item, depth + 1, seen) for item in value]}
        if isinstance(value, tuple):
            return {"tuple": [_stable(item, depth + 1, seen) for item in value]}
        if isinstance(value, dict):
            return {
                "dict": [
                    [_stable(key, depth + 1, seen), _stable(item, depth + 1, seen)]
                    for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
                ]
            }
        if isinstance(value, (set, frozenset)):
            items = [_stable(item, depth + 1, seen) for item in value]
            return {
                type(value).__name__: sorted(
                    items, key=lambda item: json.dumps(item, sort_keys=True)
                )
            }
        if inspect.isfunction(value) or inspect.ismethod(value):
            function = inspect.unwrap(value.__func__ if inspect.ismethod(value) else value)
            code = getattr(function, "__code__", None)
            # Protocols >=3 encode reference sharing: merely retaining co_consts
            # in a profiler can change their bytes. Version 2 retains the full
            # code object (including nested code and binary float constants),
            # without reference-count or string-interning-dependent encodings.
            return {
                "callable": f"{function.__module__}.{function.__qualname__}",
                "code_sha256": (
                    hashlib.sha256(marshal.dumps(code, 2)).hexdigest() if code else None
                ),
                "closure": [
                    _stable(cell.cell_contents, depth + 1, seen)
                    for cell in (getattr(function, "__closure__", None) or ())
                ],
            }
        if all(hasattr(value, attr) for attr in ("shape", "stride", "dtype", "device")):
            stride = value.stride() if callable(value.stride) else value.stride
            device = str(value.device)
            if device.startswith("cuda:"):
                device = "cuda"
            return {
                "tensor_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "shape": list(value.shape),
                "stride": list(stride),
                "dtype": str(value.dtype),
                "device": device,
            }
        state = getattr(value, "__dict__", None)
        if state:
            return {
                "object": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": {
                    key: _stable(item, depth + 1, seen)
                    for key, item in sorted(state.items())
                    if key not in _SKIP_ATTRS
                },
            }
        return {
            "object": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": _C_VOID_P.sub(
                "c_void_p(<runtime>)",
                _CUDA_ORDINAL.sub("@cuda", _ADDRESS.sub("<runtime>", repr(value))),
            ),
        }
    finally:
        seen.remove(value_id)


def _call_descriptor(namespace: str, ordinal: int, func: Any, args: tuple, kwargs: dict) -> dict:
    # NOTE: `ordinal` is deliberately NOT part of the key. It is a per-process
    # counter of compiles reaching this wrapper, so it only lines up when the
    # seeding and consuming runs compile the same sequence -- they never do,
    # because a consuming run's CuTe MLIR file cache absorbs most compiles.
    # Keying on it produced reused=0 with the objects present (aga 469320: the
    # same ordinal mapped to 8 different kernels across ranks). It is still
    # passed in and logged, purely to correlate lines within one process.
    return {
        # New keys intentionally require reseeding; never alias legacy objects
        # whose reference-sensitive code fingerprints cannot be verified here.
        "format": "mcore-cute-aot-v2",
        "namespace": namespace,
        "python": sys.version_info[:2],
        "func": _stable(func),
        "args": _stable(args),
        "kwargs": _stable(kwargs),
    }


def _descriptor_key(descriptor: dict) -> str:
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _call_key(namespace: str, ordinal: int, func: Any, args: tuple, kwargs: dict) -> str:
    return _descriptor_key(_call_descriptor(namespace, ordinal, func, args, kwargs))


def _enabled() -> bool:
    return os.environ.get("MCORE_CUTE_DSL_AOT_CACHE") == "1"


def _apply_patch(compiler_module) -> None:
    """Patch an imported CuTe compiler module."""
    if os.environ.get("MCORE_CUTE_DSL_AOT_CACHE") != "1":
        return
    namespace = os.environ.get("MCORE_CUTE_DSL_AOT_NAMESPACE")
    cache_root = os.environ.get("CUTE_DSL_CACHE_DIR")
    if not namespace or not cache_root:
        _log_status("[CUTE-AOT CACHE] disabled: namespace or CUTE_DSL_CACHE_DIR missing")
        return

    CompileCallable = compiler_module.CompileCallable
    original_compile = CompileCallable._compile
    if getattr(original_compile, "_mcore_aot_cache", False):
        return

    counter = itertools.count()
    modules = []
    object_dir = Path(cache_root) / "aot_objects"
    object_dir.mkdir(parents=True, exist_ok=True)
    shared_root = (
        os.environ.get("MCORE_CUTE_DSL_AOT_SHARED_DIR")
        if os.environ.get("MCORE_CACHE_WRITEBACK_DISABLED") != "1"
        else None
    )
    shared_dir = Path(shared_root) if shared_root else None
    if shared_dir is not None:
        shared_dir.mkdir(parents=True, exist_ok=True)

    def publish_shared(key: str, payload: bytes) -> None:
        """Publish one cluster-wide object without duplicate-rank write storms."""
        if shared_dir is None:
            return
        target = shared_dir / f"{key}.o"
        if target.is_file():
            return
        lock = shared_dir / f".{key}.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return
        os.close(fd)
        try:
            if not target.is_file():
                tmp = shared_dir / f".{key}.{os.getpid()}.tmp"
                tmp.write_bytes(payload)
                os.replace(tmp, target)
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

    def cached_compile(self, func, *args, **kwargs):
        if "--enable-tvm-ffi" not in str(kwargs.get("options", "")):
            return original_compile(self, func, *args, **kwargs)

        ordinal = next(counter)
        descriptor = _call_descriptor(namespace, ordinal, func, args, kwargs)
        key = _descriptor_key(descriptor)
        prefix = f"mcore_{key[:24]}"
        object_path = object_dir / f"{key}.o"
        rank = os.environ.get("SLURM_PROCID", "0")
        debug_ranks = {
            item.strip()
            for item in os.environ.get("MCORE_CUTE_DSL_AOT_DEBUG_RANKS", "0").split(",")
        }
        if os.environ.get("MCORE_CUTE_DSL_AOT_DEBUG") == "1" and rank in debug_ranks:
            debug_dir = (
                Path(os.environ.get("MCORE_CACHE_DIR", cache_root))
                / "aot_debug"
                / f"job_{os.environ.get('SLURM_JOB_ID', 'manual')}"
            )
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"rank_{rank}_ordinal_{ordinal}.json"
            debug_path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")
            print(
                f"[CUTE-AOT CACHE] debug ordinal={ordinal} path={debug_path}",
                file=sys.stderr,
                flush=True,
            )
        if object_path.is_file():
            try:
                import cutlass.cute as cute

                module = cute.runtime.load_module(str(object_path), enable_tvm_ffi=True)
                result = module[prefix]
                modules.append(module)
                _STATS["reused"] += 1
                _log_detail(f"[CUTE-AOT CACHE] hit ordinal={ordinal} key={key[:12]}")
                return result
            except Exception as exc:
                _STATS["load_failed"] += 1
                print(
                    f"[CUTE-AOT CACHE] load failed ordinal={ordinal} "
                    f"key={key[:12]}; recompiling: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        result = original_compile(self, func, *args, **kwargs)
        tmp = object_path.with_name(
            f".{object_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            # TVM-FFI compiles require their generated __tvm_ffi_* entrypoint;
            # dump_to_object exports only the plain C ABI and is unsafe here.
            result.export_to_c(str(tmp), prefix, enable_pic=True)
            payload = tmp.read_bytes()
            os.replace(tmp, object_path)
            publish_shared(key, payload)
            _STATS["recompiled"] += 1
            _log_detail(
                f"[CUTE-AOT CACHE] store ordinal={ordinal} key={key[:12]} bytes={len(payload)}"
            )
        except Exception as exc:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            _STATS["export_failed"] += 1
            print(
                f"[CUTE-AOT CACHE] export failed ordinal={ordinal} " f"key={key[:12]}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        return result

    cached_compile._mcore_aot_cache = True
    CompileCallable._compile = cached_compile
    global _REPORT_REGISTERED
    if not _REPORT_REGISTERED:
        atexit.register(_report_stats)
        _REPORT_REGISTERED = True
    _log_status(f"[CUTE-AOT CACHE] enabled namespace={namespace}")


class _CompilerFinder(importlib.abc.MetaPathFinder):
    TARGET = "cutlass.base_dsl.compiler"

    def find_spec(self, fullname, path, target=None):
        if fullname != self.TARGET:
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

            class _Loader(importlib.abc.Loader):
                def create_module(self, inner_spec):
                    create_module = getattr(original_loader, "create_module", None)
                    return create_module(inner_spec) if create_module else None

                def exec_module(self, module):
                    original_loader.exec_module(module)
                    _apply_patch(module)

            spec.loader = _Loader()
            return spec
        return None


def arm() -> None:
    """Patch CuTe when its compiler module is first imported."""
    global _ARMED
    if _ARMED or not _enabled():
        return
    if _CompilerFinder.TARGET in sys.modules:
        _apply_patch(sys.modules[_CompilerFinder.TARGET])
    else:
        sys.meta_path.insert(0, _CompilerFinder())
    _ARMED = True


def install() -> None:
    """Patch CuTe if loaded already, otherwise arm the lazy import hook."""
    if not _enabled():
        return
    module = sys.modules.get(_CompilerFinder.TARGET)
    if module is not None:
        _apply_patch(module)
    else:
        arm()
