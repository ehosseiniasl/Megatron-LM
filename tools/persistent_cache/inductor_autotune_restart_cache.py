"""Recover Inductor autotune results after an FX-graph cache reload.

Cold compilation stores ``*.best_config`` beside a standalone Triton kernel
file.  An FX-graph cache hit recreates the same kernel inside the larger graph
module, so upstream looks beside a different ``filename`` and benchmarks the
configs again.  This opt-in fallback finds the original result using semantic
metadata, pairing each standalone kernel source with exactly its own
``.best_config`` (the name is recomputed with the installed PyTorch's own key
functions).  Configs must agree after discarding the tuning diagnostics that
PyTorch ignores, and the selected config must be a current candidate.  When the
key functions are unavailable the lookup falls back to per-directory matching
and refuses any ambiguous result.  A second path persists runtime-selected
configs when multiple compiled launchers remain after the upstream lookup.
The original v1 reduction cache remains readable; ordinary autotunes use v2
keys covering source, specialization, compiler version and compiled candidates.
Both accept a stored winner only when exactly one current launcher matches.
An additional opt-in reuses pointwise results across x tiling scores only when
reconstructing the original strict key proves every other field is identical
and all indexed, exact-backed observations agree on the same compiled winner.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.abc
import json
import os
import sys
import tempfile
from enum import Enum
from typing import Any

_ARMED = False
_INSTALLED = False
_TRITON_INSTALLED = False
_REPORT_REGISTERED = False
_LOOKUP_CACHE: dict[tuple[str, ...], dict[str, Any] | None] = {}
_SOURCE_RECORDS: dict[str, list[tuple[str, tuple[dict[str, Any], ...]]]] = {}
_LATE_PENDING: dict[int, tuple[str, str, int, tuple[str, ...], tuple[str, int | float] | None]] = {}
_COUNTERS = {
    "native_hit": 0,
    "fallback_hit": 0,
    "fallback_miss": 0,
    "fallback_ambiguous": 0,
    "late_hit": 0,
    "late_miss": 0,
    "late_invalid": 0,
    "late_store": 0,
    "late_equivalent_hit": 0,
    "late_equivalent_invalid": 0,
    "late_equivalent_ambiguous": 0,
}


def _enabled() -> bool:
    return os.environ.get("MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE") == "1"


def _metadata_markers(inductor_meta: dict[str, Any]) -> tuple[str, ...]:
    kernel_name = inductor_meta.get("kernel_name")
    backend_hash = inductor_meta.get("backend_hash")
    tiling_scores = inductor_meta.get("tiling_scores")
    if not kernel_name or not backend_hash or tiling_scores is None:
        return ()
    return (
        f"'kernel_name': {kernel_name!r}",
        f"'backend_hash': {backend_hash!r}",
        f"'tiling_scores': {tiling_scores!r}",
    )


def _expected_config_name(module, directory: str, source_name: str) -> str | None:
    """Return the ``.best_config`` basename PyTorch derives for a kernel source.

    Recomputed with the installed version's own key functions (source basename,
    cache-key tag and ``torch_key``), so a source is paired with exactly its own
    result.  ``None`` when this PyTorch version does not expose them.
    """
    cache_type = getattr(module, "AutotuneCache", None)
    prepare_key = getattr(cache_type, "_prepare_key", None)
    make_local_key = getattr(cache_type, "_make_local_cache_key", None)
    if prepare_key is None or make_local_key is None:
        return None
    try:
        path = make_local_key(directory, prepare_key(os.path.join(directory, source_name)))
    except Exception:
        return None
    return os.path.basename(path)


def _load_records(module, cache_root: str) -> list[tuple[str, tuple[dict[str, Any], ...]]]:
    """Index every standalone kernel source with the payload(s) it may own."""
    records: list[tuple[str, tuple[dict[str, Any], ...]]] = []
    for directory, _, filenames in os.walk(cache_root):
        config_files = [name for name in filenames if name.endswith(".best_config")]
        source_files = [name for name in filenames if name.endswith(".py")]
        if not config_files or not source_files:
            continue
        payloads: dict[str, dict[str, Any]] = {}
        for config_name in config_files:
            try:
                with open(os.path.join(directory, config_name), encoding="utf-8") as config:
                    payloads[config_name] = json.load(config)
            except (OSError, ValueError):
                continue
        if not payloads:
            continue
        for source_name in source_files:
            expected = _expected_config_name(module, directory, source_name)
            if expected is None:
                # Exact pairing unavailable: every result in the directory is a
                # candidate; the ambiguity guard in _find_payload arbitrates.
                candidates = tuple(payloads.values())
            elif expected in payloads:
                candidates = (payloads[expected],)
            else:
                continue  # no result of its own (never tuned, or torch changed)
            try:
                with open(os.path.join(directory, source_name), encoding="utf-8") as source:
                    text = source.read()
            except OSError:
                continue
            records.append((text, candidates))
    return records


def _find_payload(
    cache_root: str, inductor_meta: dict[str, Any], configs_hash: str, module=None
) -> dict[str, Any] | None:
    """Return the one cached payload whose kernel matches ``inductor_meta``.

    Returns ``None`` on a miss and when several distinct configs match (the
    caller then falls through to a native benchmark rather than guessing).
    """
    markers = _metadata_markers(inductor_meta)
    if not markers or not cache_root:
        return None
    lookup_key = (cache_root, configs_hash, *markers)
    if lookup_key in _LOOKUP_CACHE:
        return _LOOKUP_CACHE[lookup_key]

    records = _SOURCE_RECORDS.get(cache_root)
    if records is None:
        records = _load_records(module, cache_root)
        _SOURCE_RECORDS[cache_root] = records

    matches: list[dict[str, Any]] = []
    for text, payloads in records:
        if not all(marker in text for marker in markers):
            continue
        for payload in payloads:
            if payload.get("configs_hash") != configs_hash:
                continue
            # PyTorch discards these diagnostics before selecting a config.
            # Independent ranks can save the same launch with different tuning
            # times or compiled-artifact hashes. Keep all other fields so real
            # config conflicts (including future options) remain ambiguous.
            config = {
                key: value
                for key, value in payload.items()
                if key not in ("time_taken_ms", "triton_cache_hash")
            }
            if config not in matches:
                matches.append(config)

    result = matches[0] if len(matches) == 1 else None
    if len(matches) > 1:
        _COUNTERS["fallback_ambiguous"] += 1
    _LOOKUP_CACHE[lookup_key] = result
    return result


def _report() -> None:
    total = sum(_COUNTERS.values())
    if total == 0:
        return
    print(
        f"[INDUCTOR-AUTOTUNE RESTART SUMMARY] rank={os.environ.get('SLURM_PROCID', '0')} "
        + " ".join(f"{key}={value}" for key, value in _COUNTERS.items()),
        file=sys.stderr,
        flush=True,
    )


def _config_fingerprint(module, config: Any) -> str:
    """Return the installed PyTorch version's exact config identity."""
    return repr(module.triton_config_to_hashable(config))


def _stable_cache_value(value: Any) -> Any:
    """Encode compiler metadata without lossy keys or address-bearing reprs."""
    if value is None or type(value) in (str, bool, int, float):
        return [type(value).__name__, value]
    if isinstance(value, Enum):
        return ["enum", type(value).__module__, type(value).__qualname__, value.name]
    if type(value) is dict:
        items = [
            [_stable_cache_value(key), _stable_cache_value(item)] for key, item in value.items()
        ]
        return ["dict", sorted(items, key=lambda item: json.dumps(item[0]))]
    if type(value) in (list, tuple, set, frozenset):
        items = [_stable_cache_value(item) for item in value]
        if type(value) in (set, frozenset):
            items.sort(key=json.dumps)
        return [type(value).__name__, items]
    raise TypeError(f"Unsupported compiler metadata: {type(value).__name__}")


def _launcher_fingerprint(module, launcher: Any, schema: int) -> str:
    if schema == 1:
        return _config_fingerprint(module, launcher.config)
    cache_hash = getattr(launcher, "cache_hash", None)
    if not isinstance(cache_hash, str) or not cache_hash:
        raise ValueError("Missing compiled kernel identity")
    return json.dumps(
        _stable_cache_value((module.triton_config_to_hashable(launcher.config), cache_hash)),
        separators=(",", ":"),
        allow_nan=False,
    )


def _late_cache_descriptor(autotuner, module) -> tuple[str, str, dict[str, Any]] | None:
    """Build the unchanged strict key and retain its descriptor for verification."""
    cache_root = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
    launchers = getattr(autotuner, "launchers", ())
    metadata = getattr(autotuner, "inductor_meta", {})
    kernel_name = metadata.get("kernel_name")
    backend_hash = metadata.get("backend_hash")
    tiling_scores = metadata.get("tiling_scores")
    source = getattr(getattr(autotuner, "fn", None), "src", None)
    if (
        not cache_root
        or len(launchers) <= 1
        or not kernel_name
        or not backend_hash
        or tiling_scores is None
        or not isinstance(source, str)
    ):
        return None

    schema = 1 if getattr(autotuner, "save_cache_hook", None) is None else 2
    descriptor = {
        "schema": 1,
        "kernel_name": kernel_name,
        "backend_hash": backend_hash,
        "tiling_scores": tiling_scores,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "size_hints": getattr(autotuner, "size_hints", None),
        "heuristic_type": repr(getattr(autotuner, "heuristic_type", None)) if schema == 1 else None,
        "candidate_configs": (
            [_config_fingerprint(module, launcher.config) for launcher in launchers]
            if schema == 1
            else None
        ),
    }
    if schema == 2:
        try:
            # Local device ordinals do not describe a specialization. Hardware
            # properties, source and each compiled candidate still must match.
            device = dict(autotuner.device_props._asdict())
            device.pop("index", None)
            triton_meta = dict(autotuner.triton_meta)
            triton_meta.pop("device", None)
            torch_key = sys.modules["torch._inductor.codecache"].torch_key()
            if not isinstance(torch_key, bytes) or not torch_key:
                return None
            descriptor.update(
                schema=2,
                torch_key=torch_key.hex(),
                device=device,
                triton_meta=triton_meta,
                inductor_meta=metadata,
                heuristic_type=autotuner.heuristic_type,
                candidate_configs=[
                    _launcher_fingerprint(module, launcher, schema) for launcher in launchers
                ],
            )
            descriptor_json = json.dumps(
                _stable_cache_value(descriptor), separators=(",", ":"), allow_nan=False
            )
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            return None  # unsupported specialization: keep upstream autotuning
    else:
        descriptor_json = json.dumps(
            descriptor, sort_keys=True, separators=(",", ":"), default=repr
        )
    cache_key = hashlib.sha256(descriptor_json.encode()).hexdigest()
    path = os.path.join(cache_root, f"mcore_late_autotune_v{schema}", f"{cache_key}.json")
    return path, cache_key, descriptor


def _late_cache_path(autotuner, module) -> tuple[str, str] | None:
    location = _late_cache_descriptor(autotuner, module)
    return location[:2] if location is not None else None


def _positive_score(value: Any) -> bool:
    return type(value) in (int, float) and 0 < value < float("inf")


def _score_key(descriptor: dict[str, Any], score: int | float) -> str:
    changed = dict(descriptor)
    changed["tiling_scores"] = {"x": score}
    changed["inductor_meta"] = dict(descriptor["inductor_meta"], tiling_scores={"x": score})
    encoded = json.dumps(_stable_cache_value(changed), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _equivalence_location(path: str, descriptor: dict[str, Any]) -> tuple[str, int | float] | None:
    """Limit score equivalence to opted-in v2 pointwise x-only metadata."""
    if os.environ.get("MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE") != "1":
        return None
    scores = descriptor["tiling_scores"]
    heuristic = descriptor["heuristic_type"]
    if (
        descriptor["schema"] != 2
        or not isinstance(heuristic, Enum)
        or heuristic.name != "POINTWISE"
        or type(scores) is not dict
        or set(scores) != {"x"}
        or not _positive_score(scores["x"])
    ):
        return None
    return (
        os.path.join(os.path.dirname(path), "equivalent", _score_key(descriptor, 0)),
        scores["x"],
    )


def _equivalent_name(payload: dict[str, Any]) -> str:
    winner_hash = hashlib.sha256(payload["selected_config"].encode()).hexdigest()
    return f"{payload['cache_key']}-{winner_hash}.json"


def _atomic_payload(path: str, payload: dict[str, Any]) -> bool:
    temporary = None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=os.path.dirname(path), prefix=".autotune-", delete=False
        ) as cache_file:
            temporary = cache_file.name
            json.dump(payload, cache_file, sort_keys=True, allow_nan=False)
        os.replace(temporary, path)
        return True
    except (OSError, TypeError, ValueError):
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        return False


def _index_equivalent(location: tuple[str, int | float] | None, payload: dict[str, Any]) -> None:
    if location is None:
        return
    directory, score = location
    observation = dict(payload, tiling_score_x=score)
    path = os.path.join(directory, _equivalent_name(observation))
    # Keep conflicting winners in different files. Existing corrupt records
    # remain a miss rather than silently being repaired by another observation.
    if not os.path.exists(path) and not _atomic_payload(path, observation):
        _COUNTERS["late_equivalent_invalid"] += 1


def _read_equivalent(
    directory: str, descriptor: dict[str, Any], fingerprints: tuple[str, ...]
) -> str | None:
    """Require a unanimous winner backed by intact strict-key observations."""
    try:
        names = sorted(name for name in os.listdir(directory) if name.endswith(".json"))
    except FileNotFoundError:
        return None
    except OSError:
        _COUNTERS["late_equivalent_invalid"] += 1
        return None
    winners = set()
    try:
        for name in names:
            with open(os.path.join(directory, name), encoding="utf-8") as cache_file:
                payload = json.load(cache_file)
            if not isinstance(payload, dict) or not _positive_score(payload.get("tiling_score_x")):
                raise ValueError("Unsupported score observation")
            score = payload["tiling_score_x"]
            selected = payload.get("selected_config")
            # Reconstructing the original strict key with only the recorded x
            # score substituted proves source, compiler, hardware, specialization
            # and every ordered compiled candidate are otherwise identical.
            if (
                payload.get("schema") != 2
                or payload.get("cache_key") != _score_key(descriptor, score)
                or payload.get("kernel_name") != descriptor["kernel_name"]
                or fingerprints.count(selected) != 1
                or name != _equivalent_name(payload)
            ):
                raise ValueError("Incompatible score observation")
            exact_path = os.path.join(
                os.path.dirname(os.path.dirname(directory)), f"{payload['cache_key']}.json"
            )
            with open(exact_path, encoding="utf-8") as cache_file:
                exact = json.load(cache_file)
            if exact != {key: value for key, value in payload.items() if key != "tiling_score_x"}:
                raise ValueError("Missing, changed or corrupt exact observation")
            winners.add(selected)
    except (OSError, TypeError, ValueError):
        _COUNTERS["late_equivalent_invalid"] += 1
        return None
    if len(winners) > 1:
        _COUNTERS["late_equivalent_ambiguous"] += 1
        return None
    return next(iter(winners)) if winners else None


def _read_late_cache(autotuner, module) -> bool:
    """Select an exact current launcher from a persisted late-autotune result."""
    location = _late_cache_descriptor(autotuner, module)
    if location is None:
        _COUNTERS["late_invalid"] += 1
        return False
    path, cache_key, descriptor = location
    schema = descriptor["schema"]
    fingerprints = tuple(
        _launcher_fingerprint(module, launcher, schema) for launcher in autotuner.launchers
    )
    equivalent = _equivalence_location(path, descriptor)
    _LATE_PENDING[id(autotuner)] = (path, cache_key, schema, fingerprints, equivalent)
    payload = None
    try:
        with open(path, encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except FileNotFoundError:
        selected = _read_equivalent(equivalent[0], descriptor, fingerprints) if equivalent else None
        if selected is None:
            _COUNTERS["late_miss"] += 1
            return False
        _COUNTERS["late_equivalent_hit"] += 1
    except (OSError, ValueError):
        _COUNTERS["late_invalid"] += 1
        return False
    else:
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != schema
            or payload.get("cache_key") != cache_key
        ):
            _COUNTERS["late_invalid"] += 1
            return False
        selected = payload.get("selected_config")
    matches = [
        launcher
        for launcher, fingerprint in zip(autotuner.launchers, fingerprints)
        if fingerprint == selected
    ]
    if len(matches) != 1:
        _COUNTERS["late_invalid"] += 1
        return False

    # Only independently observed exact results contribute to the index;
    # score-equivalent reuses must not masquerade as new benchmark evidence.
    if payload is not None:
        _index_equivalent(equivalent, payload)
    autotuner.launchers = matches
    _LATE_PENDING.pop(id(autotuner), None)
    _COUNTERS["late_hit"] += 1
    return True


def _write_late_cache(autotuner, module) -> bool:
    """Atomically persist the launcher's runtime-selected exact config."""
    location = _LATE_PENDING.pop(id(autotuner), None)
    if location is None or len(getattr(autotuner, "launchers", ())) != 1:
        return False
    path, cache_key, schema, fingerprints, equivalent = location
    try:
        selected = _launcher_fingerprint(module, autotuner.launchers[0], schema)
    except (AttributeError, TypeError, ValueError):
        _COUNTERS["late_invalid"] += 1
        return False
    if schema == 2 and fingerprints.count(selected) != 1:
        _COUNTERS["late_invalid"] += 1
        return False
    payload = {
        "schema": schema,
        "cache_key": cache_key,
        "kernel_name": autotuner.inductor_meta.get("kernel_name"),
        "selected_config": selected,
    }
    if not _atomic_payload(path, payload):
        _COUNTERS["late_invalid"] += 1
        return False
    _index_equivalent(equivalent, payload)
    _COUNTERS["late_store"] += 1
    return True


def _apply_patch(module) -> None:
    global _INSTALLED, _REPORT_REGISTERED
    if _INSTALLED or not _enabled():
        return
    cache_type = module.AutotuneCache
    original_read_best = cache_type.read_best

    def read_best(self, inductor_meta, configs):
        best = original_read_best(self, inductor_meta, configs)
        if best is not None:
            _COUNTERS["native_hit"] += 1
            return best
        if os.environ.get("MCORE_CACHE_MODE") == "write":
            return None  # seed upstream results; the runtime hook records winners

        payload = _find_payload(
            os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""), inductor_meta, self.configs_hash, module
        )
        if payload is not None:
            best = module._load_cached_autotuning(
                dict(payload), self.configs_hash, configs, inductor_meta
            )
            # Newer PyTorch may reconstruct an unmatched config. Semantic
            # fallback is weaker than an exact native cache key: require a
            # current candidate even when coordesc allows upstream expansion.
            if best is not None and not any(best is config for config in configs):
                best = None
        if best is None:
            _COUNTERS["fallback_miss"] += 1
        else:
            _COUNTERS["fallback_hit"] += 1
        return best

    cache_type.read_best = read_best
    _INSTALLED = True
    if not _REPORT_REGISTERED:
        atexit.register(_report)
        _REPORT_REGISTERED = True


def _apply_triton_patch(module) -> None:
    """Reuse exact runtime choices left after native and semantic lookups."""
    global _TRITON_INSTALLED
    if _TRITON_INSTALLED or not _enabled():
        return
    cache_type = module.CachingAutotuner
    original_precompile = cache_type.precompile
    original_autotune = cache_type.autotune_to_one_config

    def precompile(self, *args, **kwargs):
        result = original_precompile(self, *args, **kwargs)
        warm_cache_only = kwargs.get("warm_cache_only", False)
        if args:
            warm_cache_only = args[0]
        if not warm_cache_only and len(getattr(self, "launchers", ())) > 1:
            _read_late_cache(self, module)
        return result

    def autotune_to_one_config(self, *args, **kwargs):
        result = original_autotune(self, *args, **kwargs)
        _write_late_cache(self, module)
        return result

    cache_type.precompile = precompile
    cache_type.autotune_to_one_config = autotune_to_one_config
    _TRITON_INSTALLED = True


class _InductorCacheFinder(importlib.abc.MetaPathFinder):
    TARGETS = {
        "torch._inductor.runtime.autotune_cache": _apply_patch,
        "torch._inductor.runtime.triton_heuristics": _apply_triton_patch,
    }

    def find_spec(self, fullname, path, target=None):
        apply_patch = self.TARGETS.get(fullname)
        if apply_patch is None:
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
                    apply_patch(module)

            spec.loader = _Loader()
            return spec
        return None


def arm() -> None:
    """Patch on the first Inductor autotune-cache import."""
    global _ARMED
    if _ARMED or not _enabled():
        return
    finder = _InductorCacheFinder()
    for target, apply_patch in finder.TARGETS.items():
        if target in sys.modules:
            apply_patch(sys.modules[target])
    sys.meta_path.insert(0, finder)
    _ARMED = True


def install() -> None:
    """Patch an already-imported Inductor autotune-cache module, if present."""
    if not _enabled():
        return
    for target, apply_patch in _InductorCacheFinder.TARGETS.items():
        module = sys.modules.get(target)
        if module is not None:
            apply_patch(module)
