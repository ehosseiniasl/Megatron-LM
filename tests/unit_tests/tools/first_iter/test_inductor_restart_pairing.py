# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Pairing rules of the Inductor autotune restart fallback (no torch needed)."""

import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
PC_DIR = REPO_ROOT / "tools" / "persistent_cache"

META_A = {"kernel_name": "triton_red_fused_a", "backend_hash": "BH", "tiling_scores": {"x": 1}}
META_B = {"kernel_name": "triton_red_fused_b", "backend_hash": "BH", "tiling_scores": {"x": 1}}


def _source_text(meta: dict) -> str:
    return (
        "inductor_meta = {"
        + ", ".join(f"{key!r}: {value!r}" for key, value in meta.items())
        + "}\n"
    )


@pytest.fixture
def restart_cache():
    sys.path.insert(0, str(PC_DIR))
    try:
        module = importlib.import_module("inductor_autotune_restart_cache")
        importlib.reload(module)
        yield module
    finally:
        sys.modules.pop("inductor_autotune_restart_cache", None)
        sys.path.remove(str(PC_DIR))


def _exact_key_module():
    """Stand-in for torch._inductor.runtime.autotune_cache with the key functions."""

    class AutotuneCache:
        @staticmethod
        def _prepare_key(filename):
            return os.path.basename(filename).removesuffix(".py") + ":tag"

        @staticmethod
        def _make_local_cache_key(dirname, key):
            return os.path.join(dirname, f"{key.replace(':', '_')}_torchkey.best_config")

    return types.SimpleNamespace(AutotuneCache=AutotuneCache)


def _populate(root: Path, same_hash: bool = True):
    """Two kernels sharing one directory, as an FX-graph cache dir does."""
    d = root / "ab"
    d.mkdir(parents=True)
    (d / "srcA.py").write_text(_source_text(META_A))
    (d / "srcB.py").write_text(_source_text(META_B))
    hash_b = "cfg" if same_hash else "other"
    (d / "srcA_tag_torchkey.best_config").write_text(
        json.dumps({"configs_hash": "cfg", "XBLOCK": 128, "name": "A"})
    )
    (d / "srcB_tag_torchkey.best_config").write_text(
        json.dumps({"configs_hash": hash_b, "XBLOCK": 64, "name": "B"})
    )
    return d


def test_exact_pairing_returns_only_the_sources_own_result(tmp_path, restart_cache):
    _populate(tmp_path)
    module = _exact_key_module()
    found = restart_cache._find_payload(str(tmp_path), META_A, "cfg", module)
    assert found is not None and found["name"] == "A"
    found = restart_cache._find_payload(str(tmp_path), META_B, "cfg", module)
    assert found is not None and found["name"] == "B"
    # a kernel whose source is present but whose own result is missing is a miss,
    # even though another result with the same configs_hash sits beside it
    d = tmp_path / "ab"
    (d / "srcB_tag_torchkey.best_config").unlink()
    restart_cache._SOURCE_RECORDS.clear()
    restart_cache._LOOKUP_CACHE.clear()
    assert restart_cache._find_payload(str(tmp_path), META_B, "cfg", module) is None


def test_without_key_functions_ambiguous_matches_are_refused(tmp_path, restart_cache):
    """Older/newer torch without the key helpers: never return a guess."""
    d = _populate(tmp_path)
    module = types.SimpleNamespace(AutotuneCache=type("AutotuneCache", (), {}))
    # two results with the same configs_hash share the directory: a pick would be a coin toss
    assert restart_cache._find_payload(str(tmp_path), META_A, "cfg", module) is None
    assert restart_cache._COUNTERS["fallback_ambiguous"] == 1
    # once the neighbour's configs_hash differs, the single remaining match is accepted
    restart_cache._SOURCE_RECORDS.clear()
    restart_cache._LOOKUP_CACHE.clear()
    (d / "srcB_tag_torchkey.best_config").write_text(
        json.dumps({"configs_hash": "other", "name": "B"})
    )
    found = restart_cache._find_payload(str(tmp_path), META_A, "cfg", module)
    assert found is not None and found["name"] == "A"


def test_no_module_at_all_still_matches_unambiguous_directories(tmp_path, restart_cache):
    _populate(tmp_path, same_hash=False)
    found = restart_cache._find_payload(str(tmp_path), META_A, "cfg", None)
    assert found is not None and found["name"] == "A"


def _paired_copies(root, payloads):
    """One kernel is tuned on different ranks before their caches are merged."""
    module = _exact_key_module()
    for index, payload in enumerate(payloads):
        directory = root / str(index)
        directory.mkdir()
        source_name = f"rank{index}.py"
        (directory / source_name).write_text(_source_text(META_A))
        key = module.AutotuneCache._prepare_key(directory / source_name)
        config_path = module.AutotuneCache._make_local_cache_key(directory, key)
        Path(config_path).write_text(json.dumps(payload))
    return module


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"time_taken_ms": 72},
        {"triton_cache_hash": "different-compiled-artifact"},
        {"time_taken_ms": 81, "triton_cache_hash": "different-compiled-artifact"},
    ],
)
def test_rank_copies_with_identical_configs_are_cache_hits(tmp_path, restart_cache, diagnostics):
    # These fields describe the tuning run/artifact, not the selected launch.
    # Real consolidated caches contain equal configs with different values.
    config = {"configs_hash": "cfg", "XBLOCK": 1024, "num_warps": 4, "num_stages": 1}
    module = _paired_copies(tmp_path, [config, {**config, **diagnostics}])
    found = restart_cache._find_payload(str(tmp_path), META_A, "cfg", module)
    assert found == config
    assert restart_cache._COUNTERS["fallback_ambiguous"] == 0
    assert restart_cache._find_payload(str(tmp_path), META_A, "cfg", module) == config
    # Lookup must not mutate the stored source records used by other lookups.
    records = restart_cache._SOURCE_RECORDS[str(tmp_path)]
    assert any(
        payload == {**config, **diagnostics} for _, payloads in records for payload in payloads
    )


@pytest.mark.parametrize(
    "conflict",
    [
        {"XBLOCK": 512},
        {"num_warps": 8},
        {"num_stages": 2},
        {"found_by_coordesc": True},
        {"extra_options": {"enable_fp_fusion": False}},
        {"unknown_future_option": True},
    ],
)
def test_rank_copies_with_conflicting_configs_still_miss(tmp_path, restart_cache, conflict):
    config = {
        "configs_hash": "cfg",
        "XBLOCK": 1024,
        "num_warps": 4,
        "num_stages": 1,
        "found_by_coordesc": False,
        "time_taken_ms": 72,
        "triton_cache_hash": "compiled-artifact",
    }
    module = _paired_copies(tmp_path, [config, {**config, **conflict, "time_taken_ms": 81}])
    assert restart_cache._find_payload(str(tmp_path), META_A, "cfg", module) is None
    assert restart_cache._COUNTERS["fallback_ambiguous"] == 1


def test_different_candidate_sets_do_not_create_a_conflict(tmp_path, restart_cache):
    config = {"configs_hash": "cfg", "XBLOCK": 1024, "num_warps": 4, "num_stages": 1}
    module = _paired_copies(tmp_path, [config, {**config, "configs_hash": "other", "XBLOCK": 512}])
    assert restart_cache._find_payload(str(tmp_path), META_A, "cfg", module) == config
    assert restart_cache._COUNTERS["fallback_ambiguous"] == 0


@pytest.mark.parametrize(
    "current_candidate, coordesc_enabled, found_by_coordesc, expect_hit",
    [
        (True, False, False, True),
        (False, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, True, True, False),
    ],
)
def test_fallback_reconstruction_requires_a_current_candidate(
    tmp_path,
    restart_cache,
    monkeypatch,
    current_candidate,
    coordesc_enabled,
    found_by_coordesc,
    expect_hit,
):
    payload = {"configs_hash": "cfg", "XBLOCK": 1024, "found_by_coordesc": found_by_coordesc}
    module = _paired_copies(tmp_path, [payload])
    candidate = object()
    selected = candidate if current_candidate else object()
    # Installed Torch can now reconstruct configs absent from the current set.
    module.AutotuneCache.read_best = lambda self, meta, configs: None
    module._load_cached_autotuning = lambda payload, cfg_hash, configs, meta: selected
    monkeypatch.setenv("MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE", "1")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(restart_cache, "_REPORT_REGISTERED", True)
    restart_cache._apply_patch(module)
    cache = module.AutotuneCache()
    cache.configs_hash = "cfg"
    meta = {**META_A, "coordinate_descent_tuning": coordesc_enabled}
    assert cache.read_best(meta, [candidate]) is (selected if expect_hit else None)
    assert restart_cache._COUNTERS["fallback_hit"] == int(expect_hit)
    assert restart_cache._COUNTERS["fallback_miss"] == int(not expect_hit)


def test_native_hit_does_not_apply_the_fallback_candidate_guard(restart_cache, monkeypatch):
    module = _exact_key_module()
    native_result = object()
    module.AutotuneCache.read_best = lambda self, meta, configs: native_result
    monkeypatch.setenv("MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE", "1")
    monkeypatch.setattr(restart_cache, "_REPORT_REGISTERED", True)
    restart_cache._apply_patch(module)
    assert module.AutotuneCache().read_best(META_A, []) is native_result
    assert restart_cache._COUNTERS["native_hit"] == 1
    assert restart_cache._COUNTERS["fallback_miss"] == 0


def test_write_mode_skips_semantic_replay_but_registers_runtime_stats(restart_cache, monkeypatch):
    module = _exact_key_module()
    module.AutotuneCache.read_best = lambda self, meta, configs: None
    reports = []
    monkeypatch.setenv("MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE", "1")
    monkeypatch.setenv("MCORE_CACHE_MODE", "write")
    monkeypatch.setattr(restart_cache.atexit, "register", reports.append)

    def unexpected_lookup(*args):
        raise AssertionError("write mode must seed native results")

    monkeypatch.setattr(restart_cache, "_find_payload", unexpected_lookup)
    restart_cache._apply_patch(module)
    assert module.AutotuneCache().read_best(META_A, []) is None
    assert reports == [restart_cache._report]
    assert restart_cache._COUNTERS["fallback_miss"] == 0
