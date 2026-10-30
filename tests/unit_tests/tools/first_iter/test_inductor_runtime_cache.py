# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU coverage for runtime autotune replay after a native/semantic miss."""

import importlib
import json
import os
import sys
import types
from collections import namedtuple
from enum import Enum
from pathlib import Path

import pytest

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
PC_DIR = REPO_ROOT / "tools" / "persistent_cache"


class HeuristicType(Enum):
    POINTWISE = 1


DeviceProperties = namedtuple(
    "DeviceProperties",
    "type index multi_processor_count cc major regs_per_multiprocessor "
    "max_threads_per_multi_processor max_threads_per_block warp_size",
)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(PC_DIR))
    cache = importlib.import_module("inductor_autotune_restart_cache")
    importlib.reload(cache)
    monkeypatch.setenv("MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE", "1")
    monkeypatch.setenv("MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE", "1")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    compiler = types.SimpleNamespace(torch_key=lambda: b"installed-torch-key")
    monkeypatch.setitem(sys.modules, "torch._inductor.codecache", compiler)

    class Autotuner:
        def __init__(self, active_hook=True):
            self.saved = []
            self.save_cache_hook = self.saved.append if active_hook else None
            self.fn = types.SimpleNamespace(src="def kernel(x, XBLOCK):\n    return x + XBLOCK\n")
            self.device_props = DeviceProperties("cuda", 0, 152, 103, 10, 65536, 2048, 1024, 32)
            # Metadata shapes from the profiled copy/div kernel, including
            # tuple-keyed Triton attrs and set-valued Inductor hints.
            self.triton_meta = {
                "signature": {"in_ptr0": "*fp32", "xnumel": "i64", "XBLOCK": "constexpr"},
                "constants": {},
                "device": 0,
                "device_type": "cuda",
                "configs": [{(0,): [["tt.divisibility", 16]]}],
                "enable_fp_fusion": True,
                "launch_pdl": False,
                "disable_ftz": False,
            }
            self.inductor_meta = {
                "kernel_name": "triton_poi_fused_copy__div_split_unsqueeze_1",
                "backend_hash": "triton-and-backend-hash",
                "tiling_scores": {"x": 77347258368},
                "autotune_hints": set(),
                "assert_indirect_indexing": True,
            }
            self.size_hints = {"x": 8589934592}
            self.heuristic_type = HeuristicType.POINTWISE
            self.available = [
                types.SimpleNamespace(
                    config=types.SimpleNamespace(
                        kwargs={"XBLOCK": block}, num_warps=4, num_stages=1
                    ),
                    cache_hash=f"compiled-{block}",
                )
                for block in (512, 1024)
            ]
            self.launchers = []
            self.benchmarks = 0
            self.precompile_result = object()
            self.autotune_result = object()

        def precompile(self, warm_cache_only=False):
            self.launchers = [] if warm_cache_only else list(self.available)
            return self.precompile_result

        def autotune_to_one_config(self, *args, **kwargs):
            self.benchmarks += 1
            self.launchers = [self.launchers[-1]]
            if self.save_cache_hook:
                self.save_cache_hook(self.launchers[0].config)
            return self.autotune_result

    module = types.SimpleNamespace(
        CachingAutotuner=Autotuner,
        triton_config_to_hashable=lambda cfg: (
            tuple(sorted(cfg.kwargs.items())),
            cfg.num_warps,
            cfg.num_stages,
        ),
    )
    cache._apply_triton_patch(module)
    yield cache, module, compiler
    sys.modules.pop("inductor_autotune_restart_cache", None)


def _seed(runtime):
    cache, module, _ = runtime
    cold = module.CachingAutotuner()
    assert cold.precompile() is cold.precompile_result
    assert len(cold.launchers) == 2
    assert cold.autotune_to_one_config("unchanged", option=1) is cold.autotune_result
    assert cold.benchmarks == 1 and cold.saved == [cold.available[-1].config]
    assert cache._COUNTERS["late_store"] == 1
    return cold


def test_active_native_save_hook_records_then_replays_without_benchmark(runtime, tmp_path):
    cache, module, _ = runtime
    cold = _seed(runtime)
    files = list((tmp_path / "mcore_late_autotune_v2").glob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["schema"] == 2
    warm = module.CachingAutotuner()
    warm.precompile()
    assert warm.launchers == [warm.available[-1]]
    assert warm.launchers[0].config.kwargs == cold.launchers[0].config.kwargs
    assert warm.benchmarks == 0 and warm.saved == []
    assert cache._COUNTERS["late_hit"] == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda tuner: setattr(tuner.fn, "src", "def different_kernel(): pass"),
        lambda tuner: tuner.available[0].config.kwargs.update(XBLOCK=256),
        lambda tuner: tuner.available.reverse(),
        lambda tuner: setattr(tuner.available[0], "cache_hash", "changed-compiled-kernel"),
        lambda tuner: tuner.triton_meta["signature"].update(in_ptr0="*bf16"),
        lambda tuner: tuner.triton_meta["constants"].update(xnumel=32768),
        lambda tuner: tuner.triton_meta.update(configs=[{(0,): [["tt.divisibility", 8]]}]),
        lambda tuner: tuner.triton_meta.update(enable_fp_fusion=False),
        lambda tuner: setattr(tuner, "device_props", tuner.device_props._replace(cc=100)),
        lambda tuner: tuner.inductor_meta.update(backend_hash="changed-backend"),
        lambda tuner: tuner.inductor_meta.update(assert_indirect_indexing=False),
        lambda tuner: tuner.size_hints.update(x=32768),
    ],
    ids=[
        "source",
        "candidate",
        "candidate-order",
        "compiled-kernel",
        "signature",
        "constants",
        "attributes",
        "compile-option",
        "device",
        "backend",
        "assertions",
        "size",
    ],
)
def test_changed_compilation_identity_retunes(runtime, change):
    cache, module, _ = runtime
    _seed(runtime)
    warm = module.CachingAutotuner()
    change(warm)
    warm.precompile()
    assert len(warm.launchers) == 2
    warm.autotune_to_one_config()
    assert warm.benchmarks == 1 and len(warm.saved) == 1
    assert cache._COUNTERS["late_hit"] == 0


def test_changed_torch_version_retunes(runtime):
    cache, module, compiler = runtime
    _seed(runtime)
    compiler.torch_key = lambda: b"different-torch-key"
    warm = module.CachingAutotuner()
    warm.precompile()
    assert len(warm.launchers) == 2 and cache._COUNTERS["late_hit"] == 0


def test_device_ordinal_and_dictionary_order_do_not_change_specialization(runtime):
    cache, module, _ = runtime
    _seed(runtime)
    warm = module.CachingAutotuner()
    warm.device_props = warm.device_props._replace(index=3)
    warm.triton_meta = dict(reversed(list(warm.triton_meta.items())))
    warm.triton_meta["device"] = 3
    warm.precompile()
    assert warm.launchers == [warm.available[-1]] and cache._COUNTERS["late_hit"] == 1


@pytest.mark.parametrize("unsupported", [object(), float("nan")])
def test_unsupported_specialization_retunes_without_writing(runtime, tmp_path, unsupported):
    cache, module, _ = runtime
    tuner = module.CachingAutotuner()
    tuner.triton_meta["constants"]["unsupported"] = unsupported
    tuner.precompile()
    tuner.autotune_to_one_config()
    assert tuner.benchmarks == 1 and len(tuner.saved) == 1
    assert cache._COUNTERS["late_invalid"] == 1 and cache._COUNTERS["late_store"] == 0
    assert not (tmp_path / "mcore_late_autotune_v2").exists()


@pytest.mark.parametrize("warm_cache_only", [False, True])
def test_native_single_candidate_and_compile_worker_bypass(runtime, warm_cache_only):
    cache, module, _ = runtime
    tuner = module.CachingAutotuner()
    if not warm_cache_only:
        tuner.available = tuner.available[:1]
    tuner.precompile(warm_cache_only)
    assert not cache._LATE_PENDING
    assert all(
        cache._COUNTERS[key] == 0 for key in ("late_hit", "late_miss", "late_invalid", "late_store")
    )


@pytest.mark.parametrize("corruption", ["foreign-winner", "non-object", "schema"])
def test_invalid_runtime_entry_keeps_current_candidates(runtime, tmp_path, corruption):
    cache, module, _ = runtime
    _seed(runtime)
    path = next((tmp_path / "mcore_late_autotune_v2").glob("*.json"))
    payload = json.loads(path.read_text())
    if corruption == "foreign-winner":
        payload["selected_config"] = "not-a-current-compiled-launcher"
    elif corruption == "schema":
        payload["schema"] = 1
    else:
        payload = []
    path.write_text(json.dumps(payload))
    warm = module.CachingAutotuner()
    warm.precompile()
    assert len(warm.launchers) == 2 and cache._COUNTERS["late_invalid"] == 1


def test_new_launcher_not_in_precompiled_set_is_not_stored(runtime, tmp_path):
    cache, module, _ = runtime
    tuner = module.CachingAutotuner()
    tuner.precompile()
    tuner.launchers = [
        types.SimpleNamespace(config=tuner.available[0].config, cache_hash="foreign-binary")
    ]
    assert not cache._write_late_cache(tuner, module)
    assert cache._COUNTERS["late_invalid"] == 1
    assert not (tmp_path / "mcore_late_autotune_v2").exists()


def test_v1_dynamic_cache_remains_readable(runtime, tmp_path):
    cache, module, _ = runtime
    cold = module.CachingAutotuner(active_hook=False)
    cold.precompile()
    cold.autotune_to_one_config()
    path = next((tmp_path / "mcore_late_autotune_v1").glob("*.json"))
    payload = json.loads(path.read_text())
    assert payload["schema"] == 1
    assert payload["selected_config"] == repr(
        module.triton_config_to_hashable(cold.launchers[0].config)
    )
    warm = module.CachingAutotuner(active_hook=False)
    warm.precompile()
    assert warm.launchers == [warm.available[-1]] and cache._COUNTERS["late_hit"] == 1


def test_score_equivalence_requires_opt_in(runtime, monkeypatch):
    cache, module, _ = runtime
    monkeypatch.setenv("MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE", "1")
    _seed(runtime)
    monkeypatch.delenv("MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE")
    warm = module.CachingAutotuner()
    warm.inductor_meta["tiling_scores"]["x"] = 13981890560
    warm.precompile()
    assert len(warm.launchers) == 2 and cache._COUNTERS["late_equivalent_hit"] == 0


def test_changed_score_reuses_same_compiled_winner_without_claiming_new_observation(
    runtime, tmp_path
):
    cache, module, _ = runtime
    _seed(runtime)
    warm = module.CachingAutotuner()
    warm.inductor_meta["tiling_scores"]["x"] = 13981890560
    warm.precompile()
    assert warm.launchers == [warm.available[-1]]
    assert warm.benchmarks == 0 and warm.saved == []
    assert cache._COUNTERS["late_hit"] == cache._COUNTERS["late_equivalent_hit"] == 1
    assert len(list((tmp_path / "mcore_late_autotune_v2").glob("*.json"))) == 1
    assert len(list((tmp_path / "mcore_late_autotune_v2/equivalent").glob("*/*.json"))) == 1


def test_old_record_needs_exact_hit_before_score_equivalence(runtime, monkeypatch, tmp_path):
    cache, module, _ = runtime
    monkeypatch.delenv("MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE")
    _seed(runtime)
    assert not (tmp_path / "mcore_late_autotune_v2/equivalent").exists()
    monkeypatch.setenv("MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE", "1")
    unseen = module.CachingAutotuner()
    unseen.inductor_meta["tiling_scores"]["x"] = 13981890560
    unseen.precompile()
    assert len(unseen.launchers) == 2
    exact = module.CachingAutotuner()
    exact.precompile()
    assert exact.launchers == [exact.available[-1]]
    shifted = module.CachingAutotuner()
    shifted.inductor_meta["tiling_scores"]["x"] = 13981890560
    shifted.precompile()
    assert shifted.launchers == [shifted.available[-1]]
    assert cache._COUNTERS["late_equivalent_hit"] == 1


@pytest.mark.parametrize("score", [0, -1, True, "13981890560", {"x": 1}])
def test_unsupported_score_is_exact_only(runtime, score, tmp_path):
    cache, module, _ = runtime
    _seed(runtime)
    tuner = module.CachingAutotuner()
    tuner.inductor_meta["tiling_scores"]["x"] = score
    tuner.precompile()
    assert len(tuner.launchers) == 2
    tuner.autotune_to_one_config()
    assert cache._COUNTERS["late_equivalent_hit"] == 0
    assert len(list((tmp_path / "mcore_late_autotune_v2/equivalent").glob("*/*.json"))) == 1


def test_other_tiling_axes_are_not_ignored(runtime, tmp_path):
    cache, module, _ = runtime
    _seed(runtime)
    tuner = module.CachingAutotuner()
    tuner.inductor_meta["tiling_scores"].update(x=13981890560, y=1)
    tuner.precompile()
    assert len(tuner.launchers) == 2
    tuner.autotune_to_one_config()
    assert cache._COUNTERS["late_equivalent_hit"] == 0
    assert len(list((tmp_path / "mcore_late_autotune_v2/equivalent").glob("*/*.json"))) == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "bad-json",
        "wrong-score",
        "foreign-winner",
        "missing-exact",
        "changed-exact",
        "bad-exact-json",
    ],
)
def test_corrupt_or_unbacked_score_observation_retunes(runtime, tmp_path, corruption):
    cache, module, _ = runtime
    _seed(runtime)
    exact = next((tmp_path / "mcore_late_autotune_v2").glob("*.json"))
    sidecar = next((tmp_path / "mcore_late_autotune_v2/equivalent").glob("*/*.json"))
    payload = json.loads(sidecar.read_text())
    if corruption == "bad-json":
        sidecar.write_text("{")
    elif corruption == "missing-exact":
        exact.unlink()
    elif corruption == "bad-exact-json":
        exact.write_text("{")
    elif corruption == "changed-exact":
        observed = json.loads(exact.read_text())
        observed["selected_config"] = "changed"
        exact.write_text(json.dumps(observed))
    else:
        if corruption == "wrong-score":
            payload["tiling_score_x"] += 1
        else:
            payload["selected_config"] = "foreign"
        sidecar.write_text(json.dumps(payload))
    tuner = module.CachingAutotuner()
    tuner.inductor_meta["tiling_scores"]["x"] = 13981890560
    tuner.precompile()
    assert len(tuner.launchers) == 2
    assert cache._COUNTERS["late_equivalent_invalid"] == 1
    assert cache._COUNTERS["late_equivalent_hit"] == 0


def _independent_observation(runtime, monkeypatch, score, winner):
    """Mimic a independently tuned result merged from another seed process."""
    cache, module, _ = runtime
    monkeypatch.setenv("MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE", "0")
    tuner = module.CachingAutotuner()
    tuner.inductor_meta["tiling_scores"]["x"] = score
    tuner.precompile()
    tuner.launchers = [tuner.available[winner]]
    assert cache._write_late_cache(tuner, module)
    monkeypatch.setenv("MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE", "1")
    exact = module.CachingAutotuner()
    exact.inductor_meta["tiling_scores"]["x"] = score
    exact.precompile()
    return exact


def test_conflicting_observed_winners_retune_but_exact_stays_first(runtime, monkeypatch, tmp_path):
    cache, module, _ = runtime
    _seed(runtime)
    _independent_observation(runtime, monkeypatch, 13981890560, 0)
    assert len(list((tmp_path / "mcore_late_autotune_v2/equivalent").glob("*/*.json"))) == 2
    unseen = module.CachingAutotuner()
    unseen.inductor_meta["tiling_scores"]["x"] = 14567091200
    unseen.precompile()
    assert len(unseen.launchers) == 2
    assert cache._COUNTERS["late_equivalent_ambiguous"] == 1
    exact = module.CachingAutotuner()
    exact.precompile()
    assert exact.launchers == [exact.available[-1]]
    assert cache._COUNTERS["late_equivalent_hit"] == 0


def test_same_key_conflict_is_retained_and_invalidates_equivalence(runtime, monkeypatch, tmp_path):
    cache, module, _ = runtime
    cold = _seed(runtime)
    # A second independent observation for the same key picked another winner.
    exact_path = next((tmp_path / "mcore_late_autotune_v2").glob("*.json"))
    exact_path.unlink()
    _independent_observation(runtime, monkeypatch, cold.inductor_meta["tiling_scores"]["x"], 0)
    assert len(list((tmp_path / "mcore_late_autotune_v2/equivalent").glob("*/*.json"))) == 2
    unseen = module.CachingAutotuner()
    unseen.inductor_meta["tiling_scores"]["x"] = 14567091200
    unseen.precompile()
    assert len(unseen.launchers) == 2
    assert cache._COUNTERS["late_equivalent_invalid"] == 1


def test_unanimous_independent_observations_allow_equivalence(runtime, monkeypatch):
    cache, module, _ = runtime
    _seed(runtime)
    _independent_observation(runtime, monkeypatch, 13981890560, -1)
    unseen = module.CachingAutotuner()
    unseen.inductor_meta["tiling_scores"]["x"] = 14567091200
    unseen.precompile()
    assert unseen.launchers == [unseen.available[-1]]
    assert cache._COUNTERS["late_equivalent_hit"] == 1


def test_corrupt_index_cannot_override_exact_hit(runtime, tmp_path):
    cache, module, _ = runtime
    _seed(runtime)
    sidecar = next((tmp_path / "mcore_late_autotune_v2/equivalent").glob("*/*.json"))
    sidecar.write_text("{")
    exact = module.CachingAutotuner()
    exact.precompile()
    assert exact.launchers == [exact.available[-1]]
    assert cache._COUNTERS["late_hit"] == 1
    assert cache._COUNTERS["late_equivalent_hit"] == 0
    assert sidecar.read_text() == "{"


def test_non_pointwise_heuristic_remains_exact_only(runtime, tmp_path):
    cache, module, _ = runtime

    class Reduction(Enum):
        REDUCTION = 1

    cold = module.CachingAutotuner()
    cold.heuristic_type = Reduction.REDUCTION
    cold.precompile()
    cold.autotune_to_one_config()
    assert not (tmp_path / "mcore_late_autotune_v2/equivalent").exists()
    warm = module.CachingAutotuner()
    warm.heuristic_type = Reduction.REDUCTION
    warm.inductor_meta["tiling_scores"]["x"] = 13981890560
    warm.precompile()
    assert len(warm.launchers) == 2
    assert cache._COUNTERS["late_equivalent_hit"] == 0


def test_index_write_failure_preserves_exact_hit(runtime, monkeypatch, tmp_path):
    cache, module, _ = runtime
    monkeypatch.delenv("MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE")
    _seed(runtime)
    monkeypatch.setenv("MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE", "1")
    monkeypatch.setattr(cache, "_atomic_payload", lambda *args: False)
    warm = module.CachingAutotuner()
    warm.precompile()
    assert warm.launchers == [warm.available[-1]]
    assert cache._COUNTERS["late_hit"] == 1
    assert cache._COUNTERS["late_equivalent_invalid"] == 1
    assert not (tmp_path / "mcore_late_autotune_v2/equivalent").exists()


def test_index_serialization_failure_preserves_exact_hit(runtime, monkeypatch, tmp_path):
    cache, module, _ = runtime
    monkeypatch.delenv("MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE")
    _seed(runtime)
    exact_path = next((tmp_path / "mcore_late_autotune_v2").glob("*.json"))
    payload = json.loads(exact_path.read_text())
    # json.load accepts this extra field, while the optional index writer uses
    # strict JSON. Rejecting that sidecar must not break the valid exact winner.
    payload["extra"] = float("nan")
    original = json.dumps(payload)
    exact_path.write_text(original)
    monkeypatch.setenv("MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE", "1")

    warm = module.CachingAutotuner()
    assert warm.precompile() is warm.precompile_result
    assert warm.launchers == [warm.available[-1]]
    assert warm.benchmarks == 0 and warm.saved == []
    assert cache._COUNTERS["late_hit"] == 1
    assert cache._COUNTERS["late_equivalent_invalid"] == 1
    assert exact_path.read_text() == original
    assert not list((tmp_path / "mcore_late_autotune_v2/equivalent").rglob("*.json"))
    assert not list(tmp_path.rglob(".autotune-*"))
