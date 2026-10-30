import json
import os
import subprocess
import sys
from pathlib import Path

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]


def test_cache_sitecustomize_keeps_stdout_protocol_clean():
    repo_root = REPO_ROOT
    shim = repo_root / "tools" / "persistent_cache" / "_triton_patch_shim"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(shim)
    env["MCORE_CUTE_DSL_FILE_CACHE"] = "1"
    env["MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", "print('protocol-output')"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout == "protocol-output\n"


def test_cache_sitecustomize_does_not_eager_import_training_stacks(tmp_path):
    repo_root = REPO_ROOT
    shim = repo_root / "tools" / "persistent_cache" / "_triton_patch_shim"
    fake_modules = tmp_path / "fake_modules"
    for package in ("triton/runtime", "cutlass/base_dsl"):
        package_dir = fake_modules / package
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("")
    (fake_modules / "triton/runtime/autotuner.py").write_text("class Autotuner:\n    pass\n")
    (fake_modules / "cutlass/base_dsl/compiler.py").write_text(
        "class CompileCallable:\n    def _compile(self, *args, **kwargs): pass\n"
    )
    (fake_modules / "cutlass/base_dsl/dsl.py").write_text(
        "class BaseDSL:\n"
        "    def _func(self, *args, **kwargs): pass\n"
        "    def generate_mlir(self, *args, **kwargs): pass\n"
    )
    (fake_modules / "deep_ep.py").write_text(
        "class HybridEPBuffer:\n    def __init__(self, *args, **kwargs): pass\n"
    )
    hybrid_cache = tmp_path / "hybrid_cache/.deepep/hybrid_ep/jit/proc-seed"
    hybrid_cache.mkdir(parents=True)
    (hybrid_cache / "seed.so").write_bytes(b"test")
    stale_cache = hybrid_cache.parent / "proc-stale/stale.so"
    stale_cache.parent.mkdir()
    stale_cache.symlink_to(tmp_path / "missing.so")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(shim), str(fake_modules)))
    env["MCORE_CUTE_DSL_FILE_CACHE"] = "1"
    env["MCORE_CUTE_DSL_AOT_CACHE"] = "1"
    env["MCORE_CUTE_DSL_AOT_NAMESPACE"] = "startup-test"
    env["CUTE_DSL_CACHE_DIR"] = str(tmp_path / "cute_cache")
    env["MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE"] = "1"
    env["HYBRID_EP_LOAD_CACHED_KERNELS"] = "1"
    env["HYBRID_EP_CACHE_DIR"] = str(tmp_path / "hybrid_cache")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "before = {name: any(module == name or "
                "module.startswith(name + '.') for module in sys.modules) "
                "for name in ('torch', 'triton', 'cutlass', 'deep_ep')}; "
                "import cutlass.base_dsl.dsl as dsl; "
                "import cutlass.base_dsl.compiler as compiler; "
                "import deep_ep; "
                "print(json.dumps({'before': before, 'patched': {"
                "'cute_dsl': bool(getattr(dsl.BaseDSL._func, "
                "'_mcore_persistent_cache', False)), "
                "'cute_aot': bool(getattr(compiler.CompileCallable._compile, "
                "'_mcore_aot_cache', False)), "
                "'hybrid_ep': bool(getattr(deep_ep.HybridEPBuffer.__init__, "
                "'_mcore_persistent_cache', False))}}))"
            ),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload == {
        "before": {"torch": False, "triton": False, "cutlass": False, "deep_ep": False},
        "patched": {"cute_dsl": True, "cute_aot": True, "hybrid_ep": True},
    }, (payload, result.stderr)
    assert stale_cache.is_symlink()


def test_prewarm_hook_is_lazy_and_runs_through_imported_entrypoint(tmp_path):
    package = tmp_path / "megatron" / "training"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "initialize.py").write_text(
        "calls = []\n"
        "def get_args(): return 'recipe'\n"
        "def _warmup_jit_function(): calls.append('native')\n"
        "def set_jit_fusion_options(): _warmup_jit_function()\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT / "tools/persistent_cache/_triton_patch_shim"), str(tmp_path))
    )
    env["MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; assert 'torch' not in sys.modules; "
                "assert 'megatron.training.initialize' not in sys.modules; "
                "import inductor_prewarm; "
                "from megatron.training.initialize import set_jit_fusion_options; "
                "import megatron.training.initialize as init; "
                "inductor_prewarm.warmup = lambda args: init.calls.append(args); "
                "set_jit_fusion_options(); set_jit_fusion_options(); "
                "assert init.calls == ['native', 'recipe', 'native']; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        env=env,
        cwd=tmp_path,
        text=True,
    )
    assert "setup failed" not in result.stderr
