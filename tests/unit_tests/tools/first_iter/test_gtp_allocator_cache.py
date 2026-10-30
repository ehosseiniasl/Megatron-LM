# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU tests of the allocator loader with a mock compiler.

Run this file directly with Python to avoid the GPU suite's distributed fixtures.
These checks do not validate the compiled allocator or CUDA/NCCL behavior.
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
SOURCE = REPO_ROOT / "megatron/core/tensor_parallel/gtp_vmm_allocator.py"


def _load_module(load):
    """Import the actual module without importing Torch or initializing CUDA."""
    names = (
        "torch",
        "torch.utils",
        "torch.utils.cpp_extension",
        "megatron",
        "megatron.core",
        "megatron.core.nccl_allocator",
        "megatron.core.utils",
    )
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        module.__path__ = []
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    modules["torch"].cuda = types.SimpleNamespace(MemPool=object)
    modules["torch.utils.cpp_extension"].load = load
    modules["torch.utils.cpp_extension"].load_inline = mock.Mock(
        side_effect=AssertionError("load_inline must not rewrite allocator source")
    )
    modules["megatron.core.nccl_allocator"].get_func_args = lambda function: []
    modules["megatron.core.utils"].log_single_rank = lambda *args: None
    spec = importlib.util.spec_from_file_location("gtp_allocator_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class TestGTPAllocatorCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_configured_cache_and_prebuild_do_not_create_allocator(self):
        extension = mock.Mock()
        loader = mock.Mock(return_value=extension)
        module = _load_module(loader)
        with mock.patch.dict(os.environ, TORCH_EXTENSIONS_DIR=str(self.root)):
            self.assertIs(module._load_vmm_allocator_extension(), extension)
            extension.get_vmm_allocator.assert_not_called()
            self.assertIsNone(module._allocator)
            module._build_vmm_allocator()
            module._build_vmm_allocator()
        self.assertEqual(loader.call_count, 2)
        extension.get_vmm_allocator.assert_called_once_with()
        kwargs = loader.call_args.kwargs
        self.assertEqual(kwargs["build_directory"], str(self.root / "gtp_vmm_allocator"))
        self.assertEqual(kwargs["name"], "gtp_vmm_allocator")
        self.assertEqual(kwargs["extra_ldflags"], ["-lcuda"])
        self.assertTrue(kwargs["with_cuda"])
        self.assertTrue(kwargs["is_python_module"])
        source_path = SOURCE.parent / "csrc/gtp_vmm_allocator.cpp"
        self.assertEqual(kwargs["sources"], [str(source_path)])
        self.assertTrue(source_path.is_file())
        self.assertFalse((self.root / "gtp_vmm_allocator/main.cpp").exists())

    def test_legacy_path_without_configured_cache(self):
        loader = mock.Mock()
        module = _load_module(loader)
        module.__file__ = str(self.root / "core/tensor_parallel/gtp_vmm_allocator.py")
        with mock.patch.dict(os.environ):
            os.environ.pop("TORCH_EXTENSIONS_DIR", None)
            module._load_vmm_allocator_extension()
        self.assertEqual(
            loader.call_args.kwargs["build_directory"],
            str(self.root / "core/build/gtp_vmm_allocator"),
        )

    def test_source_changes_still_reach_torch_even_with_cached_library(self):
        calls = []

        def load(**kwargs):
            calls.append((kwargs, Path(kwargs["sources"][0]).read_text()))

        module = _load_module(load)
        module.__file__ = str(self.root / "core/tensor_parallel/gtp_vmm_allocator.py")
        source_path = Path(module.__file__).parent / "csrc/gtp_vmm_allocator.cpp"
        source_path.parent.mkdir(parents=True)
        source_path.write_text("// original allocator source\n")
        directory = self.root / "gtp_vmm_allocator"
        directory.mkdir()
        (directory / "gtp_vmm_allocator.so").write_bytes(b"stale artifact")
        with mock.patch.dict(os.environ, TORCH_EXTENSIONS_DIR=str(self.root)):
            module._load_vmm_allocator_extension()
            source_path.write_text("// updated allocator source\n")
            module._load_vmm_allocator_extension()
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0][1], calls[1][1])
        self.assertEqual(calls[0][0]["build_directory"], calls[1][0]["build_directory"])

    def test_build_exception_preserves_cause(self):
        failure = OSError("compiler failed")
        module = _load_module(mock.Mock(side_effect=failure))
        with mock.patch.dict(os.environ, TORCH_EXTENSIONS_DIR=str(self.root)):
            with self.assertRaisesRegex(RuntimeError, "GTP VMM allocator extension") as caught:
                module._load_vmm_allocator_extension()
        self.assertIs(caught.exception.__cause__, failure)


if __name__ == "__main__":
    unittest.main()
