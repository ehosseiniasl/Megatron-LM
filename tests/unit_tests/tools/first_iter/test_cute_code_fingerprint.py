# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CuTe callable keys must ignore profiler references without ignoring code."""

import functools
import importlib.util
import json
import marshal
import os
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

_root_override = os.environ.get("MCORE_REPO_ROOT")
_REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
_MODULE = _REPO_ROOT / "tools" / "persistent_cache" / "cute_aot_cache.py"


@pytest.fixture(scope="module")
def aot():
    spec = importlib.util.spec_from_file_location("cute_fingerprint_under_test", _MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function(expression="value + 123456789"):
    namespace = {"__name__": "kernel_fixture"}
    exec(compile(f"def kernel(value):\n    return {expression}\n", "kernel.py", "exec"), namespace)
    return namespace["kernel"]


def _key(aot, function):
    return aot._call_key("test-namespace", 0, function, (17,), {"options": "--enable-tvm-ffi"})


def test_retained_references_and_wrappers_do_not_change_key(aot):
    function = _function("(lambda inner: inner + 123456789)(value)")
    before = _key(aot, function)
    # The former default marshal protocol changed its sharing flags when a
    # profiler held these objects, although the function itself never changed.
    retained = [function.__code__.co_consts, function.__code__.co_names]
    nested = next(item for item in retained[0] if isinstance(item, types.CodeType))
    retained.extend([nested.co_consts, nested.co_names, function.__code__.co_linetable])

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    assert _key(aot, function) == before
    assert _key(aot, wrapper) == before
    del retained
    assert _key(aot, function) == before


def test_string_interning_does_not_change_key(aot):
    function = _function("'a constant with spaces'")
    before = _key(aot, function)
    retained = sys.intern(function.__code__.co_consts[1])
    assert retained is function.__code__.co_consts[1]
    assert _key(aot, function) == before


@pytest.mark.parametrize(
    "original,changed",
    [
        ("value + 2", "value * 2"),  # bytecode
        ("value + 2", "value + 3"),  # integer constant
        ("b'first'", "b'second'"),
        ("'first'", "'second'"),
        ("2.25", "2.5"),
        ("2.25j", "2.5j"),
        ("(1, ('first', 2))", "(1, ('second', 2))"),
        ("value in {'first', 'second'}", "value in {'first', 'third'}"),
        ("(lambda inner: inner + 2)(value)", "(lambda inner: inner + 3)(value)"),
        ("abs(value)", "round(value)"),  # referenced global name
    ],
)
def test_code_and_constants_still_invalidate_key(aot, original, changed):
    assert _key(aot, _function(original)) != _key(aot, _function(changed))


@pytest.mark.parametrize(
    "field,value",
    [
        ("co_filename", "other-kernel.py"),
        ("co_firstlineno", 200),
        ("co_exceptiontable", b"changed"),
    ],
)
def test_source_location_and_exception_metadata_remain_in_key(aot, field, value):
    function = _function()
    changed = types.FunctionType(function.__code__.replace(**{field: value}), function.__globals__)
    assert _key(aot, function) != _key(aot, changed)


@pytest.mark.parametrize(
    "first,second",
    [
        (0.0, -0.0),
        # Keep the exact float representation, including distinct NaN payloads.
        (
            struct.unpack("d", bytes.fromhex("010000000000f87f"))[0],
            struct.unpack("d", bytes.fromhex("020000000000f87f"))[0],
        ),
    ],
)
def test_binary_float_constants_remain_distinct(aot, first, second):
    function = _function("2.25")
    keys = []
    for value in (first, second):
        code = function.__code__.replace(co_consts=(None, value))
        keys.append(_key(aot, types.FunctionType(code, function.__globals__)))
    assert keys[0] != keys[1]


def test_closure_changes_still_invalidate_key(aot):
    def make(offset):
        def kernel(value):
            return value + offset

        return kernel

    assert _key(aot, make(2)) != _key(aot, make(3))


def test_versioned_descriptor_does_not_match_legacy_key(aot):
    function = _function()
    descriptor = aot._call_descriptor("test-namespace", 0, function, (17,), {})
    assert descriptor["format"] == "mcore-cute-aot-v2"
    legacy = json.loads(json.dumps(descriptor))
    del legacy["format"]
    legacy["func"]["code_sha256"] = aot.hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
    assert aot._descriptor_key(descriptor) != aot._descriptor_key(legacy)


def test_fresh_process_hash_seed_and_reference_retention_do_not_change_key(aot):
    # frozenset constants must remain stable across Python hash randomization,
    # and nested code must not retain reference-count-dependent serialization.
    program = f"""
import functools
import importlib.util
import os
spec = importlib.util.spec_from_file_location('cache', {_MODULE.as_posix()!r})
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
def kernel(value):
    def nested(inner):
        return inner in {{'first', 'second', 'third', 'fourth', 'fifth'}}
    return nested(value), (1.25, 2.5j, b'constant')
if os.environ['RETAIN_REFERENCES'] == '1':
    retained = [kernel.__code__.co_consts, kernel.__code__.co_names]
    @functools.wraps(kernel)
    def wrapper(*args, **kwargs):
        return kernel(*args, **kwargs)
    target = wrapper
else:
    target = kernel
print(cache._call_key('test-namespace', 0, target, ('first',), {{}}))
"""
    keys = set()
    for seed, retain in (("1", "0"), ("2", "1"), ("42", "0"), ("42", "1")):
        result = subprocess.run(
            [sys.executable, "-c", program],
            env={**os.environ, "PYTHONHASHSEED": seed, "RETAIN_REFERENCES": retain},
            capture_output=True,
            text=True,
            check=True,
        )
        keys.add(result.stdout.strip())
    assert len(keys) == 1
