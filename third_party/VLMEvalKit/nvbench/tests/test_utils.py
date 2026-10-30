"""Focused tests for shared JSON, artifact, path, and environment mechanics."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path

from orchestration.utils import (
    JsonArtifactError,
    StrictJsonError,
    dump_json,
    is_safe_relative_path,
    load_json_bytes,
    read_exact_json,
    relative_path,
    require_shared_file,
    sanitized_error_text,
    temporary_environment,
    write_json,
    write_json_once,
)


class UtilsTest(unittest.TestCase):
    def test_strict_json_round_trip_is_deterministic(self) -> None:
        stream = io.StringIO()
        dump_json({"z": "café", "a": 1}, stream, ensure_ascii=False)
        self.assertEqual(stream.getvalue(), '{\n  "a": 1,\n  "z": "café"\n}\n')
        self.assertEqual(
            load_json_bytes(stream.getvalue().encode(), maximum_bytes=128),
            {"a": 1, "z": "café"},
        )

    def test_strict_json_rejects_invalid_syntax_and_size_contracts(self) -> None:
        for raw, maximum in (
            (b'{"nested":{"x":1,"x":2}}', 128),
            (b'{"x":NaN}', 128),
            (b'{"x":Infinity}', 128),
            (b"", 1),
            (b"{}", 1),
            (b"\xff", 1),
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(StrictJsonError):
                    load_json_bytes(raw, maximum_bytes=maximum)
        with self.assertRaises(StrictJsonError):
            load_json_bytes(b"{}", maximum_bytes=True)
        with self.assertRaises(ValueError):
            dump_json({"value": float("nan")}, io.StringIO(), ensure_ascii=False)

    def test_durable_json_writes_have_expected_replace_and_once_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "record.json"
            self.assertEqual(write_json(target, {"value": 1}), target)
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            self.assertEqual(write_json(target, {"value": 2}), target)
            self.assertEqual(load_json_bytes(target.read_bytes(), maximum_bytes=128), {"value": 2})

            once = root / "once.json"
            self.assertEqual(write_json_once(once, {"value": 3}), once)
            with self.assertRaises(JsonArtifactError):
                write_json_once(once, {"value": 4})
            self.assertFalse(list(root.glob(".*.tmp")))

    def test_exact_json_read_and_contained_file_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = write_json(root / "record.json", {"value": 1})
            self.assertEqual(read_exact_json(target, {"value"}), {"value": 1})
            self.assertEqual(require_shared_file(target, root, "record"), target)
            self.assertEqual(relative_path(target, root), "record.json")
            target.chmod(0o600)
            with self.assertRaises(ValueError):
                require_shared_file(target, root, "record")

    def test_relative_path_safety_and_error_text_are_bounded(self) -> None:
        self.assertTrue(is_safe_relative_path("datasets/a.json"))
        self.assertFalse(is_safe_relative_path("../a.json"))
        self.assertFalse(is_safe_relative_path("a\\b.json"))
        root = Path("/safe/run")
        message = sanitized_error_text(RuntimeError("bad\n/safe/run/private"), root)
        self.assertEqual(message, "bad <run_directory>/private")
        self.assertEqual(
            sanitized_error_text(RuntimeError("abcdef"), root, maximum_length=3),
            "abc",
        )

    def test_temporary_environment_restores_absent_and_present_values(self) -> None:
        name = "NVBENCH_UTILS_TEST_VALUE"
        os.environ.pop(name, None)
        with temporary_environment(name, "temporary"):
            self.assertEqual(os.environ[name], "temporary")
        self.assertNotIn(name, os.environ)
        os.environ[name] = "original"
        try:
            with temporary_environment(name, None):
                self.assertNotIn(name, os.environ)
            self.assertEqual(os.environ[name], "original")
        finally:
            os.environ.pop(name, None)


if __name__ == "__main__":
    unittest.main()
