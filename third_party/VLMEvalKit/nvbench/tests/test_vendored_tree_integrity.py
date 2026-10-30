from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nvbench import verify_vendored_tree
from nvbench.verify_vendored_tree import (
    compute_tree_digest,
    read_recorded_digest,
)


NVBENCH_ROOT = Path(__file__).resolve().parents[1]


class VendoredTreeIntegrityTest(unittest.TestCase):
    def test_cli_reexports_the_one_shared_digest_implementation(self) -> None:
        self.assertIs(verify_vendored_tree.compute_tree_digest, compute_tree_digest)
        self.assertIs(
            verify_vendored_tree.read_recorded_digest,
            read_recorded_digest,
        )

    def test_checked_in_tree_matches_recorded_digest(self) -> None:
        root = NVBENCH_ROOT.parent
        recorded = verify_vendored_tree.read_recorded_digest(root / "nvbench" / "VENDOR.md")
        self.assertEqual(verify_vendored_tree.compute_tree_digest(root), recorded)

    def test_overlay_content_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "upstream.txt").write_text("upstream", encoding="utf-8")
            initial = verify_vendored_tree.compute_tree_digest(root)
            (root / "nvbench").mkdir()
            (root / "nvbench" / "local.py").write_text("local", encoding="utf-8")
            self.assertEqual(verify_vendored_tree.compute_tree_digest(root), initial)

    def test_content_and_executable_mode_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "script.py"
            target.write_text("first", encoding="utf-8")
            initial = verify_vendored_tree.compute_tree_digest(root)
            target.write_text("second", encoding="utf-8")
            content_changed = verify_vendored_tree.compute_tree_digest(root)
            self.assertNotEqual(content_changed, initial)
            target.chmod(0o755)
            self.assertNotEqual(verify_vendored_tree.compute_tree_digest(root), content_changed)


if __name__ == "__main__":
    unittest.main()
