"""Verify the pinned VLMEvalKit tree outside the NVIDIA-owned nvbench overlay."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
from pathlib import Path


_DIGEST_PATTERN = re.compile(
    r"^Upstream tree SHA-256: `([0-9a-f]{64})`$", re.MULTILINE
)
_EXCLUDED_ROOT_NAMES = frozenset({".git", "nvbench"})
_FORMAT_PREFIX = b"vlmevalkit-vendored-tree-v1\0"


def _tree_entries(root: Path) -> list[Path]:
    entries: list[Path] = []
    pending = [Path(".")]
    while pending:
        relative_directory = pending.pop()
        directory = root / relative_directory
        for child in directory.iterdir():
            relative = child.relative_to(root)
            if relative.parts[0] in _EXCLUDED_ROOT_NAMES:
                continue
            if child.is_symlink() or child.is_file():
                entries.append(relative)
            elif child.is_dir():
                pending.append(relative)
            else:
                raise ValueError(f"unsupported filesystem entry: {child}")
    return sorted(entries, key=lambda path: path.as_posix())


def _mode_and_content(path: Path) -> tuple[bytes, bytes]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        return b"120000", os.fsencode(os.readlink(path))
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"unsupported filesystem entry: {path}")
    executable = metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    mode = b"100755" if executable else b"100644"
    return mode, path.read_bytes()


def compute_tree_digest(root: Path) -> str:
    """Hash every non-overlay file's relative path, Git mode, and content.

    ``root`` must be an existing vendored or source tree. The function reads
    every included file without mutation and raises ``ValueError`` or
    ``OSError`` for unsupported or inaccessible entries. Run this developer
    audit on a CPU allocation, not an interactive host.
    """
    root = root.resolve(strict=True)
    digest = hashlib.sha256(_FORMAT_PREFIX)
    for relative in _tree_entries(root):
        path_bytes = os.fsencode(relative.as_posix())
        mode, content = _mode_and_content(root / relative)
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(mode)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def read_recorded_digest(vendor_record: Path) -> str:
    """Return the recorded vendor digest without modifying ``vendor_record``.

    Missing files and malformed records raise their native filesystem or
    ``ValueError`` failures.
    """
    match = _DIGEST_PATTERN.search(vendor_record.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"missing upstream tree digest in {vendor_record}")
    return match.group(1)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--print-digest", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Verify the recorded vendor digest and optional independent source tree."""
    args = _parse_args(argv)
    actual = compute_tree_digest(args.root)
    if args.print_digest:
        print(actual)
        return 0

    recorded = read_recorded_digest(args.root / "nvbench" / "VENDOR.md")
    if actual != recorded:
        print(f"vendored digest mismatch: recorded={recorded} actual={actual}", file=sys.stderr)
        return 1

    if args.source is not None:
        source = compute_tree_digest(args.source)
        if source != actual:
            print(f"source digest mismatch: vendored={actual} source={source}", file=sys.stderr)
            return 1

    print(f"verified VLMEvalKit upstream tree: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
