#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Build known MIMO extensions at their final image path and check fresh-process reuse.

Run inside the target image, with its matching training checkout. This only
builds/loads extensions; it does not initialize the GTP allocator or a process group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_BUILD = """
import json
import torch
from megatron.core.tensor_parallel import gtp_vmm_allocator, random
random._get_share_storage()
gtp = gtp_vmm_allocator._load_vmm_allocator_extension()
print('MCORE_EXTENSION_RESULT=' + json.dumps({
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'libraries': [random._share_storage_ext.__file__, gtp.__file__],
}))
"""


def _snapshot(root: Path) -> dict:
    """Track build outputs, including timestamps, so a rebuild cannot pass unnoticed."""
    result = {}
    for name in ("share_storage_ext", "gtp_vmm_allocator"):
        directory = root / name
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name in {"lock", ".load_inline.lock"}:
                continue
            stat = path.stat()
            result[str(path.relative_to(root))] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return result


def finalize(dest: Path, training_repo: Path) -> dict:
    """Build once, then require an independent process to reuse both extensions."""
    training_repo = training_repo.resolve()
    sources = [
        training_repo / "megatron/core/tensor_parallel/random.py",
        training_repo / "megatron/core/tensor_parallel/gtp_vmm_allocator.py",
        training_repo / "megatron/core/tensor_parallel/csrc/gtp_vmm_allocator.cpp",
    ]
    for source in sources:
        if not source.is_file():
            raise ValueError(f"matching training source is missing: {source}")
    root = dest.resolve() / "megatron_fused_kernels"
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Do not inherit a seed shim that would redirect the final build path.
    env["PYTHONPATH"] = str(training_repo)
    env["TORCH_EXTENSIONS_DIR"] = str(root)
    results = []
    before_check = None
    for stage in ("build", "verify"):
        print(f"[IMAGE CACHE] extensions: {stage} at {root}", flush=True)
        completed = subprocess.run(
            [sys.executable, "-c", _BUILD],
            cwd=training_repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        print(completed.stdout, end="", flush=True)
        print(completed.stderr, end="", file=sys.stderr, flush=True)
        completed.check_returncode()
        records = [
            line.split("=", 1)[1]
            for line in completed.stdout.splitlines()
            if line.startswith("MCORE_EXTENSION_RESULT=")
        ]
        if len(records) != 1:
            raise RuntimeError(f"{stage}: missing extension load result")
        result = json.loads(records[0])
        for library, name in zip(result["libraries"], ("share_storage_ext", "gtp_vmm_allocator")):
            if not Path(library).is_file() or Path(library).resolve().parent != root / name:
                raise RuntimeError(f"{stage}: extension loaded outside final image path: {library}")
        if len(result["libraries"]) != 2:
            raise RuntimeError(f"{stage}: expected both extension libraries")
        results.append(result)
        after = _snapshot(root)
        if stage == "verify" and (before_check != after or results[0] != results[1]):
            raise RuntimeError(
                "fresh-process extension load changed build artifacts; refusing to finish bake"
            )
        before_check = after
    report = {
        "training_repo": str(training_repo),
        "source_sha256": {
            str(p.relative_to(training_repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "runtime": results[0],
        "fresh_process_reused": True,
        "artifacts": before_check,
    }
    (dest / "extensions-finalization.json").write_text(json.dumps(report, indent=2) + "\n")
    print("[IMAGE CACHE] both extensions reused unchanged in a fresh process", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--training-repo", required=True)
    args = parser.parse_args()
    finalize(Path(args.dest), Path(args.training_repo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
