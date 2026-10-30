"""Tiny subprocess behaviors used by local process-supervisor tests."""

from __future__ import annotations

import json
import os
import signal
import sys
import time


def _ready() -> None:
    print("READY", flush=True)


def _exit_on_term(_signum: int, _frame: object) -> None:
    print("FORWARDED_TERM", flush=True)
    raise SystemExit(23)


mode = sys.argv[1]
if mode == "environment":
    names = (
        "KEEP",
        "PYTHONHOME",
        "PYTHONPATH",
        "CONDA_PREFIX",
        "_CONDA_EXE",
        "VIRTUAL_ENV",
        "PIXI_PROJECT_MANIFEST",
        "LD_LIBRARY_PATH",
        "OVERRIDE",
    )
    environment = {name: os.environ.get(name) for name in names}
    environment["__cwd__"] = os.getcwd()
    print(json.dumps(environment), flush=True)
    print("STDERR_CAPTURED", file=sys.stderr, flush=True)
elif mode == "exit_on_term":
    signal.signal(signal.SIGTERM, _exit_on_term)
    _ready()
    while True:
        time.sleep(0.01)
elif mode == "ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _ready()
    while True:
        time.sleep(0.01)
else:
    raise SystemExit(f"unknown mode: {mode}")
