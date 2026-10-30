"""Rank-gating for the persistent-cache shim's stderr logging.

`status` lines print once (global rank 0) by default and per-rank only under
MCORE_CACHE_VERBOSE=1; `detail` lines print only under that flag. Exercised in a
subprocess so the module sees a clean environment each time.
"""

import os
import subprocess
import sys
from pathlib import Path

_root_override = os.environ.get("MCORE_REPO_ROOT")
_REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
_PC_DIR = _REPO_ROOT / "tools" / "persistent_cache"

_DRIVER = "import cache_verbosity as cv\n" "cv.status('STATUS')\n" "cv.detail('DETAIL')\n"


def _run(rank=None, verbose=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PC_DIR)
    for key in ("RANK", "SLURM_PROCID", "MCORE_CACHE_VERBOSE"):
        env.pop(key, None)
    if rank is not None:
        env["RANK"] = rank
    if verbose is not None:
        env["MCORE_CACHE_VERBOSE"] = verbose
    result = subprocess.run(
        [sys.executable, "-c", _DRIVER], check=True, capture_output=True, env=env, text=True
    )
    return result.stderr


def test_status_prints_on_rank_zero_only():
    assert "STATUS" in _run(rank="0")
    assert "STATUS" not in _run(rank="5")


def test_detail_is_silent_without_verbose():
    assert "DETAIL" not in _run(rank="0")


def test_verbose_restores_status_and_detail_on_every_rank():
    stderr = _run(rank="5", verbose="1")
    assert "STATUS" in stderr
    assert "DETAIL" in stderr


def test_defaults_to_primary_when_rank_unset():
    # No RANK/SLURM_PROCID -> treated as rank 0 so a lone process still reports.
    assert "STATUS" in _run()
