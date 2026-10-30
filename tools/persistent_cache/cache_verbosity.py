"""Rank-gated stderr logging for the persistent-cache shim.

Every cache helper is armed inside each Python worker, so a status line printed
unconditionally repeats once per rank — thousands of identical lines at scale,
which buries the few messages that matter (which scopes loaded, any failure).

Route status/diagnostic lines through here:
  * ``status`` prints from the global rank-0 worker only (one line per job),
    or from every rank when ``MCORE_CACHE_VERBOSE=1``.
  * ``detail`` prints only under ``MCORE_CACHE_VERBOSE=1`` — for per-rank audit
    lines the first-iter capture harness counts (it sets that env itself).

Failures stay ungated: callers print those directly so any rank can surface a
problem regardless of this module.

Stdlib only; safe to import before torch.
"""

import os
import sys


def verbose():
    """True when full per-rank cache logging is requested."""
    return os.environ.get("MCORE_CACHE_VERBOSE", "0") == "1"


def _is_rank_zero():
    return os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")) == "0"


def status(msg):
    """Print an informational cache line once (rank 0), or per-rank when verbose."""
    if verbose() or _is_rank_zero():
        print(msg, file=sys.stderr, flush=True)


def detail(msg):
    """Print a per-rank diagnostic line only under MCORE_CACHE_VERBOSE=1."""
    if verbose():
        print(msg, file=sys.stderr, flush=True)
