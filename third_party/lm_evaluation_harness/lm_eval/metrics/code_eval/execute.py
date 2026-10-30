# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is adapted from OpenAI's release
# https://github.com/openai/human-eval/blob/master/human_eval/execution.py

import contextlib
import io
import os
import signal
import tempfile

from lm_eval.metrics.multiple_metrics.containerized_eval import eval_string_script

RELIABILITY_GUARD_STR = """
# This disables various destructive functions and prevents the generated code
# from interfering with the test (e.g. fork bomb, killing other processes,
# removing filesystem files, etc.)

# WARNING
# This function is NOT a security sandbox. Untrusted code, including, model-
# generated code, should not be blindly executed outside of one. See the
# Codex paper for more information about OpenAI's code sandbox, and proceed
# with caution.

import faulthandler
import resource
import platform

maximum_memory_bytes = None  # 1024 * 1024 * 1024  # 1 GB

if maximum_memory_bytes is not None:
    import resource

    # resource.setrlimit(
    #     resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
    # )
    resource.setrlimit(
        resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
    )
    if not platform.uname().system == "Darwin":
        resource.setrlimit(
            resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
        )

faulthandler.disable()

import builtins

builtins.exit = None
builtins.quit = None

import os

from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
from string import ascii_lowercase
from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge
from functools import reduce, cache, lru_cache
from random import randrange, shuffle
from operator import itemgetter, sub
from re import search as re_search  # Assuming 're' refers to a regex search
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator
import copy
import string
import math
import collections
import bisect
import heapq
import functools
import random
import itertools
import operator
import re
import numpy as np
import pandas as pd
from math import log, prod  # 'log' and 'prod' are functions in the math module
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle
from functools import lru_cache, reduce, partial
# from sortedcontainers import SortedList, SortedDict, SortedSet
# import sortedcontainers
from operator import iand
import sys

os.environ["OMP_NUM_THREADS"] = "1"

os.kill = None
os.system = None
os.putenv = None
os.remove = None
os.removedirs = None
os.rmdir = None
os.fchdir = None
os.setuid = None
os.fork = None
os.forkpty = None
os.killpg = None
os.rename = None
os.renames = None
os.truncate = None
os.replace = None
os.unlink = None
os.fchmod = None
os.fchown = None
os.chmod = None
os.chown = None
os.chroot = None
os.fchdir = None
os.lchflags = None
os.lchmod = None
os.lchown = None
os.getcwd = None
os.chdir = None

import shutil

shutil.rmtree = None
shutil.move = None
shutil.chown = None

import subprocess

subprocess.Popen = None  # type: ignore

__builtins__.help = None

import sys

sys.modules["ipdb"] = None
sys.modules["joblib"] = None
sys.modules["resource"] = None
sys.modules["psutil"] = None
sys.modules["tkinter"] = None

"""


def check_correctness(check_program, timeout, task_id, completion_id):
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    safety_prefixed_code = RELIABILITY_GUARD_STR + "\n" + check_program
    result = eval_string_script("python", safety_prefixed_code)
    result["passed"] = result["status"] == "OK"
    result["task_id"] = task_id
    result["completion_id"] = completion_id
    return result


def unsafe_execute(check_program, result, timeout):
    # This runs in a separate process, so we want to minimize imports
    # Only import what we need, when we need it
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        # Run program.
        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)
