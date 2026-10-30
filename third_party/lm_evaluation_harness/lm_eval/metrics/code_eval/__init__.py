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
"""The CodeEval metric estimates the pass@k metric for code synthesis.
This is an evaluation harness for the HumanEval problem solving dataset
described in the paper "Evaluating Large Language Models Trained on Code"
(https://arxiv.org/abs/2107.03374)."""

import itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .execute import check_correctness

import numpy as np
from tqdm import tqdm

_CITATION = """\
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan \
and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards \
and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray \
and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf \
and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray \
and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser \
and Mohammad Bavarian and Clemens Winter and Philippe Tillet \
and Felipe Petroski Such and Dave Cummings and Matthias Plappert \
and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss \
and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak \
and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain \
and William Saunders and Christopher Hesse and Andrew N. Carr \
and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa \
and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati \
and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei \
and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""

_DESCRIPTION = """\
This metric implements the evaluation harness for the HumanEval problem solving dataset
described in the paper "Evaluating Large Language Models Trained on Code"
(https://arxiv.org/abs/2107.03374).
"""


_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of candidates to evaluate. Each candidates should be a list
        of strings with several code candidates to solve the problem.
    references: a list with a test for each prediction. Each test should evaluate the
        correctness of a code candidate.
    k: number of code candidates to consider in the evaluation (Default: [1, 10, 100])
    num_workers: number of workers used to evaluate the canidate programs (Default: 4).
    timeout:
Returns:
    pass_at_k: dict with pass rates for each k
    results: dict with granular results of each unittest
Examples:
    >>> test_cases = ["assert add(2,3)==5"]
    >>> candidates = [["def add(a,b): return a*b", "def add(a, b): return a+b"]]
    >>> # Use individual functions directly for evaluation
    >>> # See tasks/humaneval/utils.py for usage examples
"""


_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval" metric executes untrusted model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).

Once you have read this disclaimer and taken appropriate precautions,
set the environment variable HF_ALLOW_CODE_EVAL="1". Within Python you can to this
with:

>>> import os
>>> os.environ["HF_ALLOW_CODE_EVAL"] = "1"

################################################################################\
"""

_LICENSE = """The MIT License

Copyright (c) OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE."""


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_candidates(
    candidates: list[list[str]],
    docs: list[dict],
    process_function: Callable[[int, str, dict], dict],
) -> list[list[dict]]:
    """Generic function to evaluate code candidates using ThreadPoolExecutor.

    Args:
        candidates: List of candidate lists for each document
        docs: List of documents corresponding to candidates
        process_function: Function that takes (repeat_idx, response, doc) and returns result dict
        num_workers: Number of worker threads

    Returns:
        List of result lists maintaining the same structure as candidates
    """
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = []
        task_metadata = []

        for doc_idx, (doc, candidate_list) in enumerate(zip(docs, candidates)):
            for candidate_idx, candidate in enumerate(candidate_list):
                future = executor.submit(process_function, candidate_idx, candidate, doc)
                futures.append(future)
                task_metadata.append((doc_idx, candidate_idx))

        results_by_doc = defaultdict(list)
        pbar = tqdm(total=len(futures), desc="Evaluations")
        for future, (doc_idx, candidate_idx) in zip(futures, task_metadata):
            result = future.result()
            results_by_doc[doc_idx].append((candidate_idx, result))
            pbar.update(1)

    doc_results = []
    for doc_idx in range(len(docs)):
        doc_result_list = results_by_doc[doc_idx]
        doc_result_list.sort()
        ordered_results = [r[1] for r in doc_result_list]
        doc_results.append(ordered_results)

    return doc_results


def estimate_pass_at_k_dynamic(doc: dict, results: list[list[dict]]) -> dict:
    """Pass@k estimation function with dynamic K sweep (powers of 2 up to N)."""
    result_list = results[0]  # filter_fn results
    passed = [r["passed"] for r in result_list]
    total = len(passed)
    correct = sum(passed)

    N = total
    Ks = set([2**x for x in range(0, int(np.log2(N)) + 1)] + [N])
    pass_at_k_results = {}
    for k in Ks:
        pass_at_k_results[f"pass@{k}[{N}]"] = estimate_pass_at_k([total], [correct], k)[0]
    return pass_at_k_results
