# Parametric knowledge QA

These tasks probe factual knowledge stored in model parameters. They do not
provide retrieval or search tools.

## Tasks

- `simpleqa_verified_25shot` uses the official
  `google/simpleqa-verified` dataset (1,000 examples, MIT license).
- `facts_parametric_25shot` uses version 2 of Kaggle's official
  `kaggle/facts-parametric-public-examples` dataset (1,052 examples,
  Apache-2.0 license). The task reads the public CSV directly from Kaggle's
  version-pinned download endpoint.

Both tasks use 25 fixed, hand-written demonstrations and greedy short-answer
generation for pretrained/base models. This follows the DeepSeek-V4 base-model
report, which evaluates SimpleQA Verified and FACTS Parametric as 25-shot world
knowledge tasks. The report does not publish its demonstrations, so the
examples here are harness-defined and deliberately do not come from either
evaluation set.

The task's normal metric aggregation step grades all responses after model
generation finishes. It writes `judge_accuracy,none` directly into the normal
lm-eval result. `reference_match` and `response_nonempty` are also reported as
cheap diagnostics, but they are not benchmark scores.

The grader ports the current official Kaggle starter-notebook rubrics and
defaults to `openai/gpt-oss-120b` on OpenRouter. Calls are concurrent
single-example requests, with 128 in flight by default. SimpleQA uses one grade
per response and FACTS uses three. Empty responses and normalized exact matches
take a conservative no-API fast path. OpenAI-compatible client, retry,
concurrency, and usage-accounting behavior is shared through
`lm_eval.api.judge`.

Set `OPENROUTER_API_KEY` in the regular evaluation process. Optional overrides
are:

```bash
export PARAMETRIC_QA_JUDGE_MODEL=openai/gpt-oss-120b
export PARAMETRIC_QA_JUDGE_CONCURRENCY=128
export PARAMETRIC_QA_JUDGE_BASE_URL=https://openrouter.ai/api/v1
```

To run the same `openai/gpt-oss-120b` judge through NVIDIA Inference instead,
use NVIDIA's OpenAI-compatible endpoint profile:

```bash
export NVIDIA_API_KEY=...
export PARAMETRIC_QA_JUDGE_PROVIDER=nvidia
```

This selects `https://integrate.api.nvidia.com/v1` automatically. The legacy
`NVIDIA_INFERENCE_API_KEY` variable is also accepted. NVIDIA's profile omits
the optional structured-output request field and relies on the existing
label-parser and response retries, for compatibility with the hosted API.

## Manual regrading and recovery

The standalone grader can regrade emitted `samples_*.jsonl` files:

```bash
OPENROUTER_API_KEY=... python scripts/grade_parametric_qa.py \
  /results/model_name
```

Or use NVIDIA Inference:

```bash
NVIDIA_API_KEY=... python scripts/grade_parametric_qa.py \
  --provider nvidia /results/model_name
```

This recovery path caches completed calls in adjacent
`.openrouter-cache.jsonl` (or `.nvidia-cache.jsonl`) files and writes a separate lm-eval-compatible
`results_judged_<task>.json`. `--dry-run` reports the pending call count. If a
model directory contains multiple sample logs for the same task, select the
intended file explicitly; the grader refuses to silently overwrite one run's
score with another.

## Sources

- [DeepSeek-V4 technical report](https://arxiv.org/html/2606.19348v1)
- [SimpleQA Verified paper](https://arxiv.org/abs/2509.07968) and
  [dataset](https://huggingface.co/datasets/google/simpleqa-verified)
- [FACTS Parametric benchmark](https://www.kaggle.com/benchmarks/google/facts-parametric)
  and [paper](https://arxiv.org/abs/2507.11473)

## Known limitations

- The FACTS task contains only the 1,052-example public split. It cannot
  reproduce Kaggle's score over the equally sized private split.
- DeepSeek-V4 reports internal-framework exact match for its base models. Its
  unpublished prompt and matcher prevent exact reproduction.
- `gpt-oss-120b` is a cost-conscious operational judge, not the benchmarks'
  original GPT-4.1 and Gemini 2.5 Pro judges. Changing the judge or provider
  can move scores.
- FACTS' paper averages the three judge grades, while its current Kaggle
  starter notebook also exposes a stricter score requiring all three grades
  to be correct. The regular task reports the average; the standalone grader
  also reports the strict score.
