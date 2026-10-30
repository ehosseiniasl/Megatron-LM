# BigCodeBench Task for LM Evaluation Harness

This task implements BigCodeBench in `lm-eval` and supports both benchmark modes:
- `Complete` (function completion from docstring + signature)
- `Instruct` (code generation from natural-language instructions)

## Overview

BigCodeBench is a practical code-generation benchmark with 1,140 Python tasks and unit-test-based evaluation.

- Homepage: https://bigcode-bench.github.io/
- Dataset: https://huggingface.co/datasets/bigcode/bigcodebench
- Paper: https://arxiv.org/abs/2406.15877

## Dataset Notes

BigCodeBench on Hugging Face uses:
- `dataset_path: bigcode/bigcodebench`
- default config (no `dataset_name`)
- versioned splits (not `test`)

This task currently evaluates on:
- `test_split: v0.1.4`

## Setup

### Step 1: Python Environment

```bash
# from repo root
uv venv env
source env/bin/activate
uv pip install -e ".[code_eval]"
```

### Step 2: Enable Code Evaluation

BigCodeBench executes model-generated code. Enable it explicitly:

```bash
export HF_ALLOW_CODE_EVAL=1
```

## Available Tasks

| Task | Mode | Decoding |
|------|------|----------|
| `bigcodebench_complete` | Complete | sampled |
| `bigcodebench_complete_greedy` | Complete | greedy |
| `bigcodebench_instruct` | Instruct | sampled |
| `bigcodebench_instruct_greedy` | Instruct | greedy |

## Running Evaluations

### Basic Usage

```bash
# Complete (greedy, pass@1)
lm_eval --model hf --model_args pretrained=<model_name> \
  --tasks bigcodebench_complete_greedy --batch_size 1

# Instruct (greedy, pass@1)
lm_eval --model hf --model_args pretrained=<model_name> \
  --tasks bigcodebench_instruct_greedy --batch_size 1
```

```bash
# Complete (sampled, pass@k)
lm_eval --model hf --model_args pretrained=<model_name> \
  --tasks bigcodebench_complete --batch_size 1

# Instruct (sampled, pass@k)
lm_eval --model hf --model_args pretrained=<model_name> \
  --tasks bigcodebench_instruct --batch_size 1
```

### Run Both Modes

```bash
lm_eval --model hf --model_args pretrained=<model_name> \
  --tasks bigcodebench_complete_greedy,bigcodebench_instruct_greedy \
  --batch_size 1
```

## Testing

### Quick Local Sanity Test (No Inference Server)

This checks task registration and code-eval pipeline behavior using the in-process dummy model:

```bash
cd /lustre/fsw/portfolios/llmservice/users/rislam/adlr/lm-evaluation-harness
source env/bin/activate
export HF_ALLOW_CODE_EVAL=1
python -m lm_eval --model dummy --tasks bigcodebench_complete_greedy --limit 1 --batch_size 1
python -m lm_eval --model dummy --tasks bigcodebench_instruct_greedy --limit 1 --batch_size 1
```

### NVBench Run (Real Server Path)

Run `nvbench/launch_evals.py` against an existing server as described above.
The repository's SLURM launcher is specific to MIMO VLM evaluation.

## Metrics

The task reports code-eval metrics from `estimate_pass_at_k_dynamic`:
- `pass@1[N]`
- `pass@k[N]` for configured repeats/sample count

For greedy configs (`repeats=1`), `pass@1[1]` is the main metric.

## Implementation Notes

- Prompt builders are in `utils.py`:
  - `build_text_complete()`
  - `build_text_instruct()`
- Candidate scoring uses:
  - `score_predictions()`
  - `score_predictions_instruct()`
- Generated code is sanitized before execution and checked against each task's `test`.

## Troubleshooting

### `ValueError: Code evaluation is disabled by default`

Set:

```bash
export HF_ALLOW_CODE_EVAL=1
```

### Split/dataset errors

Use `test_split: v0.1.4` and no `dataset_name` for this dataset.

### Slow runtime

Runtime is sensitive to:
- `batch_size`
- `repeats`
- `generation_kwargs.max_gen_toks`

### Server/client launch issues (Slurm)

Check:
- `data/slurm_logs/server_<job>_<array>.log`
- `data/slurm_logs/client_<job>_<array>.log`

Common issue: server port collision (`Address already in use`), usually resolved by relaunching the failed array index.

## Citation

```bibtex
@article{zhuo2024bigcodebench,
  title={BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions},
  author={Zhuo, Terry Yue and others},
  journal={arXiv preprint arXiv:2406.15877},
  year={2024}
}
```
