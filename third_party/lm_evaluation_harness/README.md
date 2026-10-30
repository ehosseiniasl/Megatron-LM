# LM Evaluation Harness (NVIDIA Fork)

Evaluation harness for Megatron-LM and vLLM base models. This fork adds NVIDIA-specific orchestration for running evaluations at scale on SLURM clusters.

## Install

Install the package using the install script, which requires your HuggingFace token:

```bash
./install.sh YOUR_HF_TOKEN
```

Get your token from https://huggingface.co/settings/tokens (required to avoid rate limiting when downloading datasets).

This submits a SLURM job that:
1. Downloads and installs `uv` (fast Python package manager)
2. Creates a virtual environment named `env` in the repository root
3. Installs `lm-eval` in editable mode
4. Configures HuggingFace authentication and bakes the token into the activate script

Once the job completes, activate the environment:

```bash
source env/bin/activate
```

The `HF_TOKEN` is automatically exported when you activate the environment.

## Usage

### Option 1: MIMO text and multilingual evaluation (Recommended)

`nvbench/launch_mimo_vlm_eval.sbatch` launches the inference server and
evaluates all 23 text and multilingual task entries in `nemo4-fast.yml` with
this repository's vendored lm-eval harness:

```bash
sbatch \
    --export=ALL,RUN_NAME=n4v6-a3b,CHECKPOINT_DIR=/path/to/checkpoints,CHECKPOINT_STEP=10000 \
    nvbench/launch_mimo_vlm_eval.sbatch
```

The one-node TP2/EP2 defaults target Nano checkpoints. For Super checkpoints,
keep the same launcher and evaluation settings while increasing expert
parallelism:

```bash
sbatch --nodes=8 \
    --export=ALL,RUN_NAME=n4v6-super,CHECKPOINT_DIR=/path/to/checkpoints,\
CHECKPOINT_STEP=10000,EXPERT_MODEL_PARALLEL_SIZE=16,\
INFERENCE_MOE_TOKEN_DISPATCHER_TYPE=nccl \
    nvbench/launch_mimo_vlm_eval.sbatch
```

The default `nvls` inference dispatcher keeps the one-node Nano path on its
NVLink-optimized implementation. Multi-node Super expert groups use `nccl`.

`LME_ENV`, `IMAGE`, `HF_HOME`, `TOKENIZER_MODEL`, and `RUN_ROOT` can be
overridden for cluster-local installations. Long-context Super tasks can lower
`INFERENCE_DYNAMIC_BATCHING_MAX_TOKENS` from its Nano-tuned default of 65,536
when their activation batches do not fit in HBM.

### Option 2: N4 VLM evaluation

`nvbench/launch_n4_vlm_eval.sbatch` runs the two-shot suite from the pinned
`n4-vlm-evals` VLMEvalKit revision. It selects each of the 13 tasks directly
from that suite and uses the validated TP1/EP4 dynamic-server path,
16,384-token context, inference-optimized language decoder, Mistral-crawl image
preprocessing, and tokenizer-provided chat template.

Build a fresh revision-stamped VLMEvalKit environment from the revision in
`nvbench/n4_vlm_eval_pins.sh`. On both AGA and CMH, prefetch the pinned tokenizer
revision into the cluster-local Hugging Face cache before submitting the array.
The launcher refuses an environment or tokenizer snapshot from any other
revision.

```bash
VLM_ENV=/path/to/vlmevalkit-n4-vlm-evals-5b14657678ef

sbatch --export=ALL,RUN_NAME=n4v6-a3b,CHECKPOINT_DIR=/path/to/checkpoints,\
CHECKPOINT_STEP=10000,OUTPUT_ROOT=/path/to/vlm-results,\
MEGATRON_COMMIT=$(git rev-parse HEAD),VLM_ENV=${VLM_ENV} \
    nvbench/launch_n4_vlm_eval.sbatch
```

Results use a unique evaluation ID and are never reused. `LMUData` defaults to
`${OUTPUT_ROOT}/cache/vlmevalkit`; point it at an existing cluster-local dataset
cache when available.

### Option 3: Manual (against running server)

If you already have a Megatron or vLLM server running, you can run evaluations directly:

**Megatron:**
```bash
python nvbench/launch_evals.py -e megatron \
    -c nvbench/nemo4-fast.yml \
    -r /path/to/results \
    -m my_model_name \
    -i $SERVER_IP -p $SERVER_PORT
```

**vLLM:**
```bash
python nvbench/launch_evals.py -e vllm \
    -c nvbench/nemo4-fast.yml \
    -r /path/to/results \
    -m my_model_name \
    -i 127.0.0.1 -p $PORT
```

The `-e` flag specifies the endpoint type, which determines the API format used to communicate with the server.

## Benchmark Configs

Benchmark configurations are YAML files in `nvbench/` that define which tasks to run and how to report results:

| Config | Description |
|--------|-------------|
| `nvbench/nemo4-fast.yml` | Fast N4 evaluation subset with optimized batch sizes and separately schedulable compression tasks |

Each config specifies tasks with their batch sizes, few-shot settings, and which metrics to report. See [CLAUDE.md](CLAUDE.md) for the config format details.

### Parametric knowledge tasks

`simpleqa_verified_25shot` and `facts_parametric_25shot` follow the
DeepSeek-V4 base-model setup: 25 fixed demonstrations, greedy short-answer
generation, and a semantic exact-match score.

Set `OPENROUTER_API_KEY` before running either task. Once normal model
generation finishes, lm-eval's metric aggregation step grades the responses
with `openai/gpt-oss-120b`. The grader sends concurrent single-example
requests, defaults to 128 in flight, and uses three independent grades per
FACTS response. The score is written directly into the normal lm-eval result
as `judge_accuracy,none`.

The defaults can be overridden with:

```bash
export PARAMETRIC_QA_JUDGE_MODEL=openai/gpt-oss-120b
export PARAMETRIC_QA_JUDGE_CONCURRENCY=128
export PARAMETRIC_QA_JUDGE_BASE_URL=https://openrouter.ai/api/v1
```

NVIDIA Inference is also supported through its OpenAI-compatible API. It uses
the documented `https://integrate.api.nvidia.com/v1` endpoint and accepts
`NVIDIA_API_KEY` (or the legacy `NVIDIA_INFERENCE_API_KEY` fallback):

```bash
export NVIDIA_API_KEY=...
export PARAMETRIC_QA_JUDGE_PROVIDER=nvidia
```

`scripts/grade_parametric_qa.py` remains available for manually regrading
logged samples or recovering a judge-only failure.

## Task Selection

Run specific tasks by index using the `-t` flag. This is useful for parallel evaluation with SLURM job arrays or for debugging specific tasks:

```bash
# Single task (e.g., task at index 5)
python nvbench/launch_evals.py ... -t 5

# Range of tasks (indices 0-9)
python nvbench/launch_evals.py ... -t 0:10

# From index 15 to end
python nvbench/launch_evals.py ... -t 15:
```

Task indices correspond to the order in the benchmark config file.

## Assembling Results

After running evaluations, use `assemble_report_results.py` to aggregate results across models into a comparison table:

```bash
python nvbench/assemble_report_results.py \
    -c nvbench/nemo4-fast.yml \
    -r /path/to/results
```

This script:
- Recursively finds all `results*.json` files in the results directory
- Groups results by model (based on parent directory names)
- Outputs a formatted table with tasks grouped by category (Code, Math, Reasoning, etc.)
- Also outputs CSV format for further analysis

The results directory should have the structure:
```
results/
├── model_a/
│   └── results_task1.json
│   └── results_task2.json
├── model_b/
│   └── results_task1.json
│   └── ...
```

## Development

See [CLAUDE.md](CLAUDE.md) for detailed documentation on:
- Adding new evaluation tasks
- Task YAML format and configuration options
- Benchmark config structure
- How `dp_env.sh` works for distributed setups

Debug task prompts to see what the model will receive:
```bash
python scripts/get_prompts.py task_name
```
