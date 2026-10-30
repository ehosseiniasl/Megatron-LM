# CLAUDE.md - Development Guide

This is the NVIDIA fork of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluating Megatron-LM and vLLM models.

## Quick Commands

```bash
# Install (get token from https://huggingface.co/settings/tokens)
./install.sh YOUR_HF_TOKEN
source env/bin/activate

# Run evaluation against a running server
python nvbench/launch_evals.py -e megatron -c nvbench/nemo4-fast.yml \
    -r /results -m model_name -i 127.0.0.1 -p 5000

# Debug a task's prompts
python scripts/get_prompts.py task_name

# Mock server for testing without a real model
python mock_server.py --port 5000
lm_eval --model notok-completions \
    --model_args base_url=http://127.0.0.1:5000,model=mock \
    --tasks <task_name> --batch_size 4 --limit 10
```

## Repository Structure

```
install.sh                  # Install script (validates HF token, submits SLURM job)
mock_server.py              # Mock OpenAI-compatible server for local testing
install_env.sbatch          # SLURM job for environment setup

nvbench/                    # NVIDIA evaluation orchestration
├── launch_evals.py         # Main evaluation runner
├── dp_env.sh               # Distributed GPU setup + server discovery
├── assemble_results.py     # Results aggregation
├── nemo4-fast.yml          # N4 benchmark config
└── *.sbatch                # SLURM job scripts

lm_eval/tasks/              # Task definitions (~200+ tasks)
├── humaneval/              # Code tasks
├── gsm8k/                  # Math tasks
├── mmlu/                   # Aggregate benchmarks
├── global_mmlu/            # Multilingual (42 languages)
└── ...

scripts/
└── get_prompts.py          # Debug task prompts
```

## Adding a New Task

### 1. Create Task YAML

Create a directory under `lm_eval/tasks/` with a YAML file:

```yaml
# lm_eval/tasks/my_task/my_task.yaml
task: my_task
dataset_path: my_dataset              # HuggingFace dataset or local path
dataset_name: main                    # Dataset config name
test_split: test

doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: "{{answer}}"

output_type: generate_until
generation_kwargs:
  do_sample: false
  temperature: 0.0
  max_gen_toks: 512
  until: ["\n\n", "</s>"]

num_fewshot: 5

metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
```

### 2. For Custom Processing

Add a `utils.py` with functions referenced via `!function`:

```yaml
# In YAML
doc_to_target: !function utils.extract_answer
process_results: !function utils.process_results
```

```python
# lm_eval/tasks/my_task/utils.py
def extract_answer(doc):
    return doc["answer"].strip()

def process_results(doc, results):
    # results[0] is the model output
    pred = results[0].strip()
    gold = doc["answer"].strip()
    return {"exact_match": pred == gold}
```

### 3. Add to Benchmark Config

Add your task to a benchmark YAML in `nvbench/`:

```yaml
# nvbench/nemo4-fast.yml
tasks:
  - task_name: my_task
    task_group: reasoning           # For grouping in reports
    batch_size: 32
    num_fewshot: 5
    report_metric:
      - alias: my_task              # Display name
        metric: exact_match,none    # format: metric_name,filter_name
```

### 4. Test Your Task

```bash
# Check prompts
python scripts/get_prompts.py my_task

# Run single task locally
lm_eval --model hf --model_args pretrained=gpt2 \
    --tasks my_task --limit 10
```

## Task YAML Reference

### Common Fields

| Field | Description |
|-------|-------------|
| `dataset_path` | HuggingFace dataset name or local path |
| `doc_to_text` | Jinja2 template for prompt |
| `doc_to_target` | Expected answer (string or `!function`) |
| `output_type` | `generate_until`, `multiple_choice`, `loglikelihood` |
| `num_fewshot` | Number of few-shot examples |
| `generation_kwargs` | `max_gen_toks`, `temperature`, `until` (stop strings) |
| `metric_list` | List of metrics with `aggregation` method |

### Few-shot Configuration

```yaml
fewshot_config:
  sampler: first_n                  # first_n, random, stratified
  samples:                          # Or explicit samples
    - question: "2+2?"
      target: "4"
```

### Metrics

Common metric formats:
- `exact_match,none` - Exact string match
- `acc,none` - Accuracy
- `acc_norm,none` - Length-normalized accuracy
- `pass@1[N],evalplus_sanitized` - Code pass rate with sanitization

## Benchmark Config Format

```yaml
version: 1.0
tasks:
  - task_name: humaneval_greedy
    task_group: code
    batch_size: 64
    num_fewshot: 0
    cache_requests: true
    overrides:
      repeats: 32                   # For sampling-based metrics
      generation_kwargs.max_gen_toks: 2048
    report_metric:
      - alias: humaneval
        metric: pass@1[1],evalplus_sanitized
```

### Override Syntax

Override any task parameter in the benchmark config:

```yaml
overrides:
  repeats: 32
  generation_kwargs.max_gen_toks: 2048
  num_fewshot: 3
```

## Model Type Architecture

`launch_evals.py` maps engine presets to lm-eval model types:

| Engine (`-e`) | lm-eval model | Class | Notes |
|---|---|---|---|
| `megatron` | `notok-completions` | `TokenizerFreeLocalCompletionsLM` | Hits OpenAI-compatible completions endpoint; supports all output types including `loglikelihood_rolling` |
| `vllm` | `local-completions` | `OpenaiCompletionsLM` | Same API, but with a HF tokenizer for proper token-level chunking |

**`megatron_api.py` (`megatron_eval_api`) is NOT used by default.** It's a separate model class that hits a custom `/likelihood` endpoint and does NOT support `loglikelihood_rolling`. Don't add rolling loglikelihood there.

For perplexity tasks (`loglikelihood_rolling`, e.g. `llm_compression`, `wikitext`): just use `-e megatron` as normal — `notok-completions` handles it via the completions endpoint with `echo=True`/`logprobs`.

## Running Evaluations

### Megatron-LM

```bash
python nvbench/launch_evals.py -e megatron \
    -c nvbench/nemo4-fast.yml \
    -r /results -m model_name \
    -i $SERVER_IP -p $SERVER_PORT
```

### vLLM

```bash
python nvbench/launch_evals.py -e vllm \
    -c nvbench/nemo4-fast.yml \
    -r /results -m model_name \
    -i 127.0.0.1 -p $PORT
```

### Task Selection

Run specific tasks by index:
```bash
# Single task
python nvbench/launch_evals.py ... -t 5

# Range
python nvbench/launch_evals.py ... -t 0:10

# From index to end
python nvbench/launch_evals.py ... -t 15:
```

### Server Discovery (new)

Instead of IP:port, use ready files for automatic discovery:
```bash
python nvbench/launch_evals.py ... --ready-dir /path/to/ready_files
```

## SLURM Integration

See `nvbench/launch_mimo_vlm_eval.sbatch` for the supported MIMO VLM server-plus-evaluation job.

Key pattern: Launch server + evaluations in parallel using `srun --overlap`:

```bash
# Launch inference server
srun -l --ntasks-per-node 8 ... bash -c "source dp_env.sh ... --launch 'python server.py'" &

# Run evaluations (waits for server via ready_dir)
srun --overlap --nodes 1 -l --ntasks 1 bash -c "python launch_evals.py --ready-dir ..."
```

## dp_env.sh

Sets up distributed GPU environment and server discovery:

```bash
source nvbench/dp_env.sh --tp 2 --pp 1 --port 5000 --launch 'python server.py'
```

Arguments:
- `--tp N` - Tensor parallelism
- `--pp N` - Pipeline parallelism
- `--port N` - Base port (0 for auto-discovery)
- `--ready-dir PATH` - Write ready file when server is up
- `--vllm` - vLLM mode (server on rank 0 only)

## Results

Results are saved as JSON in the results directory. Use `assemble_results.py` or `assemble_report_results.py` to aggregate:

```bash
python nvbench/assemble_report_results.py -c nvbench/nemo4-fast.yml -r /results
```

## Common Issues

**Task not found**: Ensure your YAML is valid and in the correct location under `lm_eval/tasks/`.

**Few-shot not working**: Check `fewshot_config` or ensure the dataset has a training split.

**Metric errors**: Verify `metric_list` uses valid metric names and the `process_results` function returns matching keys.

**Server connection**: For Megatron, ensure the server is fully initialized before evaluations start. Use `--ready-dir` for automatic synchronization.
