# MultiPL-E Task for LM Evaluation Harness

This task implements the MultiPL-E benchmark, which evaluates large language models on code generation across multiple programming languages.

## Overview

MultiPL-E is a dataset for evaluating large language models for code generation that supports 18+ programming languages. It takes the OpenAI "HumanEval" and the MBPP Python benchmarks and uses little compilers to translate them to other languages.

- Homepage: https://nuprl.github.io/MultiPL-E/
- Paper: [MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation](https://doi.org/10.1109/TSE.2023.3267446)

## Setup

### Step 1: Install Language Compilers with Pixi

The evaluation requires language-specific compilers. We use [pixi](https://pixi.sh) to manage these dependencies in an isolated environment.

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install all compiler dependencies (from repo root)
pixi install
```

This installs: Node.js, TypeScript, Go, Rust, PHP, Ruby, R, Julia, Lua, Perl, and a C++ compiler.

### Step 2: Set Up Python Environment

Create a virtual environment for the Python dependencies:

```bash
# Create and activate virtual environment
uv venv env
source env/bin/activate

# Install lm-eval and dependencies
uv pip install -e ".[code_eval]"
```

### Step 3: Enable Code Evaluation

Code evaluation is disabled by default for safety. Enable it by setting:

```bash
export HF_ALLOW_CODE_EVAL=1
```

## Running Evaluations

### Using Pixi (Recommended)

Pixi tasks automatically activate both the compiler environment and Python venv:

```bash
# Run the full multiple benchmark
pixi run lm_eval_code --model hf --model_args pretrained=bigcode/starcoder --tasks multiple --batch_size 1

# Run specific language tasks
pixi run lm_eval_code --model hf --model_args pretrained=bigcode/starcoder --tasks multiple-js,multiple-rs --batch_size 1

# Test the evaluation pipeline
pixi run test_multiple
```

### Manual Invocation

If you prefer to run commands manually:

```bash
# Activate both environments
source env/bin/activate
eval "$(pixi shell-hook)"

# Run evaluation
lm_eval --model hf --model_args pretrained=bigcode/starcoder --tasks multiple-js --batch_size 1
```

## Supported Languages

| Task | Language | Compiler |
|------|----------|----------|
| `multiple-js` | JavaScript | Node.js |
| `multiple-ts` | TypeScript | tsc + Node.js |
| `multiple-cpp` | C++ | g++ |
| `multiple-rs` | Rust | rustc |
| `multiple-go` | Go | go |
| `multiple-php` | PHP | php |
| `multiple-rb` | Ruby | ruby |
| `multiple-r` | R | Rscript |
| `multiple-sh` | Bash | bash |

## Graceful Degradation

If a compiler is not available, the evaluation will **not fail**. Instead:
- Each sample will be marked with `status: "CompilerNotFound"`
- The sample will be treated as not passing
- A warning will be logged once per language
- You can later rescore the samples when compilers are available

This allows you to:
1. Generate samples on a machine without compilers
2. Transfer the samples JSON to a machine with compilers
3. Rescore using the standalone script

## Rescoring Samples

To rescore samples from a previous run (e.g., after installing missing compilers):

```bash
# Rescore a samples file (uses pixi for compilers)
pixi run rescore_multiple path/to/samples_multiple-js.jsonl

# Rescore with custom output location
pixi run rescore_multiple samples.jsonl --output rescored.jsonl

# Rescore multiple files
pixi run rescore_multiple samples_multiple-*.jsonl
```

The rescoring script:
- Reads the original samples JSONL file
- Re-executes code evaluation for each sample
- Computes pass@k metrics
- Outputs updated samples and aggregated results

## Metrics

The task reports pass@k metrics with dynamic k values based on the number of samples (N):
- `pass@1[N]`: Fraction of problems where the first sample passes
- `pass@2[N]`, `pass@4[N]`, ...: Powers of 2 up to N
- `pass@N[N]`: Fraction where at least one of N samples passes

Default configuration uses N=64 samples per problem.

## Troubleshooting

### Compiler not found
```
Warning: Compiler for 'js' not found (node). Samples will be marked as CompilerNotFound.
```
Install the missing compiler via pixi or your system package manager.

### HF_ALLOW_CODE_EVAL error
```
ValueError: Code evaluation is disabled by default.
```
Set `export HF_ALLOW_CODE_EVAL=1` before running.

### Timeout errors
Some problems may timeout (default 5-30s depending on language). This is expected for complex solutions or infinite loops.
