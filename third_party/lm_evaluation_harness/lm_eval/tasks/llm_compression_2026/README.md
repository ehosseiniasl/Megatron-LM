# LLM Compression 2026

This task evaluates a text-only export of the 2026 multimodal refresh of the LLM compression corpora. The four configured subsets are packaged in `llm_compression_2026_eval.parquet` through Git LFS, so evaluation never contacts the private `nvidia/llm-compression-intelligence-2026` Hugging Face repository.

The task is text-only. It scores the row `content` field with `loglikelihood_rolling` and ignores the binary `images` column. Synthetic Markdown image references of the form `![...](image://...)` are stripped before scoring so BPB and perplexity are computed on text tokens rather than embedded image handles. Captions or other text that appears outside those image markers remains part of the scored text.

## Groups and Tasks

Group:

- `llm_compression_2026`

Tasks:

- `llm_compression_2026_cc_interleaved`
- `llm_compression_2026_arxiv_math_interleaved`
- `llm_compression_2026_arxiv_physics_interleaved`
- `llm_compression_2026_code_repo_packed`

## Packaged Data

The packaged Parquet contains `doc_id`, `source`, `content`, and `image_count`, plus a `dataset` discriminator. Embedded image bytes and metadata are intentionally omitted. The adjacent manifest pins the Hugging Face repository commit and records every source Parquet LFS hash, source size, row count, selected-column compressed size, and a deterministic digest of the exported rows.

Use `scripts/build_llm_compression_2026_parquet.py` to reproduce the export. The builder uses Parquet column projection, so it does not download embedded image byte columns.

Run `git lfs pull` after cloning. `LM_EVAL_LLM_COMPRESSION_2026_PARQUET_PATH` can override the packaged artifact with another combined Parquet.

## Local Overrides

For development, set `LM_EVAL_LLM_COMPRESSION_2026_DATA_DIR` to a directory containing one raw file or saved Hugging Face Dataset per subset:

- `cc_interleaved`
- `arxiv_math_interleaved`
- `arxiv_physics_interleaved`
- `code_repo_packed`

Each subset can be JSONL, JSON, Parquet, or a directory produced by `Dataset.save_to_disk`. An exact path can instead be supplied with an environment variable such as `LM_EVAL_LLM_COMPRESSION_2026_CC_INTERLEAVED_PATH`.

The loader fails locally with a path-specific message when data is missing; it does not fall back to the private Hub repository.

## Rolling Token Window

By default the benchmark leaves rolling-window sizing to the model adapter, preserving normal `max_length` behavior. To make a run self-describing from the task config, set `metadata.token_window`:

```bash
lm-eval ... \
  --model_args ...,tokenizer=<hf-tokenizer-or-local-path> \
  --tasks llm_compression_2026 \
  --override metadata.token_window=65536
```

The `notok-completions` adapter requires `tokenizer=...` for `loglikelihood_rolling` and uses it to build real token-stride windows. It fails rather than falling back to a character approximation when no tokenizer is provided. The serving backend must still be launched with a compatible context length, such as vLLM `--max-model-len 65536`.

With `nvbench/launch_evals.py`, use `-e megatron --tokenizer <hf-tokenizer-or-local-path>` for Megatron text-completions endpoints. vLLM endpoints continue to use `-e vllm`, which defaults the tokenizer to the model name unless `--tokenizer` is provided.

## Metrics

- `word_perplexity`
- `byte_perplexity`
- `bits_per_byte`

Lower values indicate better compression.
