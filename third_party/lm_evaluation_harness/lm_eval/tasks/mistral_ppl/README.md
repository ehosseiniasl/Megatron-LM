# Mistral Perplexity Evaluations

This group evaluates Mistral's `arxiv_val_ppl` and `catlm_v2` validation corpora through `loglikelihood_rolling`. The 8,192-token window produces an approximately 4K-token overlap in the `notok-completions` adapter.

## Tasks

- `mistral_ppl_arxiv_val_ppl`
- `mistral_ppl_catlm_v2`
- `mistral_ppl` (aggregate group)

## Packaged data

Both datasets are stored in `mistral_ppl_eval.parquet`, tracked with Git LFS. Run `git lfs pull` after cloning the repository. The Parquet has one row per validation sequence and the columns `dataset`, `id`, `text`, `num_tokens`, and `split`. The adjacent manifest records source URIs, byte counts, row counts, and SHA-256 digests.

`arxiv_val_ppl` comes from `s3://nvidia-mistral-sharing/arxiv_val_ppl/data.jsonl`. `catlm_v2` comes from the Lobstenizer-tokenized CatLM validation set shared under `s3://nvidia-mistral-sharing/share-bundle-muon-e32/eval_data/catlm_pretokenized_lobstenizer/merged_1099511627776b/`. It was decoded with `lobstenizer-tekken-hf` while preserving special tokens other than a leading BOS token. The detokenizer removes that token only when the first token ID equals `tokenizer.bos_token_id`; it does not assume the decoded spelling is `<s>`.

To override the packaged artifact, set `LM_EVAL_MISTRAL_PPL_PARQUET_PATH` to another combined Parquet. Separate JSONL, JSON, Parquet, or `Dataset.save_to_disk` inputs can still be supplied with `LM_EVAL_MISTRAL_PPL_DATA_DIR` or per-subset path variables.

## Rebuilding

Use `scripts/detokenize_megatron_indexed.py` to produce the CatLM JSONL and detokenization manifest, then run `scripts/build_mistral_ppl_parquet.py` with that output and the raw arXiv JSONL. Both scripts refuse to overwrite existing outputs, and the builder emits a new provenance manifest alongside the Parquet.

Rows supplied through local overrides must contain a string `content` or `text` field.
