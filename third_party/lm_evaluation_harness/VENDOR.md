# Vendored lm-evaluation-harness

This directory (`third_party/lm_evaluation_harness`) vendors a curated copy of `adlr/nemo5` from
`ssh://git@gitlab-master.nvidia.com:12051/swdl-nemollm-mlops/evals/lm-evaluation-harness.git`.

- Source branch: `adlr/nemo5`
- Source commit: `17abcb7d88e805e0edace00f3135736970dfa706`
- Imported content: tracked files only
- Local change: `lm_eval/tasks/ruler/english_words.json` is stored as
  `english_words.json.gz`, and `cwe_utils.py` loads it with `gzip.open`.
- Git LFS: intentionally not used in this vendored copy.

The import was placed under `third_party/` to keep the harness package, tests,
scripts, examples, and docs isolated from Megatron-LM's native top-level
directories.
