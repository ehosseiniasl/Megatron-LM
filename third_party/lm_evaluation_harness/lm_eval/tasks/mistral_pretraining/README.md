# Mistral pretraining evaluations

These tasks reproduce the completion prompts, few-shot selection, generation
settings, and primary scoring behavior of matching Thunderdome tasks. Run all
compatible tasks with `--tasks mistral_pretraining`, or select an individual task by
its `_mistral` name. `mmlu_redux_mistral` is a 57-subject group. When invoking
lm-eval manually against Megatron's completion endpoint, include `add_bos=true`
in `--model_args`; the provided launcher does this automatically.

## Implemented tasks

| Tasks | Thunderdome scoring reproduced here |
| --- | --- |
| `hellaswag_mistral`, `piqa_mistral`, `siqa_mistral` | Log-likelihood divided by UTF-8 byte length |
| `commonsense_qa_mistral`, `openbookqa_mistral` | Conditional log-likelihood minus the `Answer:` baseline |
| `race_high_mistral`, `race_middle_mistral` | Conditional log-likelihood minus the `A:` baseline |
| `truthfulqa_mistral`, `winogrande_mistral` | Raw log-likelihood, including TruthfulQA's multiple-correct-answer target |
| `gpqa_diamond_0shot_mistral`, `gpqa_diamond_5shot_mistral` | Raw log-likelihood, deterministic answer shuffling, and newline suffix where applicable |
| `new_mmlu_v2_mistral` | Raw log-likelihood with canonical subject-specific dev examples and newline-suffixed choices |
| `mmlu_redux_mistral` | Thunderdome row filtering/corrected labels, same-split canonical examples, and newline-suffixed choices |
| `naturalqs_5shot_mistral`, `triviaqa_5shot_mistral`, `simpleqa_5shot_mistral`, `popqa_5shot_mistral` | Thunderdome prompt, seeded few-shot sampling, stop strings, and normalized exact match |
| `gsm8k_5shot_maj@1_v3_mistral` | `Final answer` v3 extraction and exact match; `maj@1` is greedy single-sample generation |
| `drop_3shot_v2_mistral` | Thunderdome primary-answer preprocessing and token F1 (reported on Thunderdome's 0-100 scale) |
| `super_gpqa_cot_5shot_mistral` | Public SuperGPQA preprocessing, fixed five-shot CoT prompt, last-answer extraction, and exact match |

The compatibility sampler deliberately mirrors Thunderdome's persistent
`random.Random(0)`, decontamination check, and set iteration. The MMLU tasks use
their separate canonical first-example behavior instead.

## Launcher

Set `RUN_PATH` and `CHECKPOINT_ITER`, then run
`nvbench/launch_mistral_pretraining_leaderboard.sh`. The launcher submits one GB200
SLURM array task per evaluation and defaults to the `research` QoS. `IMAGE`,
`TOKENIZER`, `MLM_SRC_DIR`, and the SLURM resource variables can be overridden
through the environment.

## Not mirrored

The following apparent overlaps were not given `_mistral` aliases because a
like-named alias would imply parity that the available artifacts do not support:

- `new_arc_challenge` uses a different ARC dataset/evaluation representation.
- Thunderdome currently exposes MMLU-Pro v3, while the requested task is v2.
- AGI-Eval and BBH use locally preprocessed task collections that do not match
  the harness fork's packaged rows and choice construction closely enough.
- BoolQ, SQuAD, TyDiQA, QuAC, FLORES, translated HellaSwag/MMLU variants, CMMLU,
  JMMLU, and KMMLU do not have matching data tasks in this harness fork.
- `pile_wiki_2000points` and the crawl/code perplexity tasks depend on exact
  local/private subsets or ordering unavailable to the harness task definitions.
- HumanEval, HumanEval Plus, MBPP, CruxEval, TestEval, LiveCodeBench, and other
  code tasks require scorer, sanitizer, and execution-environment parity. A
  renamed task alone would preserve the mismatch, including the known
  HumanEval Plus discrepancy.
- MATH/Odyssey/AMC judge tasks and the remaining GPQA/MMLU-Pro chain-of-thought
  variants require Thunderdome's judge or task-specific generation scorers.
