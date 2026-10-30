# MMLU-Redux

## Overview

MMLU-Redux is a subset of 5,700 manually re-annotated questions across 57 MMLU subjects from the [edinburgh-dawg/mmlu-redux-2.0](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0) dataset.

This implementation provides three evaluation variants:

1. **`mmlu_redux`** - Standard multiple choice evaluation (log-likelihood)
2. **`mmlu_redux_str`** - Generative evaluation (generates 1 token greedily)
3. **`mmlu_redux_cot`** - Chain-of-thought evaluation with "Let's think step by step" prompting

## Usage

Run the full benchmark with all 57 subjects:

```bash
# Standard multiple choice (log-likelihood)
lm_eval --model hf --model_args pretrained=<model_name> --tasks mmlu_redux --batch_size 8

# Generative (string matching)
lm_eval --model hf --model_args pretrained=<model_name> --tasks mmlu_redux_str --batch_size 8

# Chain-of-thought
lm_eval --model hf --model_args pretrained=<model_name> --tasks mmlu_redux_cot --batch_size 8
```

Run individual subjects:

```bash
# Example: anatomy subject with standard evaluation
lm_eval --model hf --model_args pretrained=<model_name> --tasks mmlu_redux_anatomy --batch_size 8

# Example: anatomy subject with generative evaluation
lm_eval --model hf --model_args pretrained=<model_name> --tasks mmlu_redux_str_anatomy --batch_size 8

# Example: anatomy subject with CoT
lm_eval --model hf --model_args pretrained=<model_name> --tasks mmlu_redux_cot_anatomy --batch_size 8
```

## Dataset Details

- **Dataset**: [edinburgh-dawg/mmlu-redux-2.0](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0)
- **Total Questions**: 5,700 (100 per subject)
- **Subjects**: 57 (all MMLU subjects)
- **License**: CC-BY-4.0
- **Paper**: [Are We Done with MMLU?](https://arxiv.org/abs/2406.04127)

## Evaluation Variants

### Default (`mmlu_redux`)
- **Type**: Multiple choice (log-likelihood comparison)
- **Format**: Standard MMLU prompt with A/B/C/D options
- **Metric**: Accuracy

### String Matching (`mmlu_redux_str`)
- **Type**: Generative (generates next token)
- **Format**: Standard MMLU prompt, model generates "A", "B", "C", or "D"
- **Max tokens**: 2
- **Metric**: Exact match

### Chain-of-Thought (`mmlu_redux_cot`)
- **Type**: Generative with reasoning
- **Format**: "Q: ... (A) ... (B) ... (C) ... (D) ...\nA: Let's think step by step."
- **Metric**: Exact match (extracts answer from reasoning)
- **Few-shot**: 0-shot by default

## Citation

```bibtex
@misc{gema2024mmlu,
      title={Are We Done with MMLU?}, 
      author={Aryo Pradipta Gema and Joshua Ong Jun Leang and Giwon Hong and Alessio Devoto and Alberto Carlo Maria Mancino and Rohit Saxena and Xuanli He and Yu Zhao and Xiaotang Du and Mohammad Reza Ghasemi Madani and Claire Barale and Robert McHardy and Joshua Harris and Jean Kaddour and Emile van Krieken and Pasquale Minervini},
      year={2024},
      eprint={2406.04127},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Notes

- Each subject contains exactly 100 questions from the test split
- The dataset includes error annotations for question quality analysis
- All variants aggregate results across all 57 subjects
- Version: 1.0

