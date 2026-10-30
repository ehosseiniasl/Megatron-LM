# WMT24++

WMT24++ evaluates translation from English into 55 languages and locale
variants. The tasks use the complete published evaluation data after removing
canary rows and rows marked `is_bad_source`, as recommended by the dataset
authors.

Each task is zero-shot and reports corpus chrF++ on a normalized 0-1 scale.
Run one locale directly with lm-eval:

```bash
lm_eval --model hf --model_args pretrained=... \
  --tasks wmt24pp_en_xx_de_DE
```

Run all published language pairs with `--tasks wmt24pp_en_xx`.

Dataset: https://huggingface.co/datasets/google/wmt24pp
