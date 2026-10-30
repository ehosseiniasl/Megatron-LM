# MMLU-Pro 10% Subsample (5-shot)

This task is a 10% subsample of the MMLU-Pro dataset with category-based few-shot sampling. The subsampling is done randomly, and the few-shot examples are selected from the same category as the test example.

## Subsampling Details

The subsampling is implemented in the `utils.py` file using the `subsample_dataset` function. This function:

1. Takes a random sample of 10% of the examples from the dataset
2. Uses a fixed random seed (42) for reproducibility

The subsampling is controlled by two parameters:
- `sample_percentage`: The percentage of examples to include (default: 10)
- `seed`: Random seed for reproducibility (default: 42)

These parameters can be modified in the `process_docs_kwargs` section of the YAML file.

## Category-Based Few-Shot Sampling

This task uses a custom sampler class (`CategorySampler`) that selects examples from the same category (subject) as the test example. This is implemented in the `utils.py` file by extending the `ContextSampler` class from the lm-evaluation-harness framework.

The `CategorySampler` class:
1. Groups examples by category during initialization
2. Overrides the `get_context` method to select examples from the same category as the current document
3. Falls back to random sampling if no examples are available for a specific category

The few-shot sampling is configured in the YAML file with:
```yaml
fewshot_config:
  sampler: !function utils.CategorySampler
  doc_to_text: !function utils.fewshot_to_text
  doc_to_target: !function utils.empty_str
```

## Usage

To run the evaluation on this subsampled task:

```bash
python -m lm_eval --model <your-model> --tasks mmlu_pro_10p_5_shot_base
```

## Customization

If you want to create a different subsample or modify the few-shot sampling, you can:

1. Modify the `sample_percentage` parameter in the YAML file
2. Change the `num_fewshot` parameter to use a different number of few-shot examples
3. Modify the `CategorySampler` class in `utils.py` to implement a different sampling strategy

## Available Functions

The `utils.py` file provides several functions and classes:

1. `subsample_dataset`: Simple random subsampling
2. `CategorySampler`: A custom sampler class that selects examples from the same category as the current document 