# CRUXEval Task

CRUXEval (Code Reasoning, Understanding, and eXecution Evaluation) is a benchmark of 800 Python functions and input-output pairs. The benchmark consists of two tasks:

- **CRUXEval-I (input prediction)**: Generate function calls that would produce an expected output
- **CRUXEval-O (output prediction)**: Predict the output of a function given its input

## Metrics

The primary metrics for CRUXEval are:

- **pass@1**: The fraction of problems where the first generated solution is correct
- **pass@5**: The fraction of problems where at least one of the first 5 generated solutions is correct

## Implementation Details

The task uses Python code execution to verify the correctness of generated inputs and outputs. The evaluation:

1. For input prediction: Runs the function with the generated input to check if it produces the expected output
2. For output prediction: Compares the generated output with the result of executing the function on the given input

## Resource Requirements

The benchmark is designed to be lightweight:
- Functions are short and focused on core programming concepts
- Default execution timeout is 3 seconds per function
- Memory limit is set to 100MB per function

## Data Source

The benchmark data is loaded from the HuggingFace dataset: [cruxeval-org/cruxeval](https://huggingface.co/datasets/cruxeval-org/cruxeval)

## Citation

```
@article{gu2024cruxeval,
  title={CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution},
  author={Alex Gu and Baptiste Rozière and Hugh Leather and Armando Solar-Lezama and Gabriel Synnaeve and Sida I. Wang},
  year={2024},
  journal={arXiv preprint arXiv:2401.03065},
}
```

## References

- [CRUXEval Website](https://crux-eval.github.io/)
- [CRUXEval GitHub Repository](https://github.com/facebookresearch/cruxeval)
