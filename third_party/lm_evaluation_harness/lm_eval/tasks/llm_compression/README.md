# LLM Compression

### Paper

Title: `Compression Represents Intelligence Linearly`

Abstract: https://arxiv.org/abs/2404.09937

This benchmark evaluates language models' compression ability on external text corpora, which has been shown to correlate linearly with intelligence metrics. The dataset contains three subsets representing different domains:

- **arxiv_math**: Mathematical reasoning corpus from arXiv papers (1,217 samples)
- **cc**: Knowledge and commonsense corpus from CommonCrawl (20,242 samples)
- **python**: Coding corpus from Python code (11,015 samples)

Homepage: https://huggingface.co/datasets/hkust-nlp/llm-compression

### Citation

```bibtex
@misc{huang2024compression,
    title={Compression Represents Intelligence Linearly}, 
    author={Yuzhen Huang and Jinghan Zhang and Zifei Shan and Junxian He},
    year={2024},
    eprint={2404.09937},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* `llm_compression`: Evaluates all three subsets and reports aggregated metrics

#### Tasks

The following tasks evaluate specific subsets:

* `llm_compression_arxiv_math`: arXiv math papers
* `llm_compression_cc`: CommonCrawl text
* `llm_compression_python`: Python code

### Metrics

This task uses `loglikelihood_rolling` evaluation to compute:
- `word_perplexity`: Word-level perplexity
- `byte_perplexity`: Byte-level perplexity  
- `bits_per_byte`: Bits per byte (compression metric)

Lower values indicate better compression ability.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

### Usage

Evaluate all subsets:
```bash
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks llm_compression \
    --device cuda:0 \
    --batch_size 8
```

Evaluate a specific subset:
```bash
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks llm_compression_arxiv_math \
    --device cuda:0 \
    --batch_size 8
```

### Version History

* [Nov 19, 2024] Initial implementation (v1.0): Added LLM Compression benchmark with three subsets (arxiv_math, cc, python) for evaluating model compression ability via bits-per-byte, byte perplexity, and word perplexity metrics.

