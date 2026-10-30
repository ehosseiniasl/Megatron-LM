# kmmlu_pro

### Paper

Title: `From KMMLU-Redux to KMMLU-Pro: A Professional Korean Benchmark Suite for LLM Evaluation`

Abstract: `We introduce KMMLU-PRO, a challenging new benchmark comprising 2,822 problems from the official exams for Korean National Professional Licensure (KNPL), representing highly specialized professions in Korea. To bridge the gap between LLM performance and real-world applicability, our evaluation closely mirrors the official certification criteria including reporting the number of professional licenses each LLM could pass according to these standards. All data are collected directly from the official sources for each license, and human annotators rigorously review each item to ensure quality and minimize noise. KMMLU-PRO includes only the most recent year's exams and will be updated annually, ensuring both long-term reliability and minimal risk of contamination.`

Homepage: https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro

Code: https://github.com/LG-AI-EXAONE/KMMLU-Pro

### Citation

```bibtex
@misc{hong2025kmmlureduxkmmluproprofessionalkorean,
      title={From KMMLU-Redux to KMMLU-Pro: A Professional Korean Benchmark Suite for LLM Evaluation},
      author={Seokhee Hong and Sunkyoung Kim and Guijin Son and Soyeon Kim and Yeonjung Hong and Jinsik Lee},
      year={2025},
      eprint={2507.08924},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.08924},
}
```

### Groups and Tasks

#### Groups

* `kmmlu_pro`: 'All 4 KMMLU-Pro domains, evaluated with zero-shot chain-of-thought generation and exact-match answer extraction'

#### Tasks

The following tasks evaluate domains in the KMMLU-Pro dataset:
- `kmmlu_pro_law`
- `kmmlu_pro_tax_and_accounting`
- `kmmlu_pro_value_estimation`
- `kmmlu_pro_medicine`

These domains cover 14 Korean National Professional Licensure exams:
- Law: `법무사` (Certified Judicial Scrivener), `변호사` (Lawyer), `공인노무사` (Certified Public Labor Attorney), `변리사` (Certified Patent Attorney)
- Tax & Accounting: `공인회계사` (Certified Public Accountant), `세무사` (Certified Tax Accountant), `관세사` (Certified Customs Broker)
- Value Estimation: `손해사정사` (Certified Damage Adjuster), `감정평가사` (Certified Appraiser)
- Medicine: `한의사` (Doctor of Korean Medicine), `치과의사` (Dentist), `약사` (Pharmacist), `한약사` (Herb Pharmacist), `의사` (Physician)

### Notes

KMMLU-Pro contains only a `test` split, so this task uses zero-shot evaluation. The model is prompted to think step by step and to provide a final answer that can be extracted with the `custom-extract` filter.

The dataset is released under the CC-BY-NC-ND-4.0 license and requires Hugging Face access approval before use.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
