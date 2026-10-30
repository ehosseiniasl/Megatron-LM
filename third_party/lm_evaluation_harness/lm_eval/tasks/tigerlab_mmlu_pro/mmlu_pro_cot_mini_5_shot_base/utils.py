from functools import partial
import random
from lm_eval.api.samplers import ContextSampler
from lm_eval.metrics.code_eval import estimate_pass_at_k_dynamic


choices = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
]


def format_cot_example(example, including_answer=True):
    prompt = "Question: "
    question = example["question"]
    prompt += question + "\n"
    prompt += "Options: "
    options = example["options"]
    for i, opt in enumerate(options):
        prompt += f"{choices[i]}. {opt.strip()}\n"
    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)


def empty_str(x):
    return ""


def subsample_train_dataset_10p(dataset):
    if len(dataset) == 70:
        print("skip subsampling of validation set")
        return dataset

    seed, sample_percentage = 42, 10
    random.seed(seed)
    dataset = dataset.sort("question_id")
    total_size = len(dataset)
    num_examples = int(total_size * sample_percentage / 100)
    indices_to_keep = sorted(random.sample(range(total_size), num_examples))
    return dataset.select(indices_to_keep)


def exact_match(doc: dict, candidates: list[dict]) -> list[dict]:
    doc_results = []
    candidates = candidates[0]
    for pred_answer in candidates:
        result = {
            "passed": pred_answer == doc["answer"],
            "gold": doc["answer"],
            "answer": pred_answer,
        }
        doc_results.append(result)
    return estimate_pass_at_k_dynamic(doc, [doc_results])


class CategorySampler(ContextSampler):
    # This is a customized implementation of the ContextSampler class based on lm_eval/api/samplers.py
    # The main difference is that it samples examples from the same category for few-shot examples
    # Maybe we can put it in lm_eval/api/samplers.py eventually.

    def __init__(self, docs, task, fewshot_indices=None, rnd=None):
        super().__init__(docs, task, fewshot_indices, rnd)
        self.category_examples = {}
        for example in self.docs:
            category = example["category"]
            if category not in self.category_examples:
                self.category_examples[category] = []
            self.category_examples[category].append(example)

    def get_context(self, doc, num_fewshot) -> str:
        category = doc["category"]
        selected_docs = self.category_examples[category][:num_fewshot]

        labeled_examples = ""
        # no need to add target, it is already in the doc_to_text function for fewshot in the cot-content
        for example in selected_docs:
            doc_content = self.doc_to_text(example)
            labeled_examples += doc_content
            labeled_examples += self.fewshot_delimiter

        return labeled_examples
