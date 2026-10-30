from functools import partial

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


def format_cot_example(example, subject):
    prompt = f'The following is a multiple choice question about {subject}. Think step by step and then output the answer in the format of "The answer is (X)" at the end.\n\n'
    question = example["question"]
    options = example["options"]
    prompt += "Question: " + question + "\nOptions: "
    for i, opt in enumerate(options):
        prompt += f"{choices[i]}. {opt.strip()}\n"
    prompt += "Let's think step by step:"
    return prompt


doc_to_text_biology = partial(format_cot_example, subject="biology")
doc_to_text_business = partial(format_cot_example, subject="business")
doc_to_text_chemistry = partial(format_cot_example, subject="chemistry")
doc_to_text_computer_science = partial(format_cot_example, subject="computer science")
doc_to_text_economics = partial(format_cot_example, subject="economics")
doc_to_text_engineering = partial(format_cot_example, subject="engineering")
doc_to_text_health = partial(format_cot_example, subject="health")
doc_to_text_history = partial(format_cot_example, subject="history")
doc_to_text_law = partial(format_cot_example, subject="law")
doc_to_text_math = partial(format_cot_example, subject="math")
doc_to_text_other = partial(format_cot_example, subject="other")
doc_to_text_philosophy = partial(format_cot_example, subject="philosophy")
doc_to_text_physics = partial(format_cot_example, subject="physics")
doc_to_text_psychology = partial(format_cot_example, subject="psychology")


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["category"] == subject)


process_biology = partial(process_docs, subject="biology")
process_business = partial(process_docs, subject="business")
process_chemistry = partial(process_docs, subject="chemistry")
process_computer_science = partial(process_docs, subject="computer science")
process_economics = partial(process_docs, subject="economics")
process_engineering = partial(process_docs, subject="engineering")
process_health = partial(process_docs, subject="health")
process_history = partial(process_docs, subject="history")
process_law = partial(process_docs, subject="law")
process_math = partial(process_docs, subject="math")
process_other = partial(process_docs, subject="other")
process_philosophy = partial(process_docs, subject="philosophy")
process_physics = partial(process_docs, subject="physics")
process_psychology = partial(process_docs, subject="psychology")


def exact_match_passk(doc, results):
    candidates = results[0]
    doc_results = [
        {
            "passed": pred.upper() == doc["answer"].upper(),
            "gold": doc["answer"],
            "answer": pred,
        }
        for pred in candidates
    ]
    return estimate_pass_at_k_dynamic(doc, [doc_results])
