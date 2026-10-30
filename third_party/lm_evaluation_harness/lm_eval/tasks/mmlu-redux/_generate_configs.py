"""
Generate MMLU-Redux configs for all subjects.

Embeds few-shot examples from the MMLU dev split (hails/mmlu_no_train)
so that few-shot prompts are identical to standard MMLU evaluation.
"""

import yaml
from datasets import load_dataset

SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]


def load_mmlu_fewshot_samples(subject: str) -> list[dict]:
    ds = load_dataset("hails/mmlu_no_train", subject, split="dev", trust_remote_code=True)
    return [{"question": ex["question"], "choices": ex["choices"], "answer": ex["answer"]} for ex in ds]


def create_all_configs():
    fewshot_by_subject = {}
    for subject in SUBJECTS:
        print(f"Loading dev examples for {subject}...")
        fewshot_by_subject[subject] = load_mmlu_fewshot_samples(subject)

    for subject in SUBJECTS:
        samples = fewshot_by_subject[subject]
        fewshot_config = {"sampler": "first_n", "samples": samples}

        for variant, template in [("default", "_mmlu_redux_template_yaml"),
                                  ("str", "_mmlu_redux_str_template_yaml"),
                                  ("cot", "_mmlu_redux_cot_template_yaml")]:
            if variant == "default":
                task_name = f"mmlu_redux_{subject}"
                filename = f"mmlu_redux_{subject}.yaml"
            elif variant == "str":
                task_name = f"mmlu_redux_str_{subject}"
                filename = f"mmlu_redux_str_{subject}.yaml"
            else:
                task_name = f"mmlu_redux_cot_{subject}"
                filename = f"mmlu_redux_cot_{subject}.yaml"

            description = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
            config = {
                "task": task_name,
                "task_alias": subject,
                "dataset_name": subject,
                "description": description,
                "include": template,
            }
            if variant != "cot":
                config["fewshot_config"] = fewshot_config

            with open(filename, "w") as f:
                yaml.dump(config, f, sort_keys=False, allow_unicode=True, width=200)


if __name__ == "__main__":
    create_all_configs()
