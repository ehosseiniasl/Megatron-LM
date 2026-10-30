from functools import partial

choices = [
    "A",
    "B",
    "C",
    "D",
]


def format_cot_example(example, including_answer=True):
    prompt = "Question: "
    question = example["question"].strip()
    options = example["choices"]
    prompt += question + "\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i].strip(), opt)
    if including_answer:
        cot_content = "Answer: " + example["solution"].strip()
        prompt += cot_content.strip()  # + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)


def empty_str(x):
    return ""


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["subject"] == subject)


process_abstract_algebra = partial(process_docs, subject="abstract_algebra")
process_anatomy = partial(process_docs, subject="anatomy")
process_astronomy = partial(process_docs, subject="astronomy")
process_business_ethics = partial(process_docs, subject="business_ethics")
process_clinical_knowledge = partial(process_docs, subject="clinical_knowledge")
process_college_biology = partial(process_docs, subject="college_biology")
process_college_chemistry = partial(process_docs, subject="college_chemistry")
process_college_computer_science = partial(
    process_docs, subject="college_computer_science"
)
process_college_mathematics = partial(process_docs, subject="college_mathematics")
process_college_medicine = partial(process_docs, subject="college_medicine")
process_college_physics = partial(process_docs, subject="college_physics")
process_computer_security = partial(process_docs, subject="computer_security")
process_conceptual_physics = partial(process_docs, subject="conceptual_physics")
process_econometrics = partial(process_docs, subject="econometrics")
process_electrical_engineering = partial(process_docs, subject="electrical_engineering")
process_elementary_mathematics = partial(process_docs, subject="elementary_mathematics")
process_formal_logic = partial(process_docs, subject="formal_logic")
process_global_facts = partial(process_docs, subject="global_facts")
process_high_school_biology = partial(process_docs, subject="high_school_biology")
process_high_school_chemistry = partial(process_docs, subject="high_school_chemistry")
process_high_school_computer_science = partial(
    process_docs, subject="high_school_computer_science"
)
process_high_school_european_history = partial(
    process_docs, subject="high_school_european_history"
)
process_high_school_geography = partial(process_docs, subject="high_school_geography")
process_high_school_government_and_politics = partial(
    process_docs, subject="high_school_government_and_politics"
)
process_high_school_macroeconomics = partial(
    process_docs, subject="high_school_macroeconomics"
)
process_high_school_mathematics = partial(
    process_docs, subject="high_school_mathematics"
)
process_high_school_microeconomics = partial(
    process_docs, subject="high_school_microeconomics"
)
process_high_school_physics = partial(process_docs, subject="high_school_physics")
process_high_school_psychology = partial(process_docs, subject="high_school_psychology")
process_high_school_statistics = partial(process_docs, subject="high_school_statistics")
process_high_school_us_history = partial(process_docs, subject="high_school_us_history")
process_high_school_world_history = partial(
    process_docs, subject="high_school_world_history"
)
process_human_aging = partial(process_docs, subject="human_aging")
process_human_sexuality = partial(process_docs, subject="human_sexuality")
process_international_law = partial(process_docs, subject="international_law")
process_jurisprudence = partial(process_docs, subject="jurisprudence")
process_logical_fallacies = partial(process_docs, subject="logical_fallacies")
process_machine_learning = partial(process_docs, subject="machine_learning")
process_management = partial(process_docs, subject="management")
process_marketing = partial(process_docs, subject="marketing")
process_medical_genetics = partial(process_docs, subject="medical_genetics")
process_miscellaneous = partial(process_docs, subject="miscellaneous")
process_moral_disputes = partial(process_docs, subject="moral_disputes")
process_moral_scenarios = partial(process_docs, subject="moral_scenarios")
process_nutrition = partial(process_docs, subject="nutrition")
process_philosophy = partial(process_docs, subject="philosophy")
process_prehistory = partial(process_docs, subject="prehistory")
process_professional_accounting = partial(
    process_docs, subject="professional_accounting"
)
process_professional_law = partial(process_docs, subject="professional_law")
process_professional_medicine = partial(process_docs, subject="professional_medicine")
process_professional_psychology = partial(
    process_docs, subject="professional_psychology"
)
process_public_relations = partial(process_docs, subject="public_relations")
process_security_studies = partial(process_docs, subject="security_studies")
process_sociology = partial(process_docs, subject="sociology")
process_us_foreign_policy = partial(process_docs, subject="us_foreign_policy")
process_virology = partial(process_docs, subject="virology")
process_world_religions = partial(process_docs, subject="world_religions")
