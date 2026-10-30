import os
import json

HUMAN_PROMPT = None
AI_PROMPT = None

from lm_eval.tasks.livecodebench.codegen.types import CodeGenerationProblem

DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(DIR, "func.json")) as f:
    func = json.load(f)

with open(os.path.join(DIR, "stdin.json")) as f:
    stdin = json.load(f)


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = (
        "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct "
        "Python program that matches the specification and passes all tests."
    )

    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = (
        "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
        "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
    )


def get_generic_question_template_answer(question: CodeGenerationProblem):
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def base_model_prompt(question: CodeGenerationProblem):
    if question.starter_code:
        examples_json = func
    else:
        examples_json = stdin

    def get_example_prompt(example):
        prompt = ""
        prompt += "### Question\n"
        prompt += example["question"]
        prompt += "\n\n"
        if question.starter_code:
            prompt += "### Starter Code\n"
            prompt += example["sample_code"]
            prompt += "\n\n"
        prompt += "### Answer\n\n"
        prompt += example["answer"]
        if example["answer"]:
            prompt += "\n\n"
        return prompt

    prompt = ""
    prompt += get_example_prompt(examples_json[0])
    prompt += get_example_prompt(
        {
            "question": question.question_content,
            "sample_code": question.starter_code,
            "answer": "",
        }
    )
    return prompt


def aligned_model_prompt(question: CodeGenerationProblem) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
        },
        {
            "role": "user",
            "content": get_generic_question_template_answer(question),
        },
    ]
