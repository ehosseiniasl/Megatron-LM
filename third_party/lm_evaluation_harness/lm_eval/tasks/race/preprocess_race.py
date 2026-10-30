import ast

choices = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def process_ast(string):
    return ast.literal_eval(string)


def last_problem(doc):
    return process_ast(doc["problems"])[-1]


def get_answer_option(problem):
    return problem["answer"]


def doc_to_choice(doc):
    return choices[: len(last_problem(doc)["options"])]


def problem_to_text(problem, include_answer=True):
    return "\n".join(
        [
            f"Question: {problem['question'].strip()}",
            *[
                f"{choice}: {option}"
                for choice, option in zip(choices, problem["options"])
            ],
            f"Answer:{' ' + problem['answer'] if include_answer else ''}",
        ]
    )


def doc_to_text(doc):
    text = "Article: " + doc["article"] + "\n\n"
    for problem in process_ast(doc["problems"])[:-1]:
        text += problem_to_text(problem) + "\n\n"
    text += problem_to_text(last_problem(doc), include_answer=False)
    return text


def doc_to_target(doc):
    return choices.index(last_problem(doc)["answer"])
