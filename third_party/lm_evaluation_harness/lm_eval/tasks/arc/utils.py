def doc_to_text_llama( doc):
    # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
    # of {'1', '2', '3', '4', '5'}. We map them back to letters.
    letters = ["A", "B", "C", "D", "E"]
    query = f"""Question: {doc["question"]}"""
    for idx, c in enumerate(doc["choices"]["text"]):
        query += f"""\n{letters[idx]}. {doc["choices"]["text"][idx]}"""
    query += "\nAnswer:"

    return query

def doc_to_choice_llama(doc):
    letters = ["A", "B", "C", "D", "E"]
    choices = []
    for idx, _ in enumerate(doc["choices"]["text"]):
        choices.append(f"""{letters[idx]}""")
    return choices

def doc_to_target_llama(doc):
    # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
    # of {'1', '2', '3', '4', '5'}. We map them back to letters.
    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
    return ["A", "B", "C", "D", "E"].index(doc["answerKey"])