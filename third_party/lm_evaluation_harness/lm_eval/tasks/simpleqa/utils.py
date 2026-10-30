import re
import string
import unicodedata


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_NOT_ATTEMPTED_RE = re.compile(
    r"\b("
    r"i don'?t know|i do not know|i'?m not sure|i am not sure|"
    r"i'?m not certain|i am not certain|i cannot|i can'?t|"
    r"i'?m unable|i am unable|i have no (?:idea|information|knowledge)|"
    r"i don'?t have|i do not have|cannot be determined|"
    r"i need more (?:information|context|data)|"
    r"not (?:enough|sufficient) (?:information|context|data)|"
    r"unknown|not sure"
    r")\b",
    re.IGNORECASE,
)


def normalize_answer(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = _ARTICLES_RE.sub(" ", text)

    punctuation = string.punctuation.replace(".", "").replace("%", "")
    text = text.translate(str.maketrans({char: " " for char in punctuation}))
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    return " ".join(text.split())


def _tokens(text):
    return set(text.split())


def _gold_is_present(pred_norm, gold_norm):
    if not pred_norm or not gold_norm:
        return False
    if gold_norm in pred_norm:
        return True

    pred_tokens = _tokens(pred_norm)
    gold_tokens = _tokens(gold_norm)
    if pred_tokens == gold_tokens:
        return True

    if pred_norm in gold_norm and len(pred_tokens) >= max(1, len(gold_tokens) - 1):
        return True

    return False


def classify_answer(prediction, gold):
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if not pred_norm or not gold_norm:
        return "not_attempted"

    has_gold = _gold_is_present(pred_norm, gold_norm)
    if _NOT_ATTEMPTED_RE.search(str(prediction)):
        return "correct" if has_gold else "not_attempted"

    if has_gold:
        return "correct"

    return "incorrect"


def process_results(doc, results):
    grade = classify_answer(results[0], doc["answer"])
    is_correct = int(grade == "correct")
    is_incorrect = int(grade == "incorrect")
    is_not_attempted = int(grade == "not_attempted")
    return {
        "correct": is_correct,
        "incorrect": is_incorrect,
        "not_attempted": is_not_attempted,
        "f1": (is_correct, is_incorrect, is_not_attempted),
    }


def simpleqa_f1_agg(items):
    correct = sum(item[0] for item in items)
    incorrect = sum(item[1] for item in items)
    not_attempted = sum(item[2] for item in items)
    total = correct + incorrect + not_attempted
    if total == 0:
        return 0.0

    correct_rate = correct / total
    attempted = correct + incorrect
    accuracy_given_attempted = correct / attempted if attempted else 0.0
    denominator = correct_rate + accuracy_given_attempted
    if denominator == 0:
        return 0.0
    return 2 * correct_rate * accuracy_given_attempted / denominator
