from lm_eval.filters.sanitize import sanitize
from lm_eval.metrics.code_eval import check_correctness
from lm_eval.metrics.code_eval import evaluate_candidates


## Formatting functions
def build_text(doc):
    description = doc["prompt"]
    test_example = doc["test_list"][0]
    prompt = f'"""\n{description}\n{test_example}\n"""\n'
    return prompt


def build_text_full_subset(doc):
    description = doc["text"]
    test_example = doc["test_list"][0]
    prompt = f'"""\n{description}\n{test_example}\n"""\n'
    return prompt


def build_references(doc):
    return "\n".join(doc["test_list"])


# NOTE(@ganler): MBPP+ extends the original MBPP jsonl data with a "test" field which
#                includes the testing code ready for execution. Note the "test" field
#                is different from HumanEval(+) which further requires a `check` func
MBBPPLUS_USE_MBPP_TESTS = 0  # bigcode-eval-harness default


def build_references_mbppplus(doc):
    if MBBPPLUS_USE_MBPP_TESTS == "1":
        return "\n".join(doc["test_list"])
    return "\n" + doc["test"]


def process_resp(repeat_idx: int, resp: str, doc: dict):
    candidate_code = sanitize("\n" + resp)
    if "test" in doc:
        test_case = build_references_mbppplus(doc)
    else:
        test_case = build_references(doc)
    test_program = candidate_code + "\n" + test_case
    result = check_correctness(
        test_program, timeout=3, task_id=doc.get("task_id", repeat_idx), completion_id=repeat_idx
    )
    return result


def non_sanitized_predictions(resps, docs):
    preds = []
    for resp, doc in zip(resps, docs):
        pred = [r for r in resp]
        preds.append(pred)
    return preds


def score_predictions(candidates: list[list[str]], docs: list[dict]):
    return evaluate_candidates(candidates, docs, process_resp)


## Rewrite few-shot sampler to sample from `code` instead of the target (`test`)
from lm_eval.api.samplers import ContextSampler


class FewShotSampler(ContextSampler):
    def get_context(self, doc, num_fewshot):
        assert self.config.fewshot_split != self.config.test_split
        assert self.config.doc_to_choice is None

        # draw `n_samples` docs from fewshot_docs
        fewshot_examples = self.sample(num_fewshot)

        labeled_examples = ""
        for doc in fewshot_examples:
            doc_content = self.doc_to_text(doc)
            doc_target = doc["code"]
            labeled_examples += doc_content
            labeled_examples += self.target_delimiter
            labeled_examples += doc_target
            labeled_examples += self.fewshot_delimiter

        return labeled_examples
