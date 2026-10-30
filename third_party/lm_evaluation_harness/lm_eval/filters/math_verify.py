import logging
from typing import Optional

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

from math_verify import parse, verify


def safe_parse(s: str) -> list[str]:
    parsed = None
    try:
        parsed = parse(s)
    except Exception as e:
        print(e, s)
    return parsed or ["[invalid]"]


@register_filter("symbolic_match")
class MathVerifyFilter(Filter):
    """Filter that applies math_verify to each candidate answer."""

    def __init__(
        self,
        solution_fields: Optional[list[str]] = None,
    ) -> None:
        self.solution_field = None
        self.solution_fields = solution_fields or ["solution", "answer"]

    def apply(self, resps, docs):
        """Apply math verification to responses."""
        results = []
        for candidates_for_doc, doc in zip(resps, docs):
            doc_results = []
            for pred_answer in candidates_for_doc:
                # Determine ground truth solution based on available fields
                if self.solution_field is None:
                    fields = [
                        key
                        for key, value in doc.items()
                        if (
                            any(patt in key for patt in self.solution_fields)
                            and (value is not None)
                        )
                    ]
                    assert (
                        len(fields) >= 1
                    ), f"No valid ground truth solution found in document with keys {doc.keys()}"
                    self.solution_field = fields[0]
                    logging.warning(
                        f"Using field '{self.solution_field}' as ground truth solution"
                    )

                gt_solution = str(doc[self.solution_field])
                mv_is_correct = False
                result = {"passed": mv_is_correct}

                try:
                    assert isinstance(pred_answer, str)

                    mv_gold = safe_parse(gt_solution)
                    mv_answer = safe_parse(pred_answer)
                    mv_is_correct = verify(mv_gold, mv_answer)
                    result["passed"] = mv_is_correct
                    result["gold"] = mv_gold[1] if len(mv_gold) > 1 else mv_gold[0]
                    result["answer"] = mv_answer[1] if len(mv_answer) > 1 else mv_answer[0]
                except Exception as e:
                    logging.warning(f"Error verifying math_verify: {e}")

                doc_results.append(result)
            results.append(doc_results)
        return results
