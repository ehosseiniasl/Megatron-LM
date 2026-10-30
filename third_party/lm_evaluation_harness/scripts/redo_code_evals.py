import json
from collections import defaultdict

from lm_eval.metrics.code_eval import estimate_pass_at_k_dynamic
from lm_eval.tasks.humaneval.utils import (
    score_predictions as score_humaneval_predictions,
)
from lm_eval.tasks.mbpp.utils import score_predictions as score_mbpp_predictions


def re_eval(fname: str):
    if "humaneval" in fname:
        score_predictions = score_humaneval_predictions
    elif "mbpp" in fname:
        score_predictions = score_mbpp_predictions
    else:
        raise ValueError("Not supported yet!")

    print(f"Re-evaling {fname}")
    with open(fname, "r", encoding="utf-8") as fp:
        docs, resps, M = [], [], defaultdict(list)
        for line in fp:
            doc = json.loads(line.strip())
            resps.append(doc["resps"][0])
            docs.append(doc["doc"])
        filtered_resps = score_predictions(resps, docs)
        for filtered_resp, doc in zip(filtered_resps, docs):
            metrics = estimate_pass_at_k_dynamic(doc, [filtered_resp])
            doc["filtered_resps"] = {**filtered_resp[0], **metrics}
            for k, v in metrics.items():
                if "pass@1[" in k:
                    M[k].append(v)

    for k, v in M.items():
        print(k, sum(v) / len(v))


if __name__ == "__main__":
    from glob import glob

    dir = "/Users/sasatheesh/Downloads/00/model_158945_1000111_2/"
    fnames = list(glob(f"{dir}*humaneval*")) + list(glob(f"{dir}*mbpp*"))
    for fname in fnames:
        re_eval(fname)
