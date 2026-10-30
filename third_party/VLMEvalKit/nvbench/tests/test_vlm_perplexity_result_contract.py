"""Tests for conditional-perplexity alignment and pooled math."""

from __future__ import annotations
import math
import unittest

from orchestration.vlm_perplexity_result_contract import (
    ConditionalPerplexityResultError,
    ConditionalPerplexityScore,
    aggregate_conditional_perplexity,
    description_token_suffix,
    score_description,
)


class ConditionalPerplexityResultContractTest(unittest.TestCase):
    def test_smallest_unicode_suffix_and_score(self) -> None:
        prompt = list("prefix cafe\N{COMBINING ACUTE ACCENT}")
        selected = description_token_suffix(prompt, "café")
        self.assertEqual("".join(prompt[selected]), "café")
        logprobs = [None] + [-0.25] * (len(prompt) - 1)
        score = score_description(
            uid="id",
            category="chart",
            description="café",
            prompt_tokens=prompt,
            prompt_logprobs=logprobs,
        )
        self.assertEqual(score.token_count, 5)
        self.assertAlmostEqual(score.loglikelihood, -1.25)
        self.assertEqual(score.byte_count, len("café".encode("utf-8")))

    def test_pools_counts_instead_of_sample_perplexities(self) -> None:
        scores = (
            ConditionalPerplexityScore("a", "chart", -2.0, 2, 4, 1),
            ConditionalPerplexityScore("b", "photo", -6.0, 3, 6, 3),
        )
        metrics = aggregate_conditional_perplexity(scores)
        overall = metrics["overall"]
        self.assertAlmostEqual(overall["perplexity_per_word"], math.exp(2.0))
        self.assertAlmostEqual(overall["perplexity_per_byte"], math.exp(0.8))
        self.assertAlmostEqual(overall["bits_per_byte"], 0.8 / math.log(2))
        self.assertAlmostEqual(overall["nats_per_token"], 1.6)
        self.assertEqual(set(metrics["by_image_type"]), {"chart", "photo"})

    def test_rejects_missing_description_logprob(self) -> None:
        with self.assertRaisesRegex(ConditionalPerplexityResultError, "missing"):
            score_description(
                uid="id",
                category="chart",
                description="x",
                prompt_tokens=["p", "x"],
                prompt_logprobs=[-0.2, None],
            )


if __name__ == "__main__":
    unittest.main()
