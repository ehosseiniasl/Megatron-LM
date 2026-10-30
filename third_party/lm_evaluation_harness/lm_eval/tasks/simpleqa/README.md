# SimpleQA

This task evaluates short-form factual question answering on the Hugging Face
mirror of OpenAI SimpleQA, `basicv8vc/SimpleQA`.

`simpleqa_5shot` is intended for pretrained/base model evaluation. It uses five
fixed, hand-written Q/A examples and asks the model to produce a short answer.

The scorer is deterministic and local: it normalizes answers, detects common
non-attempts, and computes `correct`, `incorrect`, `not_attempted`, and the
SimpleQA-style F1 over those labels. This is not the official judge-based
SimpleQA score. It will be stricter on paraphrases and synonyms, and it cannot
reliably detect every contradictory extra claim in a longer answer. A separate
judge-graded variant can be added later using the logged samples.
