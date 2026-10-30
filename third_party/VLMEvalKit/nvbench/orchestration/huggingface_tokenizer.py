"""Load the one resolved Hugging Face tokenizer used at client boundaries."""

from __future__ import annotations


class HuggingFaceTokenizerLoadError(RuntimeError):
    """Raised when the resolved local tokenizer cannot be constructed."""


def load_huggingface_tokenizer(repository: str, revision: str):
    """Load one pinned fast tokenizer without executing repository-supplied code."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            repository,
            revision=revision,
            trust_remote_code=False,
            use_fast=True,
        )
    except Exception as error:
        raise HuggingFaceTokenizerLoadError(
            "cannot load the configured Hugging Face tokenizer"
        ) from error


__all__ = ("HuggingFaceTokenizerLoadError", "load_huggingface_tokenizer")
