"""Small, explicit adaptations for benchmark prompt contracts."""

from __future__ import annotations

import string
from collections.abc import Mapping, Sequence
from typing import Callable


class PromptAdapterError(ValueError):
    """Raised when a configured prompt adapter cannot satisfy its contract."""


PromptItem = dict[str, str]
PromptAdapter = Callable[[Sequence[PromptItem], object], list[PromptItem]]


def _record_mapping(record: object) -> Mapping[str, object]:
    if isinstance(record, Mapping):
        return record
    converter = getattr(record, "to_dict", None)
    if not callable(converter):
        raise PromptAdapterError("prompt-adapter record is not mapping-like")
    converted = converter()
    if not isinstance(converted, Mapping):
        raise PromptAdapterError("prompt-adapter record conversion is not a mapping")
    return converted


def _cvbench_explicit_options(prompt: Sequence[PromptItem], record: object) -> list[PromptItem]:
    """Restore the explicit option format used by the reference CV-Bench runs."""
    # Pandas belongs to the contained client runtime, not the login-node planner.
    import pandas as pd

    row = _record_mapping(record)
    if "question" not in row:
        raise PromptAdapterError("CV-Bench record is missing question")

    options = {
        label: row[label]
        for label in string.ascii_uppercase
        if label in row and not pd.isna(row[label])
    }
    text = f"Question: {row['question']}\n"
    if options:
        text += "Options:\n"
        text += "".join(f"{label}. {value}\n" for label, value in options.items())
        text += "Please select the correct answer from the options above. \n"

    return [
        *(
            {"type": item["type"], "value": item["value"]}
            for item in prompt
            if item["type"] == "image"
        ),
        {"type": "text", "value": text},
    ]


_ADAPTERS: dict[str, PromptAdapter] = {"cvbench_explicit_options": _cvbench_explicit_options}


def supported_prompt_adapters() -> frozenset[str]:
    """Return the immutable set of accepted suite adapter names."""
    return frozenset(_ADAPTERS)


def apply_prompt_adapter(
    name: str | None, prompt: Sequence[PromptItem], record: object
) -> list[PromptItem]:
    """Return a fresh prompt after applying one configured benchmark adapter."""
    copied = [dict(item) for item in prompt]
    if name is None:
        return copied
    try:
        adapter = _ADAPTERS[name]
    except KeyError as error:
        raise PromptAdapterError(f"unsupported prompt adapter: {name}") from error
    return adapter(copied, record)


__all__ = ("PromptAdapterError", "apply_prompt_adapter", "supported_prompt_adapters")
