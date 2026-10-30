"""Parse and normalize the one chat-template policy for generation requests."""

from __future__ import annotations
from pathlib import Path

from .configured_eval_run import EvalConfigError
from .utils import is_safe_relative_path

MINIMAL_CHAT_TEMPLATE = "assets/minimal_chat_template.jinja"
_BUILTIN_TEMPLATE_PATHS = {"minimal": MINIMAL_CHAT_TEMPLATE, "tokenizer": None}


def parse_chat_template_policy(value: object, label: str) -> str:
    """Return one strict built-in policy or nvbench-relative Jinja path."""
    if type(value) is not str or not value or value != value.strip():
        raise EvalConfigError(f"{label} must be a nonempty stripped string")
    if value not in _BUILTIN_TEMPLATE_PATHS:
        path = Path(value)
        if not is_safe_relative_path(value) or path.suffix != ".jinja":
            raise EvalConfigError(
                f"{label} must be minimal, tokenizer, or a safe nvbench-relative .jinja path"
            )
    return value


def configured_chat_template_path(policy: str) -> str | None:
    """Normalize ``policy`` to an nvbench-relative template path or ``None``."""
    validated = parse_chat_template_policy(policy, "chat-template policy")
    return _BUILTIN_TEMPLATE_PATHS.get(validated, validated)


__all__ = ("MINIMAL_CHAT_TEMPLATE", "configured_chat_template_path", "parse_chat_template_policy")
