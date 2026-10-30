"""Safe YAML primitives with strict schemas and conflict-free composition."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from .configured_eval_run import ConfigScalar, EvalConfigError


_PLACEHOLDER = re.compile(r"\$\{([^{}]*)\}")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate YAML key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def expect_mapping(
    value: Any, required: set[str], optional: set[str], label: str
) -> dict[str, Any]:
    """Require a string-keyed mapping with exactly the allowed field set."""
    if not isinstance(value, dict) or not all(type(key) is str for key in value):
        raise EvalConfigError(f"{label} must be a string-keyed mapping")
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise EvalConfigError(f"{label} is missing: {', '.join(missing)}")
    if unknown:
        raise EvalConfigError(f"{label} has unknown fields: {', '.join(unknown)}")
    return value


def expect_string_mapping(value: Any, label: str) -> dict[str, Any]:
    """Return a string-keyed mapping or raise a labeled configuration error."""
    if not isinstance(value, dict) or not all(type(key) is str for key in value):
        raise EvalConfigError(f"{label} must be a string-keyed mapping")
    return value


def expect_type(value: Any, expected: type, label: str) -> Any:
    """Require an exact scalar/container type and return the unchanged value."""
    if type(value) is not expected:
        raise EvalConfigError(f"{label} must be {expected.__name__}")
    return value


def expect_optional_string(value: Any, label: str) -> str | None:
    """Accept only a string or ``None`` and return it unchanged."""
    if value is not None and type(value) is not str:
        raise EvalConfigError(f"{label} must be a string or null")
    return value


def expect_number(value: Any, label: str) -> int | float:
    """Return a finite exact integer or float, excluding booleans and non-numbers."""
    if type(value) not in (int, float):
        raise EvalConfigError(f"{label} must be an integer or float")
    if not math.isfinite(value):
        raise EvalConfigError(f"{label} must be finite")
    return value


def expect_string_tuple(value: Any, label: str) -> tuple[str, ...]:
    """Convert a duplicate-free YAML string list into an immutable tuple."""
    if not isinstance(value, list) or not all(type(item) is str for item in value):
        raise EvalConfigError(f"{label} must be a list of strings")
    if len(value) != len(set(value)):
        raise EvalConfigError(f"{label} contains duplicates")
    return tuple(value)


def expect_scalar(value: Any, label: str) -> ConfigScalar:
    """Return an allowed finite configuration scalar with its exact Python type."""
    if type(value) not in (str, int, float, bool):
        raise EvalConfigError(f"{label} must be a string, integer, float, or boolean")
    if type(value) is float and not math.isfinite(value):
        raise EvalConfigError(f"{label} must be finite")
    return value


def resolve_config_path(
    parent: Path,
    root: Path,
    value: object,
    label: str,
) -> Path:
    """Resolve one relative config reference beneath ``root``.

    ``parent`` is the directory containing the referring file. The function
    reads no file content. Missing, absolute, and escaping paths raise
    :class:`EvalConfigError`.
    """
    if type(value) is not str:
        raise EvalConfigError(f"{label} reference must be a string")
    try:
        boundary = root.resolve(strict=True)
    except OSError as error:
        raise EvalConfigError(f"cannot resolve configuration root: {root}") from error
    candidate = Path(value)
    if candidate.is_absolute() or not candidate.parts:
        raise EvalConfigError(f"{label} reference must be a safe relative path")
    try:
        path = (parent / candidate).resolve(strict=True)
        path.relative_to(boundary)
    except (OSError, RuntimeError) as error:
        raise EvalConfigError(f"cannot resolve {label} {value}: {error}") from error
    except ValueError as error:
        raise EvalConfigError(f"{label} reference escapes the configuration root") from error
    return path


def _check_placeholders(value: Any, label: str) -> None:
    if isinstance(value, str):
        placeholders = _PLACEHOLDER.findall(value)
        unsupported = sorted(set(placeholders) - {"workspace"})
        if unsupported:
            raise EvalConfigError(f"{label} uses unsupported placeholders: {unsupported}")
        if "$" in _PLACEHOLDER.sub("", value):
            raise EvalConfigError(f"{label} contains a malformed or unsupported placeholder")
    elif isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str):
                _check_placeholders(key, f"{label}.<key>")
            _check_placeholders(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _check_placeholders(nested, f"{label}[{index}]")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as error:
        raise EvalConfigError(f"cannot safely load YAML {path}: {error}") from error
    return expect_string_mapping(data, str(path))


def load_yaml_data_bytes(raw_bytes: bytes, label: str) -> dict[str, Any]:
    """Decode duplicate-free safe YAML bytes whose strings are inert data.

    ``raw_bytes`` and ``label`` are not mutated. The returned mapping is newly
    constructed by the safe loader. Invalid UTF-8, unsafe YAML, duplicate keys,
    and non-mapping roots raise :class:`EvalConfigError` with the supplied
    label.
    """
    if type(raw_bytes) is not bytes:
        raise EvalConfigError(f"{label} content must be bytes")
    if type(label) is not str or not label.strip():
        raise EvalConfigError("YAML data label must be a nonempty string")
    try:
        text = raw_bytes.decode("utf-8")
        data = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise EvalConfigError(f"cannot safely load {label}: {error}") from error
    return expect_string_mapping(data, label)


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load one duplicate-free configuration YAML mapping from ``path``.

    The source is read without mutation. Safe ``${workspace}`` placeholders
    remain unresolved for the path-resolution phase; every other placeholder
    form is rejected. I/O, YAML, mapping, or placeholder failures raise
    :class:`EvalConfigError`.
    """
    result = _load_yaml(path)
    _check_placeholders(result, str(path))
    return result


def load_yaml_data_file(path: Path) -> dict[str, Any]:
    """Load duplicate-free safe YAML data from ``path`` without substitutions.

    Scalar strings remain inert and the source file is not modified. I/O, YAML,
    or non-mapping-root failures raise :class:`EvalConfigError`.
    """
    return _load_yaml(path)
