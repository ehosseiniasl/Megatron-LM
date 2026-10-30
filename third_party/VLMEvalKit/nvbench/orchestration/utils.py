"""Small, domain-neutral mechanics for owned evaluation artifacts."""

from __future__ import annotations
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import TextIO, TypeAlias

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SAFE_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
SLURM_JOB_ID_PATTERN = re.compile(r"[1-9][0-9]*")
VLMEVAL_FAILURE_SENTINEL = "Failed to obtain answer"
SPECIAL_TOKEN_NAMES = frozenset(
    {"unknown", "bos", "eos", "image", "pad", "image_break", "image_end"}
)
VLMEvalRowIndex: TypeAlias = int | str


class StrictJsonError(ValueError):
    """Raised when JSON bytes or values violate the strict JSON syntax contract."""


class JsonArtifactError(RuntimeError):
    """Raised when a JSON artifact cannot be safely read or durably written."""


class VLMEvalRowIdentityError(ValueError):
    """Raised when a dataset or artifact row index is not a canonical scalar."""


def require_vlmeval_row_index(value: object, *, label: str = "row index") -> VLMEvalRowIndex:
    """Return ``value`` unchanged when it is a canonical VLMEvalKit row index.

    Canonical indices are exact, non-boolean integers or nonempty stripped
    strings. The function has no side effects; invalid values raise
    :class:`VLMEvalRowIdentityError`.
    """
    if type(value) is int:
        return value
    if type(value) is str and value and value == value.strip():
        return value
    raise VLMEvalRowIdentityError(f"{label} must be a non-bool integer or nonempty stripped string")


def require_unique_vlmeval_row_indices(
    values: tuple[object, ...], *, label: str
) -> tuple[VLMEvalRowIndex, ...]:
    """Validate canonical row indices with unique textual identities.

    Integer ``1`` and string ``"1"`` conflict because demonstration and JSON
    identities use their text form. Inputs are not mutated; invalid values
    raise :class:`VLMEvalRowIdentityError`.
    """
    indices = tuple(
        require_vlmeval_row_index(value, label=f"{label} {position}")
        for position, value in enumerate(values)
    )
    text_identities = tuple(str(value) for value in indices)
    if len(text_identities) != len(set(text_identities)):
        raise VLMEvalRowIdentityError(f"{label} values must be unique")
    return indices


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise StrictJsonError(f"nonstandard JSON constant: {value}")


def load_json_bytes(raw: bytes, *, maximum_bytes: int) -> object:
    """Decode bounded UTF-8 JSON without mutating ``raw``.

    ``maximum_bytes`` must be a positive exact integer. Empty, oversized,
    malformed, duplicate-key, or nonfinite-constant inputs raise
    :class:`StrictJsonError`.
    """
    if type(raw) is not bytes:
        raise StrictJsonError("JSON input must be bytes")
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise StrictJsonError("maximum JSON bytes must be positive")
    if not raw or len(raw) > maximum_bytes:
        raise StrictJsonError("JSON input has an invalid byte size")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_invalid_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise StrictJsonError("input is not strict UTF-8 JSON") from error


def dump_json(payload: object, stream: TextIO, *, ensure_ascii: bool) -> None:
    """Write deterministic finite JSON and one newline without mutating inputs.

    The caller owns the stream lifetime and durability. ``ensure_ascii`` must
    be an exact boolean; unsupported values or nonfinite numbers raise the
    corresponding :mod:`json` ``TypeError`` or ``ValueError``.
    """
    if type(ensure_ascii) is not bool:
        raise TypeError("ensure_ascii must be a boolean")
    json.dump(payload, stream, ensure_ascii=ensure_ascii, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")


def resolve_directory(path: Path) -> Path:
    """Return the resolved existing directory for ``path`` without mutation.

    A non-:class:`Path`, unavailable path, or non-directory raises
    ``ValueError``. A symlink is resolved for callers that only need a safe
    directory root; writers separately require a real parent path.
    """
    if not isinstance(path, Path):
        raise ValueError("directory must be a Path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("directory is unavailable") from error
    if not resolved.is_dir():
        raise ValueError("path must be a directory")
    return resolved


def require_shared_file(path: Path, root: Path, label: str) -> Path:
    """Return a contained readable group-readable regular file without mutation.

    ``path`` and ``root`` must identify an existing artifact and directory.
    Paths resolving outside ``root``, inaccessible files, and invalid inputs
    raise ``ValueError`` with ``label`` identifying the failed contract.
    """
    if not isinstance(path, Path):
        raise ValueError(f"{label} path must be a Path")
    try:
        root = resolve_directory(root)
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        mode = resolved.stat().st_mode
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is unavailable or outside the run") from error
    if not stat.S_ISREG(mode) or not os.access(resolved, os.R_OK):
        raise ValueError(f"{label} must be a readable regular file")
    if not mode & stat.S_IRGRP:
        raise ValueError(f"{label} must be group-readable")
    return resolved


def is_safe_relative_path(value: object) -> bool:
    """Return whether ``value`` is a normalized, nonempty relative POSIX path."""
    if type(value) is not str or not value or value != value.strip() or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def relative_path(path: Path | None, root: Path) -> str | None:
    """Return ``path`` relative to resolved ``root`` without mutating either path.

    ``None`` remains ``None``. Missing paths or paths outside ``root`` raise
    ``ValueError``; callers that require a regular file should first use
    :func:`require_shared_file`.
    """
    if path is None:
        return None
    if not isinstance(path, Path):
        raise ValueError("artifact path must be a Path or None")
    try:
        return path.resolve(strict=True).relative_to(resolve_directory(root)).as_posix()
    except (OSError, ValueError) as error:
        raise ValueError("artifact path is unavailable or outside the run") from error


def shared_file_with_relative_path(path: Path, root: Path, label: str) -> tuple[Path, str]:
    """Return a validated shared file and its normalized path below ``root``.

    Neither input is mutated. The file must satisfy :func:`require_shared_file`;
    failures raise ``ValueError``. The returned relative path is always a
    nonempty POSIX string because ``path`` must be a file beneath ``root``.
    """
    resolved = require_shared_file(path, root, label)
    relative = relative_path(resolved, root)
    if relative is None:  # Defensive: ``resolved`` cannot be ``None``.
        raise ValueError(f"{label} has no relative path")
    return resolved, relative


def sanitized_error_text(error: Exception, root: Path, *, maximum_length: int = 500) -> str:
    """Return one bounded whitespace-normalized error message without mutating ``error``.

    The supplied run-directory spelling is replaced with ``<run_directory>``.
    ``maximum_length`` must be a positive exact integer; invalid arguments
    raise ``ValueError``.
    """
    if not isinstance(error, Exception):
        raise ValueError("error must be an Exception")
    if type(maximum_length) is not int or maximum_length <= 0:
        raise ValueError("maximum error text length must be positive")
    message = re.sub(r"\s+", " ", str(error)).strip()
    return message.replace(str(root), "<run_directory>")[:maximum_length]


@contextmanager
def temporary_environment(name: str, value: str | None) -> Iterator[None]:
    """Temporarily bind one environment variable and restore its exact prior state.

    A nonempty variable name without ``=`` and a string-or-``None`` value are
    required. The process environment is mutated only for the context duration;
    invalid input raises ``ValueError``.
    """
    if type(name) is not str or not name or "=" in name or "\x00" in name:
        raise ValueError("environment variable name is invalid")
    if value is not None and type(value) is not str:
        raise ValueError("environment variable value must be a string or None")
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _write_json_temporary(destination: Path, payload: object, ensure_ascii: bool) -> Path:
    parent = resolve_directory(destination.parent)
    if parent != destination.parent:
        raise JsonArtifactError("artifact parent must be a real directory")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o640)
            dump_json(payload, stream, ensure_ascii=ensure_ascii)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except Exception as error:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise JsonArtifactError(f"cannot serialize artifact: {destination.name}") from error


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(destination: Path, payload: object, *, ensure_ascii: bool, replace: bool) -> Path:
    if not isinstance(destination, Path):
        raise JsonArtifactError("artifact destination must be a Path")
    temporary: Path | None = None
    try:
        temporary = _write_json_temporary(destination, payload, ensure_ascii)
        if replace:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
        temporary = None
        _sync_directory(destination.parent)
    except Exception as error:
        raise JsonArtifactError(f"cannot write artifact: {destination.name}") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return destination


def write_json(destination: Path, payload: object) -> Path:
    """Durably replace ``destination`` with strict UTF-8 mode-0640 JSON.

    The parent must be an existing real directory. ``payload`` is not mutated;
    successful writes atomically replace the destination and sync the parent
    directory. Serialization or filesystem failures raise :class:`JsonArtifactError`.
    """
    return _write_json(destination, payload, ensure_ascii=False, replace=True)


def write_json_once(destination: Path, payload: object) -> Path:
    """Durably create strict ASCII mode-0640 JSON only when ``destination`` is absent.

    The parent must be an existing real directory. ``payload`` is not mutated;
    a successful return is the newly linked destination. Existing destinations,
    serialization, and filesystem failures raise :class:`JsonArtifactError`.
    """
    return _write_json(destination, payload, ensure_ascii=True, replace=False)


def append_jsonl_record(destination: Path, payload: object) -> Path:
    """Append and durably flush one finite JSON record to a mode-0640 JSONL file."""
    if not isinstance(destination, Path):
        raise JsonArtifactError("JSONL destination must be a Path")
    parent = resolve_directory(destination.parent)
    if parent != destination.parent:
        raise JsonArtifactError("JSONL parent must be a real directory")
    try:
        new_file = not destination.exists()
        with destination.open("a", encoding="utf-8") as stream:
            if new_file:
                os.fchmod(stream.fileno(), 0o640)
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception as error:
        raise JsonArtifactError(f"cannot append JSONL artifact: {destination.name}") from error
    return destination


def write_jsonl(destination: Path, records: tuple[object, ...]) -> Path:
    """Durably replace a mode-0640 JSONL artifact with finite records."""
    if not isinstance(destination, Path) or type(records) is not tuple:
        raise JsonArtifactError("JSONL destination or records are invalid")
    parent = resolve_directory(destination.parent)
    if parent != destination.parent:
        raise JsonArtifactError("JSONL parent must be a real directory")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o640)
            for record in records:
                json.dump(record, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        _sync_directory(parent)
    except Exception as error:
        raise JsonArtifactError(f"cannot write JSONL artifact: {destination.name}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def read_exact_json(
    path: Path, expected_keys: set[str], *, maximum_bytes: int = 65_536
) -> dict[str, object]:
    """Read one real group-readable JSON object with exactly ``expected_keys``.

    The file is not mutated. Symlinks, non-regular or private files, malformed
    JSON, an oversized payload, and nonmatching object keys raise
    :class:`JsonArtifactError`.
    """
    if not isinstance(path, Path) or type(expected_keys) is not set:
        raise JsonArtifactError("artifact path or expected keys are invalid")
    try:
        resolved = path.resolve(strict=True)
        mode = path.stat().st_mode
        payload = load_json_bytes(path.read_bytes(), maximum_bytes=maximum_bytes)
    except (OSError, RuntimeError, StrictJsonError) as error:
        raise JsonArtifactError(f"artifact is unavailable or malformed: {path.name}") from error
    if (
        resolved != path
        or not stat.S_ISREG(mode)
        or not mode & stat.S_IRGRP
        or type(payload) is not dict
        or set(payload) != expected_keys
    ):
        raise JsonArtifactError(f"artifact violates its exact contract: {path.name}")
    return payload
