"""Secure runtime loading for OpenAI-compatible VLMEvalKit judges."""

from __future__ import annotations

import logging
import os
import re
import stat
from importlib import import_module
from collections.abc import Callable, Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .utils import StrictJsonError, load_json_bytes


JUDGE_CREDENTIALS_FILE_ENV = "NVBENCH_JUDGE_CREDENTIALS_FILE"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_API_BASE_ENV = "OPENAI_API_BASE"
_CREDENTIAL_FIELDS = frozenset({OPENAI_API_KEY_ENV, OPENAI_API_BASE_ENV})
_MAXIMUM_CREDENTIAL_BYTES = 16_384
_UNSAFE_ARTIFACT_CHARACTERS = re.compile(r"[^a-z0-9_-]+")


class JudgeRuntimeError(RuntimeError):
    """Raised when an external judge cannot be configured securely."""


class _OpenAIJudgeAlias:
    """Expose a path-safe model label while delegating requests unchanged."""

    def __init__(self, delegate: object, artifact_model: str) -> None:
        self._delegate = delegate
        self.model = artifact_model

    def __getattr__(self, name: str) -> Any:
        delegate = object.__getattribute__(self, "_delegate")
        return getattr(delegate, name)


def default_judge_credentials_path(environment: MutableMapping[str, str]) -> Path:
    """Return the configured credential file or the user's nvbench default."""
    configured = environment.get(JUDGE_CREDENTIALS_FILE_ENV)
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            raise JudgeRuntimeError(f"{JUDGE_CREDENTIALS_FILE_ENV} must be absolute")
        return path
    home = environment.get("HOME")
    if not home or not Path(home).is_absolute():
        raise JudgeRuntimeError("HOME must be an absolute path")
    return Path(home) / ".config" / "nvbench" / "judge_credentials.json"


def _credential_value(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise JudgeRuntimeError(f"{field} must be a nonempty single-line string")
    return value


def normalize_openai_api_base(value: str) -> str:
    """Return a direct OpenAI-compatible chat-completions endpoint.

    OpenAI SDK-style base URLs such as ``https://host/v1/`` are accepted and
    expanded because VLMEvalKit's requests adapter posts to the configured URL
    directly. An already complete ``.../chat/completions`` URL is preserved.
    """
    endpoint = _credential_value(value, OPENAI_API_BASE_ENV)
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise JudgeRuntimeError(
            f"{OPENAI_API_BASE_ENV} must be an absolute HTTP(S) URL "
            "without credentials, query, or fragment"
        )
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path = f"{path}/chat/completions"
    return parsed._replace(path=path).geturl()


def judge_artifact_label(judge: str) -> str:
    """Return a readable path-safe label derived from a provider model ID."""
    model = _credential_value(judge, "judge")
    label = _UNSAFE_ARTIFACT_CHARACTERS.sub("_", model.lower()).strip("_-")
    if not label:
        raise JudgeRuntimeError("judge does not contain a path-safe model label")
    return label


@contextmanager
def _provider_model_alias(
    provider_model: str, artifact_model: str
) -> Iterator[None]:
    """Keep upstream filenames safe while sending the exact provider model ID."""
    import vlmeval.api

    original = vlmeval.api.OpenAIWrapper

    def build_aliased_wrapper(
        _configured_model: str, *args: object, **kwargs: object
    ) -> _OpenAIJudgeAlias:
        delegate = original(provider_model, *args, **kwargs)
        return _OpenAIJudgeAlias(delegate, artifact_model)

    vlmeval.api.OpenAIWrapper = build_aliased_wrapper
    try:
        yield
    finally:
        vlmeval.api.OpenAIWrapper = original


@contextmanager
def _worldvqa_partial_cache_resume(
    dataset: str | None,
) -> Iterator[None]:
    """Align WorldVQA's task list with its missing cache keys on resume.

    The vendored evaluator filters ``keys`` after loading a partial judge cache
    but accidentally retains the full task list. Its shared progress helper
    requires equal lengths. Keep the compatibility patch local to a bound
    WorldVQA evaluator and preserve the vendored source unchanged.
    """
    if dataset != "WorldVQA":
        yield
        return

    worldvqa = import_module("vlmeval.dataset.worldvqa")
    original = worldvqa.utils.track_progress_rich

    def track_progress_rich(
        func: Callable[..., object],
        tasks: object = tuple(),
        nproc: int | None = 1,
        save: str | None = None,
        keys: object = None,
        use_process: bool = False,
        **kwargs: object,
    ) -> object:
        task_list = list(tasks)
        if keys is not None and len(keys) != len(task_list):
            key_list = list(keys)
            if any(
                type(index) is not int or index < 0 or index >= len(task_list)
                for index in key_list
            ):
                raise JudgeRuntimeError("WorldVQA resume keys are invalid")
            task_list = [task_list[index] for index in key_list]
        return original(
            func,
            task_list,
            nproc=nproc,
            save=save,
            keys=keys,
            use_process=use_process,
            **kwargs,
        )

    worldvqa.utils.track_progress_rich = track_progress_rich
    try:
        yield
    finally:
        worldvqa.utils.track_progress_rich = original


def evaluate_with_judge_model(
    evaluator: Callable[..., object],
    prediction: str,
    judge: str,
    *,
    dataset: str | None = None,
) -> str:
    """Evaluate with ``judge`` and return its filesystem-safe artifact label.

    Native VLMEvalKit evaluators use the model string in output filenames. For
    provider IDs containing path separators, they receive a derived label while
    their OpenAI-compatible requests retain the exact configured model ID.
    """
    artifact_model = judge_artifact_label(judge)
    with _worldvqa_partial_cache_resume(dataset):
        if artifact_model == judge:
            evaluator(prediction, model=judge)
        else:
            with _provider_model_alias(judge, artifact_model):
                evaluator(prediction, model=artifact_model)
    return artifact_model


def _load_credentials_file(path: Path) -> dict[str, str]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise JudgeRuntimeError("judge credential file is missing") from error
    except OSError as error:
        raise JudgeRuntimeError("judge credential file is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise JudgeRuntimeError(
            "judge credential file must be a regular non-symlink file"
        )
    if metadata.st_uid != os.geteuid():
        raise JudgeRuntimeError(
            "judge credential file must be owned by the current user"
        )
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise JudgeRuntimeError("judge credential file mode must be 0600")
    try:
        payload = load_json_bytes(
            path.read_bytes(), maximum_bytes=_MAXIMUM_CREDENTIAL_BYTES
        )
    except (OSError, StrictJsonError) as error:
        raise JudgeRuntimeError(
            "judge credential file is not valid strict JSON"
        ) from error
    if type(payload) is not dict or set(payload) != _CREDENTIAL_FIELDS:
        raise JudgeRuntimeError(
            "judge credential file must contain exactly OPENAI_API_KEY and OPENAI_API_BASE"
        )
    key = _credential_value(payload[OPENAI_API_KEY_ENV], OPENAI_API_KEY_ENV)
    endpoint = normalize_openai_api_base(payload[OPENAI_API_BASE_ENV])
    return {OPENAI_API_KEY_ENV: key, OPENAI_API_BASE_ENV: endpoint}


def _bind_credentials(
    environment: MutableMapping[str, str], credentials: dict[str, str]
) -> None:
    for name, value in credentials.items():
        existing = environment.get(name)
        if existing and existing != value:
            raise JudgeRuntimeError(
                f"{name} conflicts with {JUDGE_CREDENTIALS_FILE_ENV}"
            )
        environment[name] = value


def prepare_judge_runtime(
    judge: str,
    environment: MutableMapping[str, str] | None = None,
) -> Path | None:
    """Load and validate credentials for one configured judge.

    Exact matching needs no external runtime. Judge-backed evaluation first
    uses a configured/default credential file when it exists, then falls back
    to directly injected ``OPENAI_API_*`` values for CI and external systems.
    The secret itself is never returned or logged.
    """
    if type(judge) is not str or not judge or judge != judge.strip():
        raise JudgeRuntimeError("judge must be a nonempty stripped string")
    if judge == "exact_matching":
        return None
    target = os.environ if environment is None else environment
    path = default_judge_credentials_path(target)
    explicit_file = bool(target.get(JUDGE_CREDENTIALS_FILE_ENV))
    if explicit_file or path.exists():
        _bind_credentials(target, _load_credentials_file(path))
    key = target.get(OPENAI_API_KEY_ENV)
    if not key:
        raise JudgeRuntimeError(
            "judge requires OPENAI_API_KEY or a secure nvbench credential file"
        )
    _credential_value(key, OPENAI_API_KEY_ENV)
    endpoint = target.get(OPENAI_API_BASE_ENV)
    if endpoint:
        target[OPENAI_API_BASE_ENV] = normalize_openai_api_base(endpoint)

    # The pinned VLMEvalKit OpenAI adapter logs its key at INFO during judge
    # construction. Suppress that module's INFO records in this client process;
    # forked scoring workers inherit the same logger level.
    logging.getLogger("vlmeval.api.gpt").setLevel(logging.WARNING)
    return path if explicit_file or path.exists() else None


__all__ = (
    "JUDGE_CREDENTIALS_FILE_ENV",
    "JudgeRuntimeError",
    "default_judge_credentials_path",
    "evaluate_with_judge_model",
    "judge_artifact_label",
    "prepare_judge_runtime",
    "normalize_openai_api_base",
)
