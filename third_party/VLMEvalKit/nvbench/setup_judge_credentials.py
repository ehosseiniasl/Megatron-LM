"""Interactively install and verify OpenAI-compatible judge credentials."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import stat
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from nvbench.orchestration.judge_runtime import (
    OPENAI_API_BASE_ENV,
    OPENAI_API_KEY_ENV,
    default_judge_credentials_path,
    normalize_openai_api_base,
)


class JudgeCredentialSetupError(RuntimeError):
    """Raised when credential installation or its connectivity probe fails."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise JudgeCredentialSetupError(
            "judge endpoint redirected the authenticated request"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        help="Credential file (default: ~/.config/nvbench/judge_credentials.json)",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def _probe(endpoint: str, key: str, model: str, timeout: float) -> None:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you help me?"},
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False,
        },
        allow_nan=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "OpenAI/Python nvbench-judge-setup",
        },
        method="POST",
    )
    opener = urllib.request.build_opener(_NoRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(1_048_577)
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            raw_error = error.read(16_385)
            if len(raw_error) <= 16_384:
                decoded = raw_error.decode("utf-8", errors="replace").replace(
                    key, "<redacted>"
                )
                parsed = json.loads(decoded)
                if type(parsed) is dict:
                    candidate = parsed.get("detail") or parsed.get("message")
                    nested = parsed.get("error")
                    if not candidate and type(nested) is dict:
                        candidate = nested.get("message")
                    if type(candidate) is str:
                        detail = " ".join(candidate.split())[:500]
        except (OSError, TypeError, ValueError):
            pass
        suffix = f": {detail}" if detail else ""
        raise JudgeCredentialSetupError(
            f"judge endpoint returned HTTP {error.code}{suffix}"
        ) from error
    except (OSError, urllib.error.URLError) as error:
        raise JudgeCredentialSetupError(
            f"judge connectivity test failed ({type(error).__name__})"
        ) from error
    if len(raw) > 1_048_576:
        raise JudgeCredentialSetupError("judge connectivity response is too large")
    try:
        result = json.loads(raw.decode("utf-8"))
        choices = result["choices"]
        message = choices[0]["message"]
        content = message["content"]
    except (
        KeyError,
        IndexError,
        TypeError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise JudgeCredentialSetupError(
            "judge endpoint did not return an OpenAI-compatible chat completion"
        ) from error
    if type(content) is not str:
        raise JudgeCredentialSetupError("judge response content is not text")


def _write_credentials(path: Path, endpoint: str, key: str) -> None:
    if not path.is_absolute():
        raise JudgeCredentialSetupError("credential output path must be absolute")
    parent = path.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if parent.is_symlink() or parent.stat().st_uid != os.geteuid():
        raise JudgeCredentialSetupError(
            "credential directory must be owned and not a symlink"
        )
    os.chmod(parent, 0o700)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise JudgeCredentialSetupError(
            "credential destination must be a regular non-symlink file"
        )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            json.dump(
                {OPENAI_API_BASE_ENV: endpoint, OPENAI_API_KEY_ENV: key},
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise JudgeCredentialSetupError(
            "credential file could not be installed"
        ) from error


def main(argv: list[str] | None = None) -> int:
    """Prompt, test, and atomically install one judge credential file."""
    args = _parser().parse_args(argv)
    if args.timeout <= 0:
        raise JudgeCredentialSetupError("timeout must be positive")
    default_path = default_judge_credentials_path(os.environ)
    path = Path(args.output).expanduser() if args.output else default_path
    if path.exists():
        replace = (
            input(f"Replace existing credential file {path}? [y/N]: ").strip().lower()
        )
        if replace not in {"y", "yes"}:
            raise JudgeCredentialSetupError("credential installation canceled")
    endpoint_text = input(
        "OpenAI-compatible base URL [https://api.openai.com/v1/]: "
    ).strip()
    endpoint = normalize_openai_api_base(endpoint_text or "https://api.openai.com/v1/")
    model = input("Judge model for connectivity test: ").strip()
    if not model or model != model.strip() or any(ch in model for ch in "\r\n\x00"):
        raise JudgeCredentialSetupError("judge model must be a single-line value")
    key = getpass.getpass("API key (input hidden): ").strip()
    if not key or any(ch in key for ch in "\r\n\x00"):
        raise JudgeCredentialSetupError("API key must be a nonempty single-line value")
    print("Testing endpoint compatibility...", flush=True)
    _probe(endpoint, key, model, args.timeout)
    _write_credentials(path, endpoint, key)
    mode = stat.S_IMODE(path.stat().st_mode)
    if path.stat().st_uid != os.geteuid() or mode != 0o600:
        raise JudgeCredentialSetupError("installed credential permissions are invalid")
    print(f"Judge credentials installed and verified at {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
