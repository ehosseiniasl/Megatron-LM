from __future__ import annotations

import base64
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

# ML4 Lobsternizer pattern. Unlike V2, it groups sequential digits into chunks of
# up to three from the right: "1998" -> ["1", "998"].
_LOBSTENIZER_V4_REGEX = (
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?"
    "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|"
    "\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))|\\p{N}{1,2}|"
    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)
_TIKTOKEN_PATTERNS_BY_FILENAME = {
    "n4-v6-262k.json": _LOBSTENIZER_V4_REGEX,
}
DEFAULT_TIKTOKEN_NUM_SPECIAL_TOKENS = 1000
TIKTOKEN_VOCAB_ENTRY_KEYS = {"rank", "token_bytes", "token_str"}
NEMOTRON4_SPECIAL_TOKENS = [
    "<unk>",
    "<s>",
    "</s>",
    "<pad>",
    "<|reserved_special_token_4|>",
    "<|reserved_special_token_5|>",
    "<SPECIAL_6>",
    "<SPECIAL_7>",
    "<SPECIAL_8>",
    "<SPECIAL_9>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<img>",
    "</img>",
    "<image_break>",
    "<box>",
    "</box>",
    "<ref>",
    "</ref>",
    "<quad>",
    "</quad>",
    "<so_embedding>",
    "<so_start>",
    "<so_end>",
]
TIKTOKEN_SPECIAL_TOKEN_TEMPLATE = "<SPECIAL_{id}>"


class PretokenizationTokenizerType(StrEnum):
    HUGGINGFACE = "huggingface"
    TIKTOKEN_JSON = "tiktoken_json"


def _get_auto_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer


class HuggingFacePretokenizationTokenizer:
    def __init__(self, tokenizer_path: str | Path) -> None:
        AutoTokenizer = _get_auto_tokenizer()
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self.bos_id = self._tokenizer.bos_token_id
        self.eos_id = self._tokenizer.eos_token_id
        if self.bos_id is None or self.eos_id is None:
            raise ValueError(f"Tokenizer must define BOS/EOS ids: {tokenizer_path}")

    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def token_piece(self, token_id: int) -> str:
        token = self._tokenizer.convert_ids_to_tokens(token_id)
        if token is None:
            raise ValueError(f"Tokenizer does not define token id {token_id}")
        return str(token)


class TikTokenJsonPretokenizationTokenizer:
    def __init__(self, tokenizer_path: str | Path) -> None:
        tokenizer_path = Path(tokenizer_path)
        mergeable_ranks = _tiktoken_json_mergeable_ranks(
            tokenizer_path, num_special_tokens=DEFAULT_TIKTOKEN_NUM_SPECIAL_TOKENS
        )
        if mergeable_ranks is None:
            raise ValueError(f"Unsupported tiktoken tokenizer JSON: {tokenizer_path}")
        self._special_tokens = _tiktoken_special_tokens(
            DEFAULT_TIKTOKEN_NUM_SPECIAL_TOKENS
        )
        self._encoding = _build_tiktoken_encoding(
            name=tokenizer_path.parent.name,
            pattern=_tiktoken_pattern(tokenizer_path),
            mergeable_ranks=mergeable_ranks,
            special_tokens={token: i for i, token in enumerate(self._special_tokens)},
        )
        self.bos_id = self._special_tokens.index("<s>")
        self.eos_id = self._special_tokens.index("</s>")

    def tokenize(self, text: str) -> list[int]:
        return self._encoding.encode(
            text,
            allowed_special=set(),
            disallowed_special=(),
        )

    def token_piece(self, token_id: int) -> str:
        if token_id < len(self._special_tokens):
            return self._special_tokens[token_id]
        token_bytes = self._encoding.decode_single_token_bytes(token_id)
        return token_bytes.decode("utf-8", errors="backslashreplace")


def _tiktoken_pattern(tokenizer_path: Path) -> str:
    return _TIKTOKEN_PATTERNS_BY_FILENAME[tokenizer_path.name]


def _tiktoken_special_tokens(num_special_tokens: int) -> list[str]:
    return NEMOTRON4_SPECIAL_TOKENS + [
        TIKTOKEN_SPECIAL_TOKEN_TEMPLATE.format(id=i)
        for i in range(len(NEMOTRON4_SPECIAL_TOKENS), num_special_tokens)
    ]


def _build_tiktoken_encoding(
    *,
    name: str,
    pattern: str,
    mergeable_ranks: dict[bytes, int],
    special_tokens: dict[str, int],
) -> Any:
    import tiktoken

    return tiktoken.Encoding(
        name=name,
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )


def _json_file_tokenizer_path(tokenizer_path: str | Path) -> Path | None:
    path = Path(tokenizer_path)
    if path.suffix != ".json" or not path.is_file():
        return None
    return path


def _load_json_tokenizer(tokenizer_path: Path) -> Any:
    try:
        with tokenizer_path.open(encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid tokenizer JSON: {tokenizer_path}") from exc


def _tiktoken_json_vocab_size(tokenizer_path: str | Path) -> int | None:
    path = _json_file_tokenizer_path(tokenizer_path)
    if path is None:
        return None
    vocab = _load_json_tokenizer(path)
    if not isinstance(vocab, list):
        return None
    if not vocab:
        raise ValueError(f"Tiktoken tokenizer JSON is empty: {path}")
    for entry in vocab:
        if not isinstance(entry, dict) or set(entry) != TIKTOKEN_VOCAB_ENTRY_KEYS:
            return None
    return len(vocab)


def _tiktoken_json_mergeable_ranks(
    tokenizer_path: str | Path, *, num_special_tokens: int
) -> dict[bytes, int] | None:
    path = _json_file_tokenizer_path(tokenizer_path)
    if path is None:
        return None
    vocab = _load_json_tokenizer(path)
    if not isinstance(vocab, list):
        return None
    ranks: dict[bytes, int] = {}
    for i, entry in enumerate(vocab):
        if not isinstance(entry, dict) or set(entry) != TIKTOKEN_VOCAB_ENTRY_KEYS:
            return None
        if entry["rank"] != i:
            raise ValueError(f"Tiktoken ranks must be consecutive in {path}: {entry}")
        merge = base64.b64decode(entry["token_bytes"])
        if i < 256 and merge != bytes([i]):
            raise ValueError(
                f"First 256 tiktoken entries must map to their byte value in {path}"
            )
        ranks[merge] = i + num_special_tokens
    if not ranks:
        raise ValueError(f"Tiktoken tokenizer JSON is empty: {path}")
    return ranks


def detect_pretokenization_tokenizer_type(
    tokenizer_path: str | Path,
) -> PretokenizationTokenizerType:
    json_path = _json_file_tokenizer_path(tokenizer_path)
    if json_path is None:
        return PretokenizationTokenizerType.HUGGINGFACE
    if _tiktoken_json_vocab_size(json_path) is not None:
        return PretokenizationTokenizerType.TIKTOKEN_JSON
    raise ValueError(
        f"Unsupported tokenizer JSON format: {json_path}. Expected a tiktoken "
        "vocab list with rank, token_bytes, and token_str entries, or pass a "
        "Hugging Face tokenizer directory/model."
    )


def build_pretokenization_tokenizer(
    tokenizer_path: str | Path,
) -> HuggingFacePretokenizationTokenizer | TikTokenJsonPretokenizationTokenizer:
    tokenizer_type = detect_pretokenization_tokenizer_type(tokenizer_path)
    if tokenizer_type is PretokenizationTokenizerType.TIKTOKEN_JSON:
        return TikTokenJsonPretokenizationTokenizer(tokenizer_path)
    return HuggingFacePretokenizationTokenizer(tokenizer_path)


def _required_huggingface_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer does not define required token {token!r}")
    return int(token_id)


def _huggingface_special_token_ids(tokenizer_path: str | Path) -> dict[str, int]:
    AutoTokenizer = _get_auto_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = _required_huggingface_token_id(tokenizer, "<pad>")
    if bos_id is None or eos_id is None:
        raise ValueError(f"Tokenizer must define BOS/EOS ids: {tokenizer_path}")
    return {
        "bos_id": int(bos_id),
        "eos_id": int(eos_id),
        "pad_id": int(pad_id),
        "img_id": _required_huggingface_token_id(tokenizer, "[IMG]"),
        "img_break_id": _required_huggingface_token_id(tokenizer, "[IMG_BREAK]"),
        "img_end_id": _required_huggingface_token_id(tokenizer, "[IMG_END]"),
    }


def _tiktoken_special_token_ids() -> dict[str, int]:
    return {
        "bos_id": NEMOTRON4_SPECIAL_TOKENS.index("<s>"),
        "eos_id": NEMOTRON4_SPECIAL_TOKENS.index("</s>"),
        "pad_id": NEMOTRON4_SPECIAL_TOKENS.index("<pad>"),
        "img_id": NEMOTRON4_SPECIAL_TOKENS.index("<img>"),
        "img_break_id": NEMOTRON4_SPECIAL_TOKENS.index("<image_break>"),
        "img_end_id": NEMOTRON4_SPECIAL_TOKENS.index("</img>"),
    }


def special_token_ids_from_tokenizer(tokenizer_path: str | Path) -> dict[str, int]:
    tokenizer_type = detect_pretokenization_tokenizer_type(tokenizer_path)
    if tokenizer_type is PretokenizationTokenizerType.TIKTOKEN_JSON:
        return _tiktoken_special_token_ids()
    return _huggingface_special_token_ids(tokenizer_path)
