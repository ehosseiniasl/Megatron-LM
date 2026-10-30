import base64
import json
from pathlib import Path
from typing import Any

import pytest

from mistral_fork.mistral.data.pretokenized.processing import tokenizers


def _write_tiktoken_vocab(path: Path, n_entries: int = 2) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "rank": rank,
                    "token_bytes": base64.b64encode(bytes([rank])).decode("ascii"),
                    "token_str": "",
                }
                for rank in range(n_entries)
            ]
        ),
        encoding="utf-8",
    )


def _write_digit_merging_tiktoken_vocab(path: Path) -> None:
    merges = [bytes([rank]) for rank in range(256)] + [b"99", b"998"]
    path.write_text(
        json.dumps(
            [
                {
                    "rank": rank,
                    "token_bytes": base64.b64encode(merge).decode("ascii"),
                    "token_str": "",
                }
                for rank, merge in enumerate(merges)
            ]
        ),
        encoding="utf-8",
    )


def test_build_tokenizer_detects_tiktoken_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer_path = tmp_path / "n4-v6-262k.json"
    _write_tiktoken_vocab(tokenizer_path)
    calls: dict[str, Any] = {}

    def fake_build_tiktoken_encoding(
        *,
        name: str,
        pattern: str,
        mergeable_ranks: dict[bytes, int],
        special_tokens: dict[str, int],
    ) -> Any:
        calls.update(
            {
                "name": name,
                "pattern": pattern,
                "mergeable_ranks": mergeable_ranks,
                "special_tokens": special_tokens,
            }
        )

        class FakeTiktokenEncoding:
            def encode(
                self,
                text: str,
                *,
                allowed_special: set[str],
                disallowed_special: tuple[str, ...],
            ) -> list[int]:
                assert allowed_special == set()
                assert disallowed_special == ()
                return [1000 + len(text)]

            def decode_single_token_bytes(self, token_id: int) -> bytes:
                return calls["mergeable_ranks_inverse"][token_id]

        calls["mergeable_ranks_inverse"] = {
            token_id: token_bytes for token_bytes, token_id in mergeable_ranks.items()
        }

        return FakeTiktokenEncoding()

    monkeypatch.setattr(
        tokenizers, "_build_tiktoken_encoding", fake_build_tiktoken_encoding
    )

    tokenizer = tokenizers.build_pretokenization_tokenizer(tokenizer_path)

    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.tokenize("abc") == [1003]
    assert tokenizer.token_piece(1) == "<s>"
    assert tokenizer.token_piece(1000) == "\x00"
    assert calls["name"] == tmp_path.name
    assert calls["pattern"] == tokenizers._LOBSTENIZER_V4_REGEX
    assert calls["mergeable_ranks"] == {b"\x00": 1000, b"\x01": 1001}
    assert calls["special_tokens"]["<unk>"] == 0
    assert calls["special_tokens"]["<s>"] == 1
    assert calls["special_tokens"]["</s>"] == 2
    assert calls["special_tokens"]["<pad>"] == 3
    assert calls["special_tokens"]["<|reserved_special_token_4|>"] == 4
    assert calls["special_tokens"]["<|im_start|>"] == 10
    assert calls["special_tokens"]["<img>"] == 18
    assert calls["special_tokens"]["</img>"] == 19
    assert calls["special_tokens"]["<image_break>"] == 20
    assert calls["special_tokens"]["<so_embedding>"] == 27
    assert calls["special_tokens"]["<SPECIAL_999>"] == 999
    assert len(calls["special_tokens"]) == 1000


def test_n4_v6_tokenizer_uses_lobsternizer_v4_digit_chunking(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "n4-v6-262k.json"
    _write_digit_merging_tiktoken_vocab(tokenizer_path)

    tokenizer = tokenizers.build_pretokenization_tokenizer(tokenizer_path)

    assert tokenizer.tokenize("1998") == [1049, 1257]


@pytest.mark.parametrize("text", ["<s>", "</s>", "<pad>", "<mask>"])
def test_n4_v6_tokenizer_encodes_special_token_spellings_as_text(
    tmp_path: Path, text: str
) -> None:
    tokenizer_path = tmp_path / "n4-v6-262k.json"
    _write_digit_merging_tiktoken_vocab(tokenizer_path)

    tokenizer = tokenizers.build_pretokenization_tokenizer(tokenizer_path)

    assert tokenizer.tokenize(text) == [1000 + byte for byte in text.encode()]


def test_unregistered_tiktoken_json_has_no_pattern(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "other.json"
    _write_tiktoken_vocab(tokenizer_path)

    with pytest.raises(KeyError, match="other.json"):
        tokenizers.build_pretokenization_tokenizer(tokenizer_path)


def test_build_tokenizer_uses_huggingface_for_non_json_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHuggingFaceTokenizer:
        bos_token_id = 11
        eos_token_id = 12

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [len(text)]

        def convert_ids_to_tokens(self, token_id: int) -> str | None:
            return {11: "<s>", 12: "</s>"}.get(token_id)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_path: str) -> FakeHuggingFaceTokenizer:
            assert tokenizer_path == "hf-tokenizer"
            return FakeHuggingFaceTokenizer()

    monkeypatch.setattr(tokenizers, "_get_auto_tokenizer", lambda: FakeAutoTokenizer)

    tokenizer = tokenizers.build_pretokenization_tokenizer("hf-tokenizer")

    assert tokenizer.bos_id == 11
    assert tokenizer.eos_id == 12
    assert tokenizer.tokenize("abcd") == [4]
    assert tokenizer.token_piece(11) == "<s>"


def test_nemotron4_multimodal_special_tokens_are_encoded_atomically(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "n4-v6-262k.json"
    _write_tiktoken_vocab(tokenizer_path, n_entries=256)

    tokenizer = tokenizers.TikTokenJsonPretokenizationTokenizer(tokenizer_path)

    assert tokenizer.tokenize("<img><image_break></img>") == [18, 20, 19]


def test_unsupported_json_tokenizer_format_raises(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported tokenizer JSON format"):
        tokenizers.detect_pretokenization_tokenizer_type(tokenizer_path)


def test_tiktoken_special_token_ids_use_nemotron4_special_token_order(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "n4-v6-262k.json"
    _write_tiktoken_vocab(tokenizer_path)

    assert tokenizers.special_token_ids_from_tokenizer(tokenizer_path) == {
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        "img_id": 18,
        "img_break_id": 20,
        "img_end_id": 19,
    }
