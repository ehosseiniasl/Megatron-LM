# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.completions import (
    _tokenize_user_prompt,
)


def test_string_prompt_uses_user_text_tokenization():
    tokenizer = Mock()
    tokenizer.tokenize_user_text.return_value = [1, 2, 3]

    assert _tokenize_user_prompt(tokenizer, "literal <img> text") == [1, 2, 3]
    tokenizer.tokenize_user_text.assert_called_once_with("literal <img> text")
    tokenizer.tokenize.assert_not_called()


def test_string_prompt_supports_tokenizers_without_user_text_api():
    tokenizer = Mock(spec=["tokenize"])
    tokenizer.tokenize.return_value = [4, 5, 6]

    assert _tokenize_user_prompt(tokenizer, "ordinary text") == [4, 5, 6]
    tokenizer.tokenize.assert_called_once_with("ordinary text")
