# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.run_dynamic_text_generation_server import _resolve_inference_context_from_checkpoint


def _args(**changes):
    values = {
        "inference_context_from_checkpoint": False,
        "inference_max_seq_length": 2560,
        "seq_length": 8192,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_existing_default_is_unchanged_without_opt_in():
    args = _args()

    _resolve_inference_context_from_checkpoint(args, set())

    assert args.inference_max_seq_length == 2560


def test_opt_in_uses_checkpoint_sequence_length():
    args = _args(inference_context_from_checkpoint=True)

    with patch("tools.run_dynamic_text_generation_server.print_rank_0"):
        _resolve_inference_context_from_checkpoint(args, set())

    assert args.inference_max_seq_length == 8192


def test_explicit_inference_context_takes_precedence():
    args = _args(inference_context_from_checkpoint=True, inference_max_seq_length=16384)

    _resolve_inference_context_from_checkpoint(args, {"inference_max_seq_length"})

    assert args.inference_max_seq_length == 16384


def test_opt_in_rejects_invalid_checkpoint_sequence_length():
    args = _args(inference_context_from_checkpoint=True, seq_length=None)

    with pytest.raises(ValueError, match="did not provide a positive seq_length"):
        _resolve_inference_context_from_checkpoint(args, set())
