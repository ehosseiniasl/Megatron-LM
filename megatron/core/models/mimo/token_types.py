# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Token categories used by MIMO residual tensor metrics."""

from enum import IntEnum


class MimoTokenType(IntEnum):
    """Stable IDs for MIMO residual token categories."""

    IMAGE = 0
    SPECIAL_IMAGE = 1
    SPECIAL_TEXT = 2
    PAD = 3
    TEXT = 4


MIMO_TOKEN_TYPE_NAMES = ("image", "special-image", "special-text", "pad", "text")
