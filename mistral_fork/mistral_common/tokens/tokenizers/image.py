from dataclasses import dataclass

import numpy as np


def image_size_to_num_tokens(
    size: tuple[int | float, int | float],
    max_image_size: int,
    patch_size: int,
    spatial_merge_size: int,
) -> tuple[int, int]:
    w, h = size
    ratio = max(h / max_image_size, w / max_image_size)
    if ratio > 1:
        w = round(w / ratio)
        h = round(h / ratio)

    width_tokens = int((w - 1) // (patch_size * spatial_merge_size) + 1)
    height_tokens = int((h - 1) // (patch_size * spatial_merge_size) + 1)

    return width_tokens, height_tokens


@dataclass
class ImageEncoding:
    tokens: list[int]
    image: np.ndarray

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)
