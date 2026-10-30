from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
import torch

Mask = list[bool]
from mistral_fork.mistral_common.tokens.tokenizers.base import Span


@dataclass
class SequenceOfSamples:
    """
    Concatenation of samples to reach a given size
    """

    x: list[int]
    y: list[int]
    mask: Mask
    sizes: list[int]
    images: list[torch.Tensor] = field(default_factory=list)
    image_positions: list[Span] = field(default_factory=list)
    num_pad_tokens: int = 0
    doc_ids: list[int] = field(default_factory=list)

    tags: list[set[str]] | None = None  # used for aggregating metrics in wandb
    # we using pretokenized and masking we can have some sequences ending with masked part
    is_masked_pretrain: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    image_pixels_loaded: bool = True

    def __post_init__(self) -> None:
        self._assert_sizes_invariants()
        assert self.num_pad_tokens >= 0
        if self.tags is not None:
            assert len(self.tags) == len(self.sizes), (
                f"{len(self.tags)} != {len(self.sizes)}"
            )
        if self.image_pixels_loaded:
            assert len(self.images) == len(self.image_positions), (
                f"{len(self.images)=} {len(self.image_positions)=}"
            )
        else:
            assert not self.images, "metadata-only sequences must not contain image pixels"
        offset = 0
        for i, size in enumerate(self.sizes):
            log_info = (
                i,
                offset,
                offset + size,
                self.y[offset : offset + size],
                self.mask[offset : offset + size],
                self.sizes,
            )
            if not self.is_masked_pretrain:
                assert any(self.mask[offset : offset + size]), log_info
            else:
                # if using pretokenized mask, we can have last document that is fully padded
                assert any(self.mask[offset : offset + size]) or (
                    i == len(self.sizes) - 1
                ), log_info
            # if the end is padded, check that it's the last sequence (for mult_of padding)
            assert self.mask[offset + size - 1] or i == len(self.sizes) - 1, log_info
            offset += size
        assert offset == len(self.x)

    def _assert_sizes_invariants(self) -> None:
        assert len(self.x) == len(self.y) == len(self.mask) == sum(self.sizes), (
            f"{len(self.x)=}, {len(self.y)=}, {len(self.mask)=}, {sum(self.sizes)=}"
        )
        assert all(s > 0 for s in self.sizes), self.sizes

    def check_pretrain_sample(
        self,
        seq_len: int,
        bos_id: int,
        eos_id: int,
        drop_bos: bool = False,
    ) -> None:
        self._assert_sizes_invariants()
        assert sum(self.sizes) == seq_len, (sum(self.sizes), seq_len)
        cum_sizes = np.cumsum(self.sizes)
        expected_boundary_id = eos_id if drop_bos else bos_id
        assert all(
            self.x[offset] == expected_boundary_id for offset in cum_sizes[:-1]
        ), (self.x, expected_boundary_id, cum_sizes)
        assert all(self.y[offset - 1] == eos_id for offset in cum_sizes[:-1]), (
            self.y,
            eos_id,
            cum_sizes,
        )


SequenceOfSamplesType = TypeVar("SequenceOfSamplesType", bound=SequenceOfSamples)


def pad_sequence(
    sample: SequenceOfSamplesType,
    mult_of: int,
    pad_id: int,
) -> SequenceOfSamplesType:
    assert isinstance(sample, SequenceOfSamples)
    n_pad = (mult_of - (len(sample.x) % mult_of)) % mult_of
    if n_pad == 0:
        return sample

    sample.x.extend([pad_id] * n_pad)
    sample.y.extend([pad_id] * n_pad)
    sample.mask.extend([False] * n_pad)
    sample.sizes[-1] += n_pad
    sample.num_pad_tokens += n_pad
    assert len(sample.x) % mult_of == 0
    return sample
