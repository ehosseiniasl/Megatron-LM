import logging
import math
import os
import typing
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mistral_fork.mistral.data.args import TwoPhaseArgs
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.pretokenized_args import (
    parse_filter_profiles,
    parse_masking_strategies,
    parse_per_source_map,
)
from mistral_fork.mistral.data.source import SampleType

if TYPE_CHECKING:
    from mistral_fork.mistral.data.pretokenized.pretokenized_args import (
        PretokenizedArgs,
    )

logger = logging.getLogger(__name__)
DEFAULT_TOKEN_DTYPE = np.int32
_EPOCH_INDEX_MARGIN_SIGMA = 7


def atomic_write_once(path: Path, content: str) -> None:
    """Write content to path atomically, first writer wins.

    On Linux, Path.rename() silently overwrites existing files.
    This uses os.link() which raises FileExistsError if the target
    already exists, ensuring only the first writer's content persists.
    """
    tmp_path = path.with_name(f"tmp.{uuid.uuid4()}.{path.name}")
    tmp_path.write_text(content)
    try:
        os.link(tmp_path, path)
    except FileExistsError:
        pass
    finally:
        tmp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class SourceItem:
    """Represents a data source with its weight schedule and type."""

    source_path: Path
    sample_type: SampleType
    weight_schedule: "WeightSchedule"
    percentage_data: float
    filter_profile: FilterArgsName
    pretokenized_path: Path | None = None
    image_transform_type: str | None = None
    n_epoches_to_prebuild: int | None = None

    def compute_safe_num_tokens(
        self, num_total_tokens: int, sequence_length: int
    ) -> int:
        expected = self.weight_schedule.compute_total_tokens(num_total_tokens)
        expected = max(expected, sequence_length)
        margin = int(_EPOCH_INDEX_MARGIN_SIGMA * math.sqrt(sequence_length * expected))
        return expected + margin


def get_current_weights(items: list[SourceItem], tokens_processed: int) -> "np.ndarray":
    """Get current weights for all items at given token count.

    Args:
        items: List of SourceItem to get weights for.
        tokens_processed: Number of tokens processed so far.

    Returns:
        Numpy array of weights summing to 1.
    """
    weights = np.array(
        [item.weight_schedule.get_weight(tokens_processed) for item in items]
    )
    assert np.isclose(np.sum(weights), 1.0, atol=1e-6), (
        f"Weights must sum to 1, got {np.sum(weights)}"
    )
    return weights


def format_file_mapping(items: list[SourceItem]) -> str:
    """Format unique source -> pretokenized mappings as a log string."""
    seen: set[tuple[Path, Path]] = set()
    lines = []
    for item in items:
        assert item.pretokenized_path is not None
        pretok_dir = item.pretokenized_path.parent
        key = (item.source_path, pretok_dir)
        if key not in seen:
            seen.add(key)
            lines.append(f"  {item.source_path} -> {pretok_dir}")
    return "File mapping:\n" + "\n".join(lines)


def count_file_mappings(items: list[SourceItem]) -> int:
    """Count unique source -> pretokenized mappings, as format_file_mapping lists."""
    seen: set[tuple[Path, Path]] = set()
    for item in items:
        assert item.pretokenized_path is not None
        seen.add((item.source_path, item.pretokenized_path.parent))
    return len(seen)


def parse_percentage_data_multipliers(multipliers_str: str) -> dict[Path, float]:
    """Parse a multipliers string into a dict of path -> multiplier.

    Format: "/path/a:2,/path/b:3"
    """
    return parse_per_source_map(multipliers_str, float)


def _assert_all_paths_have_specs(
    specs_by_path: dict[Path, typing.Any], data_paths: set[Path]
) -> None:
    specified_paths = set(specs_by_path.keys())
    if specified_paths != data_paths:
        raise ValueError(
            f"Mismatching paths:\n\n"
            f"{specified_paths=}\n\n"
            f"{data_paths=}\n\n"
            f"delta={specified_paths ^ data_paths}"
        )


def _compute_percentage_data(
    path: Path,
    default_percentage_data: float,
    multipliers: dict[Path, float],
) -> float:
    """Compute percentage_data for a path, applying multiplier if configured.

    If multipliers is non-empty, the path must be present in the dict (enforced
    by _assert_all_paths_have_specs requiring exact match between multiplier paths and
    data paths).
    """
    if multipliers:
        percentage = default_percentage_data * multipliers[path]
    else:
        percentage = default_percentage_data
    assert 0 < percentage <= 1, (
        f"percentage_data for {path} is {percentage}, must be in (0, 1]"
    )
    return percentage


def _create_weight_schedule(
    phase1_weight: float,
    phase2_weight: float | None,
    two_phase_args: TwoPhaseArgs | None,
) -> "WeightSchedule":
    """Create the appropriate weight schedule based on training configuration.

    Args:
        phase1_weight: The weight for phase 1 training.
        phase2_weight: The weight for phase 2 training, or None for single-phase.
        two_phase_args: Two-phase training arguments, or None for single-phase.

    Returns:
        ConstantWeightSchedule for single-phase, LinearInterpolationWeightSchedule
        for two-phase training.
    """
    if two_phase_args is None or phase2_weight is None:
        return ConstantWeightSchedule(phase1_weight)
    return LinearInterpolationWeightSchedule(
        weight_init=phase1_weight,
        weight_final=phase2_weight,
        mix_start_tok=two_phase_args.mix_start_tok,
        mix_end_tok=two_phase_args.mix_end_tok,
    )


class WeightSchedule(ABC):
    """Abstract base class for weight schedules.

    Weight schedules define how sampling weights change over the course of training.
    Each SourceItem has its own weight schedule.
    """

    @abstractmethod
    def get_weight(self, tokens_processed: int) -> float:
        """Get the weight at a given number of tokens processed.

        Args:
            tokens_processed: The number of tokens processed so far.

        Returns:
            The weight for this source at the given token count.
        """
        ...

    @abstractmethod
    def get_conservative_weight(self) -> float:
        """Get conservative weight for epoch index building.

        Returns:
            A weight representing worst-case probability for this source.
        """
        ...

    @abstractmethod
    def compute_total_tokens(self, num_total_tokens: int) -> int:
        """Compute expected tokens consumed by this source over the full run."""
        ...

    @abstractmethod
    def scale(self, factor: float) -> "WeightSchedule":
        """Create a new weight schedule with weight scaled by the given factor.

        Args:
            factor: The scaling factor to apply.

        Returns:
            A new WeightSchedule with scaled weight.
        """
        ...


@dataclass(frozen=True)
class ConstantWeightSchedule(WeightSchedule):
    """A weight schedule that returns a constant weight."""

    weight: float

    def get_weight(self, tokens_processed: int) -> float:
        return self.weight

    def get_conservative_weight(self) -> float:
        return self.weight

    def compute_total_tokens(self, num_total_tokens: int) -> int:
        return int(self.weight * num_total_tokens)

    def scale(self, factor: float) -> "ConstantWeightSchedule":
        return ConstantWeightSchedule(self.weight * factor)


@dataclass(frozen=True)
class LinearInterpolationWeightSchedule(WeightSchedule):
    """A weight schedule that linearly interpolates between initial and final weights."""

    weight_init: float
    weight_final: float
    mix_start_tok: int
    mix_end_tok: int

    def __post_init__(self) -> None:
        assert self.mix_start_tok <= self.mix_end_tok, (
            self.mix_start_tok,
            self.mix_end_tok,
        )

    def get_weight(self, tokens_processed: int) -> float:
        if tokens_processed <= self.mix_start_tok:
            return self.weight_init
        elif tokens_processed > self.mix_end_tok:
            return self.weight_final
        else:  # linearly interpolate
            t = (tokens_processed - self.mix_start_tok) / (
                self.mix_end_tok - self.mix_start_tok
            )
            return self.weight_init + (self.weight_final - self.weight_init) * t

    def get_conservative_weight(self) -> float:
        # Return worst case probability (maximum of initial and final weights)
        return max(self.weight_init, self.weight_final)

    def compute_total_tokens(self, num_total_tokens: int) -> int:
        phase1_end = min(self.mix_start_tok, num_total_tokens)
        tokens = self.weight_init * phase1_end
        if num_total_tokens <= self.mix_start_tok:
            return int(tokens)

        transition_end = min(self.mix_end_tok, num_total_tokens)
        weight_at_end = self.get_weight(transition_end)
        tokens += (
            (self.weight_init + weight_at_end)
            / 2
            * (transition_end - self.mix_start_tok)
        )
        if num_total_tokens <= self.mix_end_tok:
            return int(tokens)

        tokens += self.weight_final * (num_total_tokens - self.mix_end_tok)
        return int(tokens)

    def scale(self, factor: float) -> "LinearInterpolationWeightSchedule":
        return LinearInterpolationWeightSchedule(
            weight_init=self.weight_init * factor,
            weight_final=self.weight_final * factor,
            mix_start_tok=self.mix_start_tok,
            mix_end_tok=self.mix_end_tok,
        )


def _optimized_sample_datasource_ids(
    seq_ids: typing.Sequence[int],
    weights: np.ndarray,
    seed: int,
) -> list[int]:
    """
    IMPORTANT: Result for each seq_id depends only on (seq_id, seed, weights, CHUNK_SIZE),
    not on the number or ordering of seq_ids in this call.

    Before this we were using
    datasource_ids = [
            RandomState(seq_id).choice(range(len(self._datasources)), p=self._weights)
            for seq_id in range(
                self._state.global_sequence_id,
                self._state.global_sequence_id + num_sequences,
            )
        ]
    which was extremely slow (7s for 100_000 sequences)

    To do so we only create a rng and do rng.choice every CHUNK_SIZE, with takes only 10ms

    Note: we could optimize further using cdf + searchsort but it was only marginally faster on benches and adding a bit of complexity
    """
    CHUNK_SIZE = 10_000  # IMPORTANT: do not change it it will change the order of sampled datasource_ids

    seq_ids_arr = np.asarray(seq_ids, dtype=np.int64)
    if (seq_ids_arr < 0).any():
        raise ValueError("seq_ids must be non-negative")

    chunk_ids = seq_ids_arr // CHUNK_SIZE
    offsets = seq_ids_arr - chunk_ids * CHUNK_SIZE

    sampled = np.full(seq_ids_arr.shape, fill_value=-1, dtype=np.int64)
    for cid in np.unique(chunk_ids):
        # deterministic RNG per chunk
        rng = np.random.default_rng((seed, int(cid)))
        chunk = rng.choice(a=np.arange(len(weights)), size=CHUNK_SIZE, p=weights)
        mask = chunk_ids == cid
        sampled[mask] = chunk[offsets[mask]]

    if (sampled < 0).any():
        raise RuntimeError("something went wrong filling uniforms")

    return sampled.tolist()
