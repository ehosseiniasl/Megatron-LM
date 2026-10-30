import dataclasses
import functools
from pathlib import Path
from typing import Callable, TypeVar

from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
    MaskingStrategy,
)
from mistral_fork.shared.cluster import clusterify_path

DEFAULT_PRETOKENIZATION_FOLDER = "/mnt/pretokenized"
DEFAULT_PATH_PRETOKENIZATION_DATA = (
    f"{DEFAULT_PRETOKENIZATION_FOLDER}/pretokenized_data"
)
DEFAULT_PRETOKENIZED_EPOCH_INDEXES = f"{DEFAULT_PRETOKENIZATION_FOLDER}/epoch_indexes"

T = TypeVar("T")


def parse_per_source_map(s: str, parse_value: Callable[[str], T]) -> dict[Path, T]:
    if not s:
        return {}
    result: dict[Path, T] = {}
    for pair in s.split(","):
        raw_path, raw_value = pair.split(":", 1)
        path = Path(clusterify_path(raw_path.strip()))
        if path in result:
            raise ValueError(f"Duplicate path {raw_path!r} in per-source map: {s!r}")
        result[path] = parse_value(raw_value.strip())
    return result


def _parse_filter_profile_name(value: str) -> FilterArgsName:
    return FilterArgsName(value)


def _parse_masking_strategy_value(v: str) -> MaskingStrategy | None:
    return MaskingStrategy(v) if v != "null" else None


@functools.lru_cache
def parse_filter_profiles(s: str) -> dict[Path, FilterArgsName]:
    return parse_per_source_map(s, _parse_filter_profile_name)


@functools.lru_cache
def parse_masking_strategies(s: str) -> dict[Path, MaskingStrategy | None]:
    return parse_per_source_map(s, _parse_masking_strategy_value)


@dataclasses.dataclass
class PretokenizedArgs:
    # A comma-separated list of paths and filter profile values. ("/path/a:PRETRAIN_V1,/path/b:MULTIMODAL_V1")
    # Profile values must match FilterArgsName.value (mistral/data/pretokenized/filter_profiles.py).
    per_source_filter: (
        str  # Required. Format: "/path/a:PRETRAIN_V1,/path/b:PRETRAIN_V2"
    )
    pretokenized_data_dir: str = DEFAULT_PATH_PRETOKENIZATION_DATA
    # Where to store epoch indices.
    epoch_index_cache_path: str = DEFAULT_PRETOKENIZED_EPOCH_INDEXES
    # If true, do not shuffle the data.
    no_shuffling: bool = False
    # If true, trim the last split piece's trailing masked run so no packed
    # sequence ends on masked tokens. Only affects sources with loss masks.
    truncate_trailing_masked: bool = True
    # Percentage of tokens to be taken from the datasource. Set it to lower than
    # 1 to do repeat-aware runs on smaller horizon.
    percentage_data: float = 1.0
    # TODO: delete this field once epoch-index build behavior is handled by callers.
    # Crash at submission time if epoch indexes are not there.
    allow_build_of_epoch_index: bool = False
    # Allow startup from a contiguous cache prefix; missing later epochs still fail on access.
    allow_partial_epoch_index_cache: bool = False
    # Size of the merged files.
    merged_size: int = 2**40
    # A comma-separated list of paths and weights. The weight will be applied to
    # percentage_data on that source.
    percentage_data_multipliers: str = ""

    # A comma-separated list of "path:masking_strategy". Use "path:null" if no masking.
    # Example: "path1:MASK_FILE_SEPARATOR,path2:null".
    masking_strategies: str = ""

    # Use optimized datasource_id sampling (faster than naive sampling).
    optimized_datasource_id_sampling: bool = True

    # Legacy resume path for checkpoints written before deterministic source
    # ordering. Keep False unless resuming such a run.
    allow_non_sorted_sources_in_dataloader: bool = False

    # When True, skip image processing and only use text tokens.
    skip_images: bool = False

    # Replace document BOS inputs with EOS at runtime.
    drop_bos: bool = False

    # Last-resort fallback using a hardcoded source_set directory name instead of the
    # cache key system.
    deprecated_source_set: str | None = None

    def get_filter_profile(self, source_dir: Path) -> FilterArgsName:
        """Return the filter profile assigned to *source_dir* via per_source_filter."""
        profile_map = parse_filter_profiles(self.per_source_filter)
        if source_dir not in profile_map:
            raise KeyError(
                f"Source {source_dir} not found in per_source_filter. "
                f"Available sources: {list(profile_map.keys())}"
            )
        return profile_map[source_dir]

    def get_masking_strategy(self, source_dir: Path) -> MaskingStrategy | None:
        """Return the masking strategy for *source_dir*, or None if not configured."""
        if not self.masking_strategies:
            return None
        masking_map = parse_masking_strategies(self.masking_strategies)
        if source_dir not in masking_map:
            raise KeyError(
                f"Source {source_dir} not found in masking_strategies. "
                f"Available sources: {list(masking_map.keys())}"
            )
        return masking_map[source_dir]
