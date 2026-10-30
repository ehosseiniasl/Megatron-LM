"""Versioned document filters applied by fork-local pretokenization."""

from __future__ import annotations

import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Collection


class RareTokenFilter(StrEnum):
    """Registered rare-token filters whose values are part of the cache key.

    The lists backing a value are immutable. Add a new versioned enum value when
    changing a list so existing pretokenized caches keep their original meaning.
    """

    RARE_TOKENS_DEBUG = "rare_tokens_DEBUG"
    RARE_TOKENS_N4_V6_262K_V1 = "rare_tokens_n4_v6_262k_v1"


class PdsFilter(StrEnum):
    """Registered PDS quarantine manifests whose values enter the cache key."""

    NVIDIA_PDS_QUARANTINE_V1 = "nvidia_pds_quarantine_v1"
    NVIDIA_PDS_QUARANTINE_V2 = "nvidia_pds_quarantine_v2"
    NVIDIA_PDS_QUARANTINE_V3 = "nvidia_pds_quarantine_v3"


class BadDocumentFilter(StrEnum):
    """Registered bad-document filters whose values are part of the cache key."""

    NVIDIA_BAD_DOCUMENTS_DEBUG = "nvidia_bad_documents_DEBUG"


# DEBUG tokens sampled from 10,000 documents in the existing wikibooks pretok.
RARE_TOKEN_IDS_DEBUG = frozenset({1_320, 2_128, 2_142})

N4_V6_262K_FILTERED_TOKENS_PATH = Path(
    "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/"
    "tokenizers/n4-v6-262k/filtered_tokens.json"
)

RARE_TOKEN_FILTER_PATHS: dict[RareTokenFilter, Path] = {
    RareTokenFilter.RARE_TOKENS_N4_V6_262K_V1: N4_V6_262K_FILTERED_TOKENS_PATH,
}

NVIDIA_PDS_QUARANTINE_V1_PATHS = (
    Path(
        "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/"
        "nv-00-10603-sandia/quarantine/2026-07-28/"
        "pulse_quarantine_522926.parquet"
    ),
    Path(
        "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/"
        "nv-00-10603-sandia/quarantine-0813/pulse_quarantine_0813.parquet"
    ),
)
NVIDIA_PDS_QUARANTINE_V2_PATHS = (
    *NVIDIA_PDS_QUARANTINE_V1_PATHS,
    Path(
        "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/"
        "nv-00-10603-sandia/quarantine-0826/pulse_quarantine_0826.parquet"
    ),
    Path(
        "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/"
        "nv-00-10603-sandia/pulse_quarantine_0831/"
        "pulse_quarantine_0831.parquet"
    ),
)
NVIDIA_PDS_QUARANTINE_V3_ROOT = Path(
    "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron-v4/"
    "sandia/nv-00-10603-sandia/quarantine"
)
NVIDIA_PDS_QUARANTINE_V3_PATHS = (
    NVIDIA_PDS_QUARANTINE_V3_ROOT / "2026-07-28/pulse_quarantine_522926.parquet",
    NVIDIA_PDS_QUARANTINE_V3_ROOT / "pulse_quarantine_0813.parquet",
    NVIDIA_PDS_QUARANTINE_V3_ROOT / "pulse_quarantine_0826.parquet",
    NVIDIA_PDS_QUARANTINE_V3_ROOT / "pulse_quarantine_0831.parquet",
    NVIDIA_PDS_QUARANTINE_V3_ROOT / "pulse_quarantine_0914.parquet",
)

PDS_FILTER_PATHS: dict[PdsFilter, tuple[Path, ...]] = {
    PdsFilter.NVIDIA_PDS_QUARANTINE_V1: NVIDIA_PDS_QUARANTINE_V1_PATHS,
    PdsFilter.NVIDIA_PDS_QUARANTINE_V2: NVIDIA_PDS_QUARANTINE_V2_PATHS,
    PdsFilter.NVIDIA_PDS_QUARANTINE_V3: NVIDIA_PDS_QUARANTINE_V3_PATHS,
}

# Offsets point to the first byte of a JSONL record. Paths are matched exactly.
# DEBUG entry removes the first document from the first symlinked source chunk.
NVIDIA_BAD_DOCUMENT_BYTE_OFFSETS_DEBUG: dict[str, frozenset[int]] = {
    "/lustre/fsw/portfolios/nemotron/users/thib.ma/data/"
    "debug_pr214_document_filters/raw/wikibooks_2chunks/chunk.00000.jsonl": (
        frozenset({0})
    ),
}


@lru_cache(maxsize=None)
def _load_rare_token_ids(path: Path) -> frozenset[int]:
    """Load token IDs from a filtered-token JSON artifact."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Rare-token filter artifact not found: {path}. "
            "Pretokenization must run where this cluster path is visible."
        )
    try:
        with path.open(encoding="utf-8") as file:
            entries = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid rare-token filter JSON at {path}: {exc}") from exc

    if not isinstance(entries, list):
        raise ValueError(
            f"Rare-token filter JSON at {path} must contain a top-level list"
        )

    token_ids: set[int] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Rare-token filter entry {index} at {path} must be an object"
            )
        token_id = entry.get("token_id")
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError(
                f"Rare-token filter entry {index} at {path} has invalid "
                f"token_id: {token_id!r}"
            )
        token_ids.add(token_id)
    return frozenset(token_ids)


def rare_token_ids(rare_token_filter: RareTokenFilter) -> frozenset[int]:
    """Return the immutable token-ID set registered for a rare-token filter."""

    if rare_token_filter is RareTokenFilter.RARE_TOKENS_DEBUG:
        return RARE_TOKEN_IDS_DEBUG
    try:
        path = RARE_TOKEN_FILTER_PATHS[rare_token_filter]
    except KeyError as exc:
        raise ValueError(
            f"No token-ID artifact registered for rare-token filter "
            f"{rare_token_filter.value}"
        ) from exc
    return _load_rare_token_ids(path)


def preload_rare_token_filters(
    rare_token_filters: Collection[RareTokenFilter],
) -> None:
    """Load configured rare-token filters before worker processes are created."""

    for rare_token_filter in set(rare_token_filters):
        rare_token_ids(rare_token_filter)


def should_filter_bad_document(
    bad_document_filter: BadDocumentFilter | None,
    *,
    jsonl_path: Path,
    byte_offset: int,
) -> bool:
    """Return whether a document is excluded by its path and byte offset."""

    if bad_document_filter is not BadDocumentFilter.NVIDIA_BAD_DOCUMENTS_DEBUG:
        return False
    bad_offsets = NVIDIA_BAD_DOCUMENT_BYTE_OFFSETS_DEBUG.get(str(jsonl_path))
    return bad_offsets is not None and byte_offset in bad_offsets


def should_filter_rare_tokens(
    rare_token_filter: RareTokenFilter | None, token_ids: Collection[int]
) -> bool:
    """Return whether a tokenized document contains a registered rare token."""

    if rare_token_filter is None:
        return False
    return not rare_token_ids(rare_token_filter).isdisjoint(token_ids)
