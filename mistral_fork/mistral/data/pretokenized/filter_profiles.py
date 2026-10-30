"""Filter profile names for pretokenized data.

String values in the enum are stable — changing a value invalidates cached data on disk.
The Python identifier (enum member name) can be renamed freely without affecting the cache.
"""

from enum import StrEnum


class FilterArgsName(StrEnum):
    """Registered names for immutable filter argument sets.

    The name value is used as part of the cache key for pretokenized data.
    String values are stable — changing a value invalidates cached data on disk.
    Add new entries as needed for your filter configurations.
    """

    NO_FILTER = "no_filter"
    PRETRAIN_V3__SINKDEEP = "pretrain_v3__sinkdeep"
    PRETRAIN_V4__LOBSTRAL = "pretrain_v4__lobstral"
    PRETRAIN_V4__LOBSTRAL_NO_MAX_LENGTH = (
        "pretrain_v4__lobstral_no_max_length"
    )
    MULTIMODAL_V4__LOBSTRAL = "multimodal_v4__lobstral"
    MULTIMODAL_V1__GROUNDING = "multimodal_v1__grounding"
