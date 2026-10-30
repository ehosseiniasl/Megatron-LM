from functools import lru_cache

import torch.distributed as dist
from torch.distributed import ProcessGroup


_CONTEXT_DATA_PARALLEL_GROUP: ProcessGroup | None = None
_DATA_PARALLEL_GROUP: ProcessGroup | None = None
_MODEL_PARALLEL_GROUP: ProcessGroup | None = None


def set_parallel_groups(
    *,
    context_data_parallel_group: ProcessGroup,
    data_parallel_group: ProcessGroup,
    model_parallel_group: ProcessGroup,
) -> None:
    global _CONTEXT_DATA_PARALLEL_GROUP, _DATA_PARALLEL_GROUP, _MODEL_PARALLEL_GROUP
    _CONTEXT_DATA_PARALLEL_GROUP = context_data_parallel_group
    _DATA_PARALLEL_GROUP = data_parallel_group
    _MODEL_PARALLEL_GROUP = model_parallel_group


@lru_cache()
def get_data_parallel_rank() -> int:
    assert _DATA_PARALLEL_GROUP is not None, "parallel groups not initialized"
    return dist.get_rank(group=_DATA_PARALLEL_GROUP)


@lru_cache()
def get_data_parallel_world_size() -> int:
    assert _DATA_PARALLEL_GROUP is not None, "parallel groups not initialized"
    return dist.get_world_size(group=_DATA_PARALLEL_GROUP)


@lru_cache()
def get_model_parallel_world_size() -> int:
    assert _MODEL_PARALLEL_GROUP is not None, "parallel groups not initialized"
    return dist.get_world_size(group=_MODEL_PARALLEL_GROUP)


def get_context_data_parallel_group() -> ProcessGroup:
    assert _CONTEXT_DATA_PARALLEL_GROUP is not None, "parallel groups not initialized"
    return _CONTEXT_DATA_PARALLEL_GROUP
