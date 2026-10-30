"""Pydantic config for the Megatron pretokenized dataloader.

This config is shared between:
- pretrain_gpt.py (to create the ForkedPretokenizedIterator / ImportedPretokenizedIterator)
- prebuild_epoch_indexes_megatron.py (to prebuild epoch indexes)
- build_pretokenized_dataloader() (to create a PretokenizedDataLoader from config)

It matches the YAML format used by pretrain_gpt.py.
"""

from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from mistral_fork.mistral.data.pretokenized.components.image_transforms import (
    ImageTransformType,
    SpecialImageIDs,
)
from mistral_fork.mistral.data.pretokenized.datasource import (
    PretokenizedImageEncoder,
)
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.pretokenized_args import PretokenizedArgs
from mistral_fork.mistral.data.pretokenized.pretokenized_data_loader import (
    PretokenizedDataLoader,
)
from mistral_fork.mistral.data.pretokenized.pretokenized_utils import (
    ConstantWeightSchedule,
    SourceItem,
)
from mistral_fork.mistral.data.pretokenized.simple_image_encoder import (
    SimpleImageEncoder,
)
from mistral_fork.mistral.data.source import SampleType


class SourceConfig(BaseModel):
    prefix_path: str
    weight: float
    percentage_data: float = 1.0
    sample_type: SampleType = SampleType.PRETRAIN
    # Per-source image transform type (required for multimodal sources)
    image_transform_type: ImageTransformType | None = None


class MegatronDataloaderConfig(BaseModel):
    sources: list[SourceConfig]
    seed: int = 0
    seq_len: int
    bos_id: int
    eos_id: int
    pad_id: int
    img_id: int
    img_break_id: int
    img_end_id: int
    epoch_index_cache_path: str = "/tmp/megatron_epoch_cache"
    num_total_tokens: int
    no_shuffling: bool = False
    allow_build_of_epoch_index: bool = False
    skip_images: bool = False
    drop_bos: bool = False


def _build_image_encoder_fn(
    special_ids: SpecialImageIDs,
) -> Callable[[str], PretokenizedImageEncoder]:
    """Build an image encoder factory function from transform type names."""

    def build_encoder(transform_type: str) -> PretokenizedImageEncoder:
        return SimpleImageEncoder(
            transform_type=ImageTransformType(transform_type),
            special_ids=special_ids,
        )

    return build_encoder


def build_pretokenized_dataloader(
    config: MegatronDataloaderConfig,
    micro_batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    image_encoder_build_fn: Callable[[str], PretokenizedImageEncoder] | None = None,
) -> PretokenizedDataLoader:
    """Build a PretokenizedDataLoader from a MegatronDataloaderConfig.

    This is the same wiring that pretrain_gpt.py does manually from YAML.
    If image_encoder_build_fn is not provided but sources have image_transform_type,
    one is built automatically from the per-source config.
    """
    total_weight = sum(s.weight for s in config.sources)
    source_items = [
        SourceItem(
            source_path=Path(s.prefix_path),
            sample_type=s.sample_type,
            weight_schedule=ConstantWeightSchedule(s.weight / total_weight),
            percentage_data=s.percentage_data,
            filter_profile=FilterArgsName.NO_FILTER,
            pretokenized_path=Path(s.prefix_path),
            image_transform_type=(
                s.image_transform_type.value
                if s.image_transform_type is not None
                else None
            ),
        )
        for s in config.sources
    ]

    # Auto-build image_encoder_build_fn from per-source config if not provided
    if image_encoder_build_fn is None:
        has_mm_sources = any(s.image_transform_type is not None for s in config.sources)
        if has_mm_sources:
            image_encoder_build_fn = _build_image_encoder_fn(
                special_ids=SpecialImageIDs(
                    img=config.img_id,
                    img_break=config.img_break_id,
                    img_end=config.img_end_id,
                ),
            )

    per_source_filter = ",".join(f"{s.prefix_path}:no_filter" for s in config.sources)

    return PretokenizedDataLoader(
        source_items=source_items,
        seed=config.seed,
        num_toks_in_global_batch=micro_batch_size * config.seq_len * world_size,
        num_total_tokens=config.num_total_tokens,
        seq_len=config.seq_len,
        rank=rank,
        world_size=world_size,
        bos_id=config.bos_id,
        eos_id=config.eos_id,
        pad_id=config.pad_id,
        pretokenized_args=PretokenizedArgs(
            per_source_filter=per_source_filter,
            epoch_index_cache_path=config.epoch_index_cache_path,
            no_shuffling=config.no_shuffling,
            allow_build_of_epoch_index=config.allow_build_of_epoch_index,
            skip_images=config.skip_images,
            drop_bos=config.drop_bos,
        ),
        image_encoder_build_fn=image_encoder_build_fn,
        multiprocessed_worker_info=None,
        selection_strategy=None,
    )
