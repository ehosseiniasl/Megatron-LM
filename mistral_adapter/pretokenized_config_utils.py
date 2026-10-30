from functools import partial
from pathlib import Path
from typing import Callable

import yaml

from mistral_adapter.pretokenized_config import CONFIG_VERSION, PretokenizedConfig
from mistral_fork.mistral.data.pretokenized.components.image_transforms import (
    ImageTransformType,
    SpecialImageIDs,
)
from mistral_fork.mistral.data.pretokenized.datasource import PretokenizedImageEncoder
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.pretokenized_utils import (
    ConstantWeightSchedule,
    SourceItem,
)
from mistral_fork.mistral.data.pretokenized.simple_image_encoder import (
    SimpleImageEncoder,
)
from mistral_fork.mistral.data.source import SampleType

_EXTENDS_KEY = "extends"


def _load_pretokenized_config_mapping(
    path: Path, ancestors: tuple[Path, ...] = ()
) -> dict[str, object]:
    path = path.resolve()
    if path in ancestors:
        chain = " -> ".join(str(item) for item in (*ancestors, path))
        raise ValueError(f"Cycle in pretokenized config inheritance: {chain}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise TypeError(f"Pretokenized config must contain a YAML mapping: {path}")

    extends = raw.pop(_EXTENDS_KEY, None)
    if extends is None:
        return raw
    if not isinstance(extends, str):
        raise TypeError(f"'{_EXTENDS_KEY}' must be a path string: {path}")

    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = path.parent / base_path

    base = _load_pretokenized_config_mapping(base_path, (*ancestors, path))
    return {**base, **raw}


def load_pretokenized_config(path: Path) -> PretokenizedConfig:
    raw = _load_pretokenized_config_mapping(path)
    config = PretokenizedConfig.model_validate(raw)
    assert config.version == CONFIG_VERSION, (
        f"Pretokenized config version mismatch: file has v{config.version}, "
        f"but this Megatron build expects v{CONFIG_VERSION}."
    )
    return config


def apply_pretokenized_config_overrides(
    config: PretokenizedConfig,
    *,
    percentage_data_multiplier: float | None = None,
    epoch_index_cache_path: str | None = None,
) -> PretokenizedConfig:
    update = {}
    if percentage_data_multiplier is not None:
        if percentage_data_multiplier <= 0:
            raise ValueError(
                "percentage_data_multiplier must be > 0, got "
                f"{percentage_data_multiplier}"
            )
        update["sources"] = [
            source.model_copy(
                update={
                    "percentage_data": source.percentage_data
                    * percentage_data_multiplier
                }
            )
            for source in config.sources
        ]

    if epoch_index_cache_path is not None:
        update["epoch_index_cache_path"] = epoch_index_cache_path

    if not update:
        return config
    return config.model_copy(update=update)


def config_to_source_items(config: PretokenizedConfig) -> list[SourceItem]:
    total_weight = sum(s.weight for s in config.sources)
    return [
        SourceItem(
            source_path=Path(s.prefix_path),
            sample_type=SampleType(s.sample_type),
            weight_schedule=ConstantWeightSchedule(s.weight / total_weight),
            percentage_data=s.percentage_data,
            filter_profile=FilterArgsName.NO_FILTER,
            pretokenized_path=Path(s.prefix_path),
            image_transform_type=s.image_transform_type,
            n_epoches_to_prebuild=s.n_epoches_to_prebuild,
        )
        for s in config.sources
    ]


def _build_image_encoder(
    transform_type: str, *, special_ids: SpecialImageIDs
) -> PretokenizedImageEncoder:
    return SimpleImageEncoder(
        transform_type=ImageTransformType(transform_type), special_ids=special_ids
    )


def build_image_encoder_fn(
    config: PretokenizedConfig,
) -> Callable[[str], PretokenizedImageEncoder] | None:
    if config.skip_images:
        return None

    has_mm = any(s.image_transform_type is not None for s in config.sources)
    if not has_mm:
        return None

    special_ids = SpecialImageIDs(
        img=config.img_id, img_break=config.img_break_id, img_end=config.img_end_id
    )

    # ProcessPoolExecutor must pickle this callable when epoch-index workers use
    # the spawn start method. A partial of a module-level function is portable;
    # a nested closure is not.
    return partial(_build_image_encoder, special_ids=special_ids)
