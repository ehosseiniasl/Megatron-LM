"""
Simplified epoch index prebuilding script for the Megatron fork.

Differences from the original prebuild_epoch_indexes.py:
- No ConfStore / DataArgs: uses mistral_adapter PretokenizedConfig YAML.
- Reuses the adapter config wiring so runtime and prebuild agree on transforms.
- Accepts a Slurm rank plus a local process count from the shell launcher.
- No run_fn_as_cli: plain argparse.
"""

import argparse
import logging
from pathlib import Path

from mistral_adapter.pretokenized_config import PretokenizedConfig
from mistral_adapter.pretokenized_config_utils import (
    apply_pretokenized_config_overrides,
    build_image_encoder_fn,
    config_to_source_items,
    load_pretokenized_config,
)
from mistral_fork.mistral.data.pretokenized.distributed_prebuild_of_epoch_indexes import (
    check_prebuild,
    prebuild_epoch_index,
)
from mistral_fork.mistral.data.pretokenized.document_filters import PdsFilter

logger = logging.getLogger(Path(__file__).stem)


def _load_config(
    path: Path,
    *,
    num_total_tokens: int | None,
    num_batches: int,
    percentage_data_multiplier: float | None,
    epoch_index_cache_path: str | None,
    use_validation_sources: bool = False,
) -> tuple[PretokenizedConfig, int]:
    config = load_pretokenized_config(path)
    if use_validation_sources:
        validation_set = config.validation_set
        assert validation_set is not None and validation_set.valid_sources, (
            "--validation-sources requires a validation_set with valid_sources "
            "in the pretokenized config"
        )
        # Swap in the validation sources and seed so the (seed-keyed) validation
        # epoch indexes get built/checked instead of training's.
        config = config.model_copy(
            update={"sources": validation_set.valid_sources, "seed": validation_set.valid_seed}
        )
    config = apply_pretokenized_config_overrides(
        config,
        percentage_data_multiplier=percentage_data_multiplier,
        epoch_index_cache_path=epoch_index_cache_path,
    )
    if num_total_tokens is None:
        num_total_tokens = config.original_micro_batch_size * config.original_seq_len * num_batches
    return config, num_total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prebuild epoch indexes from a PretokenizedConfig YAML"
    )
    parser.add_argument("command", choices=["prebuild", "check"], help="Command to run")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a mistral_adapter PretokenizedConfig YAML file",
    )
    parser.add_argument(
        "--num_total_tokens",
        type=int,
        help=(
            "Override total token horizon. Required only when the default "
            "--num_batches conversion is not what you want."
        ),
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=5,
        help=(
            "Compute num_total_tokens as original_micro_batch_size * "
            "original_seq_len * num_batches when --num_total_tokens is not provided."
        ),
    )
    parser.add_argument(
        "--percentage-data-multiplier",
        type=float,
        default=None,
        help=(
            "If set, multiply every source percentage_data by this value after "
            "loading the config."
        ),
    )
    parser.add_argument(
        "--epoch-index-cache-path",
        type=str,
        default=None,
        help=(
            "If set, override epoch_index_cache_path from the config. Defaults "
            "to None, which keeps the YAML value."
        ),
    )
    parser.add_argument(
        "--validation-sources",
        action="store_true",
        help=(
            "Build/check epoch indexes for the config's validation_set "
            "(valid_sources at valid_seed) instead of the training sources. "
            "Validation epoch indexes are seed-keyed, so they are distinct from "
            "training's whenever valid_seed differs from the train seed."
        ),
    )
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--processes-per-worker",
        type=int,
        default=1,
        help="Number of local epoch-index builder processes launched by this rank.",
    )
    parser.add_argument(
        "--image-transform-workers",
        type=int,
        default=4,
        help="Processes used to precompute image transforms within each datasource.",
    )
    parser.add_argument(
        "--epoch-index-pds-filter",
        type=PdsFilter,
        choices=list(PdsFilter),
        default=None,
        help=(
            "Exclude documents containing images in this PDS quarantine set while "
            "building epoch indexes. Use a dedicated epoch-index cache root."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config, num_total_tokens = _load_config(
        args.config,
        num_total_tokens=args.num_total_tokens,
        num_batches=args.num_batches,
        percentage_data_multiplier=args.percentage_data_multiplier,
        epoch_index_cache_path=args.epoch_index_cache_path,
        use_validation_sources=args.validation_sources,
    )
    logger.info(
        "Loaded config (%s sources): sources=%s seed=%s seq_len=%s "
        "num_total_tokens=%s epoch_index_cache_path=%s",
        "validation" if args.validation_sources else "training",
        len(config.sources),
        config.seed,
        config.original_seq_len,
        num_total_tokens,
        config.epoch_index_cache_path,
    )

    if args.command == "prebuild":
        prebuild_epoch_index(
            source_items=config_to_source_items(config),
            seed=config.seed,
            num_total_tokens=num_total_tokens,
            seq_len=config.original_seq_len,
            bos_id=config.bos_id,
            eos_id=config.eos_id,
            pad_id=config.pad_id,
            epoch_index_cache_path=config.epoch_index_cache_path,
            image_encoder_build_fn=build_image_encoder_fn(config),
            skip_images=config.skip_images,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
            processes_per_worker=args.processes_per_worker,
            truncate_trailing_masked=config.truncate_trailing_masked,
            image_transform_workers=args.image_transform_workers,
            epoch_index_pds_filter=args.epoch_index_pds_filter,
        )
    elif args.command == "check":
        check_prebuild(
            source_items=config_to_source_items(config),
            seed=config.seed,
            num_total_tokens=num_total_tokens,
            seq_len=config.original_seq_len,
            bos_id=config.bos_id,
            eos_id=config.eos_id,
            pad_id=config.pad_id,
            epoch_index_cache_path=config.epoch_index_cache_path,
            image_encoder_build_fn=build_image_encoder_fn(config),
            skip_images=config.skip_images,
            truncate_trailing_masked=config.truncate_trailing_masked,
            epoch_index_pds_filter=args.epoch_index_pds_filter,
        )


if __name__ == "__main__":
    main()
