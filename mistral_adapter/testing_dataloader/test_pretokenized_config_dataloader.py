import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mistral_adapter.pretokenized_config_utils import (
    apply_pretokenized_config_overrides,
    load_pretokenized_config,
)
from mistral_fork.mistral.data.pretokenized.processing.tokenizers import (
    build_pretokenization_tokenizer,
)

DEFAULT_CONFIG = Path("mistral_adapter/example_configs/nvidia_data_sharing_1t_4096_MISTRAL.yaml")


def batch_summary(batch: dict[str, Any]) -> str:
    import torch

    parts = []
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            parts.append(f"{key}={tuple(value.shape)}:{value.dtype}")
        elif key == "images" and isinstance(value, list):
            parts.append(f"{key}=list[{len(value)}]")
        else:
            parts.append(f"{key}={type(value).__name__}")
    return ", ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretokenized-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-queue-depth", type=int, default=2)
    parser.add_argument("--num-total-tokens", type=int)
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
        "--no-allow-build-epoch-index",
        action="store_true",
        help="Fail if the requested temporary epoch indexes are missing.",
    )
    parser.add_argument(
        "--use-validation-sources",
        action="store_true",
        help=(
            "Iterate the config's validation_set instead of the training sources. "
            "Swaps in valid_sources and valid_seed so the validation-set epoch "
            "indexes (which are seed-keyed, hence distinct from training's) get "
            "built/checked."
        ),
    )
    parser.add_argument(
        "--epoch-index-build-timeout-s",
        type=float,
        default=600,
        help="Timeout in seconds for the NFS barrier after epoch-index build.",
    )
    parser.add_argument(
        "--dump-x-y-path",
        type=Path,
        default=None,
        help="Write aligned X/Y token IDs and token pieces from the first batch.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer used to decode --dump-x-y-path token pieces.",
    )
    parser.add_argument(
        "--dump-max-tokens",
        type=int,
        default=512,
        help="Maximum number of first-batch token pairs to dump.",
    )
    return parser.parse_args()


def dump_x_y_tokens(
    *, batch: dict[str, Any], output_path: Path, tokenizer_path: Path, max_tokens: int
) -> None:
    assert max_tokens > 0, f"Expected --dump-max-tokens > 0, got {max_tokens}"
    tokenizer = build_pretokenization_tokenizer(tokenizer_path)
    x_ids = batch["tokens"].reshape(-1).tolist()[:max_tokens]
    y_ids = batch["labels"].reshape(-1).tolist()[:max_tokens]
    loss_mask = batch["loss_mask"].reshape(-1).tolist()[:max_tokens]
    assert len(x_ids) == len(y_ids) == len(loss_mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t")
        writer.writerow(
            ["offset", "x_id", "x_token_repr", "y_id", "y_token_repr", "loss_mask"]
        )
        for offset, (x_id, y_id, mask) in enumerate(zip(x_ids, y_ids, loss_mask)):
            writer.writerow(
                [
                    offset,
                    x_id,
                    repr(tokenizer.token_piece(x_id)),
                    y_id,
                    repr(tokenizer.token_piece(y_id)),
                    mask,
                ]
            )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from mistral_adapter.pretokenized_iterator import MistralPretokenizedIterator

    config = load_pretokenized_config(args.pretokenized_config)
    assert (args.dump_x_y_path is None) == (args.tokenizer_path is None), (
        "--dump-x-y-path and --tokenizer-path must be provided together"
    )

    if args.use_validation_sources:
        validation_set = config.validation_set
        assert validation_set is not None and validation_set.valid_sources, (
            "--use-validation-sources requires a validation_set with valid_sources "
            "in the pretokenized config"
        )
        valid_seed = validation_set.valid_seed
        config = config.model_copy(
            update={"sources": validation_set.valid_sources, "seed": valid_seed}
        )
        print(
            f"Using validation sources: {len(validation_set.valid_sources)} sources, "
            f"valid_seed={valid_seed}",
            flush=True,
        )

    config = apply_pretokenized_config_overrides(
        config,
        percentage_data_multiplier=args.percentage_data_multiplier,
        epoch_index_cache_path=args.epoch_index_cache_path,
    )
    allow_build_of_epoch_index = not args.no_allow_build_epoch_index
    assert config.sources, "PretokenizedConfig must contain at least one source"

    num_total_tokens = args.num_total_tokens
    if num_total_tokens is None:
        tokens_per_batch = args.micro_batch_size * config.original_seq_len * args.world_size
        num_total_tokens = tokens_per_batch * args.num_batches

    print(
        "Building dataloader with "
        f"{len(config.sources)} sources, "
        f"percentage_data={sorted({s.percentage_data for s in config.sources})}, "
        f"epoch_index_cache_path={config.epoch_index_cache_path}, "
        f"allow_build_of_epoch_index={allow_build_of_epoch_index}, "
        f"num_total_tokens={num_total_tokens}, "
        f"rank={args.rank}, world_size={args.world_size}",
        flush=True,
    )
    iterator = MistralPretokenizedIterator(
        config=config,
        num_total_tokens=num_total_tokens,
        rank=args.rank,
        world_size=args.world_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        prefetch_queue_depth=args.prefetch_queue_depth,
        allow_build_of_epoch_index=allow_build_of_epoch_index,
        nfs_barrier_dir=Path(config.epoch_index_cache_path) / "barriers",
        epoch_index_build_timeout_s=args.epoch_index_build_timeout_s,
    )

    for batch_idx in range(args.num_batches):
        batch = next(iterator)
        print(f"batch {batch_idx}: {batch_summary(batch)}", flush=True)
        if batch_idx == 0 and args.dump_x_y_path is not None:
            assert args.tokenizer_path is not None
            dump_x_y_tokens(
                batch=batch,
                output_path=args.dump_x_y_path,
                tokenizer_path=args.tokenizer_path,
                max_tokens=args.dump_max_tokens,
            )
            print(f"Dumped first-batch X/Y tokens to {args.dump_x_y_path}", flush=True)

    print(f"Successfully read {args.num_batches} batches.", flush=True)


if __name__ == "__main__":
    main()
