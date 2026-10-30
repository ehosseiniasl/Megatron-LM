# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native Mistral pretokenized data adapter for heterogeneous MIMO training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import accumulate
from pathlib import Path
from typing import Any, Callable

import torch

from examples.mimo.training.topology import HeteroTopology
from examples.mimo.utils.hetero import get_data_lane_rank, get_language_batch_parallel_size
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.token_types import MimoTokenType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_pp_first_stage
from megatron.core.utils import flatten_batch_for_packed_sequences, get_pg_rank
from megatron.training.checkpointing import get_checkpoint_name
from megatron.training.tensor_metrics.data_source_loss import data_source_metrics_requested
from megatron.training.tensor_metrics.data_source_metrics import modality_metrics_requested
from megatron.training.training import in_phase_transition as _in_phase_transition
from mistral_adapter.per_source_validation import (
    build_per_source_validation_iterator,
    expand_source_ids_to_token_shape,
    resolve_per_source_validation_schedule,
    select_top_weighted_validation_sources,
)
from mistral_adapter.pretokenized_config import PretokenizedConfig
from mistral_adapter.pretokenized_config_utils import (
    apply_pretokenized_config_overrides,
    load_pretokenized_config,
)
from mistral_adapter.pretokenized_iterator import MistralPretokenizedIterator
from mistral_adapter.train_loop_helpers import (
    _validate_pretokenized_training_args,
    num_training_tokens_in_phase,
)
from mistral_fork.mistral.data.pretokenized.components.image_transforms import (
    DEFAULT_IMAGE_PATCH_SIZE,
    ImageTransformType,
)


def configure_mistral_data(args: argparse.Namespace) -> None:
    """Validate MIMO's Mistral-data contract and cache its parsed config on ``args``."""
    if args.dataset_provider != "mistral":
        return
    if not args.pretokenized_config:
        raise ValueError("--dataset-provider mistral requires --pretokenized-config")
    if args.pretokenized_allow_build_of_epoch_index:
        raise ValueError("MIMO requires prebuilt Mistral epoch indexes")

    config = load_pretokenized_config(Path(args.pretokenized_config))
    config = apply_pretokenized_config_overrides(
        config,
        percentage_data_multiplier=getattr(args, "overwrite_percentage_data_multiplier", None),
    )
    _validate_pretokenized_training_args(args, config, tensor_model_parallel_size=args.mimo_llm_tp)

    per_source_validation = config.per_source_validation
    validation_enabled = per_source_validation is not None and int(args.eval_interval or 0) > 0
    image_sharding = getattr(args, "mimo_encoder_image_sharding", False)
    if image_sharding and validation_enabled:
        raise ValueError("encoder image sharding does not yet support per-source validation")
    if validation_enabled:
        selected_validation = select_top_weighted_validation_sources(
            per_source_validation,
            getattr(args, "per_source_validation_top_n", 0),
        )
        if selected_validation is not per_source_validation:
            config = config.model_copy(update={"per_source_validation": selected_validation})
            per_source_validation = selected_validation
        if (
            getattr(args, "mimo_llm_cp", 1) != 1
            or getattr(args, "gtp_remat_num_sequence_shards", 1) != 1
        ):
            raise ValueError(
                "Per-source MIMO validation currently requires --mimo-llm-cp 1 "
                "and no GTP sequence sharding; "
                "source IDs are not context-parallel sharded"
            )
        if getattr(args, "cuda_graph_impl", None) == "full_iteration":
            raise ValueError(
                "Per-source validation does not support full-iteration CUDA graphs"
            )
        if config.validation_set is not None or config.perplexity_eval_sets:
            raise ValueError(
                "MIMO currently supports per_source_validation as its only validation mode"
            )
        eval_global_batch_size = getattr(
            args, "eval_global_batch_size", args.global_batch_size
        )
        eval_micro_batch_size = getattr(
            args, "eval_micro_batch_size", args.micro_batch_size
        )
        llm_data_parallel_size = get_language_batch_parallel_size(args)
        if eval_global_batch_size % (eval_micro_batch_size * llm_data_parallel_size):
            raise ValueError(
                "--eval-global-batch-size must be divisible by --eval-micro-batch-size * "
                "language sample lanes"
            )
        if (eval_micro_batch_size * llm_data_parallel_size) % args.mimo_encoder_dp:
            raise ValueError(
                "MIMO validation requires --eval-micro-batch-size * language sample lanes "
                "divisible by --mimo-encoder-dp"
            )
        _, _, args.eval_iters = resolve_per_source_validation_schedule(
            input_tokens_per_source=args.input_tokens_per_source,
            sequence_length=args.seq_length,
            num_sources=len(per_source_validation.sources),
            eval_global_batch_size=eval_global_batch_size,
            requested_eval_iters=int(args.eval_iters or 0),
        )
    elif (
        int(getattr(args, "eval_iters", 0) or 0) > 0
        and int(getattr(args, "eval_interval", 0) or 0) > 0
    ):
        raise ValueError("MIMO Mistral data does not yet support validation iterators")
    args._mistral_per_source_validation_enabled = validation_enabled

    has_image_data = not config.skip_images and any(
        source.image_transform_type is not None for source in config.sources
    )
    if has_image_data:
        if getattr(args, "mimo_llm_only", False):
            raise ValueError("Mistral image data requires an active encoder grid")
        llm_data_parallel_size = get_language_batch_parallel_size(args)
        if (
            not image_sharding
            and (args.micro_batch_size * llm_data_parallel_size) % args.mimo_encoder_dp
        ):
            raise ValueError(
                "Mistral image data requires --micro-batch-size * language sample lanes "
                "divisible by --mimo-encoder-dp "
                f"(got {args.micro_batch_size} * {llm_data_parallel_size} "
                f"% {args.mimo_encoder_dp} != 0)"
            )
        _validate_image_transform_alignment(args, config)

    if config.img_id is not None:
        args.image_token_id = config.img_id
    args.dataloader_save = getattr(args, "save", None)
    args._mistral_pretokenized_config = config


def build_mistral_data_loaders(
    args: argparse.Namespace, topology: HeteroTopology
) -> tuple[object | None, object | None, None]:
    """Build role-specific Mistral train and validation iterators."""
    config = args._mistral_pretokenized_config
    encoder_name = _encoder_name(topology)

    language_grid = topology.grids[MIMO_LANGUAGE_MODULE_KEY]
    language_process_groups = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
    restored_state = _load_mistral_state(args, pipeline_parallel=args.mimo_llm_pp > 1)
    in_phase_transition = restored_state is not None and _in_phase_transition(args)
    # Every language PP stage needs the same packed-sequence boundaries. Metadata-only
    # iterators are deterministic for a shared DP/GTP data rank and do not load image pixels.
    language_needs_data = language_grid.is_current_rank_in_grid()

    encoder_needs_data = False
    encoder_process_groups = None
    if encoder_name is not None:
        encoder_process_groups = topology.module_pgs[encoder_name]
        encoder_needs_data = topology.grids[encoder_name].is_current_rank_in_grid() and (
            is_pp_first_stage(encoder_process_groups.pp)
        )

    if encoder_needs_data and language_needs_data:
        raise ValueError("the Mistral data adapter requires non-colocated module grids")
    llm_data_parallel_size = get_language_batch_parallel_size(args)
    if language_needs_data:
        language_data_parallel_rank = get_data_lane_rank(language_process_groups)
        special_token_ids = {encoder_name: config.img_id} if encoder_name is not None else {}
        iterator = _MistralDataIterator(
            args,
            config,
            rank=language_data_parallel_rank,
            world_size=llm_data_parallel_size,
            micro_batch_size=args.micro_batch_size,
            num_workers=args.num_workers,
            load_image_pixels=False,
            include_loss_modality=modality_metrics_requested(getattr(args, "tensor_metrics", ())),
            include_data_source_ids=data_source_metrics_requested(
                getattr(args, "tensor_metrics", ())
            ),
            translate_batch=partial(
                build_mistral_language_batch,
                special_token_ids=special_token_ids,
                residual_token_ids=(
                    _mistral_residual_token_ids(config)
                    if _residual_token_type_metrics_requested(getattr(args, "tensor_metrics", ()))
                    else None
                ),
            ),
            checkpoint_state_enabled=_is_checkpoint_lane(language_process_groups),
            restored_state=restored_state,
            in_phase_transition=in_phase_transition,
        )
        valid_iterator = _build_mistral_validation_iterator(
            args,
            config,
            rank=language_data_parallel_rank,
            world_size=llm_data_parallel_size,
            micro_batch_size=getattr(
                args, "eval_micro_batch_size", args.micro_batch_size
            ),
            load_image_pixels=False,
            transform_batch=partial(
                build_mistral_language_batch, special_token_ids=special_token_ids
            ),
            expose_source_metadata=True,
        )
        return (iterator, valid_iterator, None)
    if encoder_needs_data:
        encoder_dp_rank = get_pg_rank(encoder_process_groups.dp)
        image_shard = None
        encoder_data_rank, encoder_data_size = encoder_dp_rank, args.mimo_encoder_dp
        if getattr(args, "mimo_encoder_image_sharding", False):
            shards_per_sequence = args.mimo_encoder_dp // llm_data_parallel_size
            image_shard = (encoder_dp_rank % shards_per_sequence, shards_per_sequence)
            encoder_data_rank = encoder_dp_rank // shards_per_sequence
            encoder_data_size = llm_data_parallel_size
        encoder_raw_micro_batch_size = (
            args.micro_batch_size * llm_data_parallel_size // encoder_data_size
        )
        encoder_num_workers = args.mistral_encoder_num_workers
        if encoder_num_workers is None:
            encoder_num_workers = args.num_workers
        iterator = _MistralDataIterator(
            args,
            config,
            rank=encoder_data_rank,
            world_size=encoder_data_size,
            micro_batch_size=encoder_raw_micro_batch_size,
            num_workers=encoder_num_workers,
            load_image_pixels=True,
            translate_batch=partial(
                build_mistral_encoder_batch,
                encoder_name=encoder_name,
                image_token_id=config.img_id,
                pin_memory=torch.cuda.is_available(),
            ),
            checkpoint_state_enabled=False,
            restored_state=restored_state,
            in_phase_transition=in_phase_transition,
            max_image_group_workers=args.mistral_image_group_workers,
            image_shard=image_shard,
        )
        eval_micro_batch_size = getattr(
            args, "eval_micro_batch_size", args.micro_batch_size
        )
        encoder_eval_micro_batch_size = (
            eval_micro_batch_size * llm_data_parallel_size // args.mimo_encoder_dp
        )
        valid_iterator = _build_mistral_validation_iterator(
            args,
            config,
            rank=encoder_dp_rank,
            world_size=args.mimo_encoder_dp,
            micro_batch_size=encoder_eval_micro_batch_size,
            load_image_pixels=True,
            transform_batch=partial(
                build_mistral_encoder_batch,
                encoder_name=encoder_name,
                image_token_id=config.img_id,
            ),
            expose_source_metadata=False,
            max_image_group_workers=args.mistral_image_group_workers,
        )
        return (iterator, valid_iterator, None)
    return (None, None, None)


def _build_mistral_validation_iterator(
    args: argparse.Namespace,
    config: PretokenizedConfig,
    *,
    rank: int,
    world_size: int,
    micro_batch_size: int,
    load_image_pixels: bool,
    transform_batch: Callable[[dict[str, Any]], dict[str, Any]],
    expose_source_metadata: bool,
    max_image_group_workers: int = 1,
) -> object | None:
    if not getattr(args, "_mistral_per_source_validation_enabled", False):
        return None
    per_source_validation = config.per_source_validation
    assert per_source_validation is not None
    eval_num_microbatches = args.eval_global_batch_size // (
        micro_batch_size * world_size
    )
    iterator, eval_iters = build_per_source_validation_iterator(
        args=args,
        config=config,
        per_source_validation=per_source_validation,
        dp_rank=rank,
        dp_world_size=world_size,
        eval_global_batch_size=args.eval_global_batch_size,
        eval_micro_batch_size=micro_batch_size,
        eval_num_microbatches=eval_num_microbatches,
        load_image_pixels=load_image_pixels,
        max_image_group_workers=max_image_group_workers,
        transform_batch=transform_batch,
        expose_source_metadata=expose_source_metadata,
    )
    assert eval_iters == args.eval_iters, (eval_iters, args.eval_iters)
    return iterator


class _MistralDataIterator:
    """Load, translate, and close one native Mistral data stream."""

    # The committed state is the underlying loader's global cursor, identical on every data
    # rank, which is why _load_mistral_state restores every rank from dprank000.
    is_save_state_rank_independent = True

    def __init__(
        self,
        args: argparse.Namespace,
        config: PretokenizedConfig,
        *,
        rank: int,
        world_size: int,
        micro_batch_size: int,
        num_workers: int,
        load_image_pixels: bool,
        include_loss_modality: bool = False,
        translate_batch: Callable[[dict[str, Any]], dict[str, Any]],
        checkpoint_state_enabled: bool,
        restored_state: dict[str, Any] | None,
        in_phase_transition: bool = False,
        max_image_group_workers: int = 1,
        include_data_source_ids: bool = False,
        image_shard: tuple[int, int] | None = None,
    ) -> None:
        self._raw_iterator = MistralPretokenizedIterator(
            config=config,
            num_total_tokens=num_training_tokens_in_phase(args),
            rank=rank,
            world_size=world_size,
            micro_batch_size=micro_batch_size,
            num_workers=num_workers,
            allow_partial_epoch_index_cache=args.pretokenized_allow_partial_epoch_index_cache,
            in_phase_transition=in_phase_transition,
            load_image_pixels=load_image_pixels,
            include_loss_modality=include_loss_modality,
            max_image_group_workers=max_image_group_workers,
            include_data_source_ids=include_data_source_ids,
            preserve_logical_seqlens=(
                getattr(args, "experimental_attention_variant", None) == "ssa"
            ),
            image_shard=image_shard,
        )
        if restored_state is not None:
            self._raw_iterator.set_state(restored_state)
        self._translate_batch = translate_batch
        self._checkpoint_state_enabled = checkpoint_state_enabled
        self._committed_state = restored_state
        self._prefetch_executor = (
            ThreadPoolExecutor(max_workers=1) if num_workers > 0 else None
        )
        self._next_batch = None

    def __iter__(self) -> "_MistralDataIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        if self._prefetch_executor is None:
            batch, state = self._load_next_batch()
        else:
            batch, state = (
                self._load_next_batch() if self._next_batch is None else self._next_batch.result()
            )
        if self._prefetch_executor is not None:
            try:
                raw_batch, next_state = self._raw_iterator.next_with_state()
            except StopIteration:
                self._next_batch = None
            else:
                self._next_batch = self._prefetch_executor.submit(
                    self._translate_with_state, raw_batch, next_state
                )
        self._committed_state = state
        return batch

    def _load_next_batch(self) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_batch, state = self._raw_iterator.next_with_state()
        return self._translate_with_state(raw_batch, state)

    def _translate_with_state(
        self, raw_batch: dict[str, Any], state: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._translate_batch(raw_batch), state

    def save_state(self) -> dict[str, Any] | None:
        """Return the state after the last batch delivered to training."""
        return self._committed_state if self._checkpoint_state_enabled else None

    def close(self) -> None:
        """Close image files and any Mistral worker processes."""
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
        self._raw_iterator.close()


def _load_mistral_state(
    args: argparse.Namespace, *, pipeline_parallel: bool
) -> dict[str, Any] | None:
    if not getattr(args, "iteration", 0):
        return None

    state_path = get_checkpoint_name(
        args.load,
        args.iteration,
        pipeline_parallel=pipeline_parallel,
        tensor_rank=0,
        pipeline_rank=0,
        expert_parallel=False,
        expert_rank=0,
        basename="train_dataloader_dprank000.pt",
    )
    if not Path(state_path).is_file():
        raise RuntimeError(f"Mistral dataloader state not found at {state_path}")
    return torch.load(state_path, map_location="cpu", weights_only=False)["dataloader_state_dict"]


def _is_checkpoint_lane(process_groups: Any) -> bool:
    """Select one TP/CP/PP lane for each language DP rank."""
    return all(
        group is None or get_pg_rank(group) == 0
        for group in (
            getattr(process_groups, "tp", None),
            getattr(process_groups, "cp", None),
            getattr(process_groups, "pp", None),
        )
    )


def _residual_token_type_metrics_requested(specifications: tuple[str, ...] | list[str]) -> bool:
    """Return whether any configured residual metric requests token-type breakdowns."""
    for specification in specifications:
        name, separator, _ = specification.strip().rpartition(":")
        if (
            separator
            and name.endswith("-by-token-type")
            and name.startswith(("residual-", "dresidual-"))
        ):
            return True
    return False


def _mistral_residual_token_ids(
    config: PretokenizedConfig,
) -> dict[MimoTokenType, tuple[int | None, ...]]:
    """Map Mistral special-token IDs to residual token categories."""
    token_ids = {
        MimoTokenType.IMAGE: (config.img_id,),
        MimoTokenType.SPECIAL_IMAGE: (config.img_break_id, config.img_end_id),
        MimoTokenType.SPECIAL_TEXT: (config.bos_id, config.eos_id),
        MimoTokenType.PAD: (config.pad_id,),
    }
    categories_by_id: dict[int, MimoTokenType] = {}
    for token_type, category_ids in token_ids.items():
        for token_id in category_ids:
            if token_id is None:
                continue
            previous = categories_by_id.setdefault(token_id, token_type)
            if previous != token_type:
                raise ValueError(
                    f"Mistral token ID {token_id} belongs to both {previous.name} "
                    f"and {token_type.name}."
                )
    return token_ids


def build_mistral_language_batch(
    raw_batch: dict[str, Any],
    *,
    special_token_ids: dict[str, int] | None = None,
    residual_token_ids: dict[MimoTokenType, tuple[int | None, ...]] | None = None,
) -> dict[str, Any]:
    """Translate a native Mistral batch for the language module."""
    language_batch = {
        name: raw_batch[name] for name in ("tokens", "labels", "loss_mask", "position_ids")
    }
    if "loss_modality" in raw_batch:
        language_batch["loss_modality"] = raw_batch["loss_modality"]
    if "source_ids" in raw_batch:
        language_batch["source_ids"] = expand_source_ids_to_token_shape(
            raw_batch["source_ids"], raw_batch["loss_mask"]
        )

    if "cu_seqlens" in raw_batch:
        for name in ("cu_seqlens", "cu_seqlens_padded", "max_seqlen"):
            if name in raw_batch:
                language_batch[name] = raw_batch[name]
        flatten_batch_for_packed_sequences(language_batch)
    language_batch["input_ids"] = language_batch.pop("tokens")
    if residual_token_ids is not None:
        residual_token_types = torch.full_like(
            language_batch["input_ids"], int(MimoTokenType.TEXT), dtype=torch.int64
        )
        for token_type, token_ids in residual_token_ids.items():
            for token_id in token_ids:
                if token_id is not None:
                    residual_token_types[language_batch["input_ids"] == token_id] = int(token_type)
        language_batch["residual_token_types"] = residual_token_types
    if "data_source_ids" in raw_batch:
        language_batch["data_source_ids"] = raw_batch["data_source_ids"].reshape_as(
            language_batch["input_ids"]
        )
    flat_input_ids = language_batch["input_ids"].reshape(-1)
    image_token_mask = raw_batch.get("image_token_mask")
    if image_token_mask is not None:
        image_token_mask = image_token_mask.reshape(-1)
    text_mask = torch.ones_like(flat_input_ids, dtype=torch.bool)
    modality_token_indices: dict[str, torch.Tensor] = {}
    for modality_name, token_id in (special_token_ids or {}).items():
        modality_mask = (
            image_token_mask if image_token_mask is not None else flat_input_ids == token_id
        )
        modality_token_indices[modality_name] = modality_mask.nonzero(as_tuple=False).flatten()
        text_mask &= ~modality_mask
    modality_token_indices["text"] = text_mask.nonzero(as_tuple=False).flatten()
    language_batch["modality_token_indices"] = modality_token_indices
    return language_batch


def build_mistral_encoder_batch(
    raw_batch: dict[str, Any], *, encoder_name: str, image_token_id: int | None = None, pin_memory: bool = False
) -> dict[str, Any]:
    """Translate a native Mistral batch for the vision encoder module."""
    input_ids = raw_batch["tokens"]
    image_token_mask = raw_batch.get("image_token_mask")
    if image_token_mask is not None and image_token_id is not None:
        input_ids = input_ids.masked_fill(
            (input_ids == image_token_id) & ~image_token_mask,
            -1,
        )
    encoder_batch: dict[str, Any] = {"input_ids": input_ids}
    images = raw_batch.get("images") or []
    if images:
        encoder_inputs = _pack_native_images(images, pin_memory=pin_memory)
        encoder_batch["modality_inputs"] = {encoder_name: {encoder_name: encoder_inputs}}
    return encoder_batch


def _pack_native_images(
    images: list[torch.Tensor], *, pin_memory: bool = False
) -> dict[str, Any]:
    patch_size = DEFAULT_IMAGE_PATCH_SIZE
    image_sizes = [(int(image.shape[-2]), int(image.shape[-1])) for image in images]
    invalid_sizes = [size for size in image_sizes if size[0] % patch_size or size[1] % patch_size]
    if invalid_sizes:
        raise ValueError(
            f"image sizes must be divisible by patch_size={patch_size}, got {invalid_sizes}"
        )
    patch_grids = [
        (image_height // patch_size, image_width // patch_size)
        for image_height, image_width in image_sizes
    ]
    patch_counts = [patch_rows * patch_columns for patch_rows, patch_columns in patch_grids]

    num_channels = images[0].shape[0]
    packed_patches = images[0].new_empty(
        (sum(patch_counts), num_channels, patch_size, patch_size),
        pin_memory=pin_memory,
    )
    patch_offset = 0
    for image, (patch_rows, patch_columns), patch_count in zip(images, patch_grids, patch_counts):
        packed_patches[patch_offset : patch_offset + patch_count].view(
            patch_rows, patch_columns, num_channels, patch_size, patch_size
        ).copy_(
            image.reshape(num_channels, patch_rows, patch_size, patch_columns, patch_size).permute(
                1, 3, 0, 2, 4
            )
        )
        patch_offset += patch_count

    cumulative_patch_counts = torch.tensor(
        [0, *accumulate(patch_counts)], dtype=torch.int32, pin_memory=pin_memory
    )
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cumulative_patch_counts,
        cu_seqlens_kv=cumulative_patch_counts,
        max_seqlen_q=max(patch_counts),
        max_seqlen_kv=max(patch_counts),
    )
    return {
        "pixel_values": packed_patches.flatten(1).unsqueeze(0),
        "imgs_sizes": image_sizes,
        "packed_seq_params": packed_seq_params,
    }


def _validate_image_transform_alignment(
    args: argparse.Namespace, config: PretokenizedConfig
) -> None:
    transform_types = {
        ImageTransformType(source.image_transform_type)
        for source in config.sources
        if source.image_transform_type is not None
    }
    if ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES in transform_types:
        if args.model_provider != "nemotron-moe-mistral-vit":
            raise ValueError(
                "crawl_default_no_high_res requires " "--model-provider nemotron-moe-mistral-vit"
            )
        required_options = (
            ("patch_dim", 14, "--patch-dim 14"),
            ("spatial_merge_size", 2, "--spatial-merge-size 2"),
            ("dynamic_resolution", True, "--dynamic-resolution"),
            ("disable_vision_class_token", True, "--disable-vision-class-token"),
        )
        for attribute, expected, option in required_options:
            if getattr(args, attribute) != expected:
                raise ValueError(f"crawl_default_no_high_res requires {option}")


def _encoder_name(topology: HeteroTopology) -> str | None:
    encoder_names = [name for name in topology.grids if name != MIMO_LANGUAGE_MODULE_KEY]
    if len(encoder_names) > 1:
        raise ValueError("the Mistral data adapter supports one encoder module")
    return encoder_names[0] if encoder_names else None
