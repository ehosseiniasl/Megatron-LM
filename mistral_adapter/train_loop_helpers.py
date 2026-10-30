"""Train loop helpers for Mistral adapter data loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
import yaml

from megatron.core import parallel_state
from megatron.training import print_rank_0

if TYPE_CHECKING:
    from mistral_adapter.pretokenized_config import (
        PretokenizedConfig,
        PretokenizedValidationConfig,
        SourceConfig,
    )
    from mistral_adapter.pretokenized_iterator import MistralPretokenizedIterator


INTERDOCUMENT_MASKING_TP_MBS_SLACK_THREAD = "https://mistralai.slack.com/archives/C0ARXARV9V0/p1783061005464749?thread_ts=1782130365.109669&cid=C0ARXARV9V0"  # noqa: E501


class _ReplayableValidationIterator(Protocol):
    def __next__(self) -> dict[str, torch.Tensor]: ...

    def ensure_started(self) -> None: ...

    def save_state(self) -> dict[str, Any] | None: ...

    def set_state(self, data: dict[str, Any] | None) -> None: ...

    def close(self) -> None: ...


def add_pretokenized_args(parser: Any) -> Any:
    """Add Mistral pretokenized dataloader arguments."""
    group = parser.add_argument_group(title="mistral pretokenized data")
    group.add_argument(
        "--pretokenized-config",
        type=str,
        default=None,
        help="Path to YAML config for Mistral pretokenized data. "
        "When set, uses Mistral's PretokenizedDataLoader instead of "
        "Megatron's built-in data pipeline. Requires --dataloader-type external.",
    )
    group.add_argument(
        "--input-tokens-per-source",
        type=int,
        default=131072,
        help=(
            "Input-token budget for each per-source validation dataset. Must be "
            "a multiple of --seq-length. Set to 0 when supplying --eval-iters."
        ),
    )
    group.add_argument(
        "--per-source-validation-top-n",
        type=int,
        default=0,
        help=(
            "Evaluate only the N highest-weight per-source validation datasets. "
            "Set to 0 to evaluate every configured source."
        ),
    )
    group.add_argument(
        "--overwrite-percentage-data-multiplier",
        type=float,
        default=None,
        help="Multiply every source's percentage_data by this factor before "
        "building the train and validation iterators. Used to subsample the "
        "mixture (e.g. 0.0001 for debug smoke tests). Must match the multiplier "
        "the epoch indexes were built with.",
    )
    group.add_argument(
        "--pretokenized-allow-build-of-epoch-index",
        action="store_true",
        default=False,
        help="Build missing pretokenized epoch indexes inside the training job.",
    )
    group.add_argument(
        "--pretokenized-allow-partial-epoch-index-cache",
        action="store_true",
        default=False,
        help=(
            "Allow training to start with a contiguous partial epoch-index cache. "
            "Missing epoch indexes still fail immediately when reached."
        ),
    )
    group.add_argument(
        "--pretokenized-build-of-epoch-index-timeout-s",
        type=int,
        default=3600,
        help=(
            "Timeout in seconds for the NFS barrier after distributed pretokenized "
            "epoch-index build."
        ),
    )
    return parser


def _sequence_aux_loss_enabled(args: Any) -> bool:
    load_balancing_type = getattr(args, "moe_router_load_balancing_type", "none")
    aux_loss_coeff = getattr(args, "moe_aux_loss_coeff", 0.0)

    if isinstance(load_balancing_type, str):
        if load_balancing_type != "seq_aux_loss":
            return False
        if isinstance(aux_loss_coeff, list):
            aux_loss_coeff = aux_loss_coeff[0] if aux_loss_coeff else 0.0
        return float(aux_loss_coeff) > 0

    if not isinstance(load_balancing_type, list):
        return False

    if isinstance(aux_loss_coeff, list):
        coeffs = aux_loss_coeff
    else:
        coeffs = [aux_loss_coeff] * len(load_balancing_type)
    return any(
        loss_type == "seq_aux_loss" and float(coeff) > 0
        for loss_type, coeff in zip(load_balancing_type, coeffs)
    )


def _validate_interdocument_masking_args(args: Any, config: PretokenizedConfig) -> None:
    assert args.dataloader_inter_document_masking == config.interdocument_masking, (
        "The pretokenized config and --dataloader-inter-document-masking must agree: "
        f"config.interdocument_masking={config.interdocument_masking}, "
        f"--dataloader-inter-document-masking="
        f"{args.dataloader_inter_document_masking}."
    )


def _assert_seq_aux_loss_interdocument_masking_supported(
    *,
    tensor_model_parallel_size: int,
    micro_batch_size: int,
) -> None:
    assert tensor_model_parallel_size == 1 or micro_batch_size == 1, (
        "Megatron seq_aux_loss with interdocument masking currently has a known "
        "bug with tensor parallelism > 1 and micro_batch_size > 1. The TP > 1 "
        "xor micro_batch_size > 1 cases are allowed. "
        f"Got TP={tensor_model_parallel_size}, micro_batch_size={micro_batch_size}. "
        f"See {INTERDOCUMENT_MASKING_TP_MBS_SLACK_THREAD}."
    )


def _assert_config_version(config: PretokenizedConfig) -> None:
    from mistral_adapter.pretokenized_config import CONFIG_VERSION

    assert config.version == CONFIG_VERSION, (
        f"Pretokenized config version mismatch: file has v{config.version}, "
        f"but this Megatron build expects v{CONFIG_VERSION}. "
        f"Regenerate the config with a matching Mistral launcher."
    )


def _validate_pretokenized_training_args(
    args: Any, config: PretokenizedConfig, *, tensor_model_parallel_size: int | None = None
) -> None:
    tensor_model_parallel_size = (
        args.tensor_model_parallel_size
        if tensor_model_parallel_size is None
        else tensor_model_parallel_size
    )
    _validate_interdocument_masking_args(args, config)
    if config.interdocument_masking and _sequence_aux_loss_enabled(args):
        _assert_seq_aux_loss_interdocument_masking_supported(
            tensor_model_parallel_size=tensor_model_parallel_size,
            micro_batch_size=args.micro_batch_size,
        )
    assert args.seq_length == config.original_seq_len, (
        f"--seq-length must match the pretokenized config's original_seq_len: "
        f"expected {config.original_seq_len}, got {args.seq_length}"
    )


def num_training_tokens_in_phase(args: Any) -> int:
    """Return the token horizon remaining from the initial or restored training state."""
    if getattr(args, "train_samples", None) is not None:
        consumed_train_samples = int(args.consumed_train_samples)
        train_samples = int(args.train_samples)
        phase_transition_samples = getattr(args, "phase_transition_samples", None)
        assert 0 <= consumed_train_samples <= train_samples, (
            "--consumed-train-samples must be between zero and --train-samples, got "
            f"{consumed_train_samples} and {train_samples}"
        )
        if phase_transition_samples is not None:
            phase_transition_samples = int(phase_transition_samples)
            assert phase_transition_samples <= train_samples, (
                "--phase-transition-samples must not exceed --train-samples"
            )
        phase_end_samples = (
            phase_transition_samples
            if phase_transition_samples is not None
            and consumed_train_samples < phase_transition_samples
            else train_samples
        )
        return (phase_end_samples - consumed_train_samples) * int(args.seq_length)
    return int(args.train_iters) * int(args.global_batch_size) * int(args.seq_length)


def _fixed_overlap_eval_seq_len_from_fields(
    *, fixed_overlap: int, fixed_overlap_eval_seq_len: int | None, original_seq_len: int
) -> int:
    if fixed_overlap_eval_seq_len is not None:
        assert fixed_overlap_eval_seq_len > 1, (
            "fixed_overlap_eval_seq_len must be > 1, got " f"{fixed_overlap_eval_seq_len}"
        )

    seq_len = (
        fixed_overlap_eval_seq_len if fixed_overlap_eval_seq_len is not None else original_seq_len
    )
    assert seq_len > 1, f"Expected fixed-overlap eval seq len > 1, got {seq_len}"
    assert 0 <= fixed_overlap < seq_len, (
        "Expected fixed_overlap in [0, fixed_overlap_eval_seq_len), got "
        f"fixed_overlap={fixed_overlap}, "
        f"fixed_overlap_eval_seq_len={seq_len}"
    )
    return seq_len


def build_pretokenized_iterator(args: Any) -> MistralPretokenizedIterator:
    """Build a pretokenized iterator from a PretokenizedConfig YAML file.

    The YAML is produced by the Mistral launcher and validated against the
    PretokenizedConfig Pydantic model defined in mistral_adapter/pretokenized_config.py.
    """
    from mistral_adapter.pretokenized_config import PretokenizedConfig
    from mistral_adapter.pretokenized_config_utils import apply_pretokenized_config_overrides
    from mistral_adapter.pretokenized_iterator import MistralPretokenizedIterator

    with open(args.pretokenized_config) as f:
        raw = yaml.safe_load(f)

    config = PretokenizedConfig.model_validate(raw)
    _assert_config_version(config)

    config = apply_pretokenized_config_overrides(
        config,
        percentage_data_multiplier=getattr(args, "overwrite_percentage_data_multiplier", None),
    )

    _validate_pretokenized_training_args(args, config)

    if args.pretokenized_allow_build_of_epoch_index:
        percentage_data_multiplier = getattr(args, "overwrite_percentage_data_multiplier", None)
        assert percentage_data_multiplier is not None and percentage_data_multiplier < 1, (
            "--pretokenized-allow-build-of-epoch-index only supports building epoch "
            "indexes on the fly for subsampled mixtures "
            "(--overwrite-percentage-data-multiplier < 1), got "
            f"{percentage_data_multiplier}. On the full data this is far too slow; "
            "prebuild the indexes beforehand with "
            "`python -m mistral_fork.mistral.data.pretokenized.processing."
            "prebuild_epoch_indexes_megatron prebuild ...`."
        )

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()
    iterator = MistralPretokenizedIterator(
        config=config,
        num_total_tokens=num_training_tokens_in_phase(args),
        rank=dp_rank,
        world_size=dp_world_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        allow_build_of_epoch_index=args.pretokenized_allow_build_of_epoch_index,
        allow_partial_epoch_index_cache=args.pretokenized_allow_partial_epoch_index_cache,
        nfs_barrier_dir=Path(args.save) / "barriers",
        epoch_index_build_timeout_s=args.pretokenized_build_of_epoch_index_timeout_s,
        preserve_logical_seqlens=getattr(args, "experimental_attention_variant", None) == "ssa",
    )

    # Restore dataloader state from checkpoint if available.
    _maybe_restore_pretokenized_state(args, iterator, dp_rank)

    print_rank_0(
        f"> built pretokenized iterator (forked): "
        f"{len(config.sources)} sources, dp_rank={dp_rank}/{dp_world_size}, "
        f"config={config}, "
        f"num_workers={args.num_workers}"
    )
    return iterator


class _FixedValidationIterator:
    """Replays the same validation batches on every evaluation round.

    Megatron's evaluate() pulls a fixed number of batches (reset_period) from this
    same object each time it runs. We snapshot the underlying loader's initial
    (global_sequence_id=0) state once and, after each full round, rewind to it via
    set_state instead of rebuilding the loader. This keeps the single-process
    loader and its open datasources/epoch-index alive across rounds while still
    serving identical batches, giving a fixed validation set with comparable
    loss/PPL across training steps.

    Requires an inner iterator built with num_workers=0 and
    allow_state_resetting=True (so set_state may be called repeatedly).
    """

    def __init__(
        self,
        inner: _ReplayableValidationIterator,
        eval_iters: int,
        reset_period: int,
        expected_eval_tokens: int | None = None,
    ) -> None:
        assert eval_iters >= 1, eval_iters
        assert reset_period >= 1, reset_period
        if expected_eval_tokens is not None:
            expected_eval_tokens = int(expected_eval_tokens)
            assert expected_eval_tokens >= 0, (
                "Expected validation token count must be non-negative, got "
                f"{expected_eval_tokens=}"
            )
        self._inner = inner
        self._eval_iters = eval_iters
        self._reset_period = reset_period
        self._expected_eval_tokens = expected_eval_tokens
        # Snapshot the initial state before consuming any batch so every round
        # can rewind to it.
        self._inner.ensure_started()
        self._initial_state = self._inner.save_state()
        assert (
            self._initial_state is not None
        ), "failed to snapshot initial validation dataloader state"
        self._count = 0

    def __iter__(self) -> "_FixedValidationIterator":
        return self

    def __len__(self) -> int:
        return self._eval_iters

    @property
    def expected_eval_tokens(self) -> int | None:
        return self._expected_eval_tokens

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._count > 0 and self._count % self._reset_period == 0:
            self._inner.set_state(self._initial_state)
        self._count += 1
        return next(self._inner)

    def close(self) -> None:
        self._inner.close()


def _eval_batching(args: Any, dp_world_size: int) -> tuple[int, int, int]:
    eval_global_batch_size = getattr(args, "eval_global_batch_size", args.global_batch_size)
    eval_micro_batch_size = getattr(args, "eval_micro_batch_size", args.micro_batch_size)
    assert eval_global_batch_size % (eval_micro_batch_size * dp_world_size) == 0, (
        eval_global_batch_size,
        eval_micro_batch_size,
        dp_world_size,
    )
    eval_num_microbatches = eval_global_batch_size // (eval_micro_batch_size * dp_world_size)
    return eval_global_batch_size, eval_micro_batch_size, eval_num_microbatches


def _build_fixed_overlap_validation_iterator(
    *,
    args: Any,
    config: PretokenizedConfig,
    name: str,
    sources: list[SourceConfig],
    fixed_overlap: int,
    fixed_overlap_eval_seq_len: int | None,
    dp_rank: int,
    dp_world_size: int,
    eval_global_batch_size: int,
    eval_micro_batch_size: int,
    eval_num_microbatches: int,
) -> tuple[_FixedValidationIterator, int]:
    from mistral_adapter.pretokenized_config_utils import apply_pretokenized_config_overrides
    from mistral_adapter.pretokenized_fixed_overlap_iterator import (
        MistralPretokenizedFixedOverlapIterator,
    )

    fixed_overlap_seq_len = _fixed_overlap_eval_seq_len_from_fields(
        fixed_overlap=fixed_overlap,
        fixed_overlap_eval_seq_len=fixed_overlap_eval_seq_len,
        original_seq_len=config.original_seq_len,
    )
    assert args.seq_length >= fixed_overlap_seq_len - 1, (
        "Fixed-overlap validation requires --seq-length to fit at least one "
        f"eval window's logprobs, got args.seq_length={args.seq_length} "
        f"and fixed_overlap_eval_seq_len={fixed_overlap_seq_len}"
    )

    valid_config = config.model_copy(update={"sources": sources})
    valid_config = apply_pretokenized_config_overrides(
        valid_config,
        percentage_data_multiplier=getattr(args, "overwrite_percentage_data_multiplier", None),
    )

    valid_iterator = MistralPretokenizedFixedOverlapIterator(
        config=valid_config,
        rank=dp_rank,
        world_size=dp_world_size,
        micro_batch_size=eval_micro_batch_size,
        window_seq_len=fixed_overlap_seq_len,
        overlap=fixed_overlap,
        batch_seq_len=args.seq_length,
        preserve_logical_seqlens=getattr(args, "experimental_attention_variant", None) == "ssa",
    )
    eval_iters = (
        valid_iterator.num_global_microbatches + eval_num_microbatches - 1
    ) // eval_num_microbatches
    reset_period = eval_iters * eval_num_microbatches
    iterator = _FixedValidationIterator(
        inner=valid_iterator,
        eval_iters=eval_iters,
        reset_period=reset_period,
        expected_eval_tokens=valid_iterator.num_scored_tokens,
    )
    print_rank_0(
        "> built fixed-overlap pretokenized perplexity eval iterator: "
        f"name={name}, sources={len(sources)}, "
        f"windows={valid_iterator.num_windows}, "
        f"scored_tokens={valid_iterator.num_scored_tokens}, "
        f"overlap={fixed_overlap}, eval_iters={eval_iters}, "
        f"fixed_overlap_seq_len={fixed_overlap_seq_len}, "
        f"fixed_overlap_eval_seq_len={fixed_overlap_eval_seq_len}, "
        f"batch_seq_len={args.seq_length}, "
        f"eval_global_batch_size={eval_global_batch_size}, "
        f"reset_period={reset_period}, dp_rank={dp_rank}/{dp_world_size}"
    )
    return iterator, eval_iters


def _build_trainlike_validation_iterator(
    *,
    args: Any,
    config: PretokenizedConfig,
    validation_set: PretokenizedValidationConfig,
    dp_rank: int,
    dp_world_size: int,
    eval_global_batch_size: int,
    eval_micro_batch_size: int,
    eval_num_microbatches: int,
) -> tuple[_FixedValidationIterator, int]:
    from mistral_adapter.pretokenized_config_utils import apply_pretokenized_config_overrides
    from mistral_adapter.pretokenized_iterator import MistralPretokenizedIterator

    tokens_per_eval_step = eval_global_batch_size * args.seq_length
    eval_iters = validation_set.valid_num_tokens // tokens_per_eval_step
    assert eval_iters >= 1, (
        f"valid_num_tokens={validation_set.valid_num_tokens} is too small for one eval step "
        f"of {tokens_per_eval_step} tokens (eval_global_batch_size * seq_length)"
    )

    # Per-rank next() calls consumed by one evaluate() pass.
    reset_period = eval_iters * eval_num_microbatches

    valid_seed = validation_set.valid_seed
    valid_config = config.model_copy(
        update={"sources": validation_set.valid_sources, "seed": valid_seed}
    )
    valid_config = apply_pretokenized_config_overrides(
        valid_config,
        percentage_data_multiplier=getattr(args, "overwrite_percentage_data_multiplier", None),
    )
    valid_total_tokens = eval_iters * eval_global_batch_size * args.seq_length

    # num_workers=0 keeps a single in-process loader so the fixed validation
    # iterator can rewind it in place via set_state each round (no subprocess
    # respawn, no epoch-index reload). allow_state_resetting permits the repeated
    # set_state calls that rewind performs.
    valid_iterator = MistralPretokenizedIterator(
        config=valid_config,
        num_total_tokens=valid_total_tokens,
        rank=dp_rank,
        world_size=dp_world_size,
        micro_batch_size=eval_micro_batch_size,
        num_workers=0,
        allow_state_resetting=True,
        allow_build_of_epoch_index=False,
        preserve_logical_seqlens=getattr(args, "experimental_attention_variant", None) == "ssa",
    )

    iterator = _FixedValidationIterator(
        inner=valid_iterator, eval_iters=eval_iters, reset_period=reset_period
    )
    print_rank_0(
        f"> built fixed pretokenized validation iterator: "
        f"{len(validation_set.valid_sources)} sources, eval_iters={eval_iters}, "
        f"eval_global_batch_size={eval_global_batch_size}, reset_period={reset_period}, "
        f"valid_total_tokens={valid_total_tokens}, valid_seed={valid_seed}, "
        f"dp_rank={dp_rank}/{dp_world_size}"
    )
    return iterator, eval_iters


def build_pretokenized_valid_iterator(
    args: Any,
) -> _FixedValidationIterator | list[_FixedValidationIterator] | None:
    """Build validation and document-PPL iterators from the pretokenized config.

    ``validation_set`` builds the normal pretokenized validation stream, matching
    training batch semantics. ``perplexity_eval_sets`` builds named document-local
    fixed-overlap PPL streams.
    """
    from mistral_adapter.pretokenized_config import PretokenizedConfig

    with open(args.pretokenized_config) as f:
        raw = yaml.safe_load(f)
    config = PretokenizedConfig.model_validate(raw)
    _assert_config_version(config)

    validation_set = config.validation_set
    has_validation_set = validation_set is not None and bool(validation_set.valid_sources)
    has_perplexity_eval_sets = bool(config.perplexity_eval_sets)
    if not has_validation_set and not has_perplexity_eval_sets:
        print_rank_0("> no validation_set or perplexity_eval_sets in pretokenized config")
        return None

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()
    eval_global_batch_size, eval_micro_batch_size, eval_num_microbatches = _eval_batching(
        args, dp_world_size
    )
    _validate_interdocument_masking_args(args, config)
    if _sequence_aux_loss_enabled(args):
        _assert_seq_aux_loss_interdocument_masking_supported(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            micro_batch_size=eval_micro_batch_size,
        )
    assert args.seq_length == config.original_seq_len, (
        f"--seq-length must match the pretokenized config's original_seq_len: "
        f"expected {config.original_seq_len}, got {args.seq_length}"
    )

    named_iterators: list[tuple[str, _FixedValidationIterator, int]] = []

    if has_validation_set:
        iterator, eval_iters = _build_trainlike_validation_iterator(
            args=args,
            config=config,
            validation_set=validation_set,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            eval_global_batch_size=eval_global_batch_size,
            eval_micro_batch_size=eval_micro_batch_size,
            eval_num_microbatches=eval_num_microbatches,
        )
        named_iterators.append(("trainlike", iterator, eval_iters))
    elif has_perplexity_eval_sets:
        print_rank_0("> no train-like validation_set in pretokenized config")

    for eval_set in config.perplexity_eval_sets:
        iterator, eval_iters = _build_fixed_overlap_validation_iterator(
            args=args,
            config=config,
            name=eval_set.name,
            sources=eval_set.sources,
            fixed_overlap=eval_set.fixed_overlap,
            fixed_overlap_eval_seq_len=eval_set.fixed_overlap_eval_seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            eval_global_batch_size=eval_global_batch_size,
            eval_micro_batch_size=eval_micro_batch_size,
            eval_num_microbatches=eval_num_microbatches,
        )
        named_iterators.append((eval_set.name, iterator, eval_iters))

    names = [name for name, _, _ in named_iterators]
    assert len(set(names)) == len(
        names
    ), f"validation/perplexity eval names must be unique, got {names}"

    if len(named_iterators) == 1:
        name, iterator, eval_iters = named_iterators[0]
        args.multiple_validation_sets = False
        args.validation_set_names = None
        args.eval_iters = eval_iters
        print_rank_0(f"> configured single pretokenized validation iterator: name={name}")
        return iterator

    args.multiple_validation_sets = True
    args.validation_set_names = names
    args.eval_iters = [eval_iters for _, _, eval_iters in named_iterators]
    print_rank_0(
        "> configured multiple pretokenized validation iterators: "
        f"names={args.validation_set_names}, eval_iters={args.eval_iters}"
    )
    return [iterator for _, iterator, _ in named_iterators]


def _maybe_restore_pretokenized_state(
    args: Any, iterator: MistralPretokenizedIterator, dp_rank: int
) -> None:
    """Restore pretokenized dataloader state from a --dataloader-save checkpoint."""
    from megatron.training.checkpointing import get_checkpoint_name

    dataloader_save = args.dataloader_save
    assert dataloader_save is not None, "please add dataloader save"
    if not args.iteration:
        print_rank_0("first iteration, no reloading")
        return

    # Same rule as maybe_save_dataloader_state: a state that does not depend on the rank is
    # written once, as dprank000, and every rank restores from it.
    state_file_rank = 0 if iterator.is_save_state_rank_independent else dp_rank
    data_state_path = get_checkpoint_name(
        dataloader_save,
        args.iteration,
        pipeline_rank=0,
        tensor_rank=0,
        expert_parallel=False,
        basename=f"train_dataloader_dprank{state_file_rank:03d}.pt",
    )

    if not os.path.exists(data_state_path):
        raise RuntimeError(f"> no dataloader state found at {data_state_path}, ")

    state_dict = torch.load(data_state_path, map_location="cpu", weights_only=False)
    iterator.set_state(state_dict["dataloader_state_dict"])
    print_rank_0(f"> restored pretokenized dataloader state from {data_state_path}")
