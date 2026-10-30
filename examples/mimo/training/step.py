# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Forward step and per-token loss for MIMO training."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial

import torch

from examples.mimo.training.batch import move_batch_to_cuda
from examples.mimo.training.encoder_prefetch import PREFETCHED_FEATURES_KEY, PROJECTION_TIMER_KEY
from megatron.core.models.mimo.token_types import MIMO_TOKEN_TYPE_NAMES
from megatron.core.tensor_observation import capture_tensor_token_types, is_observing_tensor
from megatron.core.utils import unwrap_model
from megatron.training.tensor_metrics.data_source_loss import observe_data_source_loss
from megatron.training.tensor_metrics.data_source_metrics import observe_modality_loss
from mistral_adapter.per_source_validation import (
    PER_SOURCE_LOSS_KEY,
    PER_SOURCE_NUM_SOURCES_KEY,
    build_per_source_loss_report,
)

_RESIDUAL_TOKEN_TYPE_SOURCE_KINDS = (
    "residual_summary_by_token_type",
    "dresidual_summary_by_token_type",
)


def loss_func(
    output_tensor: torch.Tensor,
    *,
    loss_mask: torch.Tensor,
    model: torch.nn.Module | None = None,
    loss_modality: torch.Tensor | None = None,
    source_ids: torch.Tensor | None = None,
    num_sources: int | None = None,
    data_source_ids: torch.Tensor | None = None,
):
    """Return summed per-token loss, integer local token count, and logging tensors."""
    if not isinstance(output_tensor, torch.Tensor):
        raise TypeError(
            "loss_func expects the terminal language stage to return a per-token loss tensor, "
            f"got {type(output_tensor).__name__}"
        )

    if not isinstance(loss_mask, torch.Tensor) or output_tensor.shape != loss_mask.shape:
        raise RuntimeError(
            "MIMO per-token loss requires a loss_mask with the same shape as the model output"
        )

    output = output_tensor.float()
    mask = loss_mask.float()
    if is_observing_tensor("modality_loss"):
        if model is None or loss_modality is None:
            raise ValueError(
                "Loss-by-modality observation requires a model and token-aligned modality IDs."
            )
        observe_modality_loss(model, output, mask, loss_modality)
    if is_observing_tensor("data_source_loss"):
        if model is None or data_source_ids is None:
            raise ValueError(
                "Train-loss-by-data-source requires a model and token-aligned source IDs."
            )
        observe_data_source_loss(model, output, mask, data_source_ids)
    masked = output * mask
    num_tokens = mask.sum().to(torch.int)
    loss_sum = masked.sum()
    report = {"lm loss": torch.stack((loss_sum.detach(), num_tokens.detach().float()))}
    per_source_mode = num_sources is not None
    if per_source_mode != (source_ids is not None):
        raise RuntimeError(
            "source_ids and num_sources must either both be present or both be absent"
        )
    if per_source_mode:
        assert source_ids is not None and num_sources is not None
        report[PER_SOURCE_LOSS_KEY] = build_per_source_loss_report(
            losses=output,
            loss_mask=mask,
            source_ids=source_ids,
            num_sources=num_sources,
        )
    return (
        loss_sum,
        num_tokens,
        report,
    )


def mimo_forward_step(data_iterator, model):
    """Run a raw-input or prefetched-feature MIMO microbatch for the pipeline schedule.

    On the last pipeline stage, the schedule passes ``output_tensor`` to the returned loss closure.
    """
    batch = next(data_iterator) if data_iterator is not None else {"input_ids": None}
    source_ids = batch.pop("source_ids", None)
    num_sources = batch.pop(PER_SOURCE_NUM_SOURCES_KEY, None)
    prefetched = batch.pop(PREFETCHED_FEATURES_KEY, None)
    projection_timer = batch.pop(PROJECTION_TIMER_KEY, None)

    if prefetched is None:
        if projection_timer is not None:
            raise RuntimeError("encoder prefetch timer has no prefetched features")
        batch = move_batch_to_cuda(batch)
        source_ids = move_batch_to_cuda(source_ids)
        data_source_ids = batch.pop("data_source_ids", None)
        packed_metadata = {
            name: batch.get(name) for name in ("cu_seqlens", "cu_seqlens_padded", "max_seqlen")
        }
        loss_modality = batch.pop("loss_modality", None)
        if loss_modality is not None and is_observing_tensor("modality_loss"):
            loss_modality = unwrap_model(model).shard_language_token_metadata(
                loss_modality, **packed_metadata
            )
        residual_token_types = batch.pop("residual_token_types", None)
        token_type_context = nullcontext()
        observe_residual_token_types = any(
            is_observing_tensor(source_kind) for source_kind in _RESIDUAL_TOKEN_TYPE_SOURCE_KINDS
        )
        if observe_residual_token_types and unwrap_model(model).language_model is not None:
            if residual_token_types is None:
                raise ValueError(
                    "Residual metrics by token type require token metadata from the data provider."
                )
            residual_token_types = unwrap_model(model).shard_language_residual_token_types(
                residual_token_types, **packed_metadata
            )
            token_type_context = capture_tensor_token_types(
                residual_token_types, MIMO_TOKEN_TYPE_NAMES
            )
        with token_type_context:
            output_tensor, loss_mask = model(**batch)
        return output_tensor, partial(
            loss_func,
            loss_mask=loss_mask,
            model=model,
            loss_modality=loss_modality,
            source_ids=source_ids,
            num_sources=num_sources,
            data_source_ids=data_source_ids,
        )

    if batch.get("modality_inputs"):
        raise ValueError("prefetched features cannot be combined with raw modality inputs")

    projection_context = projection_timer if projection_timer is not None else nullcontext()
    with projection_context:
        output_tensor = model._forward_encoders(
            batch.get("input_ids"), modality_inputs=None, input_tensors=prefetched
        )
    # Encoder ranks never evaluate the language-model loss closure.
    return output_tensor, partial(loss_func, loss_mask=None)
