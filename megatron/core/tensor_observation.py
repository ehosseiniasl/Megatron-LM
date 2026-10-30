# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scoped notifications for observing short-lived tensors in Megatron Core."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

TensorObservationCallback = Callable[[object, str, str, torch.Tensor, int | None], None]
_TensorObservationContext = tuple[TensorObservationCallback, frozenset[str]] | None


@dataclass(frozen=True)
class _TokenTypeContext:
    tensor: torch.Tensor
    names: tuple[str, ...]


_CapturedTensorObservationContext = tuple[_TensorObservationContext, _TokenTypeContext | None]


_ACTIVE_OBSERVER: ContextVar[tuple[TensorObservationCallback, frozenset[str]] | None] = ContextVar(
    "tensor_observation", default=None
)
_ACTIVE_TOKEN_TYPES: ContextVar[_TokenTypeContext | None] = ContextVar(
    "tensor_observation_token_types", default=None
)
_OBSERVATION_SUSPENDED: ContextVar[bool] = ContextVar("tensor_observation_suspended", default=False)
_REPLAYING_GRADIENT_OBSERVATIONS: ContextVar[bool] = ContextVar(
    "replaying_gradient_tensor_observations", default=False
)
_SEMANTIC_PARTITION_NAMES_ATTR = "_tensor_metric_semantic_partition_names"
_SEMANTIC_PARTITION_SIZES_ATTR = "_tensor_metric_semantic_partition_sizes"
_SEMANTIC_PARTITION_CATALOG_ATTR = "_tensor_metric_semantic_partition_catalog"


def set_tensor_metric_semantic_partitions(
    tensor: torch.Tensor,
    names: Sequence[str],
    sizes: Sequence[int],
    *,
    all_names: Sequence[str] | None = None,
) -> None:
    """Describe contiguous semantic row partitions within a fused tensor.

    Sizes describe the complete TP-local tensor before any additional GTP or optimizer sharding.
    Repeated names are allowed for interleaved layouts such as grouped-query QKV.

    Args:
        tensor: Parameter whose first dimension contains fused semantic components.
        names: Component name corresponding to each contiguous row block.
        sizes: Positive row count for each block.
        all_names: Rank-symmetric component order, including components absent from this shard.
            Defaults to first-occurrence order in ``names``.
    """
    names = tuple(names)
    sizes = tuple(sizes)
    if not names or len(names) != len(sizes):
        raise ValueError("Semantic tensor partitions require equally sized name and size lists.")
    if any(not name for name in names):
        raise ValueError("Semantic tensor partition names must not be empty.")
    if any(size <= 0 for size in sizes):
        raise ValueError("Semantic tensor partition sizes must be positive.")
    all_names = tuple(dict.fromkeys(names)) if all_names is None else tuple(all_names)
    if not all_names or len(all_names) != len(set(all_names)):
        raise ValueError("Semantic tensor partition catalog names must be nonempty and unique.")
    if any(name not in all_names for name in names):
        raise ValueError("Every semantic tensor partition must appear in its catalog.")
    setattr(tensor, _SEMANTIC_PARTITION_NAMES_ATTR, names)
    setattr(tensor, _SEMANTIC_PARTITION_SIZES_ATTR, sizes)
    setattr(tensor, _SEMANTIC_PARTITION_CATALOG_ATTR, all_names)


def get_tensor_metric_semantic_partitions(
    tensor: torch.Tensor,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]] | None:
    """Return semantic fused-tensor row partitions, if the model supplied them."""
    names = getattr(tensor, _SEMANTIC_PARTITION_NAMES_ATTR, None)
    sizes = getattr(tensor, _SEMANTIC_PARTITION_SIZES_ATTR, None)
    all_names = getattr(tensor, _SEMANTIC_PARTITION_CATALOG_ATTR, None)
    if names is None and sizes is None and all_names is None:
        return None
    if names is None or sizes is None or all_names is None:
        raise ValueError("A fused tensor must define complete semantic partition metadata.")
    return tuple(names), tuple(sizes), tuple(all_names)


@contextmanager
def capture_tensor_observations(
    observer: TensorObservationCallback, source_kinds: frozenset[str]
) -> Iterator[None]:
    """Notify ``observer`` about the selected tensor source kinds within this scope.

    Args:
        observer: Callback receiving the owner, local name, source kind, tensor, and optional
            tensor-parallel shard dimension of each observation.
        source_kinds: Source kinds to notify. Other observation sites remain no-ops.
    """
    token = _ACTIVE_OBSERVER.set((observer, source_kinds))
    try:
        yield
    finally:
        _ACTIVE_OBSERVER.reset(token)


@contextmanager
def suspend_tensor_observations() -> Iterator[None]:
    """Temporarily suppress notifications to the active tensor observer.

    Activation checkpoint implementations use this around their backward recomputation so the
    original forward is observed exactly once.
    """
    token = _OBSERVATION_SUSPENDED.set(True)
    try:
        yield
    finally:
        _OBSERVATION_SUSPENDED.reset(token)


@contextmanager
def capture_tensor_token_types(token_types: torch.Tensor, names: Sequence[str]) -> Iterator[None]:
    """Associate token-type IDs with residual observations in this forward scope.

    Args:
        token_types: Integer IDs aligned with every non-hidden dimension of the residual stream.
        names: Stable label for each integer ID, in ID order.
    """
    names = tuple(names)
    if token_types.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise TypeError("Tensor observation token types must use an integer dtype.")
    if not names or any(not name or "]" in name for name in names):
        raise ValueError("Tensor observation token-type names must be nonempty and exclude ']'.")
    if len(names) != len(set(names)):
        raise ValueError("Tensor observation token-type names must be unique.")
    token = _ACTIVE_TOKEN_TYPES.set(_TokenTypeContext(token_types.detach(), names))
    try:
        yield
    finally:
        _ACTIVE_TOKEN_TYPES.reset(token)


def get_tensor_observation_context() -> _CapturedTensorObservationContext:
    """Capture observer and token metadata contexts for activation recomputation."""
    return _ACTIVE_OBSERVER.get(), _ACTIVE_TOKEN_TYPES.get()


@contextmanager
def replay_tensor_observation_hooks(context: _CapturedTensorObservationContext) -> Iterator[None]:
    """Replay only autograd-hook registration under an activation recomputation.

    Value notifications remain suspended, so forward metrics continue to observe the original
    no-grad execution exactly once. Gradient sources need their hooks attached to the graph built
    by the grad-enabled recomputation. Restoring the captured callback explicitly also crosses an
    autograd worker-thread boundary where Python ``ContextVar`` state may not propagate.
    """
    observer_context, token_type_context = context
    observer_token = _ACTIVE_OBSERVER.set(observer_context)
    token_type_token = _ACTIVE_TOKEN_TYPES.set(token_type_context)
    suspended_token = _OBSERVATION_SUSPENDED.set(True)
    gradient_token = _REPLAYING_GRADIENT_OBSERVATIONS.set(True)
    try:
        yield
    finally:
        _REPLAYING_GRADIENT_OBSERVATIONS.reset(gradient_token)
        _OBSERVATION_SUSPENDED.reset(suspended_token)
        _ACTIVE_TOKEN_TYPES.reset(token_type_token)
        _ACTIVE_OBSERVER.reset(observer_token)


def is_observing_tensor(source_kind: str) -> bool:
    """Return whether ``source_kind`` has an active observer."""
    if _OBSERVATION_SUSPENDED.get():
        return False
    active = _ACTIVE_OBSERVER.get()
    return active is not None and source_kind in active[1]


def observe_tensor(
    owner: object,
    name: str,
    source_kind: str,
    tensor: torch.Tensor,
    *,
    tp_shard_dim: int | None = None,
) -> None:
    """Notify the active observer about one tensor produced during the logical forward.

    Activation checkpoint implementations suppress notifications during their backward
    recomputation, so the original no-grad forward remains the single observed execution.

    Args:
        owner: Model object used to resolve a canonical site name.
        name: Name local to ``owner``.
        source_kind: Stable source category used by metric selection.
        tensor: Short-lived observed tensor.
        tp_shard_dim: Tensor dimension sharded over tensor parallel ranks, or ``None`` when the
            tensor is replicated over tensor parallel ranks.
    """
    _notify_observer(owner, name, source_kind, tensor, tp_shard_dim)


def observe_layer_residuals(layer: object, accumulator: torch.Tensor, output: torch.Tensor) -> None:
    """Observe a layer's incoming residual accumulator and net contribution.

    With per-layer CUDA graphs, the contribution is reconstructed as ``output - accumulator``
    outside the captured layer. Full-iteration CUDA graphs include this call and do not replay
    Python observation notifications. The contribution is computed without autograd only when a
    fused residual summary is active.

    Args:
        layer: Layer owning the residual sites.
        accumulator: Residual stream entering the layer.
        output: Residual stream leaving the layer.
    """
    config = getattr(layer, "config", None)
    tp_shard_dim = 0 if getattr(config, "sequence_parallel", False) else None
    observe_summary = is_observing_tensor("residual_summary")
    summary_token_type_observer = _token_type_observer("residual_summary")
    if observe_summary or summary_token_type_observer is not None:
        with torch.no_grad():
            contribution = output.detach() - accumulator.detach()
        summary = _residual_row_summary(accumulator, contribution)
        summary_shard_dim = 1 if tp_shard_dim is not None else None
        if observe_summary:
            _notify_observer(
                layer, "residual_summary", "residual_summary", summary, summary_shard_dim
            )
        _notify_token_type_summary_observations(
            summary_token_type_observer,
            layer,
            "residual_summary",
            "residual_summary",
            summary,
            accumulator,
            summary_shard_dim,
        )
    _observe_gradient_pair(layer, accumulator, output, tp_shard_dim)


def observe_residual_summary_boundary(owner: object, name: str, tensor: torch.Tensor) -> None:
    """Observe row norms at a decoder residual-stream boundary.

    Args:
        owner: Decoder stack owning the boundary.
        name: Stable boundary name used in the result label.
        tensor: Residual activation whose final dimension is the hidden dimension.
    """
    source_kind = "residual_summary_boundary"
    if not is_observing_tensor(source_kind):
        return
    if tensor.ndim == 0 or tensor.shape[-1] == 0:
        raise ValueError("Residual boundary summaries require a nonempty final hidden dimension.")
    config = getattr(owner, "config", None)
    tp_shard_dim = 0 if getattr(config, "sequence_parallel", False) else None
    with torch.no_grad():
        rows = tensor.detach().float().reshape(-1, tensor.shape[-1])
        summary = rows.square().sum(dim=-1).sqrt().unsqueeze(0)
    summary_shard_dim = 1 if tp_shard_dim is not None else None
    _notify_observer(owner, name, source_kind, summary, summary_shard_dim)


def _notify_observer(
    owner: object, name: str, source_kind: str, tensor: torch.Tensor, tp_shard_dim: int | None
) -> None:
    if _OBSERVATION_SUSPENDED.get():
        return
    active = _ACTIVE_OBSERVER.get()
    if active is not None and source_kind in active[1]:
        active[0](owner, name, source_kind, tensor, tp_shard_dim)


def _token_type_observer(
    source_kind: str, *, allow_replay: bool = False
) -> tuple[TensorObservationCallback, _TokenTypeContext] | None:
    if _OBSERVATION_SUSPENDED.get() and not (
        allow_replay and _REPLAYING_GRADIENT_OBSERVATIONS.get()
    ):
        return None
    active = _ACTIVE_OBSERVER.get()
    token_types = _ACTIVE_TOKEN_TYPES.get()
    token_type_source_kind = f"{source_kind}_by_token_type"
    if active is None or token_types is None or token_type_source_kind not in active[1]:
        return None
    return active[0], token_types


def _aligned_token_types(tensor: torch.Tensor, context: _TokenTypeContext) -> torch.Tensor:
    token_types = context.tensor
    if token_types.device != tensor.device:
        raise ValueError("Residual token types and tensors must be on the same device.")
    tensor_prefix = tuple(tensor.shape[:-1])
    if tuple(token_types.shape) == tensor_prefix:
        return token_types
    if token_types.numel() == tensor.numel() // tensor.shape[-1]:
        return token_types.reshape(tensor_prefix)
    raise ValueError(
        "Residual token types must align with the tensor's non-hidden dimensions; "
        f"got token types {tuple(token_types.shape)} and tensor {tuple(tensor.shape)}."
    )


@torch.no_grad()
def _notify_token_type_summary_observations(
    observer_context: tuple[TensorObservationCallback, _TokenTypeContext] | None,
    owner: object,
    name: str,
    source_kind: str,
    summary: torch.Tensor,
    reference: torch.Tensor,
    tp_shard_dim: int | None,
) -> None:
    """Split one fused row summary into compact token-type populations."""
    if observer_context is None:
        return
    observer, token_type_context = observer_context
    token_types = _aligned_token_types(reference, token_type_context).reshape(-1)
    if summary.ndim != 2 or summary.shape[1] != token_types.numel():
        raise ValueError("Residual-summary rows must align with flattened token types.")
    for token_type_id, token_type_name in enumerate(token_type_context.names):
        observer(
            owner,
            f"{name}.token_type[{token_type_name}]",
            f"{source_kind}_by_token_type",
            summary[:, token_types == token_type_id],
            tp_shard_dim,
        )


def _observe_gradient_pair(
    owner: object, accumulator: torch.Tensor, output: torch.Tensor, tp_shard_dim: int | None
) -> None:
    if not _can_observe_gradient((accumulator, output)):
        return
    summary_source_kind = "dresidual_summary"
    summary_observer = _gradient_observer(summary_source_kind, (accumulator, output))
    summary_token_type_observer = _token_type_observer(summary_source_kind, allow_replay=True)
    if summary_observer is None and summary_token_type_observer is None:
        return
    gradients: list[torch.Tensor | None] = [None, None]

    def observe_gradient(index: int, gradient: torch.Tensor) -> torch.Tensor:
        gradients[index] = gradient
        accumulator_gradient, contribution_gradient = gradients
        if accumulator_gradient is not None and contribution_gradient is not None:
            summary = _residual_row_summary(accumulator_gradient, contribution_gradient)
            summary_shard_dim = 1 if tp_shard_dim is not None else None
            if summary_observer is not None:
                summary_observer(
                    owner, summary_source_kind, summary_source_kind, summary, summary_shard_dim
                )
            _notify_token_type_summary_observations(
                summary_token_type_observer,
                owner,
                summary_source_kind,
                summary_source_kind,
                summary,
                accumulator_gradient,
                summary_shard_dim,
            )
            gradients[:] = [None, None]
        return gradient

    accumulator.register_hook(lambda gradient: observe_gradient(0, gradient))
    output.register_hook(lambda gradient: observe_gradient(1, gradient))


def _gradient_observer(
    source_kind: str, tensors: tuple[torch.Tensor, ...]
) -> TensorObservationCallback | None:
    if not _can_observe_gradient(tensors):
        return None
    active = _ACTIVE_OBSERVER.get()
    if active is None or source_kind not in active[1]:
        return None
    return active[0]


def _can_observe_gradient(tensors: tuple[torch.Tensor, ...]) -> bool:
    """Return whether hooks may be registered in the current execution."""
    return (
        torch.is_grad_enabled()
        and all(tensor.requires_grad for tensor in tensors)
        and (not _OBSERVATION_SUSPENDED.get() or _REPLAYING_GRADIENT_OBSERVATIONS.get())
    )


def _residual_row_summary_impl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute row L2 norms and cosine similarity in one tensor program."""
    left = left.float().reshape(-1, left.shape[-1])
    right = right.float().reshape(-1, right.shape[-1])
    left_square = left.square().sum(dim=-1)
    right_square = right.square().sum(dim=-1)
    left_norm = left_square.sqrt()
    right_norm = right_square.sqrt()
    denominator = (left_square * right_square).clamp_min(0).sqrt()
    correlation = (left * right).sum(dim=-1) / denominator.clamp_min(torch.finfo(left.dtype).tiny)
    correlation = correlation.clamp(-1, 1)
    correlation = torch.where(denominator > 0, correlation, torch.zeros_like(correlation))
    return torch.stack((left_norm, right_norm, correlation))


_fused_residual_row_summary = torch.compile(_residual_row_summary_impl, dynamic=True)


@torch.no_grad()
def _residual_row_summary(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return ``[accumulator norm, contribution norm, correlation]`` for every row."""
    if left.shape != right.shape:
        raise ValueError("Paired tensor observations must have matching shapes.")
    if left.ndim == 0 or left.shape[-1] == 0:
        raise ValueError("Residual summaries require a nonempty final hidden dimension.")
    if type(left) is torch.Tensor and type(right) is torch.Tensor and left.is_cuda:
        return _fused_residual_row_summary(left, right)
    return _residual_row_summary_impl(left, right)
