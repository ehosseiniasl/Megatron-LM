import abc
import dataclasses
import hashlib
import json
from collections.abc import Iterator
from typing import Any, Generic, Optional, TypeVar, cast

import numpy as np
import torch

from mistral_fork.mistral.data.source import SampleType
from mistral_fork.mistral_common.tokens.tokenizers.base import Span
from mistral_fork.mistral_common.tokens.tokenizers.image import image_size_to_num_tokens

try:
    from stool.utils import Closable
except ImportError:
    import abc as _abc_closable

    class Closable(_abc_closable.ABC):  # type: ignore[no-redef]
        @_abc_closable.abstractmethod
        def close(self) -> None: ...

        def set_wrapped_by_closing(self, value: bool) -> None: ...


Array = TypeVar("Array", np.ndarray, torch.Tensor)


def _any_dtype(arr: Array, *dtypes) -> bool:
    # np dtypes do not play well with set comparisons
    return any(arr.dtype == dt for dt in dtypes)


def walk_dataclass(obj: Any) -> Iterator[Any]:
    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            yield from walk_dataclass(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from walk_dataclass(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from walk_dataclass(item)
    else:
        yield obj


@dataclasses.dataclass
class Batch(Generic[Array]):
    sample_type: SampleType

    x: Array
    y: Array
    sizes: np.ndarray
    srcs: list[str]

    y_mask: Array | None = None

    # for some reason torch tensors are faster to pass between processes than np arrays
    images: list[torch.Tensor] | None = None
    image_positions: list[Span] = dataclasses.field(default_factory=list)

    num_text_tokens_by_src: dict[str, int] = dataclasses.field(default_factory=dict)
    num_image_tokens_by_src: dict[str, int] = dataclasses.field(default_factory=dict)
    # is_context_parallel = True when the Batch is defined in chunk_inputs_for_context_parallelism
    is_context_parallel: bool = False
    num_pad_tokens: int = 0

    stats: dict[str, Any] = dataclasses.field(default_factory=dict)
    timings: dict[str, float] = dataclasses.field(default_factory=dict)
    sample_pretokenized_paths: list[str] | None = None
    sample_doc_ids: list[int] | None = None
    # Real document lengths before suffix padding is folded into sizes.
    logical_sizes: np.ndarray | None = None

    def __post_init__(self) -> None:
        assert self.x.ndim == 1
        assert self.x.shape == self.y.shape
        assert _any_dtype(self.x, np.int64, torch.int64), self.x.dtype
        assert _any_dtype(self.y, np.int64, torch.int64), self.y.dtype
        assert self.sizes.ndim == 1
        assert self.sizes.dtype == np.int64
        assert len(self.sizes) == len(self.srcs)
        if self.sample_pretokenized_paths is not None:
            assert len(self.sample_pretokenized_paths) == len(self.srcs), (
                f"{len(self.sample_pretokenized_paths)=} != {len(self.srcs)=}"
            )
        if self.sample_doc_ids is not None:
            assert len(self.sample_doc_ids) == len(self.srcs), (
                f"{len(self.sample_doc_ids)=} != {len(self.srcs)=}"
            )
        assert self.num_pad_tokens >= 0
        if self.logical_sizes is not None:
            assert self.logical_sizes.shape == self.sizes.shape
            assert np.all((0 <= self.logical_sizes) & (self.logical_sizes <= self.sizes))
            assert int(np.sum(self.sizes - self.logical_sizes)) == self.num_pad_tokens

        # mask check
        if self.y_mask is not None:
            mask_size = np.prod(self.y_mask.shape)
            assert mask_size == np.prod(self.y.shape), (self.y_mask.shape, self.y.shape)
            assert _any_dtype(self.y_mask, bool, torch.bool)
            assert self.sizes.sum().item() == mask_size, (
                f"{self.sizes.sum()=}, {mask_size=}"
            )
            if not self.is_context_parallel:
                assert not self.y_mask.all()
                assert self.y_mask.any()

    @property
    def is_cuda(self) -> bool:
        per_tensor_is_cuda = [
            x.is_cuda for x in walk_dataclass(self) if isinstance(x, torch.Tensor)
        ]
        if not per_tensor_is_cuda:
            return False  # no torch tensors means everything is on cpu
        if all(per_tensor_is_cuda):
            return True
        if not any(per_tensor_is_cuda):
            return False
        raise RuntimeError(
            "You can't call `is_cuda` on a batch that holds tensors of mixed devices",
        )

    def num_image_tokens(
        self, max_image_size: int, patch_size: int, spatial_merge_size: int
    ) -> int:
        if self.images is None:
            return 0
        return sum(
            int(
                np.prod(
                    image_size_to_num_tokens(
                        size=(img.shape[1], img.shape[2]),
                        max_image_size=max_image_size,
                        patch_size=patch_size,
                        spatial_merge_size=spatial_merge_size,
                    )
                )
            )
            for img in self.images
        )

    def cuda(self, non_blocking: bool) -> "Batch[torch.Tensor]":
        """Returns a copy of itself with all tensors on CUDA."""
        if self.is_cuda:
            return cast("Batch[torch.Tensor]", self)

        def cuda(x: Array) -> torch.Tensor:
            return torch.from_numpy(x).cuda(non_blocking=non_blocking)

        def maybe_cuda(x: Array | None) -> torch.Tensor | None:
            return cuda(x) if x is not None else None

        return Batch(
            sample_type=self.sample_type,
            srcs=self.srcs,
            sizes=self.sizes,
            x=cuda(self.x),
            y=cuda(self.y),
            y_mask=maybe_cuda(self.y_mask),
            images=(
                [img.cuda() for img in self.images] if self.images is not None else None
            ),
            image_positions=self.image_positions,
            num_text_tokens_by_src=self.num_text_tokens_by_src,
            num_image_tokens_by_src=self.num_image_tokens_by_src,
            num_pad_tokens=self.num_pad_tokens,
            stats=self.stats,
            is_context_parallel=self.is_context_parallel,
            sample_pretokenized_paths=self.sample_pretokenized_paths,
            sample_doc_ids=self.sample_doc_ids,
            logical_sizes=self.logical_sizes,
        )

    def cpu(self) -> "Batch[np.ndarray]":
        if not self.is_cuda:
            return cast("Batch[np.ndarray]", self)

        def cpu(x: Array) -> np.ndarray:
            assert isinstance(x, torch.Tensor), type(x)
            return x.detach().cpu().numpy()

        def maybe_cpu(x: Array | None) -> np.ndarray | None:
            if x is None:
                return None
            return cpu(x)

        return Batch(
            sample_type=self.sample_type,
            srcs=self.srcs,
            sizes=self.sizes,
            x=cpu(self.x),
            y=cpu(self.y),
            y_mask=maybe_cpu(self.y_mask),
            images=(
                [img.to(torch.device("cpu")) for img in self.images]
                if self.images is not None
                else None
            ),
            image_positions=self.image_positions,
            num_text_tokens_by_src=self.num_text_tokens_by_src,
            num_image_tokens_by_src=self.num_image_tokens_by_src,
            num_pad_tokens=self.num_pad_tokens,
            stats=self.stats,
            is_context_parallel=self.is_context_parallel,
            sample_pretokenized_paths=self.sample_pretokenized_paths,
            sample_doc_ids=self.sample_doc_ids,
            logical_sizes=self.logical_sizes,
        )

    def get_stats(self) -> dict[str, Any]:
        # Empty previous stats
        return {
            "nb_documents": len(self.sizes),
            "shortest_document_length": int(min(self.sizes)),
            "longest_document_length": int(max(self.sizes)),
            "avg_document_length": int(np.mean(self.sizes)),
        }

    def to_dict(self) -> dict[str, Any]:
        def _flat(arr: Array | None) -> list[int | float | bool] | None:
            return None if arr is None else arr.tolist()

        data = {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "sizes": self.sizes.tolist(),
            "srcs": self.srcs,
            "y_mask": _flat(self.y_mask),
        }
        return {k: v for k, v in data.items() if v is not None}

    def compute_hash(self, use_src: bool = True) -> str:
        """Compute a hash of this batch (for debugging purposes only).
        Compute and return SHA256 hash of the JSON string.
        """
        data = self.to_dict()
        if not use_src:
            data.pop("srcs")
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data_str.encode()).hexdigest()


B = TypeVar("B")
DataLoaderState = dict[str, Any]

DataLoaderStats = dict[
    str,
    dict[str, Any],
]


class DataLoader(Closable, abc.ABC, Generic[B]):
    @abc.abstractmethod
    def get_batch(self) -> B:
        pass

    @abc.abstractmethod
    def get_stats(self) -> DataLoaderStats:
        pass

    @abc.abstractmethod
    def get_state(self) -> DataLoaderState:
        """The contract is: you can call get_state only after starting the dataloader
        Also the dataloader should support the call to get_state once started, even before first Batch. And this state
        can't be None (cf DataLoaderState) so use a proper signature!
        """
        pass

    @abc.abstractmethod
    def unwrap(self) -> Optional["DataLoader"] | list[Optional["DataLoader"]]:
        pass

    @abc.abstractmethod
    def set_state(self, state: DataLoaderState) -> None:
        "The contract is: you can call set_state only BEFORE starting the dataloader, after it is too late!"
        pass

    @property
    @abc.abstractmethod
    def source_names(self) -> set[str]:
        pass

    @property
    @abc.abstractmethod
    def max_steps(self) -> int:
        pass

    @abc.abstractmethod
    def start(self) -> None:
        pass
