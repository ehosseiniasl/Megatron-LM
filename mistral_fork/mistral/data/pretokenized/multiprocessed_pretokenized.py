"""Multiprocessed wrapper for PretokenizedDataLoader.

Creates N workers, each running a PretokenizedDataLoader in a subprocess with
multiprocessed_worker_info=(worker_id, num_workers). All workers share the same
global state and coordinate via the skip logic in PretokenizedDataLoader.get_batch().

Batches are served round-robin across workers.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, Protocol

from mistral_fork.mistral.data.data_loader import (
    Batch,
    DataLoader,
    DataLoaderState,
    DataLoaderStats,
)
from mistral_fork.mistral.data.multiprocessed import MultiprocessedDataLoader
from mistral_fork.mistral.data.pretokenized.pretokenized_data_loader import (
    PretokenizedDataLoader,
)


class PretokenizedDataLoaderFn(Protocol):
    def __call__(
        self, multiprocessed_worker_info: tuple[int, int] | None
    ) -> PretokenizedDataLoader: ...


class MultiprocessedPretokenizedDataLoader(DataLoader):
    """Wraps N MultiprocessedDataLoader workers, each running a PretokenizedDataLoader.

    All dataloaders share a single global state. Each worker is assigned a
    non-overlapping stride via multiprocessed_worker_info, and the state after
    each get_batch() is identical regardless of how many workers are used.
    """

    def __init__(
        self,
        pretokenized_data_loader_fn: PretokenizedDataLoaderFn,
        num_workers: int,
        max_size: int,
        timeout: float,
    ) -> None:
        self.workers = []
        for worker_id in range(num_workers):
            self.workers.append(
                MultiprocessedDataLoader(
                    partial(
                        pretokenized_data_loader_fn,
                        multiprocessed_worker_info=(worker_id, num_workers),
                    ),
                    max_size=max_size,
                    timeout=timeout,
                )
            )

        self.next_worker = 0
        self._state: DataLoaderState | None = None
        self._started: bool = False
        self._stats: DataLoaderStats = {}

    def get_batch(self) -> Batch:
        batch = self.workers[self.next_worker].get_batch()
        state = self.workers[self.next_worker].get_state()
        self._stats = self.workers[self.next_worker].get_stats()

        self._state = state

        self.next_worker = (self.next_worker + 1) % len(self.workers)

        return batch

    def get_state(self) -> DataLoaderState:
        assert self._state is not None, "Start the dataloader before calling get_state"
        return self._state

    def set_state(self, state: DataLoaderState) -> None:
        assert not self._started, "set_state must be called before start"
        assert self._state is None, "set_state was already called"
        assert state is not None
        # Same state goes to all workers — each skips to the right position
        for worker in self.workers:
            worker.set_state(state)
        self._state = state

    def close(self) -> None:
        for loader in self.workers:
            loader.close()

    def start(self) -> None:
        assert not self._started, "dataloader was already started"
        self._started = True
        with ThreadPoolExecutor() as pool:
            list(pool.map(lambda loader: loader.start(), self.workers))
        if self._state is None:
            received_state = self.workers[0].get_state()
            assert received_state is not None
            self._state = received_state

    def set_wrapped_by_closing(self, value: bool) -> None:
        for loader in self.workers:
            loader.set_wrapped_by_closing(value)

    @property
    def max_steps(self) -> int:
        return math.inf  # type: ignore

    @property
    def source_names(self) -> set[str]:
        return {source for loader in self.workers for source in loader.source_names}

    def unwrap(self) -> Optional[DataLoader] | list[Optional[DataLoader]]:
        unwrapped_loaders: list[Optional[DataLoader]] = []
        for data_loader in self.workers:
            unwrapped = data_loader.unwrap()
            assert not isinstance(unwrapped, list), type(unwrapped)
            unwrapped_loaders.append(unwrapped)
        return unwrapped_loaders

    def get_stats(self) -> DataLoaderStats:
        return self._stats
