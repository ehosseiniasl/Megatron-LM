"""Multiprocessed data loader wrapper.

Moves a DataLoader into a separate subprocess and pre-fetches batches via a
two-stage queue pipeline:

  Worker subprocess  --[mp.Queue]--> dequeue thread --[queue.Queue]--> get_batch()

The mp.Queue is kept small (default 1) to limit shared-memory file descriptors.
The in-process queue.Queue provides the actual prefetch depth.

Ported from upstream Mistral (mistral/data/multiprocessed.py) with simplified
dependencies (no py-spy, no CompositeDataLoader/FiniteDataLoader exceptions).
"""

import atexit
import logging
import os
import time
from multiprocessing.context import SpawnProcess
from multiprocessing.synchronize import Event as EventClass
from queue import Empty, Full, Queue
from threading import Thread
from typing import Any, Callable, Optional

from torch import multiprocessing as mp

from mistral_fork.mistral.data.data_loader import (
    Batch,
    DataLoader,
    DataLoaderState,
    DataLoaderStats,
)

logger = logging.getLogger(__name__)

# Workers start with "spawn", so a child gets a fresh logging module and none of
# the levels the parent set. Carry this package's level across by hand; without
# it a caller that quiets the dataloader still gets the full output from every
# worker.
_DATA_PACKAGE = __name__.rsplit(".", 1)[0]


def _feed_queue(buffer: Queue, queue: mp.Queue, stop: EventClass) -> None:
    """Thread that drains the mp.Queue into the in-process buffer."""
    logger.debug("Queue dequeue thread started")
    while not stop.is_set():
        try:
            res = queue.get(timeout=1)
        except Empty:
            continue
        while not stop.is_set():
            try:
                buffer.put(res, timeout=1)
                break
            except Full:
                continue
    logger.debug("Queue thread stopped")


class WorkerProcess(SpawnProcess):
    # NOTE: SpawnProcess inheritance is important here, as using fork will lead to deadlocks
    def __init__(
        self,
        loader_ctr: Callable[[], DataLoader],
        stop: EventClass,
        queue: mp.Queue,
        log_level: int,
        data_log_level: int,
    ) -> None:
        super().__init__()

        # Failsafe - flag a constructor which would be unpickable down the line, fail fast
        import cloudpickle

        try:
            self.loader_ctr = cloudpickle.dumps(loader_ctr)
        except Exception as e:
            logger.error(f"loader_ctr is unpicklable: {e}")
            raise e

        self._state: bytes | None = None
        self._wrapped_by_closing = False
        self._started = False

        self._log_level = log_level
        self._data_log_level = data_log_level

        # Shared attributes accessible from the main process
        self._stop = stop
        self._queue: mp.Queue = queue

    def run(self) -> None:
        # Set up logging in subprocess
        root = logging.getLogger()
        root.setLevel(self._log_level)
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(self._log_level)
            root.addHandler(handler)
        logging.getLogger(_DATA_PACKAGE).setLevel(self._data_log_level)
        # Named under _DATA_PACKAGE so the level set just above governs it. As a bare
        # top-level name it escaped every caller that quiets the dataloader package.
        logger_ = logging.getLogger(f"{_DATA_PACKAGE}.worker")

        import cloudpickle

        try:
            loader: DataLoader = cloudpickle.loads(self.loader_ctr)()
            loader.set_wrapped_by_closing(self._wrapped_by_closing)

            if self._state is not None:
                dataloader_state = cloudpickle.loads(self._state)
                loader.set_state(dataloader_state)

            loader.start()

            initial_state: DataLoaderState = loader.get_state()
            source_names = loader.source_names
            max_steps = loader.max_steps
        except Exception as err:
            self._queue.put(err)
            raise
        self._queue.put((source_names, max_steps, initial_state))
        self._started = True

        try:
            logger_.info("Starting data worker!")
            batch: Any = None
            state: Any = None
            stats: Any = None

            while not self._stop.is_set():
                batch = loader.get_batch() if batch is None else batch
                state = loader.get_state() if state is None else state
                stats = loader.get_stats() if stats is None else stats

                try:
                    self._queue.put_nowait((batch, state, stats))
                except Full:
                    time.sleep(0.02)
                else:
                    batch, state, stats = None, None, None

        except Exception as err:
            while True:
                try:
                    self._queue.put(err, timeout=0.01)
                    break
                except Full:
                    pass

            while not self._stop.is_set():
                time.sleep(0.1)

            raise
        finally:
            loader.close()
            logger_.info("Exiting data worker!")

    @property
    def wrapped_by_closing(self) -> int:
        return self._wrapped_by_closing

    def set_state(self, state: DataLoaderState) -> None:
        assert not self._started, (
            f"Please set state before starting data loader - is started: {self._started}"
        )
        import cloudpickle

        self._state = cloudpickle.dumps(state)

    def set_wrapped_by_closing(self, value: bool) -> None:
        self._wrapped_by_closing = value


class MultiprocessedDataLoader(DataLoader):
    """Moves the underlying data loader to a separate process and pre-fetches batches."""

    def __init__(
        self,
        loader_ctr: Callable[[], DataLoader],
        max_size: int,
        timeout: float = 600,
        max_size_worker_queue: int = 1,
    ) -> None:
        assert max_size > 0

        self.ctx = mp.get_context("spawn")
        self.mp_queue = self.ctx.Queue(maxsize=max_size_worker_queue)
        self.stop: EventClass = self.ctx.Event()
        self.loader_ctr = loader_ctr

        self._wrapped_by_closing = False
        self.worker = self._new_worker()
        self._source_names: set[str] | None = None
        self._max_steps: int | None = None
        self._is_worker_started = False

        self._state: DataLoaderState | None = None
        self.stats: DataLoaderStats = {}

        self.timeout = timeout

        self.buffer: Queue[Any] = Queue(maxsize=max_size)
        self.thread: Optional[Thread] = None

        # Safety net: ensure close() runs during interpreter shutdown even if
        # the caller doesn't call it (e.g. unhandled exception). Registered
        # after multiprocessing._exit_function (which was registered at import
        # time), so LIFO order guarantees our close() runs first.
        atexit.register(self.close)

    def start(self) -> None:
        assert not self.is_started, "Worker is already started"

        if not self.worker.wrapped_by_closing:
            raise RuntimeError(
                "Please use set_wrapped_by_closing(True) before starting"
            )

        self.worker.start()
        logger.debug(f"Starting process... PID is {self.worker.pid}")
        self._is_worker_started = True

        while True:
            try:
                res = self.mp_queue.get(timeout=0.01)
            except Empty:
                if not self.worker.is_alive():
                    raise RuntimeError("Data worker (abruptly) died!")
            else:
                if isinstance(res, Exception):
                    raise RuntimeError(
                        f"Data worker died raising: {type(res).__name__}: {res}"
                    )
                self._source_names, self._max_steps, self._state = res
                assert self._state is not None
                break

        logger.debug(
            f"Started process (parent: {os.getpid()}, child: {self.worker.pid})!"
        )
        self._start_thread()

    def _start_thread(self) -> None:
        assert self.thread is None
        self.thread = Thread(
            target=_feed_queue,
            args=(self.buffer, self.mp_queue, self.stop),
            name="QueueThread",
            daemon=True,
        )
        self.thread.start()

    def get_batch(self) -> Batch:
        if not self._is_worker_started:
            self.start()

        start = time.time()
        n_retries = 0

        while not self.stop.is_set():
            try:
                res = self.buffer.get(timeout=0.1)

                if isinstance(res, Exception):
                    self.stop.set()
                    raise res

                batch, state, stats = res
            except Empty:
                if not self.worker.is_alive() and not self.stop.is_set():
                    raise RuntimeError("Data worker (abruptly) died!")
                elif time.time() - start > self.timeout:
                    raise TimeoutError("Data worker is hanging!")
                else:
                    n_retries += 1
                    time.sleep(0.01)
            else:
                self._state = state
                self.stats = stats
                return batch
        raise RuntimeError("Data worker died!")

    def get_stats(self) -> DataLoaderStats:
        return self.stats

    def get_state(self) -> DataLoaderState:
        assert self.is_started, "get_state called before the dataloader was started"
        assert self._state is not None
        return self._state

    @property
    def is_started(self) -> bool:
        if self._is_worker_started and not self.worker.is_alive():
            logger.warning("Worker either died, or it concluded")
        return self._is_worker_started

    @property
    def max_steps(self) -> int:
        assert self.is_started
        assert self._max_steps is not None
        return self._max_steps

    @property
    def source_names(self) -> set[str]:
        assert self.worker is not None and self.worker.is_alive()
        assert self._source_names is not None
        return self._source_names

    def set_state(self, state: DataLoaderState) -> None:
        assert not self.is_started, (
            "Please set state before starting multiprocessed data loader"
        )
        self.worker.set_state(state)

    def set_wrapped_by_closing(self, value: bool) -> None:
        self._wrapped_by_closing = value
        self.worker.set_wrapped_by_closing(value)

    def _drain_queue(self, q: Any) -> int:
        """Drain all items from a queue, return how many were discarded."""
        n = 0
        while True:
            try:
                _ = q.get_nowait()
                n += 1
            except Empty:
                break
            except (EOFError, OSError):
                break
        return n

    def close(self) -> None:
        logger.debug("Closing dataloader...")
        if not self._is_worker_started:
            atexit.unregister(self.close)
            return
        self.stop.set()

        if self.thread is not None:
            logger.debug("Closing thread")
            self.thread.join()
            self.thread = None
            logger.debug("Thread joined")

            n = self._drain_queue(self.buffer)
            if n:
                logger.debug(f"Drained {n} items from buffer after stop")

        # Try to join the worker, draining the mp.Queue between attempts.
        # mp.Queue holds shared memory; unread items prevent process exit.
        # NOTE: a sentinel-based approach (worker puts sentinel in queue on exit)
        # would be cleaner, but the _feed_queue thread complicates it: the thread
        # might consume the sentinel from mp_queue and block trying to forward it
        # to a full buffer, causing a deadlock with thread.join() above.
        if self.worker.is_alive():
            assert self._is_worker_started
            logger.debug("Worker is alive, joining it")

            deadline = time.time() + 10
            while time.time() < deadline:
                n = self._drain_queue(self.mp_queue)
                if n:
                    logger.debug(f"Drained {n} items from mp_queue after stop")
                self.worker.join(timeout=0.01)
                if self.worker.exitcode is not None:
                    break

            if self.worker.exitcode is None:
                logger.warning("Force killing data worker...")
                self.worker.kill()
                self.worker.join(timeout=30)

            logger.debug(f"Data worker process joined (exitcode={self.worker.exitcode})")

        self.mp_queue = self.ctx.Queue(maxsize=1)
        self.worker = self._new_worker()
        self._is_worker_started = False

        self.stop.clear()
        atexit.unregister(self.close)

    def _new_worker(self) -> WorkerProcess:
        log_level = logging.getLogger().getEffectiveLevel()
        data_log_level = logging.getLogger(_DATA_PACKAGE).getEffectiveLevel()
        worker = WorkerProcess(
            loader_ctr=self.loader_ctr,
            stop=self.stop,
            queue=self.mp_queue,
            log_level=log_level,
            data_log_level=data_log_level,
        )
        worker.set_wrapped_by_closing(self._wrapped_by_closing)
        return worker

    def unwrap(self) -> Optional["DataLoader"] | list[Optional["DataLoader"]]:
        raise ValueError("Cannot unwrap a multiprocessed dataloader")
