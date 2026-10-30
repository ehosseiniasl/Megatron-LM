# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""FS Reader that unpickles a checkpoint's metadata at most once per process."""

import gc
import os
from contextlib import contextmanager
from typing import Optional, Tuple, Union

from torch.distributed.checkpoint import FileSystemReader, Metadata

# Identity of the cached `.metadata` file: absolute path, size in bytes, modification time (ns).
_CacheKey = Tuple[str, int, int]


@contextmanager
def _cyclic_gc_paused():
    """Pauses the cyclic garbage collector for the block and restores its previous state."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class CachedMetadataFileSystemReader(FileSystemReader):
    """
    Extends FileSystemReader so that a checkpoint's `.metadata` is unpickled at most once.

    One torch_dist load reads `.metadata` from several places (the common-state load, the
    tensors-metadata lookup, the sharded load itself) and every read is a full `pickle.load` of a
    file that reaches hundreds of MiB for large MoE checkpoints. The cache keeps the `Metadata`
    of the most recently read checkpoint directory, process-wide and shared by all reader
    instances, keyed by the `.metadata` path, size and modification time: a rewritten file is
    read again and a stale object is never served. The unpickle runs with the cyclic garbage
    collector paused, since it creates no cyclic garbage and the collector's young-generation
    passes otherwise roughly double its duration.

    The returned `Metadata` object is shared between all readers of the same checkpoint. Callers
    that hand it to code which modifies it should take a copy first (see
    `TorchDistLoadShardedStrategy.load`).

    Attributes:
        _cache_key (Optional[_CacheKey]): identity of the cached `.metadata` file.
        _cache_value (Optional[Metadata]): the cached metadata.
    """

    _cache_key: Optional[_CacheKey] = None
    _cache_value: Optional[Metadata] = None

    def __init__(self, path: Union[str, os.PathLike], cache_metadata: bool = True) -> None:
        """
        Initialize with file system path.

        Args:
            path (Union[str, os.PathLike]): Path to the checkpoint directory or file.
            cache_metadata (bool): kept for backward compatibility. The cache is always consulted
                because its entry is validated against the file on disk; the flag only tells
                `TorchDistLoadShardedStrategy` whether to keep the metadata for save-side reuse.
        """
        super().__init__(path=path)
        self.cache_metadata = cache_metadata

    def _metadata_file(self, rank: Optional[int] = None) -> str:
        """Path of the `.metadata` file this reader reads (rank-local variants included)."""
        get_path = getattr(self, "_get_metadata_path", None)
        if get_path is not None:
            return os.fspath(get_path(rank))
        return os.fspath(self.fs.concat_path(self.path, ".metadata"))

    def _current_cache_key(self, rank: Optional[int] = None) -> Optional[_CacheKey]:
        """Cache key of the file about to be read, or None when it cannot be stat-ed."""
        try:
            path = self._metadata_file(rank)
            st = os.stat(path)
        except (OSError, TypeError, ValueError):
            return None
        return (os.path.abspath(path), st.st_size, st.st_mtime_ns)

    def read_metadata(self, *args, **kwargs) -> Metadata:
        """
        Read metadata from file system, caching for subsequent calls.
        Shared across instances when the checkpoint directory (and its `.metadata` file) is
        the same.

        Returns:
            Metadata: Checkpoint metadata.
        """
        key = self._current_cache_key(kwargs.get("rank"))
        if key is None:
            return super().read_metadata(*args, **kwargs)
        cls = CachedMetadataFileSystemReader
        if cls._cache_key == key and cls._cache_value is not None:
            metadata = cls._cache_value
            if getattr(metadata, "storage_meta", None) is not None:
                metadata.storage_meta.load_id = self.load_id
            return metadata
        with _cyclic_gc_paused():
            metadata = super().read_metadata(*args, **kwargs)
        cls._cache_key, cls._cache_value = key, metadata
        return metadata

    @classmethod
    def clear_metadata_cache(cls, path: Union[str, os.PathLike, None] = None):
        """
        Clear the metadata cache.

        Args:
            path (Union[str, os.PathLike, None]): If provided, the cache is cleared only if it
                holds the `.metadata` of this checkpoint directory; otherwise it is always cleared.
        """
        if path is not None and cls._cache_key is not None:
            wanted = os.path.abspath(os.path.join(os.fspath(path), ".metadata"))
            if cls._cache_key[0] != wanted:
                return
        cls._cache_key, cls._cache_value = None, None
