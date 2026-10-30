# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A torch_dist load unpickles the checkpoint's `.metadata` once, not once per reader."""

import gc
from unittest import mock

import pytest
import torch
from torch.distributed.checkpoint import FileSystemReader

from megatron.core.dist_checkpointing import (
    ShardedTensor,
    load,
    load_common_state_dict,
    load_tensors_metadata,
    save,
)
from megatron.core.dist_checkpointing.serialization import load_sharded_metadata
from megatron.core.dist_checkpointing.strategies.cached_metadata_filesystem_reader import (
    CachedMetadataFileSystemReader,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    _get_filesystem_reader,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class _BaseReadCounter:
    """Counts the full `.metadata` reads, i.e. calls of torch's `FileSystemReader.read_metadata`."""

    def __init__(self):
        self.count = 0
        self._patcher = None

    def __enter__(self):
        original = FileSystemReader.read_metadata
        counter = self

        def counted(reader, *args, **kwargs):
            counter.count += 1
            return original(reader, *args, **kwargs)

        self._patcher = mock.patch.object(FileSystemReader, 'read_metadata', counted)
        self._patcher.start()
        return self

    def __exit__(self, *exc):
        self._patcher.stop()


def _sharded_state_dict(rank, world_size, keys=('a', 'b'), scale=1.0, device='cuda'):
    sd = {}
    for i, key in enumerate(keys):
        data = (torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4) + i) * scale
        sd[key] = ShardedTensor.from_rank_offsets(key, data + rank, (0, rank, world_size))
    sd['common'] = {'iteration': 7, 'scale': scale}
    return sd


class TestMetadataCache:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        CachedMetadataFileSystemReader.clear_metadata_cache()

    def teardown_method(self, method):
        CachedMetadataFileSystemReader.clear_metadata_cache()
        Utils.destroy_model_parallel()

    def test_one_unpickle_per_load(self, tmp_path_dist_ckpt):
        """The pre-pass, the tensors-metadata lookup and the load share one `.metadata` read."""
        rank, world_size = Utils.rank, Utils.world_size
        with TempNamedDir(tmp_path_dist_ckpt / 'test_one_unpickle_per_load') as ckpt_dir:
            save(_sharded_state_dict(rank, world_size), ckpt_dir)
            CachedMetadataFileSystemReader.clear_metadata_cache()

            base_strategy = TorchDistLoadShardedStrategy(cache_metadata=True)
            load_strategy = FullyParallelLoadStrategyWrapper(
                base_strategy, per_rank_object_load=True
            )
            with _BaseReadCounter() as reads:
                # The same sequence as megatron/training/checkpointing.py::load_checkpoint:
                # args pre-pass, GTP padding check, then dist_checkpointing.load (which reads
                # the common state again and runs the sharded load).
                common = load_common_state_dict(ckpt_dir)
                tensors_md = load_tensors_metadata(ckpt_dir)
                sharded_md = load_sharded_metadata(ckpt_dir)
                loaded = load(_sharded_state_dict(rank, world_size), ckpt_dir, load_strategy)
            assert reads.count == 1, reads.count

            assert common['common'] == {'iteration': 7, 'scale': 1.0}
            assert set(tensors_md) == {'a', 'b'}
            assert tensors_md['a'].global_shape == (4 * world_size, 4)
            assert set(sharded_md) >= {'a', 'b'}
            expected = _sharded_state_dict(rank, world_size)
            for key in ('a', 'b'):
                assert torch.equal(loaded[key].cpu(), expected[key].data.cpu()), key
            assert loaded['common'] == {'iteration': 7, 'scale': 1.0}

            # The metadata kept for save-side reuse is a detached copy over the same dicts, so
            # the save path's rebinding of storage_data cannot reach the cached object.
            cached = CachedMetadataFileSystemReader._cache_value
            kept = base_strategy.cached_global_metadata
            assert cached is not None and kept is not None
            assert kept is not cached
            assert kept.state_dict_metadata is cached.state_dict_metadata
            assert kept.storage_data is cached.storage_data

    def test_readers_share_the_cache_regardless_of_flags(self, tmp_path_dist_ckpt):
        """`_get_filesystem_reader` returns the cached reader for every call site."""
        rank, world_size = Utils.rank, Utils.world_size
        with TempNamedDir(tmp_path_dist_ckpt / 'test_readers_share') as ckpt_dir:
            save(_sharded_state_dict(rank, world_size), ckpt_dir)
            CachedMetadataFileSystemReader.clear_metadata_cache()
            readers = [
                _get_filesystem_reader(ckpt_dir),
                _get_filesystem_reader(ckpt_dir, cache_metadata=True, async_strategy='mcore'),
                _get_filesystem_reader(ckpt_dir, cache_metadata=True, async_strategy='nvrx'),
                _get_filesystem_reader(str(ckpt_dir), cache_metadata=False),
            ]
            assert all(isinstance(r, CachedMetadataFileSystemReader) for r in readers)
            with _BaseReadCounter() as reads:
                metadatas = [r.read_metadata() for r in readers]
            assert reads.count == 1
            assert all(md is metadatas[0] for md in metadatas)

    def test_cache_follows_the_file(self, tmp_path_dist_ckpt):
        """A rewritten `.metadata` (same directory) is read again; a cleared cache too."""
        rank, world_size = Utils.rank, Utils.world_size
        with TempNamedDir(tmp_path_dist_ckpt / 'test_cache_follows_the_file') as ckpt_dir:
            save(_sharded_state_dict(rank, world_size, keys=('a', 'b')), ckpt_dir)
            with _BaseReadCounter() as reads:
                assert set(load_tensors_metadata(ckpt_dir)) == {'a', 'b'}
                assert set(load_tensors_metadata(ckpt_dir)) == {'a', 'b'}
            assert reads.count == 1

            # Overwrite the checkpoint in place with different keys.
            torch.distributed.barrier()
            save(_sharded_state_dict(rank, world_size, keys=('c', 'd', 'e')), ckpt_dir)
            with _BaseReadCounter() as reads:
                assert set(load_tensors_metadata(ckpt_dir)) == {'c', 'd', 'e'}
                assert set(load_tensors_metadata(ckpt_dir)) == {'c', 'd', 'e'}
            assert reads.count == 1

            # Clearing for another directory is a no-op; clearing for this one forces a re-read.
            CachedMetadataFileSystemReader.clear_metadata_cache(ckpt_dir / 'elsewhere')
            with _BaseReadCounter() as reads:
                load_tensors_metadata(ckpt_dir)
            assert reads.count == 0
            CachedMetadataFileSystemReader.clear_metadata_cache(ckpt_dir)
            with _BaseReadCounter() as reads:
                load_tensors_metadata(ckpt_dir)
            assert reads.count == 1

    def test_gc_state_is_restored(self, tmp_path_dist_ckpt):
        """The unpickle pauses the cyclic GC and leaves it as it found it."""
        rank, world_size = Utils.rank, Utils.world_size
        with TempNamedDir(tmp_path_dist_ckpt / 'test_gc_state_is_restored') as ckpt_dir:
            save(_sharded_state_dict(rank, world_size), ckpt_dir)
            was_enabled = gc.isenabled()
            try:
                gc.enable()
                CachedMetadataFileSystemReader.clear_metadata_cache()
                CachedMetadataFileSystemReader(ckpt_dir).read_metadata()
                assert gc.isenabled()

                gc.disable()
                CachedMetadataFileSystemReader.clear_metadata_cache()
                CachedMetadataFileSystemReader(ckpt_dir).read_metadata()
                assert not gc.isenabled()
            finally:
                if was_enabled:
                    gc.enable()
                else:
                    gc.disable()
