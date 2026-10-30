"""Tests for MultiprocessedPretokenizedDataLoader.

Adapted from upstream Mistral (mistral/data/pretokenized/multiprocessed_pretokenized_test.py).
"""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from mistral_fork.mistral.bs_rampup import BatchSizeSchedule
from mistral_fork.mistral.data.pretokenized.datasource_test import (
    BOS_ID,
    DebugImageEncoder,
    EOS_ID,
    PAD_ID,
)
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.multiprocessed_pretokenized import (
    MultiprocessedPretokenizedDataLoader,
    PretokenizedDataLoaderFn,
)
from mistral_fork.mistral.data.pretokenized.pretokenized_args import PretokenizedArgs
from mistral_fork.mistral.data.pretokenized.pretokenized_data_loader import (
    PretokenizedDataLoader,
)
from mistral_fork.mistral.data.pretokenized.pretokenized_data_loader_test import (
    mock_bs_schedule,
    prepare_data_sources_on_disk,
)
from mistral_fork.mistral.data.pretokenized.pretokenized_utils import (
    ConstantWeightSchedule,
    LinearInterpolationWeightSchedule,
    SourceItem,
    WeightSchedule,
)
from mistral_fork.mistral.data.source import SampleType


@contextmanager
def _closing(
    loader: MultiprocessedPretokenizedDataLoader,
) -> Iterator[MultiprocessedPretokenizedDataLoader]:
    """Context manager that calls set_wrapped_by_closing and close."""
    loader.set_wrapped_by_closing(True)
    try:
        yield loader
    finally:
        loader.close()


def make_dataloader_fn(
    rank: int,
    world_size: int,
    prefix_paths: list[str],
    weights: list[float],
    seq_len: int,
    n_batches: int,
    global_batch_size: int,
    batch_size_schedule: BatchSizeSchedule | None,
    tmp_path: Path,
    use_linear_weight_schedule: bool = False,
    add_images: bool = False,
) -> PretokenizedDataLoaderFn:
    num_total_tokens = n_batches * global_batch_size

    weights_arr = np.array(weights)
    weights_normalized = weights_arr / np.sum(weights_arr)

    if use_linear_weight_schedule:
        target_weight = 1 - weights_normalized
        target_weight = target_weight / np.sum(target_weight)
        weight_schedulers: list[WeightSchedule] = [
            LinearInterpolationWeightSchedule(
                weight_init=w, weight_final=tw, mix_start_tok=1, mix_end_tok=5
            )
            for w, tw in zip(weights_normalized, target_weight, strict=True)
        ]
    else:
        weight_schedulers = [ConstantWeightSchedule(w) for w in weights_normalized]

    source_items = [
        SourceItem(
            source_path=Path(path),
            sample_type=SampleType.PRETRAIN,
            weight_schedule=w_schedule,
            percentage_data=1.0,
            filter_profile=FilterArgsName.NO_FILTER,
            pretokenized_path=Path(path),
        )
        for path, w_schedule in zip(prefix_paths, weight_schedulers, strict=True)
    ]

    def _dataloader(
        multiprocessed_worker_info: tuple[int, int] | None,
    ) -> PretokenizedDataLoader:
        return PretokenizedDataLoader(
            source_items=source_items,
            seed=0,
            num_toks_in_global_batch=global_batch_size,
            num_total_tokens=num_total_tokens,
            rank=rank,
            world_size=world_size,
            seq_len=seq_len,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_id=PAD_ID,
            multiprocessed_worker_info=multiprocessed_worker_info,
            pretokenized_args=PretokenizedArgs(
                epoch_index_cache_path=str(tmp_path / "cache_root"),
                per_source_filter="/placeholder:no_filter",
            ),
            image_encoder_build_fn=(
                (lambda x: DebugImageEncoder()) if add_images else None
            ),
            selection_strategy=None,
            batch_size_schedule=batch_size_schedule,
        )

    return _dataloader


def test_multiprocessed_matches_single_process(tmp_path: Path) -> None:
    """Verify that N multiprocessed workers produce the same batches as a single-process loader."""
    n_batches = 20
    n_sources = 3
    seq_len = 5
    global_batch_size = 4 * seq_len
    rank = 1
    world_size = 2
    prefix_paths, _, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources
    )
    weights = [float(n) for n in n_toks_per_source]

    dataloader_fn = make_dataloader_fn(
        rank=rank,
        world_size=world_size,
        prefix_paths=prefix_paths,
        weights=weights,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=n_batches,
        tmp_path=tmp_path,
        batch_size_schedule=None,
    )

    with ExitStack() as exit_stack:
        ref_dataloader = dataloader_fn(multiprocessed_worker_info=None)
        ref_dataloader.start()

        for num_workers in [1, 3]:
            mp_loader = MultiprocessedPretokenizedDataLoader(
                pretokenized_data_loader_fn=dataloader_fn,
                num_workers=num_workers,
                max_size=2,
                timeout=10,
            )
            exit_stack.enter_context(_closing(mp_loader))
            mp_loader.start()

            for i in range(n_batches):
                ref_batch = ref_dataloader.get_batch()
                ref_state = ref_dataloader.get_state()
                batch = mp_loader.get_batch()
                other_state = mp_loader.get_state()

                assert other_state == ref_state
                np.testing.assert_allclose(batch.x, ref_batch.x)
                np.testing.assert_allclose(batch.y, ref_batch.y)
                np.testing.assert_allclose(batch.sizes, ref_batch.sizes)
                assert batch.srcs == ref_batch.srcs

            # Reset ref for next num_workers iteration
            ref_dataloader = dataloader_fn(multiprocessed_worker_info=None)
            ref_dataloader.start()


def test_exact_restart(tmp_path: Path) -> None:
    """Verify that state save/restore produces identical batches regardless of num_workers."""
    n_batches = 30
    n_sources = 3
    seq_len = 5
    global_batch_size = 4 * seq_len
    rank = 1
    world_size = 2
    prefix_paths, _, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources
    )
    weights = [float(n) for n in n_toks_per_source]

    dataloader_fn = make_dataloader_fn(
        rank=rank,
        world_size=world_size,
        prefix_paths=prefix_paths,
        weights=weights,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=n_batches,
        tmp_path=tmp_path,
        batch_size_schedule=None,
    )

    with ExitStack() as exit_stack:
        data_loader = MultiprocessedPretokenizedDataLoader(
            pretokenized_data_loader_fn=dataloader_fn,
            num_workers=2,
            max_size=2,
            timeout=10,
        )
        exit_stack.enter_context(_closing(data_loader))
        data_loader.start()

        rng = np.random.RandomState(seed=2)
        n_reset = 3
        reset = set(rng.permutation(range(n_batches))[:n_reset])

        new_num_workers = rng.randint(1, 4)
        new_data_loader = MultiprocessedPretokenizedDataLoader(
            pretokenized_data_loader_fn=dataloader_fn,
            num_workers=new_num_workers,
            max_size=2,
            timeout=10,
        )
        exit_stack.enter_context(_closing(new_data_loader))
        new_data_loader.start()

        for i in range(n_batches):
            ref_batch = data_loader.get_batch()
            ref_state = data_loader.get_state()
            batch = new_data_loader.get_batch()
            other_state = new_data_loader.get_state()
            assert other_state == ref_state

            np.testing.assert_allclose(batch.x, ref_batch.x)
            np.testing.assert_allclose(batch.y, ref_batch.y)
            np.testing.assert_allclose(batch.sizes, ref_batch.sizes)
            assert batch.srcs == ref_batch.srcs

            if i in reset:
                new_data_loader.close()
                new_num_workers = rng.randint(1, 4)
                new_data_loader = MultiprocessedPretokenizedDataLoader(
                    pretokenized_data_loader_fn=dataloader_fn,
                    num_workers=new_num_workers,
                    max_size=2,
                    timeout=10,
                )
                new_data_loader.set_state(ref_state)
                exit_stack.enter_context(_closing(new_data_loader))
                new_data_loader.start()


@pytest.mark.parametrize(
    ("use_bs_schedule", "use_linear_weight_schedule"),
    [(False, False), (True, False), (False, True)],
)
def test_multiprocessed(
    use_bs_schedule: bool, use_linear_weight_schedule: bool, tmp_path: Path
) -> None:
    """Upstream test: multiprocessed with 4 workers matches single-process.

    Parametrized with batch size schedule and linear weight schedule variants.
    """
    n_batches = 100
    n_sources = 3
    seq_len = 5
    global_batch_size = 4 * seq_len
    rank = 1
    world_size = 2
    prefix_paths, _, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources
    )
    if use_linear_weight_schedule:
        weights = [1 / len(n_toks_per_source) for _ in n_toks_per_source]
        weights[0] += 0.9
    else:
        weights = [float(n) for n in n_toks_per_source]

    if use_bs_schedule:
        local_batch_size = global_batch_size // world_size // seq_len
        bs_schedule: BatchSizeSchedule | None = mock_bs_schedule(
            global_batch_size=global_batch_size,
            n_batches=n_batches,
            local_batch_size=local_batch_size,
            first_level_batches=10,
        )
    else:
        bs_schedule = None

    dataloader_fn = make_dataloader_fn(
        rank=rank,
        world_size=world_size,
        prefix_paths=prefix_paths,
        weights=weights,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=n_batches,
        tmp_path=tmp_path,
        batch_size_schedule=bs_schedule,
        use_linear_weight_schedule=use_linear_weight_schedule,
    )

    with ExitStack() as exit_stack:
        ref_dataloader = dataloader_fn(multiprocessed_worker_info=None)
        ref_dataloader.start()

        mp_loader = MultiprocessedPretokenizedDataLoader(
            pretokenized_data_loader_fn=dataloader_fn,
            num_workers=4,
            max_size=2,
            timeout=10,
        )
        exit_stack.enter_context(_closing(mp_loader))
        mp_loader.start()

        for _ in range(n_batches):
            ref_batch = ref_dataloader.get_batch()
            ref_state = ref_dataloader.get_state()
            batch = mp_loader.get_batch()
            other_state = mp_loader.get_state()

            assert other_state == ref_state
            np.testing.assert_allclose(batch.x, ref_batch.x)
            np.testing.assert_allclose(batch.y, ref_batch.y)
            np.testing.assert_allclose(batch.sizes, ref_batch.sizes)
            assert batch.srcs == ref_batch.srcs


@pytest.mark.parametrize("use_bs_schedule", [False, True])
def test_exact_restart_with_bs_schedule(use_bs_schedule: bool, tmp_path: Path) -> None:
    """Upstream test: exact restart with batch size schedule parametrization."""
    n_batches = 100
    n_sources = 3
    seq_len = 5
    global_batch_size = 4 * seq_len
    rank = 1
    world_size = 2
    prefix_paths, _, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources
    )
    weights = [float(n) for n in n_toks_per_source]

    if use_bs_schedule:
        local_batch_size = global_batch_size // world_size // seq_len
        bs_schedule: BatchSizeSchedule | None = mock_bs_schedule(
            global_batch_size=global_batch_size,
            n_batches=n_batches,
            local_batch_size=local_batch_size,
            first_level_batches=10,
        )
    else:
        bs_schedule = None

    dataloader_fn = make_dataloader_fn(
        rank=rank,
        world_size=world_size,
        prefix_paths=prefix_paths,
        weights=weights,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=n_batches,
        tmp_path=tmp_path,
        batch_size_schedule=bs_schedule,
    )

    with ExitStack() as exit_stack:
        data_loader = MultiprocessedPretokenizedDataLoader(
            pretokenized_data_loader_fn=dataloader_fn,
            num_workers=2,
            max_size=2,
            timeout=10,
        )
        exit_stack.enter_context(_closing(data_loader))
        data_loader.start()

        rng = np.random.RandomState(seed=2)
        n_reset = 5
        reset = set(rng.permutation(range(n_batches))[:n_reset])

        new_data_loader = MultiprocessedPretokenizedDataLoader(
            pretokenized_data_loader_fn=dataloader_fn,
            num_workers=rng.randint(1, 4),
            max_size=2,
            timeout=10,
        )
        exit_stack.enter_context(_closing(new_data_loader))
        new_data_loader.start()

        for i in range(n_batches):
            ref_batch = data_loader.get_batch()
            ref_state = data_loader.get_state()
            batch = new_data_loader.get_batch()
            other_state = new_data_loader.get_state()
            assert other_state == ref_state

            np.testing.assert_allclose(batch.x, ref_batch.x)
            np.testing.assert_allclose(batch.y, ref_batch.y)
            np.testing.assert_allclose(batch.sizes, ref_batch.sizes)
            assert batch.srcs == ref_batch.srcs

            if i in reset:
                new_data_loader.close()
                new_data_loader = MultiprocessedPretokenizedDataLoader(
                    pretokenized_data_loader_fn=dataloader_fn,
                    num_workers=rng.randint(1, 4),
                    max_size=2,
                    timeout=10,
                )
                new_data_loader.set_state(ref_state)
                exit_stack.enter_context(_closing(new_data_loader))
                new_data_loader.start()


def test_images(tmp_path: Path) -> None:
    """Upstream test: multiprocessed dataloader works with image data."""
    n_batches = 100
    n_sources = 3
    seq_len = 5
    global_batch_size = 4 * seq_len
    rank = 1
    world_size = 2
    prefix_paths, _, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources, add_images=True
    )
    weights = [float(n) for n in n_toks_per_source]

    dataloader_fn = make_dataloader_fn(
        rank=rank,
        world_size=world_size,
        prefix_paths=prefix_paths,
        weights=weights,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=n_batches,
        tmp_path=tmp_path,
        batch_size_schedule=None,
        add_images=True,
    )

    with ExitStack() as exit_stack:
        data_loader = MultiprocessedPretokenizedDataLoader(
            pretokenized_data_loader_fn=dataloader_fn,
            num_workers=2,
            max_size=2,
            timeout=10,
        )
        exit_stack.enter_context(_closing(data_loader))
        data_loader.start()

        for _ in range(n_batches):
            _ = data_loader.get_batch()
