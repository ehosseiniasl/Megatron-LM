import copy
import dataclasses
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from mistral_fork.mistral.bs_rampup import BatchSizeSchedule, RampupLevel
from mistral_fork.mistral.data.interleaved_pretokenized.selection_args import (
    SelectionStrategy,
)
from mistral_fork.mistral.data.pretokenized.datasource_test import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    _build_datasource,
    parse_token_id,
)
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.pretokenized_args import PretokenizedArgs
from mistral_fork.mistral.data.pretokenized.pretokenized_data_loader import (
    PretokenizedDataLoader,
    PretokenizedDataLoaderState,
)
from mistral_fork.mistral.data.pretokenized.pretokenized_utils import (
    ConstantWeightSchedule,
    SourceItem,
)
from mistral_fork.mistral.data.source import SampleType

logger = logging.getLogger("Pretokenized Test")


class DataLoaderCallable(Protocol):
    def __call__(self, rank: int, world_size: int) -> PretokenizedDataLoader: ...


def mock_bs_schedule(
    global_batch_size: int,
    n_batches: int,
    local_batch_size: int,
    first_level_batches: int = 10,
) -> BatchSizeSchedule:
    rampup_levels = [
        RampupLevel(
            global_batch_size=global_batch_size // 2,
            steps=0,
            local_batch_size=local_batch_size // 2,
            tokens_processed=0,
            num_microbatches=1,
            max_lr_rescale=1.0,
        ),
        RampupLevel(
            global_batch_size=global_batch_size,
            steps=first_level_batches,
            local_batch_size=local_batch_size,
            tokens_processed=global_batch_size // 2 * first_level_batches,
            num_microbatches=1,
            max_lr_rescale=1.0,
        ),
    ]
    return BatchSizeSchedule(rampup_levels)


def make_dataloader_fn(
    source_items: list[SourceItem],
    seq_len: int,
    n_batches: int,
    global_batch_size: int,
    tmp_path: Path,
    with_mock_bs_schedule: bool = False,
    selection_strategy: SelectionStrategy | None = None,
    optimized_datasource_id_sampling: bool = False,
    allow_non_sorted_sources_in_dataloader: bool = False,
    multiprocessed_worker_info: tuple[int, int] | None = None,
    in_phase_transition: bool = False,
) -> DataLoaderCallable:
    def _dataloader(rank: int, world_size: int) -> PretokenizedDataLoader:
        if with_mock_bs_schedule:
            rampup_length = 10
            num_total_tokens = (
                n_batches * global_batch_size - global_batch_size // 2 * rampup_length
            )
            local_batch_size = global_batch_size // world_size // seq_len
            bs_schedule = mock_bs_schedule(
                global_batch_size=global_batch_size,
                n_batches=n_batches,
                local_batch_size=local_batch_size,
                first_level_batches=rampup_length,
            )
        else:
            num_total_tokens = n_batches * global_batch_size
            bs_schedule = None

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
                allow_build_of_epoch_index=True,
                allow_non_sorted_sources_in_dataloader=(
                    allow_non_sorted_sources_in_dataloader
                ),
            ),
            batch_size_schedule=bs_schedule,
            image_encoder_build_fn=None,
            selection_strategy=selection_strategy,
            optimized_datasource_id_sampling=optimized_datasource_id_sampling,
            in_phase_transition=in_phase_transition,
        )

    return _dataloader


def prepare_data_sources_on_disk(
    seed: int, root_path: Path, n_sources: int, add_images: bool = False
) -> tuple[list[str], set[int], list[int]]:
    rng = np.random.RandomState(seed)
    present_tokens: set[int] = set()
    prefix_paths = []
    n_toks_per_source = []
    for datasource_id in range(n_sources):
        datasource_dst = root_path / f"source_{datasource_id}"
        prefix_path, datasource_tokens, _, n_toks, datasource_imgs_ids = (
            _build_datasource(
                datasource_dst,
                original_source=f"original_source_{datasource_id}",
                max_seq_length=100,  # longer seqlen helps probabilistic tests like token and sequence dropout
                rng=rng,
                datasource_id=datasource_id,
                n_docs=(datasource_id + 1) * 20,
                add_images=add_images,
            )
        )
        prefix_paths.append(str(prefix_path))
        n_toks_per_source.append(n_toks)
        assert not (datasource_tokens & present_tokens), (
            datasource_tokens & present_tokens
        )
        present_tokens.update(datasource_tokens)
    return prefix_paths, present_tokens, n_toks_per_source


def create_source_items(
    prefix_paths: list[str], weights: list[float]
) -> list[SourceItem]:
    """Create SourceItem objects from prefix_paths and normalized weights."""
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    return [
        SourceItem(
            source_path=Path(p),
            sample_type=SampleType.PRETRAIN,
            weight_schedule=ConstantWeightSchedule(w),
            percentage_data=1.0,
            filter_profile=FilterArgsName.NO_FILTER,
            pretokenized_path=Path(p),
        )
        for p, w in zip(prefix_paths, normalized_weights, strict=True)
    ]


@pytest.mark.parametrize("optimized_datasource_id_sampling", [False, True])
def test_pretokenized_data_loader_index(
    tmp_path: Path, optimized_datasource_id_sampling: bool
) -> None:
    n_sources = 3
    seq_len = 5
    global_batch_size = 4 * seq_len
    world_size = 1
    prefix_paths, present_tokens, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources
    )
    weights = [float(n) for n in n_toks_per_source]
    source_items = create_source_items(prefix_paths, weights)

    dataloader_fn = make_dataloader_fn(
        source_items=source_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=math.ceil(2 * sum(n_toks_per_source) / global_batch_size),
        tmp_path=tmp_path,
        optimized_datasource_id_sampling=optimized_datasource_id_sampling,
    )

    data_loader = dataloader_fn(rank=0, world_size=world_size)

    # TODO(@thib): introduce concept of epochs per source, and check that there is no duplicated inside epochs and
    #  that all tokens were seen (modulo the remaing tokens) at each epoch

    seen_tokens: set[int] = set()
    seen_labels: set[int] = set()
    toks_per_source_counts: dict[int, int] = defaultdict(int)
    for _ in range(100):
        batch = data_loader.get_batch()

        assert len(batch.x) == (global_batch_size // world_size)

        for tok in batch.x:
            if tok == BOS_ID:
                continue
            assert tok != EOS_ID
            assert tok not in seen_tokens, (tok, seen_tokens)
            seen_tokens.add(tok)

            source_id, _, _ = parse_token_id(tok)
            toks_per_source_counts[source_id] += 1

        for label in batch.y:
            if label == EOS_ID:
                continue
            assert label != BOS_ID
            assert label not in seen_labels, (label, seen_labels)
            seen_labels.add(label)

    actual_proportions = np.array(
        [float(toks_per_source_counts[i]) for i in range(n_sources)]
    )
    actual_proportions /= np.sum(actual_proportions)

    expected_proportions = np.array(weights)
    expected_proportions /= np.sum(expected_proportions)

    assert np.max(np.abs(expected_proportions - actual_proportions)) < 0.05


def test_get_batch_with_params_uses_passed_weights(tmp_path: Path) -> None:
    """Test that get_batch_with_params uses the passed weights, not internal weights.

    This test verifies that when calling get_batch_with_params with specific weights,
    the dataloader respects those weights rather than using the weights from SourceItems.
    This is critical for DoublePretokenizedDataLoader which computes weights externally.
    """
    n_sources = 2
    seq_len = 5
    global_batch_size = 4 * seq_len
    world_size = 1
    n_batches = 50

    prefix_paths, _, _ = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources
    )
    # Create source items with equal weights (0.5, 0.5)
    source_items = create_source_items(prefix_paths, [1.0, 1.0])

    dataloader_fn = make_dataloader_fn(
        source_items=source_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=n_batches,
        tmp_path=tmp_path,
    )

    data_loader = dataloader_fn(rank=0, world_size=world_size)
    data_loader.start()

    # Call get_batch_with_params with weights that select ONLY source 0
    weights_only_source_0 = np.array([1.0, 0.0])
    local_batch_size = global_batch_size // world_size // seq_len

    for _ in range(10):
        batch = data_loader.get_batch_with_params(
            local_size_for_this_batch=local_batch_size,
            weights=weights_only_source_0,
            skip=False,
        )
        assert batch is not None
        # All tokens should be from source 0
        for tok in batch.x:
            if tok == BOS_ID:
                continue
            source_id, _, _ = parse_token_id(tok)
            assert source_id == 0, (
                f"Expected source 0, got source {source_id}. "
                "get_batch_with_params may not be using passed weights."
            )

    # Now call with weights that select ONLY source 1
    weights_only_source_1 = np.array([0.0, 1.0])
    for _ in range(10):
        batch = data_loader.get_batch_with_params(
            local_size_for_this_batch=local_batch_size,
            weights=weights_only_source_1,
            skip=False,
        )
        assert batch is not None
        # All tokens should be from source 1
        for tok in batch.x:
            if tok == BOS_ID:
                continue
            source_id, _, _ = parse_token_id(tok)
            assert source_id == 1, (
                f"Expected source 1, got source {source_id}. "
                "get_batch_with_params may not be using passed weights."
            )


def test_exact_restart(tmp_path: Path) -> None:
    n_sources = 3
    seq_len = 5
    global_batch_size = 4 * seq_len
    rank = 0
    world_size = 2
    n_batches = 100
    prefix_paths, present_tokens, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources
    )
    weights = [float(n) for n in n_toks_per_source]
    source_items = create_source_items(prefix_paths, weights)

    dataloader_fn = make_dataloader_fn(
        source_items=source_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=n_batches,
        tmp_path=tmp_path,
        with_mock_bs_schedule=True,
    )

    data_loader = dataloader_fn(rank=rank, world_size=world_size)

    expected_batches = []
    states = []
    for _ in range(100):
        expected_batches.append(data_loader.get_batch())
        states.append(data_loader.get_state())

    n_reset = 10

    reset = set(np.random.RandomState(seed=2).permutation(range(100))[:n_reset])
    print(f"{reset=}")

    new_data_loader = dataloader_fn(rank=rank, world_size=world_size)

    for i in range(n_batches):
        batch = new_data_loader.get_batch()

        np.testing.assert_allclose(batch.x, expected_batches[i].x)
        np.testing.assert_allclose(batch.y, expected_batches[i].y)
        np.testing.assert_allclose(batch.sizes, expected_batches[i].sizes)
        assert batch.srcs == expected_batches[i].srcs
        if i in reset:
            new_data_loader = dataloader_fn(rank=rank, world_size=world_size)
            new_data_loader.set_state(states[i])


def test_crash_if_seed_or_seqlen_different_when_reloading(tmp_path: Path) -> None:
    n_sources = 3
    seq_len = 5
    seed = 0
    global_batch_size = 4 * seq_len
    prefix_paths, present_tokens, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources
    )
    weights = [float(n) for n in n_toks_per_source]
    source_items = create_source_items(prefix_paths, weights)
    n_batches = 10

    make = make_dataloader_fn(
        source_items=source_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=n_batches,
        tmp_path=tmp_path,
    )

    first = make(rank=0, world_size=1)
    for _ in range(n_batches // 2):
        _ = first.get_batch()
    dict_state = first.get_state()
    assert dict_state is not None
    state = PretokenizedDataLoaderState.from_dict(dict_state)

    second = make(rank=0, world_size=1)
    state_with_different_seqlen: PretokenizedDataLoaderState = copy.deepcopy(state)
    for ds_state in state_with_different_seqlen.datasources_states:
        ds_state.specs = dataclasses.replace(ds_state.specs, seq_len=2 * seq_len)

    with pytest.raises(
        AssertionError,
        match="Mismatch in specs between received state and datasource specs",
    ):
        second.set_state(state_with_different_seqlen.to_dict())

    third = make(rank=0, world_size=1)
    state_with_different_seed = copy.deepcopy(state)
    for ds_state in state_with_different_seed.datasources_states:
        ds_state.specs = dataclasses.replace(ds_state.specs, seed=seed + 1)

    with pytest.raises(
        AssertionError,
        match="Mismatch in specs between received state and datasource specs",
    ):
        third.set_state(state_with_different_seed.to_dict())


def test_resume_reorders_datasources_to_saved_order(tmp_path: Path) -> None:
    n_sources = 2
    seq_len = 5
    global_batch_size = 4 * seq_len
    prefix_paths, _, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources
    )
    source_items = create_source_items(
        prefix_paths, [float(n) for n in n_toks_per_source]
    )

    original_make = make_dataloader_fn(
        source_items=source_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=10,
        tmp_path=tmp_path,
    )
    original = original_make(rank=0, world_size=1)
    original.start()
    state = original.get_state()

    reordered_items = list(reversed(source_items))
    reordered_make = make_dataloader_fn(
        source_items=reordered_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=10,
        tmp_path=tmp_path,
        allow_non_sorted_sources_in_dataloader=True,
    )
    reordered = reordered_make(rank=0, world_size=1)
    assert [ds._specs.prefix_path for ds in reordered._datasources] == [
        str(item.pretokenized_path) for item in reordered_items
    ]

    reordered.set_state(state)

    assert [ds._specs.prefix_path for ds in reordered._datasources] == [
        str(item.pretokenized_path) for item in source_items
    ]


def test_resume_reordered_datasources_fails_by_default(tmp_path: Path) -> None:
    n_sources = 2
    seq_len = 5
    global_batch_size = 4 * seq_len
    prefix_paths, _, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources
    )
    source_items = create_source_items(
        prefix_paths, [float(n) for n in n_toks_per_source]
    )

    original_make = make_dataloader_fn(
        source_items=source_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=10,
        tmp_path=tmp_path,
    )
    original = original_make(rank=0, world_size=1)
    original.start()
    state = original.get_state()

    reordered_make = make_dataloader_fn(
        source_items=list(reversed(source_items)),
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=10,
        tmp_path=tmp_path,
    )
    reordered = reordered_make(rank=0, world_size=1)

    with pytest.raises(
        AssertionError,
        match="Mismatch in specs between received state and datasource specs",
    ):
        reordered.set_state(state)


def test_phase_transition_reconciles_changed_datasource_set(tmp_path: Path) -> None:
    seq_len = 5
    global_batch_size = 4 * seq_len
    prefix_paths, _, _ = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=3
    )
    shared_path, removed_path, added_path = prefix_paths

    phase_1_items = create_source_items([shared_path, removed_path], [3.0, 1.0])
    phase_1 = make_dataloader_fn(
        source_items=phase_1_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=100,
        tmp_path=tmp_path,
    )(rank=0, world_size=1)
    phase_1.start()
    saved_state = PretokenizedDataLoaderState.from_dict(phase_1.get_state())

    phase_2_items = create_source_items([added_path, shared_path], [2.0, 1.0])
    phase_2_make = make_dataloader_fn(
        source_items=phase_2_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=1,
        tmp_path=tmp_path,
        in_phase_transition=True,
    )
    phase_2 = phase_2_make(rank=0, world_size=1)
    shared_datasource = phase_2._datasources[1]
    initial_shared_coverage = shared_datasource._cum_sequence_count_per_epoch[-1]
    shared_state = next(
        state
        for state in saved_state.datasources_states
        if state.specs.prefix_path == shared_path
    )
    shared_state.sequence_id = initial_shared_coverage - 1
    shared_state.num_toks_seen = shared_state.sequence_id * seq_len
    shared_state.num_epochs = shared_state.num_toks_seen / shared_state.num_toks_per_epoch
    removed_state = next(
        state
        for state in saved_state.datasources_states
        if state.specs.prefix_path == removed_path
    )
    removed_state.current_seq_id_and_increment_()
    saved_state.global_sequence_id = 123
    required_coverage = shared_state.sequence_id + shared_datasource._num_sequences_to_validate
    assert initial_shared_coverage <= required_coverage

    phase_2.set_state(saved_state.to_dict())

    reconciled_state = PretokenizedDataLoaderState.from_dict(phase_2.get_state())
    assert [state.specs.prefix_path for state in reconciled_state.datasources_states] == [
        added_path,
        shared_path,
    ]
    assert [state.sequence_id for state in reconciled_state.datasources_states] == [
        0,
        shared_state.sequence_id,
    ]
    assert reconciled_state.global_sequence_id == 123
    assert [item.weight_schedule.weight for item in phase_2._source_items] == pytest.approx(
        [2 / 3, 1 / 3]
    )
    assert phase_1_items[0].weight_schedule.weight == pytest.approx(3 / 4)
    assert shared_datasource._cum_sequence_count_per_epoch[-1] > required_coverage

    changed_specs_state = copy.deepcopy(saved_state)
    changed_shared_state = next(
        state
        for state in changed_specs_state.datasources_states
        if state.specs.prefix_path == shared_path
    )
    changed_shared_state.specs = dataclasses.replace(
        changed_shared_state.specs, percentage_data=0.5
    )
    with pytest.raises(
        AssertionError,
        match="Mismatch in specs between received state and datasource specs",
    ):
        phase_2_make(rank=0, world_size=1).set_state(changed_specs_state.to_dict())

    ordinary_resume = make_dataloader_fn(
        source_items=phase_2_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=1,
        tmp_path=tmp_path,
        allow_non_sorted_sources_in_dataloader=True,
    )(rank=0, world_size=1)
    with pytest.raises(ValueError, match="Datasource set mismatch"):
        ordinary_resume.set_state(saved_state.to_dict())


def test_global_batch_independent_of_topology(tmp_path: Path) -> None:
    start = time.time()

    n_batches = 20

    n_sources = 3
    seq_len = 5
    global_batch_size = 4 * seq_len
    prefix_paths, present_tokens, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources
    )
    weights = [float(n) for n in n_toks_per_source]
    source_items = create_source_items(prefix_paths, weights)

    print(time.time() - start)

    _dataloader = make_dataloader_fn(
        source_items=source_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=n_batches,
        tmp_path=tmp_path,
    )

    ref_dataloader = _dataloader(rank=0, world_size=1)
    print(time.time() - start, "ref")

    other_dataloaders = [_dataloader(rank=r, world_size=4) for r in range(4)]

    print(time.time() - start, "other_dataloaders")

    for i in range(n_batches):
        ref_batch = ref_dataloader.get_batch()
        ref_state = ref_dataloader.get_state()
        other_batches = [dl.get_batch() for dl in other_dataloaders]

        # all dataloader states are identical
        for dl in other_dataloaders:
            assert dl.get_state() == ref_state

        np.testing.assert_allclose(
            ref_batch.x, np.concatenate([b.x for b in other_batches])
        )
        np.testing.assert_allclose(
            ref_batch.y, np.concatenate([b.y for b in other_batches])
        )
        np.testing.assert_allclose(
            ref_batch.sizes, np.concatenate([b.sizes for b in other_batches])
        )


def test_pretokenized_dl_can_skip_loading_batch(tmp_path: Path) -> None:
    n_batches = 20
    n_sources = 3
    seq_len = 5
    global_batch_size = 4 * seq_len
    world_size = 1
    prefix_paths, present_tokens, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=n_sources
    )
    weights = [float(n) for n in n_toks_per_source]
    source_items = create_source_items(prefix_paths, weights)

    dataloader_fn = make_dataloader_fn(
        source_items=source_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=math.ceil(2 * sum(n_toks_per_source) / global_batch_size),
        tmp_path=tmp_path,
        optimized_datasource_id_sampling=True,
    )
    dl = dataloader_fn(rank=0, world_size=world_size)
    # We run the exact same seed in the mock, thus we can compare the states
    skip_dl = dataloader_fn(rank=0, world_size=world_size)
    dl.start()
    skip_dl.start()
    for _ in range(n_batches):
        dl._get_batch_or_skip(skip=False)
        skip_dl._get_batch_or_skip(skip=True)

    assert skip_dl.get_state() == dl.get_state()


# DISABLED: test_fim
# Requires: PretrainFIMArgs, PostTokenizationFIMArgs (FIM removed from minimal codebase)

# DISABLED: test_dataloader_state_schema_hash
# Requires: ConfStore, DataArgs, data_args_to_source_items

# DISABLED: test_pretokenized_flexible_image_encoder_uuid_stability
# Requires: SpecialImageIDs, PretokenizedFlexibleImageEncoder, ImageTransformTypeForTest


@pytest.mark.parametrize(
    "worker_info,expect_mapping",
    [(None, True), ((0, 4), True), ((1, 4), False), ((3, 4), False)],
    ids=["single_process", "worker_0", "worker_1", "worker_3"],
)
def test_file_mapping_is_logged_once_per_rank(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    worker_info: tuple[int, int] | None,
    expect_mapping: bool,
) -> None:
    """Every worker builds its own loader; only one of them logs the mapping."""
    seq_len = 5
    global_batch_size = 4 * seq_len
    prefix_paths, _, n_toks_per_source = prepare_data_sources_on_disk(
        seed=0, root_path=tmp_path, n_sources=2
    )
    source_items = create_source_items(prefix_paths, [float(n) for n in n_toks_per_source])

    dataloader_fn = make_dataloader_fn(
        source_items=source_items,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        n_batches=math.ceil(sum(n_toks_per_source) / global_batch_size),
        tmp_path=tmp_path,
        multiprocessed_worker_info=worker_info,
    )

    with caplog.at_level(logging.INFO):
        dataloader_fn(rank=0, world_size=1)

    assert ("File mapping:" in caplog.text) is expect_mapping


# TODO tests:
#  - deterministic and change with seed

if __name__ == "__main__":
    pass
