import logging
import typing
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from numpy.random import RandomState

from mistral_fork.mistral.bs_rampup import BatchSizeSchedule
from mistral_fork.mistral.data.data_loader import (
    Batch,
    DataLoader,
    DataLoaderState,
    DataLoaderStats,
)
from mistral_fork.mistral.data.interleaved_pretokenized.selection_args import (
    SelectionStrategy,
)
from mistral_fork.mistral.data.iterator.utils import SequenceOfSamples
from mistral_fork.mistral_common.tokens.tokenizers.base import Span
from mistral_fork.mistral.data.pretokenized.datasource import (
    EPOCH_INDEX_CODE_VERSION,
    Datasource,
    DatasourceSpecs,
    DatasourceState,
    PretokenizedImageEncoder,
)
from mistral_fork.mistral.data.pretokenized.pretokenized_args import PretokenizedArgs
from mistral_fork.mistral.data.pretokenized.pretokenized_utils import (
    SourceItem,
    _optimized_sample_datasource_ids,
    count_file_mappings,
    format_file_mapping,
    get_current_weights,
)
from mistral_fork.mistral.data.source import SampleType
from mistral_fork.mistral_common.base import MistralBase

if typing.TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class PretokenizedDataLoaderState(MistralBase):
    global_sequence_id: int
    datasources_states: list[DatasourceState]
    epoch_index_code_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "PretokenizedDataLoaderState":
        return cls.model_validate(data)


class PretokenizedDataLoader(DataLoader[Batch]):
    def __init__(
        self,
        source_items: list[SourceItem],
        seed: int,
        num_toks_in_global_batch: int,
        # Token budget used to size per-source epoch-index coverage.
        num_total_tokens: int,
        seq_len: int,
        rank: int,
        world_size: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        pretokenized_args: PretokenizedArgs,
        image_encoder_build_fn: Callable[[str], PretokenizedImageEncoder] | None,
        # when using multi worker per gpu, th n_workers create the same big batch but use a different stride
        # like this the n workers have the same dataloader state. This allows to have the same batches if using
        # 1 or many workers per gpu
        multiprocessed_worker_info: tuple[int, int] | None,
        selection_strategy: SelectionStrategy | None,
        batch_size_schedule: BatchSizeSchedule | None = None,
        optimized_datasource_id_sampling: bool = False,
        # When True, set_state may be called repeatedly (even after start) to
        # rewind a running loader to a previously captured state. Used by the
        # fixed validation iterator to replay the same batches each round
        # without rebuilding the loader.
        allow_state_resetting: bool = False,
        load_image_pixels: bool = True,
        max_image_group_workers: int = 1,
        in_phase_transition: bool = False,
        image_shard: tuple[int, int] | None = None,
    ) -> None:
        self._allow_non_sorted_sources_in_dataloader = (
            pretokenized_args.allow_non_sorted_sources_in_dataloader
        )
        self._source_items = source_items
        self._seed = seed
        self._global_batch_size = num_toks_in_global_batch
        self._num_total_tokens = num_total_tokens
        self._seq_len = seq_len
        self._rank = rank
        self._world_size = world_size
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._pad_id = pad_id
        self._optimized_datasource_id_sampling = optimized_datasource_id_sampling
        self._allow_state_resetting = allow_state_resetting
        self._in_phase_transition = in_phase_transition

        self.image_encoder_build_fn = image_encoder_build_fn
        self.multiprocessed_worker_info = multiprocessed_worker_info

        # TODO @baudouinMNC check this
        self.batch_size_schedule = batch_size_schedule
        self.selection_strategy = selection_strategy

        assert self._global_batch_size % (self._seq_len * self._world_size) == 0, (
            self._global_batch_size,
            self._world_size,
            self._seq_len,
        )

        self._static_num_sequences_by_global_batch = (
            (self._global_batch_size // self._seq_len)
            if self.batch_size_schedule is None
            else None
        )

        assert 0 <= self._rank < self._world_size

        self._state: PretokenizedDataLoaderState | None = None

        self._multiprocessed_worker_info = multiprocessed_worker_info
        if self._multiprocessed_worker_info is not None:
            multiprocess_worker_id, num_multiprocess_workers = (
                self._multiprocessed_worker_info
            )
            assert 0 <= multiprocess_worker_id < num_multiprocess_workers, (
                self._multiprocessed_worker_info
            )
        self._is_first_batch = True

        # Build datasources from source_items. Every multiprocessing worker
        # builds its own loader, so log the mapping from one of them rather
        # than repeating an identical block num_workers times per rank.
        if (
            self._multiprocessed_worker_info is None
            or self._multiprocessed_worker_info[0] == 0
        ):
            # One line per source is ~740 KB for a production blend, which
            # dwarfs everything else rank 0 logs. Summarise, and keep the
            # full listing available at DEBUG.
            logger.info(
                "File mapping: %d unique source -> pretokenized pairs "
                "(enable DEBUG on this logger for the full list)",
                count_file_mappings(source_items),
            )
            logger.debug(format_file_mapping(source_items))

        self._datasources = []
        for item in source_items:
            assert item.pretokenized_path is not None, "pretokenized_path must be set"
            self._datasources.append(
                Datasource(
                    specs=DatasourceSpecs(
                        prefix_path=str(item.pretokenized_path),
                        seed=seed,
                        seq_len=seq_len,
                        bos_id=bos_id,
                        eos_id=eos_id,
                        pad_id=pad_id,
                        no_shuffling=pretokenized_args.no_shuffling,
                        percentage_data=item.percentage_data,
                        no_partial_sequence=item.sample_type == SampleType.PRETRAIN_OCR,
                        truncate_trailing_masked=pretokenized_args.truncate_trailing_masked,
                    ),
                    num_toks=item.compute_safe_num_tokens(
                        num_total_tokens, sequence_length=seq_len
                    ),
                    cache_root_path=Path(pretokenized_args.epoch_index_cache_path),
                    image_encoder_build_fn=image_encoder_build_fn,
                    image_transform_type=item.image_transform_type,
                    selection_strategy=selection_strategy,
                    allow_epoch_index_building=pretokenized_args.allow_build_of_epoch_index,
                    allow_partial_epoch_index_cache=(
                        pretokenized_args.allow_partial_epoch_index_cache
                    ),
                    skip_images=pretokenized_args.skip_images,
                    drop_bos=pretokenized_args.drop_bos,
                    load_image_pixels=load_image_pixels,
                    max_image_group_workers=max_image_group_workers,
                    image_shard=image_shard,
                )
            )

    def start(self) -> None:
        if self._state is None:
            self._state = PretokenizedDataLoaderState(
                global_sequence_id=0,
                datasources_states=[ds.initial_state() for ds in self._datasources],
                epoch_index_code_version=EPOCH_INDEX_CODE_VERSION,
            )
        assert self._state is not None

    def get_batch_with_params(
        self, local_size_for_this_batch: int, weights: np.ndarray, skip: bool
    ) -> Batch | None:
        """
        We use `skip` option to update internal state, but not loading any data
        """
        assert self._state is not None
        global_size_for_this_batch = local_size_for_this_batch * self._world_size
        if self._optimized_datasource_id_sampling:
            datasource_ids = _optimized_sample_datasource_ids(
                seq_ids=range(
                    self._state.global_sequence_id,
                    self._state.global_sequence_id + global_size_for_this_batch,
                ),
                weights=weights,
                seed=self._seed,
            )
        else:
            datasource_ids = [
                RandomState(seq_id).choice(range(len(self._datasources)), p=weights)
                for seq_id in range(
                    self._state.global_sequence_id,
                    self._state.global_sequence_id + global_size_for_this_batch,
                )
            ]
        return self.get_batch_with_datasource_ids(
            local_size_for_this_batch=local_size_for_this_batch,
            datasource_ids=datasource_ids,
            skip=skip,
        )

    def get_batch_with_datasource_ids(
        self,
        local_size_for_this_batch: int,
        datasource_ids: typing.Sequence[int],
        skip: bool,
    ) -> Batch | None:
        """Build a batch from an explicit datasource choice for every global row.

        A datasource ID of -1 creates a padding row without advancing any
        datasource state.
        """

        assert self._state is not None
        global_size_for_this_batch = local_size_for_this_batch * self._world_size
        assert len(datasource_ids) == global_size_for_this_batch, (
            f"Expected {global_size_for_this_batch} datasource IDs, "
            f"got {len(datasource_ids)}"
        )
        assert all(-1 <= ds_id < len(self._datasources) for ds_id in datasource_ids), (
            "Explicit datasource IDs must be -1 for padding or index a configured "
            f"datasource, got {datasource_ids}"
        )

        self._state.global_sequence_id += global_size_for_this_batch
        seq_ids_in_datasource = [
            None
            if ds_id == -1
            else self._state.datasources_states[ds_id].current_seq_id_and_increment_()
            for ds_id in datasource_ids
        ]

        if skip:
            return None

        start = self._rank * local_size_for_this_batch
        end = (self._rank + 1) * local_size_for_this_batch

        local_datasources_ids = datasource_ids[start:end]
        local_saq_ids_ids_in_datasource = seq_ids_in_datasource[start:end]

        sequences: list[SequenceOfSamples] = []
        sequences_srcs: list[str] = []
        sequences_pretokenized_paths: list[str] = []
        for ds_id, seq_id_in_ds in zip(
            local_datasources_ids, local_saq_ids_ids_in_datasource, strict=True
        ):
            if ds_id == -1:
                sequences_srcs.append("__padding__")
                sequences_pretokenized_paths.append("")
                sequences.append(
                    SequenceOfSamples(
                        x=[self._pad_id] * self._seq_len,
                        y=[self._pad_id] * self._seq_len,
                        mask=[True] * self._seq_len,
                        sizes=[self._seq_len],
                        num_pad_tokens=self._seq_len,
                        doc_ids=[-1],
                    )
                )
                continue

            assert seq_id_in_ds is not None
            ds = self._datasources[ds_id]
            sequences_srcs.append(ds.src)
            sequences_pretokenized_paths.append(ds._prefix_path)
            sequences.append(ds.get_sequence(seq_id_in_ds))

        assert len(sequences) > 0  # it's impossible to guess the error later

        batch = batch_from_sequences(
            sample_type=SampleType.PRETRAIN,
            sequences=sequences,
            sequences_srcs=sequences_srcs,
            sequences_pretokenized_paths=sequences_pretokenized_paths,
        )
        return batch

    def _get_next_local_bsz_and_weights(self) -> tuple[int, np.ndarray]:
        assert self._state is not None
        tokens_processed = self._state.global_sequence_id * self._seq_len
        if self.batch_size_schedule is not None:
            rampup_level = self.batch_size_schedule.get_level_by_tokens(
                tokens_processed
            )
            local_bsz = rampup_level.local_batch_size
        else:
            assert self._static_num_sequences_by_global_batch is not None
            local_bsz = self._static_num_sequences_by_global_batch // self._world_size
        cur_weights = get_current_weights(self._source_items, tokens_processed)
        return local_bsz, cur_weights

    def _get_batch_or_skip(self, skip: bool) -> Batch | None:
        cur_local_bsz, cur_weights = self._get_next_local_bsz_and_weights()
        return self.get_batch_with_params(
            local_size_for_this_batch=cur_local_bsz, weights=cur_weights, skip=skip
        )

    def get_batch(self) -> Batch:
        if self._state is None:
            self.start()

        assert self._state is not None
        if self._multiprocessed_worker_info:
            # since other workers will also sample sequences, we need to skip some sequences
            # doing it here allows to have self._state always pointing to the state just after sampling
            worker_id, num_workers = self._multiprocessed_worker_info
            num_batches_to_skip = worker_id if self._is_first_batch else num_workers - 1
            for _ in range(num_batches_to_skip):
                out = self._get_batch_or_skip(skip=True)
                assert out is None

        self._is_first_batch = False
        batch = self._get_batch_or_skip(skip=False)
        assert batch is not None
        return batch

    def get_state(self) -> DataLoaderState:
        assert self._state is not None, "please start the dataloader first"
        return self._state.to_dict()

    def _maybe_reorder_datasources_to_match_saved_state(
        self, parsed_state: PretokenizedDataLoaderState
    ) -> None:
        index_by_path = {
            data_source._specs.prefix_path: i
            for i, data_source in enumerate(self._datasources)
        }
        assert len(index_by_path) == len(self._datasources), (
            "Duplicate prefix_path in datasources"
        )
        saved_paths = [
            state.specs.prefix_path for state in parsed_state.datasources_states
        ]
        if set(saved_paths) != set(index_by_path):
            raise ValueError(
                "Datasource set mismatch between checkpoint and current config. "
                f"delta={set(saved_paths) ^ set(index_by_path)}"
            )
        saved_state_order = [index_by_path[path] for path in saved_paths]
        if saved_state_order == list(range(len(saved_state_order))):
            return

        logger.warning(
            f"Reordering {len(saved_state_order)} datasources to match the "
            f"checkpoint order."
        )
        self._datasources = [self._datasources[i] for i in saved_state_order]
        self._source_items = [self._source_items[i] for i in saved_state_order]

    def _reconcile_phase_transition_state(
        self, parsed_state: PretokenizedDataLoaderState
    ) -> PretokenizedDataLoaderState:
        saved_state_by_path = {
            state.specs.prefix_path: state
            for state in parsed_state.datasources_states
        }
        if len(saved_state_by_path) != len(parsed_state.datasources_states):
            raise ValueError("Duplicate prefix_path in checkpoint datasources")

        current_paths = [data_source.specs.prefix_path for data_source in self._datasources]
        if len(set(current_paths)) != len(current_paths):
            raise ValueError("Duplicate prefix_path in datasources")

        reconciled_states = []
        for data_source in self._datasources:
            data_source_state = saved_state_by_path.get(data_source.specs.prefix_path)
            if data_source_state is None:
                data_source_state = data_source.initial_state()
            data_source.check_state(data_source_state)
            reconciled_states.append(data_source_state)

        return PretokenizedDataLoaderState(
            global_sequence_id=parsed_state.global_sequence_id,
            datasources_states=reconciled_states,
            epoch_index_code_version=parsed_state.epoch_index_code_version,
        )

    def set_state(self, state: DataLoaderState) -> None:
        if not self._allow_state_resetting:
            assert self._state is None, (
                self._state,
                "dataloader was already started, or set was already set!",
            )
        parsed_state = PretokenizedDataLoaderState.from_dict(state)
        if parsed_state.epoch_index_code_version != EPOCH_INDEX_CODE_VERSION:
            raise ValueError(
                f"Epoch index code version mismatch: checkpoint has version "
                f"{parsed_state.epoch_index_code_version}, but current code uses "
                f"version {EPOCH_INDEX_CODE_VERSION}. Cannot resume training with "
                f"different epoch index versions as data ordering would be inconsistent."
            )
        if self._in_phase_transition:
            self._state = self._reconcile_phase_transition_state(parsed_state)
            return
        if self._allow_non_sorted_sources_in_dataloader:
            self._maybe_reorder_datasources_to_match_saved_state(parsed_state)
        assert len(parsed_state.datasources_states) == len(self._datasources), (
            len(parsed_state.datasources_states),
            len(self._datasources),
            "did you change the datasources?",
        )
        for data_source, data_source_state in zip(
            self._datasources, parsed_state.datasources_states, strict=True
        ):
            data_source.check_state(data_source_state)
        self._state = parsed_state

    def close(self) -> None:
        pass

    def set_wrapped_by_closing(self, value: bool) -> None:
        pass

    @property
    def max_steps(self) -> int:
        # TODO(@thib): setting to inf (even if not true) since we don't need to know the number of max_steps (which is
        #  needed for other dataloaders.
        return float("inf")  # type: ignore

    @property
    def source_names(self) -> set[str]:
        return set(ds.src for ds in self._datasources)

    def unwrap(self) -> Optional["DataLoader"] | list[Optional["DataLoader"]]:
        return self

    def get_stats(self) -> DataLoaderStats:
        return {}


def batch_from_sequences(
    sample_type: SampleType,
    sequences: list[SequenceOfSamples],
    sequences_srcs: list[str],
    sequences_pretokenized_paths: list[str],
) -> Batch:
    x: list[int] = []
    y: list[int] = []
    masks: list[bool] = []
    srcs: list[str] = []
    pretokenized_paths: list[str] = []
    doc_ids: list[int] = []
    sizes: list[int] = []
    logical_sizes: list[int] = []
    images: list[torch.Tensor] = []
    image_positions: list[Span] = []
    num_text_tokens_by_src: dict[str, int] = defaultdict(int)
    num_image_tokens_by_src: dict[str, int] = defaultdict(int)
    num_pad_tokens = 0
    timings: dict[str, float] = defaultdict(float)

    for src, item, path in zip(
        sequences_srcs, sequences, sequences_pretokenized_paths, strict=True
    ):
        sequence_offset = len(x)
        x.extend(item.x)
        y.extend(item.y)
        masks.extend(item.mask)
        n_samples = len(item.sizes) if item.sizes else 1
        srcs.extend([src] * n_samples)
        pretokenized_paths.extend([path] * n_samples)
        doc_ids.extend(item.doc_ids)
        sizes.extend(item.sizes)
        logical_sizes.extend(item.sizes)
        # pad_sequence appends storage padding to the final document; a
        # padding-only sample is the same case with zero remaining real rows.
        if item.num_pad_tokens:
            logical_sizes[-1] -= item.num_pad_tokens
        images.extend(item.images)
        image_positions.extend(position + sequence_offset for position in item.image_positions)
        num_image_tokens = sum(span.end - span.start for span in item.image_positions)
        num_text_tokens = len(item.x) - num_image_tokens
        num_image_tokens_by_src[src] += num_image_tokens
        num_text_tokens_by_src[src] += num_text_tokens
        num_pad_tokens += item.num_pad_tokens
        for k, v in item.timings.items():
            timings[k] += v

    np_mask: np.ndarray = np.array(masks, dtype=bool)
    assert all(s >= 0 for s in sizes)

    return Batch(
        sample_type=sample_type,
        x=np.array(x, dtype=np.int64),
        y=np.array(y, dtype=np.int64),
        y_mask=None if np_mask.all() else np_mask,
        sizes=np.array(sizes, dtype=np.int64),
        srcs=srcs,
        images=(images if len(images) > 0 else None),
        image_positions=image_positions,
        num_text_tokens_by_src=dict(num_text_tokens_by_src),
        num_image_tokens_by_src=dict(num_image_tokens_by_src),
        num_pad_tokens=num_pad_tokens,
        timings=timings,
        sample_pretokenized_paths=pretokenized_paths,
        sample_doc_ids=doc_ids,
        logical_sizes=np.array(logical_sizes, dtype=np.int64) if num_pad_tokens else None,
    )
