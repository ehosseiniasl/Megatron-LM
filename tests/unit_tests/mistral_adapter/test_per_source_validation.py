# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from examples.mimo.data import mistral_provider
from examples.mimo.data.mistral_provider import (
    build_mistral_encoder_batch,
    build_mistral_language_batch,
)
from examples.mimo.training.step import loss_func
from megatron.training.training import (
    _accumulate_per_source_loss_reports,
    _finalize_per_source_loss_report,
)
from mistral_adapter import per_source_validation as validation
from mistral_adapter.per_source_validation import (
    PER_SOURCE_LOSS_KEY,
    PerSourcePretokenizedIterator,
    build_per_source_loss_report,
    expand_source_ids_to_token_shape,
    get_per_source_validation_metadata,
    resolve_per_source_validation_schedule,
    select_top_weighted_validation_sources,
)
from mistral_adapter.pretokenized_config import (
    PretokenizedPerSourceValidationConfig,
    PretokenizedPerSourceValidationSourceConfig,
)
from mistral_adapter.pretokenized_iterator import MistralPretokenizedIterator
from mistral_adapter.train_loop_helpers import add_pretokenized_args
from mistral_fork.mistral.data.iterator.utils import SequenceOfSamples
from mistral_fork.mistral.data.pretokenized import pretokenized_data_loader as loader_module
from mistral_fork.mistral.data.pretokenized.pretokenized_data_loader import (
    PretokenizedDataLoader,
    batch_from_sequences,
)
from mistral_fork.mistral.data.source import SampleType
from mistral_fork.mistral_common.tokens.tokenizers.base import Span


def test_metadata_discovery_through_wrappers_and_virtual_pipeline():
    inner = SimpleNamespace(source_names=("books", "code"), input_token_budgets_by_source=(16, 16))
    wrapped = SimpleNamespace(iterable=SimpleNamespace(_inner=inner))

    metadata = get_per_source_validation_metadata([wrapped, wrapped])

    assert metadata.source_names == ("books", "code")
    assert metadata.input_token_budgets_by_source == (16, 16)
    assert metadata.num_sources == 2


def test_metadata_disagreement_across_virtual_pipeline_fails():
    first = SimpleNamespace(source_names=("books",), input_token_budgets_by_source=(16,))
    second = SimpleNamespace(source_names=("code",), input_token_budgets_by_source=(16,))

    with pytest.raises(AssertionError, match="metadata disagrees"):
        get_per_source_validation_metadata([first, second])


def test_none_metadata_on_wrapper_is_not_per_source_validation():
    wrapped = SimpleNamespace(
        source_names=None, input_token_budgets_by_source=None, _inner=SimpleNamespace()
    )

    assert get_per_source_validation_metadata(wrapped) is None


def test_scored_token_metadata_does_not_enable_per_source_validation():
    iterator = SimpleNamespace(source_names=("books",), expected_eval_tokens_by_source=(16,))

    assert get_per_source_validation_metadata(iterator) is None


def test_fixed_reporting_iterator_replays_the_same_validation_batches():
    class Replay:
        def __init__(self):
            self.state = 0

        def ensure_started(self):
            pass

        def save_state(self):
            return {"state": self.state}

        def set_state(self, state):
            self.state = state["state"]

        def __next__(self):
            batch = {"source_ids": torch.tensor([self.state % 2])}
            self.state += 1
            return batch

        def close(self):
            pass

    fixed = validation._FixedIterator(Replay(), eval_iters=2, reset_period=2)
    iterator = validation._ReportingIterator(fixed, ["code", "docs"], 16)

    assert [next(iterator)["source_ids"].item() for _ in range(4)] == [0, 1, 0, 1]
    assert get_per_source_validation_metadata(iterator).source_names == ("code", "docs")


def test_validation_iterator_applies_transform_batch(monkeypatch):
    class Config:
        epoch_index_cache_path = "/cache"

        def model_copy(self, update):
            return self

    class RawIterator:
        def __init__(self, **_kwargs):
            self.state = 0

        def ensure_started(self):
            pass

        def save_state(self):
            return {"state": self.state}

        def set_state(self, state):
            self.state = state["state"]

        def __next__(self):
            self.state += 1
            return {"source_ids": torch.tensor([0])}

        def close(self):
            pass

    per_source = PretokenizedPerSourceValidationConfig(
        sources=[
            PretokenizedPerSourceValidationSourceConfig(
                name="code", prefix_path="/code", weight=1.0, percentage_data=1.0
            )
        ],
        valid_seed=17,
    )
    monkeypatch.setattr(validation, "PerSourcePretokenizedIterator", RawIterator)
    monkeypatch.setattr(
        validation, "apply_pretokenized_config_overrides", lambda config, **_kwargs: config
    )

    iterator, eval_iters = validation.build_per_source_validation_iterator(
        args=SimpleNamespace(
            seq_length=4,
            eval_iters=0,
            input_tokens_per_source=4,
            overwrite_percentage_data_multiplier=None,
        ),
        config=Config(),
        per_source_validation=per_source,
        dp_rank=0,
        dp_world_size=1,
        eval_global_batch_size=1,
        eval_micro_batch_size=1,
        eval_num_microbatches=1,
        transform_batch=lambda batch: {**batch, "transformed": True},
    )

    assert eval_iters == 1
    assert next(iterator)["transformed"] is True


def test_source_weights_only_size_equal_epoch_index_budgets():
    sources = [
        PretokenizedPerSourceValidationSourceConfig(
            name="code-a", prefix_path="/code/a", weight=1.0, percentage_data=1.0
        ),
        PretokenizedPerSourceValidationSourceConfig(
            name="code-b", prefix_path="/code/b", weight=3.0, percentage_data=1.0
        ),
    ]

    loader_sources = validation._sources_with_equal_index_budgets(sources)

    assert [source.prefix_path for source in loader_sources] == ["/code/a", "/code/b"]
    assert [source.weight for source in loader_sources] == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize(
    ("cli", "expected"),
    [
        ([], 131072),
        (["--input-tokens-per-source", "0"], 0),
        (["--input-tokens-per-source", "262144"], 262144),
    ],
)
def test_input_tokens_per_source_cli_is_parsed(cli, expected):
    parser = add_pretokenized_args(argparse.ArgumentParser())
    args = parser.parse_args(cli)

    assert args.input_tokens_per_source == expected


@pytest.mark.parametrize(
    ("cli", "expected"), [([], 0), (["--per-source-validation-top-n", "10"], 10)]
)
def test_per_source_validation_top_n_cli_is_parsed(cli, expected):
    parser = add_pretokenized_args(argparse.ArgumentParser())
    args = parser.parse_args(cli)

    assert args.per_source_validation_top_n == expected


def test_top_weighted_validation_source_selection():
    config = PretokenizedPerSourceValidationConfig(
        sources=[
            PretokenizedPerSourceValidationSourceConfig(
                name=name, prefix_path=f"/{name}", weight=weight, percentage_data=1.0
            )
            for name, weight in (("low", 0.1), ("high", 0.7), ("mid", 0.2))
        ],
        valid_seed=17,
    )

    assert select_top_weighted_validation_sources(config, 0) is config
    selected = select_top_weighted_validation_sources(config, 2)
    assert [source.name for source in selected.sources] == ["high", "mid"]

    with pytest.raises(ValueError, match="must be non-negative"):
        select_top_weighted_validation_sources(config, -1)
    with pytest.raises(ValueError, match="cannot exceed"):
        select_top_weighted_validation_sources(config, 4)


def test_duplicate_physical_source_paths_are_rejected():
    with pytest.raises(ValidationError, match="prefix paths must be unique"):
        PretokenizedPerSourceValidationConfig(
            sources=[
                PretokenizedPerSourceValidationSourceConfig(
                    name="first", prefix_path="/same/source", weight=0.25, percentage_data=1.0
                ),
                PretokenizedPerSourceValidationSourceConfig(
                    name="second", prefix_path="/same/source", weight=0.75, percentage_data=1.0
                ),
            ],
            valid_seed=17,
        )


def test_token_budget_derives_equal_rows_and_eval_iters():
    assert resolve_per_source_validation_schedule(
        input_tokens_per_source=16,
        sequence_length=4,
        num_sources=3,
        eval_global_batch_size=4,
        requested_eval_iters=0,
    ) == (16, 4, 3)


def test_explicit_eval_iters_requires_zero_token_budget():
    assert resolve_per_source_validation_schedule(
        input_tokens_per_source=0,
        sequence_length=4,
        num_sources=3,
        eval_global_batch_size=4,
        requested_eval_iters=2,
    ) == (8, 2, 2)

    with pytest.raises(ValueError, match="input_tokens_per_source to 0"):
        resolve_per_source_validation_schedule(
            input_tokens_per_source=16,
            sequence_length=4,
            num_sources=2,
            eval_global_batch_size=4,
            requested_eval_iters=1,
        )


def test_token_budget_must_be_a_sequence_length_multiple():
    with pytest.raises(ValueError, match="must be divisible"):
        resolve_per_source_validation_schedule(
            input_tokens_per_source=15,
            sequence_length=4,
            num_sources=2,
            eval_global_batch_size=4,
            requested_eval_iters=0,
        )


@pytest.mark.parametrize("eval_global_batch_size", [4, 8])
def test_mimo_startup_supports_independent_eval_global_batch_size(
    monkeypatch, eval_global_batch_size
):
    per_source = SimpleNamespace(sources=[SimpleNamespace(), SimpleNamespace()])
    config = SimpleNamespace(
        per_source_validation=per_source,
        validation_set=None,
        perplexity_eval_sets=[],
        skip_images=True,
        sources=[],
        img_id=None,
    )
    args = SimpleNamespace(
        dataset_provider="mistral",
        pretokenized_config="/validation.yaml",
        pretokenized_allow_build_of_epoch_index=False,
        overwrite_percentage_data_multiplier=None,
        mimo_llm_tp=1,
        mimo_llm_cp=1,
        mimo_llm_dp=4,
        mimo_encoder_dp=2,
        gtp_weight_remat_size=1,
        eval_interval=10,
        eval_iters=0,
        input_tokens_per_source=16,
        eval_global_batch_size=eval_global_batch_size,
        eval_micro_batch_size=1,
        global_batch_size=4,
        micro_batch_size=1,
        seq_length=4,
        cuda_graph_impl=None,
        save=None,
    )
    monkeypatch.setattr(mistral_provider, "load_pretokenized_config", lambda _path: config)
    monkeypatch.setattr(
        mistral_provider, "apply_pretokenized_config_overrides", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(
        mistral_provider, "_validate_pretokenized_training_args", lambda *_args, **_kwargs: None
    )

    mistral_provider.configure_mistral_data(args)
    assert args.eval_iters == 8 // eval_global_batch_size
    assert args._mistral_per_source_validation_enabled is True


def test_mimo_validation_iterator_uses_eval_global_batch_size(monkeypatch):
    built_with = {}
    expected_iterator = object()

    def build_iterator(**kwargs):
        built_with.update(kwargs)
        return expected_iterator, 1

    monkeypatch.setattr(mistral_provider, "build_per_source_validation_iterator", build_iterator)
    args = SimpleNamespace(
        _mistral_per_source_validation_enabled=True,
        global_batch_size=4,
        eval_global_batch_size=8,
        eval_iters=1,
    )
    config = SimpleNamespace(per_source_validation=object())

    iterator = mistral_provider._build_mistral_validation_iterator(
        args,
        config,
        rank=0,
        world_size=4,
        micro_batch_size=1,
        load_image_pixels=False,
        transform_batch=lambda batch: batch,
        expose_source_metadata=True,
    )

    assert iterator is expected_iterator
    assert built_with["eval_global_batch_size"] == 8
    assert built_with["eval_num_microbatches"] == 2


def test_expand_source_ids_to_token_shape():
    source_ids = torch.tensor([3, -1], dtype=torch.int64)
    loss_mask = torch.ones((2, 4), dtype=torch.float32)

    expanded = expand_source_ids_to_token_shape(source_ids, loss_mask)

    assert expanded.dtype == torch.int64
    assert expanded.shape == loss_mask.shape
    assert expanded.tolist() == [[3, 3, 3, 3], [-1, -1, -1, -1]]
    assert expanded.is_contiguous()


def test_per_source_loss_report_scatter_sums_counts_and_zero_local_source():
    losses = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    loss_mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    source_ids = torch.tensor([[0, 0, 1], [1, -1, -1]], dtype=torch.int64)

    report = build_per_source_loss_report(
        losses=losses, loss_mask=loss_mask, source_ids=source_ids, num_sources=3
    )

    assert report.dtype == torch.float64
    assert report.tolist() == [[1.0, 1.0, 2.0], [7.0, 2.0, 2.0], [0.0, 0.0, 0.0]]


def test_per_source_loss_report_has_fixed_shape_for_250_sources():
    report = build_per_source_loss_report(
        losses=torch.tensor([2.0]),
        loss_mask=torch.tensor([1.0]),
        source_ids=torch.tensor([249], dtype=torch.int64),
        num_sources=250,
    )

    assert report.shape == (250, 3)
    assert report[:249].count_nonzero().item() == 0
    assert report[249].tolist() == [2.0, 1.0, 1.0]


@pytest.mark.parametrize("invalid_source_id", [-2, 2])
def test_per_source_loss_report_rejects_invalid_scored_ids(invalid_source_id):
    with pytest.raises(AssertionError, match="must be -1 for padding"):
        build_per_source_loss_report(
            losses=torch.tensor([1.0]),
            loss_mask=torch.tensor([1.0]),
            source_ids=torch.tensor([invalid_source_id], dtype=torch.int64),
            num_sources=2,
        )


def test_per_source_loss_report_rejects_scored_padding_ids():
    with pytest.raises(AssertionError, match="padding IDs.*zero loss mask"):
        build_per_source_loss_report(
            losses=torch.tensor([1.0]),
            loss_mask=torch.tensor([1.0]),
            source_ids=torch.tensor([-1], dtype=torch.int64),
            num_sources=2,
        )


def test_per_source_loss_report_allows_padded_negative_one_ids():
    report = build_per_source_loss_report(
        losses=torch.tensor([99.0]),
        loss_mask=torch.tensor([0.0]),
        source_ids=torch.tensor([-1], dtype=torch.int64),
        num_sources=2,
    )

    assert report.count_nonzero().item() == 0


def test_per_source_loss_report_rejects_misaligned_token_shapes():
    with pytest.raises(AssertionError, match="identical token shapes"):
        build_per_source_loss_report(
            losses=torch.ones(4),
            loss_mask=torch.ones(4),
            source_ids=torch.zeros(2, dtype=torch.int64),
            num_sources=1,
        )


class _ScheduledLoader:
    def __init__(self):
        self.state = None
        self.datasource_ids = None

    def start(self):
        if self.state is None:
            self.state = {"global_sequence_id": 0}

    def get_state(self):
        return dict(self.state)

    def get_batch_with_datasource_ids(self, local_size_for_this_batch, datasource_ids, skip):
        self.datasource_ids = list(datasource_ids)
        self.state["global_sequence_id"] += len(datasource_ids)
        return SimpleNamespace(
            x=np.arange(6, dtype=np.int64),
            y=np.arange(10, 16, dtype=np.int64),
            y_mask=np.ones(6, dtype=bool),
            sizes=np.array([3, 3], dtype=np.int64),
            images=None,
        )

    def set_state(self, state):
        self.state = dict(state)


def test_scheduled_iterator_initializes_with_direct_source_schedule(monkeypatch):
    monkeypatch.setattr(
        validation.MistralPretokenizedIterator, "__init__", lambda self, **_kwargs: None
    )

    iterator = PerSourcePretokenizedIterator(
        config=SimpleNamespace(sources=[SimpleNamespace(), SimpleNamespace()]),
        rank=0,
        world_size=1,
        micro_batch_size=2,
        num_workers=0,
        global_source_id_schedule=[0, 1],
    )

    assert iterator._global_source_id_schedule == [0, 1]


def test_scheduled_iterator_uses_source_ids_directly_and_masks_padding():
    iterator = PerSourcePretokenizedIterator.__new__(PerSourcePretokenizedIterator)
    iterator._loader = _ScheduledLoader()
    iterator._micro_batch_size = 2
    iterator._seq_len = 3
    iterator._pad_id = 99
    iterator._interdocument_masking = False
    iterator._include_loss_modality = False
    iterator._rank = 0
    iterator._rows_per_global_microbatch = 2
    iterator._global_source_id_schedule = [1, -1]
    iterator._scheduled_row_index = 0

    batch = next(iterator)

    assert iterator._loader.datasource_ids == [1, -1]
    assert batch["source_ids"].tolist() == [1, -1]
    assert batch["tokens"].tolist() == [[0, 1, 2], [99, 99, 99]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]


def test_explicit_datasource_schedule_does_not_advance_padding(monkeypatch):
    class State:
        def __init__(self):
            self.sequence_id = 0

        def current_seq_id_and_increment_(self):
            sequence_id = self.sequence_id
            self.sequence_id += 1
            return sequence_id

    class Datasource:
        src = "source"
        _prefix_path = "/source"

        def get_sequence(self, sequence_id):
            return ("sequence", sequence_id)

    loader = PretokenizedDataLoader.__new__(PretokenizedDataLoader)
    loader._state = SimpleNamespace(global_sequence_id=0, datasources_states=[State(), State()])
    loader._world_size = 1
    loader._rank = 0
    loader._datasources = [Datasource(), Datasource()]
    loader._pad_id = 99
    loader._seq_len = 3
    monkeypatch.setattr(loader_module, "batch_from_sequences", lambda **kwargs: kwargs)

    batch = loader.get_batch_with_datasource_ids(2, [1, -1], skip=False)

    assert loader._state.global_sequence_id == 2
    assert [state.sequence_id for state in loader._state.datasources_states] == [0, 1]
    assert batch["sequences"][0] == ("sequence", 0)
    assert batch["sequences"][1].x == [99, 99, 99]


def test_mimo_batch_and_loss_preserve_per_source_accounting():
    raw_batch = {
        "tokens": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
        "loss_mask": torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        "position_ids": torch.arange(3).expand(2, 3),
        "source_ids": torch.tensor([0, 1]),
    }
    language_batch = build_mistral_language_batch(raw_batch)

    _, _, losses = loss_func(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        loss_mask=language_batch["loss_mask"],
        source_ids=language_batch["source_ids"],
        num_sources=2,
    )

    assert language_batch["source_ids"].tolist() == [[0, 0, 0], [1, 1, 1]]
    assert losses[PER_SOURCE_LOSS_KEY].tolist() == [[4.0, 2.0, 3.0], [9.0, 2.0, 3.0]]


def test_training_reduction_averages_each_source_over_its_scored_tokens():
    total = torch.zeros((2, 3), dtype=torch.float64)
    _accumulate_per_source_loss_reports(
        total,
        [
            {PER_SOURCE_LOSS_KEY: torch.tensor([[2, 1, 2], [3, 1, 2]], dtype=torch.float64)},
            {PER_SOURCE_LOSS_KEY: torch.tensor([[4, 1, 2], [6, 2, 2]], dtype=torch.float64)},
        ],
        num_sources=2,
    )
    metadata = get_per_source_validation_metadata(
        SimpleNamespace(source_names=("code", "docs"), input_token_budgets_by_source=(4, 4))
    )

    result = _finalize_per_source_loss_report(total, metadata)

    assert [value.item() for value in result.values()] == [3.0, 3.0]


def test_image_routing_uses_metadata_spans_not_token_id_alone():
    sequences = [
        SequenceOfSamples(
            x=[18, 1],
            y=[1, 1],
            mask=[True, True],
            sizes=[2],
            doc_ids=[0],
            image_pixels_loaded=False,
        ),
        SequenceOfSamples(
            x=[1, 18, 20, 19],
            y=[1, 1, 1, 1],
            mask=[True, True, True, True],
            sizes=[4],
            image_positions=[Span(start=1, end=4)],
            doc_ids=[1],
            image_pixels_loaded=False,
        ),
    ]
    batch = batch_from_sequences(
        SampleType.PRETRAIN, sequences, ["text", "image"], ["/text", "/image"]
    )
    iterator = MistralPretokenizedIterator.__new__(MistralPretokenizedIterator)
    iterator._img_id = 18
    image_token_mask = iterator._image_token_mask(batch, 1, 6)

    raw_batch = {
        "tokens": torch.from_numpy(batch.x.reshape(1, 6)),
        "labels": torch.from_numpy(batch.y.reshape(1, 6)),
        "loss_mask": torch.ones(1, 6),
        "position_ids": torch.arange(6).reshape(1, 6),
        "image_token_mask": image_token_mask,
    }
    language_batch = build_mistral_language_batch(raw_batch, special_token_ids={"vision": 18})
    encoder_batch = build_mistral_encoder_batch(raw_batch, encoder_name="vision", image_token_id=18)

    assert image_token_mask.tolist() == [[False, False, False, True, False, False]]
    assert language_batch["modality_token_indices"]["vision"].tolist() == [3]
    assert 0 in language_batch["modality_token_indices"]["text"].tolist()
    assert encoder_batch["input_ids"].tolist() == [[-1, 1, 1, 18, 20, 19]]
