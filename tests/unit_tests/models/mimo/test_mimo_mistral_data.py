# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests for the minimal MIMO Mistral-data adapter."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from examples.mimo.data import mistral_provider
from megatron.core.packed_seq_params import PackedSeqParams
from mistral_adapter.pretokenized_config import PretokenizedConfig, SourceConfig
from mistral_fork.mistral.data.pretokenized.components.image_transforms import (
    ImageTransformType,
    SpecialImageIDs,
)
from mistral_fork.mistral.data.pretokenized.datasource import build_sequence
from mistral_fork.mistral.data.pretokenized.image_transform_metadata import TransformImageMetadata
from mistral_fork.mistral.data.pretokenized.megatron_index_reader import _ListTokensBinReader
from mistral_fork.mistral.data.pretokenized.metadata import (
    DocumentIndexEntry,
    ImageIndexEntry,
    TextIndexEntry,
    _TarOffset,
)
from mistral_fork.mistral.data.pretokenized.simple_image_encoder import SimpleImageEncoder


def _raw_language_batch() -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "labels": torch.tensor([[2, 3, 4, 9], [6, 7, 8, 9]]),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]]),
        "position_ids": torch.tensor([[0, 1, 0, 1], [0, 0, 1, 2]]),
        "cu_seqlens": torch.tensor([[0, 2, 4, 4], [0, 1, 4, 4]], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([[0, 2, 4, 4], [0, 1, 4, 4]], dtype=torch.int32),
        "max_seqlen": torch.tensor([2, 3], dtype=torch.int32),
    }


def test_language_batch_preserves_fields_and_flattens_mbs_greater_than_one():
    raw_batch = _raw_language_batch()
    expected = {
        name: raw_batch[name].reshape(1, -1).clone()
        for name in ("tokens", "labels", "loss_mask", "position_ids")
    }

    batch = mistral_provider.build_mistral_language_batch(raw_batch)

    assert torch.equal(batch["input_ids"], expected["tokens"])
    for name in ("labels", "loss_mask", "position_ids"):
        assert torch.equal(batch[name], expected[name])
    assert set(batch) == {
        "input_ids",
        "labels",
        "loss_mask",
        "position_ids",
        "cu_seqlens",
        "cu_seqlens_padded",
        "max_seqlen",
        "modality_token_indices",
    }
    assert set(batch["modality_token_indices"]) == {"text"}
    assert torch.equal(batch["modality_token_indices"]["text"], torch.arange(8))
    assert torch.equal(batch["cu_seqlens"], torch.tensor([[0, 2, 4, 5, 8]], dtype=torch.int32))
    assert torch.equal(
        batch["cu_seqlens_padded"], torch.tensor([[0, 2, 4, 5, 8]], dtype=torch.int32)
    )
    assert torch.equal(batch["max_seqlen"], torch.tensor([3], dtype=torch.int32))


def test_language_batch_precomputes_text_and_modality_positions():
    batch = mistral_provider.build_mistral_language_batch(
        _raw_language_batch(), special_token_ids={"vision": 3, "audio": 6}
    )

    indices = batch["modality_token_indices"]
    assert torch.equal(indices["vision"], torch.tensor([2]))
    assert torch.equal(indices["audio"], torch.tensor([5]))
    assert torch.equal(indices["text"], torch.tensor([0, 1, 3, 4, 6, 7]))


@pytest.mark.parametrize("packed", (False, True))
def test_language_batch_preserves_training_source_ids(packed):
    raw = _raw_language_batch()
    if not packed:
        for name in ("cu_seqlens", "cu_seqlens_padded", "max_seqlen"):
            raw.pop(name)
    raw["data_source_ids"] = torch.tensor([[0, 0, 1, 1], [2, 0, 0, 0]])
    batch = mistral_provider.build_mistral_language_batch(raw)
    torch.testing.assert_close(
        batch["data_source_ids"], raw["data_source_ids"].reshape_as(batch["labels"])
    )


def test_language_batch_preserves_real_image_rows_and_logical_padding_lengths():
    raw_batch = _raw_language_batch()
    raw_batch["cu_seqlens"] = torch.tensor([[0, 2, 3, 3], [0, 1, 2, 2]], dtype=torch.int32)
    raw_batch["loss_mask"][0, 2] = 0  # Image input remains a real attention query.

    batch = mistral_provider.build_mistral_language_batch(
        raw_batch, special_token_ids={"vision": 3}
    )

    assert "padding_mask" not in batch
    torch.testing.assert_close(
        batch["cu_seqlens"], torch.tensor([[0, 2, 3, 4, 5]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        batch["cu_seqlens_padded"], torch.tensor([[0, 2, 4, 5, 8]], dtype=torch.int32)
    )
    assert batch["modality_token_indices"]["vision"].tolist() == [2]
    assert batch["loss_mask"][0, 2] == 0


def test_language_batch_without_interdocument_masking_preserves_standard_batch():
    raw_batch = {
        name: value
        for name, value in _raw_language_batch().items()
        if name not in ("cu_seqlens", "cu_seqlens_padded", "max_seqlen")
    }

    batch = mistral_provider.build_mistral_language_batch(raw_batch)

    assert batch["input_ids"] is raw_batch["tokens"]
    for name in ("labels", "loss_mask", "position_ids"):
        assert batch[name] is raw_batch[name]
    assert set(batch) == {
        "input_ids",
        "labels",
        "loss_mask",
        "position_ids",
        "modality_token_indices",
    }
    assert torch.equal(batch["modality_token_indices"]["text"], torch.arange(8))


def test_encoder_batch_keeps_rows_and_builds_exact_nested_kwargs():
    pin_memory = torch.cuda.is_available()
    input_ids = torch.arange(24).reshape(3, 8)
    first_image = torch.arange(3 * 28 * 28, dtype=torch.bfloat16).reshape(3, 28, 28)
    second_image = torch.arange(3 * 28 * 56, dtype=torch.bfloat16).reshape(3, 28, 56)

    batch = mistral_provider.build_mistral_encoder_batch(
        {"tokens": input_ids, "images": [first_image, second_image]},
        encoder_name="mistral_vit_encoder",
        pin_memory=pin_memory,
    )

    assert batch["input_ids"] is input_ids
    assert set(batch["modality_inputs"]) == {"mistral_vit_encoder"}
    assert set(batch["modality_inputs"]["mistral_vit_encoder"]) == {"mistral_vit_encoder"}
    encoder_kwargs = batch["modality_inputs"]["mistral_vit_encoder"]["mistral_vit_encoder"]
    assert set(encoder_kwargs) == {"pixel_values", "imgs_sizes", "packed_seq_params"}
    assert encoder_kwargs["pixel_values"].shape == (1, 12, 3 * 14 * 14)
    assert encoder_kwargs["pixel_values"].dtype == torch.bfloat16
    assert encoder_kwargs["pixel_values"].is_pinned() is pin_memory
    assert torch.equal(encoder_kwargs["pixel_values"][0, 0], first_image[:, :14, :14].reshape(-1))
    assert encoder_kwargs["imgs_sizes"] == [(28, 28), (28, 56)]
    packed_seq_params = encoder_kwargs["packed_seq_params"]
    assert isinstance(packed_seq_params, PackedSeqParams)
    assert packed_seq_params.cu_seqlens_q.is_pinned() is pin_memory
    assert torch.equal(packed_seq_params.cu_seqlens_q, torch.tensor([0, 4, 12], dtype=torch.int32))
    assert packed_seq_params.max_seqlen_q == packed_seq_params.max_seqlen_kv == 8

    text_only_batch = mistral_provider.build_mistral_encoder_batch(
        {"tokens": input_ids}, encoder_name="mistral_vit_encoder"
    )
    assert set(text_only_batch) == {"input_ids"}
    assert text_only_batch["input_ids"] is input_ids


class _FakeMistralIterator:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.global_sequence_id = 0
        self.instances.append(self)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_with_state()[0]

    def next_with_state(self):
        micro_batch_size = self.kwargs["micro_batch_size"]
        rank = self.kwargs["rank"]
        world_size = self.kwargs["world_size"]
        sequence_length = self.kwargs["config"].original_seq_len
        first_sequence = self.global_sequence_id + rank * micro_batch_size
        rows = torch.arange(first_sequence, first_sequence + micro_batch_size).unsqueeze(1)
        tokens = rows.expand(micro_batch_size, sequence_length).clone()
        cu_seqlens = torch.tensor([[0, sequence_length]] * micro_batch_size, dtype=torch.int32)
        batch = {
            "tokens": tokens,
            "labels": tokens.clone(),
            "loss_mask": torch.ones_like(tokens, dtype=torch.float32),
            "position_ids": torch.arange(sequence_length).expand_as(tokens),
            "cu_seqlens": cu_seqlens,
            "max_seqlen": torch.full((micro_batch_size,), sequence_length, dtype=torch.int32),
        }
        self.global_sequence_id += micro_batch_size * world_size
        return batch, {"global_sequence_id": self.global_sequence_id}

    def set_state(self, state) -> None:
        self.global_sequence_id = state["global_sequence_id"]

    def save_state(self):
        return {"global_sequence_id": self.global_sequence_id}

    def close(self) -> None:
        pass


def _config(*, interdocument_masking: bool = True) -> PretokenizedConfig:
    return PretokenizedConfig(
        sources=[
            SourceConfig(
                prefix_path="/unused",
                weight=1.0,
                percentage_data=1.0,
                image_transform_type="no_transform",
            )
        ],
        bos_id=1,
        eos_id=2,
        pad_id=0,
        img_id=10,
        img_break_id=11,
        img_end_id=12,
        seed=1,
        epoch_index_cache_path="/unused",
        interdocument_masking=interdocument_masking,
        drop_bos=not interdocument_masking,
        original_seq_len=4,
        truncate_trailing_masked=True,
    )


def _configuration_args(**updates) -> SimpleNamespace:
    values = {
        "dataset_provider": "mistral",
        "pretokenized_config": "config.yaml",
        "pretokenized_allow_build_of_epoch_index": False,
        "dataloader_inter_document_masking": True,
        "tensor_model_parallel_size": 1,
        "mimo_llm_tp": 1,
        "seq_length": 4,
        "micro_batch_size": 1,
        "mimo_llm_only": False,
        "mimo_encoder_dp": 2,
        "mimo_llm_dp": 4,
        "gtp_weight_remat_size": 1,
        "image_token_id": 31,
        "eval_iters": 0,
        "eval_interval": 0,
        "overwrite_percentage_data_multiplier": None,
        "save": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _args(**updates) -> SimpleNamespace:
    values = {
        "mimo_llm_dp": 4,
        "mimo_llm_tp": 1,
        "mimo_llm_pp": 1,
        "gtp_weight_remat_size": 1,
        "mimo_encoder_dp": 2,
        "micro_batch_size": 2,
        "mistral_encoder_num_workers": None,
        "mistral_image_group_workers": 1,
        "num_workers": 0,
        "pretokenized_allow_partial_epoch_index_cache": False,
        "train_samples": None,
        "train_iters": 1,
        "global_batch_size": 8,
        "seq_length": 4,
        "iteration": 0,
        "consumed_train_samples": 0,
        "phase_transition_samples": None,
        "load": None,
        "_mistral_pretokenized_config": _config(),
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _topology(
    *,
    language_rank: bool,
    encoder_rank: bool,
    dp_rank: int,
    gtp_rank: int = 0,
    gtp_size: int = 1,
    sequence_shards: int = 1,
    cp_rank: int = 0,
    cp_size: int = 1,
):
    def group(rank=0, size=1):
        return SimpleNamespace(rank=lambda: rank, size=lambda: size)

    data_parallel_group = group(rank=dp_rank)
    language_data_parallel_group = group(
        rank=dp_rank * (gtp_size // sequence_shards) + gtp_rank // sequence_shards
    )
    model_parallel_group = group()
    return SimpleNamespace(
        grids={
            "language": SimpleNamespace(is_current_rank_in_grid=lambda: language_rank),
            "mistral_vit_encoder": SimpleNamespace(is_current_rank_in_grid=lambda: encoder_rank),
        },
        module_pgs={
            "language": SimpleNamespace(
                tp=model_parallel_group,
                cp=group(rank=cp_rank, size=cp_size),
                pp=model_parallel_group,
                dp=data_parallel_group,
                gtp_remat=group(rank=gtp_rank, size=gtp_size),
                dp_gtp_remat=language_data_parallel_group,
                dp_cp_gtp_remat=language_data_parallel_group,
            ),
            "mistral_vit_encoder": SimpleNamespace(
                tp=model_parallel_group,
                cp=model_parallel_group,
                pp=model_parallel_group,
                dp=data_parallel_group,
            ),
        },
    )


def test_one_encoder_iterator_matches_contiguous_language_lane_partition(monkeypatch):
    _FakeMistralIterator.instances = []
    monkeypatch.setattr(mistral_provider, "MistralPretokenizedIterator", _FakeMistralIterator)
    monkeypatch.setattr(mistral_provider, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(mistral_provider, "is_pp_first_stage", lambda _group: True)
    args = _args()

    language_rows = []
    for language_dp_rank in (2, 3):
        language_loader = mistral_provider.build_mistral_data_loaders(
            args, _topology(language_rank=True, encoder_rank=False, dp_rank=language_dp_rank)
        )[0]
        language_rows.append(next(language_loader)["input_ids"].reshape(2, 4))

    encoder_loader = mistral_provider.build_mistral_data_loaders(
        args, _topology(language_rank=False, encoder_rank=True, dp_rank=1)
    )[0]
    encoder_rows = next(encoder_loader)["input_ids"]

    assert torch.equal(encoder_rows, torch.cat(language_rows))
    assert all(
        instance.kwargs["load_image_pixels"] is False
        for instance in _FakeMistralIterator.instances[:2]
    )
    encoder_iterator_args = _FakeMistralIterator.instances[-1].kwargs
    assert encoder_iterator_args["rank"] == 1
    assert encoder_iterator_args["world_size"] == 2
    assert encoder_iterator_args["micro_batch_size"] == 4
    assert encoder_iterator_args["load_image_pixels"] is True


@pytest.mark.parametrize("sequence_shards", [1, 2])
def test_gtp_language_lanes_match_contiguous_encoder_partition(monkeypatch, sequence_shards):
    _FakeMistralIterator.instances = []
    monkeypatch.setattr(mistral_provider, "MistralPretokenizedIterator", _FakeMistralIterator)
    monkeypatch.setattr(mistral_provider, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(mistral_provider, "is_pp_first_stage", lambda _group: True)
    args = _args(
        mimo_llm_dp=2,
        gtp_weight_remat_size=4,
        tensor_parallel_num_sequence_shards=sequence_shards,
        mimo_encoder_dp=4 // sequence_shards,
        micro_batch_size=1,
    )

    language_rows = []
    for language_gtp_rank in range(4 - 2 * sequence_shards, 4, sequence_shards):
        language_loader = mistral_provider.build_mistral_data_loaders(
            args,
            _topology(
                language_rank=True,
                encoder_rank=False,
                dp_rank=1,
                gtp_rank=language_gtp_rank,
                gtp_size=4,
                sequence_shards=sequence_shards,
            ),
        )[0]
        language_rows.append(next(language_loader)["input_ids"].reshape(1, 4))

    encoder_loader = mistral_provider.build_mistral_data_loaders(
        args, _topology(language_rank=False, encoder_rank=True, dp_rank=args.mimo_encoder_dp - 1)
    )[0]
    encoder_rows = next(encoder_loader)["input_ids"]

    assert torch.equal(encoder_rows, torch.cat(language_rows))
    language_iterator_args = [instance.kwargs for instance in _FakeMistralIterator.instances[:2]]
    sample_lanes = 8 // sequence_shards
    assert [iterator_args["rank"] for iterator_args in language_iterator_args] == list(
        range(sample_lanes - 2, sample_lanes)
    )
    assert all(
        iterator_args["world_size"] == sample_lanes for iterator_args in language_iterator_args
    )
    encoder_iterator_args = _FakeMistralIterator.instances[-1].kwargs
    assert encoder_iterator_args["rank"] == args.mimo_encoder_dp - 1
    assert encoder_iterator_args["world_size"] == args.mimo_encoder_dp
    assert encoder_iterator_args["micro_batch_size"] == 2
    assert encoder_iterator_args["load_image_pixels"] is True


@pytest.mark.parametrize("sequence_shards", [1, 2])
def test_language_cp_ranks_share_the_same_data_lane(monkeypatch, sequence_shards):
    _FakeMistralIterator.instances = []
    monkeypatch.setattr(mistral_provider, "MistralPretokenizedIterator", _FakeMistralIterator)
    monkeypatch.setattr(mistral_provider, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(mistral_provider, "is_pp_first_stage", lambda _group: True)
    args = _args(
        mimo_llm_dp=2,
        gtp_weight_remat_size=2,
        tensor_parallel_num_sequence_shards=sequence_shards,
        micro_batch_size=1,
    )

    language_batches = []
    for cp_rank in (0, 1):
        loader = mistral_provider.build_mistral_data_loaders(
            args,
            _topology(
                language_rank=True,
                encoder_rank=False,
                dp_rank=1,
                gtp_rank=2 - sequence_shards + cp_rank % sequence_shards,
                gtp_size=2,
                sequence_shards=sequence_shards,
                cp_rank=cp_rank,
                cp_size=2 * sequence_shards,
            ),
        )[0]
        language_batches.append(next(loader)["input_ids"])

    sample_lanes = 4 // sequence_shards
    assert [instance.kwargs["rank"] for instance in _FakeMistralIterator.instances] == [
        sample_lanes - 1,
        sample_lanes - 1,
    ]
    assert all(
        instance.kwargs["world_size"] == sample_lanes for instance in _FakeMistralIterator.instances
    )
    assert torch.equal(*language_batches)


def test_middle_language_stage_gets_packed_metadata_iterator(monkeypatch):
    _FakeMistralIterator.instances = []
    monkeypatch.setattr(mistral_provider, "MistralPretokenizedIterator", _FakeMistralIterator)
    monkeypatch.setattr(mistral_provider, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(mistral_provider, "is_pp_first_stage", lambda _group: False)

    language_loader = mistral_provider.build_mistral_data_loaders(
        _args(), _topology(language_rank=True, encoder_rank=False, dp_rank=2)
    )[0]

    assert language_loader is not None
    iterator_args = _FakeMistralIterator.instances[0].kwargs
    assert iterator_args["rank"] == 2
    assert iterator_args["load_image_pixels"] is False


@pytest.mark.parametrize("ssa", [False, True])
def test_source_contract_flags_are_forwarded(monkeypatch, ssa):
    _FakeMistralIterator.instances = []
    monkeypatch.setattr(mistral_provider, "MistralPretokenizedIterator", _FakeMistralIterator)

    mistral_provider._MistralDataIterator(
        _args(
            pretokenized_allow_partial_epoch_index_cache=True,
            experimental_attention_variant="ssa" if ssa else None,
        ),
        _config(),
        rank=0,
        world_size=1,
        micro_batch_size=1,
        num_workers=0,
        load_image_pixels=False,
        translate_batch=lambda batch: batch,
        checkpoint_state_enabled=True,
        restored_state=None,
    )

    iterator_args = _FakeMistralIterator.instances[0].kwargs
    assert iterator_args["allow_partial_epoch_index_cache"] is True
    assert iterator_args["preserve_logical_seqlens"] is ssa


@pytest.mark.parametrize(
    ("restored_state", "crossed_transition", "expected"),
    [
        (None, True, False),
        ({"global_sequence_id": 8}, False, False),
        ({"global_sequence_id": 8}, True, True),
    ],
)
def test_phase_transition_flag_requires_crossing_checkpoint_restore(
    monkeypatch, restored_state, crossed_transition, expected
):
    _FakeMistralIterator.instances = []
    monkeypatch.setattr(mistral_provider, "MistralPretokenizedIterator", _FakeMistralIterator)
    monkeypatch.setattr(mistral_provider, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(
        mistral_provider, "_load_mistral_state", lambda _args, *, pipeline_parallel: restored_state
    )
    monkeypatch.setattr(mistral_provider, "_in_phase_transition", lambda _args: crossed_transition)

    mistral_provider.build_mistral_data_loaders(
        _args(), _topology(language_rank=True, encoder_rank=False, dp_rank=0)
    )

    assert _FakeMistralIterator.instances[0].kwargs["in_phase_transition"] is expected


def test_checkpoint_state_stops_at_last_delivered_batch(monkeypatch):
    _FakeMistralIterator.instances = []
    monkeypatch.setattr(mistral_provider, "MistralPretokenizedIterator", _FakeMistralIterator)
    args = _args()
    args.num_workers = 1

    iterator = mistral_provider._MistralDataIterator(
        args,
        args._mistral_pretokenized_config,
        rank=0,
        world_size=2,
        micro_batch_size=2,
        num_workers=args.num_workers,
        load_image_pixels=False,
        translate_batch=lambda batch: batch,
        checkpoint_state_enabled=True,
        restored_state=None,
    )
    try:
        first = next(iterator)
        iterator._next_batch.result()

        assert first["tokens"][0, 0].item() == 0
        assert iterator.save_state() == {"global_sequence_id": 4}
        assert _FakeMistralIterator.instances[0].global_sequence_id == 8
    finally:
        iterator.close()


def test_checkpoint_state_does_not_commit_batch_when_next_prefetch_fails(monkeypatch):
    _FakeMistralIterator.instances = []
    monkeypatch.setattr(mistral_provider, "MistralPretokenizedIterator", _FakeMistralIterator)
    args = _args()
    args.num_workers = 1

    iterator = mistral_provider._MistralDataIterator(
        args,
        args._mistral_pretokenized_config,
        rank=0,
        world_size=2,
        micro_batch_size=2,
        num_workers=args.num_workers,
        load_image_pixels=False,
        translate_batch=lambda batch: batch,
        checkpoint_state_enabled=True,
        restored_state=None,
    )
    try:
        next(iterator)
        iterator._next_batch.result()

        def fail_next_batch():
            raise RuntimeError("prefetch failed")

        monkeypatch.setattr(iterator._raw_iterator, "next_with_state", fail_next_batch)
        with pytest.raises(RuntimeError, match="prefetch failed"):
            next(iterator)

        assert iterator.save_state() == {"global_sequence_id": 4}
    finally:
        iterator.close()


def test_encoder_restores_canonical_language_state(monkeypatch, tmp_path):
    _FakeMistralIterator.instances = []
    monkeypatch.setattr(mistral_provider, "MistralPretokenizedIterator", _FakeMistralIterator)
    monkeypatch.setattr(mistral_provider, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(mistral_provider, "is_pp_first_stage", lambda _group: True)
    args = _args()
    args.iteration = 3
    args.load = str(tmp_path)
    state_path = tmp_path / "iter_0000003" / "mp_rank_00" / "train_dataloader_dprank000.pt"
    state_path.parent.mkdir(parents=True)
    torch.save({"dataloader_state_dict": {"global_sequence_id": 24}}, state_path)

    encoder_loader = mistral_provider.build_mistral_data_loaders(
        args, _topology(language_rank=False, encoder_rank=True, dp_rank=0)
    )[0]
    batch = next(encoder_loader)

    assert batch["input_ids"][0, 0].item() == 24
    assert encoder_loader.save_state() is None


def test_canonical_state_reshards_across_new_language_and_encoder_dp(monkeypatch):
    _FakeMistralIterator.instances = []
    monkeypatch.setattr(mistral_provider, "MistralPretokenizedIterator", _FakeMistralIterator)
    monkeypatch.setattr(mistral_provider, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(mistral_provider, "is_pp_first_stage", lambda _group: True)

    old_args = _args()
    old_loader = mistral_provider.build_mistral_data_loaders(
        old_args, _topology(language_rank=True, encoder_rank=False, dp_rank=0)
    )[0]
    next(old_loader)
    saved_state = old_loader.save_state()
    assert saved_state == {"global_sequence_id": 8}

    new_args = _args()
    new_args.llm_dp = 2
    new_args.encoder_dp = 1
    new_args._mistral_pretokenized_config = _config()
    language_rows = []
    for dp_rank in range(new_args.llm_dp):
        loader = mistral_provider._MistralDataIterator(
            new_args,
            new_args._mistral_pretokenized_config,
            rank=dp_rank,
            world_size=new_args.llm_dp,
            micro_batch_size=new_args.micro_batch_size,
            num_workers=new_args.num_workers,
            load_image_pixels=False,
            translate_batch=lambda batch: batch,
            checkpoint_state_enabled=True,
            restored_state=saved_state,
        )
        language_rows.append(next(loader)["tokens"][:, 0])

    encoder_loader = mistral_provider._MistralDataIterator(
        new_args,
        new_args._mistral_pretokenized_config,
        rank=0,
        world_size=new_args.encoder_dp,
        micro_batch_size=new_args.micro_batch_size * new_args.llm_dp,
        num_workers=new_args.num_workers,
        load_image_pixels=True,
        translate_batch=lambda batch: batch,
        checkpoint_state_enabled=False,
        restored_state=saved_state,
    )
    encoder_rows = next(encoder_loader)["tokens"][:, 0]

    assert torch.equal(torch.cat(language_rows), torch.arange(8, 12))
    assert torch.equal(encoder_rows, torch.arange(8, 12))


def test_metadata_only_sequence_matches_pixel_tokens_and_mask_without_pixel_read(monkeypatch):
    special_ids = SpecialImageIDs(img=10, img_break=11, img_end=12)
    image_encoder = SimpleImageEncoder(ImageTransformType.NO_TRANSFORM, special_ids)
    metadata = TransformImageMetadata(width=56, height=56)
    image_tokens = image_encoder.encode_image_metadata(metadata, seed=(7,))
    assert special_ids.img in image_tokens
    assert special_ids.img_break in image_tokens
    assert special_ids.img_end in image_tokens
    image_entry = ImageIndexEntry(
        img_id_in_doc=0,
        tar_offset=_TarOffset(offset=0, size=0, tar_path="/unused"),
        pos_in_text=1,
        metadata=metadata,
        size=len(image_tokens),
        seed=(7,),
    )
    document = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=3),
        imgs=[image_entry],
        size=3 + len(image_tokens),
    )
    token_reader = _ListTokensBinReader([1, 20, 2])
    source_image = Image.fromarray(np.zeros((56, 56, 3), dtype=np.uint8))

    monkeypatch.setattr(
        "mistral_fork.mistral.data.pretokenized.datasource.load_contents_grouped",
        lambda _images, **_kwargs: [source_image],
    )
    pixel_sequence = build_sequence(
        [document], image_encoder, token_reader, bos_id=1, eos_id=2, is_masked_pretrain=False
    )
    monkeypatch.setattr(
        "mistral_fork.mistral.data.pretokenized.datasource.load_contents_grouped",
        lambda _images, **_kwargs: pytest.fail("metadata-only path loaded image pixels"),
    )
    metadata_sequence = build_sequence(
        [document],
        image_encoder,
        token_reader,
        bos_id=1,
        eos_id=2,
        is_masked_pretrain=False,
        load_image_pixels=False,
    )

    assert metadata_sequence.x == pixel_sequence.x
    assert metadata_sequence.y == pixel_sequence.y
    assert metadata_sequence.mask == pixel_sequence.mask
    assert metadata_sequence.image_positions == pixel_sequence.image_positions
    image_span = metadata_sequence.image_positions[0]
    native_image_tokens = metadata_sequence.x[image_span.start : image_span.end]
    assert native_image_tokens == image_tokens
    assert special_ids.img_break in native_image_tokens
    assert native_image_tokens[-1] == special_ids.img_end
    assert metadata_sequence.images == []
    assert metadata_sequence.image_pixels_loaded is False
    assert len(pixel_sequence.images) == 1


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"pretokenized_config": None}, "requires --pretokenized-config"),
        ({"pretokenized_allow_build_of_epoch_index": True}, "prebuilt Mistral epoch indexes"),
        ({"eval_iters": 1, "eval_interval": 10}, "does not yet support validation iterators"),
        ({"mimo_llm_only": True}, "requires an active encoder grid"),
        ({"mimo_encoder_dp": 5}, r"--micro-batch-size \* language sample lanes"),
    ],
)
def test_startup_validation_rejects_unsupported_configuration(monkeypatch, update, message):
    args = _configuration_args(**update)
    monkeypatch.setattr(mistral_provider, "load_pretokenized_config", lambda _path: _config())

    with pytest.raises(ValueError, match=message):
        mistral_provider.configure_mistral_data(args)


def test_startup_configuration_applies_generic_override_and_native_image_token(monkeypatch):
    args = _configuration_args(overwrite_percentage_data_multiplier=0.5)
    monkeypatch.setattr(mistral_provider, "load_pretokenized_config", lambda _path: _config())

    mistral_provider.configure_mistral_data(args)

    configured = args._mistral_pretokenized_config
    assert configured.sources[0].percentage_data == 0.5
    assert args.image_token_id == configured.img_id == 10


def test_startup_configuration_accepts_gtp_fanout_divisibility(monkeypatch):
    args = _configuration_args(mimo_llm_dp=2, gtp_weight_remat_size=2, mimo_encoder_dp=4)
    monkeypatch.setattr(mistral_provider, "load_pretokenized_config", lambda _path: _config())

    mistral_provider.configure_mistral_data(args)

    assert args._mistral_pretokenized_config.img_id == 10


def test_startup_configuration_supports_no_interdocument_masking(monkeypatch):
    args = _configuration_args(dataloader_inter_document_masking=False)
    monkeypatch.setattr(
        mistral_provider,
        "load_pretokenized_config",
        lambda _path: _config(interdocument_masking=False),
    )

    mistral_provider.configure_mistral_data(args)

    assert args._mistral_pretokenized_config.interdocument_masking is False


def test_startup_validation_uses_language_tensor_parallel_size(monkeypatch):
    args = _configuration_args(
        mimo_llm_tp=2,
        micro_batch_size=2,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=0.1,
    )
    monkeypatch.setattr(mistral_provider, "load_pretokenized_config", lambda _path: _config())

    with pytest.raises(AssertionError, match="TP=2, micro_batch_size=2"):
        mistral_provider.configure_mistral_data(args)


@pytest.mark.parametrize("image_count", [0, 3, 11])
def test_eight_image_shards_preserve_tokens_order_and_read_only_owned_images(
    monkeypatch, image_count
):
    special_ids = SpecialImageIDs(img=10, img_break=11, img_end=12)
    # Use the production transform, which emits CHW pixels expected by the
    # adapter. NO_TRANSFORM is metadata-test-only and leaves pixels in HWC.
    encoder = SimpleImageEncoder(ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES, special_ids)
    entries, source_images = [], []
    for index in range(image_count):
        width = 28 * (1 + index % 3)
        metadata = TransformImageMetadata(width=width, height=28)
        image_tokens = encoder.encode_image_metadata(metadata, seed=(index,))
        entries.append(
            ImageIndexEntry(
                img_id_in_doc=index,
                tar_offset=_TarOffset(offset=0, size=0, tar_path="/unused"),
                pos_in_text=1,
                metadata=metadata,
                size=len(image_tokens),
                seed=(index,),
            )
        )
        source_images.append(Image.fromarray(np.full((28, width, 3), index, dtype=np.uint8)))
    document = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=3),
        imgs=entries,
        size=3 + sum(entry.size for entry in entries),
    )
    reads = []

    def load_owned(images, **_kwargs):
        indices = [entry.img_id_in_doc for entry in images]
        reads.extend(indices)
        return [source_images[index] for index in indices]

    monkeypatch.setattr(
        "mistral_fork.mistral.data.pretokenized.datasource.load_contents_grouped", load_owned
    )
    common = dict(
        docs=[document],
        image_encoder=encoder,
        bin_reader=_ListTokensBinReader([1, 20, 2]),
        bos_id=1,
        eos_id=2,
        is_masked_pretrain=False,
    )
    reference = build_sequence(**common)
    reads.clear()
    shards = [build_sequence(**common, image_shard=(rank, 8)) for rank in range(8)]
    assert reads == list(range(image_count))
    assert [span for shard in shards for span in shard.image_positions] == reference.image_positions
    reconstructed_images = [image for shard in shards for image in shard.images]
    for actual, expected in zip(reconstructed_images, reference.images, strict=True):
        torch.testing.assert_close(actual, expected)
    local_token_counts = []
    converter = object.__new__(mistral_provider.MistralPretokenizedIterator)
    converter._micro_batch_size, converter._seq_len, converter._img_id = 1, len(reference.x), 10
    converter._include_loss_modality = False
    for shard in shards:
        assert (shard.x, shard.y, shard.mask) == (reference.x, reference.y, reference.mask)
        native_batch = SimpleNamespace(
            x=np.array(shard.x),
            y=np.array(shard.y),
            y_mask=np.array(shard.mask),
            images=shard.images,
            image_positions=shard.image_positions,
        )
        batch = mistral_provider.build_mistral_encoder_batch(
            converter._convert_batch(native_batch), encoder_name="vision", image_token_id=10
        )
        count = int((batch["input_ids"] == 10).sum())
        local_token_counts.append(count)
        assert bool(batch.get("modality_inputs")) == bool(shard.images)
        if not shard.images:
            assert count == 0
    assert sum(local_token_counts) == reference.x.count(10)


def test_image_sharded_encoder_iterators_share_and_restore_canonical_sequence(monkeypatch):
    _FakeMistralIterator.instances = []
    monkeypatch.setattr(mistral_provider, "MistralPretokenizedIterator", _FakeMistralIterator)
    monkeypatch.setattr(mistral_provider, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(mistral_provider, "is_pp_first_stage", lambda _group: True)
    state = {"global_sequence_id": 37}
    monkeypatch.setattr(mistral_provider, "_load_mistral_state", lambda *_args, **_kwargs: state)
    args = _args(
        mimo_llm_dp=1, mimo_encoder_dp=8, micro_batch_size=1, mimo_encoder_image_sharding=True
    )
    for rank in range(8):
        loader = mistral_provider.build_mistral_data_loaders(
            args, _topology(language_rank=False, encoder_rank=True, dp_rank=rank)
        )[0]
        raw = _FakeMistralIterator.instances[-1]
        assert (raw.kwargs["rank"], raw.kwargs["world_size"], raw.kwargs["micro_batch_size"]) == (
            0,
            1,
            1,
        )
        assert raw.kwargs["image_shard"] == (rank, 8)
        assert next(loader)["input_ids"][0, 0].item() == 37
        assert next(loader)["input_ids"][0, 0].item() == 38
        assert loader.save_state() is None
        loader.close()


def test_image_sharded_encoder_iterators_map_to_two_language_lanes(monkeypatch):
    _FakeMistralIterator.instances = []
    monkeypatch.setattr(mistral_provider, "MistralPretokenizedIterator", _FakeMistralIterator)
    monkeypatch.setattr(mistral_provider, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(mistral_provider, "is_pp_first_stage", lambda _group: True)
    state = {"global_sequence_id": 37}
    monkeypatch.setattr(mistral_provider, "_load_mistral_state", lambda *_args, **_kwargs: state)
    args = _args(
        mimo_llm_dp=2, mimo_encoder_dp=8, micro_batch_size=1, mimo_encoder_image_sharding=True
    )

    for rank in range(8):
        loader = mistral_provider.build_mistral_data_loaders(
            args, _topology(language_rank=False, encoder_rank=True, dp_rank=rank)
        )[0]
        raw = _FakeMistralIterator.instances[-1]
        lane, shard = divmod(rank, 4)
        assert (raw.kwargs["rank"], raw.kwargs["world_size"], raw.kwargs["image_shard"]) == (
            lane,
            2,
            (shard, 4),
        )
        assert next(loader)["input_ids"][0, 0].item() == 37 + lane
        loader.close()
