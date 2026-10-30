from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mistral_adapter.pretokenized_config import PretokenizedConfig, SourceConfig
from mistral_adapter.pretokenized_iterator import MistralPretokenizedIterator
from mistral_fork.mistral.data.iterator.utils import SequenceOfSamples, pad_sequence
from mistral_fork.mistral.data.pretokenized.pretokenized_data_loader import batch_from_sequences
from mistral_fork.mistral.data.source import SampleType
from mistral_fork.mistral_common.tokens.tokenizers.base import Span


@pytest.mark.parametrize(("interdocument_masking", "drop_bos"), [(True, True), (False, False)])
def test_iterator_rejects_matching_document_boundary_options(
    interdocument_masking: bool, drop_bos: bool
) -> None:
    config = PretokenizedConfig(
        sources=[SourceConfig(prefix_path="/unused", weight=1.0, percentage_data=1.0)],
        bos_id=1,
        eos_id=2,
        pad_id=0,
        img_id=18,
        img_break_id=20,
        img_end_id=19,
        seed=0,
        epoch_index_cache_path="/unused",
        interdocument_masking=interdocument_masking,
        drop_bos=drop_bos,
        original_seq_len=3,
        truncate_trailing_masked=True,
        skip_images=True,
    )

    with pytest.raises(AssertionError, match="Exactly one"):
        MistralPretokenizedIterator(
            config=config, num_total_tokens=3, rank=0, world_size=1, micro_batch_size=1
        )


def test_no_interdocument_masking_batch_omits_thd_metadata() -> None:
    iterator = MistralPretokenizedIterator.__new__(MistralPretokenizedIterator)
    iterator._micro_batch_size = 2
    iterator._seq_len = 3
    iterator._interdocument_masking = False
    iterator._include_loss_modality = False
    iterator._preserve_logical_seqlens = False
    batch = SimpleNamespace(
        x=np.arange(6, dtype=np.int64), y=np.arange(1, 7, dtype=np.int64), y_mask=None, images=None
    )
    iterator._loader = SimpleNamespace(get_batch=lambda: batch, get_state=lambda: {})

    result = next(iterator)

    assert set(result) == {"tokens", "labels", "loss_mask", "position_ids"}
    assert result["tokens"].shape == (2, 3)


@pytest.mark.parametrize("packed", (False, True))
def test_training_source_ids_follow_document_order_and_preserve_batch_and_state(packed):
    iterator = MistralPretokenizedIterator.__new__(MistralPretokenizedIterator)
    iterator._micro_batch_size, iterator._seq_len = 2, 4
    iterator._interdocument_masking = packed
    iterator._include_loss_modality = False
    iterator._preserve_logical_seqlens = False
    iterator._data_source_indices = {"/data/b": 0, "/data/a": 1}
    batch = SimpleNamespace(
        x=np.arange(8),
        y=np.arange(1, 9),
        images=None,
        y_mask=np.array([1, 0, 1, 0, 1, 1, 1, 1], dtype=bool),
        sizes=np.array([2, 2, 4]),
        sample_pretokenized_paths=["/data/a", "/data/b/", "/data/a/"],
    )
    state = {"global_sequence_id": 123}
    iterator._loader = SimpleNamespace(get_batch=lambda: batch, get_state=lambda: state)
    result, returned_state = iterator.next_with_state()
    assert returned_state is state
    torch.testing.assert_close(
        result["data_source_ids"], torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    )
    iterator._data_source_indices = None
    ordinary, ordinary_state = iterator.next_with_state()
    assert ordinary_state is state
    assert set(result) - set(ordinary) == {"data_source_ids"}
    for key in ordinary:
        torch.testing.assert_close(result[key], ordinary[key])


@pytest.mark.parametrize(
    ("paths", "sizes", "message"),
    [
        (None, [4], "document-aligned"),
        (["/data/a"], None, "document-aligned"),
        (["/unknown"], [4], "Unconfigured"),
        (["/data/a"], [3], "complete token batch"),
    ],
)
def test_training_source_ids_reject_invalid_provenance(paths, sizes, message):
    iterator = MistralPretokenizedIterator.__new__(MistralPretokenizedIterator)
    iterator._micro_batch_size, iterator._seq_len = 1, 4
    iterator._data_source_indices = {"/data/a": 0}
    with pytest.raises(ValueError, match=message):
        iterator._build_data_source_ids(
            SimpleNamespace(sample_pretokenized_paths=paths, sizes=sizes)
        )


@pytest.mark.parametrize("preserve_logical_seqlens", [False, True])
@pytest.mark.parametrize("padding_source", ["none", "synthetic", "all", "unknown"])
def test_thd_input_lengths_do_not_use_label_mask(padding_source, preserve_logical_seqlens):
    iterator = MistralPretokenizedIterator.__new__(MistralPretokenizedIterator)
    iterator._micro_batch_size = 1
    iterator._seq_len = 8
    iterator._include_loss_modality = False
    iterator._preserve_logical_seqlens = preserve_logical_seqlens
    batch = SimpleNamespace(
        x=np.arange(8, dtype=np.int64),
        y=np.arange(1, 9, dtype=np.int64),
        y_mask=np.array([True, False, True, False, False, False, False, False]),
        sizes=np.array([4, 4], dtype=np.int64),
        images=None,
        num_pad_tokens=0,
    )
    if padding_source == "synthetic":
        batch.logical_sizes = np.array([4, 0], dtype=np.int64)
        batch.num_pad_tokens = 4
    elif padding_source == "all":
        batch.logical_sizes = np.array([0, 0], dtype=np.int64)
        batch.num_pad_tokens = 8
    elif padding_source == "unknown":
        batch.logical_sizes = None
        batch.num_pad_tokens = 3
        if preserve_logical_seqlens:
            with pytest.raises(ValueError, match="preserve logical document lengths"):
                iterator._convert_batch_thd(batch)
            return

    result = iterator._convert_batch_thd(batch)

    assert "padding_mask" not in result
    if preserve_logical_seqlens and batch.num_pad_tokens:
        assert result["cu_seqlens_padded"].tolist() == [[0, 4, 8]]
        assert result["cu_seqlens"].tolist() == (
            [[0, 4, 4]] if padding_source == "synthetic" else [[0, 0, 0]]
        )
    else:
        assert "cu_seqlens_padded" not in result
        assert result["cu_seqlens"].tolist() == [[0, 4, 8]]
    # A real image/prompt row has no LM target but still counts toward its document.
    if padding_source != "all":
        assert result["loss_mask"][0, 1] == 0
        assert result["cu_seqlens"][0, 1] == 4


def test_thd_padding_documents_preserve_collated_sample_offsets():
    iterator = MistralPretokenizedIterator.__new__(MistralPretokenizedIterator)
    iterator._micro_batch_size = 2
    iterator._seq_len = 8
    iterator._include_loss_modality = False
    iterator._preserve_logical_seqlens = True
    batch = SimpleNamespace(
        x=np.arange(16, dtype=np.int64),
        y=np.arange(1, 17, dtype=np.int64),
        y_mask=None,
        sizes=np.array([4, 4, 8], dtype=np.int64),
        logical_sizes=np.array([4, 0, 8], dtype=np.int64),
        images=None,
        num_pad_tokens=4,
    )
    result = iterator._convert_batch_thd(batch)
    assert result["cu_seqlens"].tolist() == [[0, 4, 4], [0, 8, 8]]
    assert result["cu_seqlens_padded"].tolist() == [[0, 4, 8], [0, 8, 8]]
    assert "padding_mask" not in result


@pytest.mark.parametrize("preserve_logical_seqlens", [False, True])
@pytest.mark.parametrize("second_row_is_padding", [False, True])
def test_native_suffix_padding_keeps_document_lengths(
    preserve_logical_seqlens, second_row_is_padding
):
    first = pad_sequence(
        SequenceOfSamples(
            x=[10, 11, 18, 19, 14],
            y=[11, 18, 19, 14, 15],
            mask=[False, True, True, False, True],
            sizes=[2, 3],
            doc_ids=[101, 102],
            image_positions=[Span(start=2, end=4)],
            image_pixels_loaded=False,
        ),
        mult_of=8,
        pad_id=0,
    )
    if second_row_is_padding:
        second = SequenceOfSamples(
            x=[0] * 8, y=[0] * 8, mask=[True] * 8, sizes=[8], num_pad_tokens=8, doc_ids=[-1]
        )
    else:
        second = pad_sequence(
            SequenceOfSamples(
                x=[20, 21, 22], y=[21, 22, 23], mask=[True, False, True], sizes=[3], doc_ids=[103]
            ),
            mult_of=8,
            pad_id=0,
        )
    batch = batch_from_sequences(
        SampleType.PRETRAIN, [first, second], ["image", "text"], ["/image", "/text"]
    )
    assert batch.sizes.tolist() == [2, 6, 8]
    assert batch.logical_sizes.tolist() == [2, 3, 0 if second_row_is_padding else 3]
    iterator = MistralPretokenizedIterator.__new__(MistralPretokenizedIterator)
    iterator._micro_batch_size = 2
    iterator._seq_len = 8
    iterator._img_id = 18
    iterator._include_loss_modality = False
    iterator._preserve_logical_seqlens = preserve_logical_seqlens
    result = iterator._convert_batch_thd(batch)
    physical = [[0, 2, 8], [0, 8, 8]]
    second_length = 0 if second_row_is_padding else 3
    if preserve_logical_seqlens:
        assert result["cu_seqlens"].tolist() == [[0, 2, 5], [0, second_length, second_length]]
        assert result["cu_seqlens_padded"].tolist() == physical
    else:
        assert result["cu_seqlens"].tolist() == physical
        assert "cu_seqlens_padded" not in result
    assert "padding_mask" not in result
    assert result["loss_mask"][0, 0] == result["loss_mask"][0, 3] == 0
    assert result["image_token_mask"][0, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Batch.cuda requires CUDA")
def test_native_logical_lengths_survive_batch_device_round_trip():
    sequence = pad_sequence(
        SequenceOfSamples(
            x=[10, 11, 12], y=[11, 12, 13], mask=[True, False, True], sizes=[3], doc_ids=[1]
        ),
        mult_of=8,
        pad_id=0,
    )
    batch = batch_from_sequences(SampleType.PRETRAIN, [sequence], ["text"], ["/text"])
    restored = batch.cuda(non_blocking=False).cpu()
    np.testing.assert_array_equal(restored.logical_sizes, np.array([3], dtype=np.int64))
    np.testing.assert_array_equal(restored.sizes, np.array([8], dtype=np.int64))
