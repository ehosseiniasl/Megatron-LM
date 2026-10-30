import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="fixed-overlap iterator tests require torch"
)

if importlib.util.find_spec("torch") is not None:
    import torch

    from mistral_adapter import pretokenized_fixed_overlap_iterator as fixed
    from mistral_adapter.pretokenized_config import PretokenizedConfig, SourceConfig
    from mistral_fork.mistral.data.pretokenized.megatron_index_reader import DType
    from mistral_fork.mistral.data.pretokenized.pretokenized_utils import DEFAULT_TOKEN_DTYPE


@pytest.mark.parametrize(
    ("token_count", "seq_len", "overlap", "expected"),
    [
        (3, 4, 2, [(0, 3, 0)]),
        (8, 4, 2, [(0, 4, 0), (2, 6, 1), (4, 8, 1)]),
        (8, 4, 0, [(0, 4, 0), (3, 7, 0), (6, 8, 0)]),
    ],
)
def test_iter_fixed_overlap_windows(
    token_count: int, seq_len: int, overlap: int, expected: list[tuple[int, int, int]]
) -> None:
    assert (
        list(
            fixed._iter_fixed_overlap_windows(
                token_count=token_count, seq_len=seq_len, overlap=overlap
            )
        )
        == expected
    )


def test_pack_windows_into_rows() -> None:
    windows = [
        fixed._FixedOverlapWindow(source_id=0, doc_id=0, start=start, end=end, skip_logprobs=0)
        for start, end in [(0, 4), (2, 6), (4, 8), (6, 10), (8, 11)]
    ]

    rows = fixed._pack_windows_into_rows(windows, row_seq_len=6)

    assert rows == [windows[:2], windows[2:4], windows[4:]]


class _FakeIndex:
    def __init__(self, tokens_with_eos: np.ndarray) -> None:
        self.sequence_count = 1
        self.sequence_lengths = np.array([len(tokens_with_eos)], dtype=np.int64)
        self.sequence_pointers = np.array([0], dtype=np.int64)


class _FakeBinReader:
    def __init__(self, tokens_with_eos: np.ndarray) -> None:
        self._tokens_with_eos = tokens_with_eos

    def read(self, *, dtype: type[np.number], offset: int, count: int) -> np.ndarray:
        assert dtype == DEFAULT_TOKEN_DTYPE
        start = offset // DType.size(DEFAULT_TOKEN_DTYPE)
        return self._tokens_with_eos[start : start + count]


def _patch_text_source(monkeypatch: pytest.MonkeyPatch, tokens_with_eos: np.ndarray) -> None:
    def load_text_source(
        source: "SourceConfig", *, skip_images: bool
    ) -> "fixed._PretokenizedTextSource":
        assert source.prefix_path == "/fake/source"
        assert skip_images
        return fixed._PretokenizedTextSource(
            prefix_path=source.prefix_path,
            index=_FakeIndex(tokens_with_eos),
            bin_reader=_FakeBinReader(tokens_with_eos),
        )

    monkeypatch.setattr(fixed, "_load_text_source", load_text_source)


def _config(*, interdocument_masking: bool = False) -> "PretokenizedConfig":
    return PretokenizedConfig(
        sources=[SourceConfig(prefix_path="/fake/source", weight=1.0, percentage_data=1.0)],
        bos_id=1,
        eos_id=2,
        pad_id=0,
        img_id=18,
        img_break_id=20,
        img_end_id=19,
        seed=0,
        epoch_index_cache_path="/unused",
        interdocument_masking=interdocument_masking,
        drop_bos=not interdocument_masking,
        original_seq_len=4,
        truncate_trailing_masked=True,
        skip_images=True,
    )


def test_fixed_overlap_iterator_scores_each_content_token_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_text_source(
        monkeypatch, np.array([1, 10, 11, 12, 13, 14, 15, 16, 2], dtype=DEFAULT_TOKEN_DTYPE)
    )

    iterator = fixed.MistralPretokenizedFixedOverlapIterator(
        config=_config(interdocument_masking=False),
        rank=0,
        world_size=1,
        micro_batch_size=2,
        window_seq_len=4,
        overlap=2,
        batch_seq_len=4,
    )

    assert iterator.num_windows == 3
    assert iterator.num_scored_tokens == 7
    assert iterator.expected_eval_tokens == 7
    assert iterator.num_global_microbatches == 2

    first = next(iterator)
    torch.testing.assert_close(first["tokens"], torch.tensor([[1, 10, 11, 0], [11, 12, 13, 0]]))
    torch.testing.assert_close(first["labels"], torch.tensor([[10, 11, 12, 0], [12, 13, 14, 0]]))
    torch.testing.assert_close(
        first["loss_mask"], torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    )
    torch.testing.assert_close(
        first["cu_seqlens"], torch.tensor([[0, 3, 4], [0, 3, 4]], dtype=torch.int32)
    )
    torch.testing.assert_close(first["max_seqlen"], torch.tensor([3, 3], dtype=torch.int32))

    second = next(iterator)
    torch.testing.assert_close(second["tokens"], torch.tensor([[13, 14, 15, 0], [0, 0, 0, 0]]))
    torch.testing.assert_close(second["labels"], torch.tensor([[14, 15, 16, 0], [0, 0, 0, 0]]))
    torch.testing.assert_close(
        second["loss_mask"], torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        second["cu_seqlens"], torch.tensor([[0, 3, 4], [0, 3, 4]], dtype=torch.int32)
    )
    torch.testing.assert_close(second["max_seqlen"], torch.tensor([3, 3], dtype=torch.int32))


def test_fixed_overlap_iterator_zero_overlap_scores_each_content_token_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_text_source(
        monkeypatch, np.array([1, 10, 11, 12, 13, 14, 15, 16, 2], dtype=DEFAULT_TOKEN_DTYPE)
    )

    iterator = fixed.MistralPretokenizedFixedOverlapIterator(
        config=_config(interdocument_masking=False),
        rank=0,
        world_size=1,
        micro_batch_size=2,
        window_seq_len=4,
        overlap=0,
        batch_seq_len=6,
    )

    assert iterator.num_windows == 3
    assert iterator.num_scored_tokens == 7
    assert iterator.num_global_microbatches == 1

    scored_labels: list[int] = []
    for _ in range(iterator.num_global_microbatches):
        batch = next(iterator)
        assert batch["tokens"].shape == (2, 6)
        scored_labels.extend(batch["labels"][batch["loss_mask"].bool()].tolist())

    assert scored_labels == list(range(10, 17))


def test_fixed_overlap_iterator_allows_interior_eos_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_text_source(monkeypatch, np.array([1, 10, 2, 12, 2], dtype=DEFAULT_TOKEN_DTYPE))

    iterator = fixed.MistralPretokenizedFixedOverlapIterator(
        config=_config(interdocument_masking=False),
        rank=0,
        world_size=1,
        micro_batch_size=1,
        window_seq_len=4,
        overlap=0,
        batch_seq_len=4,
    )

    assert iterator.num_scored_tokens == 3
    batch = next(iterator)
    torch.testing.assert_close(batch["tokens"], torch.tensor([[1, 10, 2, 0]]))
    torch.testing.assert_close(batch["labels"], torch.tensor([[10, 2, 12, 0]]))
    torch.testing.assert_close(batch["loss_mask"], torch.tensor([[1.0, 1.0, 1.0, 0.0]]))


@pytest.mark.parametrize("preserve_logical_seqlens", [False, True])
def test_fixed_overlap_iterator_emits_row_shaped_thd_batches(
    monkeypatch: pytest.MonkeyPatch, preserve_logical_seqlens: bool
) -> None:
    _patch_text_source(
        monkeypatch, np.array([1, 10, 11, 12, 13, 14, 15, 16, 2], dtype=DEFAULT_TOKEN_DTYPE)
    )

    iterator = fixed.MistralPretokenizedFixedOverlapIterator(
        config=_config(interdocument_masking=True),
        rank=0,
        world_size=1,
        micro_batch_size=1,
        window_seq_len=4,
        overlap=2,
        batch_seq_len=8,
        preserve_logical_seqlens=preserve_logical_seqlens,
    )

    assert iterator.num_windows == 3
    assert iterator.num_scored_tokens == 7
    assert iterator.num_global_microbatches == 2

    first = next(iterator)
    torch.testing.assert_close(first["tokens"], torch.tensor([[1, 10, 11, 11, 12, 13, 0, 0]]))
    torch.testing.assert_close(first["labels"], torch.tensor([[10, 11, 12, 12, 13, 14, 0, 0]]))
    torch.testing.assert_close(
        first["loss_mask"], torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(first["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1]]))
    assert first["cu_seqlens"].tolist() == (
        [[0, 3, 6, 6]] if preserve_logical_seqlens else [[0, 3, 6, 8]]
    )
    torch.testing.assert_close(first["max_seqlen"], torch.tensor([3], dtype=torch.int32))
    assert "padding_mask" not in first
    if preserve_logical_seqlens:
        assert first["cu_seqlens_padded"].tolist() == [[0, 3, 6, 8]]
    else:
        assert "cu_seqlens_padded" not in first
    # Overlap contributes to document length even when its LM target is masked.
    assert first["cu_seqlens"][0, 2] - first["cu_seqlens"][0, 1] == 3

    second = next(iterator)
    torch.testing.assert_close(second["tokens"], torch.tensor([[13, 14, 15, 0, 0, 0, 0, 0]]))
    torch.testing.assert_close(second["labels"], torch.tensor([[14, 15, 16, 0, 0, 0, 0, 0]]))
    torch.testing.assert_close(
        second["loss_mask"], torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    )
    assert second["cu_seqlens"].tolist() == (
        [[0, 3, 3, 3]] if preserve_logical_seqlens else [[0, 3, 6, 8]]
    )
    torch.testing.assert_close(second["max_seqlen"], torch.tensor([3], dtype=torch.int32))
    assert "padding_mask" not in second
    if preserve_logical_seqlens:
        assert second["cu_seqlens_padded"].tolist() == [[0, 3, 6, 8]]

    empty_overrun = next(iterator)
    assert "padding_mask" not in empty_overrun
    torch.testing.assert_close(empty_overrun["loss_mask"], torch.zeros((1, 8), dtype=torch.float32))
    assert empty_overrun["cu_seqlens"].tolist() == (
        [[0, 0, 0, 0]] if preserve_logical_seqlens else [[0, 3, 6, 8]]
    )
    if preserve_logical_seqlens:
        assert empty_overrun["cu_seqlens_padded"].tolist() == [[0, 3, 6, 8]]
    torch.testing.assert_close(empty_overrun["max_seqlen"], torch.tensor([3], dtype=torch.int32))


def test_fixed_overlap_iterator_rewinds_from_saved_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_text_source(
        monkeypatch, np.array([1, 10, 11, 12, 13, 14, 15, 16, 2], dtype=DEFAULT_TOKEN_DTYPE)
    )

    iterator = fixed.MistralPretokenizedFixedOverlapIterator(
        config=_config(interdocument_masking=True),
        rank=0,
        world_size=1,
        micro_batch_size=1,
        window_seq_len=4,
        overlap=2,
        batch_seq_len=8,
    )

    state = iterator.save_state()
    first = next(iterator)
    next(iterator)
    iterator.set_state(state)
    replayed = next(iterator)

    torch.testing.assert_close(replayed["tokens"], first["tokens"])
    torch.testing.assert_close(replayed["loss_mask"], first["loss_mask"])


def test_fixed_overlap_packed_iterator_dp_shards_same_deck(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_text_source(
        monkeypatch,
        np.array([1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2], dtype=DEFAULT_TOKEN_DTYPE),
    )

    iterators = [
        fixed.MistralPretokenizedFixedOverlapIterator(
            config=_config(interdocument_masking=True),
            rank=rank,
            world_size=2,
            micro_batch_size=1,
            window_seq_len=4,
            overlap=2,
            batch_seq_len=6,
        )
        for rank in range(2)
    ]

    assert iterators[0].num_windows == 5
    assert iterators[0].num_scored_tokens == 10
    assert iterators[0].num_global_microbatches == 2

    scored_labels: list[int] = []
    for _ in range(iterators[0].num_global_microbatches):
        for iterator in iterators:
            batch = next(iterator)
            scored_labels.extend(batch["labels"][batch["loss_mask"].bool()].tolist())

    assert scored_labels == list(range(10, 20))
