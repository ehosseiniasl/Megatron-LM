from types import SimpleNamespace

import pytest
import torch
import yaml

from mistral_adapter import train_loop_helpers as helpers
from mistral_adapter.pretokenized_config import (
    CONFIG_VERSION,
    PretokenizedConfig,
    PretokenizedValidationConfig,
    SourceConfig,
)
from mistral_adapter.train_loop_helpers import (
    _assert_config_version,
    _assert_seq_aux_loss_interdocument_masking_supported,
    _fixed_overlap_eval_seq_len_from_fields,
    _FixedValidationIterator,
    _validate_interdocument_masking_args,
    _validate_pretokenized_training_args,
)


def _source() -> SourceConfig:
    return SourceConfig(prefix_path="/fake/source", weight=1.0, percentage_data=1.0)


def _config(*, interdocument_masking: bool = True) -> PretokenizedConfig:
    return PretokenizedConfig(
        sources=[_source()],
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
        validation_set=PretokenizedValidationConfig(
            valid_sources=[_source()], valid_num_tokens=128, valid_seed=1
        ),
    )


@pytest.mark.parametrize("interdocument_masking", [True, False])
def test_interdocument_masking_must_match_cli(interdocument_masking: bool) -> None:
    _validate_interdocument_masking_args(
        SimpleNamespace(dataloader_inter_document_masking=interdocument_masking),
        _config(interdocument_masking=interdocument_masking),
    )

    with pytest.raises(AssertionError, match="must agree"):
        _validate_interdocument_masking_args(
            SimpleNamespace(dataloader_inter_document_masking=not interdocument_masking),
            _config(interdocument_masking=interdocument_masking),
        )


def test_config_version_must_match_this_megatron_build() -> None:
    _assert_config_version(_config())

    old_config = _config().model_copy(update={"version": -1})
    with pytest.raises(AssertionError, match="version mismatch"):
        _assert_config_version(old_config)


def _training_args(
    *,
    tensor_model_parallel_size: int = 1,
    micro_batch_size: int = 1,
    seq_length: int = 4,
    dataloader_inter_document_masking: bool = True,
    moe_router_load_balancing_type: str | list[str] = "none",
    moe_aux_loss_coeff: float | list[float] = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        tensor_model_parallel_size=tensor_model_parallel_size,
        micro_batch_size=micro_batch_size,
        seq_length=seq_length,
        dataloader_inter_document_masking=dataloader_inter_document_masking,
        moe_router_load_balancing_type=moe_router_load_balancing_type,
        moe_aux_loss_coeff=moe_aux_loss_coeff,
    )


def test_interdocument_masking_allows_tp_and_mbs_without_seq_aux_loss() -> None:
    _validate_pretokenized_training_args(
        _training_args(
            tensor_model_parallel_size=2,
            micro_batch_size=2,
            moe_router_load_balancing_type="aux_loss",
        ),
        _config(),
    )


def test_no_interdocument_masking_allows_tp_and_mbs_with_seq_aux_loss() -> None:
    _validate_pretokenized_training_args(
        _training_args(
            tensor_model_parallel_size=2,
            micro_batch_size=2,
            dataloader_inter_document_masking=False,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
        ),
        _config(interdocument_masking=False),
    )


@pytest.mark.parametrize(
    ("tensor_model_parallel_size", "micro_batch_size"), [(1, 1), (2, 1), (1, 2)]
)
def test_seq_aux_loss_interdocument_masking_allows_tp_xor_mbs(
    tensor_model_parallel_size: int, micro_batch_size: int
) -> None:
    _assert_seq_aux_loss_interdocument_masking_supported(
        tensor_model_parallel_size=tensor_model_parallel_size, micro_batch_size=micro_batch_size
    )


def test_seq_aux_loss_interdocument_masking_rejects_tp_and_mbs() -> None:
    with pytest.raises(AssertionError, match="1783061005464749"):
        _assert_seq_aux_loss_interdocument_masking_supported(
            tensor_model_parallel_size=2, micro_batch_size=2
        )


def test_interdocument_masking_rejects_tp_and_mbs_with_seq_aux_loss() -> None:
    with pytest.raises(AssertionError, match="1783061005464749"):
        _validate_pretokenized_training_args(
            _training_args(
                tensor_model_parallel_size=2,
                micro_batch_size=2,
                moe_router_load_balancing_type=["aux_loss", "seq_aux_loss"],
                moe_aux_loss_coeff=[1.0, 1.0],
            ),
            _config(),
        )


def test_unmasked_data_skips_interdocument_seq_aux_loss_guard() -> None:
    _validate_pretokenized_training_args(
        _training_args(
            tensor_model_parallel_size=2,
            micro_batch_size=2,
            dataloader_inter_document_masking=False,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
        ),
        _config(interdocument_masking=False),
    )


def test_interdocument_masking_uses_overridden_tp_for_seq_aux_loss() -> None:
    with pytest.raises(AssertionError, match="1783061005464749"):
        _validate_pretokenized_training_args(
            _training_args(
                tensor_model_parallel_size=1,
                micro_batch_size=2,
                moe_router_load_balancing_type="seq_aux_loss",
                moe_aux_loss_coeff=1.0,
            ),
            _config(),
            tensor_model_parallel_size=2,
        )


def test_fixed_overlap_eval_seq_len_defaults_to_training_seq_len() -> None:
    seq_len = _fixed_overlap_eval_seq_len_from_fields(
        fixed_overlap=0, fixed_overlap_eval_seq_len=None, original_seq_len=4
    )

    assert seq_len == 4


def test_fixed_overlap_eval_seq_len_can_be_overridden() -> None:
    seq_len = _fixed_overlap_eval_seq_len_from_fields(
        fixed_overlap=0, fixed_overlap_eval_seq_len=6, original_seq_len=4
    )

    assert seq_len == 6


def test_fixed_overlap_eval_seq_len_zero_fails_loudly() -> None:
    with pytest.raises(AssertionError, match="must be > 1"):
        _fixed_overlap_eval_seq_len_from_fields(
            fixed_overlap=0, fixed_overlap_eval_seq_len=0, original_seq_len=4
        )


class _CountingIterator:
    def __init__(self) -> None:
        self.value = 0
        self.closed = False

    def ensure_started(self) -> None:
        return

    def save_state(self) -> dict[str, int]:
        return {"value": self.value}

    def set_state(self, data: dict[str, int]) -> None:
        self.value = data["value"]

    def __next__(self) -> dict[str, int]:
        result = {"value": self.value}
        self.value += 1
        return result

    def close(self) -> None:
        self.closed = True


def test_fixed_validation_iterator_replays_after_reset_period() -> None:
    inner = _CountingIterator()
    iterator = _FixedValidationIterator(
        inner=inner, eval_iters=2, reset_period=3, expected_eval_tokens=123
    )

    assert len(iterator) == 2
    assert iterator.expected_eval_tokens == 123
    assert [next(iterator)["value"] for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]

    iterator.close()

    assert inner.closed


def _raw_config(*, include_perplexity_eval_sets: bool = True) -> dict:
    validation_set = {
        "valid_sources": [{"prefix_path": "/valid/source", "weight": 1.0, "percentage_data": 1.0}],
        "valid_num_tokens": 64,
        "valid_seed": 123,
    }
    raw = {
        "version": CONFIG_VERSION,
        "sources": [{"prefix_path": "/train/source", "weight": 1.0, "percentage_data": 1.0}],
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 0,
        "img_id": 18,
        "img_break_id": 20,
        "img_end_id": 19,
        "seed": 0,
        "epoch_index_cache_path": "/unused",
        "interdocument_masking": True,
        "drop_bos": False,
        "original_seq_len": 8,
        "truncate_trailing_masked": True,
        "skip_images": True,
        "validation_set": validation_set,
    }
    if include_perplexity_eval_sets:
        raw["perplexity_eval_sets"] = [
            {
                "name": "catlm",
                "sources": [
                    {"prefix_path": "/catlm/source", "weight": 1.0, "percentage_data": 1.0}
                ],
                "fixed_overlap": 2,
                "fixed_overlap_eval_seq_len": 8,
            }
        ]
    return raw


def _write_config(tmp_path, raw: dict) -> str:
    path = tmp_path / "pretokenized.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def _args(config_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        pretokenized_config=config_path,
        global_batch_size=4,
        eval_global_batch_size=4,
        micro_batch_size=2,
        eval_micro_batch_size=2,
        tensor_model_parallel_size=1,
        seq_length=8,
        dataloader_inter_document_masking=True,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        overwrite_percentage_data_multiplier=None,
        multiple_validation_sets=False,
        validation_set_names=None,
        pretokenized_allow_build_of_epoch_index=False,
        pretokenized_allow_partial_epoch_index_cache=False,
        pretokenized_build_of_epoch_index_timeout_s=3600,
        train_samples=100,
        train_iters=10,
        consumed_train_samples=0,
        phase_transition_samples=None,
        num_workers=0,
        save="/tmp/save",
    )


class _FakePretokenizedIterator:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        _FakePretokenizedIterator.instances.append(self)

    def ensure_started(self) -> None:
        return

    def save_state(self) -> dict[str, int]:
        return {"offset": 0}

    def set_state(self, data: dict[str, int]) -> None:
        return

    def __next__(self) -> dict:
        return {}

    def close(self) -> None:
        return


class _FakeFixedOverlapIterator:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.num_global_microbatches = 3
        self.num_windows = 5
        self.num_scored_tokens = 11
        _FakeFixedOverlapIterator.instances.append(self)

    def ensure_started(self) -> None:
        return

    def save_state(self) -> dict[str, int]:
        return {"offset": 0}

    def set_state(self, data: dict[str, int]) -> None:
        return

    def __next__(self) -> dict:
        return {}

    def close(self) -> None:
        return


def _patch_iterator_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    from mistral_adapter import pretokenized_fixed_overlap_iterator, pretokenized_iterator

    _FakePretokenizedIterator.instances = []
    _FakeFixedOverlapIterator.instances = []
    monkeypatch.setattr(
        pretokenized_iterator, "MistralPretokenizedIterator", _FakePretokenizedIterator
    )
    monkeypatch.setattr(
        pretokenized_fixed_overlap_iterator,
        "MistralPretokenizedFixedOverlapIterator",
        _FakeFixedOverlapIterator,
    )
    monkeypatch.setattr(helpers.parallel_state, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(helpers.parallel_state, "get_data_parallel_world_size", lambda: 1)


@pytest.mark.parametrize(
    ("consumed", "transition", "remaining"),
    [(0, None, 100), (25, None, 75), (0, 80, 80), (25, 80, 55), (80, 80, 20), (85, 80, 15)],
)
def test_num_training_tokens_in_phase_uses_remaining_sample_horizon(
    consumed: int, transition: int | None, remaining: int
) -> None:
    args = _args("/unused")
    args.consumed_train_samples = consumed
    args.phase_transition_samples = transition

    assert helpers.num_training_tokens_in_phase(args) == remaining * args.seq_length


def test_num_training_tokens_in_phase_uses_full_iteration_horizon() -> None:
    args = _args("/unused")
    args.train_samples = None
    args.phase_transition_samples = 32
    args.consumed_train_samples = 33

    assert helpers.num_training_tokens_in_phase(args) == 10 * 4 * 8


def test_build_pretokenized_iterator_uses_num_training_tokens_in_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_iterator_builders(monkeypatch)
    monkeypatch.setattr(helpers, "_maybe_restore_pretokenized_state", lambda *_args: None)
    monkeypatch.setattr(helpers, "num_training_tokens_in_phase", lambda _args: 1234)
    args = _args(_write_config(tmp_path, _raw_config()))

    helpers.build_pretokenized_iterator(args)

    assert _FakePretokenizedIterator.instances[0].kwargs["num_total_tokens"] == 1234


def test_build_valid_iterator_configures_trainlike_and_perplexity_sets(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_iterator_builders(monkeypatch)
    args = _args(_write_config(tmp_path, _raw_config()))

    iterators = helpers.build_pretokenized_valid_iterator(args)

    assert isinstance(iterators, list)
    assert len(iterators) == 2
    assert args.multiple_validation_sets is True
    assert args.validation_set_names == ["trainlike", "catlm"]
    assert args.eval_iters == [2, 2]
    assert iterators[1].expected_eval_tokens == 11

    assert len(_FakePretokenizedIterator.instances) == 1
    trainlike_config = _FakePretokenizedIterator.instances[0].kwargs["config"]
    assert trainlike_config.sources[0].prefix_path == "/valid/source"
    assert trainlike_config.seed == 123
    assert _FakePretokenizedIterator.instances[0].kwargs["num_total_tokens"] == 64

    assert len(_FakeFixedOverlapIterator.instances) == 1
    ppl_kwargs = _FakeFixedOverlapIterator.instances[0].kwargs
    assert ppl_kwargs["config"].sources[0].prefix_path == "/catlm/source"
    assert ppl_kwargs["overlap"] == 2
    assert ppl_kwargs["window_seq_len"] == 8


def test_build_valid_iterator_configures_single_trainlike_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_iterator_builders(monkeypatch)
    raw = _raw_config(include_perplexity_eval_sets=False)
    args = _args(_write_config(tmp_path, raw))

    iterator = helpers.build_pretokenized_valid_iterator(args)

    assert not isinstance(iterator, list)
    assert args.multiple_validation_sets is False
    assert args.validation_set_names is None
    assert args.eval_iters == 2
    assert len(_FakePretokenizedIterator.instances) == 1
    assert len(_FakeFixedOverlapIterator.instances) == 0


class _StateCapturingIterator:
    """Minimal stand-in for MistralPretokenizedIterator's restore contract."""

    def __init__(self, *, is_save_state_rank_independent: bool) -> None:
        self.is_save_state_rank_independent = is_save_state_rank_independent
        self.restored = None

    def set_state(self, state) -> None:
        self.restored = state


def _write_dataloader_state(root, iteration: int, file_rank: int, global_sequence_id: int) -> None:
    state_dir = root / f"iter_{iteration:07d}" / "mp_rank_00"
    state_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"dataloader_state_dict": {"global_sequence_id": global_sequence_id}},
        state_dir / f"train_dataloader_dprank{file_rank:03d}.pt",
    )


@pytest.mark.parametrize(
    ("is_save_state_rank_independent", "expected_global_sequence_id"), [(True, 1000), (False, 3000)]
)
def test_restore_reads_the_file_the_writer_produced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    is_save_state_rank_independent: bool,
    expected_global_sequence_id: int,
) -> None:
    """Rank-independent state comes from dprank000; sharded state comes from this rank's file."""
    from megatron.training import checkpointing

    # get_checkpoint_name reads these from the MPU globals whenever the caller leaves them
    # unset, and this test runs without initializing model parallel.
    monkeypatch.setattr(checkpointing.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(checkpointing.mpu, "get_expert_model_parallel_rank", lambda: 0)
    iteration = 2
    _write_dataloader_state(tmp_path, iteration, file_rank=0, global_sequence_id=1000)
    _write_dataloader_state(tmp_path, iteration, file_rank=3, global_sequence_id=3000)

    iterator = _StateCapturingIterator(
        is_save_state_rank_independent=is_save_state_rank_independent
    )
    args = SimpleNamespace(dataloader_save=str(tmp_path), iteration=iteration)

    helpers._maybe_restore_pretokenized_state(args, iterator, 3)

    assert iterator.restored == {"global_sequence_id": expected_global_sequence_id}


def test_restore_is_skipped_before_the_first_iteration(tmp_path) -> None:
    iterator = _StateCapturingIterator(is_save_state_rank_independent=True)
    args = SimpleNamespace(dataloader_save=str(tmp_path), iteration=0)

    helpers._maybe_restore_pretokenized_state(args, iterator, 3)

    assert iterator.restored is None
