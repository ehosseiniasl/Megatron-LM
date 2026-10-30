import pickle

import pytest
from pydantic import ValidationError

from mistral_adapter.pretokenized_config import CONFIG_VERSION, PretokenizedConfig
from mistral_adapter.pretokenized_config_utils import (
    build_image_encoder_fn,
    config_to_source_items,
)


def _base_config() -> dict:
    return {
        "version": CONFIG_VERSION,
        "sources": [
            {
                "prefix_path": "/tmp/source/chunks_00000-00001",
                "weight": 1.0,
                "percentage_data": 1.0,
            }
        ],
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 0,
        "img_id": 18,
        "img_break_id": 20,
        "img_end_id": 19,
        "seed": 0,
        "epoch_index_cache_path": "/tmp/epoch_indexes",
        "drop_bos": False,
        "original_seq_len": 8,
    }


@pytest.mark.parametrize("field", ["img_id", "img_break_id", "img_end_id"])
def test_image_token_ids_are_required_and_not_nullable(field: str) -> None:
    missing = _base_config()
    missing["truncate_trailing_masked"] = True
    del missing[field]
    with pytest.raises(ValidationError, match=field):
        PretokenizedConfig.model_validate(missing)

    null = _base_config()
    null["truncate_trailing_masked"] = True
    null[field] = None
    with pytest.raises(ValidationError, match=field):
        PretokenizedConfig.model_validate(null)


def test_truncate_trailing_masked_is_required() -> None:
    with pytest.raises(ValidationError, match="truncate_trailing_masked"):
        PretokenizedConfig.model_validate(_base_config())


def test_drop_bos_is_required() -> None:
    raw = _base_config()
    raw["truncate_trailing_masked"] = True
    del raw["drop_bos"]

    with pytest.raises(ValidationError, match="drop_bos"):
        PretokenizedConfig.model_validate(raw)


def test_truncate_trailing_masked_must_be_true() -> None:
    raw = _base_config()
    raw["truncate_trailing_masked"] = False

    with pytest.raises(ValidationError, match="requires truncate_trailing_masked=true"):
        PretokenizedConfig.model_validate(raw)


def test_truncate_trailing_masked_true_is_valid() -> None:
    raw = _base_config()
    raw["truncate_trailing_masked"] = True

    config = PretokenizedConfig.model_validate(raw)

    assert config.truncate_trailing_masked is True
    assert config.perplexity_eval_sets == []
    assert config.drop_bos is False


@pytest.mark.parametrize("value", [None, 2])
def test_n_epoches_to_prebuild_is_optional_and_propagated(
    value: int | None,
) -> None:
    raw = _base_config()
    raw["truncate_trailing_masked"] = True
    if value is not None:
        raw["sources"][0]["n_epoches_to_prebuild"] = value

    config = PretokenizedConfig.model_validate(raw)

    assert config.sources[0].n_epoches_to_prebuild == value
    assert config_to_source_items(config)[0].n_epoches_to_prebuild == value


@pytest.mark.parametrize("value", [0, -1])
def test_n_epoches_to_prebuild_must_be_positive(value: int) -> None:
    raw = _base_config()
    raw["truncate_trailing_masked"] = True
    raw["sources"][0]["n_epoches_to_prebuild"] = value

    with pytest.raises(ValidationError, match="n_epoches_to_prebuild"):
        PretokenizedConfig.model_validate(raw)


def test_perplexity_eval_set_name_must_be_non_empty() -> None:
    raw = _base_config()
    raw["truncate_trailing_masked"] = True
    raw["perplexity_eval_sets"] = [
        {
            "name": "",
            "sources": [
                {
                    "prefix_path": "/tmp/source/chunks_00000-00001",
                    "weight": 1.0,
                    "percentage_data": 1.0,
                }
            ],
            "fixed_overlap": 1,
        }
    ]

    with pytest.raises(ValidationError, match="name must be non-empty"):
        PretokenizedConfig.model_validate(raw)


def test_image_encoder_factory_is_pickleable_for_process_workers() -> None:
    raw = _base_config()
    raw["truncate_trailing_masked"] = True
    raw["sources"][0]["image_transform_type"] = "no_transform"
    config = PretokenizedConfig.model_validate(raw)

    factory = build_image_encoder_fn(config)
    assert factory is not None
    restored = pickle.loads(pickle.dumps(factory))

    assert restored("no_transform").uuid() == factory("no_transform").uuid()
