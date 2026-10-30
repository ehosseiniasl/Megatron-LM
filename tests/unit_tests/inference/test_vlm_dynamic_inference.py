# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    vlm_dynamic_inference,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
    chat_completions,
)


def test_detects_mimo_mistral_vit_checkpoint(monkeypatch):
    args = SimpleNamespace(vision_projection_type="mlp")
    checkpoint_args = SimpleNamespace(
        model_provider="nemotron-moe-mistral-vit",
        mistral_vit_model_size="1.6b",
        vision_projection_type="mlp2",
        img_h=1540,
        img_w=1540,
        patch_dim=14,
        image_token_id=42,
    )
    monkeypatch.setattr(
        vlm_dynamic_inference, "load_args_from_checkpoint", lambda _args: (_args, checkpoint_args)
    )

    assert vlm_dynamic_inference._detect_vlm_from_checkpoint(args)
    assert args._vlm_is_mimo
    assert args.language_model_type == "nemotron6-moe"
    assert args.vision_model_type == "mistral-vit-1p6b"
    assert args.vision_projection_type == "mlp2"
    assert args.encoder_num_layers == 30
    assert args.spatial_merge_size == 2
    assert args.dynamic_resolution
    assert args.dynamic_resolution_max_patches == 12100
    assert args.image_token_id == 42
    assert args._vlm_checkpoint_prefix_map == {
        "language_model.": "language_model.module.module.",
        "vision_model.": (
            "modality_submodules.mistral_vit_encoder.module.module." "encoders.mistral_vit_encoder."
        ),
        "vision_projection.": (
            "modality_submodules.mistral_vit_encoder.module.module.input_projections.0."
        ),
    }


def test_mimo_detection_preserves_explicit_cli_values(monkeypatch):
    args = SimpleNamespace(vision_projection_type="affine")
    checkpoint_args = SimpleNamespace(
        model_provider="nemotron-moe-mistral-vit",
        mistral_vit_model_size="1.6b",
        vision_projection_type="mlp2",
    )
    monkeypatch.setattr(
        vlm_dynamic_inference, "load_args_from_checkpoint", lambda _args: (_args, checkpoint_args)
    )

    assert vlm_dynamic_inference._detect_vlm_from_checkpoint(
        args, user_passed_attrs={"vision_projection_type"}
    )
    assert args.vision_projection_type == "affine"


def test_checkpoint_model_audit_ignores_extra_state():
    prefix = ("language_model.module.module.", "modality_submodules.")
    vlm_dynamic_inference._validate_checkpoint_model_mismatches(
        prefix,
        {"language_model.module.module.decoder.layers.0._extra_state"},
        {"vision_model.layers.0._extra_state"},
    )


@pytest.mark.parametrize(
    ("missing_keys", "unexpected_keys"),
    [
        ({"language_model.module.module.decoder.layers.0.weight"}, set()),
        (set(), {"vision_projection.linear_fc1.weight"}),
    ],
)
def test_checkpoint_model_audit_rejects_incomplete_loads(missing_keys, unexpected_keys):
    with pytest.raises(CheckpointingException, match="model-parameter audit failed"):
        vlm_dynamic_inference._validate_checkpoint_model_mismatches(
            ("language_model.module.module.", "modality_submodules."), missing_keys, unexpected_keys
        )


def test_mimo_model_load_uses_prefix_mapping_and_audit(monkeypatch):
    prefix_map = {
        "language_model.": "language_model.module.module.",
        "vision_model.": "modality_submodules.encoder.module.module.encoders.encoder.",
        "vision_projection.": ("modality_submodules.encoder.module.module.input_projections.0."),
    }
    checkpoint_prefixes = ("language_model.module.module.", "modality_submodules.encoder.")
    args = SimpleNamespace(
        _vlm_is_mimo=True,
        _vlm_checkpoint_prefix_map=prefix_map,
        _vlm_checkpoint_model_prefixes=checkpoint_prefixes,
        load="checkpoint",
        inference_ckpt_non_strict=False,
    )
    model = MagicMock()
    load_kwargs = {}

    monkeypatch.setattr(vlm_dynamic_inference, "get_args", lambda: args)
    monkeypatch.setattr(vlm_dynamic_inference, "_get_model", lambda *args, **kwargs: [model])

    def record_load_checkpoint(**kwargs):
        load_kwargs.update(kwargs)

    monkeypatch.setattr(vlm_dynamic_inference, "load_checkpoint", record_load_checkpoint)

    assert vlm_dynamic_inference.get_model(is_vlm=True) is model
    assert load_kwargs["model_sharded_state_dict_prefix_map"] == prefix_map
    validator = load_kwargs["checkpoint_mismatch_validator"]
    validator(set(), set())
    model.eval.assert_called_once_with()


def _detect(monkeypatch, args, checkpoint_args, user_passed_attrs=None):
    monkeypatch.setattr(
        vlm_dynamic_inference,
        "load_args_from_checkpoint",
        lambda runtime_args: (runtime_args, checkpoint_args),
    )
    return vlm_dynamic_inference._detect_vlm_from_checkpoint(args, user_passed_attrs)


def test_text_checkpoint_selects_hybrid_provider(monkeypatch):
    args = SimpleNamespace(
        model_provider="gpt", sequence_parallel=False, tensor_model_parallel_size=2
    )
    checkpoint_args = SimpleNamespace(
        language_model_type=None, hybrid_layer_pattern="M*-", sequence_parallel=None
    )

    assert not _detect(monkeypatch, args, checkpoint_args)
    assert args.model_provider == "hybrid"
    assert args.sequence_parallel is True


def test_text_checkpoint_preserves_explicit_provider_args(monkeypatch):
    args = SimpleNamespace(
        model_provider="gpt", sequence_parallel=False, tensor_model_parallel_size=2
    )
    checkpoint_args = SimpleNamespace(
        language_model_type=None, model_provider="hybrid", sequence_parallel=True
    )

    assert not _detect(
        monkeypatch,
        args,
        checkpoint_args,
        user_passed_attrs={"model_provider", "sequence_parallel"},
    )
    assert args.model_provider == "gpt"
    assert args.sequence_parallel is False


@pytest.mark.parametrize("checkpoint_sequence_parallel", [True, None])
def test_text_checkpoint_clamps_sequence_parallel_at_tp1(monkeypatch, checkpoint_sequence_parallel):
    """``_detect_vlm_from_checkpoint`` runs after ``validate_args``, so it must
    re-apply the TP=1 guard itself for both the checkpoint-sourced and the
    hybrid-inferred value."""
    args = SimpleNamespace(
        model_provider="gpt", sequence_parallel=False, tensor_model_parallel_size=1
    )
    checkpoint_args = SimpleNamespace(
        language_model_type=None,
        hybrid_layer_pattern="M*-",
        sequence_parallel=checkpoint_sequence_parallel,
    )

    assert not _detect(monkeypatch, args, checkpoint_args)
    assert args.model_provider == "hybrid"
    assert args.sequence_parallel is False


def _tokenizer(
    chat_template="{{ messages }}",
    image_token="[IMG]",
    image_token_id=10,
    image_break_token="[IMG_BREAK]",
    image_break_token_id=12,
    image_end_token="[IMG_END]",
    image_end_token_id=13,
):
    tokens = {
        "image_token": (image_token, image_token_id),
        "image_break_token": (image_break_token, image_break_token_id),
        "image_end_token": (image_end_token, image_end_token_id),
    }
    present = {token: token_id for token, token_id in tokens.values() if token is not None}

    tokenizer = SimpleNamespace(
        chat_template=chat_template,
        all_special_tokens=list(present),
        all_special_ids=list(present.values()),
        convert_tokens_to_ids=lambda token: present[token],
        encode=lambda token, add_special_tokens=False: [present[token]],
    )
    for name, (token, token_id) in tokens.items():
        setattr(tokenizer, name, token)
        setattr(tokenizer, f"{name}_id", token_id)
    return tokenizer


def _args(
    image_token_id=None,
    image_break_token_id=None,
    image_end_token_id=None,
    pretokenized_config=None,
    **kwargs,
):
    return SimpleNamespace(
        pretokenized_config=pretokenized_config,
        image_token_id=image_token_id,
        image_break_token_id=image_break_token_id,
        image_end_token_id=image_end_token_id,
        **kwargs,
    )


def _null_tokenizer():
    """Stand-in for the NullTokenizer used by pre-tokenized Mistral checkpoints."""
    return SimpleNamespace()


def test_resolve_vlm_special_tokens_accepts_agreeing_sources():
    args = _args(image_token_id=10, image_break_token_id=12, image_end_token_id=13)

    special_tokens = vlm_dynamic_inference.resolve_vlm_special_tokens(_tokenizer(), args)

    assert special_tokens.image_token_id == 10
    assert special_tokens.image_break_token_id == 12
    assert special_tokens.image_end_token_id == 13
    assert special_tokens.supports_native_image_tokens


def test_resolve_vlm_special_tokens_rejects_source_conflict():
    args = _args(image_token_id=99)

    with pytest.raises(ValueError, match="Conflicting image_token_id"):
        vlm_dynamic_inference.resolve_vlm_special_tokens(_tokenizer(), args)


def test_resolve_vlm_special_tokens_reads_pretokenized_config(monkeypatch):
    """The pretokenized config is authoritative for NullTokenizer checkpoints."""
    monkeypatch.setattr(
        vlm_dynamic_inference,
        "_pretokenized_image_token_ids",
        lambda args: {"image_token_id": 18, "image_break_token_id": 20, "image_end_token_id": 19},
    )
    args = _args(image_token_id=18, pretokenized_config="/some/pretokenized.yaml")

    special_tokens = vlm_dynamic_inference.resolve_vlm_special_tokens(_null_tokenizer(), args)

    assert (special_tokens.image_token_id, special_tokens.image_break_token_id) == (18, 20)
    assert special_tokens.image_end_token_id == 19
    assert special_tokens.supports_native_image_tokens


def test_pretokenized_ids_read_config_predating_current_schema(tmp_path):
    """A training-era config must not have to satisfy the current full schema.

    The N4V6 checkpoint's config is version 8 and omits ``drop_bos``, which the
    current ``PretokenizedConfig`` requires. Inference only needs the image IDs.
    """
    config = tmp_path / "pretokenized.yaml"
    config.write_text(
        "version: 8\n"
        "sources: []\n"
        "bos_id: 1\n"
        "eos_id: 2\n"
        "pad_id: 0\n"
        "img_id: 18\n"
        "img_break_id: 20\n"
        "img_end_id: 19\n"
        "seed: 0\n"
        "epoch_index_cache_path: /tmp/cache\n"
        "original_seq_len: 4096\n"
        "truncate_trailing_masked: false\n"
    )

    ids = vlm_dynamic_inference._pretokenized_image_token_ids(
        SimpleNamespace(pretokenized_config=str(config))
    )

    assert ids == {"image_token_id": 18, "image_break_token_id": 20, "image_end_token_id": 19}


def test_pretokenized_ids_skipped_when_images_disabled(tmp_path):
    config = tmp_path / "pretokenized.yaml"
    config.write_text("version: 8\nskip_images: true\nimg_id: 18\n")

    assert (
        vlm_dynamic_inference._pretokenized_image_token_ids(
            SimpleNamespace(pretokenized_config=str(config))
        )
        == {}
    )


def test_resolve_vlm_special_tokens_rejects_pretokenized_checkpoint_conflict(monkeypatch):
    monkeypatch.setattr(
        vlm_dynamic_inference, "_pretokenized_image_token_ids", lambda args: {"image_token_id": 18}
    )
    args = _args(image_token_id=99, pretokenized_config="/some/pretokenized.yaml")

    with pytest.raises(ValueError, match="Conflicting image_token_id"):
        vlm_dynamic_inference.resolve_vlm_special_tokens(_null_tokenizer(), args)


def test_resolve_vlm_special_tokens_without_break_end_disables_native():
    """A checkpoint carrying only image_token_id must not enable native tokens."""
    args = _args(image_token_id=18)

    special_tokens = vlm_dynamic_inference.resolve_vlm_special_tokens(_null_tokenizer(), args)

    assert special_tokens.image_token_id == 18
    assert special_tokens.image_break_token_id is None
    assert not special_tokens.supports_native_image_tokens


def test_resolve_vlm_special_tokens_requires_some_image_token_id():
    with pytest.raises(ValueError, match="Could not determine the VLM image token ID"):
        vlm_dynamic_inference.resolve_vlm_special_tokens(_null_tokenizer(), _args())


def test_resolve_vlm_special_tokens_tolerates_missing_chat_template():
    """NullTokenizer checkpoints have no chat template; that must not block serving."""
    args = _args(image_token_id=10, image_break_token_id=12, image_end_token_id=13)

    special_tokens = vlm_dynamic_inference.resolve_vlm_special_tokens(
        _tokenizer(chat_template=None), args
    )

    assert special_tokens.image_token_id == 10


def test_resolve_vlm_special_tokens_allows_absent_image_break():
    special_tokens = vlm_dynamic_inference.resolve_vlm_special_tokens(
        _tokenizer(image_break_token=None, image_break_token_id=None), _args()
    )

    assert special_tokens.image_break_token_id is None
    assert not special_tokens.supports_native_image_tokens


def test_resolve_vlm_special_tokens_requires_both_token_and_id():
    with pytest.raises(ValueError, match="must define both"):
        vlm_dynamic_inference.resolve_vlm_special_tokens(
            _tokenizer(image_end_token_id=None), _args()
        )


def test_resolve_vlm_special_tokens_rejects_multi_token_encoding():
    tokenizer = _tokenizer()
    tokenizer.encode = lambda token, add_special_tokens=False: [1, 2]

    with pytest.raises(ValueError, match="must encode to"):
        vlm_dynamic_inference.resolve_vlm_special_tokens(tokenizer, _args())


def test_resolve_vlm_special_tokens_unwraps_megatron_tokenizer():
    hf_tokenizer = _tokenizer()
    megatron_tokenizer = SimpleNamespace(_tokenizer=SimpleNamespace(tokenizer=hf_tokenizer))

    special_tokens = vlm_dynamic_inference.resolve_vlm_special_tokens(megatron_tokenizer, _args())

    assert special_tokens.image_token_id == 10


# --- Chat-endpoint image-token guard ----------------------------------------
#
# Image bytes and the prompt are assembled independently. If the chat template
# does not emit the image token resolved above, the vision embeddings have
# nowhere to land and the model answers about an image it never saw. The guard
# refuses that request instead.


def _guard(prompt_tokens, image_token_id=18, num_images=1):
    return chat_completions._image_token_mismatch_error(prompt_tokens, image_token_id, num_images)


def test_guard_allows_text_only_request():
    assert _guard([1, 2, 3], num_images=0) is None


def test_guard_allows_matching_token_count():
    assert _guard([1, 18, 2]) is None


def test_guard_allows_one_token_per_image():
    assert _guard([18, 1, 18], num_images=2) is None


def test_guard_rejects_missing_image_token():
    """The observed failure mode: image attached, no image token in the prompt."""
    error = _guard([1, 2, 3])

    assert error is not None
    assert "0 image token(s) for 1 attached image(s)" in error
    # Point the caller at the path the VLM pretraining workflow actually uses.
    assert "/v1/completions" in error


def test_guard_rejects_too_few_tokens_for_images():
    assert _guard([18], num_images=2) is not None


def test_guard_rejects_too_many_tokens_for_images():
    assert _guard([18, 18], num_images=1) is not None


def test_guard_rejects_image_request_to_text_only_model():
    """A model with no image token cannot serve an image request."""
    assert _guard([1, 2, 3], image_token_id=None) is not None
