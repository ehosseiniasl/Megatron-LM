# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Bitwise parity of MIMO first-stage language inputs with the previous full-sequence merge."""

from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import (
    MIMO_LANGUAGE_MODULE_KEY,
    ModuleLayout,
    ModuleStageInfo,
    RankRole,
)
from megatron.core.models.mimo.model.base import MimoModel, _global_token_index
from megatron.core.models.mimo.token_types import MimoTokenType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.experimental,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

_VOCAB = 16  # Tokens repeat often, so embedding gradients sum many rows per token.
_HIDDEN = 16
_VISION_ID = 1000  # Outside the vocabulary.
_AUDIO_ID = _VOCAB - 1  # Inside the vocabulary.
_SPECIAL_TOKEN_IDS = {"vision": _VISION_ID, "audio": _AUDIO_ID}


def _reference_inputs(model, modality_embeddings, batch, owns_mtp):
    """Previous path: embed text, scatter into a full [S, B, H] tensor, shard, SP-scatter."""
    input_ids, indices = batch["input_ids"], batch["modality_token_indices"]
    text_indices = (indices or {}).get("text")
    text = model.get_text_embeddings(
        input_ids, batch["position_ids"], model.special_token_ids, text_indices
    )
    combined = torch.zeros(*input_ids.shape, text.size(-1), dtype=text.dtype, device="cuda")
    for name, emb in {**modality_embeddings, "text": text}.items():
        if indices is not None:
            combined.view(-1, text.size(-1)).index_copy_(0, indices[name], emb.to(text.dtype))
            continue
        if name == "text":
            mask = torch.ones_like(input_ids, dtype=torch.bool)
            for token_id in model.special_token_ids.values():
                mask &= input_ids != token_id
        else:
            mask = input_ids == model.special_token_ids[name]
        combined.masked_scatter_(mask.unsqueeze(-1).expand_as(combined), emb.to(text.dtype))
    cp_batch = model._prepare_language_inputs(
        combined.transpose(0, 1).contiguous(),
        input_ids,
        batch["position_ids"],
        batch["labels"],
        batch["loss_mask"],
        batch["cu_seqlens"],
        batch["cu_seqlens_padded"],
        batch["max_seqlen"],
        owns_mtp,
        text_indices,
    )
    cfg = model.partition_adapter.cfg
    if cfg.seq_parallel:
        local = cp_batch.get_batch()
        local["decoder_input"] = tensor_parallel.scatter_to_sequence_parallel_region(
            local["decoder_input"], group=cfg.tp_group
        )
    return cp_batch


class _LanguageModel(torch.nn.Module):
    """Real vocab-parallel embedding with learned positions; returns its decoder input."""

    def __init__(self, config, max_sequence_length, tp_group):
        super().__init__()
        self.config = config
        self.embedding = LanguageModelEmbedding(
            config,
            _VOCAB,
            max_sequence_length,
            position_embedding_type="learned_absolute",
            scatter_to_sequence_parallel=False,
            tp_group=tp_group,
        )
        self.mtp_process = True
        self.last_inputs = None

    def forward(self, **kwargs):
        self.last_inputs = kwargs
        return kwargs["decoder_input"]


class _ModalityEncoder(torch.nn.Module):
    """Colocated modality submodule that returns caller-provided embeddings."""

    def forward(self, encoder_inputs):
        return encoder_inputs["embeddings"]


@pytest.fixture
def process_groups(request, set_env):
    """``request.param`` is ``(tp, cp, sequence_parallel)``."""
    tp_size, cp_size, _ = request.param
    if Utils.world_size % (tp_size * cp_size):
        pytest.skip(f"requires a world size divisible by TP={tp_size} x CP={cp_size}")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    model_parallel_cuda_manual_seed(1234)
    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        Utils.destroy_model_parallel()


def _make_model(
    pgs,
    sequence_parallel,
    seq_len,
    layout="contiguous",
    dtype=torch.float32,
    fp32_residual=False,
    deterministic=True,
    colocated=False,
    pad_documents=False,
):
    torch.manual_seed(2718)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN,
        num_attention_heads=8,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=pgs.tp.size(),
        sequence_parallel=sequence_parallel,
        context_parallel_size=pgs.cp.size(),
        linear_cp_layout=layout,
        attention_cp_layout="zigzag",
        params_dtype=dtype,
        use_cpu_initialization=True,
        deterministic_mode=deterministic,
        fp32_residual_connection=fp32_residual,
    )
    model = MimoModel(
        MimoModelConfig(
            language_model_spec=ModuleSpec(
                module=_LanguageModel,
                params={"config": config, "max_sequence_length": seq_len, "tp_group": pgs.tp},
            ),
            modality_submodules_spec={},
            special_token_ids=_SPECIAL_TOKEN_IDS,
        ),
        cp_group=pgs.cp,
        tp_group=pgs.tp,
        tp_cp_group=pgs.tp_cp,
    ).cuda()
    model = model.to(dtype)  # Also casts the position table, which is created in fp32.
    if pad_documents:
        # As for SSA: the partitioner pads each packed document and fills padding rows with 0.
        cfg = model.partition_adapter.cfg
        model.partition_adapter.cfg = replace(cfg, pad_packed_documents=True)
    if colocated:
        for name in _SPECIAL_TOKEN_IDS:
            model.modality_submodules[name] = _ModalityEncoder()
    else:
        model.role = RankRole(
            modules={MIMO_LANGUAGE_MODULE_KEY: ModuleStageInfo(True, True)},
            mode=ModuleLayout.NON_COLOCATED,
        )
    return model


def _make_batch(
    seq_len,
    batch_size,
    pattern="mixed",
    modalities=("vision",),
    index_mode="indices",
    packed=False,
    mrope=False,
    emb_dtype=torch.float32,
    negative_text_id=False,
):
    """Return ``(batch, modality_embeddings)``; row ``k`` belongs at ``indices[name][k]``.

    ``index_mode`` is "indices", "mask", or "stray_audio" (mask mode with audio tokens but no
    audio embeddings). ``packed`` is False, True, "padded", or "unaligned" (a document length
    that CP cannot split evenly).
    """
    generator = torch.Generator().manual_seed(7)
    input_ids = torch.randint(0, _AUDIO_ID, (batch_size, seq_len), generator=generator)
    owner = torch.zeros(batch_size, seq_len, dtype=torch.long)  # 0 text, 1 vision, 2 audio
    if pattern == "mixed":
        for b in range(batch_size):
            start = (3 + 5 * b) % (seq_len // 2)
            owner[b, start : start + seq_len // 4] = 1  # spans CP/SP shard boundaries
            if "audio" in modalities or index_mode == "stray_audio":
                owner[b, seq_len * 5 // 8 + b : seq_len * 3 // 4 + b] = 2
    elif pattern == "all":
        owner[:] = 1
    elif pattern != "empty":
        raise ValueError(pattern)
    input_ids[owner == 1] = _VISION_ID
    input_ids[owner == 2] = _AUDIO_ID
    if index_mode == "indices" and owner[0, -1] == 0:
        input_ids[0, -1] = _AUDIO_ID  # Text may share a special ID when indices are given.
    if negative_text_id:
        assert owner[0, 1] == 0
        input_ids[0, 1] = -1

    flat_owner = owner.flatten()
    indices = {"text": (flat_owner == 0).nonzero().squeeze(1)}
    for code, name in enumerate(("vision", "audio"), start=1):
        if name in modalities:
            # Producer order need not be sequence order.
            indices[name] = (flat_owner == code).nonzero().squeeze(1).flip(0)
    modality_embeddings = {
        name: torch.randn(indices[name].numel(), _HIDDEN, generator=generator).to(emb_dtype)
        for name in modalities
    }

    positions = torch.arange(seq_len).expand(batch_size, -1)
    if mrope:
        positions = torch.stack((positions, positions // 2, positions // 4))
    batch = {
        "input_ids": input_ids,
        "position_ids": positions,
        "labels": input_ids.roll(-1, dims=-1),
        "loss_mask": (owner == 0).float(),
        "modality_token_indices": indices if index_mode == "indices" else None,
        "cu_seqlens": None,
        "cu_seqlens_padded": None,
        "max_seqlen": None,
    }
    if packed:
        assert batch_size == 1
        first = seq_len // 4 - 3 if packed == "unaligned" else seq_len // 4
        padded = torch.tensor([[0, first, seq_len]], dtype=torch.int32)
        batch["cu_seqlens_padded"] = padded
        # "padded": each document ends two tokens before its padded boundary.
        batch["cu_seqlens"] = padded - 2 * (padded > 0).int() if packed == "padded" else padded.clone()
        batch["max_seqlen"] = padded.diff().max().view(1)
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if batch["modality_token_indices"] is not None:
        batch["modality_token_indices"] = {
            k: v.cuda() for k, v in batch["modality_token_indices"].items()
        }
    return batch, {name: emb.cuda() for name, emb in modality_embeddings.items()}


def _mismatch(name, actual, expected):
    """Describe any bitwise difference, including signs of zeros, or return None."""
    if actual is None or expected is None:
        return None if actual is expected else f"{name}: {actual!r} vs {expected!r}"
    if not isinstance(expected, torch.Tensor):
        return None
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return f"{name}: {actual.dtype}{tuple(actual.shape)} vs {expected.dtype}{tuple(expected.shape)}"
    if not torch.equal(actual, expected):
        diff = (actual.double() - expected.double()).abs().max().item()
        return f"{name}: values differ, max abs diff {diff}"
    if actual.is_floating_point() and not torch.equal(actual.signbit(), expected.signbit()):
        return f"{name}: signs of zeros differ"
    return None


def _assert_no_mismatches(errors):
    """Agree on failure across ranks before asserting, so no rank hangs in a collective."""
    errors = [e for e in errors if e]
    failed = torch.tensor([int(bool(errors))], device="cuda")
    dist.all_reduce(failed)
    assert not errors, f"rank {dist.get_rank()}: " + "; ".join(errors)
    assert failed.item() == 0, "another rank reported a mismatch"


def _prepare(model, modality_embeddings, batch, owns_mtp=True):
    return model._prepare_multimodal_language_inputs(
        modality_embeddings,
        batch["input_ids"],
        batch["position_ids"],
        batch["labels"],
        batch["loss_mask"],
        batch["cu_seqlens"],
        batch["cu_seqlens_padded"],
        batch["max_seqlen"],
        owns_mtp,
        batch["modality_token_indices"],
    )


def _compare(model, batch, modality_embeddings, owns_mtp=True, word_embedding_rtol=None):
    """Compare sharded inputs and gradients with the reference; all bitwise unless noted.

    ``word_embedding_rtol`` relaxes only the word-embedding gradient, for default mode.
    """
    reference_embs = {k: v.clone().requires_grad_() for k, v in modality_embeddings.items()}
    expected = _reference_inputs(model, reference_embs, batch, owns_mtp)
    model_embs = {k: v.clone().requires_grad_() for k, v in modality_embeddings.items()}
    actual = _prepare(model, dict(model_embs), batch, owns_mtp)

    errors = []
    assert actual.batches_by_layout.keys() == expected.batches_by_layout.keys()
    for layout, expected_batch in expected.batches_by_layout.items():
        actual_batch = actual.get_batch(layout)
        assert actual_batch.keys() == expected_batch.keys()
        errors += [
            _mismatch(f"{layout}.{k}", actual_batch[k], v) for k, v in expected_batch.items()
        ]

    output = actual.get_batch()["decoder_input"]
    generator = torch.Generator(device="cuda").manual_seed(1000 + dist.get_rank())
    grad = torch.randn(output.shape, device="cuda", generator=generator).to(output.dtype)
    names = [name for name, _ in model.named_parameters()] + list(modality_embeddings)
    parameters = tuple(model.parameters())
    actual_grads = torch.autograd.grad(output, (*parameters, *model_embs.values()), grad)
    expected_grads = torch.autograd.grad(
        expected.get_batch()["decoder_input"], (*parameters, *reference_embs.values()), grad
    )
    for name, a, e in zip(names, actual_grads, expected_grads):
        if word_embedding_rtol is not None and name.endswith("word_embeddings.weight"):
            scale = e.abs().max().item()
            if not torch.allclose(a, e, rtol=word_embedding_rtol, atol=1e-6 * scale):
                errors.append(f"grad {name}: beyond summation-order rounding")
            continue
        errors.append(_mismatch(f"grad {name}", a, e))
    _assert_no_mismatches(errors)


# (tp, cp, sequence_parallel): no parallelism, CP only, SP only, SP with CP, TP=4 with CP.
_PARALLEL = [(1, 1, False), (1, 8, False), (2, 1, True), (2, 2, True), (4, 2, True)]

_CASES = {
    "zigzag_two_modalities_mrope_bf16_modalities": dict(
        layout="zigzag", modalities=("vision", "audio"), mrope=True, emb_dtype=torch.bfloat16
    ),
    "packed_with_padding": dict(layout="zigzag", packed="padded"),
    "packed_document_padding": dict(layout="zigzag", packed="padded", pad_documents=True),
    "packed_unaligned_document_padding": dict(
        layout="zigzag", packed="unaligned", pad_documents=True
    ),
    "packed_bf16_fp32_residual": dict(
        packed=True, dtype=torch.bfloat16, emb_dtype=torch.bfloat16, fp32_residual=True
    ),
    "mask_mode_stray_special_tokens": dict(index_mode="stray_audio"),
    "empty_modality_negative_text_id_no_mtp": dict(
        pattern="empty", negative_text_id=True, owns_mtp=False
    ),
    "all_modality": dict(layout="zigzag", pattern="all"),
}


@pytest.mark.parametrize("process_groups", _PARALLEL, indirect=True)
@pytest.mark.parametrize("case", list(_CASES))
def test_first_stage_inputs_match_reference_bitwise(process_groups, request, case):
    """Decoder input, sharded metadata, and all gradients match the reference exactly."""
    sequence_parallel = request.node.callspec.params["process_groups"][2]
    c = dict(_CASES[case])
    packed = c.pop("packed", False)
    owns_mtp = c.pop("owns_mtp", True)
    model = _make_model(
        process_groups,
        sequence_parallel,
        64,
        c.pop("layout", "contiguous"),
        c.pop("dtype", torch.float32),
        c.pop("fp32_residual", False),
        pad_documents=c.pop("pad_documents", False),
    )
    batch, modality_embeddings = _make_batch(64, 1 if packed else 2, packed=packed, **c)
    _compare(model, batch, modality_embeddings, owns_mtp)


@pytest.mark.parametrize("process_groups", [(1, 2, False), (2, 2, True)], indirect=True)
@pytest.mark.parametrize("deterministic", [True, False])
def test_long_sequence_gradients(process_groups, request, deterministic):
    """Thousands of repeats per token: bitwise under deterministic_mode.

    Without it, F.embedding's backward sums repeated tokens in chunks that depend on the lookup
    list, so the word-embedding gradient may differ by rounding; everything else stays exact.
    """
    sequence_parallel = request.node.callspec.params["process_groups"][2]
    model = _make_model(
        process_groups, sequence_parallel, 8192, "zigzag", deterministic=deterministic
    )
    batch, modality_embeddings = _make_batch(8192, 1, modalities=("vision", "audio"))
    _compare(model, batch, modality_embeddings, word_embedding_rtol=None if deterministic else 1e-5)


@pytest.mark.parametrize("process_groups", [(1, 2, False), (2, 2, True)], indirect=True)
@pytest.mark.parametrize("colocated", [False, True])
def test_forward_passes_reference_inputs(process_groups, request, colocated):
    """Both first-stage forwards hand the reference inputs to the language model."""
    sequence_parallel = request.node.callspec.params["process_groups"][2]
    model = _make_model(process_groups, sequence_parallel, 64, "zigzag", colocated=colocated)
    batch, modality_embeddings = _make_batch(64, 2, modalities=("vision", "audio"), mrope=True)
    expected = _reference_inputs(model, modality_embeddings, batch, owns_mtp=True).get_batch()
    if colocated:
        inputs = {"modality_inputs": {k: {"embeddings": v} for k, v in modality_embeddings.items()}}
    else:
        model.set_input_tensor(dict(modality_embeddings))
        inputs = {}
    output, loss_mask = model(**batch, attention_mask=None, **inputs)
    lm_inputs = model.language_model.last_inputs
    errors = [
        _mismatch("output", output, expected["decoder_input"]),
        _mismatch("returned loss_mask", loss_mask, expected["loss_mask"]),
        _mismatch("input_ids", lm_inputs["input_ids"], expected["tokens"]),
    ]
    for key in ("position_ids", "labels", "loss_mask", "mtp_input_mask"):
        errors.append(_mismatch(key, lm_inputs[key], expected[key]))
    _assert_no_mismatches(errors)


@pytest.mark.parametrize("process_groups", _PARALLEL, indirect=True)
@pytest.mark.parametrize(
    "packed, pad_documents",
    [(False, False), ("padded", False), ("padded", True), ("unaligned", True)],
)
def test_residual_token_types_follow_decoder_input(process_groups, request, packed, pad_documents):
    """Residual token types follow the decoder-input rows; padding rows are PAD."""
    _, cp_size, sequence_parallel = request.node.callspec.params["process_groups"]
    model = _make_model(
        process_groups, sequence_parallel, 64, "zigzag", pad_documents=pad_documents
    )
    batch, _ = _make_batch(64, 1 if packed else 2, packed=packed)
    packed_metadata = {k: batch[k] for k in ("cu_seqlens", "cu_seqlens_padded", "max_seqlen")}
    input_ids = batch["input_ids"]
    generator = torch.Generator(device="cuda").manual_seed(3)
    # Types below PAD, so PAD marks only padding rows.
    token_types = torch.randint(
        0, MimoTokenType.PAD, input_ids.shape, device="cuda", generator=generator
    )
    # Embeddings hold each token's index plus one; the partitioner fills padding rows with 0.
    index = torch.arange(1, input_ids.numel() + 1, device="cuda", dtype=torch.float64)
    rows = model.partition_adapter.partition(
        index.view_as(input_ids).t().unsqueeze(-1), None, None, None, None, None, **packed_metadata
    ).get_batch()["decoder_input"]
    cfg = model.partition_adapter.cfg
    if cfg.seq_parallel:
        rows = tensor_parallel.scatter_to_sequence_parallel_region(rows, group=cfg.tp_group)
    rows = rows.squeeze(-1).long() - 1
    expected = token_types.reshape(-1)[rows.clamp_min(0)].masked_fill(rows < 0, MimoTokenType.PAD)
    if pad_documents and cp_size > 1:
        has_padding = (rows < 0).any().int().view(1)
        dist.all_reduce(has_padding)  # Some CP ranks hold no padding rows.
        assert has_padding.item(), "expected padding rows"
    actual = model.shard_language_residual_token_types(token_types, **packed_metadata)
    _assert_no_mismatches([_mismatch("token types", actual, expected)])


@pytest.mark.parametrize("process_groups", [(1, 2, False)], indirect=True)
def test_non_last_stage_wraps_output(process_groups):
    """A first, non-last language stage keys its output for the next pipeline stage."""
    model = _make_model(process_groups, False, 64)
    model.role = RankRole(
        modules={MIMO_LANGUAGE_MODULE_KEY: ModuleStageInfo(True, False)},
        mode=ModuleLayout.NON_COLOCATED,
    )
    batch, modality_embeddings = _make_batch(64, 1)
    model.set_input_tensor(dict(modality_embeddings))
    output, loss_mask = model(**batch, attention_mask=None)
    assert output.keys() == {MIMO_LANGUAGE_MODULE_KEY}
    assert output[MIMO_LANGUAGE_MODULE_KEY].shape == (32, 1, _HIDDEN)
    assert loss_mask.shape == (1, 32)


@pytest.mark.parametrize("process_groups", [(1, 2, False)], indirect=True)
def test_embedding_dropout_drops_text_only(process_groups):
    """Dropout applies to local text embeddings; modality rows pass through unchanged."""
    model = _make_model(process_groups, False, 64)
    batch, modality_embeddings = _make_batch(64, 2, modalities=("vision", "audio"))
    clean = _prepare(model, dict(modality_embeddings), batch).get_batch()["decoder_input"]
    model.language_model.embedding.embedding_dropout.p = 0.5
    dropped = _prepare(model, dict(modality_embeddings), batch).get_batch()["decoder_input"]
    input_ids = batch["input_ids"]
    local_index = model.partition_adapter.partition(
        None, None, None, None, None, None, global_token_index=_global_token_index(input_ids)
    ).get_batch()["global_token_index"]
    is_text = torch.zeros(input_ids.numel(), dtype=torch.bool, device="cuda")
    is_text[batch["modality_token_indices"]["text"]] = True
    is_text = is_text[local_index.t()].unsqueeze(-1)
    kept = (dropped == clean * 2) | (dropped == 0)  # p=0.5 scales kept values by exactly 2
    assert bool(torch.where(is_text, kept, dropped == clean).all())
    assert bool((dropped == 0)[is_text.expand_as(dropped)].any())


@pytest.mark.parametrize("process_groups", [(1, 1, False)], indirect=True)
def test_invalid_token_metadata_raises(process_groups):
    """Errors match the previous validation for precomputed and special-token metadata."""
    model = _make_model(process_groups, False, 64)
    batch, modality_embeddings = _make_batch(64, 1)
    indices = batch["modality_token_indices"]
    vision = modality_embeddings["vision"]

    def prepare(embs, token_indices):
        return _prepare(model, embs, {**batch, "modality_token_indices": token_indices})

    with pytest.raises(ValueError, match="same modalities"):
        prepare(modality_embeddings, {"vision": indices["vision"]})
    with pytest.raises(ValueError, match="does not match"):
        prepare({"vision": vision[1:]}, indices)
    with pytest.raises(ValueError, match="must cover"):
        prepare(modality_embeddings, {**indices, "text": indices["text"][1:]})
    with pytest.raises(ValueError, match="No special token ID"):
        prepare({"video": vision}, None)
    with pytest.raises(ValueError, match="does not match"):
        prepare({"vision": vision[1:]}, None)
