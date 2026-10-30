# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the Nemotron6-MoE VLM model provider.

Covers the post-parse derived knobs and the config parity gate: the from-args
language config must reproduce the reference Nemotron architecture
field-for-field, except the two fields that
``core_transformer_config_from_args`` correctly supplies (documented below).
"""

import argparse
import sys
from types import SimpleNamespace

import pytest
import torch

from examples.mimo.model_providers.nemotron_moe_vlm import (
    NEMOTRON_MODEL_PROVIDER,
    _nemotron_bridge_recv_shape,
    add_model_provider_args,
    build_nemotron_communicator,
)
from examples.mimo.model_providers.radio_encoder import RADIO_ENCODER_MODULE_NAME
from examples.mimo.training.args import add_hetero_grid_args
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

# (num_layers, hybrid_layer_pattern) is the ONLY architecture delta between the
# 20L and 54L Nemotron presets; every other field is shared. num_layers follows
# the pattern length (get_hybrid_total_layer_count): 20 and 54 layer-tokens.
_PRESET_20L = (20, "MEMEM*EMEMEM*EMEMEM*")
_PRESET_54L = (54, "MEMEM*EMEM*EMEM*EMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEME")

# Shared Nemotron6-MoE architecture (the reference fixture): the exact values the
# run script passes as stock CLI flags.
_NEMOTRON_ARCH = dict(
    hidden_size=2688,
    num_attention_heads=32,
    num_query_groups=8,
    ffn_hidden_size=1856,
    kv_channels=128,
    num_moe_experts=128,
    moe_router_topk=6,
    moe_grouped_gemm=True,
    moe_ffn_hidden_size=1856,
    moe_router_score_function="sigmoid",
    moe_router_topk_scaling_factor=2.5,
    moe_router_enable_expert_bias=True,
    moe_router_dtype="fp32",
    moe_router_load_balancing_type="seq_aux_loss",
    moe_router_fusion=True,
    moe_aux_loss_coeff=1.0e-4,
    moe_shared_expert_intermediate_size=3712,
    moe_shared_expert_overlap=True,
    moe_token_dispatcher_type="alltoall",
    moe_flex_dispatcher_backend="deepep",
    moe_permute_fusion=True,
    use_fused_weighted_squared_relu=True,
    mamba_num_heads=64,
    mamba_head_dim=64,
    mamba_num_groups=8,
    mamba_state_dim=128,
    linear_conv_kernel_dim=4,
    normalization="RMSNorm",
    init_method_std=0.0173,
    add_bias_linear=False,
    gated_linear_unit=False,
    calculate_per_token_loss=True,
    cross_entropy_loss_fusion=True,
)


def _parse(argv):
    """Parse provider args then backfill stock-arg defaults (simulating stock parse)."""
    parser = argparse.ArgumentParser()
    add_model_provider_args(parser)
    args = parser.parse_args(argv)
    for key, value in dict(hidden_size=None, num_layers=None, fp16=False).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def test_dynamic_resolution_defaults_off():
    # --dynamic-resolution is a radio_encoder flag (store_true), registered via
    # add_radio_encoder_args; default off, passed explicitly to enable.
    args = _parse(["--model-provider", NEMOTRON_MODEL_PROVIDER])
    assert args.dynamic_resolution is False
    on = _parse(["--model-provider", NEMOTRON_MODEL_PROVIDER, "--dynamic-resolution"])
    assert on.dynamic_resolution is True


def test_freeze_flags_drive_tower_freezing():
    # The freeze interface is the --freeze-* flags.
    args = _parse(["--model-provider", NEMOTRON_MODEL_PROVIDER, "--freeze-vit", "--freeze-lm"])
    assert args.freeze_vit is True
    assert args.freeze_lm is True
    assert args.freeze_projection is False


def test_bridge_receive_shape_uses_local_image_token_count():
    batch = {"input_ids": torch.tensor([[42, 7, 42], [3, 42, 5]])}

    assert _nemotron_bridge_recv_shape(batch, image_token_id=42, hidden_size=2688) == (3, 2688)


@pytest.mark.parametrize("enabled", [False, True], ids=["legacy", "receiver_derived"])
def test_build_communicator_wires_bridge_receive_shape(monkeypatch, enabled):
    import examples.mimo.model_providers.nemotron_moe_vlm as provider

    captured = {}
    communicator = object()
    language_grid = object()
    topology = SimpleNamespace(
        grids={RADIO_ENCODER_MODULE_NAME: object(), MIMO_LANGUAGE_MODULE_KEY: language_grid},
        timeout=None,
    )
    args = SimpleNamespace(
        mimo_bridge_skip_shape_exchange=enabled, image_token_id=42, hidden_size=2688
    )
    language_config = SimpleNamespace(params_dtype=torch.bfloat16, pipeline_dtype=torch.float32)

    monkeypatch.setattr(
        provider,
        "language_model_spec",
        lambda args, pg_collection, grid: SimpleNamespace(params={"config": language_config}),
    )

    def capture_communicator(*args, **kwargs):
        captured.update(kwargs)
        return communicator

    monkeypatch.setattr(provider, "MultiModulePipelineCommunicator", capture_communicator)

    assert build_nemotron_communicator(args, topology) is communicator
    assert captured["bridge_comm_dtypes"] == {RADIO_ENCODER_MODULE_NAME: torch.bfloat16}
    shape_fns = captured["bridge_recv_shape_fns"]
    if enabled:
        assert shape_fns[RADIO_ENCODER_MODULE_NAME]({"input_ids": torch.tensor([[42, 7, 42]])}) == (
            2,
            2688,
        )
    else:
        assert shape_fns is None


# --- Config parity gate (requires torch; runs in CI) ----------------------

pytest.importorskip("torch")


def _build_argv(num_layers, hybrid_pattern):
    """Full stock + provider CLI for the Nemotron preset (mirrors the run script)."""
    return [
        "--model-provider",
        NEMOTRON_MODEL_PROVIDER,
        "--pixel-shuffle",
        "--disable-vision-class-token",
        "--num-layers",
        str(num_layers),
        "--hybrid-layer-pattern",
        hybrid_pattern,
        "--hidden-size",
        "2688",
        "--num-attention-heads",
        "32",
        "--group-query-attention",
        "--num-query-groups",
        "8",
        "--ffn-hidden-size",
        "1856",
        "--kv-channels",
        "128",
        "--squared-relu",
        "--disable-bias-linear",
        "--normalization",
        "RMSNorm",
        "--init-method-std",
        "0.0173",
        "--num-experts",
        "128",
        "--moe-router-topk",
        "6",
        "--moe-grouped-gemm",
        "--moe-ffn-hidden-size",
        "1856",
        "--moe-router-score-function",
        "sigmoid",
        "--moe-router-topk-scaling-factor",
        "2.5",
        "--moe-router-enable-expert-bias",
        "--moe-router-dtype",
        "fp32",
        "--moe-router-load-balancing-type",
        "seq_aux_loss",
        "--moe-router-fusion",
        "--moe-aux-loss-coeff",
        "1e-4",
        "--moe-shared-expert-intermediate-size",
        "3712",
        "--moe-shared-expert-overlap",
        "--moe-token-dispatcher-type",
        "alltoall",
        "--moe-flex-dispatcher-backend",
        "deepep",
        "--moe-permute-fusion",
        "--use-fused-weighted-squared-relu",
        "--mamba-num-heads",
        "64",
        "--mamba-head-dim",
        "64",
        "--mamba-num-groups",
        "8",
        "--mamba-state-dim",
        "128",
        "--linear-conv-kernel-dim",
        "4",
        "--position-embedding-type",
        "none",
        "--attention-backend",
        "flash",
        "--calculate-per-token-loss",
        "--cross-entropy-loss-fusion",
        "--seq-length",
        "8192",
        "--max-position-embeddings",
        "8192",
        "--micro-batch-size",
        "1",
        "--vocab-size",
        "131072",
        "--tokenizer-type",
        "NullTokenizer",
        "--bf16",
    ]


def _parse_validate(argv):
    """Build args via the production pipeline so validate_args-derived fields
    (params_dtype, padded_vocab_size, ...) resolve exactly as in a real run.

    Mirrors examples/mimo/pretrain_mimo.py: parse_args -> validate_args. Runs at
    world_size=1, tp=pp=cp=1 so validate_args' divisibility checks pass with no
    distributed/mpu init.
    """
    from megatron.training.arguments import parse_args, validate_args

    saved = sys.argv
    sys.argv = ["pytest"] + argv
    try:
        args = parse_args(
            lambda parser: add_hetero_grid_args(add_model_provider_args(parser)),
            ignore_unknown_args=True,
        )
    finally:
        sys.argv = saved
    validate_args(args)
    return args


def _without_flag(argv, flag):
    return [arg for arg in argv if arg != flag]


@pytest.mark.parametrize("num_layers,hybrid_pattern", [_PRESET_20L, _PRESET_54L])
def test_language_config_parity(num_layers, hybrid_pattern):
    """from-args language config == reference arch, modulo 2 documented fields.

    ``deallocate_pipeline_outputs`` and ``inference_sampling_seed`` are supplied
    by ``core_transformer_config_from_args`` and intentionally differ from a raw
    hardcoded config: deallocate=True is the stock-correct value (inert at PP=1,
    matches pretrain_gpt/vlm) and inference_sampling_seed tracks --seed. We assert
    those took the from-args values and exclude them from the field compare.
    """
    from examples.mimo.model_providers.nemotron_moe_vlm import nemotron_language_config

    args = _parse_validate(_build_argv(num_layers, hybrid_pattern))

    config = nemotron_language_config(args, tp_size=1, pp_size=1, ep_size=1, expt_tp_size=1)

    assert config.num_layers == num_layers
    assert config.is_hybrid_model is True
    for field, expected in _NEMOTRON_ARCH.items():
        assert getattr(config, field) == expected, field

    # The two documented from-args fields.
    assert config.deallocate_pipeline_outputs is True
    assert config.inference_sampling_seed == args.seed

    # Code-only overrides. (seq_length / max_position_embeddings are NOT
    # TransformerConfig fields; the seq-length contract is covered by
    # test_language_model_spec_builds_mamba via max_sequence_length.)
    assert config.position_embedding_type == "none"
    assert config.tensor_model_parallel_size == 1


@pytest.mark.parametrize("disable_mtp", [False, True])
def test_language_config_preserves_mtp_shortcut_option(disable_mtp, monkeypatch):
    """Preserve the flag with Ultra's repeated MTP and recompute/offload configuration."""
    from examples.mimo.model_providers.nemotron_moe_vlm import nemotron_language_config

    monkeypatch.setenv("NVTE_CPU_OFFLOAD_V1", "1")
    recompute_modules = [
        "residual_stream",
        "moe_act",
        "gdp_in_proj",
        "gdp_qkv",
        "moe_experts",
        "shared_experts",
        "mlp",
        "core_attn_output",
        "qkv_proj",
    ]
    argv = _without_flag(_build_argv(2, "*E/*E/*E"), "--moe-shared-expert-overlap")
    argv += [
        "--moe-shortcut-connection",
        "--moe-shortcut-parallel",
        "--moe-latent-size",
        "512",
        "--moe-use-norm-before-up-proj",
        "--activation-func-tanh-clamp-scale",
        "16",
        "--mtp-use-repeated-layer",
        "--mtp-hsm",
        "--mtp-loss-early-backward",
        "--mtp-normalize-loss-by-main-tokens",
        "--wide-residual",
        "3",
        "--recompute-granularity",
        "selective",
        "--recompute-modules",
        *recompute_modules,
        "--residual-stream-recompute-num-layers",
        "16",
        "--use-transformer-engine-op-fuser",
        "--fine-grained-activation-offloading",
        "--offload-modules",
        "fused_group_mlp",
        "--activation-offload-fraction",
        "0.6",
    ]
    if disable_mtp:
        argv.append("--moe-shortcut-disable-mtp")
    args = _parse_validate(argv)
    config = nemotron_language_config(args, tp_size=1, pp_size=1, ep_size=1, expt_tp_size=1)

    assert args.moe_shortcut_disable_mtp is disable_mtp
    assert config.moe_shortcut_disable_mtp is disable_mtp
    assert config.moe_shortcut_connection
    assert config.moe_shortcut_parallel
    assert config.moe_latent_size == 512
    assert config.moe_use_norm_before_up_proj
    assert config.mtp_num_layers == 2
    assert config.mtp_use_repeated_layer
    assert config.mtp_hsm
    assert config.mtp_loss_early_backward
    assert config.mtp_normalize_loss_by_main_tokens
    assert config.wide_residual.num_streams == 3
    assert config.recompute_modules == recompute_modules
    assert config.use_transformer_engine_op_fuser
    assert config.fine_grained_activation_offloading
    assert config.offload_modules == ["fused_group_mlp"]
    assert config.activation_offload_fraction == 0.6


def test_configs_follow_stock_dtype_args():
    """The provider does not add precision flags; tower configs inherit stock dtype args."""
    import torch

    from examples.mimo.model_providers.nemotron_moe_vlm import (
        nemotron_language_config,
        nemotron_projection_config,
        vision_submodules_spec,
    )

    bf16_args = _parse_validate(_build_argv(*_PRESET_20L))
    bf16_configs = [
        nemotron_language_config(bf16_args, tp_size=1, pp_size=1, ep_size=1, expt_tp_size=1),
        nemotron_projection_config(bf16_args, tp_size=1, projection_input_size=5120),
        vision_submodules_spec(bf16_args, pg_collection=None, encoder_grid=None)
        .submodules["encoders"][RADIO_ENCODER_MODULE_NAME]
        .params["transformer_config"],
    ]
    for config in bf16_configs:
        assert config.params_dtype is torch.bfloat16
        assert config.pipeline_dtype is torch.bfloat16
        assert config.bf16 is True

    fp32_args = _parse_validate(_without_flag(_build_argv(*_PRESET_20L), "--bf16"))
    fp32_configs = [
        nemotron_language_config(fp32_args, tp_size=1, pp_size=1, ep_size=1, expt_tp_size=1),
        nemotron_projection_config(fp32_args, tp_size=1, projection_input_size=5120),
    ]
    for config in fp32_configs:
        assert config.params_dtype is torch.float32
        assert config.pipeline_dtype is torch.float32
        assert config.bf16 is False


def test_modality_configs_do_not_inherit_language_fp32_residuals():
    """FP32 language residuals must not change the frozen tower or projector math."""
    from examples.mimo.model_providers.nemotron_mistral_vit import (
        mistral_vit_projection_config,
        mistral_vit_vision_config,
    )
    from examples.mimo.model_providers.nemotron_moe_vlm import (
        nemotron_language_config,
        nemotron_projection_config,
        vision_submodules_spec,
    )

    args = _parse_validate(_build_argv(*_PRESET_20L) + ["--fp32-residual-connection"])
    language_config = nemotron_language_config(
        args, tp_size=1, pp_size=1, ep_size=1, expt_tp_size=1
    )
    modality_configs = [
        nemotron_projection_config(args, tp_size=1, projection_input_size=5120),
        vision_submodules_spec(args, pg_collection=None, encoder_grid=None)
        .submodules["encoders"][RADIO_ENCODER_MODULE_NAME]
        .params["transformer_config"],
        mistral_vit_projection_config(args, tp_size=1),
        mistral_vit_vision_config(args, tp_size=1, pp_size=1),
    ]

    assert language_config.fp32_residual_connection is True
    assert language_config.pipeline_dtype is torch.float32
    for config in modality_configs:
        assert config.fp32_residual_connection is False
        assert config.params_dtype is torch.bfloat16
        assert config.pipeline_dtype is torch.bfloat16


def test_modality_configs_do_not_inherit_language_activation_tanh_clamp():
    """The decoder clamp must not change pretrained vision-tower or projector activations."""
    from examples.mimo.model_providers.nemotron_mistral_vit import (
        mistral_vit_projection_config,
        mistral_vit_vision_config,
    )
    from examples.mimo.model_providers.nemotron_moe_vlm import (
        nemotron_language_config,
        nemotron_projection_config,
        vision_submodules_spec,
    )

    args = _parse_validate(_build_argv(*_PRESET_20L) + ["--activation-func-tanh-clamp-scale", "16"])
    language_config = nemotron_language_config(
        args, tp_size=1, pp_size=1, ep_size=1, expt_tp_size=1
    )
    modality_configs = [
        nemotron_projection_config(args, tp_size=1, projection_input_size=5120),
        vision_submodules_spec(args, pg_collection=None, encoder_grid=None)
        .submodules["encoders"][RADIO_ENCODER_MODULE_NAME]
        .params["transformer_config"],
        mistral_vit_projection_config(args, tp_size=1),
        mistral_vit_vision_config(args, tp_size=1, pp_size=1),
    ]

    assert language_config.activation_func_tanh_clamp_scale == 16.0
    for config in modality_configs:
        assert config.activation_func_tanh_clamp_scale is None
    assert modality_configs[0].gated_linear_unit is False


def test_dense_tower_config_drops_language_hybrid_settings():
    """Vision and projector configs must not inherit decoder-only hybrid architectures."""
    from types import SimpleNamespace

    from examples.mimo.model_providers.radio_encoder import _make_dense_non_hybrid

    config = SimpleNamespace(
        activation_func_tanh_clamp_scale=16.0,
        num_moe_experts=128,
        moe_ffn_hidden_size=1856,
        moe_shared_expert_intermediate_size=3712,
        moe_grouped_gemm=True,
        moe_router_fusion=True,
        moe_permute_fusion=True,
        moe_shared_expert_overlap=True,
        moe_shortcut_connection=True,
        moe_shortcut_parallel=True,
        moe_shortcut_post_norm=True,
        fp32_residual_connection=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.float32,
        is_hybrid_model=True,
        use_fused_weighted_squared_relu=True,
        wide_residual=object(),
        residual_stream_recompute_num_layers=4,
        recompute_modules=["moe_act", "residual_stream", "shortcut_pre_mlp_layernorm"],
        offload_modules=["core_attn", "shortcut_post_norm"],
    )

    _make_dense_non_hybrid(config)

    assert config.activation_func_tanh_clamp_scale is None
    assert config.wide_residual is None
    assert config.moe_shortcut_connection is False
    assert config.moe_shortcut_parallel is False
    assert config.moe_shortcut_post_norm is False
    assert config.fp32_residual_connection is False
    assert config.pipeline_dtype is torch.bfloat16
    assert config.residual_stream_recompute_num_layers is None
    assert config.recompute_modules == ["moe_act"]
    assert config.offload_modules == ["core_attn"]


def test_language_model_spec_builds_mamba():
    """language_model_spec returns a MambaModel spec carrying the preset config."""
    from examples.mimo.model_providers.nemotron_moe_vlm import language_model_spec
    from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
    from megatron.core.models.mamba.mamba_model import MambaModel

    args = _parse_validate(_build_argv(*_PRESET_20L))
    args.mimo_llm_ep = 2
    args.mimo_llm_expt_tp = 2
    args.expert_tensor_parallel_num_weight_shards = args.mimo_llm_expt_tp
    spec = language_model_spec(args, pg_collection=None, llm_grid=None)
    assert spec.module is MambaModel
    assert spec.params["mamba_stack_spec"] is mamba_stack_spec
    assert spec.params["config"].num_layers == 20
    assert spec.params["config"].expert_model_parallel_size == 2
    assert spec.params["config"].expert_tensor_parallel_size == 2
    assert spec.params["max_sequence_length"] == args.seq_length
    assert spec.params["logit_dtype"] is None


@pytest.mark.parametrize(
    ("cli_dtype", "expected_dtype"), [("bf16", torch.bfloat16), ("fp32", torch.float32)]
)
def test_language_model_spec_propagates_logit_dtype(cli_dtype, expected_dtype):
    """The requested output-logit dtype reaches the MIMO language model."""
    from examples.mimo.model_providers.nemotron_moe_vlm import language_model_spec

    args = _parse_validate(_build_argv(*_PRESET_20L) + ["--output-logit-dtype", cli_dtype])
    spec = language_model_spec(args, pg_collection=None, llm_grid=None)

    assert spec.params["logit_dtype"] is expected_dtype


def test_language_model_spec_honors_custom_stack_spec():
    from examples.mimo.model_providers.nemotron_moe_vlm import language_model_spec
    from megatron.core.models.hybrid.hybrid_layer_specs import gated_delta_product_stack_spec

    args = _parse_validate(
        _build_argv(*_PRESET_20L)
        + [
            "--spec",
            "megatron.core.models.hybrid.hybrid_layer_specs",
            "gated_delta_product_stack_spec",
        ]
    )
    spec = language_model_spec(args, pg_collection=None, llm_grid=None)

    assert spec.params["mamba_stack_spec"] is gated_delta_product_stack_spec


def test_vision_submodules_spec_wires_radio_encoder():
    """vision_submodules_spec wires the RADIO encoder + affine projector, and the
    preset's pixel-shuffle / class-token-drop knobs reach the wrapper params."""
    from examples.mimo.model_providers.nemotron_moe_vlm import vision_submodules_spec
    from examples.mimo.model_providers.radio_encoder import RADIOEncoderWrapper

    args = _parse_validate(_build_argv(*_PRESET_20L))
    spec = vision_submodules_spec(args, pg_collection=None, encoder_grid=None)

    encoder = spec.submodules["encoders"][RADIO_ENCODER_MODULE_NAME]
    assert encoder.module is RADIOEncoderWrapper
    assert encoder.params["apply_pixel_shuffle"] is True
    assert encoder.params["drop_class_token"] is True

    projection = spec.submodules["input_projections"][0]
    assert projection.params["projector_type"] == "affine"
    assert projection.params["input_size"] == encoder.params["transformer_config"].hidden_size * 4
    assert projection.params["config"].ffn_hidden_size == projection.params["input_size"] * 4


@pytest.mark.parametrize(
    "pixel_shuffle,expected_projection_input_size", [(True, 5120), (False, 1280)]
)
def test_projection_input_size_tracks_pixel_shuffle(pixel_shuffle, expected_projection_input_size):
    """The projector input width follows the encoder output width."""
    from examples.mimo.model_providers.nemotron_moe_vlm import vision_submodules_spec

    argv = _build_argv(*_PRESET_20L)
    if not pixel_shuffle:
        argv = _without_flag(argv, "--pixel-shuffle")
    args = _parse_validate(argv)
    spec = vision_submodules_spec(args, pg_collection=None, encoder_grid=None)

    encoder = spec.submodules["encoders"][RADIO_ENCODER_MODULE_NAME]
    projection = spec.submodules["input_projections"][0]

    assert encoder.params["apply_pixel_shuffle"] is pixel_shuffle
    assert projection.params["input_size"] == expected_projection_input_size
    assert projection.params["config"].ffn_hidden_size == 4 * expected_projection_input_size


def test_language_rank_placement_uses_language_parallelism():
    from examples.mimo.model_providers import resolve_provider
    from examples.mimo.model_providers.nemotron_moe_vlm import (
        language_input_projection_spec,
        nemotron_language_config,
        vision_submodules_spec,
    )
    from megatron.core.transformer.spec_utils import ModuleSpec

    args = _parse_validate(_build_argv(*_PRESET_20L))
    args.mimo_run_input_projections_on_llm_ranks = True
    encoder_spec = vision_submodules_spec(args, pg_collection=None, encoder_grid=None)

    assert encoder_spec.submodules["input_projections"] == []

    args.tensor_parallel_num_weight_shards = 4
    language_config = nemotron_language_config(
        args, tp_size=2, pp_size=1, ep_size=1, expt_tp_size=1
    )
    assert language_config.gtp_remat_num_sequence_shards == 1
    language_spec = ModuleSpec(module=object, params={"config": language_config})
    projection = language_input_projection_spec(args, None, None, language_spec)

    assert projection.params["input_size"] == 5120
    assert projection.params["config"].tensor_model_parallel_size == 2
    assert projection.params["config"].gtp_weight_remat_size == 2
    assert projection.params["config"].gtp_remat_num_sequence_shards == 1
    provider = resolve_provider(args)
    assert provider.language_input_projection_specs[RADIO_ENCODER_MODULE_NAME] is (
        language_input_projection_spec
    )


# A full model instantiation (constructing MambaModel / RADIOEncoderWrapper) needs
# TE + a distributed init and is left to the cog functional check.


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("use_groups", [False, True])
@pytest.mark.parametrize("sequence_shards", [1, 2])
def test_language_cp_comes_from_its_grid(monkeypatch, cp_size, use_groups, sequence_shards):
    from types import SimpleNamespace

    from examples.mimo.model_providers import nemotron_moe_vlm as provider

    # Deliberately disagree with the grid to catch inheritance of stock CP.
    args = SimpleNamespace(
        mimo_llm_ep=1,
        mimo_llm_expt_tp=1,
        tensor_parallel_num_weight_shards=sequence_shards,
        tensor_parallel_num_sequence_shards=sequence_shards,
        spec=None,
        vocab_size=64,
        seq_length=32,
        hybrid_layer_pattern="*",
        logit_dtype=None,
    )
    monkeypatch.setattr(
        provider,
        "_base_config",
        lambda args: SimpleNamespace(
            context_parallel_size=8,
            calculate_per_token_loss=True,
            gtp_remat_num_sequence_shards=sequence_shards,
        ),
    )
    grid = SimpleNamespace(shape=[1, cp_size, 1], dim_names=["tp", "cp", "pp"])
    if sequence_shards > 1:
        grid.shape.append(sequence_shards)
        grid.dim_names.append("gtp_remat_sequence_shards")
    cp_size *= sequence_shards
    groups = None
    if use_groups:
        groups = SimpleNamespace(
            **{
                name: SimpleNamespace(size=lambda size=size: size, rank=lambda: 0)
                for name, size in {"tp": 1, "cp": cp_size, "pp": 1, "ep": 1, "expt_tp": 1}.items()
            }
        )
        monkeypatch.setattr(provider, "get_pg_size", lambda pg: pg.size())
        monkeypatch.setattr(provider, "get_pg_rank", lambda pg: pg.rank())
    spec = provider.language_model_spec(args, groups, grid)
    assert spec.params["config"].context_parallel_size == cp_size


def test_encoder_and_projection_do_not_inherit_language_cp():
    from examples.mimo.model_providers.nemotron_moe_vlm import nemotron_projection_config
    from examples.mimo.model_providers.radio_encoder import radio_vision_config

    args = _parse_validate(_build_argv(*_PRESET_20L))
    args.context_parallel_size = 2
    args.tensor_parallel_num_weight_shards = 2
    args.tensor_parallel_num_sequence_shards = 2
    vision = radio_vision_config(args, tp_size=1, pp_size=1)
    projection = nemotron_projection_config(args, tp_size=1, projection_input_size=5120)
    assert vision.context_parallel_size == 1
    assert projection.context_parallel_size == 1
    assert vision.gtp_remat_num_sequence_shards == 1
    assert projection.gtp_remat_num_sequence_shards == 1


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("sequence_shards", [1, 4])
def test_language_sequence_shards_are_excluded_from_projection(tp, sequence_shards):
    from examples.mimo.model_providers.nemotron_moe_vlm import (
        nemotron_language_config,
        nemotron_projection_config,
    )

    args = _parse_validate(_build_argv(*_PRESET_20L))
    args.tensor_model_parallel_size = tp
    args.sequence_parallel = tp > 1
    args.tensor_parallel_num_weight_shards = 8
    args.tensor_parallel_num_sequence_shards = tp * sequence_shards
    args.context_parallel_size = 2 * sequence_shards

    language = nemotron_language_config(
        args, tp, pp_size=1, ep_size=1, expt_tp_size=1, cp_size=args.context_parallel_size
    )
    assert language.gtp_remat_num_sequence_shards == sequence_shards
    assert language.context_parallel_size == 2 * sequence_shards
    projection = nemotron_projection_config(
        args, tp, projection_input_size=5120, base_config=language
    )
    assert projection.gtp_remat_num_sequence_shards == 1
    assert projection.context_parallel_size == 1
    assert projection.gtp_weight_remat_size == language.gtp_weight_remat_size
