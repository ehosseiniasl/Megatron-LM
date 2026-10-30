# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid import hybrid_block as hybrid_block_module
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.models.hybrid.shortcut_block import (
    ShortcutMoEBlock,
    group_layers_into_shortcut_blocks,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import TwoStageAttentionLayer
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from megatron.core.transformer.wide_residual_layer import StreamwiseSigmoidWideResidualConnection


@pytest.mark.parametrize("is_mtp_layer", [False, True])
@pytest.mark.parametrize("disable_mtp", [False, True])
def test_disable_mtp_shortcuts_preserves_decoder(monkeypatch, is_mtp_layer, disable_mtp):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=1,
        num_moe_experts=2,
        moe_shortcut_connection=True,
        moe_shortcut_parallel=True,
        moe_shortcut_post_norm=True,
        moe_shortcut_disable_mtp=disable_mtp,
    )
    grouped_configs = []

    class FakeLayer(torch.nn.Module):
        def __init__(self, config, layer_number, **kwargs):
            super().__init__()
            self.config = config
            self.layer_number = layer_number

    def record_grouping(layers, layer_types, config, **kwargs):
        grouped_configs.append(config)
        return layers

    monkeypatch.setattr(hybrid_block_module, "group_layers_into_shortcut_blocks", record_grouping)
    stack = hybrid_block_module.HybridStack(
        config=config,
        submodules=hybrid_block_module.HybridStackSubmodules(
            attention_layer=FakeLayer, moe_layer=FakeLayer
        ),
        layer_type_list=[LayerSymbols.ATTENTION, LayerSymbols.MOE],
        post_layer_norm=False,
        pg_collection=SimpleNamespace(
            pp=None, tp=None, cp=SimpleNamespace(size=lambda: 1), tp_cp=None
        ),
        is_mtp_layer=is_mtp_layer,
    )
    disabled = is_mtp_layer and disable_mtp
    assert len(grouped_configs) == (0 if disabled else 1)
    for flag in ("moe_shortcut_connection", "moe_shortcut_parallel", "moe_shortcut_post_norm"):
        assert getattr(stack.config, flag) == (not disabled)
        assert getattr(config, flag), "The shared decoder config must remain unchanged"
    assert all(layer.config is stack.config for layer in stack.layers)
    assert (stack.config is config) == (not disabled)


def _shortcut_config(
    *, parallel: bool = False, normalization: str = "RMSNorm", post_norm: bool = False
):
    return SimpleNamespace(
        moe_shortcut_parallel=parallel,
        moe_shortcut_post_norm=post_norm,
        fp32_residual_connection=False,
        hidden_size=8,
        layernorm_epsilon=1e-5,
        normalization=normalization,
        sequence_parallel=True,
        wide_residual=None,
        recompute_granularity=None,
        recompute_modules=[],
        fine_grained_activation_offloading=False,
        offload_modules=[],
    )


class _FakeCompute(torch.nn.Module, TwoStageAttentionLayer):
    def __init__(self, config, *, supports_two_stage: bool = True):
        super().__init__()
        self.config = config
        self.layer_number = 1
        self.is_first_layer = True
        self.is_last_layer = False
        self.supports_two_stage = supports_two_stage

    def supports_two_stage_attention(self) -> bool:
        return self.supports_two_stage

    def forward_pre_attn_and_core_attn(self, hidden_states, **kwargs):
        return (hidden_states,)

    def forward_post_core_attn(self, hidden_states, **kwargs):
        return hidden_states


class _FakeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tp_group = None


class _FakeNorm:
    def __new__(cls, *, config, hidden_size, eps):
        norm_cls = {"LayerNorm": torch.nn.LayerNorm, "RMSNorm": torch.nn.RMSNorm}[
            config.normalization
        ]
        norm = norm_cls(hidden_size, eps=eps)
        for parameter in norm.parameters():
            setattr(parameter, "sequence_parallel", config.sequence_parallel)
        return norm


class _FakeMoE(torch.nn.Module):
    def __init__(self, config, *, layer_number: int = 2):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.is_first_layer = layer_number == 1
        self.is_last_layer = False
        self.pre_mlp_layernorm = _FakeNorm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )
        self.submodules_config = SimpleNamespace(pre_mlp_layernorm=_FakeNorm)
        self.mlp = _FakeMLP()


@pytest.mark.parametrize(
    ("compute_symbol", "parallel"),
    [
        pytest.param(LayerSymbols.MAMBA, False, id="mamba-serial"),
        pytest.param(LayerSymbols.GDN, True, id="gdn-overlap"),
        pytest.param(LayerSymbols.ATTENTION, False, id="attention-serial"),
        pytest.param(LayerSymbols.DS_ATTENTION, False, id="dsa-serial"),
        pytest.param(LayerSymbols.MLA, True, id="mla-overlap"),
    ],
)
def test_group_layers_into_shortcut_blocks(compute_symbol, parallel):
    """Supported pairs are registered together while an initial MoE keeps its normal path."""
    config = _shortcut_config(parallel=parallel)
    leading_moe = _FakeMoE(config, layer_number=1)
    compute = _FakeCompute(config)
    paired_moe = _FakeMoE(config, layer_number=3)
    trailing_layer = torch.nn.Identity()

    grouped = group_layers_into_shortcut_blocks(
        torch.nn.ModuleList([leading_moe, compute, paired_moe, trailing_layer]),
        [LayerSymbols.MOE, compute_symbol, LayerSymbols.MOE, LayerSymbols.MLP],
        config,
    )

    assert len(grouped) == 3
    assert grouped[0] is leading_moe
    assert grouped[2] is trailing_layer
    shortcut = grouped[1]
    assert isinstance(shortcut, ShortcutMoEBlock)
    assert shortcut.compute_layer is compute
    assert shortcut.moe_layer is paired_moe
    assert shortcut.attn_layer_num == compute.layer_number - 1
    assert shortcut.moe_layer_num == paired_moe.layer_number - 1
    assert shortcut.overlap_mode is parallel
    assert shortcut.shortcut_pre_mlp_layernorm is not paired_moe.pre_mlp_layernorm
    assert isinstance(shortcut.shortcut_post_norm, IdentityOp)
    assert all(
        parameter.sequence_parallel
        for parameter in shortcut.shortcut_pre_mlp_layernorm.parameters()
    )

    state_keys = set(grouped.state_dict())
    assert "1.shortcut_pre_mlp_layernorm.weight" in state_keys
    assert "1.shortcut_post_norm.weight" not in state_keys


@pytest.mark.parametrize(
    ("normalization", "norm_type"),
    [
        pytest.param("LayerNorm", torch.nn.LayerNorm, id="layer-norm"),
        pytest.param("RMSNorm", torch.nn.RMSNorm, id="rms-norm"),
    ],
)
def test_shortcut_post_norm_follows_config(normalization, norm_type):
    config = _shortcut_config(normalization=normalization, post_norm=True)
    grouped = group_layers_into_shortcut_blocks(
        torch.nn.ModuleList([_FakeCompute(config), _FakeMoE(config)]),
        [LayerSymbols.ATTENTION, LayerSymbols.MOE],
        config,
    )

    assert isinstance(grouped[0].shortcut_post_norm, norm_type)
    assert all(
        parameter.sequence_parallel for parameter in grouped[0].shortcut_post_norm.parameters()
    )


def test_group_layers_rejects_pipeline_split_before_moe():
    config = _shortcut_config()

    with pytest.raises(ValueError, match="cannot cross pipeline-stage boundaries"):
        group_layers_into_shortcut_blocks(
            torch.nn.ModuleList([_FakeMoE(config)]), [LayerSymbols.MOE], config, pp_layer_offset=1
        )


def test_wide_shortcut_mtp_pair_stays_ordinary_width():
    """MTP auxiliary stacks do not inherit the decoder's wide residual stream."""

    config = TransformerConfig(
        num_layers=2,
        hidden_size=4,
        num_attention_heads=1,
        use_cpu_initialization=True,
        wide_residual=WideResidualConfig(num_streams=3),
    )
    compute = _FakeCompute(config)
    compute.is_mtp_layer = True
    moe = _FakeMoE(config)
    moe.is_mtp_layer = True

    block = ShortcutMoEBlock(compute, moe, overlap_a2a=False)
    ordinary_hidden = torch.randn(3, config.hidden_size)

    assert block.is_mtp_layer
    assert block.shortcut_residual_read is None
    assert block._read_shortcut_hidden(ordinary_hidden, recompute_context=None) is ordinary_hidden
    assert not any("shortcut_residual_read" in key for key in block.state_dict())


def test_shortcut_pair_rejects_mixed_mtp_ownership():
    config = _shortcut_config()
    compute = _FakeCompute(config)
    compute.is_mtp_layer = True

    with pytest.raises(ValueError, match="agree.*MTP"):
        ShortcutMoEBlock(compute, _FakeMoE(config), overlap_a2a=False)


def test_parallel_stream_is_initialized_once(monkeypatch):
    streams = []

    def make_stream(*, priority):
        stream = SimpleNamespace(priority=priority)
        streams.append(stream)
        return stream

    monkeypatch.setattr(ShortcutMoEBlock, "_parallel_stream", None)
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)

    first = ShortcutMoEBlock._get_a2a_overlap_stream()
    second = ShortcutMoEBlock._get_a2a_overlap_stream()

    assert first is second
    assert streams == [first]
    assert first.priority == -1


@pytest.mark.parametrize(
    ("compute_symbol", "compute", "error"),
    [
        pytest.param(
            LayerSymbols.MLP,
            lambda config: _FakeCompute(config),
            "must be preceded",
            id="unsupported-predecessor",
        ),
        pytest.param(
            LayerSymbols.MAMBA,
            lambda config: _FakeCompute(config, supports_two_stage=False),
            "does not support two-stage attention",
            id="atomic-forward-only",
        ),
    ],
)
def test_group_layers_rejects_invalid_shortcut_pair(compute_symbol, compute, error):
    config = _shortcut_config()
    with pytest.raises(ValueError, match=error):
        group_layers_into_shortcut_blocks(
            torch.nn.ModuleList([compute(config), _FakeMoE(config)]),
            [compute_symbol, LayerSymbols.MOE],
            config,
        )


def test_eager_overlap_matches_serial_output_and_gradients(monkeypatch):
    """Both schedules preserve physical-layer state, numerics, gradients, and join priority."""
    config = TransformerConfig(
        num_layers=2, hidden_size=4, num_attention_heads=1, use_cpu_initialization=True
    )
    active_quant_layers = []
    active_overlap = [None]
    marker_calls = []
    join_events = []
    inference_compute_calls = []
    dispatcher_selections = []
    compute_packed_seq_params = object()
    moe_packed_seq_params = object()

    def assert_quant_layer(layer_number):
        assert active_quant_layers[-1] == layer_number

    class FakeCompute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layer_number = 1
            self.input_scale = torch.nn.Parameter(torch.tensor(2.0))
            self.output_scale = torch.nn.Parameter(torch.tensor(3.0))

        def forward_pre_attn_and_core_attn(self, hidden_states, **kwargs):
            assert_quant_layer(0)
            assert kwargs["packed_seq_params"] is compute_packed_seq_params
            assert kwargs["residual_stream_recompute_context"] is None
            return hidden_states * self.input_scale, hidden_states

        def forward_post_core_attn(self, projected, residual, **kwargs):
            assert_quant_layer(0)
            assert kwargs["residual_stream_recompute_context"] is None
            return projected * self.output_scale

        def forward(self, hidden_states, **kwargs):
            assert_quant_layer(0)
            inference_compute_calls.append(kwargs)
            return hidden_states * self.input_scale * self.output_scale

    class FakeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tp_group = None
            self.dispatch_scale = torch.nn.Parameter(torch.tensor(5.0))
            self.expert_scale = torch.nn.Parameter(torch.tensor(7.0))
            self.combine_scale = torch.nn.Parameter(torch.tensor(11.0))
            self.route_scale = torch.nn.Parameter(torch.tensor(13.0))
            self.prob_scale = torch.nn.Parameter(torch.tensor(17.0))
            self.preprocess_scale = torch.nn.Parameter(torch.tensor(19.0))
            self.shared_scale = torch.nn.Parameter(torch.tensor(23.0))

        def route(self, shortcut_input, padding_mask):
            assert_quant_layer(1)
            assert padding_mask is None
            return shortcut_input * self.route_scale, shortcut_input * self.prob_scale

        def select_token_dispatcher(self):
            dispatcher_selections.append(InferenceMode.is_active())

        def preprocess(self, shortcut_input, probs, routing_map):
            assert_quant_layer(1)
            return shortcut_input * self.preprocess_scale, probs + routing_map

        def dispatch(self, route_input, route_probs):
            assert_quant_layer(1)
            return route_input * self.dispatch_scale, route_probs

        def routed_experts_compute(self, dispatched_input, dispatched_probs):
            assert_quant_layer(1)
            return dispatched_input + dispatched_probs * self.expert_scale, None

        def combine(self, routed_output):
            assert_quant_layer(1)
            return routed_output * self.combine_scale

        def postprocess(self, combined_output, shared_expert_output):
            assert_quant_layer(1)
            join_events.append((active_overlap[0], "postprocess"))
            return combined_output + shared_expert_output

        def shared_experts_compute(self, hidden_states):
            assert_quant_layer(1)
            join_events.append((active_overlap[0], "shared-experts"))
            return hidden_states * self.shared_scale

    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 2

            class FakeNorm:
                def __new__(cls, **kwargs):
                    return torch.nn.Identity()

            self.submodules_config = SimpleNamespace(pre_mlp_layernorm=FakeNorm)
            self.mlp = FakeMLP()
            self.residual_scale = torch.nn.Parameter(torch.tensor(29.0))
            self.mlp_state_scale = torch.nn.Parameter(torch.tensor(31.0))
            self.reflatten_scale = torch.nn.Parameter(torch.tensor(37.0))

        def _maybe_unflatten_for_moe(self, hidden_states, padding_mask, packed_seq_params):
            assert_quant_layer(1)
            assert packed_seq_params is moe_packed_seq_params
            return hidden_states, padding_mask, 4

        def _maybe_reflatten_from_moe(self, hidden_states, packed_seq_params, moe_unflatten_mbs):
            assert_quant_layer(1)
            assert packed_seq_params is moe_packed_seq_params
            assert moe_unflatten_mbs == 4
            return hidden_states * self.reflatten_scale

        def _pre_mlp_layernorm_and_residual(
            self, hidden_states, *, residual_stream_recompute_context
        ):
            assert_quant_layer(1)
            assert residual_stream_recompute_context is None
            return (
                hidden_states,
                hidden_states * self.residual_scale,
                (hidden_states * self.mlp_state_scale,),
            )

        def _apply_mlp_bda_step(
            self, output_with_bias, residual, mlp_state, *, residual_stream_recompute_context
        ):
            assert_quant_layer(1)
            assert residual_stream_recompute_context is None
            return output_with_bias[0] + residual + mlp_state[0]

    @contextmanager
    def quant_context_factory(config, layer_number):
        active_quant_layers.append(layer_number)
        try:
            yield
        finally:
            active_quant_layers.pop()

    def run(overlap):
        active_overlap[0] = overlap
        block = ShortcutMoEBlock(FakeCompute(), FakeMoE(), overlap_a2a=overlap)
        if overlap:

            def wait_combine(combined_output):
                join_events.append((overlap, "wait-combine"))
                return combined_output

            monkeypatch.setattr(
                block,
                "_launch_dispatch",
                lambda route_input, route_probs, async_op=False: block.moe_layer.mlp.dispatch(
                    route_input, route_probs
                ),
            )
            monkeypatch.setattr(
                block,
                "_wait_dispatch",
                lambda dispatched_input, dispatched_probs: (dispatched_input, dispatched_probs),
            )
            monkeypatch.setattr(
                block,
                "_launch_combine",
                lambda output, async_op=False: block.moe_layer.mlp.combine(output),
            )
            monkeypatch.setattr(block, "_wait_combine", wait_combine)

        hidden_states = torch.arange(1.0, 5.0, requires_grad=True)
        shortcut_hidden = hidden_states + 0.25
        output = block(
            hidden_states=hidden_states,
            shortcut_hidden=shortcut_hidden,
            attention_mask=None,
            inference_context=None,
            rotary_pos_emb=None,
            sequence_len_offset=None,
            compute_packed_seq_params=compute_packed_seq_params,
            moe_packed_seq_params=moe_packed_seq_params,
            padding_mask=None,
            quant_context_factory=quant_context_factory,
            quant_config=None,
            prepare_moe_input=lambda compute_output: compute_output + 0.5,
            compute_recompute_context=None,
            moe_recompute_context=None,
        )
        output.sum().backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in block.named_parameters()
            if parameter.grad is not None
        }
        gradients["hidden_states"] = hidden_states.grad.detach().clone()
        return output, gradients

    def record_marker(tensor, sequence):
        assert_quant_layer(1)
        join_events.append((active_overlap[0], "marker"))
        marker_calls.append((active_overlap[0], tensor, sequence))

    monkeypatch.setattr(
        torch.cuda, "Event", lambda *args, **kwargs: SimpleNamespace(record=lambda stream: None)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(
        "megatron.core.models.hybrid.shortcut_block.set_tensor_grad_fn_sequence_sr", record_marker
    )
    serial_output, serial_gradients = run(overlap=False)
    overlap_output, overlap_gradients = run(overlap=True)

    torch.testing.assert_close(overlap_output, serial_output)
    assert overlap_gradients.keys() == serial_gradients.keys()
    for name in overlap_gradients:
        torch.testing.assert_close(overlap_gradients[name], serial_gradients[name])

    assert [(overlap, sequence) for overlap, _, sequence in marker_calls] == [
        (False, torch.iinfo(torch.int).max),
        (True, torch.iinfo(torch.int).max),
    ]
    assert all(tensor.grad_fn is not None for _, tensor, _ in marker_calls)
    assert join_events == [
        (False, "shared-experts"),
        (False, "marker"),
        (False, "postprocess"),
        (True, "shared-experts"),
        (True, "wait-combine"),
        (True, "marker"),
        (True, "postprocess"),
    ]

    # Inference always uses the serial schedule and the compute layer's ordinary atomic forward,
    # even when the checkpoint was trained with side-stream overlap enabled.
    active_overlap[0] = True
    join_events.clear()
    inference_context = object()
    sequence_len_offset = object()
    rotary_pos_emb = object()
    inference_block = ShortcutMoEBlock(FakeCompute(), FakeMoE(), overlap_a2a=True)
    inference_block.eval()
    marker_count = len(marker_calls)
    hidden_states = torch.arange(1.0, 5.0)
    with torch.inference_mode(), InferenceMode.active():
        inference_output = inference_block(
            hidden_states=hidden_states,
            shortcut_hidden=hidden_states + 0.25,
            attention_mask=None,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            sequence_len_offset=sequence_len_offset,
            compute_packed_seq_params=compute_packed_seq_params,
            moe_packed_seq_params=moe_packed_seq_params,
            padding_mask=None,
            quant_context_factory=quant_context_factory,
            quant_config=None,
            prepare_moe_input=lambda compute_output: compute_output + 0.5,
            compute_recompute_context=None,
            moe_recompute_context=None,
        )

    torch.testing.assert_close(inference_output, serial_output.detach())
    assert dispatcher_selections[-1]
    assert len(inference_compute_calls) == 1
    compute_call = inference_compute_calls[0]
    assert compute_call["inference_context"] is inference_context
    assert compute_call["sequence_len_offset"] is sequence_len_offset
    assert compute_call["rotary_pos_emb"] is rotary_pos_emb
    assert compute_call["packed_seq_params"] is compute_packed_seq_params
    assert len(marker_calls) == marker_count
    assert join_events == [(True, "shared-experts"), (True, "postprocess")]


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "overlap"])
def test_wide_shortcut_owns_independent_registered_read(monkeypatch, parallel):
    """The routed shortcut reads wide X_l through its own ordinary-width controller."""

    config = TransformerConfig(
        num_layers=2,
        hidden_size=4,
        num_attention_heads=1,
        use_cpu_initialization=True,
        wide_residual=WideResidualConfig(num_streams=3),
    )
    pg_collection = ProcessGroupCollection()

    class WideFakeCompute(_FakeCompute):
        def __init__(self):
            super().__init__(config)

    class WideFakeMoE(_FakeMoE):
        def __init__(self):
            super().__init__(config)
            self.pg_collection = pg_collection
            self.residual_connection_mlp = StreamwiseSigmoidWideResidualConnection(
                config=config,
                layer_number=self.layer_number,
                branch_name="mlp",
                pg_collection=pg_collection,
            )

    if parallel:
        monkeypatch.setattr(
            torch.cuda, "Event", lambda *args, **kwargs: SimpleNamespace(record=lambda stream: None)
        )

    block = ShortcutMoEBlock(WideFakeCompute(), WideFakeMoE(), overlap_a2a=parallel)

    class RegisteredOwner(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layers = torch.nn.ModuleList([layer])

    owner = RegisteredOwner(block)
    state_keys = set(owner.state_dict())
    assert "layers.0.shortcut_residual_read.read_map.logit" in state_keys
    assert "layers.0.moe_layer.residual_connection_mlp.read_map.logit" in state_keys
    assert (
        block.shortcut_residual_read.read_map.logit
        is not block.moe_layer.residual_connection_mlp.read_map.logit
    )

    wide_hidden = torch.randn(5, 3 * config.hidden_size, requires_grad=True)
    shortcut_hidden = block._read_shortcut_hidden(wide_hidden, recompute_context=None)
    assert shortcut_hidden.shape == (5, config.hidden_size)

    shortcut_hidden.square().mean().backward()
    read_grad = block.shortcut_residual_read.read_map.logit.grad
    assert read_grad is not None
    assert torch.count_nonzero(read_grad[: config.wide_residual.num_streams]) > 0
    assert wide_hidden.grad is not None
    assert torch.isfinite(wide_hidden.grad).all()


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "overlap"])
def test_shortcut_standalone_recompute_and_offload(monkeypatch, parallel):
    """Standalone memory controls preserve packed-MBS boundaries in both schedules."""
    from megatron.core.models.hybrid import shortcut_block as shortcut_block_module

    calls = {"checkpoints": [], "offloads": [], "unflatten_shapes": []}

    class RecordingCheckpoint:
        def __init__(self, fp8=False, ckpt_manager=None, retain_input_tensors=False):
            del fp8, retain_input_tensors
            self.ckpt_manager = ckpt_manager
            calls["checkpoints"].append(self)

        def checkpoint(self, run_function, hidden_states):
            calls["checkpoint_input"] = hidden_states
            calls["checkpoint_output"] = run_function(hidden_states)
            return calls["checkpoint_output"]

        def discard_output_and_register_recompute(self, hook_tensor):
            calls["recompute_hook"] = hook_tensor

    class RecordingOffload:
        def __init__(self, enabled, tensor, name):
            self.enabled = enabled
            self.tensor = tensor
            self.name = name
            self.record = {"enabled": enabled, "input": tensor, "name": name}
            calls["offloads"].append(self.record)

        def __enter__(self):
            self.record["entered"] = True
            return self.tensor

        def __exit__(self, *_args):
            self.record["exited"] = True

        def group_offload(self, tensor, forced_released_tensors=None):
            self.record["output"] = tensor
            self.record["forced_released_tensors"] = forced_released_tensors
            return tensor

        @staticmethod
        def mark_not_offload(tensor):
            calls.setdefault("not_offloaded", []).append(tensor)

    monkeypatch.setattr(shortcut_block_module, "CheckpointWithoutOutput", RecordingCheckpoint)
    monkeypatch.setattr(
        shortcut_block_module, "_get_offloading_interface", lambda: RecordingOffload
    )
    monkeypatch.setattr(shortcut_block_module, "set_tensor_grad_fn_sequence_sr", lambda *_: None)
    if parallel:
        monkeypatch.setattr(
            torch.cuda, "Event", lambda *args, **kwargs: SimpleNamespace(record=lambda stream: None)
        )
        monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())

    config = TransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=2,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_shortcut_connection=True,
        moe_shortcut_post_norm=True,
        recompute_granularity="selective",
        recompute_modules=["shortcut_pre_mlp_layernorm"],
        fine_grained_activation_offloading=True,
        offload_modules=["shortcut_post_norm"],
    )
    packed_seq_params = object()
    micro_batch_size = 2
    sequence_length = 3

    moe_layer = _FakeMoE(config)
    mlp = moe_layer.mlp

    def route(hidden_states, padding_mask=None):
        calls["route_shape"] = tuple(hidden_states.shape)
        assert padding_mask is None
        return torch.ones_like(hidden_states), torch.ones_like(hidden_states, dtype=torch.bool)

    def combine(routed_output):
        calls["combined_output"] = routed_output * 2
        return calls["combined_output"]

    def postprocess(combined_output, shared_expert_output):
        calls["postprocess_output"] = combined_output + shared_expert_output
        return calls["postprocess_output"]

    def maybe_unflatten(hidden_states, padding_mask, packed):
        assert packed is packed_seq_params
        calls["unflatten_shapes"].append(tuple(hidden_states.shape))
        unflattened = hidden_states.reshape(
            micro_batch_size, sequence_length, config.hidden_size
        ).transpose(0, 1)
        return unflattened, padding_mask, micro_batch_size

    def maybe_reflatten(hidden_states, packed, unflatten_mbs):
        assert packed is packed_seq_params
        assert unflatten_mbs == micro_batch_size
        calls["reflatten_shape"] = tuple(hidden_states.shape)
        return hidden_states.transpose(0, 1).reshape(
            micro_batch_size * sequence_length, 1, config.hidden_size
        )

    def pre_mlp(hidden_states, *, residual_stream_recompute_context):
        assert residual_stream_recompute_context is None
        return hidden_states, hidden_states, (hidden_states * 0.5,)

    def apply_mlp_bda(output_with_bias, residual, mlp_state, *, residual_stream_recompute_context):
        assert residual_stream_recompute_context is None
        return output_with_bias[0] + residual + mlp_state[0]

    mlp.route = route
    mlp.preprocess = lambda hidden_states, probs, routing_map: (hidden_states, probs)
    mlp.dispatch = lambda hidden_states, probs: (hidden_states, probs)
    mlp.routed_experts_compute = lambda hidden_states, probs: (hidden_states + probs, None)
    mlp.combine = combine
    mlp.shared_experts_compute = lambda hidden_states: hidden_states * 0.25
    mlp.postprocess = postprocess
    moe_layer._maybe_unflatten_for_moe = maybe_unflatten
    moe_layer._maybe_reflatten_from_moe = maybe_reflatten
    moe_layer._pre_mlp_layernorm_and_residual = pre_mlp
    moe_layer._apply_mlp_bda_step = apply_mlp_bda

    block = ShortcutMoEBlock(_FakeCompute(config), moe_layer, overlap_a2a=parallel)
    if parallel:
        monkeypatch.setattr(
            block,
            "_launch_dispatch",
            lambda hidden_states, probs, async_op=False: mlp.dispatch(hidden_states, probs),
        )
        monkeypatch.setattr(
            block, "_wait_dispatch", lambda hidden_states, probs: (hidden_states, probs)
        )
        monkeypatch.setattr(
            block, "_launch_combine", lambda output, async_op=False: mlp.combine(output)
        )
        monkeypatch.setattr(block, "_wait_combine", lambda output: output)

    hidden_states = torch.randn(
        micro_batch_size * sequence_length, 1, config.hidden_size, requires_grad=True
    )

    @contextmanager
    def quant_context_factory(*_args):
        yield

    output = block(
        hidden_states=hidden_states,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        sequence_len_offset=None,
        compute_packed_seq_params=packed_seq_params,
        moe_packed_seq_params=packed_seq_params,
        padding_mask=None,
        quant_context_factory=quant_context_factory,
        quant_config=None,
        compute_recompute_context=None,
        moe_recompute_context=None,
    )
    output.sum().backward()

    assert len(calls["checkpoints"]) == 1
    assert calls["checkpoints"][0].ckpt_manager is None
    assert calls["checkpoint_input"] is hidden_states
    assert calls["recompute_hook"] is calls["combined_output"]
    assert block.shortcut_pre_mlp_layernorm_checkpoint is None
    assert calls["unflatten_shapes"] == [
        (micro_batch_size * sequence_length, 1, config.hidden_size),
        (micro_batch_size * sequence_length, 1, config.hidden_size),
    ]
    assert calls["route_shape"] == (sequence_length, micro_batch_size, config.hidden_size)
    assert calls["reflatten_shape"] == (sequence_length, micro_batch_size, config.hidden_size)
    assert output.shape == hidden_states.shape
    assert len(calls["offloads"]) == 1
    offload = calls["offloads"][0]
    assert offload["enabled"]
    assert offload["name"] == "shortcut_post_norm"
    assert offload["input"] is calls["postprocess_output"]
    assert tuple(offload["input"].shape) == (sequence_length, micro_batch_size, config.hidden_size)
    assert offload["entered"] and offload["exited"]
    assert len(offload["forced_released_tensors"]) == 1
    assert offload["forced_released_tensors"][0] is offload["input"]
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


@pytest.mark.parametrize("explicit_pre_norm_recompute", [False, True], ids=["replay", "both"])
def test_residual_replay_owns_shortcut_norms(monkeypatch, explicit_pre_norm_recompute):
    """Residual replay supersedes standalone recompute and shortcut post-norm offload."""
    from megatron.core.models.hybrid import shortcut_block as shortcut_block_module

    calls = {"checkpoints": [], "offloads": [], "not_offloaded": []}
    replay_manager = SimpleNamespace(checkpoints=[])

    class RecordingCheckpoint:
        def __init__(self, fp8=False, ckpt_manager=None, retain_input_tensors=False):
            del fp8, retain_input_tensors
            self.ckpt_manager = ckpt_manager
            calls["checkpoints"].append(self)

        def checkpoint(self, run_function, hidden_states):
            output = run_function(hidden_states)
            if self.ckpt_manager is not None:
                self.ckpt_manager.checkpoints.append(self)
            return output

        def discard_output_and_register_recompute(self, _hook_tensor):
            calls.setdefault("standalone_discards", 0)
            calls["standalone_discards"] += 1

    class RecordingOffload:
        def __init__(self, enabled, tensor, name):
            self.tensor = tensor
            self.record = {"enabled": enabled, "input": tensor, "name": name}
            calls["offloads"].append(self.record)

        def __enter__(self):
            return self.tensor

        def __exit__(self, *_args):
            return None

        def group_offload(self, tensor, forced_released_tensors=None):
            self.record["forced_released_tensors"] = forced_released_tensors
            return tensor

        @staticmethod
        def mark_not_offload(tensor):
            calls["not_offloaded"].append(tensor)

    class RecordingReplayContext:
        def __init__(self):
            self.manager = replay_manager
            self.inputs = []

        def checkpoint(self, run_function, hidden_states):
            self.inputs.append(hidden_states)
            return run_function(hidden_states)

    monkeypatch.setattr(shortcut_block_module, "CheckpointWithoutOutput", RecordingCheckpoint)
    monkeypatch.setattr(
        shortcut_block_module, "_get_offloading_interface", lambda: RecordingOffload
    )

    config = _shortcut_config()
    config.moe_shortcut_post_norm = True
    config.recompute_granularity = "selective"
    config.recompute_modules = ["shortcut_pre_mlp_layernorm"] if explicit_pre_norm_recompute else []
    config.fine_grained_activation_offloading = True
    config.offload_modules = ["shortcut_post_norm"]
    moe_layer = _FakeMoE(config)
    moe_layer.mlp.route = lambda hidden_states, padding_mask=None: (
        torch.ones_like(hidden_states),
        torch.ones_like(hidden_states, dtype=torch.bool),
    )
    moe_layer.mlp.preprocess = lambda hidden_states, probs, routing_map: (hidden_states, probs)

    def postprocess(combined_output, shared_expert_output):
        calls["postprocess_output"] = combined_output + shared_expert_output
        return calls["postprocess_output"]

    moe_layer.mlp.postprocess = postprocess
    moe_layer._maybe_unflatten_for_moe = lambda hidden_states, padding_mask, packed_seq_params: (
        hidden_states,
        padding_mask,
        None,
    )
    moe_layer._maybe_reflatten_from_moe = (
        lambda hidden_states, packed_seq_params, unflatten_mbs: hidden_states
    )

    def apply_mlp_bda(output_with_bias, residual, mlp_state, *, residual_stream_recompute_context):
        assert mlp_state == ("state",)
        assert residual_stream_recompute_context is replay_context
        calls["bda_input"] = output_with_bias[0]
        return output_with_bias[0] + residual

    moe_layer._apply_mlp_bda_step = apply_mlp_bda
    block = ShortcutMoEBlock(_FakeCompute(config), moe_layer, overlap_a2a=False)
    replay_context = RecordingReplayContext()
    hidden_states = torch.randn(4, 1, config.hidden_size, requires_grad=True)

    route_input, _ = block._moe_router_preprocess(hidden_states, recompute_context=replay_context)
    combined_output = route_input * 2
    output = block._postprocess(
        combined_output,
        torch.zeros_like(combined_output),
        torch.zeros_like(combined_output),
        ("state",),
        moe_recompute_context=replay_context,
    )
    output.sum().backward()

    assert len(calls["checkpoints"]) == 1
    assert calls["checkpoints"][0].ckpt_manager is replay_manager
    assert replay_manager.checkpoints == calls["checkpoints"]
    assert block.shortcut_pre_mlp_layernorm_checkpoint is None
    assert calls.get("standalone_discards", 0) == 0
    assert len(replay_context.inputs) == 1
    assert replay_context.inputs[0] is calls["postprocess_output"]
    assert len(calls["offloads"]) == 1
    assert calls["offloads"][0]["name"] == "shortcut_post_norm"
    assert calls["offloads"][0]["enabled"] is False
    assert calls["offloads"][0]["input"] is calls["postprocess_output"]
    assert len(calls["not_offloaded"]) == 2
    assert calls["not_offloaded"][0] is route_input
    assert calls["not_offloaded"][1] is calls["bda_input"]
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def test_identity_shortcut_norms_disable_memory_controls(monkeypatch):
    """Identity shortcut norms never discard or offload aliased input storage."""
    from megatron.core.models.hybrid import shortcut_block as shortcut_block_module

    class IdentityNorm:
        def __new__(cls, **_kwargs):
            return IdentityOp()

    def fail_checkpoint(*_args, **_kwargs):
        raise AssertionError("identity shortcut pre-norm must not be checkpointed")

    monkeypatch.setattr(shortcut_block_module, "CheckpointWithoutOutput", fail_checkpoint)
    config = _shortcut_config()
    config.recompute_granularity = "selective"
    config.recompute_modules = ["shortcut_pre_mlp_layernorm"]
    config.fine_grained_activation_offloading = True
    config.offload_modules = ["shortcut_post_norm"]
    moe_layer = _FakeMoE(config)
    moe_layer.submodules_config = SimpleNamespace(pre_mlp_layernorm=IdentityNorm)
    moe_layer.mlp.route = lambda hidden_states, padding_mask=None: (
        torch.ones_like(hidden_states),
        torch.ones_like(hidden_states, dtype=torch.bool),
    )
    moe_layer.mlp.preprocess = lambda hidden_states, probs, routing_map: (hidden_states, probs)
    moe_layer._maybe_unflatten_for_moe = lambda hidden_states, padding_mask, packed_seq_params: (
        hidden_states,
        padding_mask,
        None,
    )

    block = ShortcutMoEBlock(_FakeCompute(config), moe_layer, overlap_a2a=False)
    hidden_states = torch.randn(4, 1, config.hidden_size)
    original = hidden_states.clone()
    block._moe_router_preprocess(hidden_states)

    assert not block.recompute_shortcut_pre_mlp_layernorm
    assert not block.offload_shortcut_post_norm
    torch.testing.assert_close(hidden_states, original)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        pytest.param(
            {
                "recompute_granularity": "selective",
                "recompute_modules": ["shortcut_pre_mlp_layernorm"],
            },
            "require moe_shortcut_connection=True",
            id="pre-norm-without-shortcut",
        ),
        pytest.param(
            {"fine_grained_activation_offloading": True, "offload_modules": ["shortcut_post_norm"]},
            "require moe_shortcut_connection=True",
            id="post-norm-without-shortcut",
        ),
        pytest.param(
            {"moe_shortcut_connection": True, "recompute_modules": ["shortcut_pre_mlp_layernorm"]},
            "requires recompute_granularity='selective'",
            id="pre-norm-requires-selective",
        ),
        pytest.param(
            {
                "moe_shortcut_connection": True,
                "fine_grained_activation_offloading": True,
                "offload_modules": ["shortcut_post_norm"],
            },
            "requires moe_shortcut_post_norm=True",
            id="post-norm-offload-requires-post-norm",
        ),
    ],
)
def test_shortcut_memory_controls_validate_configuration(overrides, error):
    kwargs = {
        "num_layers": 2,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_moe_experts": 2,
        "moe_router_topk": 1,
        "moe_router_pre_softmax": True,
    }
    kwargs.update(overrides)
    with pytest.raises((AssertionError, ValueError), match=error):
        TransformerConfig(**kwargs)


def test_residual_replay_reports_shortcut_post_norm_offload_ownership():
    with pytest.warns(UserWarning, match="shortcut_post_norm"):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=8,
            num_attention_heads=2,
            num_moe_experts=2,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_shortcut_connection=True,
            moe_shortcut_post_norm=True,
            wide_residual=WideResidualConfig(num_streams=3),
            recompute_granularity="selective",
            recompute_modules=["residual_stream", "shortcut_pre_mlp_layernorm"],
            residual_stream_recompute_num_layers=1,
            fine_grained_activation_offloading=True,
            offload_modules=["shortcut_post_norm"],
        )

    assert config.recompute_modules == ["residual_stream", "shortcut_pre_mlp_layernorm"]
    assert config.offload_modules == ["shortcut_post_norm"]
