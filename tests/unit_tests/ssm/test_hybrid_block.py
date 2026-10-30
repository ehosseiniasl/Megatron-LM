# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch

import megatron.core.models.hybrid.hybrid_block as hybrid_block_module
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import (
    gated_delta_product_stack_spec,
    hybrid_inference_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA as HAVE_GDN
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.ssm.gated_delta_product import HAVE_FLA as HAVE_GDP
from megatron.core.ssm.gated_delta_product import HAVE_MAMBA_SSM as HAVE_GDP_MAMBA
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig, WideResidualConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils

TWO_STAGE_ATTENTION_CASES = [
    pytest.param(Symbols.MAMBA, hybrid_stack_spec, {}, id="mamba"),
    pytest.param(
        Symbols.GDN,
        hybrid_stack_spec,
        {"bf16": True, "params_dtype": torch.bfloat16, "activation_func": torch.nn.functional.silu},
        marks=pytest.mark.skipif(not HAVE_GDN, reason="FLA is not installed"),
        id="gdn",
    ),
    pytest.param(Symbols.ATTENTION, hybrid_stack_spec, {}, id="attention"),
    pytest.param(
        Symbols.MAMBA,
        gated_delta_product_stack_spec,
        {
            "bf16": True,
            "params_dtype": torch.bfloat16,
            "mamba_num_heads": 4,
            "mamba_head_dim": 64,
            "mamba_num_groups": 4,
            "mamba_state_dim": 16,
        },
        marks=pytest.mark.skipif(
            not (HAVE_GDP and HAVE_GDP_MAMBA), reason="GDP dependencies are not installed"
        ),
        id="gdp",
    ),
]


class _HostPhysicalLayer(torch.nn.Module):
    """Minimal registered physical layer used by the host scheduling test."""

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number


@pytest.fixture
def host_hybrid_stack_builder(monkeypatch):
    """Run the real stack constructor with host-only physical layer builders."""
    moe_spec = object()
    submodules = SimpleNamespace(
        mamba_layer=object(), attention_layer=object(), mlp_layer=object(), moe_layer=moe_spec
    )

    def build_layer(spec, *, layer_number, **kwargs):
        layer = _HostPhysicalLayer(layer_number)
        if spec is moe_spec:
            layer.mlp = SimpleNamespace(router=SimpleNamespace(moe_layer_number=None))
        return layer

    monkeypatch.setattr(hybrid_block_module, "build_module", build_layer)
    group = SimpleNamespace(size=lambda: 1)
    pg_collection = SimpleNamespace(pp=group, tp=group, cp=group, tp_cp=group)

    def build(pattern, *, pp_layer_offset=0, moe_layer_offset=None, num_layers=2):
        config = SimpleNamespace(
            wide_residual=None,
            linear_cp_layout="contiguous",
            moe_router_orthogonality_loss_num_layers=num_layers,
            fp8=None,
            fp4=None,
            cuda_graph_impl="none",
            moe_shortcut_connection=False,
        )
        return HybridStack(
            config=config,
            submodules=submodules,
            layer_type_list=validate_segment_layers(pattern),
            pp_layer_offset=pp_layer_offset,
            moe_layer_offset=moe_layer_offset,
            post_layer_norm=False,
            post_process=False,
            pg_collection=pg_collection,
        )

    return build


@pytest.mark.parametrize(
    "pattern, block_offset, moe_offset, expected",
    [
        pytest.param("M-E*E", 0, None, [(3, 1), (5, 2)], id="first-segment"),
        pytest.param("*EM-E", 7, 3, [(9, 4), (12, 5)], id="later-segment"),
    ],
)
def test_orthogonality_moe_numbers_skip_other_hybrid_layers(
    host_hybrid_stack_builder, pattern, block_offset, moe_offset, expected
):
    """Physical block offsets and global MoE ordinals remain independent across PP/VPP."""
    stack = host_hybrid_stack_builder(
        pattern, pp_layer_offset=block_offset, moe_layer_offset=moe_offset
    )

    assert [layer.layer_number for layer in stack.layers] == list(
        range(block_offset + 1, block_offset + len(pattern) + 1)
    )
    assert [
        (layer.layer_number, layer.mlp.router.moe_layer_number)
        for layer, symbol in zip(stack.layers, pattern)
        if symbol == Symbols.MOE
    ] == expected


def test_orthogonality_limit_requires_global_moe_offset(host_hybrid_stack_builder):
    """A later pipeline segment must not silently restart the first-N selection."""
    with pytest.raises(ValueError, match="moe_layer_offset is required"):
        host_hybrid_stack_builder("*E", pp_layer_offset=4)


@pytest.mark.parametrize("num_layers", [None, 0])
def test_unrestricted_or_disabled_orthogonality_needs_no_moe_offset(
    host_hybrid_stack_builder, num_layers
):
    """Legacy all-layer behavior and disabling the loss need no global MoE numbering."""
    stack = host_hybrid_stack_builder("*E", pp_layer_offset=4, num_layers=num_layers)

    assert [layer.layer_number for layer in stack.layers] == [5, 6]
    assert stack.layers[1].mlp.router.moe_layer_number is None


class _HostShortcutBlock(ShortcutMoEBlock):
    """Shortcut double that exposes host scheduling inputs without CUDA kernels."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.layer_number = 1
        self.attn_layer_num = 0
        self.moe_layer_num = 1
        self.compute_layer = _HostPhysicalLayer(layer_number=1)
        self.moe_layer = _HostPhysicalLayer(layer_number=2)
        self.received = None
        self.midpoint = None

    def forward(self, **kwargs):
        self.received = kwargs
        self.midpoint = kwargs["hidden_states"] + 10.0
        moe_input = kwargs["prepare_moe_input"](self.midpoint)
        return moe_input + 20.0


class _HostCPState:
    """Record each physical CP transition and make its result distinguishable."""

    def __init__(self):
        self.prepare_calls = []
        self.finalize_calls = []

    def build_forward_state(
        self, packed_seq_params, packed_seq_params_by_layout=None, thd_plan=None
    ):
        assert packed_seq_params is None
        assert packed_seq_params_by_layout is None
        assert thd_plan is None
        return self

    def prepare_layer(self, layer_index, hidden_states):
        call_number = len(self.prepare_calls) + 1
        converted = hidden_states + float(call_number)
        packed = object()
        self.prepare_calls.append(
            (layer_index, hidden_states.detach().clone(), converted.detach().clone(), packed)
        )
        return converted, packed

    def finalize_layer(self, layer_index, hidden_states):
        self.finalize_calls.append((layer_index, hidden_states.detach().clone()))
        return hidden_states + 50.0


class _HostReplayContext:
    def __init__(self, manager, *, is_block_end):
        self.manager = manager
        self.is_block_end = is_block_end
        self.finalize_calls = []

    def finalize(self, hidden_states):
        self.finalize_calls.append(hidden_states.detach().clone())


def _host_shortcut_stack(shortcut, hidden_states, cp_state=None):
    stack = HybridStack.__new__(HybridStack)
    torch.nn.Module.__init__(stack)
    stack.config = SimpleNamespace(
        cuda_graph_impl="none",
        flash_decode=False,
        fp8=False,
        fp4=None,
        moe_shortcut_connection=True,
        recompute_granularity="selective",
        recompute_modules=["residual_stream"],
        residual_stream_recompute_num_layers=1,
    )
    stack.layers = torch.nn.ModuleList([shortcut])
    stack.layer_type_list = [Symbols.MAMBA, Symbols.MOE]
    stack.num_layers_per_pipeline_rank = 2
    stack.pre_process = False
    stack.post_process = False
    stack.post_layer_norm = False
    stack.uses_wide_residual_stream = True
    stack.input_tensor = hidden_states
    stack.residual_stream_readout = None
    stack._has_linear_layer_with_chunkwise_cp = False
    stack._cp_layout_manager = cp_state
    stack.train()
    return stack


def test_shortcut_host_schedule_uses_physical_cp_and_replay_boundaries(monkeypatch):
    """X_l branches and X_mid convert separately; only the MoE boundary is finalized."""

    shortcut = _HostShortcutBlock()
    cp_state = _HostCPState()
    initial_hidden = torch.arange(6.0).view(2, 3).requires_grad_()
    stack = _host_shortcut_stack(shortcut, initial_hidden, cp_state)

    shared_manager = object()
    compute_context = _HostReplayContext(shared_manager, is_block_end=False)
    moe_context = _HostReplayContext(shared_manager, is_block_end=True)
    plan_calls = []

    def build_plan(num_layers, block_size, *, atomic_layer_pairs):
        plan_calls.append((num_layers, block_size, atomic_layer_pairs))
        return [compute_context, moe_context]

    observations = []

    def observe(layer, before, after):
        observations.append((layer.layer_number, before.detach().clone(), after.detach().clone()))

    monkeypatch.setattr(hybrid_block_module, "build_residual_stream_recompute_plan", build_plan)
    monkeypatch.setattr(hybrid_block_module, "observe_layer_residuals", observe)

    output = stack(torch.empty_like(initial_hidden), attention_mask=None)

    assert plan_calls == [(2, 1, ((0, 1),))]
    assert [call[0] for call in cp_state.prepare_calls] == [0, 1, 1]
    torch.testing.assert_close(cp_state.prepare_calls[0][1], initial_hidden)
    torch.testing.assert_close(cp_state.prepare_calls[1][1], cp_state.prepare_calls[0][2])
    torch.testing.assert_close(cp_state.prepare_calls[2][1], shortcut.midpoint)

    assert shortcut.received["hidden_states"] is not shortcut.received["shortcut_hidden"]
    torch.testing.assert_close(shortcut.received["hidden_states"], cp_state.prepare_calls[0][2])
    torch.testing.assert_close(shortcut.received["shortcut_hidden"], cp_state.prepare_calls[1][2])
    assert shortcut.received["compute_packed_seq_params"] is cp_state.prepare_calls[0][3]
    assert shortcut.received["moe_packed_seq_params"] is cp_state.prepare_calls[1][3]
    assert shortcut.received["compute_recompute_context"] is compute_context
    assert shortcut.received["moe_recompute_context"] is moe_context

    assert [call[0] for call in cp_state.finalize_calls] == [1]
    assert compute_context.finalize_calls == []
    assert len(moe_context.finalize_calls) == 1
    torch.testing.assert_close(moe_context.finalize_calls[0], initial_hidden + 84.0)
    assert [observation[0] for observation in observations] == [1, 2]
    torch.testing.assert_close(observations[0][1], cp_state.prepare_calls[0][2])
    torch.testing.assert_close(observations[0][2], shortcut.midpoint)
    torch.testing.assert_close(observations[1][1], cp_state.prepare_calls[2][2])
    torch.testing.assert_close(observations[1][2], cp_state.finalize_calls[0][1])
    torch.testing.assert_close(output, initial_hidden + 84.0)


def test_decoder_residual_boundaries_straddle_wide_readout(monkeypatch):
    """The final-layer and wide-readout summaries observe opposite sides of the readout."""
    initial_hidden = torch.arange(6.0).view(2, 3)
    stack = _host_shortcut_stack(_HostShortcutBlock(), initial_hidden)
    stack.config.recompute_modules = []
    stack.post_process = True
    stack.name = "decoder"
    stack.is_mtp_layer = False

    class ScaleReadout(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 0.25

    stack.residual_stream_readout = ScaleReadout()
    observations = []
    monkeypatch.setattr(
        hybrid_block_module,
        "observe_residual_summary_boundary",
        lambda owner, name, tensor: observations.append((owner, name, tensor.detach().clone())),
    )

    output = stack(torch.empty_like(initial_hidden), attention_mask=None)

    assert [name for _, name, _ in observations] == ["last_layer", "last_layer_widereadout"]
    assert all(owner is stack for owner, _, _ in observations)
    torch.testing.assert_close(observations[0][2], initial_hidden + 30.0)
    torch.testing.assert_close(observations[1][2], (initial_hidden + 30.0) * 0.25)
    torch.testing.assert_close(output, observations[1][2])


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2 or int(os.environ.get("WORLD_SIZE", "1")) % 2 != 0,
    reason="Run this test with an even WORLD_SIZE >= 2.",
)
@pytest.mark.parametrize(
    ("fp32_residual_connection", "residual_replay", "batch_size"),
    [(False, False, 2), (True, True, 16)],
    ids=("native", "fp32_replay"),
)
def test_attention_shortcut_wide_residual_ep2_serial_overlap_parity(
    fp32_residual_connection, residual_replay, batch_size
):
    """Real EP=2 all-to-all preserves wide-shortcut eager serial/overlap numerics."""

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(123)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        def build(parallel):
            replay_config = {}
            if residual_replay:
                replay_config = {
                    "recompute_granularity": "selective",
                    "recompute_modules": ["residual_stream"],
                    "residual_stream_recompute_num_layers": 1,
                }
            config = TransformerConfig(
                hidden_size=256,
                num_layers=2,
                num_attention_heads=4,
                bf16=fp32_residual_connection,
                params_dtype=(torch.bfloat16 if fp32_residual_connection else torch.float32),
                fp32_residual_connection=fp32_residual_connection,
                num_moe_experts=2,
                expert_model_parallel_size=2,
                moe_ffn_hidden_size=512,
                moe_router_topk=1,
                moe_router_pre_softmax=True,
                moe_router_load_balancing_type="none",
                moe_token_dispatcher_type="alltoall",
                moe_shortcut_connection=True,
                moe_shortcut_parallel=parallel,
                moe_shared_expert_intermediate_size=256,
                add_bias_linear=False,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                use_cpu_initialization=True,
                wide_residual=WideResidualConfig(num_streams=3),
                **replay_config,
            )
            return (
                HybridStack(
                    config,
                    hybrid_stack_spec.submodules,
                    layer_type_list=[Symbols.ATTENTION, Symbols.MOE],
                    pp_layer_offset=0,
                    pg_collection=pg_collection,
                )
                .cuda()
                .to(dtype=config.params_dtype)
            )

        serial = build(parallel=False)
        overlap = build(parallel=True)
        overlap.load_state_dict(serial.state_dict(), strict=True)
        serial.train()
        overlap.train()

        torch.manual_seed(456)
        sequence_length = 16
        base_hidden = torch.randn(
            sequence_length,
            batch_size,
            256,
            device=torch.cuda.current_device(),
            dtype=(torch.bfloat16 if fp32_residual_connection else torch.float32),
        )
        torch.distributed.broadcast(base_hidden, src=0)
        attention_mask = torch.triu(
            torch.ones(
                1, 1, sequence_length, sequence_length, dtype=torch.bool, device=base_hidden.device
            ),
            diagonal=1,
        )

        residual_write_dtypes = {"serial": [], "overlap": []}

        def record_residual_writes(model, mode):
            def hook(_module, _args, kwargs, output):
                if kwargs.get("operation") == "write":
                    residual_write_dtypes[mode].append(output.dtype)

            model.layers[0].moe_layer.residual_connection_mlp.register_forward_hook(
                hook, with_kwargs=True
            )

        record_residual_writes(serial, "serial")
        record_residual_writes(overlap, "overlap")

        def run(model):
            model.zero_grad(set_to_none=True)
            hidden_states = base_hidden.detach().clone().requires_grad_(True)
            output = model(hidden_states, attention_mask=attention_mask)
            output.float().square().mean().backward()
            gradients = {
                name: parameter.grad.detach().float().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            return output.detach().float(), hidden_states.grad.detach().float(), gradients

        serial_output, serial_input_grad, serial_gradients = run(serial)
        overlap_output, overlap_input_grad, overlap_gradients = run(overlap)

        expected_residual_dtype = torch.float32 if fp32_residual_connection else base_hidden.dtype
        assert residual_write_dtypes["serial"]
        assert residual_write_dtypes["overlap"]
        assert all(dtype == expected_residual_dtype for dtype in residual_write_dtypes["serial"])
        assert all(dtype == expected_residual_dtype for dtype in residual_write_dtypes["overlap"])
        if fp32_residual_connection:
            assert serial.layers[0].shortcut_residual_read.read_map.logit.dtype == torch.bfloat16
            assert overlap.layers[0].shortcut_residual_read.read_map.logit.dtype == torch.bfloat16
        torch.testing.assert_close(overlap_output, serial_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(overlap_input_grad, serial_input_grad, rtol=1e-5, atol=1e-6)
        assert overlap_gradients.keys() == serial_gradients.keys()
        for name in serial_gradients:
            torch.testing.assert_close(
                overlap_gradients[name], serial_gradients[name], rtol=1e-5, atol=1e-6
            )

        shortcut = overlap.layers[0]
        shortcut_grad = shortcut.shortcut_residual_read.read_map.logit.grad
        assert shortcut_grad is not None
        assert torch.isfinite(shortcut_grad).all()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
class TestHybridBlock:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def get_pg_collection(self):
        return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])

    def get_hybrid_block(self, layer_pattern, *, stack_spec=hybrid_stack_spec, **config_kwargs):
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            **config_kwargs,
        )
        modules = stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def get_dsa_hybrid_block(self, layer_pattern):
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = MLATransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_type_list),
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            add_bias_linear=False,
        )
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def get_mla_hybrid_block(self, layer_pattern):
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = MLATransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_type_list),
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
        )
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        """Test GPU forward pass."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_hybrid_block(layer_pattern)
        block.cuda()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, block.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        output = block(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == block.config.hidden_size
        assert output.dtype == torch.float32

    def _run_forward(self, block, sequence_length=32, micro_batch_size=2):
        block.cuda()
        block.train()
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, block.config.hidden_size)
        ).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()
        return block(hidden_states, attention_mask=attention_mask)

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "recompute_kwargs",
        [
            dict(recompute_granularity="full", recompute_method="block", recompute_num_layers=2),
            dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=2),
            dict(recompute_granularity="selective", recompute_modules=["core_attn", "mlp"]),
        ],
        ids=["full_block", "full_uniform", "selective"],
    )
    @pytest.mark.parametrize(
        "layer_pattern",
        [
            Symbols.MAMBA * 5,
            Symbols.ATTENTION * 5,
            Symbols.MLP * 5,
            Symbols.ATTENTION + Symbols.MLP + Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP,
            Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP,
        ],
    )
    def test_recompute(self, recompute_kwargs: dict, layer_pattern: str):
        seed = 123
        sequence_length, micro_batch_size = 32, 2

        # When 'mlp' is in recompute_modules, the wrapped MLP's `(out, bias_param)`
        # output triggers a reentrant-backward deadlock in CheckpointFunction.
        # All three in-tree MoE recipes that use `recompute_modules=[..., 'mlp']`
        # set `--disable-bias-linear: true`, so we match that usage pattern here.
        arch_kwargs = {}
        if recompute_kwargs.get(
            "recompute_granularity"
        ) == "selective" and "mlp" in recompute_kwargs.get("recompute_modules", []):
            arch_kwargs["add_bias_linear"] = False

        def build_inputs():
            torch.manual_seed(seed)
            hs = torch.randn(
                (sequence_length, micro_batch_size, 256), device="cuda", requires_grad=True
            )
            am = torch.ones(
                (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool, device="cuda"
            )
            return hs, am

        hs, am = build_inputs()

        def run(block, hs, am):
            out = block(hs, attention_mask=am)
            out.float().sum().backward()
            grads = {
                n: p.grad.detach().float().cpu()
                for n, p in block.named_parameters()
                if p.grad is not None
            }
            return out.detach().float().cpu(), grads

        # --- Baseline (no recompute) ---
        model_parallel_cuda_manual_seed(seed)
        torch.manual_seed(seed)
        base = self.get_hybrid_block(layer_pattern, **arch_kwargs).cuda()
        base.train()
        base_logits, base_grads = run(base, hs, am)
        del base
        torch.cuda.empty_cache()

        # --- Recompute ---
        model_parallel_cuda_manual_seed(seed)
        torch.manual_seed(seed)
        rec = self.get_hybrid_block(layer_pattern, **arch_kwargs, **recompute_kwargs).cuda()
        rec.train()
        rec_logits, rec_grads = run(rec, hs, am)

        # --- Numerical equivalence ---
        assert torch.equal(rec_logits, base_logits), f"Logits should be bitwise matched"
        assert set(rec_grads.keys()) == set(base_grads.keys())
        for name in base_grads:
            gb, gr = base_grads[name], rec_grads[name]
            assert torch.equal(gr, gb), f"Grad should be bitwise matched for {name}"

    def test_layer_types(self):
        """
        Make sure that the layer types specified with layer_pattern
        were honored.
        """
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_hybrid_block(layer_pattern)
        layers = block.layers
        # Note that this matches the order specified by layer_pattern above
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, SelfAttention)
        assert isinstance(layers[2], TransformerLayer)
        assert isinstance(layers[2].mlp, MLP)

    @pytest.mark.parametrize(
        ("compute_symbol", "stack_spec", "compute_config"), TWO_STAGE_ATTENTION_CASES
    )
    def test_two_stage_attention_matches_atomic_forward_bitwise(
        self, compute_symbol, stack_spec, compute_config
    ):
        block = self.get_hybrid_block(
            compute_symbol,
            stack_spec=stack_spec,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            **compute_config,
        ).cuda()
        layer = block.layers[0]
        layer.train()
        assert layer.supports_two_stage_attention()

        hidden_states = torch.randn(16, 2, block.config.hidden_size, device="cuda")
        attention_mask = None
        if compute_symbol == Symbols.ATTENTION:
            attention_mask = torch.triu(
                torch.ones(1, 1, 16, 16, dtype=torch.bool, device="cuda"), diagonal=1
            )

        with torch.no_grad():
            model_parallel_cuda_manual_seed(123)
            atomic_output = layer(hidden_states, attention_mask=attention_mask)

            model_parallel_cuda_manual_seed(123)
            stage_one_state = layer.forward_pre_attn_and_core_attn(
                hidden_states, attention_mask=attention_mask
            )
            two_stage_output = layer.forward_post_core_attn(*stage_one_state)

        def assert_bitwise_equal(actual, expected):
            assert type(actual) is type(expected)
            if isinstance(actual, tuple):
                assert len(actual) == len(expected)
                for actual_item, expected_item in zip(actual, expected):
                    assert_bitwise_equal(actual_item, expected_item)
            elif actual is None:
                assert expected is None
            else:
                assert torch.equal(actual, expected)

        assert_bitwise_equal(two_stage_output, atomic_output)

    @pytest.mark.parametrize(
        ("compute_symbol", "stack_spec", "compute_config"), TWO_STAGE_ATTENTION_CASES
    )
    @pytest.mark.parametrize("parallel", [False, True], ids=["serial", "overlap"])
    @pytest.mark.parametrize(
        ("wide_residual", "residual_replay", "fp32_residual_connection"),
        [(False, False, False), (True, True, False), (True, False, True), (True, True, True)],
        ids=["ordinary", "wide-replay", "wide-fp32", "wide-replay-fp32"],
    )
    def test_shortcut_pair_eager_forward_backward(
        self,
        monkeypatch,
        compute_symbol,
        stack_spec,
        compute_config,
        parallel,
        wide_residual,
        residual_replay,
        fp32_residual_connection,
    ):
        shortcut_config = dict(
            num_moe_experts=1,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="allgather",
            moe_shortcut_connection=True,
            moe_shortcut_parallel=parallel,
            moe_shortcut_post_norm=True,
            moe_shared_expert_intermediate_size=256,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            **compute_config,
        )
        if wide_residual:
            shortcut_config.update(
                bf16=fp32_residual_connection or shortcut_config.get("bf16", False),
                params_dtype=(
                    torch.bfloat16
                    if fp32_residual_connection
                    else shortcut_config.get("params_dtype", torch.float32)
                ),
                fp32_residual_connection=fp32_residual_connection,
                wide_residual=WideResidualConfig(num_streams=3),
            )
        if residual_replay:
            shortcut_config.update(
                recompute_granularity="selective",
                recompute_modules=["residual_stream"],
                residual_stream_recompute_num_layers=1,
            )
        block = self.get_hybrid_block(
            compute_symbol + Symbols.MOE, stack_spec=stack_spec, **shortcut_config
        )

        assert len(block.layers) == 1
        assert block.num_layers_per_pipeline_rank == 2
        shortcut = block.layers[0]
        assert isinstance(shortcut, ShortcutMoEBlock)
        assert shortcut.overlap_mode is parallel
        assert shortcut.compute_layer.supports_two_stage_attention()
        assert isinstance(shortcut.moe_layer, TransformerLayer)
        state_keys = set(block.state_dict())
        assert any(key.startswith("layers.0.compute_layer.") for key in state_keys)
        assert any(key.startswith("layers.0.moe_layer.") for key in state_keys)
        assert any(key.startswith("layers.0.shortcut_pre_mlp_layernorm.") for key in state_keys)
        assert "layers.0.shortcut_post_norm.weight" in state_keys
        shortcut_read_key = "layers.0.shortcut_residual_read.read_map.logit"
        assert (shortcut_read_key in state_keys) is wide_residual

        residual_write_dtypes = []
        if fp32_residual_connection:

            def record_residual_write_dtype(_module, _args, kwargs, output):
                if kwargs.get("operation") == "write":
                    residual_write_dtypes.append(output.dtype)

            shortcut.moe_layer.residual_connection_mlp.register_forward_hook(
                record_residual_write_dtype, with_kwargs=True
            )

        block = block.cuda()
        block.train()

        hidden_states = torch.randn(
            16,
            2,
            block.config.hidden_size,
            device=torch.cuda.current_device(),
            dtype=(block.config.params_dtype if fp32_residual_connection else torch.float32),
            requires_grad=True,
        )
        attention_mask = None
        if compute_symbol == Symbols.ATTENTION:
            attention_mask = torch.triu(
                torch.ones(1, 1, 16, 16, dtype=torch.bool, device=hidden_states.device), diagonal=1
            )
            compute_layer = shortcut.compute_layer

            def fail_if_mlp_runs(*args, **kwargs):
                pytest.fail("attention shortcut output projection must not execute an MLP")

            monkeypatch.setattr(compute_layer, "_forward_mlp", fail_if_mlp_runs)

        output = block(hidden_states, attention_mask=attention_mask)
        output.float().square().mean().backward()

        assert output.shape == hidden_states.shape
        if fp32_residual_connection:
            assert hidden_states.dtype == torch.bfloat16
            assert residual_write_dtypes
            assert all(dtype == torch.float32 for dtype in residual_write_dtypes)
        logical_norms = (
            shortcut.shortcut_pre_mlp_layernorm,
            shortcut.moe_layer.pre_mlp_layernorm,
            shortcut.shortcut_post_norm,
        )
        assert len({id(norm.weight) for norm in logical_norms}) == len(logical_norms)
        for norm in logical_norms:
            assert norm.weight.grad is not None
            assert torch.isfinite(norm.weight.grad).all()

        if wide_residual:
            compute_read = (
                shortcut.compute_layer.residual_connection
                if compute_symbol == Symbols.MAMBA
                else shortcut.compute_layer.residual_connection_self_attn
            )
            independent_reads = (
                shortcut.shortcut_residual_read,
                compute_read,
                shortcut.moe_layer.residual_connection_mlp,
            )
            assert all(read is not None for read in independent_reads)
            assert len({id(read.read_map.logit) for read in independent_reads}) == len(
                independent_reads
            )
            for read in independent_reads:
                read_grad = read.read_map.logit.grad
                assert read_grad is not None
                assert torch.isfinite(read_grad).all()
                assert torch.count_nonzero(read_grad[: block.config.wide_residual.num_streams]) > 0
        else:
            assert shortcut.shortcut_residual_read is None
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()

    def test_shortcut_pair_wide_fp32_replay_matches_eager_forward_backward(self):
        common_config = dict(
            num_moe_experts=1,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="allgather",
            moe_shortcut_connection=True,
            moe_shortcut_parallel=False,
            moe_shortcut_post_norm=True,
            moe_shared_expert_intermediate_size=256,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            fp32_residual_connection=True,
            wide_residual=WideResidualConfig(num_streams=3),
        )
        torch.manual_seed(1234)
        reference = self.get_hybrid_block(Symbols.MAMBA + Symbols.MOE, **common_config).cuda()
        torch.manual_seed(1234)
        recomputed = self.get_hybrid_block(
            Symbols.MAMBA + Symbols.MOE,
            recompute_granularity="selective",
            recompute_modules=["residual_stream"],
            residual_stream_recompute_num_layers=1,
            **common_config,
        ).cuda()
        recomputed.load_state_dict(reference.state_dict())

        reference_input = torch.randn(
            128,
            2,
            reference.config.hidden_size,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        recomputed_input = reference_input.detach().clone().requires_grad_(True)
        read_dtypes = {"eager": [], "replay": []}
        state_dtypes = {"eager": [], "replay": []}
        read_grad_fns = {"eager": [], "replay": []}

        def record_mamba_read(mode):
            def hook(_module, _args, kwargs, output):
                if kwargs.get("operation") == "read":
                    branch_input, state = output
                    read_dtypes[mode].append(branch_input.dtype)
                    state_dtypes[mode].append(state[0].dtype)
                    read_grad_fns[mode].append(type(branch_input.grad_fn).__name__)

            return hook

        reference.layers[0].compute_layer.residual_connection.register_forward_hook(
            record_mamba_read("eager"), with_kwargs=True
        )
        recomputed.layers[0].compute_layer.residual_connection.register_forward_hook(
            record_mamba_read("replay"), with_kwargs=True
        )

        def run(block, hidden_states):
            block.train()
            model_parallel_cuda_manual_seed(4321)
            output = block(hidden_states, attention_mask=None)
            output.float().square().mean().backward()
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in block.named_parameters()
                if parameter.grad is not None
            }
            return output.detach(), gradients

        reference_output, reference_gradients = run(reference, reference_input)
        recomputed_output, recomputed_gradients = run(recomputed, recomputed_input)

        assert read_dtypes["eager"]
        assert read_dtypes["replay"]
        assert all(dtype == torch.bfloat16 for dtype in read_dtypes["eager"])
        assert all(dtype == torch.bfloat16 for dtype in read_dtypes["replay"])
        assert all(dtype == torch.float32 for dtype in state_dtypes["eager"])
        assert all(dtype == torch.float32 for dtype in state_dtypes["replay"])
        assert all(name == "_StreamwiseSigmoidReadBackward" for name in read_grad_fns["eager"])
        assert "_StreamwiseSigmoidReadBackward" in read_grad_fns["replay"]
        assert len(read_dtypes["replay"]) > len(read_dtypes["eager"])
        # FP32 applies to the internal residual state; final normalization returns params_dtype
        # after the residual readout.
        assert reference_output.dtype == reference.config.params_dtype
        assert recomputed_output.dtype == recomputed.config.params_dtype
        torch.testing.assert_close(recomputed_output, reference_output)
        torch.testing.assert_close(recomputed_input.grad, reference_input.grad)
        assert recomputed_gradients.keys() == reference_gradients.keys()
        for name in reference_gradients:
            torch.testing.assert_close(recomputed_gradients[name], reference_gradients[name])

    def test_invalid_layer_types_cause_failure(self):
        invalid_symbol = 'X'
        assert invalid_symbol not in Symbols.VALID_LAYERS  # sanity check.
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP + invalid_symbol
        # validate_segment_layers() in hybrid_layer_allocation.py throws a ValueError.
        with pytest.raises(ValueError):
            block = self.get_hybrid_block(layer_pattern)

    def test_gdn_layer_types(self):
        """
        Make sure that G creates a TransformerLayer wrapping GatedDeltaNet,
        while * creates a TransformerLayer wrapping SelfAttention.
        """
        layer_pattern = Symbols.GDN + Symbols.ATTENTION + Symbols.MAMBA
        block = self.get_hybrid_block(layer_pattern)
        layers = block.layers
        assert isinstance(layers[0], TransformerLayer)
        assert isinstance(layers[0].self_attention, GatedDeltaNet)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, SelfAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_gdn_inference_spec(self):
        """The inference stack must materialize GDN rather than its IdentityOp default."""
        gdn_spec = hybrid_inference_stack_spec.submodules.gdn_layer
        assert gdn_spec.module is TransformerLayer
        assert gdn_spec.submodules.self_attention.module is GatedDeltaNet

    def test_gdn_gpu_forward(self):
        """Test GPU forward pass with GDN, attention, and Mamba layers."""
        layer_pattern = Symbols.GDN + Symbols.ATTENTION + Symbols.MAMBA
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
        )
        modules = hybrid_stack_spec.submodules
        block = HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )
        block.cuda()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, block.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        output = block(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == block.config.hidden_size
        assert output.dtype == torch.float32

    def test_dsa_layer_types(self):
        """D symbol creates a TransformerLayer with absorbed MLA and DSA core attention."""
        layer_pattern = Symbols.MAMBA + Symbols.DS_ATTENTION + Symbols.MAMBA
        block = self.get_dsa_hybrid_block(layer_pattern)
        layers = block.layers
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, AbsorbedMLASelfAttention)
        assert isinstance(layers[1].self_attention.core_attention, DSAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_mixed_attention_and_dsa_layer_types(self):
        """* and D in the same block fail."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.DS_ATTENTION + Symbols.MAMBA
        with pytest.raises(ValueError):
            block = self.get_dsa_hybrid_block(layer_pattern)

    def test_mla_layer_types(self):
        """+ symbol creates a TransformerLayer with MLASelfAttention but
        standard (non-DSA) core attention."""
        layer_pattern = Symbols.MAMBA + Symbols.MLA + Symbols.MAMBA
        block = self.get_mla_hybrid_block(layer_pattern)
        layers = block.layers
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, MLASelfAttention)
        assert isinstance(layers[1].self_attention.core_attention, TEDotProductAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_mixed_attention_and_mla_layer_types(self):
        """* and + in the same block fail (same reason as * and D)."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLA + Symbols.MAMBA
        with pytest.raises(ValueError):
            block = self.get_mla_hybrid_block(layer_pattern)
