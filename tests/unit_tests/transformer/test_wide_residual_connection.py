# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for streamwise wide-residual branch connections."""

import pytest
import torch
from torch import nn

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.residual_connection import build_residual_read
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.streamwise_residual_ops import streamwise_sigmoid_writeback
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from megatron.core.transformer.transformer_layer import (
    MoETransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.wide_residual_layer import (
    LearnedWideResidualRetention,
    StreamwiseSigmoidResidualReadout,
    StreamwiseSigmoidWideResidualConnection,
    StreamwiseSigmoidWideResidualRead,
    expand_wide_residual_stream,
    specialize_wide_residual_layer_spec,
)
from tests.unit_tests.test_utilities import Utils


def _wide_config(
    *, init_scale: float = 0.0, learned_retention: bool = False, **config_overrides
) -> TransformerConfig:
    values = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        hidden_dropout=0.0,
        bias_dropout_fusion=False,
        use_cpu_initialization=True,
        wide_residual=WideResidualConfig(
            num_streams=3,
            streamwise_sigmoid_init_scale=init_scale,
            learned_retention=learned_retention,
            retention_init=0.999,
            retention_max_forget=0.10,
        ),
    )
    values.update(config_overrides)
    return TransformerConfig(**values)


def _process_groups() -> ProcessGroupCollection:
    return ProcessGroupCollection()


class _ActiveBranch(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return hidden_states


class _AdditiveAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.last_hidden_size = None

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        self.last_hidden_size = hidden_states.shape[-1]
        bias = torch.ones(
            hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        return 2.0 * hidden_states, bias


class _AdditiveMLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.last_hidden_size = None

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        self.last_hidden_size = hidden_states.shape[-1]
        return 3.0 * hidden_states, None


class _ParameterizedAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.scale = nn.Parameter(torch.randn(()))

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return self.scale * hidden_states, None


class _ParameterizedMLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.scale = nn.Parameter(torch.randn(()))

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return self.scale * hidden_states, None


class _ResidualReturningNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.returns_residual = True

    def forward(self, hidden_states):
        return hidden_states, hidden_states


@pytest.mark.parametrize("num_streams", [True, 1, 0, -1])
def test_wide_residual_config_rejects_invalid_num_streams(num_streams):
    error = TypeError if isinstance(num_streams, bool) else ValueError
    with pytest.raises(error):
        WideResidualConfig(num_streams=num_streams)


@pytest.mark.parametrize(
    ("retention_init", "max_forget"), [(0.999, 0.0), (0.999, 1.0), (0.90, 0.10), (1.0, 0.10)]
)
def test_wide_residual_config_validates_bounded_retention(retention_init, max_forget):
    with pytest.raises(ValueError):
        WideResidualConfig(
            num_streams=3,
            learned_retention=True,
            retention_init=retention_init,
            retention_max_forget=max_forget,
        )


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        ({"enable_mhc_connections": True}, "mutually exclusive"),
        ({"inference_fuse_tp_communication": True}, "fuse_tp_communication"),
        ({"heterogeneous_block_specs": True}, "heterogeneous_block_specs"),
        (
            {"num_layers": 2, "pipeline_model_parallel_size": 2, "pipeline_dtype": torch.bfloat16},
            "pipeline_model_parallel_size",
        ),
    ],
)
def test_transformer_config_rejects_unsupported_wide_residual_modes(override, expected_error):
    with pytest.raises((ValueError, NotImplementedError), match=expected_error):
        _wide_config(**override)


def test_transformer_config_accepts_mtp_with_wide_residual_replay():
    config = _wide_config(
        mtp_num_layers=2,
        recompute_granularity="selective",
        recompute_modules=["residual_stream"],
        residual_stream_recompute_num_layers=1,
    )

    assert config.mtp_num_layers == 2
    assert config.residual_stream_recompute_num_layers == 1


def test_transformer_config_accepts_fp32_wide_residual_stream():
    config = _wide_config(
        bf16=True,
        fp32_residual_connection=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )

    assert config.fp32_residual_connection
    assert config.pipeline_dtype == torch.float32


class TestStreamwiseSigmoidWideResidualConnection:
    def test_fp32_state_promotion_aliases_an_already_fp32_stream(self):
        """The defensive FP32 promotion must not allocate for normal FP32 model ingress."""

        config = _wide_config(fp32_residual_connection=True)
        connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        residual_stream = torch.randn(
            2, connection.residual_stream_hidden_size, dtype=torch.float32
        )

        _, state = connection(residual_stream, operation="read", fp32_residual_connection=True)

        assert connection.residual_stream(state).dtype == torch.float32
        assert connection.residual_stream(state) is residual_stream
        assert connection.residual_stream(state).data_ptr() == residual_stream.data_ptr()

    def test_read_output_dtype_does_not_change_fp32_connection_state(self):
        config = _wide_config(fp32_residual_connection=True)
        connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        residual_stream = torch.randn(
            2, connection.residual_stream_hidden_size, dtype=torch.float32
        )

        branch_input, state = connection(
            residual_stream,
            operation="read",
            fp32_residual_connection=True,
            branch_input_dtype=torch.bfloat16,
        )

        assert branch_input.dtype == torch.bfloat16
        assert connection.residual_stream(state) is residual_stream

    @pytest.mark.parametrize(
        ("has_bias", "dropout_probability", "training", "expected_update_dtype"),
        [
            (False, 0.0, True, torch.bfloat16),
            (False, 0.1, False, torch.bfloat16),
            (True, 0.0, True, torch.float32),
            (False, 0.1, True, torch.float32),
            (True, 0.1, True, torch.float32),
        ],
        ids=(
            "no_bias_no_dropout",
            "eval_dropout",
            "bias",
            "active_dropout",
            "bias_and_active_dropout",
        ),
    )
    def test_write_defers_update_cast_only_when_bias_and_dropout_are_inactive(
        self, monkeypatch, has_bias, dropout_probability, training, expected_update_dtype
    ):
        config = _wide_config(fp32_residual_connection=True)
        connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        residual_stream = torch.randn(
            2, connection.residual_stream_hidden_size, dtype=torch.float32
        )
        branch_update = torch.randn(2, connection.branch_hidden_size, dtype=torch.bfloat16)
        bias = (
            torch.randn(connection.branch_hidden_size, dtype=torch.bfloat16) if has_bias else None
        )
        observed = {}

        def record_writeback(residual, update, *args, **kwargs):
            del args, kwargs
            observed["update_dtype"] = update.dtype
            observed["update"] = update
            return residual

        def preserve_dropout_input(update, *args, **kwargs):
            del args, kwargs
            observed["dropout_input"] = update.detach().clone()
            return update

        monkeypatch.setattr(
            "megatron.core.transformer.wide_residual_layer.streamwise_sigmoid_writeback",
            record_writeback,
        )
        monkeypatch.setattr(
            "megatron.core.transformer.wide_residual_layer.F.dropout", preserve_dropout_input
        )
        connection(
            (branch_update, bias),
            operation="write",
            state=(residual_stream,),
            dropout_probability=dropout_probability,
            training=training,
        )

        assert observed["update_dtype"] == expected_update_dtype
        if expected_update_dtype == torch.bfloat16:
            assert observed["update"] is branch_update
            assert observed["update"].data_ptr() == branch_update.data_ptr()
        else:
            expected_dropout_input = branch_update.float()
            if bias is not None:
                expected_dropout_input = expected_dropout_input + bias.float()
            assert torch.equal(observed["dropout_input"], expected_dropout_input)

    def test_bias_and_dropout_keep_explicit_fp32_preprocessing_semantics(self):
        """The guarded path retains the original FP32 bias/dropout graph and gradients."""

        torch.manual_seed(8765)
        config = _wide_config(fp32_residual_connection=True, init_scale=0.01)
        connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        residual = torch.randn(
            4, connection.residual_stream_hidden_size, dtype=torch.float32, requires_grad=True
        )
        update = torch.randn(
            4, connection.branch_hidden_size, dtype=torch.bfloat16, requires_grad=True
        )
        bias = torch.randn(connection.branch_hidden_size, dtype=torch.bfloat16, requires_grad=True)
        reference_residual = residual.detach().clone().requires_grad_()
        reference_update = update.detach().clone().requires_grad_()
        reference_bias = bias.detach().clone().requires_grad_()
        reference_logits = connection.write_map.logit.detach().clone().requires_grad_()
        grad_output = torch.randn_like(residual)

        torch.manual_seed(4321)
        output = connection(
            (update, bias),
            operation="write",
            state=(residual,),
            dropout_probability=0.25,
            training=True,
        )
        torch.manual_seed(4321)
        reference_preprocessed = torch.nn.functional.dropout(
            reference_update.float() + reference_bias.float(), p=0.25, training=True
        )
        reference = streamwise_sigmoid_writeback(
            reference_residual, reference_preprocessed, reference_logits, connection.num_streams
        )
        gradients = torch.autograd.grad(
            output, (residual, update, bias, connection.write_map.logit), grad_output
        )
        reference_gradients = torch.autograd.grad(
            reference,
            (reference_residual, reference_update, reference_bias, reference_logits),
            grad_output,
        )

        assert torch.equal(output, reference)
        for gradient, reference_gradient in zip(gradients, reference_gradients):
            assert gradient.dtype == reference_gradient.dtype
            assert torch.equal(gradient, reference_gradient)

    def test_initial_read_write_and_readout_preserve_base_stream(self):
        config = _wide_config(init_scale=0.0)
        connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        readout = StreamwiseSigmoidResidualReadout(config)
        base = torch.randn(2, config.hidden_size)
        residual_stream = expand_wide_residual_stream(base, 3)

        branch_input, state = connection(residual_stream, operation="read")
        branch_update = torch.randn_like(base)
        output = connection(
            branch_update, operation="write", state=state, dropout_probability=0.0, training=False
        )

        assert torch.allclose(branch_input, base)
        assert torch.allclose(output, expand_wide_residual_stream(base + branch_update, 3))
        assert torch.allclose(readout(output), base + branch_update)

    @pytest.mark.parametrize("sequence_parallel", [False, True])
    def test_controllers_have_replicated_tp_gradient_metadata(self, sequence_parallel):
        config = _wide_config(learned_retention=True)
        config.sequence_parallel = sequence_parallel
        connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )

        parameters = (
            connection.read_map.logit,
            connection.write_map.logit,
            connection.retention.retention_logit,
        )
        for parameter in parameters:
            assert parameter.allreduce
            assert not parameter.tensor_model_parallel
            assert parameter.sequence_parallel == sequence_parallel
            assert parameter.average_gradients_across_tp_domain is not sequence_parallel
        assert connection.retention.retention_logit.is_wide_residual_retention_parameter

    def test_nested_module_hooks_run_before_controller_access(self):
        config = _wide_config(learned_retention=True)
        connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        operations = []
        connection.read_map.register_forward_pre_hook(lambda *_: operations.append("read"))
        connection.write_map.register_forward_pre_hook(lambda *_: operations.append("write"))
        connection.retention.register_forward_pre_hook(lambda *_: operations.append("retention"))

        residual_stream = torch.randn(2, 3 * config.hidden_size)
        branch_input, state = connection(residual_stream, operation="read")
        connection(
            branch_input, operation="write", state=state, dropout_probability=0.0, training=False
        )

        assert operations == ["read", "retention", "write"]

    def test_gradients_reach_active_controllers_but_not_padding(self):
        config = _wide_config(init_scale=0.01, learned_retention=True)
        connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        residual_stream = torch.randn(4, 3 * config.hidden_size, requires_grad=True)
        branch_input, state = connection(residual_stream, operation="read")
        output = connection(
            branch_input.square(),
            operation="write",
            state=state,
            dropout_probability=0.0,
            training=True,
        )
        output.square().mean().backward()

        parameters = (
            connection.read_map.logit,
            connection.write_map.logit,
            connection.retention.retention_logit,
        )
        for parameter in parameters:
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad[: connection.num_streams]) > 0
            assert torch.count_nonzero(parameter.grad[connection.num_streams :]) == 0

    def test_fp32_residual_multi_layer_forward_backward_uses_fp32_carry(self):
        config = _wide_config(
            init_scale=0.01, learned_retention=True, fp32_residual_connection=True
        )
        connections = nn.ModuleList(
            [
                StreamwiseSigmoidWideResidualConnection(
                    config=config,
                    layer_number=layer_number,
                    branch_name="test",
                    pg_collection=_process_groups(),
                )
                for layer_number in (1, 2)
            ]
        )
        base = torch.randn(4, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
        hidden_states = expand_wide_residual_stream(base, config.wide_residual.num_streams)

        branch_dtypes = []
        for connection in connections:
            branch_input, state = connection(
                hidden_states,
                operation="read",
                fp32_residual_connection=config.fp32_residual_connection,
            )
            branch_dtypes.append(branch_input.dtype)
            assert connection.residual_stream(state).dtype == torch.float32
            hidden_states = connection(
                branch_input.square(),
                operation="write",
                state=state,
                dropout_probability=0.0,
                training=True,
            )
            assert hidden_states.dtype == torch.float32

        assert branch_dtypes == [torch.bfloat16, torch.float32]
        hidden_states.square().mean().backward()

        assert base.grad is not None
        assert torch.isfinite(base.grad).all()
        for connection in connections:
            for parameter in connection.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()

    def test_checkpoint_round_trip_is_strict(self):
        config = _wide_config(learned_retention=True)
        source = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        destination = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        with torch.no_grad():
            source.read_map.logit[0].add_(0.25)
            source.write_map.logit[1].sub_(0.125)
            source.retention.retention_logit[2].add_(0.5)

        state = source.state_dict()
        destination.load_state_dict(state, strict=True)
        assert set(state) == {"read_map.logit", "write_map.logit", "retention.retention_logit"}
        for name, value in state.items():
            assert torch.equal(destination.state_dict()[name], value)

    def test_retention_initializes_to_requested_factor(self):
        config = _wide_config(learned_retention=True)
        retention = LearnedWideResidualRetention(
            config, layer_number=1, branch_name="test", num_streams=3
        )

        assert torch.allclose(retention(), torch.full((3,), 0.999), atol=1.0e-7)


class TestStreamwiseSigmoidWideResidualRead:
    @staticmethod
    def _read(config: TransformerConfig) -> StreamwiseSigmoidWideResidualRead:
        return StreamwiseSigmoidWideResidualRead(
            config=config,
            layer_number=2,
            branch_name="shortcut_routed",
            pg_collection=_process_groups(),
        )

    def test_initial_read_preserves_replicated_base_stream(self):
        config = _wide_config()
        residual_read = self._read(config)
        base = torch.randn(2, 3, config.hidden_size)

        shortcut_hidden = residual_read(expand_wide_residual_stream(base, 3))

        assert shortcut_hidden.shape == base.shape
        assert torch.allclose(shortcut_hidden, base)

    def test_build_helper_constructs_declared_read(self):
        config = _wide_config()

        residual_read = build_residual_read(
            ModuleSpec(module=StreamwiseSigmoidWideResidualRead),
            config=config,
            layer_number=2,
            branch_name="shortcut_routed",
            pg_collection=_process_groups(),
            name=None,
        )

        assert isinstance(residual_read, StreamwiseSigmoidWideResidualRead)
        assert residual_read.residual_stream_hidden_size == 3 * config.hidden_size
        assert residual_read.branch_hidden_size == config.hidden_size

    def test_owns_only_independent_read_control_with_live_gradients(self):
        config = _wide_config()
        residual_read = self._read(config)
        residual_stream = torch.randn(
            3, 2, config.wide_residual.num_streams * config.hidden_size, requires_grad=True
        )

        shortcut_hidden = residual_read(residual_stream)
        shortcut_hidden.square().mean().backward()

        assert set(dict(residual_read.named_parameters())) == {"read_map.logit"}
        assert not hasattr(residual_read, "write_map")
        assert not hasattr(residual_read, "retention")
        assert residual_read.read_map.logit.grad is not None
        assert (
            torch.count_nonzero(
                residual_read.read_map.logit.grad[: config.wide_residual.num_streams]
            )
            > 0
        )
        assert (
            torch.count_nonzero(
                residual_read.read_map.logit.grad[config.wide_residual.num_streams :]
            )
            == 0
        )
        assert residual_stream.grad is not None

    @pytest.mark.parametrize("sequence_parallel", [False, True])
    def test_read_control_has_replicated_tp_gradient_metadata(self, sequence_parallel):
        config = _wide_config()
        config.sequence_parallel = sequence_parallel
        parameter = self._read(config).read_map.logit

        assert parameter.is_residual_map_parameter
        assert parameter.use_muon is False
        assert parameter.allreduce
        assert not parameter.tensor_model_parallel
        assert parameter.sequence_parallel == sequence_parallel
        assert parameter.average_gradients_across_tp_domain is not sequence_parallel

    def test_checkpoint_round_trip_is_strict_and_read_only(self):
        config = _wide_config()
        source = self._read(config)
        destination = self._read(config)
        with torch.no_grad():
            source.read_map.logit[1].add_(0.25)

        state = source.state_dict()
        destination.load_state_dict(state, strict=True)

        assert set(state) == {"read_map.logit"}
        assert torch.equal(destination.read_map.logit, source.read_map.logit)


class TestWideResidualSpecSpecialization:
    def test_disabled_config_returns_original_spec(self):
        config = TransformerConfig(
            num_layers=1, hidden_size=8, num_attention_heads=2, use_cpu_initialization=True
        )
        original = ModuleSpec(module=TransformerLayer, submodules=TransformerLayerSubmodules())

        assert specialize_wide_residual_layer_spec(original, config) is original

    def test_active_branches_receive_independent_connections(self):
        original = ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=_ActiveBranch, cross_attention=IdentityOp, mlp=_ActiveBranch
            ),
        )
        specialized = specialize_wide_residual_layer_spec(original, _wide_config())

        assert original.submodules.residual_connection_self_attn is None
        assert original.submodules.residual_connection_mlp is None
        assert (
            specialized.submodules.residual_connection_self_attn.module
            is StreamwiseSigmoidWideResidualConnection
        )
        assert (
            specialized.submodules.residual_connection_mlp.module
            is StreamwiseSigmoidWideResidualConnection
        )
        assert (
            specialized.submodules.residual_connection_self_attn
            is not specialized.submodules.residual_connection_mlp
        )

    def test_cross_attention_is_rejected_explicitly(self):
        original = ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=_ActiveBranch, cross_attention=_ActiveBranch, mlp=_ActiveBranch
            ),
        )

        with pytest.raises(NotImplementedError, match="cross-attention"):
            specialize_wide_residual_layer_spec(original, _wide_config())

    def test_moe_uses_one_outer_branch_connection(self):
        original = ModuleSpec(
            module=MoETransformerLayer,
            submodules=TransformerLayerSubmodules(self_attention=IdentityOp, mlp=MoELayer),
        )
        specialized = specialize_wide_residual_layer_spec(original, _wide_config())

        assert specialized.submodules.residual_connection_self_attn is None
        assert (
            specialized.submodules.residual_connection_mlp.module
            is StreamwiseSigmoidWideResidualConnection
        )


class TestWideResidualLayerIntegration:
    @staticmethod
    def _layer_spec() -> ModuleSpec:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=_AdditiveAttention,
                self_attn_bda=get_bias_dropout_add,
                mlp=_AdditiveMLP,
                mlp_bda=get_bias_dropout_add,
            ),
        )

    def test_disabled_path_preserves_ordinary_transformer_behavior(self):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            hidden_dropout=0.0,
            bias_dropout_fusion=False,
            use_cpu_initialization=True,
        )
        layer = build_module(
            self._layer_spec(),
            config=config,
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        )
        hidden_states = torch.randn(2, 3, config.hidden_size, requires_grad=True)

        output, context = layer(hidden_states=hidden_states, attention_mask=None)

        assert context is None
        assert layer.residual_connection_self_attn is None
        assert layer.residual_connection_mlp is None
        assert torch.allclose(output, 12.0 * hidden_states + 4.0)

    @pytest.mark.parametrize(
        ("norm_field", "expected_error"),
        [
            ("input_layernorm", "self-attention residual connection"),
            ("pre_mlp_layernorm", "MLP residual connection"),
        ],
    )
    def test_residual_returning_norm_is_rejected_at_construction(self, norm_field, expected_error):
        config = _wide_config()
        layer_spec = self._layer_spec()
        setattr(layer_spec.submodules, norm_field, _ResidualReturningNorm)

        with pytest.raises(ValueError, match=expected_error):
            build_module(
                specialize_wide_residual_layer_spec(layer_spec, config),
                config=config,
                layer_number=1,
                add_layer_offset=False,
                pg_collection=_process_groups(),
            )

    def test_connection_construction_preserves_base_parameter_rng_order(self):
        base_config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            hidden_dropout=0.0,
            use_cpu_initialization=True,
        )
        wide_config = _wide_config()
        layer_spec = ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=_ParameterizedAttention,
                self_attn_bda=get_bias_dropout_add,
                mlp=_ParameterizedMLP,
                mlp_bda=get_bias_dropout_add,
            ),
        )

        torch.manual_seed(1234)
        baseline = build_module(
            layer_spec,
            config=base_config,
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        )
        torch.manual_seed(1234)
        wide = build_module(
            specialize_wide_residual_layer_spec(layer_spec, wide_config),
            config=wide_config,
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        )

        assert torch.equal(wide.self_attention.scale, baseline.self_attention.scale)
        assert torch.equal(wide.mlp.scale, baseline.mlp.scale)

    def test_layer_carries_wide_stream_while_branches_remain_ordinary_width(self):
        config = _wide_config(init_scale=0.0)
        layer = build_module(
            specialize_wide_residual_layer_spec(self._layer_spec(), config),
            config=config,
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        )
        base = torch.randn(2, 3, config.hidden_size, requires_grad=True)
        residual_stream = expand_wide_residual_stream(base, 3)

        output, context = layer(hidden_states=residual_stream, attention_mask=None)

        expected = expand_wide_residual_stream(12.0 * base + 4.0, 3)
        assert context is None
        assert output.shape[-1] == 3 * config.hidden_size
        assert torch.allclose(output, expected)
        assert layer.self_attention.last_hidden_size == config.hidden_size
        assert layer.mlp.last_hidden_size == config.hidden_size

        output.sum().backward()
        connection_parameters = {
            name: parameter
            for name, parameter in layer.named_parameters()
            if "residual_connection" in name
        }
        assert connection_parameters
        assert all(parameter.grad is not None for parameter in connection_parameters.values())

        config.cuda_graph_modules = [CudaGraphModule.attn, CudaGraphModule.mlp]
        graph_submodules = layer._get_submodules_under_cudagraphs()
        assert layer.residual_connection_self_attn in graph_submodules
        assert layer.residual_connection_mlp in graph_submodules

    def test_transformer_block_expands_and_reads_out_the_wide_stream(self):
        config = _wide_config(init_scale=0.01)
        block = TransformerBlock(
            config, self._layer_spec(), post_layer_norm=False, pg_collection=_process_groups()
        )
        hidden_states = torch.randn(2, 3, config.hidden_size, requires_grad=True)

        output = block(hidden_states=hidden_states, attention_mask=None)

        assert output.shape == hidden_states.shape
        assert torch.allclose(output, 12.0 * hidden_states + 4.0)
        assert block.residual_stream_readout is not None
        assert block.layers[0].residual_stream_hidden_size == 3 * config.hidden_size

        output.sum().backward()
        assert block.residual_stream_readout.exit_map.logit.grad is not None

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_recompute_preserves_wide_block_forward_and_backward(self):
        Utils.initialize_model_parallel(1, 1)
        try:
            model_parallel_cuda_manual_seed(123)
            config = _wide_config(
                init_scale=0.01,
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
            )
            block = TransformerBlock(
                config,
                self._layer_spec(),
                post_layer_norm=False,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            ).cuda()
            hidden_states = torch.randn(2, 3, config.hidden_size, device="cuda", requires_grad=True)

            output = block(hidden_states=hidden_states, attention_mask=None)

            assert torch.allclose(output, 12.0 * hidden_states + 4.0)
            output.sum().backward()
            assert hidden_states.grad is not None
            assert block.residual_stream_readout.exit_map.logit.grad is not None
        finally:
            Utils.destroy_model_parallel()
