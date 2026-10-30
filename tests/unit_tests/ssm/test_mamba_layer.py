# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import replace

import pytest
import torch

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import TwoStageAttentionLayer
from megatron.core.transformer.residual_recompute import build_residual_stream_recompute_plan
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_config import WideResidualConfig
from megatron.core.transformer.wide_residual_layer import specialize_wide_residual_layer_spec
from tests.unit_tests.test_utilities import Utils


class _TwoStageMixer(torch.nn.Module, TwoStageAttentionLayer):
    """Small sequence mixer for two-stage wide-residual tests."""

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.input_projection = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.output_projection = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward_pre_attn_and_core_attn(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return torch.tanh(self.input_projection(hidden_states))

    def forward_post_core_attn(self, mixer_intermediate):
        return self.output_projection(mixer_intermediate), None

    def forward(self, hidden_states, *args, **kwargs):
        return self.forward_post_core_attn(
            self.forward_pre_attn_and_core_attn(hidden_states, *args, **kwargs)
        )


@pytest.mark.internal
class TestMambaLayer:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=1,
            num_attention_heads=1,
            layernorm_epsilon=1e-6,
            use_cpu_initialization=True,
        )
        assert isinstance(hybrid_stack_spec.submodules, HybridStackSubmodules)
        assert isinstance(hybrid_stack_spec.submodules.mamba_layer.submodules, MambaLayerSubmodules)
        # Use an explicit norm so the test can verify the configured epsilon.
        mamba_submodules = replace(
            hybrid_stack_spec.submodules.mamba_layer.submodules, norm=WrappedTorchNorm
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.layer = MambaLayer(transformer_config, mamba_submodules, pg_collection=pg_collection)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_configured_layernorm_epsilon(self):
        assert self.layer.norm.eps == self.layer.config.layernorm_epsilon

    def test_gpu_forward(self):
        layer = self.layer
        layer.cuda()
        layer.eval()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == layer.config.hidden_size
        assert output.dtype == torch.float32

    def test_wide_residual_two_stage_attention_matches_atomic_with_replay(self):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            hidden_dropout=0.0,
            bias_dropout_fusion=False,
            use_cpu_initialization=True,
            wide_residual=WideResidualConfig(num_streams=3),
        )
        layer_spec = specialize_wide_residual_layer_spec(
            ModuleSpec(
                module=MambaLayer,
                submodules=MambaLayerSubmodules(
                    norm=WrappedTorchNorm, mixer=_TwoStageMixer, mamba_bda=get_bias_dropout_add
                ),
            ),
            config,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        torch.manual_seed(1234)
        atomic_layer = build_module(
            layer_spec, config=config, layer_number=1, pg_collection=pg_collection
        ).cuda()
        two_stage_layer = build_module(
            layer_spec, config=config, layer_number=1, pg_collection=pg_collection
        ).cuda()
        two_stage_layer.load_state_dict(atomic_layer.state_dict())

        atomic_input = torch.randn(4, 2, 3 * config.hidden_size, device="cuda", requires_grad=True)
        two_stage_input = atomic_input.detach().clone().requires_grad_(True)
        atomic_output = atomic_layer(hidden_states=atomic_input)

        replay_context, replay_terminal = build_residual_stream_recompute_plan(2, 2)
        two_stage_state = two_stage_layer.forward_pre_attn_and_core_attn(
            hidden_states=two_stage_input, residual_stream_recompute_context=replay_context
        )
        assert len(two_stage_state) == 3
        assert isinstance(two_stage_state[2], tuple) and two_stage_state[2]
        two_stage_output = two_stage_layer.forward_post_core_attn(
            *two_stage_state,
            inference_context=None,
            padding_mask=None,
            residual_stream_recompute_context=replay_context,
        )

        assert atomic_layer.supports_two_stage_attention()
        assert two_stage_layer.supports_two_stage_attention()
        assert len(replay_context.manager.checkpoints) == 3
        torch.testing.assert_close(two_stage_output, atomic_output)

        # Model the later shared/MoE work that forms the actual replay boundary.
        two_stage_boundary = two_stage_output + 0.0
        replay_terminal.finalize(two_stage_boundary)
        atomic_output.square().mean().backward()
        two_stage_boundary.square().mean().backward()
        torch.testing.assert_close(two_stage_input.grad, atomic_input.grad)
        atomic_gradients = {
            name: parameter.grad
            for name, parameter in atomic_layer.named_parameters()
            if parameter.grad is not None
        }
        two_stage_gradients = {
            name: parameter.grad
            for name, parameter in two_stage_layer.named_parameters()
            if parameter.grad is not None
        }
        assert two_stage_gradients.keys() == atomic_gradients.keys()
        for name in atomic_gradients:
            torch.testing.assert_close(two_stage_gradients[name], atomic_gradients[name])
