# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import contextlib
import dataclasses
from typing import cast

import pytest
import torch

import megatron.core.transformer.moe.router as router_mod
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_observation import capture_tensor_observations
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import (
    get_updated_expert_bias,
    router_gating_linear,
    topk_routing_with_score_function,
)
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.moe.router_diagnostics import (
    ROUTER_DIAGNOSTIC_CHANNEL_COUNT,
    RouterDiagnosticChannel,
    build_router_diagnostics,
)
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

try:
    # Check availability of TE fused router ops
    from megatron.core.extensions.transformer_engine import (
        fused_topk_with_score_function as _fused_topk_with_score_function,
    )

    HAVE_ROUTER_FUSION = _fused_topk_with_score_function is not None
except Exception:  # pragma: no cover - defensive
    HAVE_ROUTER_FUSION = False


class _ProcessGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class TestTop2Router:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")
        num_moe_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=num_moe_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        self.sequential_mlp = MoELayer(self.transformer_config, submodules)
        self.router = cast(Router, self.sequential_mlp.router)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert num_weights == 12 * 4, num_weights

    @pytest.mark.internal
    def test_skip_muon(self):
        assert getattr(self.router.weight, 'use_muon', True)

        self.transformer_config.moe_router_skip_muon = True
        self.transformer_config.add_bias_linear = True
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False).mlp
        )
        router = cast(Router, MoELayer(self.transformer_config, submodules).router)

        assert getattr(router.weight, 'use_muon', True) is False
        assert router.bias is not None
        assert getattr(router.bias, 'use_muon', True) is False

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("moe_router_pre_softmax", [(True), (False)])
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward(self, moe_router_pre_softmax, score_function):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.config.moe_router_pre_softmax = moe_router_pre_softmax
            self.router.config.moe_router_score_function = score_function
            # [num tokens, hidden size]
            hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()
            scores, indices = self.router(hidden_states)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_forward_observes_compact_diagnostics_when_aux_loss_is_disabled(
        self, monkeypatch
    ):
        self.router = self.router.cuda()
        # A complete sequence may still be marked sequence-parallel when TP has only one rank.
        # The compact diagnostics have batch as dimension zero and therefore remain replicated.
        self.router.config.sequence_parallel = True
        self.router.tp_group = _ProcessGroup(1)
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()
        observed = []
        score_grad_modes = []
        compute_scores = router_mod.compute_routing_scores_for_aux_loss

        def record_score_grad_mode(*args, **kwargs):
            score_grad_modes.append(torch.is_grad_enabled())
            return compute_scores(*args, **kwargs)

        monkeypatch.setattr(
            router_mod, "compute_routing_scores_for_aux_loss", record_score_grad_mode
        )

        with capture_tensor_observations(
            lambda *args: observed.append(args), frozenset({"router_diagnostics"})
        ):
            self.router(hidden_states)

        assert len(observed) == 1
        assert score_grad_modes == [False]
        owner, name, source_kind, diagnostics, tp_shard_dim = observed[0]
        assert owner is self.router
        assert name == source_kind == "router_diagnostics"
        assert diagnostics.shape == (2, ROUTER_DIAGNOSTIC_CHANNEL_COUNT, 4)
        torch.testing.assert_close(
            diagnostics[:, RouterDiagnosticChannel.VALID_TOKEN_COUNT, 0],
            torch.full((2,), 32.0, device="cuda"),
        )
        torch.testing.assert_close(
            diagnostics[:, RouterDiagnosticChannel.AUX_ACTUAL_OVERLAP, 0],
            torch.ones(2, device="cuda"),
        )
        assert tp_shard_dim is None

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        ("sequence_parallel", "tp_size", "cp_size", "unsupported_axis"),
        ((True, 2, 2, "context"), (False, 1, 2, "context")),
    )
    def test_router_forward_rejects_diagnostics_for_partitioned_sequences(
        self, sequence_parallel, tp_size, cp_size, unsupported_axis
    ):
        self.router = self.router.cuda()
        self.router.config.sequence_parallel = sequence_parallel
        self.router.tp_group = _ProcessGroup(tp_size)
        self.router.cp_group = _ProcessGroup(cp_size)
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()

        with capture_tensor_observations(lambda *args: None, frozenset({"router_diagnostics"})):
            with pytest.raises(
                NotImplementedError,
                match=rf"complete local sequences.*{unsupported_axis} parallel ranks",
            ):
                self.router(hidden_states)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_forward_validates_diagnostics_during_no_grad_forward(self):
        self.router = self.router.cuda()
        self.router.config.sequence_parallel = True
        self.router.tp_group = _ProcessGroup(2)
        self.router.cp_group = _ProcessGroup(2)
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()

        with (
            capture_tensor_observations(lambda *args: None, frozenset({"router_diagnostics"})),
            torch.no_grad(),
            pytest.raises(NotImplementedError, match="complete local sequences"),
        ):
            self.router(hidden_states)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="TE fused router ops not available",
    )
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward_fusion_equivalence(self, score_function):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.config.moe_router_score_function = score_function
            hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()

            # Unfused
            self.router.config.moe_router_fusion = False
            scores_ref, routing_ref = self.router(hidden_states)

            # Fused
            self.router.config.moe_router_fusion = True
            scores_fused, routing_fused = self.router(hidden_states)

            assert torch.equal(routing_ref, routing_fused), "Routing map mismatch"
            torch.testing.assert_close(scores_ref, scores_fused)
            # restore the config
            self.router.config.moe_router_fusion = False

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_aux_loss(self):
        self.sequential_mlp = self.sequential_mlp.cuda()

        # Without aux loss
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda().bfloat16()
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() == 0

        # With aux loss
        self.transformer_config.moe_aux_loss_coeff = 1
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0

        # With Z loss
        self.transformer_config.moe_aux_loss_coeff = 0
        self.transformer_config.moe_z_loss_coeff = 1
        self.sequential_mlp.router.weight.grad.fill_(0)
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_with_padding_mask(self):
        """Test that padding mask correctly excludes padding tokens from routing."""
        self.router = self.router.cuda()
        seq_len = 32
        batch_size = 2
        hidden_size = self.router.config.hidden_size

        # Create input with shape [seq_len, batch_size, hidden_size]
        hidden_states = torch.randn((seq_len, batch_size, hidden_size)).cuda().bfloat16()

        # Create padding mask: first half valid, second half padding
        # padding_mask shape: [seq_len, batch_size]
        # Convention: True = padding (exclude), False = valid (include)
        padding_mask = torch.zeros((seq_len, batch_size), dtype=torch.bool, device='cuda')
        padding_mask[seq_len // 2 :, :] = True  # Second half is padding

        # Test forward pass with padding mask
        with torch.no_grad():
            probs_with_mask, routing_map_with_mask = self.router(
                hidden_states, padding_mask=padding_mask
            )

            # Test forward pass without padding mask (only valid tokens)
            hidden_states_valid = hidden_states[: seq_len // 2, :, :]
            probs_without_mask, routing_map_without_mask = self.router(hidden_states_valid)

            # The valid part of routing with mask should match routing without mask
            probs_valid_part = probs_with_mask.reshape(seq_len, batch_size, -1)[
                : seq_len // 2, :, :
            ]
            probs_valid_part = probs_valid_part.reshape(-1, probs_valid_part.shape[-1])

            # Check that shapes are as expected
            assert probs_with_mask.shape == (
                seq_len * batch_size,
                self.router.config.num_moe_experts,
            )
            assert routing_map_with_mask.shape == (
                seq_len * batch_size,
                self.router.config.num_moe_experts,
            )

            # Verify that probs for valid tokens are similar
            assert torch.equal(probs_valid_part, probs_without_mask)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("with_padding_mask", [False, True])
    def test_expert_bias_token_counts_with_padding_mask(self, with_padding_mask):
        """Test expert-bias counts in grad-enabled masked and unmasked forwards."""
        config = dataclasses.replace(
            self.transformer_config,
            moe_router_enable_expert_bias=True,
            moe_router_load_balancing_type="none",
            moe_router_score_function="sigmoid",
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(
                num_experts=config.num_moe_experts, moe_grouped_gemm=False
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        router = cast(Router, MoELayer(config, submodules).router).cuda().train()

        seq_len = 5
        batch_size = 2
        hidden_states = torch.randn(
            (seq_len, batch_size, config.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        assert seq_len * batch_size != config.num_moe_experts
        padding_mask = None
        if with_padding_mask:
            padding_mask = torch.zeros((seq_len, batch_size), dtype=torch.bool, device="cuda")
            padding_mask[-2:, 0] = True

        probs, routing_map = router(hidden_states, padding_mask=padding_mask)
        valid_routing_map = routing_map
        if padding_mask is not None:
            valid_routing_map = routing_map[~padding_mask.reshape(-1)]

        expected_tokens_per_expert = valid_routing_map.sum(dim=0)
        torch.testing.assert_close(
            router.local_tokens_per_expert,
            expected_tokens_per_expert.to(router.local_tokens_per_expert.dtype),
        )
        expected_valid_tokens = seq_len * batch_size
        if padding_mask is not None:
            expected_valid_tokens -= padding_mask.sum().item()
        assert (
            router.local_tokens_per_expert.sum() == expected_valid_tokens * config.moe_router_topk
        )

        probs.sum().backward()
        assert hidden_states.grad is not None
        assert hidden_states.grad.isfinite().all()
        assert router.weight.grad is not None
        assert router.weight.grad.isfinite().all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_dtype(self):
        self.router = self.router.cuda()
        self.sequential_mlp = self.sequential_mlp.cuda()
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size), dtype=torch.bfloat16)
        hidden_states = hidden_states.cuda()

        # Test with default setting (bf16)
        self.router.config.moe_router_dtype = None
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.bfloat16, "Router output should be bf16 by default"
            assert out[0].dtype == torch.bfloat16

        # Test with fp32 enabled
        self.router.config.moe_router_dtype = 'fp32'
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.float32, "Router output should be fp32 when enabled"
            assert out[0].dtype == torch.bfloat16
            self.sequential_mlp.config.moe_token_dispatcher_type = "alltoall"
            out = self.sequential_mlp(hidden_states)
            assert out[0].dtype == torch.bfloat16
            self.sequential_mlp.config.moe_token_dispatcher_type = "allgather"

        # Test with fp64 enabled
        self.router.config.moe_router_dtype = 'fp64'
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.float64, "Router output should be fp64 when enabled"
            assert out[0].dtype == torch.bfloat16

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_force_load_balancing(self):
        hidden_states = torch.randn(
            (32, 2, self.router.config.hidden_size), device="cuda", dtype=torch.bfloat16
        )
        hidden_states.requires_grad = True

        # First forward pass with normal routing
        normal_scores, normal_routing_map = self.router(hidden_states)

        # Second forward pass with force load balancing
        self.router.config.moe_router_force_load_balancing = True
        force_scores, force_routing_map = self.router(hidden_states)

        assert normal_scores.shape == force_scores.shape
        assert normal_routing_map.shape == force_routing_map.shape
        assert torch.equal(normal_scores, force_scores) == False

        # Backward pass for force load balancing
        self.router.zero_grad()
        force_scores.sum().backward()
        assert hidden_states.grad is not None
        assert self.router.weight.grad.norm() > 0

        self.router.config.moe_router_force_load_balancing = False

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("capacity_factor", [None, 1.0, 2.0])
    @pytest.mark.parametrize("drop_policy", ["probs", "position"])
    @pytest.mark.parametrize("pad_to_capacity", [True, False])
    def test_token_dropping(self, capacity_factor, drop_policy, pad_to_capacity):
        if capacity_factor is None and pad_to_capacity:
            pytest.skip("Capacity factor is None, so no token dropping should be applied")

        num_tokens = 32
        self.router = self.router.cuda()
        self.router.config.moe_expert_capacity_factor = capacity_factor
        self.router.config.moe_token_drop_policy = drop_policy
        self.router.config.moe_pad_expert_input_to_capacity = pad_to_capacity

        hidden_states = torch.randn(
            (num_tokens, self.router.config.hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        hidden_states.requires_grad = True
        probs, routing_map = self.router(hidden_states)

        if capacity_factor is not None:
            if pad_to_capacity:
                assert (
                    routing_map.sum().item()
                    == num_tokens * self.router.config.moe_router_topk * capacity_factor
                )
            else:
                assert (
                    routing_map.sum().item()
                    <= num_tokens * self.router.config.moe_router_topk * capacity_factor
                )
        else:
            assert routing_map.sum().item() == num_tokens * self.router.config.moe_router_topk

        # restore the config
        self.router.config.moe_expert_capacity_factor = None
        self.router.config.moe_token_drop_policy = "probs"
        self.router.config.moe_pad_expert_input_to_capacity = False


@pytest.mark.parametrize("tp_size", [2, 4])
def test_router_sequence_parallel_observation_is_replicated_and_transparent(tp_size):
    """Exercise the real router hook and check that observing preserves output/backward."""
    if Utils.world_size % tp_size:
        pytest.skip("The distributed test world must be divisible by TP.")
    Utils.initialize_model_parallel(tp_size, 1, expert_tensor_parallel_size=1)
    try:
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=1,
            sequence_parallel=True,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_score_function="sigmoid",
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0,
            add_bias_linear=False,
            use_cpu_initialization=True,
        )
        router = router_mod.TopKRouter(
            cfg, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        group = parallel_state.get_tensor_model_parallel_group()
        generator = torch.Generator(device="cuda").manual_seed(1234)
        full_input = torch.randn(16, 2, 16, device="cuda", generator=generator)
        local_input = full_input.chunk(tp_size, dim=0)[group.rank()].clone().requires_grad_()
        expected_probs, expected_map = router(local_input)
        expected_grads = torch.autograd.grad(
            expected_probs.square().sum(), (local_input, router.weight)
        )
        observations = []
        with capture_tensor_observations(
            lambda *args: observations.append(args), frozenset({"router_diagnostics"})
        ):
            probs, routing = router(local_input)
        grads = torch.autograd.grad(probs.square().sum(), (local_input, router.weight))
        torch.testing.assert_close(probs, expected_probs, rtol=0, atol=0)
        assert torch.equal(routing, expected_map)
        for actual, expected in zip(grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        with torch.no_grad():
            full_logits = router.gating(full_input).flatten(0, 1)
            aux_map, aux_scores = router_mod.compute_routing_scores_for_aux_loss(
                full_logits, router.topk, router.score_function, fused=False
            )
            _, full_map = router.routing(full_logits.reshape(16, 2, 8))
            expected_diagnostics = build_router_diagnostics(
                aux_scores, aux_map, full_map, router.expert_bias, 16, 2
            )
        assert len(observations) == 1
        owner, name, source_kind, actual_diagnostics, tp_shard_dim = observations[0]
        assert owner is router
        assert name == source_kind == "router_diagnostics"
        assert tp_shard_dim is None
        torch.testing.assert_close(actual_diagnostics, expected_diagnostics, rtol=1e-5, atol=1e-7)
    finally:
        Utils.destroy_model_parallel()


class TestGroupLimitedRouter:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            context_parallel_size=1,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")

        num_moe_experts = 16
        self.transformer_config = TransformerConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            context_parallel_size=1,
            num_moe_experts=num_moe_experts,
            moe_router_topk=4,
            moe_router_group_topk=2,
            moe_router_num_groups=8,
            moe_router_pre_softmax=True,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0,
            moe_router_dtype='fp32',
            moe_token_dispatcher_type="alltoall",
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )

        # init MoE layer
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=num_moe_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        self.moe_layer = MoELayer(self.transformer_config, submodules).cuda()
        self.router = cast(Router, self.moe_layer.router)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert (
            num_weights
            == self.transformer_config.hidden_size * self.transformer_config.num_moe_experts
        ), num_weights

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("moe_router_group_topk,moe_router_num_groups", [(3, 8), (2, 4)])
    @pytest.mark.parametrize("moe_router_pre_softmax", [(True), (False)])
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward(
        self, moe_router_group_topk, moe_router_num_groups, moe_router_pre_softmax, score_function
    ):
        with torch.no_grad():
            self.router.config.moe_router_group_topk = moe_router_group_topk
            self.router.config.moe_router_num_groups = moe_router_num_groups
            self.router.config.moe_router_pre_softmax = moe_router_pre_softmax
            self.router.config.moe_router_score_function = score_function
            if moe_router_pre_softmax:
                self.router.config.moe_router_topk_scaling_factor = 16.0

            seq_len = 128
            batch_size = 4
            num_tokens = seq_len * batch_size
            # hidden_states shape: [seq_len, batch_size, hidden_size]
            hidden_states = (
                torch.randn((seq_len, batch_size, self.router.config.hidden_size)).cuda().bfloat16()
            )
            scores, routing_map = self.router(hidden_states)
            assert scores.shape == (num_tokens, self.router.config.num_moe_experts), scores.shape
            assert routing_map.shape == (
                num_tokens,
                self.router.config.num_moe_experts,
            ), routing_map.shape

            group_routing_map = (
                routing_map.reshape(num_tokens, moe_router_num_groups, -1).max(dim=-1).values
            )
            assert torch.all(group_routing_map.sum(dim=-1) <= moe_router_group_topk)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="TE fused router ops not available",
    )
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward_fusion_equivalence(self, score_function):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.score_function = score_function
            seq_len = 32
            batch_size = 4
            hidden_states = torch.randn((seq_len, batch_size, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()

            # Unfused
            self.router.config.moe_router_fusion = False
            scores_ref, routing_ref = self.router(hidden_states)

            # Fused
            self.router.config.moe_router_fusion = True
            scores_fused, routing_fused = self.router(hidden_states)

            assert torch.equal(routing_ref, routing_fused), "Routing map mismatch"
            torch.testing.assert_close(scores_ref, scores_fused)
            # restore the config
            self.router.config.moe_router_fusion = False


class TestAuxLossFreeTop2Router:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=8)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")
        num_moe_experts = 8
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            expert_model_parallel_size=8,
            moe_router_load_balancing_type="none",  # No aux loss
            moe_router_score_function="sigmoid",  # Using sigmoid scoring
            moe_router_enable_expert_bias=True,  # Enable expert bias
            moe_router_bias_update_rate=0.1,  # Set bias update rate
            moe_router_topk=2,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=num_moe_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        self.moe_layer = MoELayer(self.transformer_config, submodules)
        self.router = cast(Router, self.moe_layer.router)
        assert self.router.expert_bias is not None
        assert self.router.local_tokens_per_expert is not None

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_forward_aux_free(self):
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda().bfloat16()
        self.router = self.router.cuda()

        # First forward pass
        initial_bias = self.router.expert_bias.clone()
        scores1, indices1 = self.router(hidden_states)
        initial_tokens = self.router.local_tokens_per_expert.clone()
        updated_bias = get_updated_expert_bias(
            self.router.local_tokens_per_expert,
            self.router.expert_bias,
            self.router.config.moe_router_bias_update_rate,
        )

        # Verify expert bias was updated
        assert not torch.equal(initial_bias, updated_bias), "Expert bias should be updated"

        # Basic output checks
        assert scores1.shape == (64, 8), "Router scores shape mismatch"
        assert indices1.shape == (64, 8), "Router indices shape mismatch"

        # Print some debug info
        print("Updated bias after first forward pass:", updated_bias)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_expert_bias_update_preserves_large_integer_count_ordering(self):
        counts = torch.tensor([2**24 + 1, 2**24], dtype=torch.int64, device="cuda")
        bias = torch.zeros(2, dtype=torch.float32, device="cuda")

        updated_bias = get_updated_expert_bias(
            counts, bias, self.router.config.moe_router_bias_update_rate
        )

        expected = torch.tensor([-0.1, 0.1], dtype=torch.float32, device="cuda")
        torch.testing.assert_close(updated_bias, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_expert_bias_token_counts_survive_float16_module(self):
        wrapped = Float16Module(self.transformer_config, self.moe_layer.cuda())
        router = cast(Router, wrapped.module.router)

        assert router.local_tokens_per_expert.dtype == torch.int64

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="TE fused router ops not available",
    )
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward_fusion_equivalence(self, score_function):
        with torch.no_grad():
            # Build two fresh routers to avoid bias update interference
            submodules = get_submodules(
                get_gpt_layer_local_submodules(
                    num_experts=self.transformer_config.num_moe_experts, moe_grouped_gemm=False
                ).mlp
            )
            assert isinstance(submodules, MoESubmodules)
            moe_layer_ref = MoELayer(self.transformer_config, submodules)
            moe_layer_fused = MoELayer(self.transformer_config, submodules)
            router_ref = moe_layer_ref.router.cuda()
            router_fused = moe_layer_fused.router.cuda()

            # Ensure identical initial parameters/state
            router_fused.weight.copy_(router_ref.weight)
            expert_bias_sample = torch.randn_like(router_ref.expert_bias)
            router_ref.expert_bias.copy_(expert_bias_sample)
            router_fused.expert_bias.copy_(expert_bias_sample)

            router_ref.config.moe_router_score_function = score_function
            router_fused.config.moe_router_score_function = score_function

            hidden_states = torch.randn((32, 2, router_ref.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()

            # Unfused
            router_ref.config.moe_router_fusion = False
            scores_ref, routing_ref = router_ref(hidden_states)

            # Fused
            router_fused.config.moe_router_fusion = True
            scores_fused, routing_fused = router_fused(hidden_states)

            assert torch.equal(routing_ref, routing_fused)
            torch.testing.assert_close(scores_ref, scores_fused)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("router_dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_router_gating_linear(router_dtype):
    tols = dict(rtol=2.0e-2, atol=1.0e-3)

    ref_inp = torch.randn((4096, 7168), dtype=torch.bfloat16, device="cuda")
    ref_weight = torch.randn((256, 7168), dtype=torch.bfloat16, device="cuda")
    ref_inp.requires_grad = True
    ref_weight.requires_grad = True
    bwd_input = torch.randn((4096, 256), dtype=router_dtype, device="cuda")

    ref_output = torch.nn.functional.linear(ref_inp.to(router_dtype), ref_weight.to(router_dtype))
    ref_output.backward(bwd_input)

    inp = ref_inp.detach()
    weight = ref_weight.detach()
    inp.requires_grad = True
    weight.requires_grad = True
    bias = None
    output = router_gating_linear(inp, weight, bias, router_dtype)
    output.backward(bwd_input)

    assert output.dtype == router_dtype
    assert ref_inp.grad.dtype == ref_inp.dtype
    assert ref_weight.grad.dtype == ref_weight.dtype
    # Relax atol for float32: TE general_gemm produces results ~6.5e-3 away from
    # torch.nn.functional.linear, which exceeds the default 1e-3 atol.
    if router_dtype == torch.float32:
        tols = dict(rtol=2.0e-2, atol=1.0e-2)
    assert torch.allclose(output, ref_output, **tols)
    assert torch.allclose(inp.grad, ref_inp.grad, **tols)
    assert torch.allclose(weight.grad, ref_weight.grad, **tols)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("router_dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_router_gating_linear_bias(router_dtype):
    tols = dict(rtol=2.0e-2, atol=1.0e-3)

    ref_inp = torch.randn((4096, 7168), dtype=router_dtype, device="cuda")
    ref_weight = torch.randn((256, 7168), dtype=router_dtype, device="cuda")
    ref_bias = torch.randn((256,), dtype=router_dtype, device="cuda")
    ref_inp.requires_grad = True
    ref_weight.requires_grad = True
    ref_bias.requires_grad = True
    bwd_input = torch.randn((4096, 256), dtype=router_dtype, device="cuda")

    ref_output = torch.nn.functional.linear(
        ref_inp.to(router_dtype), ref_weight.to(router_dtype), ref_bias.to(router_dtype)
    )
    ref_output.backward(bwd_input)

    inp = ref_inp.detach()
    weight = ref_weight.detach()
    bias = ref_bias.detach()
    inp.requires_grad = True
    weight.requires_grad = True
    bias.requires_grad = True
    output = router_gating_linear(inp, weight, bias, router_dtype)
    output.backward(bwd_input)

    assert output.dtype == router_dtype
    assert ref_inp.grad.dtype == ref_inp.dtype
    assert ref_weight.grad.dtype == ref_weight.dtype
    assert ref_bias.grad.dtype == ref_bias.dtype
    assert torch.allclose(output, ref_output, **tols)
    assert torch.allclose(inp.grad, ref_inp.grad, **tols)
    assert torch.allclose(weight.grad, ref_weight.grad, **tols)
    assert torch.allclose(bias.grad, ref_bias.grad, **tols)


@pytest.mark.internal
@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("use_pre_softmax", [True, False])
@pytest.mark.parametrize("topk", [1, 2])
def test_topk_routing_precomputed_indices_equivalence(score_function, use_pre_softmax, topk):
    """Passing precomputed_indices that match the function's own selection must reproduce
    the standard output. Guards the shared post-top-k path reused by quantile balancing."""
    if score_function != "softmax" and use_pre_softmax:
        pytest.skip("pre_softmax only applies to softmax scoring")

    torch.manual_seed(123)
    num_tokens, num_experts = 64, 8
    logits = torch.randn(num_tokens, num_experts)

    kwargs = dict(use_pre_softmax=use_pre_softmax, score_function=score_function, fused=False)
    probs_ref, map_ref = topk_routing_with_score_function(logits, topk, **kwargs)
    _, top_indices = topk_routing_with_score_function(logits, topk, dense_output=True, **kwargs)
    probs_pre, map_pre = topk_routing_with_score_function(
        logits, topk, precomputed_indices=top_indices, **kwargs
    )

    # Natural top-k indices reproduce the standard output.
    assert torch.equal(map_ref, map_pre)
    torch.testing.assert_close(probs_ref, probs_pre)

    # Indices that differ from the natural top-k must route to exactly those experts. This
    # catches a regression where the precomputed_indices branch is dropped and the function
    # silently recomputes its own top-k instead of honoring the caller's indices. Bottom-k is
    # disjoint from top-k since 2 * topk <= num_experts.
    alt_indices = logits.topk(topk, dim=1, largest=False).indices
    _, map_alt = topk_routing_with_score_function(
        logits, topk, precomputed_indices=alt_indices, **kwargs
    )
    expected_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, alt_indices, True)
    assert torch.equal(map_alt, expected_map)


def test_moe_router_aux_loss_fusion_defaults_to_router_fusion():
    """Unset follows moe_router_fusion, including after post-construction mutation."""
    kwargs = dict(num_layers=1, hidden_size=8, num_attention_heads=1, num_moe_experts=4)

    config = TransformerConfig(**kwargs)
    assert config.moe_router_aux_loss_fusion is None
    assert config.moe_router_aux_loss_fusion_enabled is False
    config.moe_router_fusion = True
    assert config.moe_router_aux_loss_fusion_enabled is True

    for routing, aux in ((True, False), (False, True)):
        config = TransformerConfig(
            moe_router_fusion=routing, moe_router_aux_loss_fusion=aux, **kwargs
        )
        assert config.moe_router_aux_loss_fusion_enabled is aux


def _fp32_weights_config(fp32_weights, **overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        num_moe_experts=4,
        use_cpu_initialization=True,
        moe_router_load_balancing_type="aux_loss",
        moe_router_topk=2,
        moe_aux_loss_coeff=0,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        moe_router_dtype='fp32',
        moe_router_fp32_weights=fp32_weights,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_moe_router_fp32_weights_requires_fp32_routing_arithmetic():
    """moe_router_fp32_weights requires fp32 routing arithmetic."""
    for router_dtype in (None, 'fp64'):
        with pytest.raises(ValueError, match="moe_router_dtype"):
            _fp32_weights_config(True, moe_router_dtype=router_dtype)


def test_moe_router_fp32_weights_requires_reduced_precision_run():
    with pytest.raises(ValueError, match="reduced-precision"):
        _fp32_weights_config(True, bf16=False, params_dtype=torch.float32)


def test_fp32_router_gating_matches_fp32_reference():
    """An FP32 gate gives logits closer to an exact FP32 reference than a BF16 one."""
    hidden = 64
    inp = torch.randn(128, hidden, dtype=torch.bfloat16)
    weight = torch.randn(8, hidden, dtype=torch.float32)

    fp32_logits = router_gating_linear(inp, weight, None, torch.float32, True)
    bf16_logits = router_gating_linear(inp, weight.to(torch.bfloat16), None, torch.float32, True)
    reference = inp.to(torch.float32) @ weight.t()

    assert fp32_logits.dtype == torch.float32
    torch.testing.assert_close(fp32_logits, reference, rtol=1e-5, atol=1e-5)
    assert (fp32_logits - reference).abs().max() < (bf16_logits - reference).abs().max()


def test_fp32_router_gating_backward_preserves_dtypes():
    hidden = 32
    inp = torch.randn(16, hidden, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4, hidden, dtype=torch.float32, requires_grad=True)

    router_gating_linear(inp, weight, None, torch.float32, True).sum().backward()

    assert weight.grad is not None and weight.grad.dtype == torch.float32
    assert inp.grad is not None and inp.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_router_gating_passes_te_matching_operand_dtypes(monkeypatch, weight_dtype):
    """cuBLASLt has no mixed-operand GEMM, so A and B must reach TE with equal dtypes."""
    import megatron.core.transformer.moe.moe_utils as moe_utils_mod

    seen = {}

    def fake_te_general_gemm(A, B, out_dtype=None, layout="TN", bias=None, **kwargs):
        seen.update(A=A.dtype, B=B.dtype, out=out_dtype, layout=layout)
        return [torch.zeros(B.shape[0], A.shape[0], dtype=out_dtype)]

    monkeypatch.setattr(moe_utils_mod, 'te_general_gemm', fake_te_general_gemm)

    inp = torch.randn(8, 16, dtype=torch.bfloat16)
    weight = torch.randn(4, 16, dtype=weight_dtype)
    out = router_gating_linear(inp, weight, None, torch.float32)

    assert seen['A'] == seen['B'] == weight_dtype
    assert seen['out'] == torch.float32 and seen['layout'] == "TN"
    assert out.dtype == torch.float32


def test_force_fp32_keeps_both_gating_gemms_off_te(monkeypatch):
    """force_fp32 must bypass TE in forward and backward, since TE would use TF32."""
    import megatron.core.transformer.moe.moe_utils as moe_utils_mod

    calls = []

    def tripwire_te_general_gemm(A, B, out_dtype=None, **kwargs):
        calls.append((A.dtype, B.dtype, out_dtype))
        raise AssertionError("TE was called despite force_fp32")

    monkeypatch.setattr(moe_utils_mod, 'te_general_gemm', tripwire_te_general_gemm)

    inp = torch.randn(32, 16, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4, 16, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(4, dtype=torch.float32, requires_grad=True)

    out = router_gating_linear(inp, weight, bias, torch.float32, True)
    out.sum().backward()

    assert calls == []
    assert out.dtype == torch.float32
    assert weight.grad.dtype == torch.float32 and bias.grad.dtype == torch.float32
    assert inp.grad.dtype == torch.bfloat16


def test_force_fp32_gating_is_exact_to_fp32_reference():
    """The torch path is exact IEEE FP32, not TF32."""
    from megatron.core.transformer.moe.moe_utils import _highest_fp32_matmul_precision

    inp = torch.randn(64, 32, dtype=torch.bfloat16)
    weight = torch.randn(8, 32, dtype=torch.float32)
    bias = torch.randn(8, dtype=torch.float32)

    out = router_gating_linear(inp, weight, bias, torch.float32, True)
    with _highest_fp32_matmul_precision():
        reference = torch.addmm(bias, inp.float(), weight.t())

    torch.testing.assert_close(out, reference, rtol=0, atol=0)


def _matmul_precision():
    """Read the per-backend setting; the legacy getter raises once this one is in use."""
    return torch.backends.cuda.matmul.fp32_precision


@contextlib.contextmanager
def _global_tf32_optin(how):
    """Opt the whole process into TF32 the way a library in the container might."""
    previous = _matmul_precision()
    if how == 'legacy':
        torch.set_float32_matmul_precision('high')
    else:
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
    try:
        yield _matmul_precision()
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous


@pytest.mark.parametrize('how', ['legacy', 'per_backend'])
def test_force_fp32_gating_overrides_global_tf32_optin(monkeypatch, how):
    """The gating GEMMs must run at full precision even when the process opts into TF32."""
    seen = {}
    real_addmm, real_mm = torch.addmm, torch.mm

    def recording_addmm(*args, **kwargs):
        seen['forward'] = _matmul_precision()
        return real_addmm(*args, **kwargs)

    def recording_mm(*args, **kwargs):
        seen.setdefault('backward', _matmul_precision())
        return real_mm(*args, **kwargs)

    monkeypatch.setattr(torch, 'addmm', recording_addmm)
    monkeypatch.setattr(torch, 'mm', recording_mm)

    with _global_tf32_optin(how) as outside:
        inp = torch.randn(16, 8, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(4, dtype=torch.float32, requires_grad=True)

        router_gating_linear(inp, weight, bias, torch.float32, True).sum().backward()

        assert seen['forward'] == 'ieee'
        assert seen['backward'] == 'ieee'
        # the override is scoped, so the rest of the model keeps the global setting
        assert _matmul_precision() == outside == 'tf32'


def test_gating_without_force_fp32_leaves_matmul_precision_alone(monkeypatch):
    """The guard must not change precision for runs that did not ask for FP32 weights."""
    import megatron.core.transformer.moe.moe_utils as moe_utils_mod

    monkeypatch.setattr(moe_utils_mod, 'te_general_gemm', None)
    seen = {}
    real_mm = torch.mm

    def recording_mm(*args, **kwargs):
        seen.setdefault('forward', _matmul_precision())
        return real_mm(*args, **kwargs)

    monkeypatch.setattr(torch, 'mm', recording_mm)

    with _global_tf32_optin('legacy'):
        inp = torch.randn(8, 16, dtype=torch.bfloat16)
        router_gating_linear(inp, torch.randn(4, 16, dtype=torch.bfloat16), None, torch.float32)
        assert seen['forward'] == 'tf32'


def test_force_fp32_defaults_off_so_bf16_gates_keep_te_path(monkeypatch):
    """A BF16 gate stays on TE, which accumulates BF16 operands in FP32."""
    import megatron.core.transformer.moe.moe_utils as moe_utils_mod

    calls = []

    def fake_te_general_gemm(A, B, out_dtype=None, layout="TN", bias=None, **kwargs):
        calls.append((A.dtype, B.dtype, out_dtype))
        return [torch.zeros(B.shape[0], A.shape[0], dtype=out_dtype)]

    monkeypatch.setattr(moe_utils_mod, 'te_general_gemm', fake_te_general_gemm)

    inp = torch.randn(8, 16, dtype=torch.bfloat16)
    router_gating_linear(inp, torch.randn(4, 16, dtype=torch.bfloat16), None, torch.float32)

    assert calls == [(torch.bfloat16, torch.bfloat16, torch.float32)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("fp32_weights", [False, True])
def test_moe_router_fp32_weights_survives_float16_module(fp32_weights):
    """The gate dtype must survive Float16Module, which re-sweeps the whole module."""
    Utils.initialize_model_parallel(1, 1)
    try:
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        config = _fp32_weights_config(fp32_weights, add_bias_linear=True)
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        moe_layer = MoELayer(config, submodules)
        router = cast(Router, moe_layer.router)
        expected = torch.float32 if fp32_weights else torch.bfloat16

        assert router.weight.dtype == expected
        assert getattr(router.weight, 'keep_in_fp32', False) is fp32_weights
        assert router.bias is not None and router.bias.dtype == expected

        wrapped = Float16Module(config, moe_layer)
        wrapped_router = cast(Router, wrapped.module.router)
        assert wrapped_router.weight.dtype == expected
        assert wrapped_router.bias.dtype == expected
    finally:
        Utils.destroy_model_parallel()
