# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real-distributed test for the grad_sync vision partial-participation correction.

The dual-finalize per-token-mean path is validated end-to-end by
test_mimo_colocated_correctness (which wires configure_grad_sync into its
dp1-reference oracle). This file covers the participation-count helper directly
on grid-derived process groups (no parallel_state).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.mimo.training.encoder_prefetch import PREFETCHED_FEATURES_KEY, EncoderPrefetchLoader
from examples.mimo.training.grad_sync import (
    _token_normalization_scale,
    _vision_participation_count,
    configure_grad_sync,
    mark_modality_participation,
    reset_modality_participation,
)
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    create_hypercomm_grid,
    destroy_all_grids,
    get_pg_collection,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(("num_tokens", "expected"), [(0.0, 1.0), (4.0, 0.25)])
def test_token_normalization_scale(num_tokens, expected):
    scale = _token_normalization_scale(torch.tensor(num_tokens))

    assert scale.ndim == 0
    torch.testing.assert_close(scale, torch.tensor(expected))


def test_configure_grad_sync_installs_production_overlap_hooks():
    def no_sync():
        return None

    def start_grad_sync(*_unused):
        return None

    config = SimpleNamespace(no_sync_func=None, grad_sync_func=None)
    model = SimpleNamespace(
        config=config,
        no_sync=no_sync,
        start_grad_sync=start_grad_sync,
        language_model=None,
        modality_submodules={},
    )
    language_grid = SimpleNamespace(get_rank_enum=lambda _name: [[0]])
    topology = SimpleNamespace(grids={MIMO_LANGUAGE_MODULE_KEY: language_grid}, module_pgs={})

    configure_grad_sync(
        SimpleNamespace(overlap_grad_reduce=True, align_grad_reduce=True), model, topology
    )

    assert config.no_sync_func is no_sync
    assert config.grad_sync_func is start_grad_sync


class TestVisionParticipation:
    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        destroy_all_grids()

    @pytest.mark.parametrize("use_distributed_optimizer", [False, True])
    def test_image_sharded_projector_ddp_with_empty_rank(self, use_distributed_optimizer):
        """Empty image shards join real DDP SUM despite having no backward graph."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=8)
        pg_collection = get_pg_collection(grid)
        rank = dist.get_rank(pg_collection.dp)
        torch.manual_seed(71)
        backbone = torch.nn.Linear(4, 4, bias=False, device="cuda").requires_grad_(False)
        projector = torch.nn.Linear(4, 3, bias=False, device="cuda")
        reference = torch.nn.Linear(4, 3, bias=False, device="cuda")
        with torch.no_grad():
            reference.weight.copy_(projector.weight)
        config = TransformerConfig(
            num_layers=1, num_attention_heads=1, hidden_size=4, calculate_per_token_loss=True
        )
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                use_distributed_optimizer=use_distributed_optimizer,
                overlap_grad_reduce=False,
                average_in_collective=False,
            ),
            projector,
            pg_collection=pg_collection,
        )
        ddp.zero_grad_buffer()
        local_batches, local_upstreams = [], []
        global_token_count = torch.tensor(101.0, device="cuda")
        for sizes in ([0, 1, 0, 3, 2, 0, 4, 1], [0, 2, 0, 1, 1, 0, 2, 2]):
            inputs = torch.randn(sum(sizes), 4)
            upstream = torch.randn(sum(sizes), 3, device="cuda")
            (reference(backbone(inputs.cuda())) * upstream).sum().div(global_token_count).backward()
            local_inputs = torch.split(inputs, sizes)[rank]
            local_upstreams.append(torch.split(upstream, sizes)[rank])
            batch = {"input_ids": torch.zeros((1, 1), dtype=torch.long)}
            if sizes[rank]:
                batch["modality_inputs"] = {"vision": {"x": local_inputs}}
            local_batches.append(batch)

        def encode(inputs):
            with torch.no_grad():
                return backbone(inputs["x"])

        loader = EncoderPrefetchLoader(
            source=local_batches, encoder_name="vision", feature_producer=encode, depth=4
        )
        loader.start()
        try:
            for upstream in local_upstreams:
                batch = next(loader)
                features = batch.get(PREFETCHED_FEATURES_KEY)
                if features is not None:
                    ddp(features["vision"]).backward(upstream)
                else:
                    assert upstream.size(0) == 0
        finally:
            loader.close()
        if rank in (0, 2, 5):
            assert not torch.count_nonzero(projector.weight.main_grad)
        # Match the production non-overlap finalize path. Force all-reduce gives
        # a full-gradient oracle even with distributed-optimizer buffer layout.
        ddp.finish_grad_sync(force_all_reduce=True)
        ddp.scale_gradients(_token_normalization_scale(global_token_count))
        torch.testing.assert_close(projector.weight.main_grad, reference.weight.grad)
        assert all(parameter.grad is None for parameter in backbone.parameters())

    def test_scalar_fp32_scale_applies_to_bf16_gradients(self):
        grad = torch.ones(4, dtype=torch.bfloat16, device="cuda")
        scale = _token_normalization_scale(torch.tensor(2.0, dtype=torch.float32, device="cuda"))

        grad.mul_(scale)

        assert scale.ndim == 0
        torch.testing.assert_close(grad, torch.full_like(grad, 0.5))

    def test_vision_participation_correction(self):
        """Partial participation: text-only ranks upscale present ranks.

        With only some DP ranks holding image input, the participation count is
        < dp_size and the correction factor dp_size/participation is applied.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=self.world_size)
        vision_dp = grid.get_pg("dp")
        dp_size = dist.get_world_size(vision_dp)

        submodule = SimpleNamespace()
        fake_model = SimpleNamespace(modality_submodules={"images": submodule})

        rank = dist.get_rank(vision_dp)
        has_image = rank < dp_size // 2
        batch = (
            {"modality_inputs": {"images": {"hidden_states": torch.ones(1, device="cuda")}}}
            if has_image
            else {"modality_inputs": {}}
        )
        reset_modality_participation(fake_model)
        mark_modality_participation(fake_model, batch)

        count = _vision_participation_count(submodule, vision_dp)
        assert count.ndim == 0
        torch.testing.assert_close(
            count, torch.tensor(float(dp_size // 2), dtype=count.dtype, device=count.device)
        )
        factor = dp_size / count
        torch.testing.assert_close(
            factor, torch.tensor(2.0, dtype=factor.dtype, device=factor.device)
        )

        reset_modality_participation(fake_model)
        assert getattr(submodule, "_mimo_rank_processed_input") is False
