# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""SSA checkpoint traversal with explicit groups and no global MPU initialization."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention import (
    SimplifiedSparseAttentionCore,
    SimplifiedSparseIndexer,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def explicit_groups(monkeypatch):
    """Match CP1 MIMO ownership without initializing global model-parallel state."""
    Utils.initialize_distributed()
    parallel_state.destroy_model_parallel()
    groups = [
        torch.distributed.new_group([rank]) for rank in range(torch.distributed.get_world_size())
    ]
    own_group = groups[torch.distributed.get_rank()]

    def unexpected_global_group(*args, **kwargs):
        pytest.fail("SSA checkpointing must use its module's explicit process groups")

    for name in (
        "get_tensor_model_parallel_group",
        "get_context_parallel_group",
        "get_data_parallel_group",
    ):
        monkeypatch.setattr(parallel_state, name, unexpected_global_group)
    try:
        yield ProcessGroupCollection(
            tp=own_group, cp=own_group, dp_cp=torch.distributed.group.WORLD
        )
    finally:
        torch.distributed.destroy_process_group(own_group)


@pytest.mark.parametrize("nested", [False, True], ids=["indexer", "attention_core"])
def test_ssa_checkpoint_uses_explicit_groups(explicit_groups, nested):
    """Exercise actual TE projections and recursive core-attention checkpoint traversal."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=16,
        num_query_groups=1,
        kv_channels=256,
        attention_dropout=0.0,
        experimental_attention_variant="ssa",
        dsa_indexer_topk=512,
        dsa_indexer_loss_coeff=0.1,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    if nested:
        module = SimplifiedSparseAttentionCore(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=explicit_groups,
        )
        indexer = module.indexer
        prefix = "mtp.layers.0.mtp_model_layer.self_attention.core_attention."
        indexer_prefix = prefix + "indexer."
    else:
        module = indexer = SimplifiedSparseIndexer(config, explicit_groups)
        prefix = "decoder.layers.0.self_attention.core_attention.indexer."
        indexer_prefix = prefix

    projections = [child for child in indexer.children() if isinstance(child, TELinear)]
    assert len(projections) == 2
    assert all(projection.parallel_mode is None for projection in projections)
    assert all(projection._tp_group is None for projection in projections)
    state = module.sharded_state_dict(
        prefix=prefix, metadata={"dp_cp_group": explicit_groups.dp_cp}
    )
    expected_names = {indexer_prefix + name for name, _ in indexer.named_parameters()}
    actual_names = {key for key, value in state.items() if isinstance(value, ShardedTensor)}
    assert actual_names == expected_names
    for name, parameter in indexer.named_parameters():
        key = indexer_prefix + name
        shard = state[key]
        assert shard.key == key
        assert shard.data is parameter
        assert shard.global_shape == tuple(parameter.shape)
        assert shard.replica_id == (0, 0, explicit_groups.dp_cp.rank())
