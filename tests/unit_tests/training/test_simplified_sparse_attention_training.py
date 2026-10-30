# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Phase-1 freezing and checkpoint policy without changing dense training."""

from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.state_dict_utils import load_preprocess
from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training import checkpointing, training
from megatron.training.simplified_sparse_attention import (
    add_initialized_indexer_state,
    freeze_backbone_for_indexer_training,
    initializing_indexer_from_dense,
    is_simplified_sparse_checkpoint,
    num_indexer_layers,
    retain_initialized_indexer,
    validate_checkpoint,
)


def _args(**overrides):
    values = dict(
        experimental_attention_variant="ssa", ssa_train_indexer_only=True, dsa_indexer_topk=512
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "variant,coefficient,logs_indexer",
    [
        ("ssa", 0.0, True),
        ("ssa", 0.2, True),
        ("ssa", None, False),
        (None, None, False),
        (None, 0.0, False),
        ("dsa", 0.0, False),
        ("dsa", 0.2, True),
    ],
)
def test_training_log_drains_zero_coefficient_ssa_tracker(
    monkeypatch, variant, coefficient, logs_indexer
):
    args = _args(
        experimental_attention_variant=variant,
        dsa_indexer_loss_coeff=coefficient,
        log_interval=100,
        timing_log_level=0,
        perform_rl_step=False,
        micro_batch_size=1,
        data_parallel_size=1,
        gtp_weight_remat_size=1,
        world_size=1,
        seq_length=16,
        num_experts=None,
        mtp_num_layers=None,
        num_layers=1,
        hybrid_layer_pattern=None,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    for getter in (
        "get_timers",
        "get_tensorboard_writer",
        "get_wandb_writer",
        "get_one_logger",
        "get_energy_monitor",
    ):
        monkeypatch.setattr(training, getter, lambda: None)
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 2)
    monkeypatch.setattr(training.one_logger_utils, "track_app_tag", lambda *args: None)
    monkeypatch.setattr(
        training, "reduce_max_stat_across_model_parallel_group", lambda value, **kwargs: value
    )
    groups = ProcessGroupCollection()
    groups.pp = mock.sentinel.pp_group
    groups.dp = mock.sentinel.dp_group
    groups.cp = mock.Mock()
    groups.cp.size.return_value = 1
    reduce = mock.Mock()
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)

    helper = training.DSAIndexerLossLoggingHelper
    # Preallocate CPU storage, retaining the production save/reduce/report/clear path.
    monkeypatch.setattr(helper, "tracker", {"values": torch.zeros(1)})
    loss = torch.tensor(4.0 * coefficient if coefficient is not None else 1.0)
    helper.save_loss_to_tracker(loss, 1, 1, reduce_group=mock.sentinel.loss_group)
    total = {}
    training.training_log(
        {}, total, 0.1, 1, 1.0, False, 0, None, None, None, None, pg_collection=groups
    )

    if logs_indexer:
        torch.testing.assert_close(total["indexer loss"], loss / 2)
        assert helper.tracker["values"].count_nonzero() == 0
        assert helper.tracker["reduce_group"] is None
        assert reduce.called
        if variant == "ssa":
            torch.testing.assert_close(
                total["indexer raw loss"], torch.tensor(2.0 if coefficient > 0 else 0.0)
            )
        else:
            assert "indexer raw loss" not in total
    else:
        assert "indexer loss" not in total and "indexer raw loss" not in total
        torch.testing.assert_close(helper.tracker["values"], loss.reshape(1))
        assert helper.tracker["reduce_group"] is mock.sentinel.loss_group
        reduce.assert_not_called()


class _Model(torch.nn.Module):
    def __init__(self, with_indexer=True):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2, bias=False)
        if with_indexer:
            self.indexer = torch.nn.Linear(2, 2, bias=False)
        self.router = torch.nn.Module()
        self.router.frozen_expert_bias = False
        self.router.register_buffer("expert_bias", torch.zeros(2))
        self.router.register_buffer("local_tokens_per_expert", torch.tensor([2.0, 0.0]))


def test_phase1_updates_only_indexer_and_freezes_router_buffers():
    model = _Model()
    before = deepcopy(model.state_dict())
    freeze_backbone_for_indexer_training([model], _args())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model.indexer(torch.ones(2)).sum().backward()
    optimizer.step()
    # Exercise the same outside-the-optimizer bias updater used in training.
    _update_router_expert_bias([model], SimpleNamespace(moe_router_bias_update_rate=0.1))
    after = model.state_dict()
    assert not torch.equal(after["indexer.weight"], before["indexer.weight"])
    for key in ("backbone.weight", "router.expert_bias", "router.local_tokens_per_expert"):
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    assert model.backbone.weight.grad is None


def test_phase1_preserves_explicit_indexer_freeze():
    model = _Model()
    model.indexer.weight.requires_grad_(False)
    freeze_backbone_for_indexer_training([model], _args())
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("variant,phase1", [(None, False), (None, True), ("ssa", False)])
def test_inactive_phase1_has_no_model_traversal_or_collectives(variant, phase1):
    model = mock.Mock()
    with mock.patch("torch.distributed.all_reduce") as reduce:
        freeze_backbone_for_indexer_training(
            [model], _args(experimental_attention_variant=variant, ssa_train_indexer_only=phase1)
        )
    model.named_parameters.assert_not_called()
    model.modules.assert_not_called()
    reduce.assert_not_called()


def test_pipeline_stage_without_indexer_is_allowed_if_a_peer_has_one():
    model = _Model(with_indexer=False)
    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_backend", return_value="gloo"),
        mock.patch("torch.distributed.all_reduce", side_effect=lambda value, **_: value.fill_(1)),
    ):
        freeze_backbone_for_indexer_training([model], _args())
    assert not model.backbone.weight.requires_grad


def test_phase1_rejects_model_without_any_indexer():
    with mock.patch("torch.distributed.is_initialized", return_value=False):
        with pytest.raises(RuntimeError, match="no sparse-attention indexer"):
            freeze_backbone_for_indexer_training([_Model(False)], _args())


@pytest.mark.parametrize("phase1,load_optimizer", [(False, False), (True, True)])
def test_dense_initialization_requires_phase1_and_a_fresh_optimizer(phase1, load_optimizer):
    dense = SimpleNamespace(experimental_attention_variant=None)
    with pytest.raises(AssertionError, match="dense checkpoint"):
        validate_checkpoint(
            _args(ssa_train_indexer_only=phase1), dense, load_optimizer=load_optimizer
        )


def test_phase_transition_requires_fresh_optimizer_but_resume_can_load_optimizer():
    saved = _args()
    validate_checkpoint(_args(), saved, load_optimizer=True)
    phase2 = _args(ssa_train_indexer_only=False)
    validate_checkpoint(phase2, saved, load_optimizer=False)
    with pytest.raises(AssertionError, match="training phase changed"):
        validate_checkpoint(phase2, saved, load_optimizer=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dsa_indexer_topk": 2048},
        {"dsa_indexer_mode": "standard"},
        {"dsa_simplified_use_learned_k": False},
        {"multi_latent_attention": True},
    ],
)
def test_sparse_model_only_loading_still_checks_architecture(overrides):
    with pytest.raises(AssertionError):
        validate_checkpoint(_args(), _args(**overrides), load_optimizer=False)


def test_old_learned_indexer_checkpoint_keeps_fixed_architecture_without_cli_aliases():
    saved = _args(dsa_indexer_mode="simplified", dsa_simplified_use_learned_k=True)
    saved.experimental_attention_variant = "dsa"
    validate_checkpoint(_args(), saved, load_optimizer=False)


def test_old_sparse_checkpoint_cannot_fall_through_dense_loading():
    saved = _args(dsa_indexer_mode="simplified", dsa_simplified_use_learned_k=True)
    saved.experimental_attention_variant = "dsa"
    dense = SimpleNamespace(experimental_attention_variant=None)
    assert is_simplified_sparse_checkpoint(saved)
    with pytest.raises(AssertionError, match="requires experimental_attention_variant"):
        validate_checkpoint(dense, saved, load_optimizer=False)


def test_dense_checkpoint_metadata_keeps_progress_and_sparse_initialization_arguments(monkeypatch):
    runtime = _args(
        load="checkpoint",
        no_load_optim=True,
        finetune=False,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        add_position_embedding=False,
        vocab_file=None,
        data_parallel_random_init=False,
        use_dist_ckpt=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        spec=[
            "megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_layer_specs",
            "ssa_stack_spec",
        ],
    )
    saved = deepcopy(runtime)
    saved.experimental_attention_variant = None
    saved.spec = ["megatron.core.models.hybrid.hybrid_layer_specs", "hybrid_stack_spec"]
    sparse_spec = list(runtime.spec)
    monkeypatch.setattr(checkpointing, "get_args", lambda: runtime)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *_a, **_kw: ({"args": saved, "iteration": 37}, "checkpoint", False, None),
    )
    restored, _ = checkpointing.load_args_from_checkpoint(runtime)
    checkpointing.check_checkpoint_args(saved)
    assert restored.iteration == 37
    assert not hasattr(restored, "dsa_gqa_backend")
    assert restored.experimental_attention_variant == "ssa"
    assert restored.ssa_train_indexer_only
    assert restored.spec == sparse_spec
    assert not restored.finetune
    saved.hidden_size = 128
    with pytest.raises(AssertionError, match="hidden_size"):
        checkpointing.check_checkpoint_args(saved)


@pytest.mark.parametrize("legacy_metadata", [False, True])
@pytest.mark.parametrize("phase1", [False, True])
def test_sparse_checkpoint_args_restore_model_selection_but_preserve_training_phase(
    monkeypatch, legacy_metadata, phase1
):
    saved = _args(
        spec=[
            "megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_layer_specs",
            "ssa_stack_spec",
        ]
    )
    expected_spec = list(saved.spec)
    if legacy_metadata:
        saved.experimental_attention_variant = "dsa"
        saved.dsa_indexer_mode = "simplified"
        saved.dsa_simplified_use_learned_k = True
        saved.dsa_train_indexer_only = saved.ssa_train_indexer_only
        del saved.ssa_train_indexer_only
        saved.spec = [
            "megatron.core.transformer.experimental_attention_variant.dsa_layer_specs",
            "dsa_stack_spec",
        ]
    runtime = _args(
        experimental_attention_variant=None,
        ssa_train_indexer_only=phase1,
        dsa_indexer_topk=2048,
        spec=None,
        load="checkpoint",
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *_a, **_kw: ({"args": saved, "iteration": 37}, "checkpoint", False, None),
    )

    restored, _ = checkpointing.load_args_from_checkpoint(runtime)

    assert restored.experimental_attention_variant == "ssa"
    assert not hasattr(restored, "dsa_gqa_backend")
    assert restored.spec == expected_spec
    assert restored.dsa_indexer_topk == saved.dsa_indexer_topk
    assert restored.ssa_train_indexer_only == phase1
    assert restored.iteration == 37
    validate_checkpoint(restored, saved, load_optimizer=phase1)


def test_checkpoint_ignores_settings_for_layers_the_simplified_indexer_does_not_have():
    validate_checkpoint(
        _args(dsa_indexer_k_norm_epsilon=None, dsa_indexer_k_norm_fp32=False),
        _args(dsa_indexer_k_norm_epsilon=1e-6, dsa_indexer_k_norm_fp32=True),
        load_optimizer=True,
    )


def test_active_indexer_count_is_cached(monkeypatch):
    from megatron.core.models.hybrid import hybrid_layer_allocation

    num_indexer_layers.cache_clear()
    with mock.patch.object(
        hybrid_layer_allocation,
        "parse_hybrid_pattern",
        wraps=hybrid_layer_allocation.parse_hybrid_pattern,
    ) as parse:
        assert num_indexer_layers(5, "M*-M*") == 2
        assert num_indexer_layers(5, "M*-M*") == 2
        parse.assert_called_once()


def test_dense_checkpoint_args_keep_existing_spec_restore_without_sparse_fields(monkeypatch):
    saved = SimpleNamespace(spec=["dense_module", "dense_spec"])
    runtime = SimpleNamespace(
        spec=["requested_module", "requested_spec"],
        load="checkpoint",
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *_a, **_kw: ({"args": saved, "iteration": 37}, "checkpoint", False, None),
    )

    restored, _ = checkpointing.load_args_from_checkpoint(runtime)

    assert restored.spec == saved.spec
    assert not any(name.startswith("dsa_") for name in vars(restored))


def test_release_model_only_phase_transition_does_not_require_optimizer_resume(monkeypatch):
    runtime = _args(ssa_train_indexer_only=False)
    saved = _args()
    # The production loader supplies its actual decision: a release checkpoint
    # restores model weights even if --no-load-optim was not explicitly set.
    validate_checkpoint(runtime, saved, load_optimizer=False)


def test_distributed_dense_template_retains_indexer_but_requests_backbone():
    backbone = torch.ones(2, 2)
    indexer = torch.full((2, 2), 3.0)
    template = {
        "backbone.weight": ShardedTensor.from_rank_offsets("backbone.weight", backbone),
        "indexer.weight": ShardedTensor.from_rank_offsets("indexer.weight", indexer),
        "indexer._extra_state": ShardedObject("indexer._extra_state", None, (1,), (0,)),
    }
    retain_initialized_indexer(template)
    requested, local, _ = load_preprocess(template)
    assert isinstance(requested["backbone.weight"], ShardedTensor)
    assert "indexer.weight" not in requested
    torch.testing.assert_close(local["indexer.weight"], indexer)
    assert "indexer._extra_state" in local
