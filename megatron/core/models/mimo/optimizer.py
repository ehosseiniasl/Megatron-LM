# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optimizer for MIMO models with heterogeneous parallelism."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.utils import add_prefix_for_sharding
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32
from megatron.core.optimizer.optimizer import (
    SEPARATE_GRAD_NORM_GROUPS,
    MegatronOptimizer,
    _get_group_clip_grad,
    _get_param_grad_norm_group,
    _is_separate_grad_norm_group,
    _reported_grad_norm,
)
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import unwrap_model

if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


@dataclass
class ModuleOptimizerInfo:
    """Optimizer info for a single module."""

    optimizer: Optional[MegatronOptimizer]
    grid: Optional[HyperCommGrid]
    pg_collection: Optional[ProcessGroupCollection]
    is_active: bool


class MimoOptimizer(MegatronOptimizer):
    """
    Optimizer for MimoModel with heterogeneous parallelism.

    Each module gets its own optimizer. Global gradient norm is computed
    across all modules via all_reduce MAX, unless ``config.mimo_module_local_grad_clip``
    is set, in which case each module optimizer steps on its own.
    """

    def __init__(self, module_infos: Dict[str, ModuleOptimizerInfo], config: OptimizerConfig):
        self._validate_config(config)
        self.module_infos = module_infos
        self.config = config
        self._active_optimizers: List[MegatronOptimizer] = [
            info.optimizer
            for info in module_infos.values()
            if info.is_active and info.optimizer is not None
        ]
        self.is_stub_optimizer = len(self._active_optimizers) == 0
        self.optimizer = None  # Base class compat

    @staticmethod
    def _validate_config(config: OptimizerConfig) -> None:
        """Reject settings that could skip one module's update independently."""
        if config.mimo_module_local_grad_clip and (
            config.fp16 or config.loss_scale or math.isfinite(config.grad_norm_skip_threshold)
        ):
            raise ValueError(
                "mimo_module_local_grad_clip is not supported with loss scaling (fp16 or "
                "loss_scale) or a finite grad_norm_skip_threshold."
            )

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Prepare gradients for all active module optimizers."""
        found_inf = False
        for opt in self._active_optimizers:
            found_inf |= opt.prepare_grads()
        return found_inf

    def capture_grad_preparation_state(self) -> List[Any]:
        """Capture preparation-only state for every active module optimizer."""
        return [opt.capture_grad_preparation_state() for opt in self._active_optimizers]

    def restore_grad_preparation_state(self, state: List[Any]) -> None:
        """Restore preparation-only state for every active module optimizer."""
        assert len(state) == len(
            self._active_optimizers
        ), f"Expected {len(self._active_optimizers)} optimizer states, got {len(state)}"
        for opt, optimizer_state in zip(self._active_optimizers, state):
            opt.restore_grad_preparation_state(optimizer_state)

    def discard_prepared_grads(self) -> None:
        """Discard prepared gradients before a forward-backward rerun."""
        for opt in self._active_optimizers:
            opt.discard_prepared_grads()

    @torch.no_grad()
    def get_grad_norm(self) -> float:
        """Compute global gradient norm across all modules via all_reduce MAX."""
        num_modules = len(self.module_infos)
        norm_sq = torch.zeros(num_modules, device="cuda", dtype=torch.float32)

        for i, (name, info) in enumerate(sorted(self.module_infos.items())):
            if info.is_active and info.optimizer:
                module_norm = info.optimizer.get_grad_norm()
                if module_norm is not None:
                    module_norm = torch.as_tensor(
                        module_norm, device=norm_sq.device, dtype=norm_sq.dtype
                    ).reshape(())
                    norm_sq[i].copy_(module_norm.square())

        torch.distributed.all_reduce(norm_sq, op=torch.distributed.ReduceOp.MAX)
        return torch.sqrt(norm_sq.sum()).item()

    @torch.no_grad()
    def _compute_grad_norms_by_group(self) -> Dict[str, float]:
        """Combine each module's separate norms without counting its grid replicas twice."""
        norm_sq = torch.zeros(
            (len(self.module_infos), len(SEPARATE_GRAD_NORM_GROUPS)),
            device="cuda",
            dtype=torch.float32,
        )
        for i, (_, info) in enumerate(sorted(self.module_infos.items())):
            if info.is_active and info.optimizer is not None:
                norms = info.optimizer._compute_grad_norms_by_group()
                for j, group in enumerate(SEPARATE_GRAD_NORM_GROUPS):
                    norm_sq[i, j] = norms.get(group, 0.0) ** 2
        # Inactive encoder ranks must join this collective too.
        torch.distributed.all_reduce(norm_sq, op=torch.distributed.ReduceOp.MAX)
        norms = norm_sq.sum(dim=0).sqrt().tolist()
        self.grad_norms_by_group = dict(zip(SEPARATE_GRAD_NORM_GROUPS, norms))
        return self.grad_norms_by_group

    def _clip_separate_grads(self, main_grad_norm: float) -> None:
        """Apply the global main/indexer/MTP limits to each active module's parameters."""
        for opt in self._active_optimizers:
            if getattr(opt, "is_stub_optimizer", False):
                continue
            params_by_group = {}
            for param in opt.get_parameters():
                group = _get_param_grad_norm_group(param)
                if not _is_separate_grad_norm_group(group):
                    group = None
                params_by_group.setdefault(group, []).append(param)
            for group, params in params_by_group.items():
                clip_grad = _get_group_clip_grad(opt.config, group, opt.config.clip_grad)
                if clip_grad <= 0.0:
                    continue
                norm = main_grad_norm if group is None else self.grad_norms_by_group[group]
                clip_grad_by_total_norm_fp32(
                    params,
                    max_norm=clip_grad,
                    total_norm=norm,
                    use_decoupled_grad=opt.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                    or (
                        opt.config.use_precision_aware_optimizer
                        and getattr(params[0], "__fsdp_param__", False)
                    ),
                )

    @torch.no_grad()
    def prepare_grads_for_step(self) -> Tuple[bool, Optional[float]]:
        """Prepare gradients and measure the unclipped global gradient norm."""
        if self.config.mimo_module_local_grad_clip:
            raise NotImplementedError("mimo_module_local_grad_clip requires step().")
        self.grad_norms_by_group = {}
        self._prepared_main_grad_norm = None
        found_inf = self.prepare_grads()
        # Synchronize found_inf across all ranks to prevent deadlock:
        # if encoder ranks detect inf but LLM ranks don't, the early return
        # would skip the all_reduce in get_grad_norm(), causing a hang.
        found_inf_tensor = torch.tensor([found_inf], dtype=torch.float32, device="cuda")
        torch.distributed.all_reduce(found_inf_tensor, op=torch.distributed.ReduceOp.MAX)
        found_inf = found_inf_tensor.item() > 0
        if found_inf:
            return True, None

        main_grad_norm = self.get_grad_norm()
        if self.config.ssa_separate_indexer_grad_clip:
            self._prepared_main_grad_norm = main_grad_norm
            self._compute_grad_norms_by_group()
            return False, _reported_grad_norm(main_grad_norm, self.grad_norms_by_group)
        return False, main_grad_norm

    @torch.no_grad()
    def step_with_prepared_grads(
        self, found_inf: bool, grad_norm: Optional[float]
    ) -> Tuple[bool, Optional[float], Optional[int]]:
        """Clip and step after gradients have already been prepared and measured."""
        if found_inf:
            self.discard_prepared_grads()
            return False, None, None
        assert grad_norm is not None, "Finite prepared gradients require a gradient norm"

        should_skip_update = False

        if self.config.ssa_separate_indexer_grad_clip:
            # Reuse the norms already reported to the training rerun gate.
            main_grad_norm = self._prepared_main_grad_norm
            assert main_grad_norm is not None, "Separate clipping requires prepared group norms"
            should_skip_update = grad_norm > self.config.grad_norm_skip_threshold
            self._clip_separate_grads(main_grad_norm)
        else:
            # Preserve the existing shared clipping behavior when separate clipping is off.
            for opt in self._active_optimizers:
                if getattr(opt, "is_stub_optimizer", False):
                    continue
                params = opt.get_parameters()
                if params and opt.config.clip_grad > 0.0:
                    clip_grad_by_total_norm_fp32(
                        params,
                        max_norm=opt.config.clip_grad,
                        total_norm=grad_norm,
                        use_decoupled_grad=opt.config.use_precision_aware_optimizer,
                    )

        num_zeros = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        success = False if should_skip_update else self.step_with_ready_grads()
        # found_inf and the skip decision are world values, and a step with finite gradients does
        # not fail, so ranks agree without a collective (it would hold each module for the slowest).
        if not (success or should_skip_update):
            raise RuntimeError("A MIMO module optimizer step failed with finite gradients.")

        return success, grad_norm, num_zeros

    @torch.no_grad()
    def step(self) -> Tuple[bool, float | torch.Tensor | None, Optional[int]]:
        """Run one optimizer step across all active module optimizers."""
        if self.config.mimo_module_local_grad_clip:
            return self._step_module_local()
        found_inf, grad_norm = self.prepare_grads_for_step()
        return self.step_with_prepared_grads(found_inf, grad_norm)

    def _step_module_local(self) -> Tuple[bool, float | torch.Tensor | None, Optional[int]]:
        """Step each active module optimizer on its own process groups.

        Combines the results of this rank's modules. The grad norm or zero count is None if a
        module returns None for it. Tensor norms stay on device when combined.
        """
        results = [opt.step() for opt in self._active_optimizers]
        if not results:
            return True, None, None
        if len(results) == 1:
            return results[0]
        successes, grad_norms, zero_counts = zip(*results)
        grad_norm: float | torch.Tensor | None = None
        if all(norm is not None for norm in grad_norms):
            tensor_norms = [norm for norm in grad_norms if isinstance(norm, torch.Tensor)]
            scalar_norms = [norm for norm in grad_norms if not isinstance(norm, torch.Tensor)]
            if tensor_norms:
                grad_norm = tensor_norms[0]
                for norm in tensor_norms[1:]:
                    grad_norm = torch.hypot(grad_norm, norm)
                if scalar_norms:
                    grad_norm = torch.hypot(
                        grad_norm, torch.full_like(grad_norm, math.hypot(*scalar_norms))
                    )
            else:
                grad_norm = math.hypot(*scalar_norms)
        num_zeros = None if None in zero_counts else sum(zero_counts)
        return all(successes), grad_norm, num_zeros

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step active optimizers after gradients have been prepared."""
        success = True
        for opt in self._active_optimizers:
            success &= opt.step_with_ready_grads()
        return success

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients on all active module optimizers."""
        for opt in self._active_optimizers:
            opt.zero_grad(set_to_none)

    @property
    def chained_optimizers(self) -> List[MegatronOptimizer]:
        """Expose leaf optimizers to stock training, checkpoint, and rerun lifecycle hooks."""
        optimizers = []
        for opt in self._active_optimizers:
            optimizers.extend(getattr(opt, 'chained_optimizers', [opt]))
        return optimizers

    def prepare_model_params_for_param_sync(self) -> None:
        """Stage parameters for explicit synchronization in all active module optimizers."""
        for opt in self._active_optimizers:
            opt.prepare_model_params_for_param_sync()

    def quantize_and_sync_model_params_from_main_params(self) -> None:
        """Re-derive model params from main params in all active module optimizers."""
        for opt in self._active_optimizers:
            opt.quantize_and_sync_model_params_from_main_params()

    def get_loss_scale(self) -> torch.Tensor:
        """Return the loss scale tensor from the first active optimizer."""
        if self._active_optimizers:
            return self._active_optimizers[0].get_loss_scale()
        return torch.tensor([1.0], dtype=torch.float32, device="cuda")

    def count_zeros(self) -> int:
        """Count zero gradients per module (world-MAX so disjoint grids agree), then sum."""
        module_counts = torch.zeros(len(self.module_infos), device="cuda", dtype=torch.int64)
        for index, (_, info) in enumerate(sorted(self.module_infos.items())):
            if info.is_active and info.optimizer is not None:
                module_counts[index] = info.optimizer.count_zeros()
        torch.distributed.all_reduce(module_counts, op=torch.distributed.ReduceOp.MAX)
        return int(module_counts.sum().item())

    @property
    def param_groups(self) -> List[dict]:
        """Combined param groups from all active module optimizers."""
        groups = []
        for opt in self._active_optimizers:
            groups.extend(opt.param_groups)
        return groups

    # Checkpointing

    def state_dict(self):
        """Return per-module optimizer state dicts."""
        return {
            name: info.optimizer.state_dict() if info.is_active and info.optimizer else None
            for name, info in self.module_infos.items()
        }

    def load_state_dict(self, state_dict: Dict):
        """Load per-module optimizer state dicts.

        Reassembles param_groups and grad_scaler that were extracted and saved
        as ShardedObjects by sharded_state_dict(), then delegates to each
        per-module optimizer's load_state_dict.
        """
        for name, info in self.module_infos.items():
            if not (info.is_active and info.optimizer):
                continue
            module_sd = state_dict.get(name)
            if module_sd is None:
                continue

            for sub_sd, inner_opt in _iter_optimizer_sub_dicts(module_sd, info.optimizer):
                _restore_param_groups(sub_sd, inner_opt, name)
                _restore_param_state_sharding_type(sub_sd)
                _restore_grad_scaler(sub_sd)

            info.optimizer.load_state_dict(module_sd)

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        """Build sharded state dict, routing param_groups and grad_scaler
        through distributed save as ShardedObjects (common.pt is rank-0 only,
        which misses LLM optimizer state in non-colocated mode).
        """
        sharded_state = {}
        for name, info in self.module_infos.items():
            if info.is_active and info.optimizer:
                module_sd = info.optimizer.sharded_state_dict(
                    model_sharded_state_dict, is_loading, **kwargs
                )
                replica_id = _get_replica_id(info.pg_collection)

                for idx, (sub_sd, _) in enumerate(
                    _iter_optimizer_sub_dicts(module_sd, info.optimizer)
                ):
                    suffix = f'.{idx}' if idx > 0 else ''
                    _extract_param_groups(sub_sd, name, suffix, replica_id)
                    _extract_param_state_sharding_type(sub_sd, name, suffix, replica_id)
                    _extract_grad_scaler(sub_sd, name, suffix, replica_id)

                add_prefix_for_sharding(module_sd, f'mimo.{name}.')
                sharded_state[name] = module_sd
            else:
                sharded_state[name] = {}
        return sharded_state

    def reload_model_params(self, state_dict=None):
        """Reload model parameters in all active module optimizers."""
        for opt in self._active_optimizers:
            opt.reload_model_params(state_dict)


def _iter_optimizer_sub_dicts(module_sd, optimizer):
    """Yield (sub_state_dict, inner_optimizer) pairs, one per leaf optimizer.

    ChainedOptimizer nests: its state dict is keyed by integer index when the chain
    holds more than one optimizer, and delegates straight to the single child when it
    holds exactly one. A child can itself be a ChainedOptimizer, so this recurses.

    Descending only one level hands the caller an integer-keyed dict where a leaf
    state dict is expected. That is silent on save -- _extract_param_groups looks for
    an 'optimizer' key, finds none, and writes no param_groups -- and raises
    AttributeError on load when the restore helpers call str.startswith on an int key.
    """
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    if isinstance(optimizer, ChainedOptimizer):
        inner_optimizers = optimizer.chained_optimizers
        if len(inner_optimizers) > 1:
            for idx, inner_opt in enumerate(inner_optimizers):
                yield from _iter_optimizer_sub_dicts(module_sd[idx], inner_opt)
            return
        if len(inner_optimizers) == 1:
            # Both state_dict() and sharded_state_dict() return the single child's
            # state dict directly, so module_sd already belongs to that child.
            yield from _iter_optimizer_sub_dicts(module_sd, inner_optimizers[0])
            return
    yield module_sd, optimizer


def _extract_param_groups(sub_sd, module_name, suffix, replica_id):
    """Save: extract param_groups from optimizer sub-dict into a ShardedObject."""
    opt_sub = sub_sd.get('optimizer')
    if isinstance(opt_sub, dict) and 'param_groups' in opt_sub:
        pg = deepcopy(opt_sub['param_groups'])
        for group in pg:
            group['params'] = []
        sub_sd[f'_mimo_param_groups{suffix}'] = ShardedObject(
            f'optimizer.mimo.{module_name}{suffix}.param_groups',
            pg,
            (1,),
            (0,),
            replica_id=replica_id,
        )
        del opt_sub['param_groups']
        if not opt_sub:
            del sub_sd['optimizer']


def _extract_grad_scaler(sub_sd, module_name, suffix, replica_id):
    """Save: extract grad_scaler into a ShardedObject."""
    if 'grad_scaler' in sub_sd and sub_sd['grad_scaler'] is not None:
        sub_sd[f'_mimo_grad_scaler{suffix}'] = ShardedObject(
            f'optimizer.mimo.{module_name}{suffix}.grad_scaler',
            sub_sd.pop('grad_scaler'),
            (1,),
            (0,),
            replica_id=replica_id,
        )


def _extract_param_state_sharding_type(sub_sd, module_name, suffix, replica_id):
    """Save: extract param_state_sharding_type into a ShardedObject."""
    if 'param_state_sharding_type' in sub_sd:
        sub_sd[f'_mimo_param_state_sharding_type{suffix}'] = ShardedObject(
            f'optimizer.mimo.{module_name}{suffix}.param_state_sharding_type',
            sub_sd.pop('param_state_sharding_type'),
            (1,),
            (0,),
            replica_id=replica_id,
        )


def _restore_param_groups(sub_sd, inner_optimizer, module_name):
    """Load: restore param_groups with current param IDs from the inner optimizer."""
    # Find the _mimo_param_groups key (may have a suffix for chained optimizers)
    pg_key = None
    for k in list(sub_sd.keys()):
        if isinstance(k, str) and k.startswith('_mimo_param_groups'):
            pg_key = k
            break
    if pg_key is None:
        return

    loaded_pg = sub_sd.pop(pg_key)
    # Get current param IDs from the inner torch optimizer's state_dict
    current_pg = inner_optimizer.optimizer.state_dict()['param_groups']
    if len(loaded_pg) != len(current_pg):
        raise ValueError(
            f"Optimizer '{module_name}': checkpoint has {len(loaded_pg)} param_groups "
            f"but current optimizer has {len(current_pg)}"
        )
    for loaded_g, current_g in zip(loaded_pg, current_pg):
        loaded_g['params'] = current_g['params']
    # `sub_sd['optimizer']` may be absent on load: when the per-module state_dict
    # produced by DistributedOptimizer.state_dict() only contains `param_groups`
    # under the 'optimizer' key, `_extract_param_groups` removes it at save time
    # and the resulting empty dict can be dropped during dist_checkpointing
    # common-state save/load. Use setdefault so the restored param_groups land
    # in the right place regardless.
    sub_sd.setdefault('optimizer', {})['param_groups'] = loaded_pg


def _restore_param_state_sharding_type(sub_sd):
    """Load: restore param_state_sharding_type from ShardedObject key."""
    for k in list(sub_sd.keys()):
        if isinstance(k, str) and k.startswith('_mimo_param_state_sharding_type'):
            sub_sd['param_state_sharding_type'] = sub_sd.pop(k)
            break


def _restore_grad_scaler(sub_sd):
    """Load: restore grad_scaler from ShardedObject key."""
    for k in list(sub_sd.keys()):
        if isinstance(k, str) and k.startswith('_mimo_grad_scaler'):
            sub_sd['grad_scaler'] = sub_sd.pop(k)
            break


def _get_replica_id(pg_collection: Optional[ProcessGroupCollection]) -> tuple:
    """Build replica_id tuple for ShardedObject deduplication.

    Returns (tp_gtp_rank, pp_rank, dp_cp_rank) so only (0, 0, 0) within each
    module's parallelism group is the main replica; all other ranks
    in the same module are non-main replicas of the same object.
    """
    assert pg_collection is not None, "pg_collection required for checkpoint replica_id"
    assert (
        hasattr(pg_collection, 'tp') and pg_collection.tp is not None
    ), "pg_collection.tp must be set for checkpoint deduplication"
    assert (
        hasattr(pg_collection, 'pp') and pg_collection.pp is not None
    ), "pg_collection.pp must be set for checkpoint deduplication"
    assert (
        hasattr(pg_collection, 'dp_cp') and pg_collection.dp_cp is not None
    ), "pg_collection.dp_cp must be set for checkpoint deduplication"
    gtp_group = getattr(pg_collection, 'gtp_remat', None)
    gtp_rank = gtp_group.rank() if gtp_group is not None else 0
    gtp_size = gtp_group.size() if gtp_group is not None else 1
    tp_gtp_rank = pg_collection.tp.rank() * gtp_size + gtp_rank
    return (tp_gtp_rank, pg_collection.pp.rank(), pg_collection.dp_cp.rank())


def _optimizer_config_for_module(
    config: OptimizerConfig, module: torch.nn.Module
) -> OptimizerConfig:
    """Derive an optimizer config whose param-gather overlap matches the module's DDP config."""
    ddp_config = getattr(module, 'ddp_config', None)
    if ddp_config is None:
        raise ValueError("Active MIMO modules must be DDP-wrapped before optimizer construction.")
    overlap_param_gather = ddp_config.overlap_param_gather
    return replace(
        config,
        overlap_param_gather=overlap_param_gather,
        overlap_param_gather_with_optimizer_step=(
            config.overlap_param_gather_with_optimizer_step and overlap_param_gather
        ),
    )


def get_mimo_optimizer(mimo_model: "MimoModel", config: OptimizerConfig) -> MimoOptimizer:
    """Create optimizer for MimoModel with heterogeneous parallelism."""
    # Validate before module optimizer construction and its collectives.
    MimoOptimizer._validate_config(config)
    from megatron.core.optimizer import get_megatron_optimizer

    grid_map = mimo_model.mimo_config.module_to_grid_map
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

    lang_key = MIMO_LANGUAGE_MODULE_KEY

    module_infos: Dict[str, ModuleOptimizerInfo] = {}

    for module_name, grid in grid_map.items():
        optimizer = None
        pg_collection = None
        is_active = False

        if grid.is_current_rank_in_grid():
            if module_name == lang_key:
                module = mimo_model.language_model
            else:
                module = mimo_model.modality_submodules[module_name]

            if module is not None:
                pg_collection = getattr(unwrap_model(module), 'pg_collection', None)
                assert (
                    pg_collection is not None
                ), f"Module '{module_name}' must own a ProcessGroupCollection for optimizer setup"
                assert (
                    getattr(pg_collection, 'intra_dist_opt', None) is not None
                ), f"Module '{module_name}' must own an intra_dist_opt group for optimizer setup"
                # Parameter-group alignment and gradient statistics include all PP stages.
                # Keep empty local stages participating unless the entire module is frozen.
                has_trainable_params = torch.tensor(
                    any(parameter.requires_grad for parameter in module.parameters()),
                    dtype=torch.int64,
                    device="cuda",
                )
                torch.distributed.all_reduce(
                    has_trainable_params,
                    op=torch.distributed.ReduceOp.MAX,
                    group=pg_collection.intra_dist_opt,
                )
                is_active = bool(has_trainable_params.item())

            if is_active:
                module_config = _optimizer_config_for_module(config, module)
                assert module.ddp_config.num_distributed_optimizer_instances == 1, (
                    "MIMO optimizer does not yet support "
                    "num_distributed_optimizer_instances > 1. "
                    f"Module '{module_name}' has "
                    f"{module.ddp_config.num_distributed_optimizer_instances} instances."
                )
                optimizer = get_megatron_optimizer(
                    config=module_config,
                    model_chunks=[module],
                    pg_collection=pg_collection,
                    use_gloo_process_groups=False,
                    param_group_process_group=pg_collection.intra_dist_opt,
                )

        module_infos[module_name] = ModuleOptimizerInfo(
            optimizer=optimizer, grid=grid, pg_collection=pg_collection, is_active=is_active
        )

    return MimoOptimizer(module_infos, config)
