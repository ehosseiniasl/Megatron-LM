# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import torch

from megatron.core import utils
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.ssm.context_parallel.chunkwise import PackedSequenceCPMetadata
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule, TwoStageAttentionLayer
from megatron.core.transformer.moe.shared_experts import set_tensor_grad_fn_sequence_sr
from megatron.core.transformer.residual_connection import ResidualRead, build_residual_read
from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_layer import StreamwiseSigmoidWideResidualRead
from megatron.core.typed_torch import apply_module


def _get_offloading_interface():
    """Get the fine-grained activation offloading interface lazily."""
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        FineGrainedActivationOffloadingInterface,
    )

    return FineGrainedActivationOffloadingInterface


def group_layers_into_shortcut_blocks(
    layers: torch.nn.ModuleList,
    layer_type_list: Sequence[str],
    config: TransformerConfig,
    *,
    pp_layer_offset: int = 0,
) -> torch.nn.ModuleList:
    """Group physical layers into their registered shortcut-block hierarchy.

    Grouping updates the layer names through the returned ``ModuleList`` hierarchy. Layers not
    followed by an MoE remain direct children of the returned ``ModuleList``.

    Args:
        layers: Physical layers in execution order.
        layer_type_list: Physical layer symbols in execution order.
        config: Transformer configuration controlling shortcut scheduling.
        pp_layer_offset: Global offset of the first physical layer on this pipeline stage.

    Returns:
        The registered logical layers.

    Raises:
        ValueError: If an MoE has an unsupported predecessor or a shortcut pair is split across
            pipeline stages.
    """
    if pp_layer_offset > 0 and layer_type_list and layer_type_list[0] == LayerSymbols.MOE:
        raise ValueError(
            "Shortcut MoE pairs cannot cross pipeline-stage boundaries; this pipeline stage "
            "begins with an MoE layer."
        )

    grouped_layers = torch.nn.ModuleList()
    physical_index = 0
    while physical_index < len(layers):
        next_is_moe = (
            physical_index + 1 < len(layers)
            and layer_type_list[physical_index + 1] == LayerSymbols.MOE
        )
        if not next_is_moe:
            grouped_layers.append(layers[physical_index])
            physical_index += 1
            continue

        compute_layer = layers[physical_index]
        paired_type = layer_type_list[physical_index]
        supported_predecessors = {
            LayerSymbols.MAMBA,
            LayerSymbols.GDN,
            *LayerSymbols.ATTENTION_LAYERS,
        }
        if paired_type not in supported_predecessors:
            raise ValueError(
                "Shortcut MoE must be preceded by a Mamba, GDN, or supported attention layer"
            )

        supports_two_stage = (
            isinstance(compute_layer, TwoStageAttentionLayer)
            and compute_layer.supports_two_stage_attention()
        )
        if not supports_two_stage:
            raise ValueError(
                f"Shortcut compute layer {paired_type!r} does not support two-stage attention"
            )
        moe_layer = layers[physical_index + 1]
        grouped_layers.append(
            ShortcutMoEBlock(compute_layer, moe_layer, overlap_a2a=config.moe_shortcut_parallel)
        )
        physical_index += 2

    return grouped_layers


class ShortcutMoEBlock(MegatronModule):
    """Own and execute one compute-layer/shortcut-MoE pair."""

    _parallel_stream: torch.cuda.Stream | None = None

    @classmethod
    def _get_a2a_overlap_stream(cls) -> torch.cuda.Stream:
        """Return the process-wide high-priority shortcut stream."""
        if cls._parallel_stream is None:
            cls._parallel_stream = torch.cuda.Stream(priority=-1)
        return cls._parallel_stream

    def __init__(self, compute_layer, moe_layer, overlap_a2a: bool):
        super().__init__(compute_layer.config)

        self.overlap_mode = overlap_a2a
        self.layer_number = compute_layer.layer_number
        self.attn_layer_num = compute_layer.layer_number - 1
        self.moe_layer_num = moe_layer.layer_number - 1
        self.is_first_layer = getattr(compute_layer, "is_first_layer", False)
        self.is_last_layer = getattr(moe_layer, "is_last_layer", False)
        self.tp_group = moe_layer.mlp.tp_group
        self.compute_layer = compute_layer
        self.moe_layer = moe_layer
        self.off_interface = _get_offloading_interface()
        self.shortcut_pre_mlp_layernorm_checkpoint = None

        compute_is_mtp = bool(getattr(compute_layer, "is_mtp_layer", False))
        moe_is_mtp = bool(getattr(moe_layer, "is_mtp_layer", False))
        if compute_is_mtp != moe_is_mtp:
            raise ValueError("A ShortcutMoE pair must agree on whether it belongs to MTP.")
        self.is_mtp_layer = moe_is_mtp

        self.shortcut_residual_read = None
        if self.config.wide_residual is not None and not self.is_mtp_layer:
            self.shortcut_residual_read = build_residual_read(
                ModuleSpec(module=StreamwiseSigmoidWideResidualRead),
                config=self.config,
                layer_number=moe_layer.layer_number,
                branch_name="shortcut_routed",
                pg_collection=moe_layer.pg_collection,
                name=None,
            )
        if self.shortcut_residual_read is not None:
            outer_connection = getattr(moe_layer, "residual_connection_mlp", None)
            if outer_connection is None:
                raise ValueError(
                    "A wide-residual shortcut read requires an outer MoE residual connection."
                )
            if (
                self.shortcut_residual_read.residual_stream_hidden_size
                != outer_connection.residual_stream_hidden_size
            ):
                raise ValueError(
                    "The shortcut read and outer MoE residual connection must consume the "
                    "same residual-stream width."
                )

        # The shortcut path uses the same normalization implementation and configuration as
        # the MoE path, but owns an independent parameter.
        self.shortcut_pre_mlp_layernorm = build_module(
            moe_layer.submodules_config.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.shortcut_post_norm = (
            build_module(
                moe_layer.submodules_config.pre_mlp_layernorm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
            if self.config.moe_shortcut_post_norm
            else IdentityOp()
        )
        self.recompute_shortcut_pre_mlp_layernorm = (
            self.config.recompute_granularity == "selective"
            and "shortcut_pre_mlp_layernorm" in (self.config.recompute_modules or [])
            and not isinstance(self.shortcut_pre_mlp_layernorm, IdentityOp)
        )
        self.offload_shortcut_post_norm = (
            self.config.fine_grained_activation_offloading
            and "shortcut_post_norm" in (self.config.offload_modules or [])
            and not isinstance(self.shortcut_post_norm, IdentityOp)
        )
        self.route_ready_event = torch.cuda.Event() if self.overlap_mode else None

    def _read_shortcut_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        recompute_context: ResidualStreamRecomputeContext | None,
    ) -> torch.Tensor:
        """Read the pair input used only by shortcut routing."""

        residual_read: ResidualRead | None = self.shortcut_residual_read
        if residual_read is None:
            return hidden_states
        if recompute_context is not None and not residual_read.read_is_identity:
            return recompute_context.checkpoint(apply_module(residual_read), hidden_states)
        return apply_module(residual_read)(hidden_states)

    def _moe_router_preprocess(
        self,
        shortcut_hidden,
        padding_mask=None,
        packed_seq_params=None,
        recompute_context: ResidualStreamRecomputeContext | None = None,
    ):
        """Run shortcut normalization, routing, and dispatch preprocessing."""
        replay_pre_norm = recompute_context is not None and not isinstance(
            self.shortcut_pre_mlp_layernorm, IdentityOp
        )
        if replay_pre_norm:
            pre_norm_checkpoint = CheckpointWithoutOutput(ckpt_manager=recompute_context.manager)
            shortcut_input = pre_norm_checkpoint.checkpoint(
                apply_module(self.shortcut_pre_mlp_layernorm), shortcut_hidden
            )
        elif (
            self.training
            and not InferenceMode.is_active()
            and self.recompute_shortcut_pre_mlp_layernorm
            and recompute_context is None
        ):
            self.shortcut_pre_mlp_layernorm_checkpoint = CheckpointWithoutOutput()
            shortcut_input = self.shortcut_pre_mlp_layernorm_checkpoint.checkpoint(
                apply_module(self.shortcut_pre_mlp_layernorm), shortcut_hidden
            )
        else:
            shortcut_input = apply_module(self.shortcut_pre_mlp_layernorm)(shortcut_hidden)
        if recompute_context is not None and self.config.fine_grained_activation_offloading:
            self.off_interface.mark_not_offload(shortcut_input)
        shortcut_input, padding_mask, _ = self.moe_layer._maybe_unflatten_for_moe(
            shortcut_input, padding_mask, packed_seq_params
        )
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()
        probs, routing_map = self.moe_layer.mlp.route(shortcut_input, padding_mask)
        return self.moe_layer.mlp.preprocess(shortcut_input, probs, routing_map)

    def _moe_shared_experts(
        self,
        hidden_states,
        *,
        padding_mask=None,
        packed_seq_params=None,
        recompute_context: ResidualStreamRecomputeContext | None = None,
    ):
        """Run the paired MoE layer's pre-MLP norm and shared experts."""
        pre_mlp_output, residual, mlp_state = self.moe_layer._pre_mlp_layernorm_and_residual(
            hidden_states, residual_stream_recompute_context=recompute_context
        )
        pre_mlp_output, _, moe_unflatten_mbs = self.moe_layer._maybe_unflatten_for_moe(
            pre_mlp_output, padding_mask, packed_seq_params
        )
        shared_expert_output = self.moe_layer.mlp.shared_experts_compute(pre_mlp_output)
        return shared_expert_output, residual, mlp_state, moe_unflatten_mbs

    def _postprocess(
        self,
        combined_output,
        shared_expert_output,
        residual,
        mlp_state,
        *,
        moe_packed_seq_params=None,
        moe_unflatten_mbs=None,
        moe_recompute_context: ResidualStreamRecomputeContext | None = None,
    ):
        """Join routed/shared output, apply shortcut post-norm, and finish residual/BDA."""
        output = self.moe_layer.mlp.postprocess(combined_output, shared_expert_output)
        post_norm_input = output
        replay_post_norm = moe_recompute_context is not None and not isinstance(
            self.shortcut_post_norm, IdentityOp
        )
        post_norm_manager = self.off_interface(
            self.training
            and not InferenceMode.is_active()
            and self.offload_shortcut_post_norm
            and moe_recompute_context is None,
            post_norm_input,
            "shortcut_post_norm",
        )
        with post_norm_manager as post_norm_input:
            if replay_post_norm:
                output = moe_recompute_context.checkpoint(
                    apply_module(self.shortcut_post_norm), post_norm_input
                )
            else:
                output = apply_module(self.shortcut_post_norm)(post_norm_input)
        output = post_norm_manager.group_offload(output, forced_released_tensors=[post_norm_input])
        if moe_recompute_context is not None and self.config.fine_grained_activation_offloading:
            self.off_interface.mark_not_offload(output)
        output = self.moe_layer._maybe_reflatten_from_moe(
            output, moe_packed_seq_params, moe_unflatten_mbs
        )
        output = self.moe_layer._apply_mlp_bda_step(
            (output, None),
            residual,
            mlp_state,
            residual_stream_recompute_context=moe_recompute_context,
        )
        return output[0] if isinstance(output, tuple) else output

    def _launch_dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, async_op: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch dispatch on the current stream or the shortcut side stream."""
        if not async_op:
            return self.moe_layer.mlp.dispatch(hidden_states, probs)

        assert self.route_ready_event is not None
        dispatch_stream = self._get_a2a_overlap_stream()
        dispatch_stream.wait_event(self.route_ready_event)
        hidden_states.record_stream(dispatch_stream)
        probs.record_stream(dispatch_stream)

        with torch.cuda.stream(dispatch_stream):
            return self.moe_layer.mlp.dispatch(hidden_states, probs)

    def _wait_dispatch(
        self, dispatched_input: torch.Tensor, dispatched_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wait for dispatch and return its outputs on the main stream."""
        assert self.overlap_mode

        torch.cuda.current_stream().wait_stream(self._get_a2a_overlap_stream())
        dispatched_input.record_stream(torch.cuda.current_stream())
        dispatched_probs.record_stream(torch.cuda.current_stream())
        return dispatched_input, dispatched_probs

    def _launch_combine(self, output: torch.Tensor, async_op: bool = False) -> torch.Tensor:
        """Launch combine on the current stream or the shortcut side stream."""
        if not async_op:
            return self.moe_layer.mlp.combine(output)

        combine_stream = self._get_a2a_overlap_stream()
        combine_stream.wait_stream(torch.cuda.current_stream())
        output.record_stream(combine_stream)
        with torch.cuda.stream(combine_stream):
            return self.moe_layer.mlp.combine(output)

    def _wait_combine(self, combined_output: torch.Tensor) -> torch.Tensor:
        """Wait for the asynchronous combine and return its output on the main stream."""
        torch.cuda.current_stream().wait_stream(self._get_a2a_overlap_stream())
        combined_output.record_stream(torch.cuda.current_stream())
        return combined_output

    def _forward_compute_atomic(
        self,
        hidden_states,
        *,
        attention_mask,
        inference_context,
        rotary_pos_emb,
        sequence_len_offset,
        packed_seq_params,
        padding_mask,
    ):
        """Run the compute layer's cache-aware atomic forward for inference."""

        compute_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "inference_context": inference_context,
            "rotary_pos_emb": rotary_pos_emb,
            "packed_seq_params": packed_seq_params,
        }
        if not isinstance(self.compute_layer, MambaLayer):
            compute_kwargs.update(
                sequence_len_offset=sequence_len_offset, padding_mask=padding_mask
            )
        output = self.compute_layer(**compute_kwargs)
        return output[0] if isinstance(output, tuple) else output

    def _forward_inference_serial(
        self,
        hidden_states,
        *,
        shortcut_hidden,
        attention_mask,
        inference_context,
        rotary_pos_emb,
        sequence_len_offset,
        compute_packed_seq_params,
        moe_packed_seq_params,
        padding_mask,
        quant_context_factory,
        quant_config,
        prepare_moe_input,
        compute_recompute_context,
        moe_recompute_context,
    ):
        """Run Shortcut-MoE serially while preserving attention/SSM inference state."""

        if compute_recompute_context is not None or moe_recompute_context is not None:
            raise RuntimeError("Shortcut-MoE inference cannot use residual-stream recomputation.")
        if (
            self.config.mlp_chunks_for_prefill > 1
            and inference_context is not None
            and not inference_context.is_decode_only()
        ):
            raise RuntimeError(
                "Shortcut-MoE inference does not support mlp_chunks_for_prefill greater than 1."
            )
        if shortcut_hidden is None:
            shortcut_hidden = hidden_states

        # Route from X_l (or the dedicated wide-residual read R_s(X_l)) before the compute
        # layer transforms the main residual stream. No side-stream overlap is used in inference.
        with quant_context_factory(quant_config, self.moe_layer_num):
            shortcut_hidden = self._read_shortcut_hidden(shortcut_hidden, recompute_context=None)
            route_input, route_probs = self._moe_router_preprocess(
                shortcut_hidden=shortcut_hidden,
                padding_mask=padding_mask,
                packed_seq_params=moe_packed_seq_params,
                recompute_context=None,
            )

        # The two-stage compute methods are training-only for recurrent layers. The ordinary
        # atomic forward updates dynamic attention KV caches and Mamba/GDP recurrent states.
        with quant_context_factory(quant_config, self.attn_layer_num):
            compute_output = self._forward_compute_atomic(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=compute_packed_seq_params,
                padding_mask=padding_mask,
            )
        if prepare_moe_input is not None:
            compute_output = prepare_moe_input(compute_output)

        with quant_context_factory(quant_config, self.moe_layer_num):
            dispatched_input, dispatched_probs = self._launch_dispatch(
                route_input, route_probs, async_op=False
            )
            routed_output, _ = self.moe_layer.mlp.routed_experts_compute(
                dispatched_input, dispatched_probs
            )
            combined_output = self._launch_combine(routed_output, async_op=False)
            shared_expert_output, residual, mlp_state, moe_unflatten_mbs = self._moe_shared_experts(
                compute_output,
                padding_mask=padding_mask,
                packed_seq_params=moe_packed_seq_params,
                recompute_context=None,
            )
            return self._postprocess(
                combined_output,
                shared_expert_output,
                residual,
                mlp_state,
                moe_packed_seq_params=moe_packed_seq_params,
                moe_unflatten_mbs=moe_unflatten_mbs,
                moe_recompute_context=None,
            )

    def forward(
        self,
        hidden_states,
        attention_mask,
        inference_context,
        rotary_pos_emb,
        sequence_len_offset,
        compute_packed_seq_params,
        moe_packed_seq_params,
        padding_mask,
        quant_context_factory,
        quant_config,
        shortcut_hidden=None,
        compute_packed_sequence_cp_metadata: PackedSequenceCPMetadata | None = None,
        prepare_moe_input: Callable[[torch.Tensor], torch.Tensor] | None = None,
        compute_recompute_context: ResidualStreamRecomputeContext | None = None,
        moe_recompute_context: ResidualStreamRecomputeContext | None = None,
    ):
        """Run the eager schedule with each physical layer's quantization context."""
        utils.ensure_params_ready(self.parameters())
        select_token_dispatcher = getattr(self.moe_layer.mlp, "select_token_dispatcher", None)
        if select_token_dispatcher is not None:
            select_token_dispatcher()

        if InferenceMode.is_active():
            return self._forward_inference_serial(
                hidden_states,
                shortcut_hidden=shortcut_hidden,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                compute_packed_seq_params=compute_packed_seq_params,
                moe_packed_seq_params=moe_packed_seq_params,
                padding_mask=padding_mask,
                quant_context_factory=quant_context_factory,
                quant_config=quant_config,
                prepare_moe_input=prepare_moe_input,
                compute_recompute_context=compute_recompute_context,
                moe_recompute_context=moe_recompute_context,
            )

        if (compute_recompute_context is None) != (moe_recompute_context is None):
            raise ValueError("Shortcut replay requires contexts for both layers in the pair.")
        if compute_recompute_context is not None:
            assert moe_recompute_context is not None
            if compute_recompute_context.manager is not moe_recompute_context.manager:
                raise ValueError("A shortcut pair must share one residual replay manager.")
            if compute_recompute_context.is_block_end:
                raise ValueError("A residual replay block cannot end at the shortcut X_mid state.")

        if shortcut_hidden is None:
            shortcut_hidden = hidden_states

        # Launch the moe_router
        with quant_context_factory(quant_config, self.moe_layer_num):
            shortcut_hidden = self._read_shortcut_hidden(
                shortcut_hidden, recompute_context=compute_recompute_context
            )
            route_input, route_probs = self._moe_router_preprocess(
                shortcut_hidden=shortcut_hidden,
                padding_mask=padding_mask,
                packed_seq_params=moe_packed_seq_params,
                recompute_context=compute_recompute_context,
            )
            if self.overlap_mode:
                assert self.route_ready_event is not None
                self.route_ready_event.record(torch.cuda.current_stream())

        # Launch the input and attn of the attention layer
        with quant_context_factory(quant_config, self.attn_layer_num):
            compute_kwargs = dict(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=compute_packed_seq_params,
                residual_stream_recompute_context=compute_recompute_context,
            )
            if compute_packed_sequence_cp_metadata is not None:
                compute_kwargs["packed_sequence_cp_metadata"] = compute_packed_sequence_cp_metadata
            paired_state = self.compute_layer.forward_pre_attn_and_core_attn(**compute_kwargs)
        if torch.is_tensor(paired_state):
            paired_state = (paired_state,)
        elif not isinstance(paired_state, tuple):
            raise TypeError("Shortcut stage one must return a tensor or a tuple of tensors.")
        if not paired_state:
            raise RuntimeError("Shortcut stage one returned an empty state.")

        # Launch the dispatch, experts, and combine
        with quant_context_factory(quant_config, self.moe_layer_num):
            dispatched_input, dispatched_probs = self._launch_dispatch(
                route_input, route_probs, async_op=self.overlap_mode
            )
            if self.overlap_mode:
                dispatched_input, dispatched_probs = self._wait_dispatch(
                    dispatched_input, dispatched_probs
                )

            output, _ = self.moe_layer.mlp.routed_experts_compute(
                dispatched_input, dispatched_probs
            )
            combined_output = self._launch_combine(output, async_op=self.overlap_mode)
            if self.shortcut_pre_mlp_layernorm_checkpoint is not None:
                self.shortcut_pre_mlp_layernorm_checkpoint.discard_output_and_register_recompute(
                    combined_output
                )
                self.shortcut_pre_mlp_layernorm_checkpoint = None

        # launch the output layer of the attention layer
        with quant_context_factory(quant_config, self.attn_layer_num):
            attn_layer_output = self.compute_layer.forward_post_core_attn(
                *paired_state, residual_stream_recompute_context=compute_recompute_context
            )
            if isinstance(attn_layer_output, tuple):
                attn_layer_output = attn_layer_output[0]
        if prepare_moe_input is not None:
            attn_layer_output = prepare_moe_input(attn_layer_output)

        # launch the moe shared experts and combine attn and moe layer outputs
        with quant_context_factory(quant_config, self.moe_layer_num):
            shared_expert_output, residual, mlp_state, moe_unflatten_mbs = self._moe_shared_experts(
                attn_layer_output,
                padding_mask=padding_mask,
                packed_seq_params=moe_packed_seq_params,
                recompute_context=moe_recompute_context,
            )
            if self.overlap_mode:
                combined_output = self._wait_combine(combined_output)
            # Ensure the combine autograd node is scheduled first before shared_experts
            set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)
            return self._postprocess(
                combined_output,
                shared_expert_output,
                residual,
                mlp_state,
                moe_packed_seq_params=moe_packed_seq_params,
                moe_unflatten_mbs=moe_unflatten_mbs,
                moe_recompute_context=moe_recompute_context,
            )
