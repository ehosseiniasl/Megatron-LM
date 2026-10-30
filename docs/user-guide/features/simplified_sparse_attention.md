# Simplified sparse attention

Simplified sparse attention replaces the attention core while retaining the
ordinary query, key, value and output projections. A separate learned indexer
projects the normalized attention input into one query/key head. It scores causal
keys by a scaled dot product and selects the top-k keys for all attention heads.
The indexer has no positional rotation, learned head weighting or scoring ReLU.

The optional `simplified-sparse-attention>=0.1.0.dev6` package supplies the CuTe kernels.
Megatron Core owns projections, training policy and checkpoint integration; the
package owns sparse selection, attention and the indexer loss and gradients.
The package is imported only when this backend executes. Install a matching wheel
with `python -m pip install --no-deps simplified_sparse_attention-0.1.0.dev6-py3-none-any.whl`.
This version reuses compiled filtered-indexer kernels and stream-local workspaces
across changing packed sequence lengths, avoiding repeated shape-specific compilation.

## Training phases

In **phase 1**, `ssa_train_indexer_only=True` freezes the backbone, including
router state updated outside the optimizer. Attention remains dense. A KL loss
trains the indexer against the teacher distribution over all causal keys.
FlashAttention 4 computes the dense teacher output and log-sum-exp values reused
by the indexer loss. The teacher does not receive indexer-loss gradients.

In **phase 2**, `ssa_train_indexer_only=False` trains the backbone through sparse
attention and trains the indexer with KL over the selected keys. Indexer inputs
are detached in both phases, so the auxiliary objective cannot update the
backbone or its normalization weights. Indexer weights and projections remain
BF16, including inside an MXFP8 backbone.

`dsa_indexer_loss_coeff` scales both auxiliary loss and gradients. It must be
nonnegative and must be positive in phase 1. Mean-loss mode divides by the number
of real query tokens across the context-parallel group; per-token-loss mode leaves
normalization to the training loop. Storage padding contributes neither loss nor
indexer gradients.
Optional separate indexer clipping uses `ssa_separate_indexer_grad_clip` and
`ssa_indexer_clip_grad`, with the same optimizer step as the rest of the model.

### Model FLOP accounting

Throughput counts useful model GEMMs with two FLOPs per multiply-add. For each
unpadded sequence of length `S`, causal scoring has `S*(S+1)/2` pairs and sparse
attention has `S*k-k*(k-1)/2` pairs, where `k=min(topk,S)`. Phase 2 counts sparse
attention forward/backward, full causal indexer scoring, and selected-support
indexer gradients. The two indexer projections count forward and weight gradients,
without input gradients because their inputs are detached.

Phase 1 counts the frozen backbone forward only, dense teacher attention, and
full-support indexer training. With zero auxiliary coefficient in phase 2, only
the indexer forward work contributes. These are useful model FLOPs, not executed
hardware FLOPs: selection, elementwise operations, communication, optimizer work,
rematerialization and zero or frozen-backbone gradient propagation are excluded.
Other layer estimates retain their existing conventions, including the GDP
recurrent-core lower bound.

## Selection and supported inputs

Select `experimental_attention_variant="ssa"`. For hybrid stacks, use
`ssa_stack_spec` or `gdp_ssa_stack_spec` from
`megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_layer_specs`.
These specs select `SimplifiedSparseAttention` for attention layers. Other layers
retain their existing implementations. Ordinary attention remains the default.
`dsa_indexer_topk` controls selection size. The indexer architecture is fixed to
one learned query/key head of dimension 128.

The current integration supports:

| Property | Supported values |
| --- | --- |
| Layout | SBHD with batch size 1, or packed THD |
| Parallel attention | Fixed context parallelism with zigzag token partitioning; tensor model parallel size 1 |
| Attention heads | 16, 32 or 96 query heads; one shared K/V head |
| Attention head dimension | 256 |
| Indexer | One learned Q/K head, dimension 128 |
| Top-k | 512, 1024 or 2048 |
| Kernel inputs | BF16 on SM103 |
| Mask | Causal self-attention, with per-document causal masking for THD; no custom mask or bias |

## Packed sequences and context parallelism

Both training phases support context parallelism. Queries remain local, while
keys, values and indexer keys are gathered into global sequence order. Backward
uses reduce-scatter to return each rank's key/value and indexer-key gradients.
Communication preserves the BF16 activation dtype.

Packed inputs describe document boundaries through `PackedSeqParams`. Logical
lengths distinguish real tokens from physical storage padding at each document's
suffix. SSA derives and caches local row validity from these lengths and the CP
layout. Producers must retain both logical lengths and physical offsets; SSA
does not require a separate input-validity tensor in the model interface. A
language-model loss mask cannot supply this information: image or prompt tokens
can be valid attention inputs without a language-model target.

The shared packed-batch preparation aligns documents for zigzag CP and aligns
local rows for the configured projection precision. All token-aligned tensors
must use the same partition. In phase 1, each local query segment uses its own
document's causal key prefix for the dense teacher. In phase 2, sparse selection
is restricted to the same document and causal prefix.

MIMO preserves logical document lengths separately from padded storage offsets.
MTP shifts stop at real document ends. When linear layers use contiguous CP and
attention uses zigzag CP, the activation conversion adds and removes temporary
attention padding. That padding does not enter contiguous MoE layers. Existing
framework handling of source padding is unchanged. This attention integration
does not add router support for genuinely padded model inputs.

Attention dropout, sliding-window attention, QK clipping, logit recording,
decode caching and CUDA graph capture are not supported. Backward uses atomic
accumulation and does not support deterministic
mode. The kernel package must be version `0.1.0.dev6` or later with a
compatible CuTe runtime; phase 1 additionally requires FlashAttention 4.

## Checkpoints

A compatible dense checkpoint can initialize phase 1 with a model-only load
(`no_load_optim` or `finetune`). Only missing indexer entries retain their freshly
initialized values; backbone tensors still load strictly. A same-phase resume
restores the normal model, optimizer and RNG state. Changing phases requires a
fresh optimizer because the trainable parameter set changes. A model-only load
can preserve training progress; `finetune` or `pretrained_checkpoint` starts a
fresh schedule.
