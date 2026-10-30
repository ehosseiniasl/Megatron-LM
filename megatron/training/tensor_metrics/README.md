TensorMetrics is a system to observe parts of the training process and generate
metrics about it.  It consists of:

- An Observer, which provides hooks for the model to expose a tensor to metrics
- A user-extendable set of Metrics, which specify distributed computations on
  those tensors
- An Executor, which executes the distributed computations specified by each
  Metric on tensors supplied by the Observer

The goal is for adding metrics to be quick and low-friction, so that you can
quickly add them mid-investigation.  This directory contains metrics which have
proven useful or are good examples of metric-writing techniques.

## Usage

To use an existing metric, specify it along with the step interval to run it.  For example:

```
--tensor-metrics router-health:50 router-decision-entropy:100
```

### MoE output contributions

```sh
--tensor-metrics moe-contributions:100
```

This opt-in metric measures the actual forward outputs of the shared branch and
the selected routed experts, before they are added to the residual stream. Let
`S` be the shared output, `c_j` the weighted output of selected expert rank `j`,
and (without latent projections) `R = sum_j c_j`. All routing normalization/scaling and any shared-expert gate
are included. Expert rank is descending **final mixture weight among selected
experts**, not bias-corrected selection score, expert id, or output magnitude.
Equal weights use ascending expert id. A genuinely selected zero-weight expert
keeps its rank; dispatch alignment padding does not.

Each numbered MoE layer emits:

- `shared-rms`, `routed-rms`: RMS **token L2 norm**, `sqrt(E ||S||²)` and
  `sqrt(E ||R||²)`, using the same norm convention for all ranks and branches.
- `shared-over-routed-rms`: `sqrt(E ||S||² / E ||R||²)`.
- `shared-routed-alignment`: `E(S·R) / sqrt(E ||S||² E ||R||²)`.
- `top-over-last-rms`: `sqrt(E ||c_1||² / E ||c_k||²)`.
- `rank_j/rms`, `rank_j/energy-share`, `rank_j/mean-weight-share`, and
  `rank_j/zero-fraction` for every selected rank. Energy share divides each
  rank's squared output norm by the sum of the individual-expert squared norms.
  Mean weight share averages `w_j / sum_i w_i` over tokens, excluding the common
  router scale from the weight profile but retaining it in the output norms.
- `top-three-energy-share`: sum of energy shares for ranks 1 through `min(3,k)`.
- `tokens`, `shared-zero-fraction`, `routed-zero-fraction`, and
  `zero-weight-sum-fraction`. Tokens with all-zero mixture weights contribute
  zero to each mean weight share and are counted explicitly.
- `empty-population`, `zero-shared-energy`, `zero-routed-energy`,
  `zero-last-rank-energy`, and `zero-individual-energy`: flags for undefined
  denominators. An undefined ratio is **NaN**, not a large epsilon-dependent
  number; these diagnostic NaNs do not indicate a NaN in training itself.

Moments/counts are summed across microbatches and token-parallel populations
before ratios are taken. Neither per-token ratios nor per-rank ratios are
averaged. MTP prediction depths are separate, including when they share one
physical layer, with labels ending in `/mtp_1`, `/mtp_2`, etc. Expert padding is
excluded. All original model token positions are included, regardless of loss
mask or modality; this is not a valid-loss-token-only statistic. These measure
output magnitude, not causal importance to loss. In particular, individual
energy shares do not decompose `||R||²`, because expert outputs can cancel.

Supported layouts are **dropless `alltoall`** (including expert TP) and
**dropless `flex/hybridep` with expert TP=1 and equal per-rank token counts**,
including quantization alignment padding. The optional uneven-input padding
mode is not supported by this metric. HybridEP's static rank capacity is rejected because it can drop tokens.
Other Flex backends, AllGather, and capacity-based token dropping are unsupported.
Shared-expert overlap is supported only without latent projections. The MoE
router/preprocess/combine must execute eagerly; attention-only CUDA graphs are
allowed. Pipeline-parallel limitations of the tensor-metric observer still apply.

For latent MoE, shared/routed norms and alignment compare the actual model-space
outputs, **after** the combined routed output is normalized and up-projected.
Ranked-expert energies are measured in **latent space before summation and
normalization**; their tags (including ranked weight shares and zero flags) gain
an explicit `latent/` prefix, for example `latent/rank_1/rms` and
`latent/top-over-last-rms`. These are not individual post-projection energies.

HybridEP reuses the scalar channel of `combine_with_unpermute`, normally used
for router gradients, to return each weighted expert output's squared norm in
the ordinary combine call. A negative sentinel preserves exact-zero counts even
when tiny nonzero outputs' squared norms underflow. The original top-k selections
exclude dispatch padding. This requires no HybridEP/TE changes or additional
activation communication; transient storage includes the dense returned scalar
map `[tokens, experts]`. Diagnostics are detached and do not modify backward.

AlltoAll observations reuse weighted expert outputs already returned before
they merge. A local scalar-label permutation tracks rank; no extra expert
communication or `[tokens, topk, hidden]` gather is introduced. CUDA reductions
read the outputs with FP32 accumulation without a full FP32 activation copy,
then keep only `7 + 3*topk` additive values per layer/microbatch in FP64. Only
these summaries enter metric collectives. There are `15 + 4*topk` scalar tags per
layer/depth. Unscheduled steps and backward recomputation do no summary work.

#### Validation and large-run rollout

This metric does not require logit-space routing, calibration, or a modified
Transformer Engine/HybridEP build. It observes whichever experts and final
mixture weights the model actually uses. Enable it by adding
`moe-contributions:100` to the existing metric list; the interval is an example,
not a measured overhead guarantee.

The implementation has been exercised in CPU reference/observer tests, a
16-GPU HybridEP smoke using EP8 groups, and a 72-GPU Nano checkpoint continuation
using AlltoAll/EP64. The native HybridEP on/off check produced identical forward
outputs and gradients. The paired end-to-end training smoke was healthy but
was not bitwise deterministic. The Nano continuation produced eight complete
collections in each routing configuration, with distinct repeated-MTP depths.
These runs predate the standalone port to current `main`; they do not establish
production EP64 HybridEP correctness or Ultra-scale overhead. Both used one
expert-weight shard; Ultra's eight-way expert-weight rematerialization still
needs an integration check with this metric.

Before enabling this on a large model:

1. Run the CUDA reference tests at the model's hidden and latent widths (Ultra:
   10,240 and 3,072), then the native HybridEP on/off check with the deployment
   image, EP domain, permutation-fusion setting and quantization padding.
2. Exercise scheduled and unscheduled steps, repeated MTP calls, activation
   offloading/recomputation, and the actual GTP/expert-GTP layout. Compare model
   outputs and gradients with diagnostics disabled and verify each site's token
   count and the rank energy/weight-share sums. Suppress backward-recomputation
   observations, rather than counting them as another forward.
3. Measure peak allocated/reserved/device memory and whole-step time. HybridEP
   returns a dense FP32 `[local_tokens, experts]` diagnostic map: 8,192 tokens
   and 512 experts require 16 MiB for that map alone. Rank selection, scalar
   payloads and native communication workspaces add further transient memory.
   The large expert output matrices are read, not copied to FP32 or retained
   across layers. The logging timer excludes this forward work.
4. Start with one explicitly sampled iteration in an isolated replay, then a
   sparse cadence. A configuration with 68 main MoE sites and two repeated-MTP
   invocations emits 3,850 scalars at top-10, not one scalar per physical expert.
   Large-grad-norm reruns can force collection outside that cadence.

The configured guards reject token dropping, uneven HybridEP inputs, HybridEP
expert TP greater than one, latent shared-expert overlap, and captured MoE
router/preprocess paths. The observer also rejects pipeline parallelism and
full-iteration CUDA graphs. Latent energy shares remain pre-normalization,
pre-projection diagnostics; they are not a decomposition of the final output
or of the loss. Intentional NaNs for zero denominators must be interpreted with
the accompanying flags.

With `--rerun-large-grad-norm-steps`, a large-grad-norm rerun collects all configured
tensor metrics even if that step is outside their intervals or explicit iteration
schedule. Forward and backward observations come from the replay, and metrics are
emitted once before the optimizer update. Subsequent steps keep their normal
schedules. The replay uses the existing `tensor_metrics` fault-tolerance timeout
section when step monitoring is active.

For GDP gate statistics, start with:

```sh
--tensor-metrics gdp-g-stats:1000 gdp-beta-stats:1000
```

`gdp-g-stats` emits `mean` and population `std` (`ddof=0`) per layer/global value
head. `gdp-beta-stats` emits approximate `p05`, `p50`, and `p95` separately per
Householder update, using a fixed 1,024-bin histogram over `[0, 1]`. Example tags:

```
tensor-metrics-breakdowns/gdp-g-stats/decoder.layers.0/head_0/std
tensor-metrics-breakdowns/gdp-beta-stats/decoder.layers.0/head_0/update_1/p50
```

The source layouts are `gdp_g: [batch, token, head]` and
`gdp_beta: [batch, token, update, head]`. Both reduce only batch/token axes;
updates and heads are never pooled. This pair emits
`2 + 3 * gdp_num_householder` scalars per layer/head.

All microbatches in a selected iteration contribute, including padding if present;
there is no accumulation across iterations. Statistics use the actual activated
gates and sample-weighted moments/counts. Compact detached contributions survive
until logging; unscheduled steps and checkpoint recomputation do no summary work.
Invalid values are not filtered. There are no nonfinite counters, histograms,
percentiles, key overlap, state replay, or rank/SVD calculations.

Headwise CP gathers head partitions; chunkwise CP reduces token populations for
the same heads. DP/GTP are sample populations and TP heads remain distinct.
Head IDs are global value-head IDs; Q/K group is
`head // (mamba_num_heads / mamba_num_groups)`. The existing run configuration
records these dimensions, update count, backend, and CP mode/layout.

Existing restrictions remain: PP=1 and eager GDP observation sites; no captured
`mamba` or full-iteration CUDA graphs. Hooks cover `GatedDeltaProductMixer`,
including one update, rather than the separate GDN/GDN2 modules. Benchmark the
actual workload before choosing cadence: the finalization timer excludes forward
preparation, and 1,000 steps is a starting point, not an overhead guarantee.

Metric names identify the source and statistic, while result labels identify aggregation scopes.
The opt-in `wgrad-summary` metric reports the signed mean, mean absolute value, and root mean
square of finalized model gradients separately by layer, by parameter family, and globally.
Families pool matching parameters across layer and expert indices while preserving module
structure and semantic components. Each statistic weights gradient elements equally; RMS is
computed from pooled squared gradients, not averaged from per-parameter RMS values. These are
separate layer and family breakdowns, not layer-by-family intersections. Enable it explicitly
with `--tensor-metrics wgrad-summary:INTERVAL`.
The opt-in `wgrad-summary-individual` metric reports the mean, mean absolute value, and root mean
square of each finalized model gradient. It is intended for targeted diagnosis because its
logging cardinality grows with the model.

The opt-in `residual-summary` and `dresidual-summary` metrics compact each residual row to its
accumulator L2 norm, contribution L2 norm, and accumulator/contribution cosine similarity in one
tensor program. They report the root mean square, p75, p90, p95, and maximum of each
per-token population for every layer and globally. RMS and maximum are exact; quantiles use
distributed histograms with one power-of-two bin per observed norm octave and 20 linear bins over
the fixed correlation range `[-1, 1]`. `residual-summary` also reports the same norm statistics for
the final decoder-layer output under `last-layer/{statistic}`. When the decoder uses a wide
residual stream, it reports the activation after the stream readout and immediately before the
final layer norm under `last-layer-wide-readout/{statistic}`. Use
`residual-summary-by-token-type` and
`dresidual-summary-by-token-type` for the same distributions within each MIMO token category. The
base and token-type summaries reuse the same fused row calculation when both are enabled, and
uneven category populations reduce the same compact histogram state. The Mistral MIMO data adapter
supplies five categories: `image`, `special-image`, `special-text`, `pad`, and `text`. Token
metadata follows the same context- and sequence-parallel partition as the residual stream; every
category is emitted even when its local shard is empty. With CP enabled, token-type summaries
require matching linear and attention CP layouts because the categories do not yet follow
per-layer layout conversions. With five MIMO categories and 92 layers, each fused token-type
summary can emit up to 8,370 scalar keys per sampled iteration, including layer and global results.
Select a sparse interval and account for extra samples requested with
`--tensor-metrics-iterations`.

Optimizer-backed metrics treat attention QKV and SSM input projections as their semantic component
matrices. Component names are inserted before the final parameter suffix, for example
`self_attention.linear_qkv.query.weight` and `mixer.in_proj.dt.weight`. The component mapping is
supplied by the owning module and remains valid through TP, GTP, and flat DistributedOptimizer
sharding. The observer prepares one fused tensor at a time and releases its semantic copies before
advancing.

`logging-rank-local-expert-0-relu-health` is a deliberately local grouped-expert diagnostic. It
observes only `local_experts.0` on the last global rank, where Megatron creates its TensorBoard and
W&B writers, and requests no cross-rank communication. For each numbered MoE layer it accumulates
all observations in the selected iteration and emits the mean, population standard deviation,
p01, p05, and p50 of the rank-local per-neuron pre-ReLU means. For the per-neuron exact-zero output
fraction it emits only p50, p90, and p99. The `global` results pool those local neuron
populations across observed layers, without incorporating any other rank or local expert. ReLU and
squared ReLU have the same zero set. The metric currently requires the non-op-fuser,
non-grouped-tensor grouped-expert path, and those expert observation sites must remain outside
local CUDA graphs.


To sample MIMO training loss by configured data source, add:

```
--tensor-metrics train-loss-by-data-source:100
```

This supports the MIMO Mistral and mock loaders, including dense and packed THD
batches. Document provenance supplies a source ID for each token; the ordinary
target loss mask selects scored tokens. The metric sums loss and token counts
over all microbatches and data/GTP ranks before dividing, and counts TP replicas
once. It reports the main LM loss, excluding MTP and auxiliary losses.

Results use `tensor-metrics/train-loss-by-data-source/source/<index>-<name>` and a
token-weighted `tensor-metrics/train-loss-by-data-source/global` cross-check. Indices
follow training-config source order and names derive from source paths. Sources
with no scored tokens in the sampled step are omitted. Only the sampled step is
measured, not the interval between samples. Per-source validation remains separate.

This path requires CP=1 and the observer's existing PP=1 / no-full-iteration-CUDA-graph
restrictions. Loss bucketing runs only on scheduled steps; enabling the metric
adds token-aligned CPU source metadata to every training batch for loader prefetch.

## Dense tensor and sequence parallelism

Dense TP with sequence parallelism is supported by router diagnostics. On sampled
steps, the router combines raw per-sequence score sums, assignment counts, overlap,
selected mass, and top-k boundary statistics across its explicit TP group before
normalizing them. This uses one FP32 all-reduce of `batch_size * (3 * num_experts + 5)`
elements per observed router invocation, without gathering token-by-expert tensors.
The resulting complete-sequence summaries and expert biases are TP replicas, so
downstream health, expert-state, sequence-auxiliary, and routing-balance metrics
count each sequence once. Batch entries must correspond across SP ranks; sequence
statistics retain the same packed-block semantics as TP=1. No diagnostic collective
is added on unsampled steps or when sequences are already local.

Expert TP is independent of dense TP: `moe-contributions` with HybridEP still
requires expert TP=1, dropless routing, and equal per-rank token counts. Context
parallel router diagnostics remain unsupported. Existing pipeline-parallel and
CUDA-graph restrictions are unchanged.

The model's own TP and precision constraints still apply. In particular, GDP
head/group counts must be divisible by TP, and MXFP8 requires TP-local GEMM dimensions
divisible by 32. A fused GDP input projection can violate the latter after
increasing TP even when its head counts divide evenly; use a compatible shape
or an explicit BF16 precision override for that projection.

## TensorMetric interface

This is a high-level description; for details, refer to the code.

A TensorMetric specifies a distributed computation in a way that can be
efficiently executed on a running model.  We especially care about the size and
timing of computation and communication, and we need to ensure that we don't
keep large raw tensors around beyond their normal lifespan.

Each metric specifies a set of `source_kinds` that it cares about, such as
`parameter` or `router_diagnostics`.  During model execution, we will check
whether any enabled metric is observing a certain kind.  If so, we will call
`metric.accepts()` with metadata about the site, such as the module name.  If
it returns `True`, then that metric cares about that observation site.

In that case, we will immediately call `metric.prepare()` with the relevant
tensor. This is a local-only function which should reduce the main tensor into
a smaller tensor that we will hold onto until the end of the step.  For
example, for an L2 norm, this might compute the sum of squares, but not the
final sqrt since we may only have a shard of the complete tensor.

At the end of the step, all prepared tensors are supplied together to
`metric.start()`. A global L2 norm can reduce all of them into one result; a
per-layer metric can organize them by layer and start one result for each
layer; and a multi-granularity metric can reuse the same prepared contribution
in tensor, family, layer, and global results. Each tensor carries metadata,
including its observation-site name, for making those distinctions.

`metric.start()` must not call collectives internally. It returns completed
`MetricResult` objects or `CollectiveStage` objects specifying an
AllGather or AllReduce.

Each tensor is given an abstract representation of the ways it is sharded.  For
example, a weight tensor might be sharded along TP with `Shard(dim=1)` and
along DP with `Replica`.  A gradient in the distributed optimizer might have
been flattened and partitioned, but not cleanly along any individual dimension,
which is represented with `FlatShard(logical_shape, start, end)`.  An expert
might only exist on one rank, which is represented with `Owned(rank=...)`. Metrics
are expected to handle these cases explicitly, but they don't need to know what
underlying parallelism caused a particular sharding relation.

`MetricTensor.logical_shape` retains the observed tensor's shape before optimizer
storage sharding, including on placeholder ranks. It survives `with_tensor()` so
a metric can use it after replacing a large tensor with a compact contribution.

This can have nontrivial complexity, but many metrics can use a convenience
subclass called LogicalReductionMetric that reduces across these in the
"logical" manner: ignore other replicas, allow logical grouping of tensors, and
use the same reduction function across shards and other prepared tensors. Other
metrics need the full power: for example, a metric to measure replica drift has
special requirements for reducing across `Replica` dimensions.

The executor runs the collectives that were requested with `metric.start()`
(batching any compatible collectives). It then calls `metric.resume()` once for
each completed stage. This function again returns a `CollectiveStage` or a final
`MetricResult`. This continues in a loop until every metric computation has
returned its `MetricResult`.

Finally, the observer writes those MetricResults to TensorBoard or W&B. All
results remain available in both sinks. Global results use the
`tensor-metrics-global` section, while layer, family, and other detailed results
use `tensor-metrics-breakdowns`. The latter section can normally remain
collapsed and expanded when a global result warrants investigation.
All default metric logging uses this layout and batches TensorBoard writes.
Parameter-family labels replace layer and expert indices with the literal `ALL`,
for example `family/decoder.layers.ALL.mlp.linear_fc1.weight`.


| TensorMetric Method  | Description                                                                     |
| -------------------- | ------------------------------------------------------------------------------- |
| `accepts()`          | Filter whether this value is relevant to this metric.                           |
| `prepare()`          | Perform local reduction of the parameter.                                       |
| `start()`            | Organize prepared values and start reducing across ranks.                       |
| `resume()`           | Continue reducing across ranks (called in a loop until it returns MetricResult) |

Important notes:

- While no metric functions should call collectives directly, it's still
  important that the returned collective stages match (including tensor
  sizes) across ranks, or else your job will hang.

- `prepare()` and `start()` receive batches, allowing local batched kernels
  where useful. The executor separately packs compatible requests across
  collective stages before resuming each computation independently.

- If you need to observe a tensor which is not yet exposed, try to add it with
  minimal changes to core.  Metrics can include a lot of complexity, try to
  keep that in the training/tensor_metrics directory when possible.

- Cudagraph compatibility can be a challenge.  If you're observing something
  inside a cudagraph'd section, be especially careful of this.
