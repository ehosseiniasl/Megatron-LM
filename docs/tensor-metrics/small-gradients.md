# Small-gradient fractions

These opt-in metrics use the same optimizer-backed gradient snapshot and semantic tensor
boundaries as `wgrad-zero-fraction` and `wgrad-summary-individual`:

| Metric | Population and output |
| --- | --- |
| `wgrad-small-fraction` | Fraction with `abs(g) <= threshold`, by layer, parameter family and globally |
| `wgrad-small-fraction-individual` | The same, per named tensor/semantic component |
| `wgrad-relative-small-fraction` | Fraction with `abs(g) <= fraction * RMS(tensor)`, pooled by layer, family and globally |
| `wgrad-relative-small-fraction-individual` | The same, per named tensor/semantic component |

Each name accepts its own `:INTERVAL`; `--tensor-metrics-iterations` adds explicit steps.
For example:

```text
--tensor-metrics \
    wgrad-small-fraction:100 wgrad-small-fraction-individual:100 \
    wgrad-relative-small-fraction:100 wgrad-relative-small-fraction-individual:100 \
--tensor-metrics-wgrad-small-thresholds 0 1e-10 1e-8 1e-6 \
--tensor-metrics-wgrad-relative-small-thresholds 0.001 0.01 0.1
```

Absolute thresholds are required when the absolute metrics are enabled. Relative thresholds
have a default of `[0.01]`, meaning 1% of RMS. Lists must be nonempty, finite and nonnegative;
duplicates are removed and values sorted. Output suffixes are `le_0`, `le_1e-8`,
`rms_fraction_0.01`, etc. Threshold zero exactly matches zero-fraction.

The reference RMS is computed for the **complete logical named tensor**, including its
TP/GTP and optimizer storage shards, with no epsilon floor. Each tensor keeps its own
threshold when combining counts into larger scopes. Aggregation divides summed counts
by summed element counts, so larger tensors have proportionally more weight. Existing
semantic splits (for example Q/K/V) remain separate reference tensors; this does not add a
new per-expert split to a grouped expert parameter.

An entirely zero tensor is 100% small, including at relative threshold zero. Empty shards
and non-owner placeholders add no elements. Nonfinite entries never count as small.
A nonfinite tensor RMS makes its relative results, and every containing summary scope,
NaN. The absolute fractions still count finite entries against the full population.

The snapshot is observed before optimizer clipping, just like existing wgrad metrics.
Absolute cutoffs use the snapshot's units: if gradients are still loss-scaled, the cutoff
is in those scaled units. Relative fractions are invariant to uniform gradient scaling.
These metrics describe the accumulated gradient at the observation step, not persistent
optimizer momentum or individual microbatch gradients.

All computation and communication happens in the end-of-step metric commit. Absolute
counts and RMS moments share a first visit to optimizer storage; relative counts require
a second visit after the compact RMS reduction. No full gradients are gathered or retained
across parameters. Scratch arithmetic is chunked, and counts use int64 while squared sums
use FP64 to preserve tiny gradients. Summary and individual outputs share preparation
and per-tensor reductions. More thresholds add local scans and output scalars, but do not
add a distributed result branch per threshold. The existing pipeline-parallelism limitation
of the tensor-metric observer still applies.
