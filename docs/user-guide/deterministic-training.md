<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Deterministic Training

Deterministic training guarantees that two runs with identical inputs produce identical outputs at every step. Useful for debugging regressions and for reproducibility studies.

Pass `--deterministic-mode` to any Megatron training entry point (e.g. `pretrain_gpt.py`):

```bash
python pretrain_gpt.py \
  --deterministic-mode \
  <other args ...>
```

When enabled, Megatron applies the env vars and config overrides below via `megatron.training.determinism.apply_determinism_to_args` (called from `validate_args`).

## Environment variables

Each variable may be set by the launcher or left unset. If set, the value must be one that has been validated as deterministic — anything else fails hard with an assertion. If unset, `apply_determinism_env` fills the canonical default (except `MAMBA_DETERMINISTIC` and `CAUSAL_CONV1D_DETERMINISTIC`, which their kernels auto-detect from `torch.are_deterministic_algorithms_enabled()`). Must be set before the first cuBLAS / Transformer Engine call — `apply_determinism_to_args` runs early in `validate_args` to guarantee this.

| Variable | Accepted values (or unset) | Default filled if unset | Reason |
|---|---|---|---|
| `NCCL_ALGO` | empty, or a subset of `{Ring, CollnetDirect, CollnetChain, ^NVLS}` | `""` (empty) | Empty keeps NCCL's symmetric (NVLS/LDMC) kernels available. **Not** unset — unset makes PyTorch ban NVLS. Any non-empty value, `Ring` included, costs the symmetric kernels — see [below](#why-nccl_algo-defaults-to-empty) |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | `0` | Forces Transformer Engine to use deterministic algorithms |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` or `:16:8` | `:4096:8` | Deterministic cuBLAS workspace (both sizes are reproducible per NVIDIA docs; `:4096:8` is faster, `:16:8` uses less memory) |
| `TRITON_CACHE_AUTOTUNING` | `0` or `1` | *(none — opt-in)* | Persists each Triton autotune winner so every rank reuses one choice instead of re-timing it. Left unset, deterministic mode instead pins the cheapest config, which needs no cache — see [Triton autotuning](#triton-autotuning) |
| `TRITON_CACHE_DIR` | any shared-filesystem path | *(none — required only with `TRITON_CACHE_AUTOTUNING=1`)* | No safe default exists: unset, Triton uses a node-local directory and each node autotunes on its own. Required rather than filled in |
| `TRITON_PRINT_AUTOTUNING` | `1` | *(none — recommended, not set)* | Logs the config each rank selected. Changes no numerics, so it is recommended rather than forced; when `TRITON_CACHE_AUTOTUNING=1` and this is unset, a startup line reminds you. See [Verifying kernel-config agreement](#verifying-kernel-config-agreement) |
| `MAMBA_DETERMINISTIC` | any string starting with `'1'` | *(none — SSM auto-detects)* | Mamba SSM auto-follows `torch.are_deterministic_algorithms_enabled()` when unset; only an explicit non-deterministic override is rejected |
| `CAUSAL_CONV1D_DETERMINISTIC` | any string starting with `'1'` | *(none — the kernel auto-detects)* | causal_conv1d ≥ 1.6.0 auto-follows `torch.are_deterministic_algorithms_enabled()` when unset, reducing the conv weight/bias gradients through a workspace instead of `atomicAdd`; the Mamba and GDP mixers reject a deterministic run without it |

### Why `NCCL_ALGO` defaults to empty

`--deterministic-mode` defaults `NCCL_ALGO` to the empty string, which keeps NCCL's
symmetric (NVLS/LDMC) kernels available — roughly twice as fast on the reduce-scatter,
on every collective in the job.

**Empty, not unset.** Unset is the worst of both: PyTorch bans NVLS *and* you do not get
Ring's guarantee. Empty is the only value that works, for two independent reasons.

**It must be set.** PyTorch's `ProcessGroupNCCL` constructor exports `NCCL_ALGO=^NVLS`
when torch determinism is enabled *and* the variable is unset
([ProcessGroupNCCL.cpp](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1010-L1025)),
because NVLS "can lead to non-deterministic reduction" — upstream's own words, carrying a
`TODO` to drop the ban once NVLS supports deterministic mode. It names no hardware and no
specific defect, and it applies the ban everywhere. The export is guarded by
`if (!get_env("NCCL_ALGO").has_value())`, and `c10/util/env.cpp` returns an engaged
optional for an empty string — so an empty value keeps that line unreached. It fires in
the *constructor*, so it applies to every `new_group`, not just `init_process_group`.

**It must be empty.** Any non-empty value makes NCCL report `forced = 1`, and a forced
decision resolves inside the legacy algorithm candidates (tuning ids 0..20), which are
disjoint from the symmetric kernel ids (21..38, e.g. id 36 = `ReduceScatter_LDMC`). The
exclusion syntax is not special-cased: `^Tree` loses the symmetric kernels just as `Ring`
does, even though it only means "do not use Tree".

Measured at 4 GPUs / 1 node, kernel read from `torch.profiler`:

| `NCCL_ALGO` | torch determinism | reduce-scatter kernel | `forced` |
|---|---|---|---|
| *(unset)* | off | `ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16` | 0 |
| `Ring` | off | `ncclDevKernel_ReduceScatter_bf16_RING` | 1 |
| `^Tree` | off | `ncclDevKernel_ReduceScatter_bf16_RING` | 1 |
| *(unset)* | **on** | `ncclDevKernel_ReduceScatter_bf16_RING` — PyTorch wrote `^NVLS` | 1 |
| `""` | **on** | `ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16` | 0 |

**Hopper is non-deterministic. Blackwell is not.** Upstream bans NVLS on both. That
asymmetry is the entire basis for overriding the ban, so be explicit about it:

| Architecture | Reduce path under `NCCL_ALGO=""` | |
|---|---|---|
| **Hopper** (H100 / H800) | NVLS | **Non-deterministic — keep the ban.** [NVIDIA/nccl#2360](https://github.com/NVIDIA/nccl/issues/2360): a BF16 NVLS AllReduce on H800 returns different results depending on how many unrelated communicators were initialized first, and clears under `NCCL_NVLS_ENABLE=0`. Do not set `NCCL_ALGO=""` here |
| **Blackwell** (GB200 NVL72) | symmetric LDMC kernels | **Deterministic — measured.** The verification below is bit-exact. This is the only configuration `NCCL_ALGO=""` is supported on |

So the override is a measurement on one architecture, not a claim that NVLS is
deterministic in general. **Re-verify after an NCCL upgrade, and on any architecture not
listed above.**

**What this gives up.** `Tree` becomes selectable, and bit-exactness rests on measurement
rather than on construction. Verified at 64 nodes / 256 ranks with `NCCL_ALGO=""` —
375/375 float32 points identical on A/A (steps 1..25) and 375/375 on RESUME (steps
26..50). That scope is logged scalars only; the tolerance-free `.distcp` sha256
comparison was not run. Determinism cost over the same runs fell from roughly 4-5% to
about 1%.

`Ring` remains accepted as an explicit launcher override, but is not recommended: it costs
the symmetric kernels (~2x on the reduce-scatter) **and** deadlocks against the Spectrum-X
net plugin that the training images bake in, which is why `apply_determinism_env` has to
fill in `NCCL_IB_SKIP_RECV_COMPLETION=0` whenever that plugin is active.

**Safeguard.** Because the mechanism relies on incidental PyTorch behaviour, Megatron
re-reads `NCCL_ALGO` through libc after the first process group is created and raises if
the ban applied anyway (`assert_nvls_not_banned`). A PyTorch upgrade that changes this
fails at startup instead of silently costing ~2x on every collective.

If you override `NCCL_ALGO`, the value must be a subset of `{Ring, CollnetDirect, CollnetChain, ^NVLS}`. `Tree` itself is intentionally excluded: its intra-node chain reduction order is not user-controllable, and the inter-node tree topology can vary across runs without a pinned topology file, so it cannot be vouched for as bit-exact across stacks. `^NVLS` is accepted (banning NVLS is a legitimate user choice on hardware that exposes it); the user is responsible for ensuring whatever NCCL falls back to is deterministic on their environment.

## Config requirements

Checked against the parsed `args` Namespace in `apply_determinism_to_args`. Incompatible options are rejected with an explicit error rather than silently flipped off — you must disable them yourself so the run matches the config you asked for:

| Flag | Behavior under `--deterministic-mode` |
|---|---|
| `--cross-entropy-loss-fusion` | Must be off — asserted (fused CE is non-deterministic); drop the flag yourself |
| `--tp-comm-overlap` | Must be off — asserted (the overlap path is not bit-exact); drop the flag yourself |
| `moe_router_aux_loss_fusion` | Must be off — asserted (the fused aux loss reaches the backward); follows `moe_router_fusion` when unset |
| `torch.use_deterministic_algorithms` | Set to `True` |
| `torch.utils.deterministic.fill_uninitialized_memory` | Set to `False` — see below |

Flash attention is permitted: Transformer Engine's flash-attention backend is deterministic when `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (see the [Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html)).

## Uninitialized-memory fill

Determinism costs an *independent* output buffer: a reduction that would otherwise accumulate into shared memory with unordered atomics writes into its own buffer instead, fixing the summation order run to run. That is what makes training reproducible, and `--deterministic-mode` keeps it.

`torch.use_deterministic_algorithms(True)` also switches on a separate knob, `torch.utils.deterministic.fill_uninitialized_memory`, which fills every uninitialized allocation — `torch.empty`, `empty_like`, `empty_strided`, `Tensor.resize_` — with NaN or MAX_INT, so a kernel *reading* memory it never wrote reads the same bytes every run. Reproducibility does not need that, and it is not free: one extra fill kernel per empty allocation, serialized between real work, which suppresses the overlap between compute and communication and so costs far more wall time than GPU time. Clearing it is worth roughly **15% TFLOP/s** on large configs, and more the more `torch.empty` calls a step makes. `apply_determinism_to_args` clears it immediately after enabling deterministic algorithms.

Padding matters only if a computation reads it. **Benign — nothing reads it:** computed values are bit-identical and only saved bytes differ. Checkpoints are the example — some saved tensors carry trailing pad slots that no kernel writes, so two runs of the same configuration write files differing in those bytes while every trained value matches. **Harmful — a computation consumes it:** a reduction over a padded tail, a GEMM with a rounded-up K, an unmasked attention region. Results then differ run to run, and the fill does not make them correct, only repeatably wrong — every run reads the same NaN instead of different garbage. Fix the read; re-enabling the fill hides it.

Set the fill back to `True` while hunting such a read — turning garbage into a loud NaN is the one thing it is good for:

```python
import torch.utils.deterministic
torch.utils.deterministic.fill_uninitialized_memory = True
```

## The SPCX net plugin is disabled under `--deterministic-mode`

`apply_determinism_env()` sets `NCCL_NET_PLUGIN=none` by default. **The plugin breaks
bit-exactness** once `NCCL_ALGO` is empty, which is the default above: with the algorithm
unconstrained, NCCL routes reduce-scatter through its symmetric kernels, and the plugin's
IB path makes that non-reproducible. It is also unreliable on its own account, segfaulting
in `ncclIbMultiSend`, and it has been seen to hang on GB300.

**The default is not a performance tradeoff.** GB200 training does not run the plugin, so
`NCCL_NET_PLUGIN=none` matches what these runs already do — it makes that explicit instead
of leaving it to whatever the container image happens to set. Every bit-exactness result in
this document was produced with the plugin off.

It is a `setdefault`, so `NCCL_NET_PLUGIN=spcx` in the launcher still wins. Note that
exporting it in the sbatch environment is not always enough: pyxis does not reliably
forward arbitrary host variables into the container, so a launcher that needs a specific
value should export it inside the command `srun` runs.

### It also deadlocks under `Ring`

`Ring` deadlocks against the plugin on a multi-MNNVL-domain world: every ring connects,
then the job goes silent. There is no error, no traceback and no NCCL warning, so the
allocation is held until `--distributed-timeout-minutes` fires. **This is not an exotic
configuration** — the training images bake `NCCL_NET_PLUGIN=spcx` into the container
environment.

`--deterministic-mode` no longer selects `Ring` (see
[Why `NCCL_ALGO` defaults to empty](#why-nccl_algo-defaults-to-empty)), so this can only
happen if a launcher sets `NCCL_ALGO=Ring` explicitly — one more reason not to. Megatron
no longer fills in a workaround for it: a launcher that chooses `Ring` on spcx must also
set `NCCL_IB_SKIP_RECV_COMPLETION=0` or `NCCL_PROTO=simple`, both of which were measured
to clear the deadlock.

## Triton autotuning

Triton picks a kernel config by timing its candidates, so the winner depends on the machine at that instant and ranks can disagree. Deterministic mode offers two ways to remove that variance:

| Strategy | How to select it | Determinism rests on |
|---|---|---|
| **Pinned config** (default) | leave `TRITON_CACHE_AUTOTUNING` unset | Nothing external. `autotune_configs` picks the cheapest config by a pure function of the candidate list, so every rank computes the same answer without timing anything. Slower, since the pinned config is not necessarily the fastest one. |
| **Cached autotuning** | `TRITON_CACHE_AUTOTUNING=1` **and** `TRITON_CACHE_DIR=<shared path>` | Every rank reading one warm cache. Autotuning still runs and still picks fast configs, but a rank that misses the cache re-times the selection on its own and can pick differently. |

Cached autotuning is opt-in because its determinism is conditional: the pinned default holds by construction, the cached path holds only while the shared cache does. Setting `TRITON_CACHE_AUTOTUNING=1` without `TRITON_CACHE_DIR` is rejected — unset, Triton falls back to a node-local directory, which is exactly the case the cache is meant to prevent.

## Verifying kernel-config agreement

Applies to cached autotuning; the pinned default has nothing to compare. `TRITON_PRINT_AUTOTUNING=1` makes each rank log the config it selects per kernel; group those lines by kernel and key across the per-rank logs, and every group should hold exactly one distinct config.

Note the limit: a rank only logs when it *tunes*, so a run where some ranks hit the cache and others miss cannot be compared this way — the hitting ranks log nothing.

## Bitwise resume

Bit-exactness **across a checkpoint resume** is a stronger property than run-to-run bit-exactness, and a run-to-run A/A check cannot detect a violation of it. Two fresh processes share the same history, so any bug whose trigger is *process age* affects both identically and they agree. The asymmetry only appears between a process that has trained N steps and one that has just loaded a checkpoint.

The class to watch for is **process-local state that is not checkpointed and not re-initialised per iteration**. A concrete instance, fixed in this series: `TEGroupedLinear.is_first_microbatch` is `True` at construction and `False` after the first forward, and is only re-armed per iteration for quantized configs. Transformer Engine derives the weight-gradient accumulation mode from it — `accumulate = fuse_wgrad_accumulation and not is_first_microbatch` — so a stale `True` makes the wgrad GEMM **overwrite** `main_grad` instead of accumulating into it. Harmless for a module invoked once per microbatch (it overwrites a freshly zeroed buffer); destructive for one invoked twice, such as an MTP block sharing a layer across depths under `--mtp-use-repeated-layer`, where the backward runs in reverse and the overwrite discards the other depth's contribution.

To verify bitwise resume, compare three legs of the same configuration:

1. **continuous** — run to step *N*, saving at step *M*;
2. **save** — run to step *M* only, saving there;
3. **load** — resume from step *M* and run to *N*.

Leg 1 vs leg 2 over steps 1..*M* is the A/A baseline; if it diverges, nothing downstream is interpretable. Leg 1 vs leg 3 over steps *M+1*..*N* is the resume verdict. Compare the `iter_N` checkpoints byte-wise as well as the logged scalars: rank 0's shard differs by construction (it carries the pickled args, which differ in `--exit-interval`/`--save`/`--load`), so ranks 1..*N-1* are the verdict. Logged scalars are float32 and are reductions over the whole model, so they can miss a large relative change confined to a small-magnitude subspace — the checkpoint comparison has no such ceiling.

### Running the check

[`tools/determinism/verify_bitwise_resume.sh`](../../tools/determinism/verify_bitwise_resume.sh) is a worked example of everything below — edit the two marked variables (the args under test and how your site launches a distributed job) and run it. It lives in `tools/` rather than `tests/` because it is operator-run, not CI-wired: there is no `model_config.yaml`, no golden values, and nothing for `run_ci_test.sh` to drive. The rest of this section describes the procedure itself, so the check can be reconstructed or adapted without it.

The script compares **checkpoint shards byte for byte** rather than logged scalars. That is the direct question — bitwise resume is bitwise — and it needs nothing but `sha256sum`. Comparing tensorboard scalars instead is a float32 proxy for the same thing, with a real ceiling: scalars are reductions over the whole model, so they can miss a large relative change confined to a small-magnitude subspace. Use them only to locate *which step* first diverged, never as the verdict.

Read the other way, the checkpoint hash is itself a proxy for the thing you actually care about — that training proceeds identically. It is a strict one, because the shard carries the parameters, the fp32 masters, the optimizer state, `main_grad` and the per-rank RNG, so a difference anywhere in the forward or backward reaches it by the next optimizer step. But it is pass/fail at a single iteration: it will not tell you *when* the runs parted or by how much.

So on a real run, do both. Compare the per-step scalars end to end — `lm loss`, `grad norm`, and `num-zeros-in-grad` — alongside the checkpoint hashes. The scalars localize the first diverging step and show the magnitude; the hashes decide whether anything diverged at all. Neither substitutes for the other: a run can log identical `lm loss` at print precision while its weights differ in the last bits, and a checkpoint that differs tells you nothing about where to start looking. This script ships only the hash comparison because it is the half with no tolerance to argue about and no log parsing to maintain.

The three legs must be three separate processes in one allocation. Splitting them across jobs lets the legs land on different nodes, and "the two runs used different hardware" is not a finding about checkpoint resume. Take the configuration under test and vary only `--exit-interval`, `--save`, `--load`, and `--tensorboard-dir`:

| leg | added arguments |
|---|---|
| `cont` | `--exit-interval N --save ckpt_cont --tensorboard-dir tb/cont` |
| `save` | `--exit-interval M --save ckpt_save --tensorboard-dir tb/save` |
| `load` | `--exit-interval N --save ckpt_load --load ckpt_save --tensorboard-dir tb/load` |

with `--save-interval M --save-retain-interval M` on all three. Use at least three post-resume steps (`N - M >= 3`): the divergence shows up in grad-norm one step *after* the resume, not at the resume step, so a one- or two-step window can come back clean on genuinely broken code.

A comparison that measured nothing must never be read as a pass, so keep three outcomes distinct — identical, diverged, and *not evaluable* (a missing tag, no overlapping steps, or a leg that wrote no events). Treat only the first as success, and let a confirmed divergence outrank a later check that had nothing to read.

Four traps are worth knowing before writing the driver:

- **Read scalars from the tensorboard event files at full float32, never from the printed log.** Grad norm prints to four significant figures, which reports `1.724` vs `1.724` for two values differing at 2.8e-5 — a false pass.
- **Omit `--save` when it is empty rather than passing `--save ""`.** argparse consumes the next flag as the value and dies with "expected one argument".
- **`grep -c` exits 1 on zero matches**, so `grep -c ... || echo 0` emits a second line and a caller reading it as a number sees a spurious failure. Use `|| true`.
- **Quote or `set -f` around any unquoted expansion of the argument list.** `--hybrid-layer-pattern`'s value contains `*`, and an unquoted expansion also runs pathname expansion — if a file in the working directory matches, the pattern is silently replaced by that filename. An unmatched glob survives intact, so this passes by luck almost everywhere.

Exercise the check on code known to be broken before trusting a clean result from it. Reverting the `is_first_microbatch` fix above is a convenient positive control: it diverges in grad-norm at the first post-resume step while the loss still matches.

### Configurations worth covering

Bitwise resume is not one property; different configurations exercise different mechanisms, and a clean result in one says nothing about another:

- **bf16 + Muon** — the baseline. The `is_first_microbatch` defect above reproduces here and is the configuration it was root-caused and fixed in.
- **Cached Triton autotune** — see [Triton autotune selection](#triton-autotune-selection). The fallback pin and the cached strategy are different code paths; the cached one is only deterministic if every rank reads an already-warm cache, so it needs a prewarm pass, `MEGATRON_AUTOTUNE_PREWARM=1` on that pass alone, and a check afterwards that no later leg re-benchmarked anything. A cached run that silently autotuned from cold otherwise looks identical to one that worked.
- **MXFP8 + fp8 param gather** — the only configuration in which the fp8/fp4 param-gather re-derivation path executes at all; the bf16 cells leave it dead. Requires Adam: MXFP8 with Muon cannot build, because Muon's layer-wise split makes both halves uint8 under `--fp8-param-gather` and `partition_buckets` asserts `fp8_buffer is None`.
- **CuTe DSL** (`--gdp-cutedsl-kernel`, plus `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` for the fused grouped MLP) — replaces the kernels the other configurations autotune, so the autotune question does not arise on this path.

## Verifying determinism

The bit-exact correctness suite lives at `tests/unit_tests/determinism/correctness/`. It parametrizes over model presets (GPT-like, Llama-like, Hybrid/Mamba) × parallelism cells (TP, PP, VPP, EP, FSDP, and composites) and asserts that two runs of the same configuration produce bit-identical outputs and gradients. FP8 / FP4 recipes (`tensorwise`, `delayed`, `mxfp8`, `nvfp4`) are covered by `tests/unit_tests/determinism/correctness/test_fp8_determinism.py`; the Blackwell-only recipes are capability-skipped on Hopper.

The cost of `--deterministic-mode` is measured outside pytest by an nsys-driven per-NVTX-range breakdown: `tests/performance_tests/shell_test_utils/determinism/run_nsys_breakdown.sh` wraps any training entry point (e.g. `pretrain_gpt.py --profile`) under nsys for a det-vs-nondet comparison, and `tests/performance_tests/shell_test_utils/determinism/print_nsys_leaderboard.py` joins the two CSVs into a side-by-side table. The CI invocation lives at `tests/test_utils/recipes/h100/determinism-perf.yaml`.
