<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Deterministic Training for Nemotron-4

The setup only. For the mechanism behind any of it, see [Deterministic Training](deterministic-training.md).

## Example

```bash
# ===== determinism setup =====
DET_ARGS=""
if [ "${DETERMINISTIC:-0}" = 1 ]; then
    DET_ARGS="--deterministic-mode --no-moe-router-aux-loss-fusion"

    # also drop --cross-entropy-loss-fusion and --cross-entropy-fusion-impl
    # keep --moe-router-fusion

    # Already applied by apply_determinism_env(); exported here so the whole
    # determinism setup reads in one place.
    export NCCL_ALGO=""                     # empty, NOT unset, and NOT Ring:
                                            # keeps NCCL's symmetric NVLS/LDMC kernels
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export NCCL_NET_PLUGIN=none             # spcx off; see the table below

    # SSM kernels. Unset already resolves to torch's deterministic flag, so
    # these are explicit rather than required.
    export MAMBA_DETERMINISTIC=1
    export CAUSAL_CONV1D_DETERMINISTIC=1
fi

srun ... python -u pretrain_hybrid.py ${DET_ARGS} <rest of the NT4 config>
```

If a job sits at 0 iterations just after `finished creating GPT datasets`, check whether the launcher set `NCCL_ALGO=Ring`: it deadlocks against the spcx plugin — see [It also deadlocks under `Ring`](deterministic-training.md#it-also-deadlocks-under-ring).

## Changes to an NT4 config

| Change | Reason |
|---|---|
| **add** `--deterministic-mode` | entry point for everything below |
| **add** `--no-moe-router-aux-loss-fusion` | the fused aux loss reaches the backward through `MoEAuxLossAutoScaler`; asserted off |
| **remove** `--cross-entropy-loss-fusion` and `--cross-entropy-fusion-impl` | fused CE is non-deterministic; asserted off |
| **keep** `--moe-router-fusion` | fused TopK routing only selects experts, so determinism never required it off. It is kept by setting the aux-loss flag explicitly above, which otherwise inherits from it |
| **env** `NCCL_ALGO=""`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | already applied by `apply_determinism_env()` — nothing breaks if you leave them out. `NCCL_ALGO` must be **empty**, not unset: unset makes PyTorch ban NVLS. See [deterministic-training.md](deterministic-training.md#why-nccl_algo-defaults-to-empty) |
| **env** `NCCL_NET_PLUGIN=none` | also applied by `apply_determinism_env()`. The Spectrum-X plugin breaks bit-exactness once `NCCL_ALGO` is empty. See [The SPCX net plugin is disabled under `--deterministic-mode`](deterministic-training.md#the-spcx-net-plugin-is-disabled-under---deterministic-mode) |
| **env** `MAMBA_DETERMINISTIC=1`, `CAUSAL_CONV1D_DETERMINISTIC=1` | the SSM kernels read these, and unset falls through to `torch.are_deterministic_algorithms_enabled()`, which `--deterministic-mode` has already set. Only an explicit `0` is rejected |
| **replace** `--attention-backend flash --flash-attention-version 4` **with** `--attention-backend fused` | FA4's SM100 backward has no deterministic path at `head_dim=256`, which `--kv-channels 256` selects: `sm100_hd256_2cta_fmha_backward.py:143` asserts *"SM100 backward with head_dim=256 does not support deterministic mode"*. cuDNN fused is deterministic under `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` |
| **remove** `--no-load-rng`, if the config sets it | a restarted run would rebuild RNG state from `--seed` instead of restoring the checkpoint's, so dropout, data order and every stochastic kernel diverge from the trajectory the run would have taken had it not stopped. Nothing errors — the run simply continues as a different run. Any config that saves and resumes needs this dropped, not just the resume check |

`--tp-comm-overlap` is also asserted off, but NT4 configs never set it.

Nothing to do about `--seed`: determinism reproduces a *given* run, so the compared runs only have to share a seed, which an unchanged config already does.

## Checking it

Run the configuration twice and compare the tensorboard event files at full float32 — not the printed log, which rounds to four significant figures.

For resume, see [Bitwise resume](deterministic-training.md#bitwise-resume), with one caveat: that section's verdict is a byte-wise comparison of checkpoint shards, and clearing the uninitialized-memory fill makes shards differ in inert pad bytes. Until the pad slots are written deterministically at save time, the shard hash reports a mismatch it cannot explain, so the scalar comparison is what is left — and it is weaker, since scalars are whole-model reductions and can miss a change confined to a small-magnitude subspace.
