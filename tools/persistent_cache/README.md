# First-iteration kernel cache

The first training iterations after a (re)start can pay for Triton/Inductor
compilation and autotuning, CUDA PTX→SASS, CuTe DSL compilation, HybridEP NVCC
JIT and fused-kernel extensions. This tool collects reusable artifacts during
a bounded seed, consolidates them, and **bakes them into the container image**.
Production can then reuse the covered cases without fetching separate cache
archives from shared storage. Later input shapes or compiler specializations
can still require compilation or tuning; some graph preparation remains even
on cache hits. A read-only container may copy the cache to local scratch.

Start with the [Nemotron-4 seed-and-bake walkthrough](../../nemotron4/scripts/kernel_cache/README.md#one-command-seed-and-bake)
to build an image from a production launcher. Use [the consume instructions](#using-it-consume)
below to activate it in normal training, or the
[step-by-step edits for an existing launcher](../../nemotron4/scripts/kernel_cache/README.md#adapt-an-existing-launcher-step-by-step).

```
seed (write)  ->  consolidate (seal)  ->  bake into image  ->  consume (read)
 one bounded       automatic in the        one 1-node job       every job:
 run               seed helper             -> cached.sqsh       `source enable.sh`
                   -> consolidated/        (/opt/mcore_cache)   before python
```

Runtime cache misses and restoration failures generally fall back to native
compilation for the affected paths; other cache scopes may still be reused.
Invalid configuration passed to `enable.sh` can instead fail activation.
Check the cache summaries as well as job completion.

## Using it (consume)

### Cache baked into the image

An image produced by `bake_cache_image.sh` carries the compiled caches and
their `MANIFEST` at `/opt/mcore_cache`. Cache-loading code stays in the
matching nv-mistral checkout. Select that image in the production launcher
and source the repository's `enable.sh` inside the container, before `exec python`:

```bash
IMAGE_PATH="/path/to/cache-run/cached.sqsh"
RUN_CMD="set -f; cd ${REPO_DIR}; \
source tools/persistent_cache/enable.sh && \
exec python -u -m examples.mimo.pretrain_mimo ${OPTIONS}"
```

If the baked image is already selected, adding `source .../enable.sh` before
Python is the only required edit. `&&` prevents training if activation fails.
With no cache overrides inherited, `enable.sh` defaults to read mode and the
image cache. Optionally export `MCORE_CACHE_STATS=1` when submitting to include
CuTe reuse counters; it is not required to use the cache. If reusing a shell from
seeding or another cache experiment, first unset its `MCORE_CACHE_*` overrides
and `MCORE_CUTE_DSL_AOT_NAMESPACE` / `MCORE_CUTE_DSL_AOT_NS_FROM`.

Keep the normal production arguments, data, checkpoint/resume settings and
allocation. The image includes caches and metadata; the training checkout,
cache tools and resources come from the production mounts and must match the seeded
configuration. Normal training does not call the seed or bake helpers.
The checkout must include this branch's training-code changes, including GTP
allocator support for `TORCH_EXTENSIONS_DIR`; the image does not replace it.

Previously built images may also contain `/opt/mcore_cache/tools/persistent_cache`.
They remain usable without rebuilding: sourcing the repository script selects
its adjacent shim and helpers, so the bundled copy is unused. New bakes do not
copy the tool directory. Keep the repository revision compatible with the seed;
`tools_commit` in `MANIFEST` records the tools used during baking.
For a nondefault bake destination, set `MCORE_CACHE_IMAGE_DIR` to that path
before sourcing the repository script.

New bakes retain the seed's two optional Inductor settings in their `MANIFEST`.
When that image is selected, its defaults take effect before the cache hooks
are installed. A consumer needs no extra feature exports; an explicit `0`
still disables either feature. Generic images and older manifests do not
enable these features automatically. See [the feature settings below](#optional-startup-warmup-and-score-equivalent-autotuning).

No additional cache mount or `--container-env` is needed. With pyxis on
aws-cmh the container rootfs is writable and already unpacked on node-local
NVMe, so the compilers read and write the baked directories in place
(`[CACHE NODE SETUP] image cache in place ...`). On a read-only rootfs the
scope dirs are copied once per node to `/dev/shm` or `/tmp` instead.
New runtime artifacts stay in the running container or local scratch; normal
training does not update the immutable `.sqsh` image or reseal the namespace.

### From a shared-filesystem namespace

Until an image is baked, or for a namespace under test:

```bash
export MCORE_CACHE_DIR="/path/to/cache-run/cache" # holds consolidated/
export MCORE_CACHE_MODE=read MCORE_CACHE_STATS=1
unset MCORE_CACHE_LOCAL MCORE_CACHE_IMAGE_DIR MCORE_CACHE_PREFER_IMAGE
RUN_CMD="set -e; set -f; cd ${REPO_DIR}; \
source ${REPO_DIR}/tools/persistent_cache/enable.sh; \
exec python -u -m examples.mimo.pretrain_mimo ${OPTIONS}"
```

The first process per node extracts `consolidated/<scope>.tar.zst` to
node-local scratch with `tar -xf` (gzip-sealed by default, so no image needs
zstd; a zstd tarball also works where python-`zstandard` is installed).
`MCORE_CACHE_PREFER_IMAGE=1` makes an image cache win when both are present.
`enable.sh` may also be sourced on the host side of `srun`; then forward
`${MCORE_CACHE_ENV_FORWARD}` with `--container-env`.

### What to look for

One `[CACHE NODE SETUP] image cache in place|copied|extracted ...` line (rank 0;
every node with `MCORE_CACHE_VERBOSE=1`), an iteration-1 time well below the
cold number, and with `MCORE_CACHE_STATS=1` the CuTe exit summary
`[CUTE-AOT CACHE] summary ... reused=N recompiled=0`.
Inductor prints its compact exit summary whenever its counters are nonzero:
`INDUCTOR-AUTOTUNE RESTART SUMMARY ... native_hit=N fallback_hit=N late_hit=N`.
That normal status line also appears with `MCORE_CACHE_STATS=0`; the flag
does not suppress Inductor's counters or imply compiler debug tracing.
An ambiguous semantic lookup can still appear in `fallback_miss` and then be
satisfied by `late_hit`; the latter checks the exact runtime specialization.

Inductor runtime choices are recorded by default in write mode and replayed in
read mode. A seed should report `late_store` for kernels that still have multiple
compiled candidates. Runtime replay requires identical source, specialization,
hardware properties, compiler version and ordered compiled candidates; it
selects exactly one current launcher. Conflicting semantic records are never
chosen arbitrarily. Unsupported metadata or invalid runtime records leave
native autotuning active (`late_invalid`).

Ordinary runtime choices use `inductor/mcore_late_autotune_v2`; existing v1
dynamic-reduction records remain readable. Older namespaces lack v2 records:
reseed to retain these choices across future jobs and baked images. Runtime
replay skips benchmarking after candidate compilation; it does not eliminate
candidate loading or compilation.

### Optional startup warmup and score-equivalent autotuning

These options are **off unless explicitly requested by the producer or caller**.
To include them in a new seed and baked image, export them before invoking
`seed_and_bake.sh` or the individual `seed_from_run_script.sbatch` helper:

```bash
export MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU=1
export MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE=1
```

The seed helper records `inductor_prewarm_weighted_relu` and
`inductor_autotune_equivalent_cache` in `SEED_INFO`. The bake reads those exact
0/1 values and writes `runtime_prewarm_weighted_relu` and
`runtime_equivalent_autotune_cache` in the image's `MANIFEST`. It does not infer
the settings from the bake job's environment. Consumers only source
`tools/persistent_cache/enable.sh` from the training checkout; no repeated exports are
needed. Explicit caller values, including `0`, take precedence. Invalid,
incomplete or duplicate image-default entries suppress both baked defaults
without changing explicit caller settings.

These defaults apply only when the image cache is selected. Shared-filesystem
consumers and older images still need explicit exports to use the features.
For direct `install_image_cache.py` use, request the defaults with
`--prewarm-weighted-relu --equivalent-autotune-cache`; both default off.
For a seed launched without these helpers, sourcing `enable.sh` uses the
exported settings but does not write `SEED_INFO`. Before `bake_cache_image.sh`,
record each of the two `SEED_INFO` fields above exactly once as `0` or `1`,
or use the direct installer flags. Missing seed metadata leaves baked defaults off.

`MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU=1` exercises the ordinary compiled
weighted, tanh-clamped squared-ReLU functions during startup, before model
allocation. Disk caches do not populate a new process's Dynamo guards. For
Nano's local expert width of 4096, a tensor with 524288 rows crosses the
2^31-element indexing boundary; a later training batch can therefore need a
new graph even with a warm disk cache. The helper exercises both indexing
ranges and the forward/backward input modes without changing compiler guards,
numerics or training data. Its weights have no view base, matching checkpoint
recomputation, so warmed graphs also accept the detached weights seen there.
It restores Torch RNG state and frees its buffers.

The helper requires BF16, fused squared-ReLU MoE, a positive tanh clamp and a
validated local expert width. Only language ranks participate in MIMO runs.
Unsupported recipes or insufficient free GPU memory produce a `skipped` log;
execution errors propagate. The memory check budgets 32 bytes per element of
the largest warmup shape plus headroom. Explicitly enabling this for a recipe
whose Transformer Engine path fully fuses the activation may warm unused
functions. Look for `[INDUCTOR PREWARM] ... completed` on every participating
rank. Set the flag to `0` to disable it.

`MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE=1` additionally permits an exact
runtime-cache miss to reuse a pointwise winner when **only** `tiling_scores.x`
changed. It still requires identical source, specialization, compiler,
hardware and every ordered compiled candidate. Sidecar observations under
`inductor/mcore_late_autotune_v2/equivalent/` must be backed by intact exact
records and unanimously select one current launcher. Conflicts, corruption or
unsupported metadata keep native autotuning. The counter
`late_equivalent_hit` identifies these reuses; an ordinary `late_hit` alone
does not establish coverage. Identical candidates ensure a valid launch, but
do not prove the same winner is fastest at every shape: validate representative
shapes before opting in. Set the flag to `0` to keep strict-key reuse only.

### Kill switches

`MCORE_CUTE_DSL_AOT_CACHE=0`, `MCORE_CUTE_DSL_FILE_CACHE=0`,
`MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE=0`, `MCORE_TRITON_GROUP_REBASE=0`,
`TRITON_AUTOTUNE_PREHOOK_PATCH_DISABLE=1`; `MCORE_CACHE_IMAGE_COPY=1` forces the
copy path. For a native control, use a clean environment without the shim on
`PYTHONPATH` or inherited cache settings, and do not source `enable.sh`.

The repository's `enable.sh` defaults to `/opt/mcore_cache`; set
`MCORE_CACHE_IMAGE_DIR` for a custom image-cache location. Compatibility with
the script bundled in older images is retained, but current launchers use the
repository copy.

## What is cached (scopes)

| Scope | Env var pinned by the shim | Contents |
|---|---|---|
| `triton` | `TRITON_CACHE_DIR` | compiled cubins + autotune choices (+ rebased `__grp__` manifests); autotune results for `pre_hook` configs, which upstream refuses to cache, are persisted by the shim's Triton patch |
| `inductor` | `TORCHINDUCTOR_CACHE_DIR` | FX-graph + autograd caches, generated Triton kernels' `best_config`, exact runtime autotune choices |
| `cuda_ptx` | `CUDA_CACHE_PATH` | driver PTX→SASS |
| `cutedsl` | `CUTE_DSL_CACHE_DIR` | CuTe DSL IR + exported TVM-FFI AOT objects (+ `.aot_namespace` key) |
| `hybrid_ep` | `HYBRID_EP_CACHE_DIR` / `HYBRID_EP_JIT_DIR` | DeepEP HybridEP NVCC-JIT `.so` |
| `cudnn_fe` | `CUDNN_FRONTEND_CACHE_DIR` | cuDNN frontend heuristics |
| `megatron_fused_kernels` | `TORCH_EXTENSIONS_DIR` | `torch.utils.cpp_extension.load` build dir |

Keys include library-specific source, dtype, shape, compiler and hardware
metadata. Reuse depends on preserving the model configuration and per-rank
kernel shapes (micro-batch, sequence/image lengths, parallelism and weight
sharding). Reducing independent replicas can preserve these shapes, but a
node-count reduction alone does not establish cache compatibility or coverage.
Not covered: NCCL communicator setup, HybridEP buffer and
symmetric-memory registration, checkpoint load, CUDA-graph capture.

### Moving a cache to another cluster

Cluster names are not a compatibility guarantee. Keep the same base-image
contents, training code, GPU architecture and kernel-shaping configuration,
then verify a bounded warm run on the destination. Different image versions
need separate seeds; do not combine caches seeded from different images.

Preserve the training checkout's **absolute path inside the container** as
well as its revision: the current CuTe AOT function hash includes the Python
code object's source filename. Moving the checkout can therefore cause misses
even when file contents match. To keep an isolated seed checkout while matching
the production path, set `SEED_REPO_DIR` to that host checkout and
`SEED_CONTAINER_REPO_DIR` to the production `REPO_DIR`. The wrapper bind-mounts
the former at the latter for seed, verification and bake; it does not modify
the production checkout. The default visible path is `SEED_REPO_DIR` when no
alias is supplied. The seed's CuTe namespace is stored in
`cutedsl/.aot_namespace` and restored automatically; leave
`MCORE_CUTE_DSL_AOT_NAMESPACE` and `MCORE_CUTE_DSL_AOT_NS_FROM` unset when
activating a copied or baked cache. The bake finalizes the known C++ extensions
at the image path; copying that tree to another path can still cause a rebuild.
Byte-identical archives prove transfer integrity, not runtime hits.

### Optional NCCL topology in an image

The image baker can add a separately captured, cluster-specific NCCL topology
dump alongside the kernel caches. This uses stock `NCCL_TOPO_FILE`; it does
not replace NCCL or replay a communication graph. The seven ordinary kernel
cache scopes and seed jobs are unchanged.

Pass both optional settings to the bake job, with a **new** output image:

```bash
BASE_IMAGE=/path/to/recipe-base.sqsh \
MCORE_CACHE_DIR=/path/to/sealed-kernel-namespace \
OUTPUT_IMAGE=/path/to/new-cmh-cached.sqsh \
CACHE_NCCL_TOPO_FILE=/path/to/cmh/topo.xml \
CACHE_NCCL_TOPO_NAMESPACE=aws-cmh-gb300-nccl2.31.2 \
TOOLS_DIR="$PWD/tools/persistent_cache" \
  sbatch --account=YOUR_ACCOUNT --partition=batch tools/persistent_cache/bake_cache_image.sh
```

Run the bake on the intended cluster and node type. An older `nccl_topo`
archive may supply `topo.xml`; it is not necessary to reseed the unrelated
kernel scopes. A raw multi-node NVLink dump must not be set directly as
`NCCL_TOPO_FILE`. The repository helper extracts one host's static PCI/NVLink
layout and removes runtime identity. During baking it checks the XML against
the current host and records a template checksum and PCI-device fingerprint.
Only the prepared XML and metadata are added under `/opt/mcore_cache/nccl_topo`;
the helper remains in the training checkout.

The image manifest opts into topology reuse. Consumers still only select the
new image and source this checkout's `enable.sh`. Every worker checks the
template checksum, namespace and current PCI fingerprint before setting
`NCCL_TOPO_FILE`. Validation failure keeps the kernel cache active and falls
back to native NCCL discovery. Read-only images also copy this optional data
to node-local storage. The enabled topology path clears inherited topology
dump and communication-graph settings, so they cannot trigger extra discovery
or graph replay. Set `MCORE_NCCL_TOPO_CACHE=0` to disable the image feature;
disabled/older images leave externally managed NCCL variables alone.

The namespace is a provenance label, not automatic NCCL-version detection.
Use the intended base image/NCCL/network stack and validate it on the target
cluster. Static PCI checks do not establish every link's live state or
compatibility with an externally replaced NCCL library. Partial dumps are
allowed (NCCL discovers missing devices), but do not establish full topology
coverage or a performance gain. Prefer a fresh dump from a communicator
covering every local GPU and check bounded collective/training runs before
adopting the derived image. This feature does not persist bootstrap endpoints,
connections or communicator resources.

## Seed

A seed is one bounded run of the production recipe with
`MCORE_CACHE_MODE=write`; the default 20 iterations exercise startup and some
later specializations. Validate coverage against the intended inputs; real
data beyond the seed window can introduce new cases. Scale may shrink as long as
per-rank kernel shapes match production (same micro-batch, sequence/image
shapes, model, precision, TP/PP/CP/EP and weight-shard sizes). The Nemotron-4
helper preserves language-model options and the data path but reduces the
encoder batch, so its seed does not establish full encoder shape coverage.
For the Nemotron-4 launchers the flow is scripted, see
[`nemotron4/scripts/kernel_cache/README.md`](../../nemotron4/scripts/kernel_cache/README.md):

```bash
SEED_REPO_DIR="/path/to/pinned-checkout" \
SEED_CONTAINER_REPO_DIR="/path/where/production/mounts/its/checkout" \
bash nemotron4/scripts/kernel_cache/seed_and_bake.sh \
    --run-script /path/to/launcher.sh \
    --output-dir /path/to/new-cache-run --account YOUR_ACCOUNT --verify
```

Run the wrapper on the login node. It derives the node count and base image,
then submits seed, warm read, bake and warm image jobs with success dependencies.
Omit `--verify` for seed and bake only; add `--dry-run` to preview without
submitting. Outputs include `cache/`, `logs/`, `jobs.tsv` and `cached.sqsh`;
`--output-image PATH` names the image instead.

Slurm runs a spool copy of the submitted script, so helper paths cannot come
from the script's own location. The launcher and `bake_cache_image.sh` resolve
the submitted path through `scontrol` and fail with an explicit hint otherwise;
the wrapper passes `SEED_TOOLS` / `TOOLS_DIR` explicitly.

The launcher sources the production `run.sh` in a restricted-shell sandbox,
takes its environment and `OPTIONS` verbatim, rewrites only geometry, batch
size, checkpoint flags, run
length and log sinks, then packages one shard per node and consolidates them.
For individual stages, `SEED_DRY_RUN=1` previews the seed helper. For the
pipeline wrapper, use its `--dry-run` option; it explicitly disables the
helper's dry-run flag when submitting real stages.

Manual equivalent for any launcher: `MCORE_CACHE_MODE=write MCORE_CACHE_DIR=<ns>`
plus `source enable.sh` around a bounded run (caches land under
`<ns>/cache_write/_node/node-<rank>/`, or node-locally with
`MCORE_CACHE_LOCAL=/host_tmp/mcore_cache_${SLURM_JOB_ID}` followed by one
`package_node_cache.sh` per node), then `MCORE_CACHE_DIR=<ns> bash consolidate_cache.sh`.

A namespace is valid for exactly one image + code + geometry; encode them in
its name and re-seed into a new one whenever `run.sh` moves its image or code.

### Collecting a cache while profiling

External Nsight Systems profiling can run alongside cache collection. Inside
each training container, set `MCORE_CACHE_MODE=write` and a **new**
`MCORE_CACHE_DIR`, then source `enable.sh` before the existing training command.
Wrap that command with `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none`
and give each profiled rank a distinct output path. Keep the same image, code,
container-visible source paths and model configuration as the intended consumer.

New CuTe objects are exported as compilation completes. After successful
training, package once per node when using `MCORE_CACHE_LOCAL`, then consolidate
as described above. A read-mode run in a disposable container does not update
the original image; a `.nsys-rep` stores profiling data, not the kernel cache.

CuTe fingerprints now use reference-independent code serialization
(`mcore-cute-aot-v2`). Retaining Python code metadata, as a profiler or timing
wrapper can do, no longer changes the key. This changes the key format:
**create a fresh seed and bake a new image when upgrading**. Old CuTe objects
are not aliased to the new keys. Source filenames and semantic code changes
remain part of the fingerprint.

A CMH GPU check seeded three FlashAttention CuTe objects under Nsight with
the settings above. Fresh ordinary, retained-metadata, and
Nsight-plus-retained-metadata processes each reused all three with no
recompiles or export/load failures. Their synthetic outputs and gradients were
bitwise equal to the seed. This checks profiled export and reuse for those
functions, not every profiler setting or production shape; verify cache
counters for the intended recipe. A separate 32-node OCI-AGA check retained
Python metadata on all 64 language ranks and reported 832 CuTe reuses, zero
recompiles, and zero load/export failures.

## Bake the cache into the image

```bash
BASE_IMAGE="/path/to/base.sqsh" \
MCORE_CACHE_DIR="/path/to/cache-run/cache" \
OUTPUT_IMAGE="/path/to/cache-run/cached.sqsh" \
TOOLS_DIR="${PWD}/tools/persistent_cache" \
CACHE_BUILD_REPO="/path/to/matching-training-checkout" \
CACHE_CONTAINER_REPO_DIR="/path/where/production/mounts/its/checkout" \
sbatch --wait --account=YOUR_ACCOUNT --partition=batch \
    tools/persistent_cache/bake_cache_image.sh
```

Run from the tools checkout after consolidation has succeeded, using the same
base image as the seed and a new output filename. One GPU node (the image's
architecture), typically around 15 min for the tested image: pyxis starts the base image,
`install_image_cache.py` extracts the sealed scopes to `/opt/mcore_cache/`,
rebases the Triton group manifests, finalizes the C++ extensions, copies this directory beside them, writes
`/opt/mcore_cache/MANIFEST` (namespace, source image, per-scope sha256 and
file counts, seed provenance and the two runtime feature defaults), self-checks the shim, and `--container-save`
writes the new squashfs (saved as `.partial-<jobid>.sqsh` and linked into place
only on success). The same installer can run as a Dockerfile build step
against a mounted namespace.

`CACHE_BUILD_REPO` defaults to the checkout containing `TOOLS_DIR`;
`CACHE_CONTAINER_REPO_DIR` defaults to the same path. The seed-and-bake wrapper
passes `SEED_REPO_DIR` and `SEED_CONTAINER_REPO_DIR` automatically, preserving
the seed's visible source path during extension finalization. Use the matching
training checkout, including the GTP allocator cache support. The bake builds
`share_storage_ext` and `gtp_vmm_allocator` under the final
`/opt/mcore_cache/megatron_fused_kernels` path, then loads them in a second Python
process. It rejects the bake if that load changes build artifacts or loads a
library outside the image cache. `extensions-finalization.json` records the
source hashes, Torch/CUDA versions, loaded paths and checked artifacts.

The GTP allocator respects `TORCH_EXTENSIONS_DIR` and holds a process-shared
lock around source generation, compilation and loading. A warm image run needs
no separate allocator prebuild. This preserves normal Torch/Ninja invalidation;
an extension filename alone does not establish compatibility with another
image or toolchain. Final-path reuse applies to writable, in-place image
consumption; relocation during a read-only-image copy may still rebuild.

For direct installer or Dockerfile use, pass
`--training-repo /path/to/matching-checkout` to finalize the extensions. Without
that argument the installer only extracts the existing artifacts. The finalizer
imports Torch and builds extensions; it does not initialize the GTP allocator
or distributed process groups.

## How it works

1. `enable.sh` exports the mode/feature env and puts `_triton_patch_shim/`
   first on `PYTHONPATH`.
2. CPython's `site` runs `sitecustomize.py` in every process before any user
   import. It first chains to the container's own `sitecustomize.py` (some
   images add site directories there), then calls
   `mcore_cache_node_setup.setup()`:
   * source: `MCORE_CACHE_DIR` → shared namespace (unless
     `MCORE_CACHE_PREFER_IMAGE=1`); else image if
     `${MCORE_CACHE_IMAGE_DIR:-/opt/mcore_cache}/MANIFEST` exists; else inert;
   * once per node (fcntl lock + `.mcore_ready` sentinel stamped with the job
     id): image in place → only rebase Triton groups (never wipes); image copy
     or namespace → wipe stale scratch from another job, `cp -a` or `tar -xf`
     the scopes, rebase;
   * every process: pin the scope env vars, set `HYBRID_EP_LOAD_CACHED_KERNELS`
     from a `.so` scan, recover the CuTe AOT namespace from
     `cutedsl/.aot_namespace`, and apply the selected image's allowlisted
     runtime defaults without replacing explicit caller settings.
3. Lazy patches arm as `MetaPathFinder`s that fire on the first import of their
   target (`deep_ep`, `cutlass.base_dsl.dsl`, `triton.runtime.autotuner`,
   Inductor's autotune cache), so helper processes never import torch: HybridEP
   `load_cached_kernels` with the persisted `.so` linked in, CuTe DSL file cache
   + AOT object reuse, Inductor autotune recovery after an FX-graph reload,
   Triton disk caching of `pre_hook` autotune configs (Mamba SSD kernels).

## Layout

| File | Role |
|---|---|
| `enable.sh` | the runtime interface (source it) |
| `_triton_patch_shim/sitecustomize.py`, `mcore_cache_node_setup.py` | per-process arming, per-node cache setup |
| `cute_aot_cache.py`, `inductor_autotune_restart_cache.py`, `triton_autotune_disk_cache.py`, `cache_verbosity.py` | lazy patches, rank-0 logging |
| `rebase_triton_groups.sh` | rewrite absolute paths in Triton `__grp__*.json` |
| `package_node_cache.sh`, `consolidate_cache.sh`, `lib.sh` | seed side: publish per-node shards, seal them |
| `install_image_cache.py`, `bake_cache_image.sh`, `finalize_image_extensions.py` | install and finalize artifacts before saving the image |

The Nemotron-4 seed-and-bake wrapper (`seed_and_bake.sh`) and stage helper
(`seed_from_run_script.sbatch`) live in `nemotron4/scripts/kernel_cache/`.

## Tests

`tests/unit_tests/tools/first_iter/`: no-GPU tests for the shim (image and
namespace sources, once-per-node races, write mode, container-sitecustomize
ordering, in-place locking), the installer, consolidation (including a truncated
shard), packaging, logging, the AOT key, and the seed launcher (dry run against
a synthetic production `run.sh`, spool-copy path resolution). The pipeline
tests replace `sbatch` and check dependencies, image selection, frozen recipes,
environment propagation, dry-run isolation and submission failures.

These tests run on CPU, but `tests/unit_tests/__init__.py` imports Torch, and
`--confcutdir` does not avoid that import even with pytest's importlib mode.
Copy the test modules outside the package and point `MCORE_REPO_ROOT` at the
checkout:

```bash
suite=$(mktemp -d)
cp tests/unit_tests/tools/first_iter/test_*.py "$suite/"
MCORE_REPO_ROOT="$PWD" python -m pytest --confcutdir="$suite" "$suite"
```

The launcher tests need Linux Bash 4 or newer. The archive checks skip unless
the Python `zstandard` package and the `zstd` CLI are installed. These tests
stub GPU dependencies and do not validate distributed training or numerical
behavior on hardware.
