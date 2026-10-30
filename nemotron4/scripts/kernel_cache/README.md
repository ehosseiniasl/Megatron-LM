# Nemotron-4 first-iteration kernel cache

`seed_and_bake.sh` builds a container image that carries a first-iteration kernel
cache for one Nemotron-4 MIMO launcher, such as
[`nemotron4/scripts/derisking/nano_21t/phase1.sh`](../derisking/nano_21t/phase1.sh).
`seed_from_run_script.sbatch` runs the individual stages. Both read the launcher,
keep its model, kernel and data options, and rewrite only what a bounded seed
must change. See the [cache reference](../../../tools/persistent_cache/README.md)
for runtime behavior and cache scopes.

To use a baked image in training, follow the
[step-by-step edits](#adapt-an-existing-launcher-step-by-step).

## One command: seed and bake

Run `seed_and_bake.sh` with Bash on the login node, from a checkout visible to
the compute nodes. It submits `seed -> bake`; `--verify` adds warm checks for
`seed -> warm read -> bake -> warm image`. Each job runs only if the previous
one succeeded. The command returns after submission.

Prerequisites: Linux Bash 4+, Slurm with Pyxis, a training checkout that
includes these tools, and read access to everything the launcher uses: its
image, configuration files, vision checkpoint and training data. The seed reads
the launcher's `--pretokenized-config`, so that file must already exist; it is
written when the launcher itself first runs. The image must already contain the
launcher's training dependencies.

```bash
LAUNCHER=/path/to/launcher.sh          # e.g., a copy of nemotron4/scripts/derisking/nano_21t/phase1.sh
OUTPUT_DIR=/path/to/new-cache-run      # must not exist yet
OUTPUT_IMAGE=/path/to/images/base+cache.sqsh
export SEED_REPO_DIR="$PWD"            # checkout to seed from
export SEED_CONTAINER_REPO_DIR=/path/where/production/mounts/its/checkout

# Preview the derived recipe and submissions first.
bash nemotron4/scripts/kernel_cache/seed_and_bake.sh \
  --run-script "$LAUNCHER" --output-dir "$OUTPUT_DIR" --output-image "$OUTPUT_IMAGE" \
  --account YOUR_ACCOUNT --verify --dry-run

# Submit the chain.
bash nemotron4/scripts/kernel_cache/seed_and_bake.sh \
  --run-script "$LAUNCHER" --output-dir "$OUTPUT_DIR" --output-image "$OUTPUT_IMAGE" \
  --account YOUR_ACCOUNT --verify
```

`--partition`, `--qos` and `--reservation` select the allocation; the
launcher's own `#SBATCH` directives are not used. The base image is the
launcher's `IMAGE_PATH` unless `SEED_IMAGE` overrides it, and the bake uses the
same base image.

`SEED_REPO_DIR` is the checkout on the host. `SEED_CONTAINER_REPO_DIR` is where
that checkout appears inside every seed and bake container; set it to the path
production jobs will mount (the launcher's `REPO_DIR`). Both the code revision
and this path are part of the cache keys, so seed from the commit and path
that production will run. Without the setting, the path defaults to
`SEED_REPO_DIR`.

The wrapper reads the launcher once and passes the frozen result to every
queued stage, so later edits to the launcher do not affect the chain. It does
not freeze the contents of the checkout, image or configuration files; keep
those unchanged until the chain finishes.

Outputs:

```text
OUTPUT_DIR/
  input_run_script.txt                 # original launcher path
  preflight.log
  preflight/write-dryrun/recipe.run.sh  # launcher copy
  preflight/write-dryrun/resolved_recipe.sh
  cache/                              # node shards, SEED_INFO, consolidated/
  logs/                               # per-stage scheduler and training logs
  jobs.tsv                            # stage and job ID
  cached.sqsh                         # baked image, unless --output-image is given
```

`--output-image PATH` writes the baked image to `PATH`, for example next to its
base image. The directory must exist and the file must not.

`--dry-run` writes diagnostics to a separate `OUTPUT_DIR.preview.*` directory, so
the same output name stays available for the real submission. A failed preview
submits nothing; a failed job prevents the jobs that depend on it from running.

### Optional Inductor features

Exporting `MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU=1` and
`MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE=1` before running the wrapper enables
weighted-ReLU startup warmup and score-equivalent autotune reuse. The seed
records both in `SEED_INFO` and the bake stores them as the image's defaults,
so consumers need no extra exports. See
[requirements, memory cost and fallback behavior](../../../tools/persistent_cache/README.md#optional-startup-warmup-and-score-equivalent-autotuning).
Measure whether they help a given recipe before relying on them.

### Monitor and inspect the result

```bash
cat "$OUTPUT_DIR/jobs.tsv"
JOB_IDS=$(awk 'NR > 1 {print $2}' "$OUTPUT_DIR/jobs.tsv" | paste -sd, -)
sacct --jobs="$JOB_IDS" --format=JobID,State,ExitCode,Elapsed
```

Downstream jobs show `Dependency` until their predecessor succeeds. The image
file existing means the bake finished, not that the image check passed.

With `--verify`, inspect the image stage's training log once it finishes:

```bash
IMAGE_JOB_ID=$(awk '$1 == "image" {print $2}' "$OUTPUT_DIR/jobs.tsv")
grep -E 'CACHE ENABLE|CACHE NODE SETUP|CUTE-AOT CACHE.*summary|INDUCTOR-AUTOTUNE RESTART SUMMARY|elapsed time per iteration' \
  "$OUTPUT_DIR/logs/image-$IMAGE_JOB_ID/train_$IMAGE_JOB_ID.log"
```

Look for `mode=read source=image`, `image cache in place` or `image cache copied`,
the reuse and recompile counts, and iteration 1's time compared with the
seed's. See [the cache-counter reference](../../../tools/persistent_cache/README.md#what-to-look-for)
for how `fallback_miss` and `late_hit` interact. Completing without errors
does not by itself mean nothing was recompiled.

### Supported launchers

The launcher must define shell variables `OPTIONS`, `IMAGE_PATH` and
`REPO_DIR`, and may define `VISION_CHECKPOINT_DIR`. It is read in a restricted
shell with side effects disabled, so guards such as `|| exit 1` and a
`SLURM_JOB_ID` check are skipped rather than fatal. The entrypoint comes from
its `exec python ...` line; if it launches differently, set `SEED_ENTRYPOINT`.
`OPTIONS` must keep `--micro-batch-size`, `--seq-length` and
`--tensor-parallel-num-weight-shards`. Containers mount `/scratch` and, when
present, `/lustre`, so everything the launcher uses must sit under one of them.
Paths used in container commands must not contain whitespace or shell
metacharacters.

The seed drops checkpoint loading and saving, TensorBoard, W&B and other
logging outputs, and the batch-size schedule, and loads only the vision encoder
from `VISION_CHECKPOINT_DIR`. Kernels that only run on a resume or checkpoint
path are therefore not seeded.

## Geometry

The seed keeps each rank's kernel shapes (micro-batch, sequence length,
precision, EP and weight shards) and reduces the number of LLM replicas.
Encoder ranks see a smaller batch than in production, so encoder shapes that
only appear at the production batch are not seeded.

| Option | Variable | Default | Meaning |
|---|---|---|---|
| `--llm-dp N` | `SEED_LLM_DP` | 1 | LLM data-parallel replicas; each is `--tensor-parallel-num-weight-shards` ranks |
| `--encoder-dp N` | `SEED_ENCODER_DP` | 64 | vision-encoder ranks; must divide `micro_batch × llm_dp × weight_shards` |
| `--gpus-per-node N` | `SEED_GPUS_PER_NODE` | 4 | node count is `(encoder_dp + llm_dp × weight_shards) / gpus_per_node` |
| – | `SEED_MICROBATCHES` | 1 | `--global-batch-size = micro_batch × llm_dp × weight_shards × this` |
| – | `SEED_ITERS` | 20 | iterations; later specializations can still miss |
| – | `SEED_ENV_OVERRIDE` | – | environment applied after the launcher's, e.g., `MIMALLOC_PURGE_DELAY=0` |

With one LLM replica, the distributed optimizer state is sharded over only
`weight_shards` ranks. If that does not fit, as for larger models, raise
`--llm-dp`. The wrapper rejects geometries that do not fill whole nodes or
divide evenly before submitting anything.

A cold seed also uses a lot of host memory, from activation offload and
compile workers. If the launcher sets `MIMALLOC_PURGE_DELAY=-1`, which never
returns freed memory, a cold seed can be killed for host OOM; seed with
`SEED_ENV_OVERRIDE="MIMALLOC_PURGE_DELAY=0"`.

## Individual stages

The wrapper handles ordering and the frozen recipe. To run stages by hand, set
the same variables the wrapper sets and submit `seed_from_run_script.sbatch`
once per stage. Each command below waits for its job and stops on failure.

```bash
set -e
export RUN_SH=/path/to/launcher.sh
export SEED_REPO_DIR="$PWD"
export SEED_CONTAINER_REPO_DIR=/path/where/production/mounts/its/checkout
export SEED_TOOLS="$SEED_REPO_DIR/tools/persistent_cache"
export SEED_IMAGE=/path/to/base.sqsh
export SEED_NAMESPACE=/path/to/new-cache-run/cache
export SEED_OUT_ROOT=/path/to/new-cache-run/logs
export SEED_ENCODER_DP=64 SEED_LLM_DP=1 SEED_GPUS_PER_NODE=4
export SEED_MICROBATCHES=1 SEED_ITERS=20
export SEED_CONSOLIDATE=1 MCORE_CACHE_STATS=1
unset SEED_DRY_RUN MCORE_CACHE_DIR MCORE_CACHE_MODE MCORE_CACHE_LOCAL
unset MCORE_CACHE_IMAGE_DIR MCORE_CACHE_PREFER_IMAGE SEED_ENABLE_SH
mkdir -p "$SEED_OUT_ROOT"
BAKED_IMAGE=/path/to/images/base+cache.sqsh
SEED_SCRIPT="$SEED_REPO_DIR/nemotron4/scripts/kernel_cache/seed_from_run_script.sbatch"
NODES=$(( (SEED_ENCODER_DP + SEED_LLM_DP * 64) / SEED_GPUS_PER_NODE ))  # 64 weight shards
SUBMIT=(sbatch --wait --account=YOUR_ACCOUNT --nodes=$NODES
        --ntasks-per-node=$SEED_GPUS_PER_NODE --gpus-per-node=$SEED_GPUS_PER_NODE)

# Preview, then seed and consolidate.
SEED_MODE=write SEED_DRY_RUN=1 bash "$SEED_SCRIPT"
SEED_MODE=write "${SUBMIT[@]}" --output="$SEED_OUT_ROOT/slurm-write-%j.out" "$SEED_SCRIPT"

# Warm check against the shared cache, then bake from the same base image.
SEED_MODE=read "${SUBMIT[@]}" --output="$SEED_OUT_ROOT/slurm-read-%j.out" "$SEED_SCRIPT"
BASE_IMAGE="$SEED_IMAGE" MCORE_CACHE_DIR="$SEED_NAMESPACE" \
OUTPUT_IMAGE="$BAKED_IMAGE" TOOLS_DIR="$SEED_TOOLS" \
CACHE_BUILD_REPO="$SEED_REPO_DIR" CACHE_CONTAINER_REPO_DIR="$SEED_CONTAINER_REPO_DIR" \
  sbatch --wait --account=YOUR_ACCOUNT --nodes=1 \
  --output="$SEED_OUT_ROOT/slurm-bake-%j.out" "$SEED_TOOLS/bake_cache_image.sh"

# Warm check with the image's own defaults, not inherited settings.
unset MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE
SEED_MODE=image SEED_IMAGE="$BAKED_IMAGE" "${SUBMIT[@]}" \
  --output="$SEED_OUT_ROOT/slurm-image-%j.out" "$SEED_SCRIPT"
```

Each training stage ends with a `FIRST-ITERATION REPORT` (initialization time,
first iteration times, `[CACHE NODE SETUP]` lines and reuse summaries). Run by
hand, each stage reads the launcher when its job starts rather than using a
frozen copy, so keep the launcher unchanged until the last stage finishes.

## Adapt an existing launcher step by step

Two edits enable the cache: select the baked image and source `enable.sh`
before Python. Training arguments, mounts and the `srun` line stay as they are.

1. **Copy the launcher.**

   ```bash
   cp /path/to/launcher.sh /path/to/launcher_with_cache.sh
   ```

2. **Select the baked image.**

   ```bash
   IMAGE_PATH="/path/to/images/base+cache.sqsh"
   ```

   Keep the training checkout at the code revision and container path used by
   the seed. The image supplies caches; the checkout supplies training code and
   the cache tools.

3. **Source `enable.sh` in the container command.**

   Before:

   ```bash
   RUN_CMD="set -f; cd ${REPO_DIR}; exec python -u -m examples.mimo.pretrain_mimo ${OPTIONS}"
   ```

   After:

   ```bash
   RUN_CMD="set -f; cd ${REPO_DIR}; source tools/persistent_cache/enable.sh && exec python -u -m examples.mimo.pretrain_mimo ${OPTIONS}"
   ```

   This runs inside the container after entering `REPO_DIR`; `&&` stops the job
   if activation fails. With no cache settings inherited, `enable.sh` reads the
   image's cache. It also overrides any `TRITON_CACHE_DIR` the launcher sets.

4. **Check and submit the copy.**

   ```bash
   bash -n /path/to/launcher_with_cache.sh
   diff -u /path/to/launcher.sh /path/to/launcher_with_cache.sh
   sbatch /path/to/launcher_with_cache.sh
   ```

5. **Check reuse in the training log.** Look for `mode=read source=image` and
   `image cache in place` or `image cache copied`. Set `MCORE_CACHE_STATS=1` when
   submitting to print per-rank reuse counters at exit. Data the seed did not
   see can still need new kernels.

To turn off the Inductor features baked into an image for one run, submit with
`MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU=0 MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE=0`.

If submitting from a shell used for seeding, clear its cache settings first:

```bash
unset MCORE_CACHE_DIR MCORE_CACHE_MODE MCORE_CACHE_LOCAL
unset MCORE_CACHE_IMAGE_DIR MCORE_CACHE_PREFER_IMAGE MCORE_CACHE_IMAGE_COPY
unset MCORE_CACHE_SCOPES MCORE_CUTE_DSL_AOT_NAMESPACE MCORE_CUTE_DSL_AOT_NS_FROM
unset MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE
```

Seed each launcher separately and keep one image per base image, code revision
and kernel-shaping configuration; see
[cross-cluster compatibility](../../../tools/persistent_cache/README.md#moving-a-cache-to-another-cluster).
Production can compile additional cases inside its container but never
modifies the `.sqsh`. Re-seed after changing code, dependencies or kernel
shapes.

### Training checkout requirements

The baked image includes CuTe fingerprints, saved Inductor choices and the
finalized `share_storage_ext` and GTP allocator extensions. The training
checkout selected by `REPO_DIR` supplies the cache tools and must include the
GTP allocator's `TORCH_EXTENSIONS_DIR` support; an image cannot change older
allocator code. If using an isolated checkout, bind it at the seeded
container path.
