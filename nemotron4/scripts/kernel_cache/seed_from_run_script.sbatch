#!/bin/bash
# seed_from_run_script.sbatch — build, verify, or A/B the first-iteration kernel
# cache for a Nemotron-4 MIMO launcher (e.g., nemotron4/scripts/derisking/nano_21t/phase1.sh)
# WITHOUT transcribing its recipe.
#
# The production run.sh is sourced in a sandbox (restricted shell, every
# side-effecting command neutralised, output redirection forbidden) so its
# exported environment and its OPTIONS string are taken verbatim; only the
# knobs a bounded seed must change are rewritten (geometry, batch size,
# checkpointing, run length). The recipe's data path is kept, so the seed reads
# production data. Language-model kernel options are preserved; reduced encoder
# batches may leave production encoder shapes unseeded.
#
# Usage (submit-time args carry account/reservation/node count; --nodes must
# equal (SEED_ENCODER_DP + SEED_LLM_DP * weight_shards * tp * pp * cp) / gpus_per_node):
#
#   RUN_SH=/path/to/launcher.sh \
#   SEED_NAMESPACE=/path/to/cache-run/cache \
#   SEED_OUT_ROOT=/path/to/cache-run/logs \
#   SEED_MODE=write \
#   sbatch -A <acct> [--reservation ... --qos ...] --nodes=32 \
#          --output=<dir>/slurm-%j.out seed_from_run_script.sbatch
#
# Modes (SEED_MODE):
#   write   seed: compile cold into node-local scratch, package one shard per
#           node into ${SEED_NAMESPACE}/cache_write/_node/, then consolidate into
#           ${SEED_NAMESPACE}/consolidated/ (SEED_CONSOLIDATE=1, default).
#   read    warm verify from the Lustre namespace (MCORE_CACHE_STATS=1).
#   image   warm verify from the cache baked into SEED_IMAGE (bake_cache_image.sh):
#           no MCORE_CACHE_DIR; enable.sh comes from the training checkout.
#   native  control run with no cache (same geometry) for A/B timing.
#
# Knobs (env): SEED_ENCODER_DP (64), SEED_LLM_DP (1), SEED_ITERS (20),
#   SEED_MICROBATCHES (2), SEED_IMAGE (override the
#   recipe's IMAGE_PATH), SEED_REPO_DIR (override REPO_DIR, e.g., a pinned clone),
#   SEED_CONTAINER_REPO_DIR (default: resolved SEED_REPO_DIR; bind the checkout
#   at this production-visible path to preserve compiled source paths),
#   SEED_TOOLS (default: tools/persistent_cache in the checkout holding this file),
#   SEED_GPUS_PER_NODE (4), SEED_ENTRYPOINT (auto from run.sh), SEED_CONSOLIDATE (1),
#   SEED_CACHE_COMPRESSOR (gzip; zstd needs the zstd CLI or python-zstandard in the
#   consuming image), SEED_ENV_OVERRIDE ("K=V K=V", applied after the recipe env).

#SBATCH -p batch
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --segment=16
#SBATCH -t 01:30:00
#SBATCH --job-name=first_iter_cache_seed

set -euo pipefail
# OPTIONS carries literal shell globs (--ddp-param-name-patterns-for-fp32-local-accumulation
# *output_layer* *router*); they must never expand, exactly as in the production RUN_CMD.
set -f

: "${RUN_SH:?RUN_SH=<production run.sh> is required}"
: "${SEED_OUT_ROOT:?SEED_OUT_ROOT=<writable dir for the logs of this submission> is required}"
SEED_MODE="${SEED_MODE:-write}"
# Slurm executes a spool copy of an sbatch script, so BASH_SOURCE points at the
# spool directory, not this checkout. Resolve the submitted path via scontrol;
# SEED_TOOLS=<checkout>/tools/persistent_cache overrides either way.
_self="${BASH_SOURCE[0]}"
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    _cmd="$( (timeout 30 scontrol show job "${SLURM_JOB_ID}" 2>/dev/null || true) | sed -n 's/^ *Command=//p' | head -1)"
    _cmd="${_cmd%% *}"
    [[ -f "${_cmd}" ]] && _self="${_cmd}"
fi
SEED_TOOLS="${SEED_TOOLS:-$(cd "$(dirname "${_self}")/../../.." && pwd)/tools/persistent_cache}"
SEED_ENCODER_DP="${SEED_ENCODER_DP:-64}"
SEED_LLM_DP="${SEED_LLM_DP:-1}"
SEED_ITERS="${SEED_ITERS:-20}"
SEED_MICROBATCHES="${SEED_MICROBATCHES:-2}"
SEED_GPUS_PER_NODE="${SEED_GPUS_PER_NODE:-4}"
SEED_CONSOLIDATE="${SEED_CONSOLIDATE:-1}"
# SEED_DRY_RUN=1: derive and rewrite the recipe, print the container command, exit
# before any srun. Works outside Slurm (login node) for reviewing a new run.sh.
SEED_DRY_RUN="${SEED_DRY_RUN:-0}"
if [[ "${SEED_DRY_RUN}" == "1" ]]; then
    # Derivation runs in a child shell; production launchers may require this
    # even when srun itself is neutralised for preflight.
    export SLURM_JOB_ID="${SLURM_JOB_ID:-dryrun}"
fi

case "${SEED_MODE}" in
    write|read) : "${SEED_NAMESPACE:?SEED_NAMESPACE=<cache namespace dir> is required for SEED_MODE=${SEED_MODE}}" ;;
    image) : "${SEED_IMAGE:?SEED_IMAGE=<image baked with bake_cache_image.sh> is required for SEED_MODE=image}" ;;
    native) ;;
    *) echo "[FATAL] SEED_MODE=${SEED_MODE}; expected write|read|image|native" >&2; exit 2 ;;
esac
test -r "${RUN_SH}" || { echo "[FATAL] cannot read RUN_SH=${RUN_SH}" >&2; exit 2; }
test -f "${SEED_TOOLS}/enable.sh" || { echo "[FATAL] SEED_TOOLS=${SEED_TOOLS} lacks enable.sh; export SEED_TOOLS=<checkout>/tools/persistent_cache" >&2; exit 2; }

RUN_DIR="${SEED_OUT_ROOT}/${SEED_MODE}-${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}/tensorboard"
cp "${RUN_SH}" "${RUN_DIR}/recipe.run.sh"

# ---------------------------------------------------------------------------
# 1. Derive the recipe from the production script, in a sandbox.
#
# `bash -r` forbids output redirection, cd, exec and slash-qualified commands,
# so the production script's `awk ... > ${RUN_DIR}/x.tmp`, `mkdir -p`, `tee -a`
# and `srun` cannot touch the production run directory; the neutralised
# functions turn the rest into no-ops. Expect one
# "<run.sh>: line N: <path>.tmp: restricted: cannot redirect output" message
# on stderr per blocked redirect -- that is the sandbox working. Because `.`
# may not take a path in a restricted shell, the script's directory is put on
# PATH first. A no-op `exit` lets the source continue past guards such as
# `cmd >> file || exit 1` (the redirect is blocked) or a missing SLURM_JOB_ID
# check, so it still reaches the OPTIONS assignment.
# ---------------------------------------------------------------------------
derive_recipe() {
    local run_sh="$1"
    PATH="$(dirname "${run_sh}"):${PATH}" bash --noprofile --norc -r -c '
        run_sh_name="$1"; run_sh_path="$2"
        # No redirections here either: a restricted shell rejects them.
        srun() { :; }; sbatch() { :; }; mkdir() { :; }; tee() { while IFS= read -r _line; do :; done; }
        mv() { :; }; cp() { :; }; rm() { :; }; git() { :; }; env() { :; }
        sacct() { :; }; scontrol() { printf "Command=%s\n" "${run_sh_path}"; }
        exit() { :; }
        SLURM_JOB_ID="${SLURM_JOB_ID:-0}"
        before="$(export -p)"
        . "${run_sh_name}" || true
        after="$(export -p)"
        printf "%s\n" "__RECIPE_VARS__"
        declare -p OPTIONS IMAGE_PATH REPO_DIR QUANT_CONFIG VISION_CHECKPOINT_DIR SEQ_LEN MAX_BATCH_SIZE NAME || true
        printf "%s\n" "__ENV_BEFORE__"; printf "%s\n" "${before}"
        printf "%s\n" "__ENV_AFTER__"; printf "%s\n" "${after}"
    ' _ "$(basename "${run_sh}")" "${run_sh}"
}

recipe_dump="${RUN_DIR}/recipe_dump.txt"
derive_recipe "${RUN_SH}" > "${recipe_dump}" || {
    echo "[FATAL] the sandboxed source of ${RUN_SH} exited with status $?" >&2
    exit 10
}
# Variables: evaluate at top level (declare -- ... becomes a global here).
eval "$(sed -n '/^__RECIPE_VARS__$/,/^__ENV_BEFORE__$/p' "${recipe_dump}" | sed '1d;$d')"
# Environment: preserve removed exports as well as additions and changes.
sed -n '/^__ENV_BEFORE__$/,/^__ENV_AFTER__$/p' "${recipe_dump}" | sed '1d;$d' | sort > "${RUN_DIR}/env_before.txt"
sed -n '/^__ENV_AFTER__$/,$p' "${recipe_dump}" | sed '1d' | sort > "${RUN_DIR}/env_after.txt"
{
    comm -23 "${RUN_DIR}/env_before.txt" "${RUN_DIR}/env_after.txt" \
        | sed -nE 's/^declare -x ([a-zA-Z_][a-zA-Z_0-9]*)(=.*)?$/\1/p' \
        | grep -vE '^(OLDPWD|PWD|SHLVL|_|PATH)$' \
        | while IFS= read -r name; do printf 'unset %s\n' "${name}"; done
    comm -13 "${RUN_DIR}/env_before.txt" "${RUN_DIR}/env_after.txt" \
        | grep -vE '^declare -x (OLDPWD|PWD|SHLVL|_|PATH)='
} > "${RUN_DIR}/recipe_env.sh" || true
eval "$(cat "${RUN_DIR}/recipe_env.sh")"
# Seed-only environment overrides, applied after the recipe env. They never
# change compiled artifacts; use them to keep a cold seed inside host memory,
# e.g., SEED_ENV_OVERRIDE="MIMALLOC_PURGE_DELAY=0" (with -1, freed host memory
# is never returned, and cold seeds have been OOM-killed at ~330 GiB RSS per rank).
if [[ -n "${SEED_ENV_OVERRIDE:-}" ]]; then
    for _kv in ${SEED_ENV_OVERRIDE}; do
        export "${_kv?}"
        echo "[RECIPE] env override ${_kv}"
    done
fi

[[ -n "${OPTIONS:-}" ]] || { echo "[FATAL] the sandboxed source of ${RUN_SH} produced no OPTIONS" >&2; exit 10; }
IMAGE_PATH="${SEED_IMAGE:-${IMAGE_PATH:?run.sh defined no IMAGE_PATH}}"
REPO_DIR="${SEED_REPO_DIR:-${REPO_DIR:?run.sh defined no REPO_DIR}}"
SEED_CONTAINER_REPO_DIR="${SEED_CONTAINER_REPO_DIR:-${REPO_DIR}}"
for path in "${REPO_DIR}" "${SEED_CONTAINER_REPO_DIR}"; do
    [[ "${path}" =~ ^/[a-zA-Z0-9_./+@=-]+$ && "${path}" != / && "${path}" != *//* &&
       "/${path}/" != */../* && "/${path}/" != */./* ]] || {
        echo "[FATAL] checkout paths must be safe absolute paths without dot segments: ${path}" >&2
        exit 2
    }
done
REPO_DIR="${REPO_DIR%/}"
SEED_CONTAINER_REPO_DIR="${SEED_CONTAINER_REPO_DIR%/}"
test -f "${IMAGE_PATH}" || { echo "[FATAL] image not found: ${IMAGE_PATH}" >&2; exit 2; }
test -d "${REPO_DIR}" || { echo "[FATAL] repo not found: ${REPO_DIR}" >&2; exit 2; }
if [[ -z "${SEED_ENTRYPOINT:-}" ]]; then
    SEED_ENTRYPOINT="$(grep -oE 'exec python[^$"]*' "${RUN_SH}" | head -1 | sed -E 's/^exec //; s/ +$//')"
    SEED_ENTRYPOINT="${SEED_ENTRYPOINT:-python -u -m examples.mimo.pretrain_mimo}"
fi
# Module entrypoints remain unchanged; absolute script/interpreter paths under
# the isolated checkout must use the same visible alias as imports and cwd.
host_repo_prefix="${REPO_DIR}/"
container_repo_prefix="${SEED_CONTAINER_REPO_DIR}/"
SEED_ENTRYPOINT="${SEED_ENTRYPOINT//${host_repo_prefix}/${container_repo_prefix}}"
echo "[RECIPE] run.sh=${RUN_SH}"
echo "[RECIPE] image=${IMAGE_PATH}"
echo "[RECIPE] repo=${REPO_DIR} commit=$(git -c safe.directory='*' -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[RECIPE] container_repo=${SEED_CONTAINER_REPO_DIR}"
echo "[RECIPE] entrypoint=${SEED_ENTRYPOINT}"
echo "[RECIPE] env added by run.sh: $(sed -E 's/^declare -x ([^=]+)=.*/\1/' "${RUN_DIR}/recipe_env.sh" | tr '\n' ' ')"

# ---------------------------------------------------------------------------
# 2. Rewrite only what a bounded seed must change.
# ---------------------------------------------------------------------------
opt_value() {  # opt_value --flag [default]  -> the token after --flag in OPTIONS
    local flag="$1" default="${2:-}" prev="" tok
    for tok in ${OPTIONS}; do
        if [[ "${prev}" == "${flag}" ]]; then echo "${tok}"; return; fi
        prev="${tok}"
    done
    echo "${default}"
}
MBS="$(opt_value --micro-batch-size 1)"
WEIGHT_SHARDS="$(opt_value --tensor-parallel-num-weight-shards 1)"
LLM_TP="$(opt_value --mimo-llm-tp 1)"; LLM_PP="$(opt_value --mimo-llm-pp 1)"; LLM_CP="$(opt_value --mimo-llm-cp 1)"
LLM_EP="$(opt_value --mimo-llm-ep 1)"
LLM_RANKS=$(( SEED_LLM_DP * WEIGHT_SHARDS * LLM_TP * LLM_PP * LLM_CP ))
TOTAL_RANKS=$(( SEED_ENCODER_DP + LLM_RANKS ))
LLM_DATA_PARALLEL=$(( SEED_LLM_DP * WEIGHT_SHARDS ))
GBS=$(( MBS * LLM_DATA_PARALLEL * SEED_MICROBATCHES ))
SLURM_NNODES="${SLURM_NNODES:-$(( TOTAL_RANKS / SEED_GPUS_PER_NODE ))}"
(( TOTAL_RANKS == SLURM_NNODES * SEED_GPUS_PER_NODE )) || {
    echo "[FATAL] geometry needs $(( TOTAL_RANKS / SEED_GPUS_PER_NODE )) nodes (encoder_dp=${SEED_ENCODER_DP} + llm ${LLM_RANKS}), job has ${SLURM_NNODES}" >&2
    exit 11
}
(( LLM_RANKS % LLM_EP == 0 )) || { echo "[FATAL] llm ranks ${LLM_RANKS} not divisible by ep ${LLM_EP}" >&2; exit 11; }
(( (MBS * LLM_DATA_PARALLEL) % SEED_ENCODER_DP == 0 )) || {
    echo "[FATAL] micro_batch*llm_dp*weight_shards (${MBS}*${LLM_DATA_PARALLEL}) must be divisible by encoder_dp ${SEED_ENCODER_DP}" >&2
    exit 11
}

# Flags dropped with all their values (checkpointing, logging sinks, schedule,
# and geometry -- each re-added below in its bounded form). Every --wandb-* flag
# is also dropped: the seed has no W&B credentials, and a production experiment
# name would resume the production W&B run.
declare -A DROP=(
    [--save]=1 [--load]=1 [--load-vision-from]=1 [--save-interval]=1 [--save-retain-interval]=1
    [--async-save]=1 [--use-persistent-ckpt-worker]=1 [--async-ckpt-use-cpu-shm]=1
    [--ckpt-fully-parallel-save]=1 [--ckpt-fully-parallel-load]=1 [--ckpt-assume-constant-structure]=1
    [--ckpt-fully-parallel-save-process-group]=1 [--ckpt-fully-parallel-load-process-group]=1
    [--ckpt-fully-parallel-load-per-rank-objects]=1 [--ckpt-drop-redundant-extra-state]=1
    [--no-ckpt-load-validate-sharding-integrity]=1 [--result-rejected-tracker-filename]=1
    [--exit-duration-in-mins]=1 [--exit-interval]=1 [--tensorboard-dir]=1 [--tensor-metrics]=1
    [--log-interval]=1 [--step-batch-size-schedule]=1 [--global-batch-size]=1
    [--mimo-encoder-dp]=1 [--mimo-llm-offset]=1 [--mimo-llm-dp]=1
)
seed_options=""
dropping=0
for tok in ${OPTIONS}; do
    if [[ "${tok}" == --* ]]; then
        if [[ -n "${DROP[${tok}]:-}" || "${tok}" == --wandb-* ]]; then dropping=1; continue; fi
        dropping=0
    elif (( dropping )); then
        continue
    fi
    seed_options+=" ${tok}"
done
seed_options+=" --mimo-encoder-dp ${SEED_ENCODER_DP} --mimo-llm-offset ${SEED_ENCODER_DP} --mimo-llm-dp ${SEED_LLM_DP}"
seed_options+=" --global-batch-size ${GBS}"
[[ -n "${VISION_CHECKPOINT_DIR:-}" ]] && seed_options+=" --load-vision-from ${VISION_CHECKPOINT_DIR}"
seed_options+=" --exit-interval ${SEED_ITERS} --log-interval 1 --tensorboard-dir ${RUN_DIR}/tensorboard"

# Hard gates: the flags that set per-rank shapes and the seed geometry must survive
# derivation. Optional features such as FP8, recompute and offload are kept when the
# recipe sets them but are not required.
for required in --micro-batch-size --seq-length --tensor-parallel-num-weight-shards; do
    case " ${seed_options} " in
        *" ${required} "*) ;;
        *) echo "[FATAL] production flag missing from derived options: ${required}" >&2; exit 12 ;;
    esac
done
printf '%s\n' ${seed_options} > "${RUN_DIR}/derived_options.txt"
echo "[SEED] mode=${SEED_MODE} nodes=${SLURM_NNODES} encoder_dp=${SEED_ENCODER_DP} llm_dp=${SEED_LLM_DP} weight_shards=${WEIGHT_SHARDS} ep=${LLM_EP} mbs=${MBS} gbs=${GBS} iters=${SEED_ITERS}"

if [[ "${SEED_DRY_RUN}" == "1" ]]; then
    # The pipeline wrapper reuses resolved values instead of sourcing a mutable
    # production launcher for each queued stage. Keep original OPTIONS so the
    # normal per-mode rewriting still runs. declare/printf quote shell values.
    {
        cat "${RUN_DIR}/recipe_env.sh"
        declare -p OPTIONS IMAGE_PATH REPO_DIR
        for name in QUANT_CONFIG VISION_CHECKPOINT_DIR SEQ_LEN MAX_BATCH_SIZE NAME; do
            declare -p "${name}" 2>/dev/null || true
        done
    } > "${RUN_DIR}/resolved_recipe.sh"
    {
        printf 'SEED_IMAGE=%q\n' "${IMAGE_PATH}"
        printf 'SEED_REPO_DIR=%q\n' "${REPO_DIR}"
        printf 'SEED_CONTAINER_REPO_DIR=%q\n' "${SEED_CONTAINER_REPO_DIR}"
        printf 'SEED_ENTRYPOINT=%q\n' "${SEED_ENTRYPOINT}"
        printf 'SEED_NODES=%q\n' "${SLURM_NNODES}"
    } > "${RUN_DIR}/resolved_config.sh"
fi

# ---------------------------------------------------------------------------
# 3. Cache wiring per mode. enable.sh is sourced INSIDE the container command,
#    so no --container-env allow-list is needed (env exported here reaches the
#    container through srun as usual).
# ---------------------------------------------------------------------------
mounts="/scratch:/scratch"
[[ -d /lustre ]] && mounts="${mounts},/lustre:/lustre"
if [[ "${SEED_CONTAINER_REPO_DIR}" != "${REPO_DIR}" ]]; then
    mounts="${mounts},${REPO_DIR}:${SEED_CONTAINER_REPO_DIR}"
fi
enable_line=""
case "${SEED_MODE}" in
    write)
        export MCORE_CACHE_MODE=write MCORE_CACHE_DIR="${SEED_NAMESPACE}"
        # Compile into host /tmp (mounted as /host_tmp); package_node_cache.sh
        # publishes one verified shard per node after training succeeds.
        export MCORE_CACHE_LOCAL="/host_tmp/mcore_cache_${SLURM_JOB_ID}"
        mounts="${mounts},/tmp:/host_tmp"
        mkdir -p "${SEED_NAMESPACE}"
        if [[ -d "${SEED_NAMESPACE}/cache_write/_node" ]] && \
           [[ -n "$(find "${SEED_NAMESPACE}/cache_write/_node" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
            echo "[FATAL] ${SEED_NAMESPACE}/cache_write/_node is not empty; inspect, then rm -rf ${SEED_NAMESPACE}/cache_write to reseed" >&2
            exit 14
        fi
        enable_line="source ${SEED_TOOLS}/enable.sh;"
        ;;
    read)
        export MCORE_CACHE_MODE=read MCORE_CACHE_DIR="${SEED_NAMESPACE}" MCORE_CACHE_STATS=1
        [[ -d "${SEED_NAMESPACE}/consolidated" ]] || { echo "[FATAL] ${SEED_NAMESPACE}/consolidated missing -- seed + consolidate first" >&2; exit 14; }
        # The recipe is re-derived from the launcher at run time, so its image can
        # have changed since the seed. A cache is only valid for the image it was
        # compiled under; say so loudly.
        seed_image="$(sed -n 's/^image=//p' "${SEED_NAMESPACE}/SEED_INFO" 2>/dev/null || true)"
        if [[ -n "${seed_image}" && "${seed_image}" != "$(basename "${IMAGE_PATH}")" ]]; then
            echo "[WARN] namespace was seeded on ${seed_image} but this run uses $(basename "${IMAGE_PATH}"):" >&2
            echo "[WARN] expect misses; verify on the seed image with SEED_IMAGE=<path to ${seed_image}>, or re-seed a new namespace for the new image" >&2
        fi
        enable_line="source ${SEED_TOOLS}/enable.sh;"
        ;;
    image)
        export MCORE_CACHE_MODE=read MCORE_CACHE_STATS=1
        unset MCORE_CACHE_DIR
        enable_line="source ${SEED_ENABLE_SH:-${SEED_CONTAINER_REPO_DIR}/tools/persistent_cache/enable.sh};"
        ;;
    native)
        unset MCORE_CACHE_DIR MCORE_CACHE_MODE
        ;;
esac

# ---------------------------------------------------------------------------
# 4. Train (bounded window). Same srun shape as the production launcher.
# ---------------------------------------------------------------------------
TRAIN_LOG="${RUN_DIR}/train_${SLURM_JOB_ID}.log"
RUN_CMD="set -e; set -f; cd ${SEED_CONTAINER_REPO_DIR}; export PYTHONPATH=${SEED_CONTAINER_REPO_DIR}\${PYTHONPATH:+:\${PYTHONPATH}}; ${enable_line} exec ${SEED_ENTRYPOINT} ${seed_options}"
echo "[SEED] RUN_CMD=${RUN_CMD}" > "${RUN_DIR}/run_cmd.txt"
if [[ "${SEED_DRY_RUN}" == "1" ]]; then
    echo "[DRY RUN] image=${IMAGE_PATH}"
    echo "[DRY RUN] mounts=${mounts}"
    echo "[DRY RUN] env: $(sed -E 's/^declare -x //' "${RUN_DIR}/recipe_env.sh" | tr '\n' ' ')"
    echo "[DRY RUN] container command:"
    printf '    %s\n' ${RUN_CMD}
    echo "[DRY RUN] nothing submitted; artifacts in ${RUN_DIR}"
    exit 0
fi

if [[ "${SEED_MODE}" == "write" ]]; then
    # Per-node precheck: host /tmp must be fresh, executable and roomy.
    export MCORE_LOCAL_MIN_BYTES=$((64 * 1024 * 1024 * 1024))
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --mpi=none \
        --output="${RUN_DIR}/precheck-${SLURM_JOB_ID}-%N.log" \
        bash -ceu '
            base="/tmp/mcore_cache_${SLURM_JOB_ID}"
            [[ ! -e "${base}" && ! -e "${base}.setup.lock" ]]
            available=$(df --output=avail -B1 /tmp | tail -n 1 | tr -d " ")
            (( available >= MCORE_LOCAL_MIN_BYTES ))
            probe="/tmp/.mcore_exec_probe_${SLURM_JOB_ID}"; printf "#!/bin/sh\nexit 0\n" > "${probe}"; chmod 700 "${probe}"; "${probe}"; rm -f "${probe}"
            echo "LOCAL_PRECHECK_OK host=$(hostname) available_bytes=${available}"'
fi

set +e
srun -l --mpi=none \
    --container-image="${IMAGE_PATH}" \
    --container-mounts="${mounts}" \
    --no-container-mount-home \
    --output="${TRAIN_LOG}" \
    bash -c "${RUN_CMD}"
train_rc=$?
set -e
echo "[SEED] training srun exit=${train_rc} log=${TRAIN_LOG}"

# ---------------------------------------------------------------------------
# 5. Report (all modes).
# ---------------------------------------------------------------------------
set +o pipefail
echo "===================== FIRST-ITERATION REPORT (${SEED_MODE}) ====================="
grep -h -m1 -E 'time to initialize megatron' "${TRAIN_LOG}" | sed 's/^[0-9]*: *//' || true
grep -hoE 'iteration +[0-9]+/ *[0-9]+ .*elapsed time per iteration \(ms\): [0-9.]+' "${TRAIN_LOG}" \
    | sed -E 's/ \| consumed samples.*elapsed time per iteration \(ms\)/ elapsed_ms/' | head -5 | sed 's/^/    /'
grep -h '\[CACHE ENABLE\]' "${TRAIN_LOG}" | head -2 | sed 's/^[0-9]*: *//'
grep -h '\[CACHE NODE SETUP\]' "${TRAIN_LOG}" | sed 's/^[0-9]*: *//' | sort | uniq -c | sort -rn | head -6
grep -hE '\[CUTE-AOT CACHE\] summary|INDUCTOR-AUTOTUNE RESTART SUMMARY|HYBRID-EP CACHE\] enabled' "${TRAIN_LOG}" | sed -E 's/^[0-9]*: *//; s/pid=[0-9]+/pid=P/; s/dir=[^ ]+//' | sort | uniq -c | sort -rn | head -8
echo "================================================================================="
set -o pipefail
(( train_rc == 0 )) || { echo "[FATAL] training failed (rc=${train_rc}); see ${TRAIN_LOG}" >&2; exit "${train_rc}"; }

# ---------------------------------------------------------------------------
# 6. Write mode: publish + seal.
# ---------------------------------------------------------------------------
if [[ "${SEED_MODE}" == "write" ]]; then
    for iteration in 1 "${SEED_ITERS}"; do
        grep -Eq " iteration[[:space:]]+${iteration}/" "${TRAIN_LOG}" || {
            echo "[TRAIN GATE] iteration ${iteration} marker missing from ${TRAIN_LOG}" >&2; exit 20; }
    done
    export MCORE_CACHE_PACKAGE_LOCAL="/tmp/mcore_cache_${SLURM_JOB_ID}"
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --mpi=none \
        --output="${RUN_DIR}/package-${SLURM_JOB_ID}-%N.log" \
        bash "${SEED_TOOLS}/package_node_cache.sh"
    published=$(find "${SEED_NAMESPACE}/cache_write/_node" -mindepth 1 -maxdepth 1 -type d -name 'node-*' 2>/dev/null | wc -l || true)
    (( published == SLURM_NNODES )) || {
        echo "[PUBLISH GATE] ${published}/${SLURM_NNODES} node shards -- namespace incomplete, not consolidating" >&2; exit 22; }
    echo "[SEED] ${published}/${SLURM_NNODES} node shards published under ${SEED_NAMESPACE}/cache_write/_node"

    inductor_prewarm=0; inductor_equivalent_cache=0
    [[ "${MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU:-0}" != 1 ]] || inductor_prewarm=1
    [[ "${MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE:-0}" != 1 ]] || inductor_equivalent_cache=1
    {
        echo "run_sh=${RUN_SH}"
        echo "run_sh_sha256=$(sha256sum "${RUN_SH}" | cut -d' ' -f1)"
        echo "repo_dir=${REPO_DIR}"
        echo "repo_container_dir=${SEED_CONTAINER_REPO_DIR}"
        echo "repo_commit=$(git -c safe.directory='*' -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "inductor_prewarm_weighted_relu=${inductor_prewarm}"
        echo "inductor_autotune_equivalent_cache=${inductor_equivalent_cache}"
        echo "image=$(basename "${IMAGE_PATH}")"
        echo "geometry=encoder_dp:${SEED_ENCODER_DP} llm_dp:${SEED_LLM_DP} weight_shards:${WEIGHT_SHARDS} ep:${LLM_EP} tp:${LLM_TP} mbs:${MBS} gbs:${GBS} nodes:${SLURM_NNODES}"
        echo "iterations=${SEED_ITERS}"
        echo "env_override=${SEED_ENV_OVERRIDE:-}"
        echo "job_id=${SLURM_JOB_ID}"
        echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "tools=${SEED_TOOLS} $(git -c safe.directory='*' -C "${SEED_TOOLS}" rev-parse --short HEAD 2>/dev/null || true)"
    } > "${SEED_NAMESPACE}/SEED_INFO"

    if [[ "${SEED_CONSOLIDATE}" == "1" ]]; then
        # gzip by default: the consuming image may ship neither the zstd CLI nor
        # python-zstandard (26.07 deps v8 has neither), and `tar -xf` reads gzip
        # everywhere. The sealed cache is ~10 MB, so the ratio does not matter.
        srun --nodes=1 --ntasks=1 --mpi=none --output="${RUN_DIR}/consolidate-${SLURM_JOB_ID}.log" \
            env MCORE_CACHE_DIR="${SEED_NAMESPACE}" CACHE_COMPRESSOR="${SEED_CACHE_COMPRESSOR:-gzip}" \
            bash "${SEED_TOOLS}/consolidate_cache.sh"
        cat "${RUN_DIR}/consolidate-${SLURM_JOB_ID}.log"
        ls -la "${SEED_NAMESPACE}/consolidated"
        echo "[SEED COMPLETE] sealed cache: ${SEED_NAMESPACE}/consolidated"
        echo "[NEXT] verify:  SEED_MODE=read  (same submit line)"
        echo "[NEXT] bake:    BASE_IMAGE=${IMAGE_PATH} MCORE_CACHE_DIR=${SEED_NAMESPACE} CACHE_BUILD_REPO=${REPO_DIR} CACHE_CONTAINER_REPO_DIR=${SEED_CONTAINER_REPO_DIR} OUTPUT_IMAGE=<new.sqsh> sbatch ${SEED_TOOLS}/bake_cache_image.sh"
    else
        echo "[NEXT] MCORE_CACHE_DIR=${SEED_NAMESPACE} bash ${SEED_TOOLS}/consolidate_cache.sh"
    fi
fi
