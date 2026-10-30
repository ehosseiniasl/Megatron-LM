#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Submit a bounded seed and image bake from a Nemotron4/MIMO production recipe.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash seed_and_bake.sh --run-script RUN_SH --output-dir NEW_DIR --account ACCOUNT [options]

Run on the Slurm login node, from a checkout on /scratch or /lustre.
RUN_SH must use the Nemotron4/MIMO launcher format (OPTIONS, IMAGE_PATH,
REPO_DIR); it is derived with seed_from_run_script.sbatch, not submitted.

  --run-script PATH   Production launcher to derive the seed from
  --output-dir PATH   New shared directory for cache/ and logs/
  --output-image PATH New .sqsh for the baked image (default: OUTPUT_DIR/cached.sqsh)
  --account NAME      Slurm account for all stages
  --partition NAME    Slurm partition (default: batch)
  --qos NAME          Optional Slurm QOS for all stages
  --reservation NAME  Optional Slurm reservation for all stages
  --llm-dp N          LLM data-parallel replicas in the seed (default: 1); each
                      replica is --tensor-parallel-num-weight-shards ranks
  --encoder-dp N      Vision-encoder ranks in the seed (default: 64)
  --gpus-per-node N   GPUs per node, used for the node count (default: 4)
  --verify            Add warm checks: seed -> read -> bake -> image
  --dry-run           Derive the recipe and print submissions without submitting
  -h, --help          Show this help

Default chain: seed -> bake. Jobs use afterok dependencies; this command returns
after submission. A failed stage prevents its dependents from running.
The node count and global batch follow from the geometry options and the
recipe's micro-batch size and weight shards. Raise --llm-dp when the optimizer
state of one 64-way replica does not fit, as for larger models. SEED_* overrides
are supported (e.g., SEED_REPO_DIR, SEED_IMAGE, SEED_ITERS); the geometry options
override SEED_LLM_DP, SEED_ENCODER_DP and SEED_GPUS_PER_NODE.
SEED_CONTAINER_REPO_DIR binds the isolated
checkout at a production-visible path (default: resolved SEED_REPO_DIR).
This wrapper defaults SEED_MICROBATCHES to 1
and always consolidates. Use a pinned checkout and an immutable base image.
EOF
}

fail() { echo "[PIPELINE ERROR] $*" >&2; exit 2; }
run_script=""; output_dir=""; output_image=""; account=""; partition=batch; qos=""; reservation=""
llm_dp="${SEED_LLM_DP:-1}"; encoder_dp="${SEED_ENCODER_DP:-64}"; gpus_per_node="${SEED_GPUS_PER_NODE:-4}"
verify=0; dry_run=0
while (( $# )); do
    case "$1" in
        --run-script|--output-dir|--output-image|--account|--partition|--qos|--reservation|\
        --llm-dp|--encoder-dp|--gpus-per-node)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || fail "$1 needs a value"
            case "$1" in
                --run-script) run_script="$2" ;;
                --output-dir) output_dir="$2" ;;
                --output-image) output_image="$2" ;;
                --account) account="$2" ;;
                --partition) partition="$2" ;;
                --qos) qos="$2" ;;
                --reservation) reservation="$2" ;;
                --llm-dp) llm_dp="$2" ;;
                --encoder-dp) encoder_dp="$2" ;;
                --gpus-per-node) gpus_per_node="$2" ;;
            esac
            shift 2 ;;
        --verify) verify=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown argument: $1 (see --help)" ;;
    esac
done
[[ -n "${run_script}" && -n "${output_dir}" && -n "${account}" ]] || {
    usage >&2; fail "--run-script, --output-dir and --account are required";
}
[[ -r "${run_script}" && -f "${run_script}" ]] || fail "cannot read ${run_script}"
for value in "${llm_dp}" "${encoder_dp}" "${gpus_per_node}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "--llm-dp, --encoder-dp and --gpus-per-node take positive integers, got: ${value}"
done
[[ -z "${SLURM_JOB_ID:-}" ]] || fail "run this command on the login node, outside a Slurm job"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "output directory already exists: ${output_dir}"
(( dry_run )) || command -v sbatch >/dev/null || fail "sbatch is not available"
run_script="$(cd "$(dirname "${run_script}")" && pwd)/$(basename "${run_script}")"
mkdir -p "$(dirname "${output_dir}")"
output_dir="$(cd "$(dirname "${output_dir}")" && pwd)/$(basename "${output_dir}")"
if [[ -n "${output_image}" ]]; then
    [[ "${output_image}" == *.sqsh ]] || fail "--output-image must end in .sqsh: ${output_image}"
    [[ -d "$(dirname "${output_image}")" ]] || fail "--output-image directory does not exist: $(dirname "${output_image}")"
    output_image="$(cd "$(dirname "${output_image}")" && pwd)/$(basename "${output_image}")"
    [[ ! -e "${output_image}" && ! -L "${output_image}" ]] || fail "output image already exists: ${output_image}"
else
    output_image="${output_dir}/cached.sqsh"
fi

# A preview keeps diagnostics beside the intended output, leaving that output
# available for the identical command without --dry-run.
if (( dry_run )); then
    work_dir="$(mktemp -d "${output_dir}.preview.XXXXXX")"
else
    mkdir "${output_dir}"
    work_dir="${output_dir}"
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SEED_TOOLS="$(cd "${script_dir}/../../.." && pwd)/tools/persistent_cache"
seed_script="${script_dir}/seed_from_run_script.sbatch"
export RUN_SH="${run_script}" SEED_OUT_ROOT="${work_dir}/preflight"
export SEED_NAMESPACE="${work_dir}/cache"
export SEED_MICROBATCHES="${SEED_MICROBATCHES:-1}"
export SEED_LLM_DP="${llm_dp}" SEED_ENCODER_DP="${encoder_dp}" SEED_GPUS_PER_NODE="${gpus_per_node}"
export SEED_CONSOLIDATE=1 MCORE_CACHE_STATS=1
unset MCORE_CACHE_DIR MCORE_CACHE_MODE MCORE_CACHE_LOCAL SEED_ENABLE_SH
unset MCORE_CACHE_IMAGE_DIR MCORE_CACHE_PREFER_IMAGE
unset SLURM_NNODES

echo "[PIPELINE] deriving ${run_script}"
if ! SEED_MODE=write SEED_DRY_RUN=1 bash "${seed_script}" > "${work_dir}/preflight.log" 2>&1; then
    cat "${work_dir}/preflight.log" >&2
    fail "recipe preflight failed; no jobs submitted (see ${work_dir}/preflight.log)"
fi
plan_dir="${SEED_OUT_ROOT}/write-dryrun"
# These files contain shell-quoted values emitted by the helper, not log text.
source "${plan_dir}/resolved_config.sh"
# The underlying launchers interpolate these paths into container commands.
for path in "${output_dir}" "${output_image}" "${SEED_TOOLS}" "${SEED_REPO_DIR}" "${SEED_CONTAINER_REPO_DIR}" "${SEED_IMAGE}"; do
    [[ "${path}" =~ ^/[a-zA-Z0-9_./+@=-]+$ ]] || fail "unsupported path (use absolute paths without whitespace or shell metacharacters): ${path}"
done
export SEED_IMAGE SEED_REPO_DIR SEED_CONTAINER_REPO_DIR SEED_ENTRYPOINT
export RUN_SH="${plan_dir}/resolved_recipe.sh"
export SEED_NAMESPACE="${output_dir}/cache" SEED_OUT_ROOT="${output_dir}/logs"
printf '%s\n' "${run_script}" > "${work_dir}/input_run_script.txt"
echo "[PIPELINE] ${SEED_NODES} seed nodes; base image: ${SEED_IMAGE}"
echo "[PIPELINE] resolved recipe: ${RUN_SH}"

common=(--parsable --export=ALL --account="${account}" --partition="${partition}")
[[ -z "${qos}" ]] || common+=(--qos="${qos}")
[[ -z "${reservation}" ]] || common+=(--reservation="${reservation}")
if (( ! dry_run )); then
    mkdir -p "${SEED_OUT_ROOT}"
    printf 'stage\tjob_id\n' > "${work_dir}/jobs.tsv"
fi
previous_id=""
submit() {
    local stage="$1" nodes="${SEED_NODES}" tasks_per_node="${SEED_GPUS_PER_NODE}"
    local script="${seed_script}" reply job_id
    local command=(env SEED_DRY_RUN=0)
    if [[ "${stage}" == bake ]]; then
        nodes=1; tasks_per_node=1
        script="${SEED_TOOLS}/bake_cache_image.sh"
        command+=(BASE_IMAGE="${SEED_IMAGE}" MCORE_CACHE_DIR="${SEED_NAMESPACE}"
                  OUTPUT_IMAGE="${output_image}" TOOLS_DIR="${SEED_TOOLS}"
                  CACHE_BUILD_REPO="${SEED_REPO_DIR}"
                  CACHE_CONTAINER_REPO_DIR="${SEED_CONTAINER_REPO_DIR}"
                  INSTALL_DEST=/opt/mcore_cache)
    else
        command+=(SEED_MODE="${stage}")
        [[ "${stage}" != image ]] || command+=(SEED_IMAGE="${output_image}")
    fi
    command+=(sbatch "${common[@]}" --nodes="${nodes}"
              --ntasks="$((nodes * tasks_per_node))" --ntasks-per-node="${tasks_per_node}"
              --gpus-per-node="${SEED_GPUS_PER_NODE}"
              --output="${SEED_OUT_ROOT}/slurm-${stage}-%j.out")
    [[ -z "${previous_id}" ]] || command+=(--dependency="afterok:${previous_id}" --kill-on-invalid-dep=yes)
    command+=("${script}")
    if (( dry_run )); then
        printf '[DRY RUN] '; printf '%q ' "${command[@]}"; printf '\n'
        previous_id="${stage}_JOB_ID"
        return
    fi
    if ! reply="$("${command[@]}")"; then
        fail "${stage} submission failed; earlier jobs remain submitted (see ${work_dir}/jobs.tsv)"
    fi
    job_id="${reply%%;*}" # --parsable may append ;cluster
    [[ "${job_id}" =~ ^[0-9]+$ ]] || fail "unexpected sbatch response: ${reply}; inspect the queue before retrying"
    printf '%s\t%s\n' "${stage}" "${job_id}" >> "${work_dir}/jobs.tsv"
    echo "[PIPELINE] submitted ${stage}: ${job_id}"
    previous_id="${job_id}"
}

submit write
(( ! verify )) || submit read
submit bake
(( ! verify )) || submit image

if (( dry_run )); then
    echo "[DRY RUN] no jobs submitted; diagnostics: ${work_dir}"
else
    echo "[PIPELINE] job IDs: ${work_dir}/jobs.tsv"
    echo "[PIPELINE] image after successful bake: ${output_image}"
    echo "[PIPELINE] production: select that image and source tools/persistent_cache/enable.sh from the training checkout before Python"
fi
