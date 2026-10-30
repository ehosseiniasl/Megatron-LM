#!/bin/bash
# bake_cache_image.sh — derive a container image that carries a sealed
# first-iteration kernel cache at /opt/mcore_cache, so
# consuming jobs need neither Lustre cache traffic nor per-node extraction:
# the cache is already on node-local storage the moment the container starts.
#
# Mechanism: pyxis runs the installer inside the BASE image and
# --container-save writes the resulting filesystem out as a new squashfs
# (the same route dependencies/install_sbatch.sh uses for the training image).
# The container must run on the image's architecture (aarch64 here), so this
# is a one-GPU-node job, ~15 min for a 28 GB image.
#
#   BASE_IMAGE=/path/to/base.sqsh \
#   MCORE_CACHE_DIR=/path/to/cache-run/cache \
#   OUTPUT_IMAGE=/path/to/base+cache.sqsh \
#   sbatch -A <account> [--reservation ... --qos ...] tools/persistent_cache/bake_cache_image.sh
#
# Optional: TOOLS_DIR (default: the directory of this script), MCORE_CACHE_SCOPES,
#           INSTALL_DEST (default /opt/mcore_cache),
#           CACHE_BUILD_REPO (matching training checkout, default TOOLS_DIR/../..),
#           CACHE_CONTAINER_REPO_DIR (its container-visible path, default same),
#           CACHE_NCCL_TOPO_FILE and CACHE_NCCL_TOPO_NAMESPACE (optional pair;
#             topology XML validated against the bake node, independent of the
#             seven kernel-cache archives).
# The two optional Inductor settings are inherited from the sealed SEED_INFO,
# so consuming this image needs no extra feature flags in the training script.
#
# Afterwards, a consuming launcher only needs, inside its container command:
#   cd "${REPO_DIR}" && source tools/persistent_cache/enable.sh
# before `exec python ...` (no MCORE_CACHE_DIR, no --container-env).
# Cache tools stay in the matching training checkout; they are not copied into
# the image. For a custom INSTALL_DEST, also set MCORE_CACHE_IMAGE_DIR to it.

#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 00:45:00
#SBATCH --job-name=first_iter_cache_bake_image

set -euo pipefail

: "${BASE_IMAGE:?BASE_IMAGE=<base .sqsh> is required}"
: "${MCORE_CACHE_DIR:?MCORE_CACHE_DIR=<cache namespace with consolidated/> is required}"
: "${OUTPUT_IMAGE:?OUTPUT_IMAGE=<new .sqsh> is required}"
INSTALL_DEST="${INSTALL_DEST:-/opt/mcore_cache}"
# Slurm executes a spool copy of an sbatch script, so BASH_SOURCE points at the
# spool directory, not this checkout. Resolve the submitted path via scontrol;
# TOOLS_DIR=<checkout>/tools/persistent_cache overrides either way.
_self="${BASH_SOURCE[0]}"
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    _cmd="$( (timeout 30 scontrol show job "${SLURM_JOB_ID}" 2>/dev/null || true) | sed -n 's/^ *Command=//p' | head -1)"
    _cmd="${_cmd%% *}"
    [[ -f "${_cmd}" ]] && _self="${_cmd}"
fi
TOOLS_DIR="${TOOLS_DIR:-$(cd "$(dirname "${_self}")" && pwd)}"
CACHE_BUILD_REPO="${CACHE_BUILD_REPO:-$(cd "${TOOLS_DIR}/../.." && pwd)}"
CACHE_CONTAINER_REPO_DIR="${CACHE_CONTAINER_REPO_DIR:-${CACHE_BUILD_REPO}}"
for path in "${CACHE_BUILD_REPO}" "${CACHE_CONTAINER_REPO_DIR}"; do
    [[ "${path}" =~ ^/[a-zA-Z0-9_./+@=-]+$ && "${path}" != / && "${path}" != *//* &&
       "/${path}/" != */../* && "/${path}/" != */./* ]] || {
        echo "error: checkout paths must be safe absolute paths without dot segments" >&2
        exit 2
    }
done
test -d "${CACHE_BUILD_REPO}"

test -f "${BASE_IMAGE}"
if [[ ! -f "${TOOLS_DIR}/install_image_cache.py" ]]; then
    echo "error: ${TOOLS_DIR} has no install_image_cache.py; export TOOLS_DIR=<checkout>/tools/persistent_cache" >&2
    exit 2
fi
if [[ ! -d "${MCORE_CACHE_DIR}/consolidated" ]]; then
    echo "error: ${MCORE_CACHE_DIR} has no consolidated/ -- run consolidate_cache.sh first" >&2
    exit 2
fi
# Read only allowlisted 0/1 settings. Do not source cache metadata as shell code
# or infer settings from the bake job's inherited runtime environment.
installer_flags=""
if [[ -n "${CACHE_NCCL_TOPO_FILE:-}" || -n "${CACHE_NCCL_TOPO_NAMESPACE:-}" ]]; then
    if [[ -z "${CACHE_NCCL_TOPO_FILE:-}" || -z "${CACHE_NCCL_TOPO_NAMESPACE:-}" ]]; then
        echo "error: CACHE_NCCL_TOPO_FILE and CACHE_NCCL_TOPO_NAMESPACE must be supplied together" >&2
        exit 2
    fi
    [[ "${CACHE_NCCL_TOPO_FILE}" =~ ^/[a-zA-Z0-9_./+@=-]+$ &&
       "${CACHE_NCCL_TOPO_FILE}" != *//* &&
       "/${CACHE_NCCL_TOPO_FILE}/" != */../* && "/${CACHE_NCCL_TOPO_FILE}/" != */./* ]] || {
        echo "error: CACHE_NCCL_TOPO_FILE must be a safe absolute path without dot segments" >&2
        exit 2
    }
    [[ "${CACHE_NCCL_TOPO_NAMESPACE}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
        echo "error: CACHE_NCCL_TOPO_NAMESPACE must be an alphanumeric label with optional '.', '_' or '-'" >&2
        exit 2
    }
    [[ -r "${CACHE_NCCL_TOPO_FILE}" && -f "${CACHE_NCCL_TOPO_FILE}" ]] || {
        echo "error: CACHE_NCCL_TOPO_FILE is not a readable file" >&2; exit 2;
    }
    [[ -f "${TOOLS_DIR}/prepare_nccl_topology_template.py" ]] || {
        echo "error: topology preparation helper is missing from TOOLS_DIR" >&2; exit 2;
    }
    installer_flags+=" --nccl-topo-file '${CACHE_NCCL_TOPO_FILE}' --nccl-topo-namespace '${CACHE_NCCL_TOPO_NAMESPACE}'"
fi
for setting in inductor_prewarm_weighted_relu inductor_autotune_equivalent_cache; do
    value=0
    if [[ -f "${MCORE_CACHE_DIR}/SEED_INFO" ]]; then
        # Count records before command substitution can strip empty trailing
        # values and hide a duplicate such as "setting=1\nsetting=\n".
        matches="$(awk -v key="${setting}=" 'index($0, key) == 1 { count++ } END { print count + 0 }' "${MCORE_CACHE_DIR}/SEED_INFO")"
        case "${matches}" in
            0) ;;
            1) value="$(sed -n "s/^${setting}=//p" "${MCORE_CACHE_DIR}/SEED_INFO")" ;;
            *) echo "error: invalid or repeated ${setting} in SEED_INFO" >&2; exit 2 ;;
        esac
    fi
    case "${value}" in
        0) ;;
        1)
            case "${setting}" in
                inductor_prewarm_weighted_relu) installer_flags+=' --prewarm-weighted-relu' ;;
                inductor_autotune_equivalent_cache) installer_flags+=' --equivalent-autotune-cache' ;;
            esac ;;
        *) echo "error: invalid or repeated ${setting} in SEED_INFO" >&2; exit 2 ;;
    esac
done
# --container-save overwrites its target. Refuse to clobber an image jobs may
# already be running against; pick a new name instead.
if [[ -e "${OUTPUT_IMAGE}" ]]; then
    echo "error: ${OUTPUT_IMAGE} already exists. Choose a new OUTPUT_IMAGE." >&2
    exit 1
fi
mkdir -p "$(dirname "${OUTPUT_IMAGE}")"
# --container-save writes the filesystem out whatever the step exited with, so a
# failed or cancelled install would leave a plausible-looking image behind. Save
# under a name that says so and hard-link it into place only after the installer
# returned zero; OUTPUT_IMAGE then exists only for a complete bake (same pattern
# as dependencies/install_sbatch.sh).
PARTIAL_IMAGE="${OUTPUT_IMAGE%.sqsh}.partial-${SLURM_JOB_ID:-$$}.sqsh"
for leftover in "$(dirname "${OUTPUT_IMAGE}")"/*.partial-*.sqsh; do
    [[ -e "${leftover}" ]] || break
    echo "warning: ${leftover} is a partial image from an earlier bake (safe to delete once unused)" >&2
done

# Mount every filesystem the namespace and tools might live on.
mounts="/scratch:/scratch"
[[ -d /lustre ]] && mounts="${mounts},/lustre:/lustre"
if [[ "${CACHE_CONTAINER_REPO_DIR}" != "${CACHE_BUILD_REPO}" ]]; then
    mounts="${mounts},${CACHE_BUILD_REPO}:${CACHE_CONTAINER_REPO_DIR}"
fi
if [[ -n "${CACHE_NCCL_TOPO_FILE:-}" ]]; then
    mounts="${mounts},${CACHE_NCCL_TOPO_FILE}:${CACHE_NCCL_TOPO_FILE}:ro"
fi

echo "[BAKE] base   : ${BASE_IMAGE}"
echo "[BAKE] cache  : ${MCORE_CACHE_DIR}"
echo "[BAKE] tools  : ${TOOLS_DIR}"
echo "[BAKE] output : ${OUTPUT_IMAGE}"

install_status=0
srun --nodes=1 --ntasks=1 --mpi=none \
    --container-image="${BASE_IMAGE}" \
    --container-mounts="${mounts}" \
    --container-writable \
    --container-save="${PARTIAL_IMAGE}" \
    --no-container-mount-home \
    bash -c "set -euo pipefail
        python3 '${TOOLS_DIR}/install_image_cache.py' \
            --namespace '${MCORE_CACHE_DIR}' --dest '${INSTALL_DEST}' --tools '${TOOLS_DIR}' \
            --training-repo '${CACHE_CONTAINER_REPO_DIR}' \
            --source-image '$(basename "${BASE_IMAGE}")' ${installer_flags} ${MCORE_CACHE_SCOPES:+--scopes \"${MCORE_CACHE_SCOPES}\"}
        echo '[BAKE] MANIFEST:'; cat '${INSTALL_DEST}/MANIFEST'
        du -sh '${INSTALL_DEST}'" || install_status=$?

if (( install_status != 0 )); then
    echo "error: installer exited ${install_status}; removing ${PARTIAL_IMAGE}" >&2
    rm -f "${PARTIAL_IMAGE}"
    exit "${install_status}"
fi
test -s "${PARTIAL_IMAGE}"
# link() fails if OUTPUT_IMAGE appeared meanwhile, so a concurrent bake can never
# replace an image that jobs may already be reading.
if ! ln "${PARTIAL_IMAGE}" "${OUTPUT_IMAGE}"; then
    echo "error: ${OUTPUT_IMAGE} appeared while this bake ran; the finished image is ${PARTIAL_IMAGE}" >&2
    exit 1
fi
rm -f "${PARTIAL_IMAGE}"
echo "[BAKE] wrote ${OUTPUT_IMAGE} ($(du -h "${OUTPUT_IMAGE}" | cut -f1))"
echo "[BAKE] consume: from the matching training checkout, source tools/persistent_cache/enable.sh before exec python"
if [[ "${INSTALL_DEST}" != /opt/mcore_cache ]]; then
    echo "[BAKE] custom cache location: export MCORE_CACHE_IMAGE_DIR=${INSTALL_DEST} before sourcing enable.sh"
fi
