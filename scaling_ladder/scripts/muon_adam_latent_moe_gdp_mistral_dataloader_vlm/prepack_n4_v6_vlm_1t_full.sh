#!/bin/bash

# Build and check the 8k, 1T, full-data VLM epoch indexes.

#SBATCH --partition=cpu
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --dependency=singleton
#SBATCH --job-name=prepack_n4v6_vlm_1t_full
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
unset PIXI_PROJECT_MANIFEST PIXI_PROJECT_ROOT PIXI_ENVIRONMENT_NAME VIRTUAL_ENV
unset AWS_PROFILE MISTRAL_LOCAL_IMAGE_ROOT MISTRAL_S3_ENDPOINT_URL
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

readonly REPO_DIR="${SLURM_SUBMIT_DIR}"
readonly EXPECTED_COMMIT="${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT set to the tested revision}"
readonly SOURCE_CONFIG="${REPO_DIR}/mistral_adapter/strawman_60t_v0.6_20260901_phase1_80_phase2_20_smart/phase1.yaml"
readonly PRETOKENIZED_DATA="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron-v4/sandia/tokenized/n4_v6_262k_ml4_regex"
readonly OUTPUT_ROOT="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/sasatheesh/data/m-lm-mistralai"
readonly IMAGE_PATH="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron4/images/nvidia+pytorch+26.07-py3+deps+v5.sqsh"
readonly S3_NAMESPACE_CONFIG="${REPO_DIR}/mistral_adapter/example_configs/n4_coalition_image_namespaces.json"
readonly NUM_TOTAL_TOKENS=999555072000
readonly PERCENTAGE_DATA_MULTIPLIER=1.0
readonly NUM_WORKERS="${SLURM_NTASKS}"
readonly PROCESSES_PER_WORKER=12
readonly IMAGE_TRANSFORM_WORKERS=4

readonly CLUSTER_NAME="$(scontrol show config | awk '$1 == "ClusterName" { print $3 }')"
case "${CLUSTER_NAME}" in
    oci-aga-slurm-1)
        CLUSTER_TAG=aga
        PYTHON_PACKAGES="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/pengx/projects/nv-mistralai-megatron/examples/multimodal/migration/running_text_chain/outputs/container-python-packages"
        LOCAL_IMAGE_ROOT="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron-v4/sandia/image_buckets"
        ;;
    aws-cmh-slurm-1)
        CLUSTER_TAG=cmh
        PYTHON_PACKAGES="/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/sasatheesh/data/m-lm-mistralai/container-python-packages-nvrx-0.6.0-py312"
        LOCAL_IMAGE_ROOT=""
        ;;
    *)
        echo "Unsupported cluster: ${CLUSTER_NAME}" >&2
        exit 1
        ;;
esac
readonly CLUSTER_TAG PYTHON_PACKAGES LOCAL_IMAGE_ROOT
readonly EPOCH_INDEXES="${OUTPUT_ROOT}/epoch_indexes_vlm_8k_1t_n4_v6_262k_strawman_v06_20260901_phase1_80_phase2_20_smart_${CLUSTER_TAG}_full"
readonly RUNTIME_CONFIG_DIR="${OUTPUT_ROOT}/runtime_configs"
readonly RUNTIME_CONFIG="${RUNTIME_CONFIG_DIR}/strawman_60t_v0.6_20260901_phase1_80_phase2_20_smart_phase1_${CLUSTER_TAG}_full.yaml"
readonly CONFIG="${RUNTIME_CONFIG}"
readonly FINGERPRINT_EXTRA_CONFIG=""
readonly COMPLETION_MARKER="${EPOCH_INDEXES}/.prepack_complete"

[[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
    echo "Worktree HEAD does not match EXPECTED_COMMIT" >&2
    exit 1
}
[[ -z "$(git -C "${REPO_DIR}" status --short)" ]] || {
    echo "Worktree is not clean" >&2
    exit 1
}
((PROCESSES_PER_WORKER * (IMAGE_TRANSFORM_WORKERS + 1) < SLURM_CPUS_PER_TASK)) || {
    echo "Packing processes plus image-transform workers exceed the CPU allocation" >&2
    exit 1
}
for path in "${REPO_DIR}" "${SOURCE_CONFIG}" "${PRETOKENIZED_DATA}" \
    "${IMAGE_PATH}" "${PYTHON_PACKAGES}" "${S3_NAMESPACE_CONFIG}"; do
    [[ -e "${path}" ]] || { echo "Required path does not exist: ${path}" >&2; exit 1; }
done

if [[ -n "${LOCAL_IMAGE_ROOT}" ]]; then
    [[ -d "${LOCAL_IMAGE_ROOT}" ]] || {
        echo "Local image root does not exist: ${LOCAL_IMAGE_ROOT}" >&2
        exit 1
    }
    export MISTRAL_LOCAL_IMAGE_ROOT="${LOCAL_IMAGE_ROOT}"
else
    [[ -f /home/sasatheesh/.aws/config ]] || {
        echo "AWS config does not exist: /home/sasatheesh/.aws/config" >&2
        exit 1
    }
    export AWS_PROFILE=nvidia-sandia-cmh
    export MISTRAL_S3_ENDPOINT_URL=https://s3.us-east-2.amazonaws.com
fi
export MISTRAL_S3_NAMESPACE_CONFIG="${S3_NAMESPACE_CONFIG}"
export PYTHONPATH="${REPO_DIR}:${PYTHON_PACKAGES}"

mkdir -p "${EPOCH_INDEXES}" "${RUNTIME_CONFIG_DIR}"
rm -f "${COMPLETION_MARKER}"

awk \
    -v cache_path="${EPOCH_INDEXES}" \
    '/^epoch_index_cache_path:/ { $0 = "epoch_index_cache_path: " cache_path }
     { print }' \
    "${SOURCE_CONFIG}" > "${RUNTIME_CONFIG}.tmp"
mv "${RUNTIME_CONFIG}.tmp" "${RUNTIME_CONFIG}"

common_args=(
    --config "${CONFIG}"
    --epoch-index-cache-path "${EPOCH_INDEXES}"
    --num_total_tokens "${NUM_TOTAL_TOKENS}"
    --image-transform-workers "${IMAGE_TRANSFORM_WORKERS}"
)

srun -l \
    --kill-on-bad-exit=1 \
    --mpi=none \
    --ntasks="${NUM_WORKERS}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --container-image="${IMAGE_PATH}" \
    --container-mounts="/scratch:/scratch,/lustre:/lustre,/home:/home" \
    --container-workdir="${REPO_DIR}" \
    /bin/bash -c '
        worker_count="$1"
        process_count="$2"
        shift 2
        exec /usr/bin/python -u -m \
            mistral_fork.mistral.data.pretokenized.processing.prebuild_epoch_indexes_megatron \
            prebuild \
            --worker_id "${SLURM_PROCID}" \
            --num_workers "${worker_count}" \
            --processes-per-worker "${process_count}" \
            "$@"
    ' bash "${NUM_WORKERS}" "${PROCESSES_PER_WORKER}" "${common_args[@]}"

srun -l \
    --kill-on-bad-exit=1 \
    --mpi=none \
    --ntasks=1 \
    --container-image="${IMAGE_PATH}" \
    --container-mounts="/scratch:/scratch,/lustre:/lustre,/home:/home" \
    --container-workdir="${REPO_DIR}" \
    /usr/bin/python -u -m \
        mistral_fork.mistral.data.pretokenized.processing.prebuild_epoch_indexes_megatron \
        check "${common_args[@]}"

{
    sha256sum "${CONFIG}" | awk '{ print $1 }'
    if [[ -n "${FINGERPRINT_EXTRA_CONFIG}" ]]; then
        sha256sum "${FINGERPRINT_EXTRA_CONFIG}" | awk '{ print $1 }'
    fi
    sha256sum "${S3_NAMESPACE_CONFIG}" | awk '{ print $1 }'
    echo "num_total_tokens=${NUM_TOTAL_TOKENS}"
    echo "percentage_data_multiplier=${PERCENTAGE_DATA_MULTIPLIER}"
} | sha256sum | awk '{ print $1 }' > "${COMPLETION_MARKER}"
echo "Wrote completion marker: ${COMPLETION_MARKER}"
