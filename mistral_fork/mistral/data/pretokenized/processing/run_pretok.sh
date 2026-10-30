#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)

GROUP_NAME=${GROUP_NAME-nemotron_coalition_mistral}
SLURM_ACCOUNT=${SLURM_ACCOUNT-${GROUP_NAME}}

if [[ -n "${GROUP_NAME}" ]] && ! id -Gn | tr ' ' '\n' | grep -qx "${GROUP_NAME}"; then
  echo "Please run this from a ${GROUP_NAME} shell first:"
  echo "  newgrp ${GROUP_NAME}"
  exit 1
fi

cd "${REPO_ROOT}"

usage() {
  echo "Usage: $0 CONFIG [pretok|merge]"
  echo
  echo "Optional env overrides:"
  echo "  PRETOKENIZED_DATA_DIR WORKDIR CONTAINER_IMAGE CONTAINER_MOUNTS"
  echo "  GROUP_NAME SLURM_ACCOUNT SLURM_PARTITION SLURM_QOS"
  echo "  NUM_SHARDS MAX_PARALLEL_SHARDS CPUS_PER_SHARD SLURM_TIME JOB_NAME"
  echo "  MERGE_NUM_SHARDS MERGE_MAX_PARALLEL_SHARDS MERGE_CPUS_PER_SHARD MERGE_SLURM_TIME MERGE_JOB_NAME"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

CONFIG=$1
mode=${2:-pretok}
CONFIG_STEM=$(basename "${CONFIG}")
CONFIG_STEM=${CONFIG_STEM%.yaml}
CONFIG_STEM=${CONFIG_STEM%.yml}

PRETOKENIZED_DATA_DIR=${PRETOKENIZED_DATA_DIR:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/pretokenized_data}
WORKDIR=${WORKDIR:-${REPO_ROOT}}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nemotron/users/szymon.ma/images/nvidia+pytorch+26.04-py3+dependencies+mamba.sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-${HOME}:${HOME},/scratch:/scratch,/lustre:/lustre}
SLURM_PARTITION=${SLURM_PARTITION:-cpu}
SLURM_QOS=${SLURM_QOS:-cpu-free}
GROUP_ARGS=()
ACCOUNT_ARGS=()
if [[ -n "${GROUP_NAME}" ]]; then
  GROUP_ARGS=(--group "${GROUP_NAME}")
fi
if [[ -n "${SLURM_ACCOUNT}" ]]; then
  ACCOUNT_ARGS=(--slurm-account "${SLURM_ACCOUNT}")
fi

case "${mode}" in
  pretok)
    NUM_SHARDS=${NUM_SHARDS:-64}
    MAX_PARALLEL_SHARDS=${MAX_PARALLEL_SHARDS:-${NUM_SHARDS}}
    CPUS_PER_SHARD=${CPUS_PER_SHARD:-16}
    SLURM_TIME=${SLURM_TIME:-48:00:00}
    JOB_NAME=${JOB_NAME:-pretok_${CONFIG_STEM}_${NUM_SHARDS}shards_${CPUS_PER_SHARD}cpu_free}
    python3 -m mistral_fork.mistral.data.pretokenized.processing.pretokenize distributed_process \
      --config "${CONFIG}" \
      --pretokenized-data-dir "${PRETOKENIZED_DATA_DIR}" \
      --num-shards "${NUM_SHARDS}" \
      --max-parallel-shards "${MAX_PARALLEL_SHARDS}" \
      --cpus-per-shard "${CPUS_PER_SHARD}" \
      "${ACCOUNT_ARGS[@]}" \
      --slurm-partition "${SLURM_PARTITION}" \
      --slurm-qos "${SLURM_QOS}" \
      --slurm-time "${SLURM_TIME}" \
      "${GROUP_ARGS[@]}" \
      --job-name "${JOB_NAME}" \
      --workdir "${WORKDIR}" \
      --container-image "${CONTAINER_IMAGE}" \
      --container-name nemo_pytorch_2604_mamba_user \
      --no-container-remap-root \
      --container-mounts "${CONTAINER_MOUNTS}" \
      --container-env HOME,LANG,LANGUAGE,LC_ALL
    ;;
  merge)
    MERGE_NUM_SHARDS=${MERGE_NUM_SHARDS:-8}
    MERGE_MAX_PARALLEL_SHARDS=${MERGE_MAX_PARALLEL_SHARDS:-${MERGE_NUM_SHARDS}}
    MERGE_CPUS_PER_SHARD=${MERGE_CPUS_PER_SHARD:-32}
    MERGE_SLURM_TIME=${MERGE_SLURM_TIME:-48:00:00}
    MERGE_JOB_NAME=${MERGE_JOB_NAME:-merge_${CONFIG_STEM}_${MERGE_NUM_SHARDS}shards_pool${MERGE_CPUS_PER_SHARD}_cpu_free}
    python3 -m mistral_fork.mistral.data.pretokenized.processing.merge distributed_merge \
      --config "${CONFIG}" \
      --pretokenized-data-dir "${PRETOKENIZED_DATA_DIR}" \
      --num-shards "${MERGE_NUM_SHARDS}" \
      --max-parallel-shards "${MERGE_MAX_PARALLEL_SHARDS}" \
      --cpus-per-shard "${MERGE_CPUS_PER_SHARD}" \
      "${ACCOUNT_ARGS[@]}" \
      --slurm-partition "${SLURM_PARTITION}" \
      --slurm-qos "${SLURM_QOS}" \
      --slurm-time "${MERGE_SLURM_TIME}" \
      "${GROUP_ARGS[@]}" \
      --job-name "${MERGE_JOB_NAME}" \
      --workdir "${WORKDIR}" \
      --container-image "${CONTAINER_IMAGE}" \
      --container-name nemo_pytorch_2604_mamba_user \
      --no-container-remap-root \
      --container-mounts "${CONTAINER_MOUNTS}" \
      --container-env HOME,LANG,LANGUAGE,LC_ALL
    ;;
  *)
    usage
    exit 1
    ;;
esac
