#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

if ! id -Gn | tr ' ' '\n' | grep -qx nemotron_coalition_mistral; then
  echo "Run this from a nemotron_coalition_mistral group shell first:"
  echo "  sg nemotron_coalition_mistral -c 'bash -lc \"cd ${REPO_ROOT} && $0 both\"'"
  exit 1
fi

usage() {
  echo "Usage: $0 [pretok|merge PRETOK_JOB_ID|both]"
  echo
  echo "Debug outputs default to:"
  echo "  /lustre/fsw/portfolios/nemotron/users/thib.ma/debug_pretokenized_data/260805_multimodal_n4_v6"
}

MODE=${1:-both}
WORKDIR=${WORKDIR:-${REPO_ROOT}}
PYTHON=${PYTHON:-/lustre/fsw/portfolios/nemotron/users/thib.ma/code/nv-mistralai-megatron/.venv-pretok-launch/bin/python}
CONFIG=${CONFIG:-mistral_scratch/thib/260805_debug_multimodal_n4_v6_pretok/debug_multimodal_n4_v6.yaml}
DEBUG_ROOT=${DEBUG_ROOT:-/lustre/fsw/portfolios/nemotron/users/thib.ma/debug_pretokenized_data/260805_multimodal_n4_v6}
PRETOKENIZED_DATA_DIR=${PRETOKENIZED_DATA_DIR:-${DEBUG_ROOT}/pretokenized_data}
LOGS_DIR=${LOGS_DIR:-${DEBUG_ROOT}/logs}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nemotron/users/szymon.ma/images/nvidia+pytorch+26.04-py3+dependencies+mamba.sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-${HOME}:${HOME},/scratch:/scratch,/lustre:/lustre}
PRETOK_JOB_NAME=${PRETOK_JOB_NAME:-pretok_debug_multimodal_n4_v6_1shard}
MERGE_JOB_NAME=${MERGE_JOB_NAME:-merge_debug_multimodal_n4_v6_1source}

json_field() {
  local field=$1
  python3 -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "${field}"
}

submit_pretok() {
  "$PYTHON" -m mistral_fork.mistral.data.pretokenized.processing.pretokenize distributed_process \
    --config "$CONFIG" \
    --pretokenized-data-dir "$PRETOKENIZED_DATA_DIR" \
    --num-shards 1 \
    --max-parallel-shards 1 \
    --cpus-per-shard 8 \
    --slurm-account nemotron_coalition_mistral \
    --slurm-partition cpu \
    --slurm-qos cpu-free \
    --slurm-time 02:00:00 \
    --logs-dir "$LOGS_DIR" \
    --group nemotron_coalition_mistral \
    --job-name "$PRETOK_JOB_NAME" \
    --workdir "$WORKDIR" \
    --container-image "$CONTAINER_IMAGE" \
    --container-name nemo_pytorch_2604_mamba_user \
    --no-container-remap-root \
    --container-mounts "$CONTAINER_MOUNTS" \
    --container-env HOME,LANG,LANGUAGE,LC_ALL
}

prepare_merge_submission() {
  "$PYTHON" -m mistral_fork.mistral.data.pretokenized.processing.merge distributed_merge \
    --config "$CONFIG" \
    --pretokenized-data-dir "$PRETOKENIZED_DATA_DIR" \
    --num-shards 1 \
    --max-parallel-shards 1 \
    --cpus-per-shard 8 \
    --slurm-account nemotron_coalition_mistral \
    --slurm-partition cpu \
    --slurm-qos cpu-free \
    --slurm-time 02:00:00 \
    --logs-dir "$LOGS_DIR" \
    --group nemotron_coalition_mistral \
    --job-name "$MERGE_JOB_NAME" \
    --workdir "$WORKDIR" \
    --container-image "$CONTAINER_IMAGE" \
    --container-name nemo_pytorch_2604_mamba_user \
    --no-container-remap-root \
    --container-mounts "$CONTAINER_MOUNTS" \
    --container-env HOME,LANG,LANGUAGE,LC_ALL \
    --dry-run
}

submit_merge_after() {
  local pretok_job_id=$1
  local merge_output
  local submission
  merge_output=$(prepare_merge_submission)
  echo "$merge_output"
  submission=$(printf '%s' "$merge_output" | json_field submission_path)
  sbatch --parsable --dependency="afterok:${pretok_job_id}" "$submission"
}

mkdir -p "$PRETOKENIZED_DATA_DIR" "$LOGS_DIR"
cd "$WORKDIR"

case "$MODE" in
  pretok)
    submit_pretok
    ;;
  merge)
    if [[ $# -ne 2 ]]; then
      usage
      exit 1
    fi
    submit_merge_after "$2"
    ;;
  both)
    pretok_output=$(submit_pretok)
    echo "$pretok_output"
    pretok_job_id=$(printf '%s' "$pretok_output" | json_field job_id)
    echo "Submitting merge after pretok job ${pretok_job_id}"
    submit_merge_after "$pretok_job_id"
    ;;
  *)
    usage
    exit 1
    ;;
esac
