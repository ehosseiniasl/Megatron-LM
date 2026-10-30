#!/usr/bin/env bash
set -euo pipefail

USER_ROOT=/lustre/fsw/portfolios/nemotron/users/thib.ma
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
CONFIG_DIR=${SCRIPT_DIR}
DATA_ROOT=${USER_ROOT}/data/debug_pr214_document_filters
PRETOK_ROOT=${PRETOK_ROOT:-${DATA_ROOT}/pretokenized_DEBUG}
LOG_DIR=${LOG_DIR:-${PRETOK_ROOT}/submit_logs}
VENV_BIN=${USER_ROOT}/code/nv-mistralai-megatron/.venv-pretok-launch/bin
HELPER=${REPO}/mistral_fork/mistral/data/pretokenized/processing/run_pretok.sh

mkdir -p "${PRETOK_ROOT}" "${LOG_DIR}"

launch() {
  local name=$1
  local config=$2
  env \
    PATH="${VENV_BIN}:${PATH}" \
    PRETOKENIZED_DATA_DIR="${PRETOK_ROOT}" \
    WORKDIR="${REPO}" \
    NUM_SHARDS=1 \
    MAX_PARALLEL_SHARDS=1 \
    CPUS_PER_SHARD=8 \
    SLURM_TIME=00:30:00 \
    JOB_NAME="pr214_debug_${name}" \
    bash "${HELPER}" "${config}" pretok >"${LOG_DIR}/submit_${name}.log" 2>&1
}

launch control "${CONFIG_DIR}/pr214_debug_wikibooks_control.yaml" &
control_pid=$!
# The helper names its submission directory with one-second timestamp precision.
sleep 2
launch rare "${CONFIG_DIR}/pr214_debug_wikibooks_rare.yaml" &
rare_pid=$!

wait "${control_pid}"
wait "${rare_pid}"

cat "${LOG_DIR}/submit_control.log"
cat "${LOG_DIR}/submit_rare.log"
