#!/usr/bin/env bash

# Submit every Thunderdome-compatible lm-eval task as a SLURM array.
# This script only submits jobs; the worker starts the model server and lm-eval.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LME_SRC_DIR="${LME_SRC_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
WORKER="${WORKER:-${SCRIPT_DIR}/run_mistral_pretraining_leaderboard.sh}"
DEFAULT_MEGATRON_SRC="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

: "${RUN_PATH:?Set RUN_PATH to the Megatron run directory}"
: "${CHECKPOINT_ITER:?Set CHECKPOINT_ITER to the zero-padded checkpoint iteration}"
MLM_SRC_DIR="${MLM_SRC_DIR:-${DEFAULT_MEGATRON_SRC}}"
RESULT_ROOT="${RESULT_ROOT:-${RUN_PATH}/evals/lm_eval_mistral_pretraining_${CHECKPOINT_ITER}}"

if [[ ! -x "${WORKER}" ]]; then
  echo "ERROR: worker is missing or not executable: ${WORKER}" >&2
  exit 2
fi
if [[ ! -d "${RUN_PATH}/checkpoints/iter_${CHECKPOINT_ITER}" ]]; then
  echo "ERROR: checkpoint not found: ${RUN_PATH}/checkpoints/iter_${CHECKPOINT_ITER}" >&2
  exit 2
fi

TASK_COUNT="$("${WORKER}" --task-count)"
ARRAY_END=$((TASK_COUNT - 1))
ARRAY_SPEC="${ARRAY_SPEC:-0-${ARRAY_END}}"
mkdir -p "${RESULT_ROOT}/slurm"

export RUN_PATH CHECKPOINT_ITER RESULT_ROOT LME_SRC_DIR MLM_SRC_DIR

SBATCH_CMD=(
  sbatch
  --job-name mistral-pretraining
  --account "${ACCOUNT:-pretraining_text}"
  --partition "${PARTITION:-gb200}"
  --qos "${QOS:-research}"
  --nodes 1
  --ntasks-per-node 1
  --gpus-per-node "${GPUS_PER_NODE:-1}"
  --cpus-per-task "${CPUS_PER_TASK:-36}"
  --mem "${MEMORY:-576G}"
  --time "${TIME_LIMIT:-24:00:00}"
  --array "${ARRAY_SPEC}${ARRAY_THROTTLE:+%${ARRAY_THROTTLE}}"
  --output "${RESULT_ROOT}/slurm/%A_%a.out"
  --error "${RESULT_ROOT}/slurm/%A_%a.err"
  --export ALL
  "${WORKER}"
)

echo "Submitting ${TASK_COUNT} Mistral pretraining tasks"
echo "Checkpoint: ${RUN_PATH}/checkpoints/iter_${CHECKPOINT_ITER}"
echo "Megatron source: ${MLM_SRC_DIR}"
echo "Harness source: ${LME_SRC_DIR}"
echo "Results: ${RESULT_ROOT}"
printf ' %q' "${SBATCH_CMD[@]}"
printf '\n'
"${SBATCH_CMD[@]}"
