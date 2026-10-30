#!/usr/bin/env bash

# Run one Thunderdome-compatible lm-eval task against the leaderboard checkpoint.
# This is the SLURM array worker; submit it with launch_mistral_pretraining_leaderboard.sh.

set -euo pipefail

TASKS=(
  hellaswag_mistral
  commonsense_qa_mistral
  openbookqa_mistral
  piqa_mistral
  siqa_mistral
  race_high_mistral
  race_middle_mistral
  naturalqs_5shot_mistral
  triviaqa_5shot_mistral
  truthfulqa_mistral
  winogrande_mistral
  gpqa_diamond_0shot_mistral
  gpqa_diamond_5shot_mistral
  drop_3shot_v2_mistral
  'gsm8k_5shot_maj@1_v3_mistral'
  simpleqa_5shot_mistral
  popqa_5shot_mistral
  new_mmlu_v2_mistral
  mmlu_redux_mistral
  super_gpqa_cot_5shot_mistral
)

if [[ "${1:-}" == "--task-count" ]]; then
  printf '%s\n' "${#TASKS[@]}"
  exit 0
fi

# torch.distributed.run re-enters this file once per server rank. Import the
# checkpoint's model-construction hook before Megatron builds the model.
if [[ "${1:-}" == "__server_rank" ]]; then
  shift
  megatron_src_dir="$1"
  tekken_compat_path="$2"
  megatron_args_json="$3"
  shift 3
  cd "${megatron_src_dir}"
  if ! python -c 'import causal_conv1d, hypercorn, mamba_ssm, quart' 2>/dev/null; then
    for install_attempt in 1 2 3; do
      if python -m pip install --root-user-action=ignore --no-build-isolation \
        'causal-conv1d~=1.5' 'hypercorn==0.18.0' 'mamba-ssm~=2.2' 'quart==0.20.0'; then
        break
      fi
      if (( install_attempt == 3 )); then
        exit 1
      fi
      echo "Server dependency install failed (attempt ${install_attempt}/3); retrying..." >&2
      sleep 5
    done
  fi
  exec python -u -c '
import importlib.util
import runpy

compat_spec = importlib.util.spec_from_file_location("mistral_tekken_compat", __import__("sys").argv.pop(1))
compat_module = importlib.util.module_from_spec(compat_spec)
compat_spec.loader.exec_module(compat_module)
compat_module.install_mistral_tekken_compat(__import__("sys").argv.pop(1))
runpy.run_path("tools/run_dynamic_text_generation_server.py", run_name="__main__")
' "${tekken_compat_path}" "${megatron_args_json}" "$@"
fi

# Run the lm-eval client inside the same ARM Megatron container as the server.
if [[ "${1:-}" == "__lm_eval_client" ]]; then
  shift
  lme_src_dir="$1"
  task_name="$2"
  model_name="$3"
  endpoint="$4"
  output_path="$5"
  shift 5

  cd "${lme_src_dir}"
  export PYTHONPATH="${lme_src_dir}${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ "${LME_INSTALL:-auto}" == "always" ]] || {
    [[ "${LME_INSTALL:-auto}" == "auto" ]] &&
      ! python -c 'import accelerate, datasets, evaluate, openai, transformers' 2>/dev/null
  }; then
    for install_attempt in 1 2 3; do
      if python -m pip install --root-user-action=ignore --no-build-isolation \
        -e "${lme_src_dir}"; then
        break
      fi
      if (( install_attempt == 3 )); then
        exit 1
      fi
      echo "lm-eval dependency install failed (attempt ${install_attempt}/3); retrying..." >&2
      sleep 5
    done
  fi
  exec python -u -m lm_eval \
    --model notok-completions \
    --model_args "base_url=${endpoint},model=${model_name},max_concurrent_per_client=1,add_bos=true" \
    --tasks "${task_name}" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}" \
    "$@"
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MEGATRON_SRC="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

: "${RUN_PATH:?Set RUN_PATH to the Megatron run directory}"
: "${CHECKPOINT_ITER:?Set CHECKPOINT_ITER to the zero-padded checkpoint iteration}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_PATH}/checkpoints}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/iter_${CHECKPOINT_ITER}"
MODEL_NAME="${MODEL_NAME:-$(basename "${RUN_PATH}")_iter_${CHECKPOINT_ITER}}"

LME_SRC_DIR="${LME_SRC_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
MLM_SRC_DIR="${MLM_SRC_DIR:-${DEFAULT_MEGATRON_SRC}}"
MEGATRON_ARGS_JSON="${MEGATRON_ARGS_JSON:-${RUN_PATH}/megatron_args.json}"
TOKENIZER="${TOKENIZER:-/mnt/misc/tokenizers/lobstenizer.tekken.json}"
IMAGE="${IMAGE:-/mnt/misc/containers/megatron_muon_blc.sqsh}"

TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${TASK_INDEX}" || ! "${TASK_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: TASK_INDEX or SLURM_ARRAY_TASK_ID must be an integer." >&2
  exit 2
fi
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#TASKS[@]} )); then
  echo "ERROR: task index ${TASK_INDEX} is outside 0-$(( ${#TASKS[@]} - 1 ))." >&2
  exit 2
fi
TASK_NAME="${TASKS[$TASK_INDEX]}"

for required_dir in "${CHECKPOINT_PATH}" "${LME_SRC_DIR}/lm_eval" "${MLM_SRC_DIR}/megatron"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "ERROR: required directory not found: ${required_dir}" >&2
    exit 2
  fi
done
for required_file in \
  "${TOKENIZER}" \
  "${IMAGE}" \
  "${MEGATRON_ARGS_JSON}" \
  "${MLM_SRC_DIR}/tools/run_dynamic_text_generation_server.py"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file not found: ${required_file}" >&2
    exit 2
  fi
done
if [[ ! -f "${CHECKPOINT_PATH}/.metadata" && ! -f "${CHECKPOINT_PATH}/common.pt" ]]; then
  echo "ERROR: incomplete checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

RESULT_ROOT="${RESULT_ROOT:-${RUN_PATH}/evals/lm_eval_mistral_pretraining_${CHECKPOINT_ITER}}"
TASK_OUTPUT_DIR="${RESULT_ROOT}/results/${TASK_NAME}"
LOG_DIR="${RESULT_ROOT}/logs/${TASK_NAME}"
mkdir -p "${TASK_OUTPUT_DIR}" "${LOG_DIR}"

source "${LME_SRC_DIR}/nvbench/sbatch_utils.sh"
nvbench_derive_ports 48500 1500
COORDINATOR_PORT=$((HTTP_PORT + 1500))

DEFAULT_CONTAINER_MOUNTS=(
  /home:/home
  /lustre01/tenant1:/lustre01/tenant1
  /lustre01/tenant1/runs:/mnt/runs
  /lustre01/tenant1/misc:/mnt/misc
  /lustre01/tenant1/home_blc2:/lustre01/tenant1/home_blc2
  /run/slurm:/run/slurm
  /etc/slurm:/etc/slurm
  /var/spool/slurmd:/var/spool/slurmd
)
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$(IFS=,; echo "${DEFAULT_CONTAINER_MOUNTS[*]}")}"

export HF_HOME="${HF_HOME:-${RUN_PATH}/evals/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RESULT_ROOT}/triton_cache/${TASK_NAME}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUN_PATH}/evals/pip_cache}"
PIP_WHEELHOUSE_DIR="${PIP_WHEELHOUSE_DIR:-${PIP_CACHE_DIR}/wheelhouse}"
export PIP_FIND_LINKS="${PIP_FIND_LINKS:-${PIP_WHEELHOUSE_DIR}}"
mkdir -p \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${TRITON_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${PIP_WHEELHOUSE_DIR}"

ITER_NUMBER="$((10#${CHECKPOINT_ITER}))"
SERVER_ARGS=(
  --load "${CHECKPOINT_DIR}"
  --ckpt-step "${ITER_NUMBER}"
  --use-checkpoint-args
  --no-load-optim
  --no-load-rng
  --micro-batch-size 1
  --global-batch-size 1
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 1
  --context-parallel-size 1
  --model-provider hybrid
  --tokenizer-type TikTokenizer
  --tokenizer-model "${TOKENIZER}"
  --no-use-tokenizer-model-from-checkpoint-args
  --bf16
  --attention-backend flash
  --moe-router-dtype fp32
  --moe-token-dispatcher-type alltoall
  --te-rng-tracker
  --inference-rng-tracker
  --dist-ckpt-strictness log_unexpected
  --cuda-graph-impl none
  --inference-cuda-graph-scope none
  --inference-max-seq-length "${MAX_SEQ_LEN:-8192}"
  --inference-dynamic-batching-buffer-size-gb "${INFERENCE_BUFFER_GB:-5}"
  --inference-dynamic-batching-max-requests "${MAX_SERVER_REQUESTS:-4}"
  --inference-coordinator-port "${COORDINATOR_PORT}"
  --port "${HTTP_PORT}"
)
if [[ -n "${SERVER_EXTRA_ARGS:-}" ]]; then
  read -r -a parsed_server_extra_args <<< "${SERVER_EXTRA_ARGS}"
  SERVER_ARGS+=("${parsed_server_extra_args[@]}")
fi

echo "Task $((TASK_INDEX + 1))/${#TASKS[@]}: ${TASK_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Megatron source: ${MLM_SRC_DIR}"
echo "Harness source: ${LME_SRC_DIR}"
echo "Results: ${TASK_OUTPUT_DIR}"

SERVER_STEP_PID=""
cleanup() {
  if [[ -n "${SERVER_STEP_PID}" ]] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    kill "${SERVER_STEP_PID}" 2>/dev/null || true
    wait "${SERVER_STEP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

srun --nodes 1 --ntasks-per-node 1 \
  --output "${LOG_DIR}/server_%A_%a.log" \
  --error "${LOG_DIR}/server_%A_%a.log" \
  --no-container-entrypoint \
  --container-image "${IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
  --export="ALL,MASTER_ADDR=${MASTER_ADDR},MASTER_PORT=${MASTER_PORT},HTTP_PORT=${HTTP_PORT},COORDINATOR_PORT=${COORDINATOR_PORT}" \
  python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node 1 \
    --node_rank 0 \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    --no_python \
    "${SCRIPT_PATH}" __server_rank "${MLM_SRC_DIR}" \
    "${LME_SRC_DIR}/nvbench/mistral_tekken_compat.py" \
    "${MEGATRON_ARGS_JSON}" "${SERVER_ARGS[@]}" &
SERVER_STEP_PID=$!
NVBENCH_SERVER_STEP_PID="${SERVER_STEP_PID}"
export NVBENCH_SERVER_STEP_PID

nvbench_wait_for_server "http://${MASTER_ADDR}:${HTTP_PORT}" "${SERVER_STARTUP_TIMEOUT_MINUTES:-60}"

srun --overlap --nodes 1 --ntasks 1 \
  --output "${LOG_DIR}/client_%A_%a.log" \
  --error "${LOG_DIR}/client_%A_%a.log" \
  --no-container-entrypoint \
  --container-image "${IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
  --export="ALL,HF_HOME=${HF_HOME},HF_DATASETS_CACHE=${HF_DATASETS_CACHE},PYTHONNOUSERSITE=1" \
  "${SCRIPT_PATH}" __lm_eval_client \
    "${LME_SRC_DIR}" \
    "${TASK_NAME}" \
    "${MODEL_NAME}" \
    "http://${MASTER_ADDR}:${HTTP_PORT}" \
    "${TASK_OUTPUT_DIR}"
