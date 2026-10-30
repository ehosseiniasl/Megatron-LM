#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --array=0-12
#SBATCH --job-name=n4-vlm-eval
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

# The pinned two-shot VLMEvalKit suite is the source of truth for the task list,
# prompt contract, generation settings, and reported metrics.

set -euo pipefail

if [[ -f /home/sasatheesh/.env ]]; then
    set +u
    set -a
    source /home/sasatheesh/.env >/dev/null 2>&1
    set +a
    set -u
fi
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

RUN_NAME=${RUN_NAME:?RUN_NAME is required}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}
CHECKPOINT_STEP=${CHECKPOINT_STEP:?CHECKPOINT_STEP is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
VLM_ENV=${VLM_ENV:?VLM_ENV must name a revision-stamped VLMEvalKit environment}
TASK_INDEX=${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}
NUM_SHOTS=2

MEGATRON_DIR=${MEGATRON_DIR:-${SLURM_SUBMIT_DIR:?Submit from the Megatron repository root}}
MEGATRON_COMMIT=${MEGATRON_COMMIT:?MEGATRON_COMMIT is required}
LME_DIR=${MEGATRON_DIR}/third_party/lm_evaluation_harness
source "${LME_DIR}/nvbench/n4_vlm_eval_pins.sh"

VLM_SOURCE=${VLM_ENV}/src/vlmevalkit
VLM_SUITE=${VLM_SOURCE}/nvbench/vlm_2shot_evals.yml
SERVER_DEPS=${SERVER_DEPS:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/sasatheesh/data/envs/megatron-eval-server-deps/quart-0.20.0_msgpack-1.1.1-py312}
HF_HOME=${HF_HOME:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/sasatheesh/data/huggingface}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-${HF_HOME}/hub/models--nvidia--n4-v6-262k/snapshots/${TOKENIZER_REVISION}}
IMAGE=${IMAGE:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron4/images/nvidia+pytorch+26.07-py3+deps+v7.sqsh}
LMUData=${LMUData:-${OUTPUT_ROOT}/cache/vlmevalkit}
EXPECTED_CHECKPOINT_SHARDS=${EXPECTED_CHECKPOINT_SHARDS:-1600}
FLASHINFER_CACHE=${OUTPUT_ROOT}/cache/flashinfer
TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}

[[ "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "${CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]
[[ "${EXPECTED_CHECKPOINT_SHARDS}" =~ ^[1-9][0-9]*$ ]]
test "$(git -C "${MEGATRON_DIR}" rev-parse HEAD)" = "${MEGATRON_COMMIT}"
test "$(cat "${VLM_ENV}/VLMEVALKIT_GIT_REVISION")" = "${VLMEVALKIT_REVISION}"
test "$(git -C "${VLM_SOURCE}" rev-parse HEAD)" = "${VLMEVALKIT_REVISION}"
test -x "${VLM_ENV}/bin/python"
test -r "${VLM_SUITE}"
test -r "${VLM_SOURCE}/nvbench/render_api_eval_config.py"
test -r "${VLM_SOURCE}/run.py"
test "$(basename "${TOKENIZER_MODEL%/}")" = "${TOKENIZER_REVISION}"
test -r "${TOKENIZER_MODEL}/tokenizer_config.json"
test -r "${IMAGE}"
test -d "${SERVER_DEPS}"

mapfile -t SUITE_VALUES < <(
    "${VLM_ENV}/bin/python" -P - "${VLM_SUITE}" "${TASK_INDEX}" <<'PY'
import sys

import yaml

suite_path, task_index = sys.argv[1], int(sys.argv[2])
with open(suite_path) as suite_file:
    config = yaml.safe_load(suite_file)

suite = config["suite"]
execution = suite["execution"]
tasks = config["tasks"]
assert suite["few_shot"]["num_shots"] == 2
assert len(tasks) == 13
assert not execution["reuse"]
assert execution["require_unique_eval_id"]
if task_index not in range(len(tasks)):
    raise SystemExit(f"task index {task_index} is outside 0:{len(tasks)}")

print(tasks[task_index]["task_name"])
print(execution["api_nproc"])
print(execution["judge"])
print(suite["checkpoint_type"])
PY
)
if (( ${#SUITE_VALUES[@]} != 4 )); then
    echo "Failed to resolve task ${TASK_INDEX} from ${VLM_SUITE}" >&2
    exit 2
fi
TASK_NAME=${SUITE_VALUES[0]}
API_NPROC=${SUITE_VALUES[1]}
JUDGE=${SUITE_VALUES[2]}
CHECKPOINT_TYPE=${SUITE_VALUES[3]}

CHECKPOINT_NAME=$(printf 'iter_%07d' "${CHECKPOINT_STEP}")
CHECKPOINT_PATH=${CHECKPOINT_DIR}/${CHECKPOINT_NAME}
TRACKER_FILE=${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt
test -s "${TRACKER_FILE}"
TRACKER_STEP=$(<"${TRACKER_FILE}")
[[ "${TRACKER_STEP}" =~ ^[0-9]+$ ]]
(( TRACKER_STEP >= CHECKPOINT_STEP ))
test -s "${CHECKPOINT_PATH}/metadata.json"
test -s "${CHECKPOINT_PATH}/.metadata"
SHARD_COUNT=$(find "${CHECKPOINT_PATH}" -maxdepth 1 -type f -name '*.distcp' -printf '.' | wc -c)
if (( SHARD_COUNT != EXPECTED_CHECKPOINT_SHARDS )); then
    echo "Expected ${EXPECTED_CHECKPOINT_SHARDS} checkpoint shards, found ${SHARD_COUNT}" >&2
    exit 2
fi

MODEL_NAME=${RUN_NAME}_${CHECKPOINT_NAME}
RUN_DIR=${OUTPUT_ROOT}/runs/${MODEL_NAME}/shot_${NUM_SHOTS}/${TASK_NAME}/${SLURM_ARRAY_JOB_ID}_${TASK_INDEX}
WORK_DIR=${OUTPUT_ROOT}/results/${MODEL_NAME}/shot_${NUM_SHOTS}
VLMEVAL_EVAL_ID=${MODEL_NAME}-shot${NUM_SHOTS}-${TASK_NAME}-${SLURM_ARRAY_JOB_ID}-${TASK_INDEX}
if [[ -e "${RUN_DIR}" ]]; then
    echo "Refusing to reuse run directory: ${RUN_DIR}" >&2
    exit 2
fi

read -r MASTER_ADDR < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
read -r MASTER_PORT HTTP_PORT COORDINATOR_PORT < <(/usr/bin/python - <<'PY'
import socket

sockets = [socket.socket() for _ in range(3)]
for sock in sockets:
    sock.bind(("", 0))
print(*(sock.getsockname()[1] for sock in sockets))
PY
)
export MASTER_ADDR MASTER_PORT HF_HOME LMUData VLMEVAL_EVAL_ID GPUS_PER_NODE=4

mkdir -p "${RUN_DIR}" "${WORK_DIR}" "${FLASHINFER_CACHE}" "${TRITON_CACHE_DIR}" "${LMUData}"

{
    echo "Cluster: $(hostname)"
    echo "Model: ${MODEL_NAME}"
    echo "Checkpoint: ${CHECKPOINT_PATH} (read-only)"
    echo "Checkpoint tracker: ${TRACKER_STEP}; shards: ${SHARD_COUNT}"
    echo "Dataset: ${TASK_NAME} (${NUM_SHOTS}-shot benchmark)"
    echo "Megatron: ${MEGATRON_COMMIT}"
    echo "VLMEvalKit: ${VLMEVALKIT_REVISION}"
    echo "Tokenizer: ${TOKENIZER_REPOSITORY}@${TOKENIZER_REVISION}"
    echo "Prompt: tokenizer-provided chat template"
    echo "Suite: ${VLM_SUITE}"
    echo "Run identity: ${VLMEVAL_EVAL_ID}"
    echo "Resources: 1 node, 4 GPUs, TP1/EP4"
} | tee "${RUN_DIR}/manifest.log"

SERVER_PID=
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT TERM INT

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 NVTE_FUSED_ATTN=0 WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR
unset CUDA_DEVICE_MAX_CONNECTIONS PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
unset PIXI_PROJECT_MANIFEST PIXI_PROJECT_ROOT PIXI_ENVIRONMENT_NAME
export PYTHONPATH=${SERVER_DEPS}
CONTAINER_MOUNTS=${MEGATRON_DIR}:${MEGATRON_DIR},${CHECKPOINT_DIR}:${CHECKPOINT_DIR},${OUTPUT_ROOT}:${OUTPUT_ROOT},${HF_HOME}:${HF_HOME},${FLASHINFER_CACHE}:/root/.cache/flashinfer,${SERVER_DEPS}:${SERVER_DEPS}

srun \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --gpus-per-task=4 --kill-on-bad-exit=1 --mpi=none \
    --output="${RUN_DIR}/server_%t.out" --error="${RUN_DIR}/server_%t.err" \
    --container-image="${IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir="${MEGATRON_DIR}" \
    /usr/bin/python -u -m torch.distributed.run --nproc-per-node=4 \
        tools/run_megatron_checkpoint_eval_server.py \
        --load "${CHECKPOINT_DIR}" --ckpt-step "${CHECKPOINT_STEP}" --ckpt-format torch_dist \
        --use-checkpoint-args \
        --tokenizer-type HuggingFaceTokenizer --tokenizer-model "${TOKENIZER_MODEL}" \
        --no-use-tokenizer-model-from-checkpoint-args \
        --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 --expert-model-parallel-size 4 \
        --expert-tensor-parallel-size 1 --moe-router-dtype fp32 \
        --moe-token-dispatcher-type alltoall \
        --inference-allow-gtp-padding \
        --vlm-language-transformer-impl inference_optimized \
        --vlm-image-preprocess-preset mistral-crawl-eval \
        --inference-max-seq-length 16384 \
        --inference-dynamic-batching-buffer-size-gb 5 --attention-backend flash \
        --distributed-timeout-minutes 10 --port "${HTTP_PORT}" \
        --inference-coordinator-port "${COORDINATOR_PORT}" \
        --inference-text-gen-server-logging --dist-ckpt-strictness log_unexpected &
SERVER_PID=$!

deadline=$((SECONDS + 1200))
until timeout 3 bash -c "exec 3<>/dev/tcp/${MASTER_ADDR}/${HTTP_PORT}" 2>/dev/null; do
    kill -0 "${SERVER_PID}" 2>/dev/null || { wait "${SERVER_PID}"; exit $?; }
    (( SECONDS < deadline )) || { echo "Server startup timed out" >&2; exit 1; }
    sleep 10
done

export PATH=${VLM_ENV}/bin:/usr/bin:/bin
unset PYTHONPATH
export PYTHONSAFEPATH=1
export OPENAI_API_BASE=http://${MASTER_ADDR}:${HTTP_PORT}/v1/chat/completions
export OPENAI_API_KEY=EMPTY
export NO_PROXY=${MASTER_ADDR},localhost,127.0.0.1,.cm.cluster
export no_proxy=${NO_PROXY}
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

TASK_CONFIG=${RUN_DIR}/config.json
"${VLM_ENV}/bin/python" -P "${VLM_SOURCE}/nvbench/render_api_eval_config.py" \
    --suite "${VLM_SUITE}" --task "${TASK_NAME}" \
    --api-base "${OPENAI_API_BASE}" --output "${TASK_CONFIG}"

"${VLM_ENV}/bin/python" -P "${VLM_SOURCE}/run.py" \
    --config "${TASK_CONFIG}" --work-dir "${WORK_DIR}" --api-nproc "${API_NPROC}" \
    --judge "${JUDGE}" --ckpt-type "${CHECKPOINT_TYPE}" \
    >"${RUN_DIR}/client.log" 2>&1

if grep -Eq 'combination failed|Evaluation failed for model|Failed to obtain answer via API|Failed with all possible conversation turns' "${RUN_DIR}/client.log"; then
    echo "VLMEvalKit skipped a failed model/dataset combination" >&2
    exit 1
fi

RESULT_DIR=${WORK_DIR}/megatron_vlm/${VLMEVAL_EVAL_ID}
test -d "${RESULT_DIR}"
SCORE_COUNT=$(find "${RESULT_DIR}" -maxdepth 1 -type f \( -name '*acc.csv' -o -name '*score.json' \) -size +0c | wc -l)
PREDICTION_COUNT=$(
    find "${RESULT_DIR}" -maxdepth 1 -type f \
        \( -name "megatron_vlm_${TASK_NAME}.xlsx" \
        -o -name "megatron_vlm_${TASK_NAME}.tsv" \
        -o -name "megatron_vlm_${TASK_NAME}.json" \) \
        -size +0c | wc -l
)
if (( SCORE_COUNT == 0 || PREDICTION_COUNT == 0 )); then
    echo "Missing nonempty score or prediction artifacts in ${RESULT_DIR}" >&2
    exit 1
fi

touch "${RUN_DIR}/COMPLETED"
echo "VLM evaluation complete: ${MODEL_NAME}, ${TASK_NAME} (${NUM_SHOTS}-shot)"
