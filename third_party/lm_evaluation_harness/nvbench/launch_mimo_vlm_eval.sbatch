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
#SBATCH --array=0-22
#SBATCH --job-name=n4v6-eval
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

# N4V6 MIMO VLM checkpoint evaluation with the repository-local lm-eval harness.
# Defaults use the tuned iteration-80000 text/VLM benchmark profile: TP2/EP2,
# FlashAttention, an 8 GiB KV buffer, and 256 client/server requests in flight
# in batches of 32. CUDA graphs are disabled because N4 shortcut-MoE checkpoints
# require eager inference. Larger checkpoints can increase the allocation and
# expert parallelism without changing the evaluation contract, for example:
#
# sbatch --export=ALL,RUN_NAME=n4v6-a3b,CHECKPOINT_DIR=/path/to/checkpoints,CHECKPOINT_STEP=10000 \
#   third_party/lm_evaluation_harness/nvbench/launch_mimo_vlm_eval.sbatch
# sbatch --nodes=8 --export=ALL,RUN_NAME=n4v6-super,CHECKPOINT_DIR=/path/to/checkpoints,\
# CHECKPOINT_STEP=10000,EXPERT_MODEL_PARALLEL_SIZE=16,INFERENCE_MOE_TOKEN_DISPATCHER_TYPE=nccl \
#   third_party/lm_evaluation_harness/nvbench/launch_mimo_vlm_eval.sbatch

set -euo pipefail

RUN_NAME=${RUN_NAME:?RUN_NAME is required}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}
CHECKPOINT_STEP=${CHECKPOINT_STEP:?CHECKPOINT_STEP is required}
TASK_INDEX=${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}

REPO_DIR=${SLURM_SUBMIT_DIR:?Submit from the repository root}
LME_DIR=${REPO_DIR}/third_party/lm_evaluation_harness
LME_ENV=${LME_ENV:-${LME_DIR}/env}
EVAL_CONFIG=nemo4-fast.yml
HF_HOME=${HF_HOME:-/home/sasatheesh/data/huggingface}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-nvidia/n4-v6-262k}
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Warning: set HF_TOKEN if ${TOKENIZER_MODEL} is not already accessible or cached." >&2
fi

IMAGE=${IMAGE:-/home/sasatheesh/data/images/m_lm_energon_cmh_gb300_pre_vlm_05_pre_energon_07_efa_pt2602_nvrx_0_6_0_v2.sqsh}
RUN_ROOT=${RUN_ROOT:-/home/sasatheesh/data/eval_runs/n4v6}
EVAL_DIR=${RUN_ROOT}/results
RUN_DIR=${RUN_ROOT}/job_${SLURM_ARRAY_JOB_ID}_${TASK_INDEX}
FLASHINFER_CACHE=${RUN_ROOT}/cache/flashinfer
TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
GPUS_PER_NODE=4
NNODES=${SLURM_NNODES:?Slurm allocation size is required}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-2}
INFERENCE_MOE_TOKEN_DISPATCHER_TYPE=${INFERENCE_MOE_TOKEN_DISPATCHER_TYPE:-nvls}
INFERENCE_DYNAMIC_BATCHING_MAX_TOKENS=${INFERENCE_DYNAMIC_BATCHING_MAX_TOKENS:-65536}
[[ "${TENSOR_MODEL_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]
[[ "${EXPERT_MODEL_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]
[[ "${INFERENCE_DYNAMIC_BATCHING_MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]]
[[ "${INFERENCE_MOE_TOKEN_DISPATCHER_TYPE}" == nvls || "${INFERENCE_MOE_TOKEN_DISPATCHER_TYPE}" == nccl ]]
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
(( WORLD_SIZE % TENSOR_MODEL_PARALLEL_SIZE == 0 ))
DATA_PARALLEL_SIZE=$((WORLD_SIZE / TENSOR_MODEL_PARALLEL_SIZE))
(( DATA_PARALLEL_SIZE % EXPERT_MODEL_PARALLEL_SIZE == 0 ))

CHECKPOINT_ITER=$(printf 'iter_%07d' "${CHECKPOINT_STEP}")
MODEL_NAME=${RUN_NAME}_${CHECKPOINT_ITER}

[[ -f "${CHECKPOINT_DIR}/${CHECKPOINT_ITER}/metadata.json" ]]
[[ -f "${CHECKPOINT_DIR}/${CHECKPOINT_ITER}/.metadata" ]]

read -r MASTER_ADDR < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
MASTER_PORT=$((10000 + SLURM_JOB_ID % 10000))
HTTP_PORT=$((20000 + SLURM_JOB_ID % 10000))
COORDINATOR_PORT=$((30000 + SLURM_JOB_ID % 10000))
export MASTER_ADDR MASTER_PORT HTTP_PORT COORDINATOR_PORT HF_HOME GPUS_PER_NODE
export NNODES CHECKPOINT_DIR CHECKPOINT_STEP TOKENIZER_MODEL
export TENSOR_MODEL_PARALLEL_SIZE EXPERT_MODEL_PARALLEL_SIZE INFERENCE_MOE_TOKEN_DISPATCHER_TYPE
export INFERENCE_DYNAMIC_BATCHING_MAX_TOKENS

mkdir -p "${RUN_DIR}" "${EVAL_DIR}" "${FLASHINFER_CACHE}" "${TRITON_CACHE_DIR}"

echo "Commit: $(git -C "${REPO_DIR}" rev-parse HEAD)"
echo "Checkpoint: ${CHECKPOINT_DIR}/${CHECKPOINT_ITER}"
echo "Evaluation: ${EVAL_CONFIG} task ${TASK_INDEX}"
echo "Resources: ${NNODES} node(s), ${WORLD_SIZE} GPUs, TP${TENSOR_MODEL_PARALLEL_SIZE}/EP${EXPERT_MODEL_PARALLEL_SIZE}"

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
CONTAINER_MOUNTS=${REPO_DIR}:${REPO_DIR},${CHECKPOINT_DIR}:${CHECKPOINT_DIR},${RUN_ROOT}:${RUN_ROOT},${HF_HOME}:${HF_HOME},${FLASHINFER_CACHE}:/root/.cache/flashinfer

srun \
    --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
    --gpus-per-task="${GPUS_PER_NODE}" --kill-on-bad-exit=1 --mpi=none \
    --output="${RUN_DIR}/server_%t.out" --error="${RUN_DIR}/server_%t.err" \
    --container-image="${IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir="${REPO_DIR}" \
    /bin/bash -c '/usr/bin/python -u -m torch.distributed.run \
        --nnodes="${NNODES}" --nproc-per-node="${GPUS_PER_NODE}" \
        --node-rank="${SLURM_NODEID}" --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        tools/run_megatron_checkpoint_eval_server.py \
        --load "${CHECKPOINT_DIR}" --ckpt-step "${CHECKPOINT_STEP}" --ckpt-format torch_dist \
        --use-checkpoint-args --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "${TOKENIZER_MODEL}" --no-use-tokenizer-model-from-checkpoint-args \
        --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}" \
        --expert-tensor-parallel-size 1 --moe-router-dtype fp32 \
        --moe-token-dispatcher-type alltoall \
        --inference-moe-token-dispatcher-type "${INFERENCE_MOE_TOKEN_DISPATCHER_TYPE}" \
        --inference-allow-gtp-padding \
        --cuda-graph-impl none \
        --inference-max-seq-length 8192 \
        --inference-dynamic-batching-max-requests 256 \
        --inference-dynamic-batching-max-tokens "${INFERENCE_DYNAMIC_BATCHING_MAX_TOKENS}" \
        --inference-dynamic-batching-buffer-size-gb 8 --attention-backend flash \
        --distributed-timeout-minutes 10 --port "${HTTP_PORT}" \
        --inference-coordinator-port "${COORDINATOR_PORT}" \
        --inference-text-gen-server-logging --dist-ckpt-strictness log_unexpected' &
SERVER_PID=$!

deadline=$((SECONDS + 1200))
until timeout 3 bash -c "exec 3<>/dev/tcp/${MASTER_ADDR}/${HTTP_PORT}" 2>/dev/null; do
    kill -0 "${SERVER_PID}" 2>/dev/null || { wait "${SERVER_PID}"; exit $?; }
    (( SECONDS < deadline )) || { echo "Server startup timed out" >&2; exit 1; }
    sleep 10
done

export PATH=${LME_ENV}/bin:/usr/bin:/bin
export PYTHONPATH=${LME_DIR}
export HF_HUB_CACHE=${HF_HOME}/hub HF_DATASETS_CACHE=${HF_HOME}/datasets
export NO_PROXY=${MASTER_ADDR},localhost,127.0.0.1,.cm.cluster
export no_proxy=${NO_PROXY}
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "${LME_DIR}"
"${LME_ENV}/bin/python" nvbench/launch_evals.py \
    --config "nvbench/${EVAL_CONFIG}" --results_dir "${EVAL_DIR}" \
    --batch-size 32 \
    --endpoint megatron_dyn_c256_b32 --model_name "${MODEL_NAME}" \
    --ip "${MASTER_ADDR}" --port "${HTTP_PORT}" --tokenizer "${TOKENIZER_MODEL}" \
    -t "${TASK_INDEX}" --no-log-samples >"${RUN_DIR}/client.log" 2>&1

echo "Evaluation complete: ${MODEL_NAME}, task ${TASK_INDEX}"
