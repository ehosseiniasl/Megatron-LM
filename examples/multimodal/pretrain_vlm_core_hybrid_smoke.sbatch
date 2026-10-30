#!/bin/bash
#SBATCH -A nemotron_n4_pre
#SBATCH -p batch
#SBATCH -t 01:00:00
#SBATCH --mem=0
#SBATCH --nodes=16
#SBATCH --segment=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --dependency=singleton
#SBATCH --job-name=vlmcore_mock_smoke
#SBATCH --output=/home/sasatheesh/data/workspace/output/slurm/%x_%j.sbatch.log

set -euo pipefail
set -a

RANK_MODE=0
if [[ "${1:-}" == "--rank" ]]; then
  RANK_MODE=1
fi

ts() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

echo "STARTUP_MARK sbatch_begin $(ts) job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} nodelist=${SLURM_JOB_NODELIST}"

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}
export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export TORCHINDUCTOR_WORKER_START=${TORCHINDUCTOR_WORKER_START:-fork}
export QUANTIZATION_TYPE_DEBUG=${QUANTIZATION_TYPE_DEBUG:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:${CPATH:-}
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

MEGATRON_ROOT=${MEGATRON_ROOT:-/home/sasatheesh/src/m-lm-main/pre-vlm-core-training-main-mock}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/home/sasatheesh/data/images/m_lm_energon_pre_vlm_07_gb200.sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/lustre:/lustre,/home:/home}
CONTAINER_LABEL=${CONTAINER_LABEL:-gb200_bigmock}
SRUN_MPI_TYPE=${SRUN_MPI_TYPE:-none}
RUN_NAME=${RUN_NAME:-pre_vlm_core_mock_hybrid_${CONTAINER_LABEL}_${SLURM_JOB_NUM_NODES}node_smoke_${SLURM_JOB_ID}}
OUTPUT_DIR=${OUTPUT_DIR:-/home/sasatheesh/data/workspace/output/${RUN_NAME}}
LOGS_DIR="${OUTPUT_DIR}/logs"
TOKENIZER_MODEL=${TOKENIZER_MODEL:-/home/sasatheesh/data/huggingface/hub/models--sanjeevnv--multimodal-pretraining/snapshots/26f81d5db838eb6dee2ff8692db83a2fbc76f3ff}
VISION_ROOT=${VISION_ROOT:-/home/sasatheesh/data/checkpoints/encoders_v2}
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_${RUN_NAME}}

if [[ "${RANK_MODE}" -eq 0 ]]; then
  read -r MASTER_ADDR < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
else
  MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR must be set by the batch launcher}
fi
MASTER_PORT=${MASTER_PORT_OVERRIDE:-$((28500 + SLURM_JOB_ID % 1000))}

DATALOADER_SEQ_LENGTH=${DATALOADER_SEQ_LENGTH:-512}
SEQ_LENGTH=${SEQ_LENGTH:-${DATALOADER_SEQ_LENGTH}}
DECODER_SEQ_LENGTH=${DECODER_SEQ_LENGTH:-${DATALOADER_SEQ_LENGTH}}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-${DECODER_SEQ_LENGTH}}

TP=${TP:-2}
EP=${EP:-8}
ETP=${ETP:-1}
PP=${PP:-1}
MBS=${MBS:-1}
GBS=${GBS:-${SLURM_NTASKS}}
NUM_EXPERTS=${NUM_EXPERTS:-128}
MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-6}
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:-3712}
HYBRID_LAYER_PATTERN=${HYBRID_LAYER_PATTERN:-MEMEM*EMEM*EMEM*EMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEME}
MAMBA_NUM_HEADS=${MAMBA_NUM_HEADS:-64}
MAMBA_HEAD_DIM=${MAMBA_HEAD_DIM:-64}
DECODER_NUM_LAYERS=${DECODER_NUM_LAYERS:-54}
DECODER_HIDDEN_SIZE=${DECODER_HIDDEN_SIZE:-2688}
DECODER_NUM_ATTENTION_HEADS=${DECODER_NUM_ATTENTION_HEADS:-32}
DECODER_NUM_QUERY_GROUPS=${DECODER_NUM_QUERY_GROUPS:-2}
DECODER_FFN_HIDDEN_SIZE=${DECODER_FFN_HIDDEN_SIZE:-1856}
DECODER_KV_CHANNELS=${DECODER_KV_CHANNELS:-128}
VISION_PROJECTION_FFN_HIDDEN_SIZE=${VISION_PROJECTION_FFN_HIDDEN_SIZE:-${DECODER_FFN_HIDDEN_SIZE}}

TRAIN_ITERS=${TRAIN_ITERS:-1}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-${TRAIN_ITERS}}
LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS:-${TRAIN_ITERS}}
LOG_INTERVAL=${LOG_INTERVAL:-1}
EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-55}
DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-60}
NUM_WORKERS=${NUM_WORKERS:-1}
ENABLE_SAVE=${ENABLE_SAVE:-0}
ENABLE_ONE_LOGGER=${ENABLE_ONE_LOGGER:-0}
DISABLE_GLOO_PROCESS_GROUPS=${DISABLE_GLOO_PROCESS_GROUPS:-1}
ENABLE_DISTRIBUTED_OPTIMIZER=${ENABLE_DISTRIBUTED_OPTIMIZER:-1}
ENABLE_OVERLAP_ARGS=${ENABLE_OVERLAP_ARGS:-1}
ENABLE_SEQUENCE_PARALLEL=${ENABLE_SEQUENCE_PARALLEL:-1}
VISION_DYNAMIC_RESOLUTION_MAX_PATCHES=${VISION_DYNAMIC_RESOLUTION_MAX_PATCHES:-}
EXTRA_MEGATRON_ARGS=${EXTRA_MEGATRON_ARGS:-}

mkdir -p "${LOGS_DIR}" /home/sasatheesh/data/workspace/output/slurm

SAVE_ARGS=()
if [[ "${ENABLE_SAVE}" -eq 1 ]]; then
  SAVE_ARGS+=(--save "${OUTPUT_DIR}/checkpoints" --save-interval 1000000 --ckpt-format torch_dist)
  mkdir -p "${OUTPUT_DIR}/checkpoints"
fi

LOGGER_ARGS=()
if [[ "${ENABLE_ONE_LOGGER}" -eq 0 ]]; then
  LOGGER_ARGS+=(--no-one-logger)
fi

GLOO_ARGS=()
if [[ "${DISABLE_GLOO_PROCESS_GROUPS}" -eq 1 ]]; then
  GLOO_ARGS+=(--disable-gloo-process-groups)
fi

OPTIMIZER_ARGS=()
if [[ "${ENABLE_DISTRIBUTED_OPTIMIZER}" -eq 1 ]]; then
  OPTIMIZER_ARGS+=(--use-distributed-optimizer)
fi

OVERLAP_ARGS=()
if [[ "${ENABLE_OVERLAP_ARGS}" -eq 1 ]]; then
  OVERLAP_ARGS+=(--overlap-grad-reduce --overlap-param-gather --ddp-pad-buckets-for-high-nccl-busbw)
fi

SP_ARGS=()
if [[ "${ENABLE_SEQUENCE_PARALLEL}" -eq 1 ]]; then
  SP_ARGS+=(--sequence-parallel)
fi

VISION_MAX_PATCH_ARGS=()
if [[ -n "${VISION_DYNAMIC_RESOLUTION_MAX_PATCHES}" ]]; then
  VISION_MAX_PATCH_ARGS+=(
    --vision-dynamic-resolution-max-patches "${VISION_DYNAMIC_RESOLUTION_MAX_PATCHES}"
  )
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA_MEGATRON_ARGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_MEGATRON_ARGS}"
fi

MEGATRON_ARGS=(
  pretrain_vlm.py
  --transformer-impl transformer_engine
  --mock-data
  --freeze-ViT
  --dataloader-type external
  --vlm-dataloader mock
  --language-model-type nemotron6-moe
  --dataloader-seq-length "${DATALOADER_SEQ_LENGTH}"
  --vision-projection-type mlp
  --vision-projection-ffn-hidden-size "${VISION_PROJECTION_FFN_HIDDEN_SIZE}"
  --vision-model-type post-c-radio-omni
  "${VISION_MAX_PATCH_ARGS[@]}"
  --disable-vision-class-token
  --eod-mask-loss
  --image-tag-type internvl
  --moe-token-dispatcher-type alltoall
  --moe-router-score-function sigmoid
  --moe-grouped-gemm
  --num-experts "${NUM_EXPERTS}"
  --moe-router-topk "${MOE_ROUTER_TOPK}"
  --moe-aux-loss-coeff 1e-4
  --moe-router-topk-scaling-factor 2.5
  --moe-router-enable-expert-bias
  --moe-router-dtype fp32
  --moe-router-load-balancing-type seq_aux_loss
  --moe-shared-expert-intermediate-size "${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE}"
  --enable-experimental
  --attention-backend flash
  --is-hybrid-model
  --mamba-num-heads "${MAMBA_NUM_HEADS}"
  --mamba-head-dim "${MAMBA_HEAD_DIM}"
  --hybrid-layer-pattern "${HYBRID_LAYER_PATTERN}"
  --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
  --tiktoken-pattern v2
  --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}"
  --use-mcore-models
  --untie-embeddings-and-output-weights
  --disable-bias-linear
  --init-method-std 0.0173
  --position-embedding-type none
  --squared-relu
  --num-layers "${DECODER_NUM_LAYERS}"
  --hidden-size "${DECODER_HIDDEN_SIZE}"
  --num-attention-heads "${DECODER_NUM_ATTENTION_HEADS}"
  --group-query-attention
  --num-query-groups "${DECODER_NUM_QUERY_GROUPS}"
  --ffn-hidden-size "${DECODER_FFN_HIDDEN_SIZE}"
  --kv-channels "${DECODER_KV_CHANNELS}"
  --normalization RMSNorm
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --exit-duration-in-mins "${EXIT_DURATION_IN_MINS}"
  --tensor-model-parallel-size "${TP}"
  --expert-model-parallel-size "${EP}"
  --expert-tensor-parallel-size "${ETP}"
  --pipeline-model-parallel-size "${PP}"
  "${SP_ARGS[@]}"
  --seq-length "${SEQ_LENGTH}"
  --decoder-seq-length "${DECODER_SEQ_LENGTH}"
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
  --micro-batch-size "${MBS}"
  --global-batch-size "${GBS}"
  --train-iters "${TRAIN_ITERS}"
  --lr 1.2e-3
  --min-lr 1.2e-5
  --weight-decay 0.1
  --clip-grad 1.0
  --lr-decay-style WSD
  --lr-warmup-iters 0
  --lr-decay-iters "${LR_DECAY_ITERS}"
  --lr-wsd-decay-style minus_sqrt
  --lr-wsd-decay-iters "${LR_WSD_DECAY_ITERS}"
  --log-interval "${LOG_INTERVAL}"
  --eval-iters 0
  --eval-interval 999999999
  --tokenizer-type MultimodalTokenizer
  --tokenizer-model "${TOKENIZER_MODEL}"
  --tokenizer-prompt-format nemotron-4
  "${SAVE_ARGS[@]}"
  --bf16
  --adam-beta1 0.9
  --adam-beta2 0.95
  "${OPTIMIZER_ARGS[@]}"
  "${OVERLAP_ARGS[@]}"
  --num-workers "${NUM_WORKERS}"
  --tensorboard-dir "${OUTPUT_DIR}/tensorboard"
  "${LOGGER_ARGS[@]}"
  "${GLOO_ARGS[@]}"
  --no-check-for-nan-in-loss-and-grad
  --manual-gc
  --manual-gc-interval 10
  --log-progress
  --log-params-norm
  --log-throughput
  --vision-encoder-checkpoint-root "${VISION_ROOT}"
  "${EXTRA_ARGS[@]}"
)

if [[ "${RANK_MODE}" -eq 1 ]]; then
  export RANK="${SLURM_PROCID}"
  export WORLD_SIZE="${SLURM_NTASKS}"
  export LOCAL_RANK="${SLURM_LOCALID}"
  export LOCAL_WORLD_SIZE=4
  export NODE_RANK="${SLURM_NODEID}"
  export TRITON_HOME="${TRITON_CACHE_DIR}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
  echo "STARTUP_MARK container_task_begin $(ts) rank=${RANK} world=${WORLD_SIZE} node=${SLURMD_NODENAME} local_rank=${LOCAL_RANK} image=${CONTAINER_IMAGE}"
  echo "STARTUP_MARK python_exec $(ts) rank=${RANK} master=${MASTER_ADDR}:${MASTER_PORT}"
  cd "${MEGATRON_ROOT}"
  exec python -u "${MEGATRON_ARGS[@]}"
fi

SRUN_MPI_ARGS=()
if [[ "${SRUN_MPI_TYPE}" != "default" ]]; then
  SRUN_MPI_ARGS+=(--mpi="${SRUN_MPI_TYPE}")
fi

echo "STARTUP_MARK before_srun $(ts) job=${SLURM_JOB_ID} master=${MASTER_ADDR}:${MASTER_PORT} image=${CONTAINER_IMAGE} output=${OUTPUT_DIR} tp=${TP} ep=${EP} layers=${DECODER_NUM_LAYERS} hidden=${DECODER_HIDDEN_SIZE} vision_projection=mlp vision_projection_ffn=${VISION_PROJECTION_FFN_HIDDEN_SIZE} gbs=${GBS}"

srun -l --verbose \
  --ntasks="${SLURM_NTASKS}" \
  --ntasks-per-node=4 \
  --export=ALL \
  "${SRUN_MPI_ARGS[@]}" \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
  --container-workdir "${MEGATRON_ROOT}" \
  --output="${LOGS_DIR}/%x_%j.log" \
  bash /home/sasatheesh/src/m-lm-main/pre-vlm-core-training-main-mock/examples/multimodal/pretrain_vlm_core_hybrid_smoke.sbatch --rank
