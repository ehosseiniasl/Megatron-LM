#!/bin/bash

# Text-only counterpart to the N4-v6 VLM ladder rung.

#SBATCH -p batch
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=128
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --segment=4
#SBATCH --overcommit
#SBATCH --dependency=singleton
#SBATCH --job-name=a3b_30b_moe_1t_n4_v6_text

set -euo pipefail

unset CUDA_DEVICE_MAX_CONNECTIONS NCCL_NET
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FUSED_ATTN=0
export TORCHINDUCTOR_WORKER_START=fork
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

readonly REPO_DIR="${SLURM_SUBMIT_DIR}"
readonly RUN_NAME="a3b_30b_moe_1t_n4_v6_text"
readonly OUTPUT_ROOT="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/sasatheesh/data/nemo3_nano"
readonly PREPACK_ROOT="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/sasatheesh/data/m-lm-mistralai"
readonly IMAGE_PATH="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron4/images/nvidia+pytorch+26.07-py3+deps+v5.sqsh"
readonly TERMINAL_ITERATION=158875
readonly GLOBAL_BATCH_SIZE=768
readonly MICRO_BATCH_SIZE=3
readonly SAVE_INTERVAL=500
readonly LOG_INTERVAL=100
readonly EXIT_DURATION_MINS=235

readonly CLUSTER_NAME="$(scontrol show config | awk '$1 == "ClusterName" { print $3 }')"
case "${CLUSTER_NAME}" in
    oci-aga-slurm-1)
        CLUSTER_TAG=aga
        PYTHON_PACKAGES="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/pengx/projects/nv-mistralai-megatron/examples/multimodal/migration/running_text_chain/outputs/container-python-packages"
        ;;
    aws-cmh-slurm-1)
        CLUSTER_TAG=cmh
        PYTHON_PACKAGES="/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/sasatheesh/data/m-lm-mistralai/container-python-packages-nvrx-0.6.0-py312"
        ;;
    *)
        echo "Unsupported cluster: ${CLUSTER_NAME}" >&2
        exit 1
        ;;
esac
readonly CLUSTER_TAG PYTHON_PACKAGES
readonly BASE_PRETOKENIZED_CONFIG="${PREPACK_ROOT}/runtime_configs/strawman_60t_v0.6_20260901_phase1_80_phase2_20_smart_phase1_${CLUSTER_TAG}_text_base.yaml"
readonly SOURCE_PRETOKENIZED_CONFIG="${PREPACK_ROOT}/runtime_configs/strawman_60t_v0.6_20260901_phase1_80_phase2_20_smart_phase1_${CLUSTER_TAG}_text_only.yaml"

for path in "${REPO_DIR}" "${IMAGE_PATH}" "${PYTHON_PACKAGES}" \
    "${BASE_PRETOKENIZED_CONFIG}" "${SOURCE_PRETOKENIZED_CONFIG}"; do
    [[ -e "${path}" ]] || { echo "Required path does not exist: ${path}" >&2; exit 1; }
done

readonly RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
readonly LOGS_DIR="${RUN_DIR}/logs"
readonly CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
readonly TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
readonly WANDB_DIR="${RUN_DIR}/wandb"
readonly TRITON_CACHE_ROOT="/tmp/${RUN_NAME}_${SLURM_JOB_ID}_triton"
readonly FLASHINFER_CACHE="${OUTPUT_ROOT}/../m-lm-mistralai/caches/flashinfer_a0b59cc90"
readonly RUN_BASE_CONFIG="${RUN_DIR}/$(basename "${BASE_PRETOKENIZED_CONFIG}")"
readonly PRETOKENIZED_CONFIG="${RUN_DIR}/$(basename "${SOURCE_PRETOKENIZED_CONFIG}")"

mkdir -p \
    "${LOGS_DIR}" \
    "${CHECKPOINT_DIR}" \
    "${TENSORBOARD_DIR}" \
    "${WANDB_DIR}/cache" \
    "${FLASHINFER_CACHE}"
cp -f "${BASE_PRETOKENIZED_CONFIG}" "${RUN_BASE_CONFIG}"
cp -f "${SOURCE_PRETOKENIZED_CONFIG}" "${PRETOKENIZED_CONFIG}"

set +u
set -a
# shellcheck disable=SC1091
source /home/sasatheesh/.env >/dev/null 2>&1
set +a
set -u

read -r MASTER_ADDR < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR MASTER_PORT=$((10000 + SLURM_JOB_ID % 10000))
export PYTHONPATH="${REPO_DIR}:${PYTHON_PACKAGES}"
export WANDB_CACHE_DIR="${WANDB_DIR}/cache"
export WANDB_ENTITY="hwinf_dcm"
export WANDB_PROJECT="vlm-scaling-ladder"
export WANDB_RUN_ID="${RUN_NAME}"
export WANDB_RESUME=allow
export NCCL_NET_SHARED_COMMS=0

if [[ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    LOAD_OPTIONS="--load ${CHECKPOINT_DIR}"
else
    LOAD_OPTIONS=""
fi

options=" \
    --use-mcore-models \
    --hybrid-layer-pattern MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME/*E/*E \
    --spec megatron.core.models.hybrid.hybrid_layer_specs gated_delta_product_stack_spec \
    --hidden-size 2688 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 2 \
    --mamba-num-heads 16 \
    --mamba-num-groups 16 \
    --mamba-head-dim 64 \
    --mamba-state-dim 128 \
    --ffn-hidden-size 1856 \
    --kv-channels 128 \
    --squared-relu \
    --untie-embeddings-and-output-weights \
    --init-method-std 0.0173 \
    --position-embedding-type none \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --normalization RMSNorm \
    \
    --num-experts 512 \
    --moe-latent-size 672 \
    --moe-router-topk 22 \
    --moe-ffn-hidden-size 1856 \
    --moe-shared-expert-intermediate-size 3712 \
    --moe-token-dispatcher-type alltoall \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 5.0 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    \
    --mtp-use-repeated-layer \
    --mtp-loss-scaling-factor 0.1 \
    --calculate-per-token-loss \
    \
    --bf16 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --train-samples $((TERMINAL_ITERATION * GLOBAL_BATCH_SIZE)) \
    --lr-decay-style WSD \
    --lr-decay-samples 122016000 \
    --lr-warmup-samples 1024000 \
    --lr-wsd-decay-style minus_sqrt \
    --lr-wsd-decay-samples 18302208 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 2.16e-4 \
    --min-lr 2.16e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --optimizer muon \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --muon-extra-scale-factor 0.2 \
    --muon-momentum 0.9 \
    --muon-scale-mode spectral \
    --muon-num-ns-steps 16 \
    --muon-coefficient-type polar_express \
    --muon-scalar-optimizer adam \
    --no-use-layer-wise-param-layout \
    --lion-beta1 0.95 \
    --lion-beta2 0.98 \
    --eval-interval 1000 \
    --eval-iters 0 \
    \
    --te-rng-tracker \
    --no-load-rng \
    \
    --dataloader-type external \
    --pretokenized-config ${PRETOKENIZED_CONFIG} \
    --tokenizer-type NullTokenizer \
    --vocab-size 262272 \
    --padded-vocab-size 262272 \
    --null-tokenizer-eod-id 2 \
    --null-tokenizer-pad-id 3 \
    --num-workers 4 \
    --no-create-attention-mask-in-dataloader \
    \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --expert-model-parallel-size 32 \
    --expert-tensor-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --high-priority-stream-groups ep \
    --ddp-num-buckets 8 \
    --attention-backend flash \
    \
    --ckpt-format torch_dist \
    ${LOAD_OPTIONS} \
    --save ${CHECKPOINT_DIR} \
    --save-interval ${SAVE_INTERVAL} \
    --save-retain-interval 2000 \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load \
    --async-save \
    --use-persistent-ckpt-worker \
    --ckpt-assume-constant-structure \
    \
    --log-interval ${LOG_INTERVAL} \
    --log-memory-interval 1000 \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-progress \
    --log-energy \
    --logging-level 20 \
    --timing-log-option minmax \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --wandb-entity ${WANDB_ENTITY} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-exp-name ${RUN_NAME} \
    --wandb-save-dir ${WANDB_DIR} \
    \
    --manual-gc \
    --manual-gc-interval 10 \
    --distributed-timeout-minutes 10 \
    --exit-duration-in-mins ${EXIT_DURATION_MINS} \
    --rerun-mode disabled \
    --disable-gloo-process-groups \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 "

export REPO_DIR TRITON_CACHE_ROOT options
srun -l \
    --kill-on-bad-exit=1 \
    --mpi=none \
    --container-image="${IMAGE_PATH}" \
    --container-mounts="/scratch:/scratch,/lustre:/lustre,/home:/home,${FLASHINFER_CACHE}:/root/.cache/flashinfer" \
    --container-workdir="${REPO_DIR}" \
    --output="${LOGS_DIR}/%x_%j_%t.out" \
    --error="${LOGS_DIR}/%x_%j_%t.err" \
    bash -c '
        export PATH=/usr/bin:${PATH}
        export RANK=${SLURM_PROCID} WORLD_SIZE=${SLURM_NTASKS}
        export LOCAL_RANK=${SLURM_LOCALID} LOCAL_WORLD_SIZE=4 NODE_RANK=${SLURM_NODEID}
        rank_cache="${TRITON_CACHE_ROOT}/rank_${RANK}"
        mkdir -p "${rank_cache}"
        export TRITON_HOME="${rank_cache}" TRITON_CACHE_DIR="${rank_cache}"
        export TORCHINDUCTOR_CACHE_DIR="${rank_cache}"
        cd "${REPO_DIR}"
        set -f
        exec /usr/bin/python -u "${REPO_DIR}/pretrain_hybrid.py" ${options}
    '
