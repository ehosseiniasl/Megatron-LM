#!/bin/bash

# Shared A3B/30B GDP + Muon/Adam N4-v6 VLM training recipe. Submit one of
# the cluster/data-specific sbatch entry points next to this file.

set -euo pipefail

unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
unset PIXI_PROJECT_MANIFEST PIXI_PROJECT_ROOT PIXI_ENVIRONMENT_NAME VIRTUAL_ENV
unset CUDA_DEVICE_MAX_CONNECTIONS NCCL_NET
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FUSED_ATTN=0
export TORCHINDUCTOR_WORKER_START=fork
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

readonly EXPECTED_CLUSTER_NAME="${EXPECTED_CLUSTER_NAME:?set by the sbatch entry point}"
readonly RUN_NAME="${RUN_NAME:?set by the sbatch entry point}"
readonly IMAGE_PATH="${IMAGE_PATH:?set by the sbatch entry point}"
readonly PYTHON_PACKAGES="${PYTHON_PACKAGES:?set by the sbatch entry point}"
readonly NEW_VISION_WEIGHTS="${NEW_VISION_WEIGHTS:?set by the sbatch entry point}"
readonly OLD_VISION_WEIGHTS="${OLD_VISION_WEIGHTS:?set by the sbatch entry point}"
readonly SOURCE_PRETOKENIZED_CONFIG="${SOURCE_PRETOKENIZED_CONFIG:?set by the sbatch entry point}"
readonly PERCENTAGE_DATA="${PERCENTAGE_DATA:?set by the sbatch entry point}"
readonly EPOCH_INDEX_CACHE="${EPOCH_INDEX_CACHE:?set by the sbatch entry point}"
declare -p EXTRA_REQUIRED_PATHS >/dev/null 2>&1 || {
    echo "EXTRA_REQUIRED_PATHS must be set by the sbatch entry point" >&2
    exit 1
}
declare -p RUNTIME_ENVIRONMENT >/dev/null 2>&1 || {
    echo "RUNTIME_ENVIRONMENT must be set by the sbatch entry point" >&2
    exit 1
}
declare -p UNSET_ENVIRONMENT >/dev/null 2>&1 || {
    echo "UNSET_ENVIRONMENT must be set by the sbatch entry point" >&2
    exit 1
}
readonly -a EXTRA_REQUIRED_PATHS
readonly -a RUNTIME_ENVIRONMENT
readonly -a UNSET_ENVIRONMENT
readonly REPO_DIR="${SLURM_SUBMIT_DIR}"
readonly OUTPUT_ROOT="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/sasatheesh/data/nemo3_nano"
readonly VLM_SFT_TOKENIZED_ROOT="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron-v4/sandia/tokenized/n4_v6_262k_ml4_regex/multimodal_v4__lobstral_rare_tokens_n4_v6_262k_v1_nvidia_pds_quarantine_v1/jsonl/n4/nvidia/multimodal-sft/vlm_sft_25b_jsonl_0825"
readonly S3_NAMESPACE_CONFIG="${REPO_DIR}/mistral_adapter/example_configs/n4_coalition_image_namespaces.json"
readonly TERMINAL_ITERATION=47683
readonly GLOBAL_BATCH_SIZE=768
readonly MICRO_BATCH_SIZE=3
readonly SAVE_INTERVAL="${SAVE_INTERVAL_OVERRIDE:-500}"
readonly LOG_INTERVAL=100
readonly EVAL_INTERVAL="${EVAL_INTERVAL_OVERRIDE:-1000}"
readonly PER_SOURCE_VALIDATION_TOP_N="${PER_SOURCE_VALIDATION_TOP_N_OVERRIDE:-0}"
readonly EXIT_DURATION_MINS="${EXIT_DURATION_MINS_OVERRIDE:-235}"

readonly CLUSTER_NAME="$(scontrol show config | awk '$1 == "ClusterName" { print $3 }')"
[[ "${CLUSTER_NAME}" == "${EXPECTED_CLUSTER_NAME}" ]] || {
    echo "This launcher targets ${EXPECTED_CLUSTER_NAME}; got ${CLUSTER_NAME}" >&2
    exit 1
}
readonly VISION_CHECKPOINT="${OLD_VISION_WEIGHTS}"

for path in "${REPO_DIR}" "${IMAGE_PATH}" "${PYTHON_PACKAGES}" \
    "${VLM_SFT_TOKENIZED_ROOT}" \
    "${NEW_VISION_WEIGHTS}" "${OLD_VISION_WEIGHTS}" "${SOURCE_PRETOKENIZED_CONFIG}" \
    "${EPOCH_INDEX_CACHE}" \
    "${S3_NAMESPACE_CONFIG}" /home/sasatheesh/.env "${EXTRA_REQUIRED_PATHS[@]}"; do
    [[ -e "${path}" ]] || { echo "Required path does not exist: ${path}" >&2; exit 1; }
done
[[ "$(grep -c '^- prefix_path:' "${SOURCE_PRETOKENIZED_CONFIG}")" == 2449 ]] || {
    echo "Pretokenized config must contain exactly 2449 no-binhvq sources" >&2
    exit 1
}
! grep -q '/binhvq_news_train/' "${SOURCE_PRETOKENIZED_CONFIG}" || {
    echo "Pretokenized config still contains binhvq_news_train" >&2
    exit 1
}
for prefix in \
    "${VLM_SFT_TOKENIZED_ROOT}/band-0-to-1b/merged_1099511627776b/chunks_00000-01024" \
    "${VLM_SFT_TOKENIZED_ROOT}/band-1b-to-5b/merged_1099511627776b/chunks_00000-01024" \
    "${VLM_SFT_TOKENIZED_ROOT}/band-5b-to-25b/merged_1099511627776b/chunks_00000-02048"; do
    for extension in bin idx metadata; do
        [[ -f "${prefix}.${extension}" ]] || {
            echo "VLM-SFT source does not exist: ${prefix}.${extension}" >&2
            exit 1
        }
    done
    grep -Fq "prefix_path: ${prefix}" "${SOURCE_PRETOKENIZED_CONFIG}" || {
        echo "The runtime blend does not contain the full VLM-SFT source: ${prefix}" >&2
        exit 1
    }
done
for checkpoint in "${NEW_VISION_WEIGHTS}" "${OLD_VISION_WEIGHTS}"; do
    shard="${checkpoint}/__0_0.distcp"
    [[ -f "${shard}" ]] || { echo "Vision checkpoint shard does not exist: ${shard}" >&2; exit 1; }
    [[ "$(stat -c %s "${shard}")" == 3308064120 ]] || {
        echo "Vision checkpoint shard has an unexpected size: ${shard}" >&2
        exit 1
    }
done

readonly RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
readonly LOGS_DIR="${RUN_DIR}/logs"
readonly CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
readonly TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
readonly WANDB_DIR="${RUN_DIR}/wandb"
readonly TRITON_CACHE_ROOT="/tmp/${RUN_NAME}_${SLURM_JOB_ID}_triton"
readonly FLASHINFER_CACHE="${OUTPUT_ROOT}/../m-lm-mistralai/caches/flashinfer_a0b59cc90"
readonly PRETOKENIZED_CONFIG="${RUN_DIR}/pretokenized.yaml"

mkdir -p \
    "${LOGS_DIR}" \
    "${CHECKPOINT_DIR}" \
    "${TENSORBOARD_DIR}" \
    "${WANDB_DIR}/cache" \
    "${FLASHINFER_CACHE}"
awk -v cache_path="${EPOCH_INDEX_CACHE}" -v percentage_data="${PERCENTAGE_DATA}" '
    /^  percentage_data:/ { $0 = "  percentage_data: " percentage_data }
    /^epoch_index_cache_path:/ { $0 = "epoch_index_cache_path: " cache_path }
    { print }
' "${SOURCE_PRETOKENIZED_CONFIG}" > "${PRETOKENIZED_CONFIG}.tmp"
mv "${PRETOKENIZED_CONFIG}.tmp" "${PRETOKENIZED_CONFIG}"

set +u
set -a
# shellcheck disable=SC1091
source /home/sasatheesh/.env >/dev/null 2>&1
set +a
set -u
unset "${UNSET_ENVIRONMENT[@]}"
export "${RUNTIME_ENVIRONMENT[@]}"

read -r MASTER_ADDR < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR MASTER_PORT=$((10000 + SLURM_JOB_ID % 10000))
export MISTRAL_S3_NAMESPACE_CONFIG="${S3_NAMESPACE_CONFIG}"
export PYTHONPATH="${REPO_DIR}:${PYTHON_PACKAGES}"
export WANDB_CACHE_DIR="${WANDB_DIR}/cache"
export WANDB_ENTITY="hwinf_dcm"
export WANDB_PROJECT="vlm-scaling-ladder"
export WANDB_RUN_ID="${RUN_NAME}"
export WANDB_RESUME=allow
export NCCL_NET_SHARED_COMMS=0

readonly ENCODER_TP=1
readonly ENCODER_DP=32
readonly LLM_OFFSET=32
readonly LLM_TP=2
readonly LLM_CP=1
readonly LLM_PP=1
readonly LLM_DP=256
readonly LLM_EP=32
readonly LLM_EXPT_TP=1

if [[ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    LOAD_OPTIONS="--load ${CHECKPOINT_DIR}"
else
    LOAD_OPTIONS="--load-vision-from ${VISION_CHECKPOINT}"
fi

vlm_options=" \
    --model-provider nemotron-moe-mistral-vit \
    --mistral-vit-model-size 1.6b \
    --dataset-provider mistral \
    --patch-dim 14 \
    --spatial-merge-size 2 \
    --img-h 1540 \
    --img-w 1540 \
    --disable-vision-class-token \
    --dynamic-resolution \
    --vision-projection-type mlp2 \
    --freeze-vit \
    --mimo-encoder-prefetch \
    --mimo-encoder-prefetch-depth 16 \
    --mimo-bridge-skip-shape-exchange \
    --mimo-encoder-tp ${ENCODER_TP} \
    --mimo-encoder-dp ${ENCODER_DP} \
    --mimo-llm-offset ${LLM_OFFSET} \
    --mimo-llm-tp ${LLM_TP} \
    --mimo-llm-cp ${LLM_CP} \
    --mimo-llm-pp ${LLM_PP} \
    --mimo-llm-dp ${LLM_DP} \
    --mimo-llm-ep ${LLM_EP} \
    --mimo-llm-expt-tp ${LLM_EXPT_TP} "

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
    --eval-interval "${EVAL_INTERVAL}" \
    --eval-iters 0 \
    --per-source-validation-top-n "${PER_SOURCE_VALIDATION_TOP_N}" \
    \
    --te-rng-tracker \
    --no-load-rng \
    \
    --dataloader-type external \
    --pretokenized-config ${PRETOKENIZED_CONFIG} \
    --tokenizer-type NullTokenizer \
    --no-save-tokenizer-assets \
    --vocab-size 262272 \
    --padded-vocab-size 262272 \
    --null-tokenizer-eod-id 2 \
    --null-tokenizer-pad-id 3 \
    --num-workers 4 \
    --mistral-encoder-num-workers 16 \
    --mistral-image-group-workers 4 \
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
    --flash-attention-version 2 \
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

export REPO_DIR TRITON_CACHE_ROOT vlm_options options
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
        exec /usr/bin/python -u -m examples.mimo.pretrain_mimo ${vlm_options} ${options}
    '
