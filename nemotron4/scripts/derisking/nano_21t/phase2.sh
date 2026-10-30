#!/bin/bash

#SBATCH -p batch_long
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=400
#SBATCH --reservation=sla_res_nemotron_sw_pre
#SBATCH --qos=hero-res
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --segment=16
#SBATCH --dependency=singleton
#SBATCH --job-name=nemotron4_derisking_nano_21t_phase2
#SBATCH --comment='{"APS":{}}'


# CUDA_DEVICE_MAX_CONNECTIONS is deliberately unset: Blackwell does not need it,
# and tensor model parallelism is 1 here in any case.
export NVTE_FWD_LAYERNORM_SM_MARGIN=40
export NVTE_BWD_LAYERNORM_SM_MARGIN=40
export NVTE_INF_LAYERNORM_SM_MARGIN=40
export TORCHINDUCTOR_WORKER_START=fork
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NVLS supplies the symmetric windows that --gtp-remat-nccl-ub registers against;
# without it that flag has nothing to hook into. Also, disable the CopyEngine
# collectives since they are slower.
export NCCL_NVLS_ENABLE=1
export NCCL_SYM_CE_THRESHOLD=4294967296
# Asserted by --fine-grained-activation-offloading on TE >= 2.10.
export NVTE_CPU_OFFLOAD_V1=1
# Without this the TE CuTe DSL fused grouped MLP silently declines to fuse.
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=0
# HybridEP dispatcher: NVLink domain size matches the 64-rank EP group.
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64
export USE_MNNVL=1
export MIMALLOC_PURGE_DELAY=-1
export TRITON_CACHE_DIR="/tmp/triton_cache"

# Basic set of paths for run.
ROOT_DIR="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron4"
REPO_DIR="${ROOT_DIR}/code"
IMAGE_PATH="${ROOT_DIR}/images/nvidia+pytorch+26.07-py3+deps+v8.sqsh"
NAME="derisking/nano_21t/phase2"
RUN_DIR="${ROOT_DIR}/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${TENSORBOARD_DIR}
QUANT_CONFIG="${ROOT_DIR}/configs/bf16_overrides.yaml"
test -f "${IMAGE_PATH}"
test -f "${QUANT_CONFIG}"
test -f "${REPO_DIR}/examples/mimo/pretrain_mimo.py"

# Parallelization-group distribution cache: replaces the all_gather_object in
# determine_main_replica_uniform_distribution with one file read per group. Set CREATE to
# --ckpt-pg-tensors-cache-create for a single run to populate it, then back to "". The read
# path does no validation, so delete the directory and re-populate if the grid, the world
# size or the ep_dp group changes.
CKPT_PG_TENSORS_CACHE_DIR="${CHECKPOINT_DIR}/pg_tensors_cache"
CKPT_PG_TENSORS_CACHE_CREATE=""
mkdir -p ${CKPT_PG_TENSORS_CACHE_DIR}

# Data + VLM.
SOURCE_PRETOKENIZED_CONFIG="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/jkamalu/mistral-derisking/nv-mistralai-megatron.PREBUILD/derisking/configs-blend/N4-21T-blend1-Sep4-70-30-phase2-no_vlm_sft-with_think_filtering.yaml"
# Evaluation sources; override VALIDATION_CONFIG to select a different fragment.
VALIDATION_CONFIG="${VALIDATION_CONFIG:-${REPO_DIR}/nemotron4/validation/mistral_arxiv.yaml}"
INPUT_TOKENS_PER_SOURCE="${INPUT_TOKENS_PER_SOURCE:-131072}"
PERCENTAGE_DATA=1.0
# TODO: Move this into ${ROOT_DIR} down the road.
EPOCH_INDEX_CACHE="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron-v4/sandia/epoch-index-PR390"
LOCAL_IMAGE_ROOT="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron-v4/sandia/image_buckets"
S3_NAMESPACE_CONFIG="${REPO_DIR}/mistral_adapter/example_configs/n4_coalition_image_namespaces.json"
PRETOKENIZED_CONFIG="${RUN_DIR}/pretokenized.yaml"
# Lustre.
unset AWS_PROFILE MISTRAL_S3_ENDPOINT_URL
export MISTRAL_LOCAL_IMAGE_ROOT="${LOCAL_IMAGE_ROOT}"
# S3.
# unset MISTRAL_LOCAL_IMAGE_ROOT
# export AWS_CONFIG_FILE="${ROOT_DIR}/configs/aws_config"
# export AWS_PROFILE="nvidia-sandia-cmh"
# export MISTRAL_S3_ENDPOINT_URL="https://s3.us-east-2.amazonaws.com"
# export MISTRAL_S3_NAMESPACE_CONFIG="${S3_NAMESPACE_CONFIG}"

VISION_CHECKPOINT_DIR="${ROOT_DIR}/vision_encoder_checkpoints/mcore"

test -f "${VALIDATION_CONFIG}" || exit 1
if grep -q '^per_source_validation:' "${SOURCE_PRETOKENIZED_CONFIG}"; then
    echo "Training config already defines per_source_validation" >&2
    exit 1
fi

awk -v cache_path="${EPOCH_INDEX_CACHE}" -v percentage_data="${PERCENTAGE_DATA}" '
    /^  percentage_data:/ { $0 = "  percentage_data: " percentage_data }
    /^epoch_index_cache_path:/ { $0 = "epoch_index_cache_path: " cache_path }
    { print }
' "${SOURCE_PRETOKENIZED_CONFIG}" > "${PRETOKENIZED_CONFIG}.tmp"
cat "${VALIDATION_CONFIG}" >> "${PRETOKENIZED_CONFIG}.tmp" || exit 1
mv "${PRETOKENIZED_CONFIG}.tmp" "${PRETOKENIZED_CONFIG}"

test -f "${AWS_CONFIG_FILE}"
test -f "${S3_NAMESPACE_CONFIG}"
test -f "${SOURCE_PRETOKENIZED_CONFIG}"
test -d "${VISION_CHECKPOINT_DIR}"
test -d "${EPOCH_INDEX_CACHE}"


DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
# Always submitted through sbatch, so SLURM_JOB_ID is what names the env log. Assert
# it rather than falling back: the old fallback built the name from ${NAME}, which
# contains slashes, so tee wrote to a path whose parents did not exist and the env log
# was silently never produced.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: SLURM_JOB_ID is unset; submit this with sbatch rather than running it directly." >&2
    exit 1
fi
# scontrol records Command= exactly as it was typed, so a job submitted as
# ../run.sh reports a relative path and dirname collapses to "..". WorkDir is what
# that path was relative to; the case guard leaves an absolute Command= alone.
SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
SCRIPT_WORKDIR=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/WorkDir=/{print $2}')
case "${SCRIPT_PATH}" in /*) ;; *) SCRIPT_PATH="${SCRIPT_WORKDIR}/${SCRIPT_PATH}" ;; esac
SCRIPT_PATH=$(realpath -m "${SCRIPT_PATH}")
# Same stem as the srun log, which is "${LOGS_DIR}/%x_%j_${DATETIME}.log", so the two
# files for a job sort together and differ only by the .env: %x is SLURM_JOB_NAME and
# %j is SLURM_JOB_ID.
ENV_LOG_FILENAME=${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log

SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")


# Log environment.
echo "<< START PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "IMAGE_PATH=${IMAGE_PATH}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "REPO_DIR=${REPO_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "SCRIPT_PATH=${SCRIPT_PATH}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT LOG" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${REPO_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${REPO_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${REPO_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}


# Pretraining hyperparameters.
SEQ_LEN=8192
# The run is 21T tokens end to end and the learning-rate schedule spans all of it, so
# the horizon is named apart from where this phase stops. Phase 1 ends at
# --phase-transition-samples, where Megatron saves and exits so a phase-2 job can pick
# up with the phase-2 blend (training.py:4158, :4265, :5649).
# ORIGINAL_TRAIN_SAMPLES is the 21T horizon everything that has already happened was
# derived from. The batch-size ramp, LR warmup and the phase transition stay pinned to
# it so the run continues on exactly the schedule it has been following.
ORIGINAL_TRAIN_SAMPLES=2563476562 # 21T tokens at SEQ_LEN.
# The end of the run is pulled in by 54B tokens. The decay tail stays 20% of the horizon,
# so it is recomputed from the shortened TOTAL_TRAIN_SAMPLES and the decay onset moves in
# with it.
HORIZON_REDUCTION_SAMPLES=$((54000000000 / SEQ_LEN)) # 54B tokens.
TOTAL_TRAIN_SAMPLES=$((ORIGINAL_TRAIN_SAMPLES - HORIZON_REDUCTION_SAMPLES)) # ~20.95T.
PHASE_TRANSITION_SAMPLES=$((ORIGINAL_TRAIN_SAMPLES * 70 / 100)) # 70% of 21T, 14.7T.
LR_DECAY_SAMPLES=${TOTAL_TRAIN_SAMPLES}
LR_WSD_DECAY_SAMPLES=$((TOTAL_TRAIN_SAMPLES / 5)) # 20% decay.

# Batch size ramp up schedule, over 50/21 percent of the run. The odd-looking fraction
# is deliberate: 50/21 % of 21T is the same number of samples as the 5% of 10T this
# used to be, so decoupling the horizon leaves the ramp bit-identical at
# TARGET_RAMP_SAMPLES=61035156, RAMP_STEPS=13245 and the same schedule tokens.
BATCH_SIZE_RAMP_PERCENT_NUMERATOR=50
BATCH_SIZE_RAMP_PERCENT_DENOMINATOR=21
MAX_BATCH_SIZE=4608
TARGET_RAMP_SAMPLES=$((ORIGINAL_TRAIN_SAMPLES * BATCH_SIZE_RAMP_PERCENT_NUMERATOR \
    / (100 * BATCH_SIZE_RAMP_PERCENT_DENOMINATOR)))
RAMP_STEPS=$(((TARGET_RAMP_SAMPLES + (MAX_BATCH_SIZE / 2)) / MAX_BATCH_SIZE))
B1_START_TOKENS=$((RAMP_STEPS * (MAX_BATCH_SIZE / 3) * SEQ_LEN))
B2_START_TOKENS=$((RAMP_STEPS * MAX_BATCH_SIZE * SEQ_LEN))
STEP_BATCH_SIZE_SCHEDULE="0:1536,${B1_START_TOKENS}:3072,${B2_START_TOKENS}:4608"

# Warmup covers the batch size ramp.
LR_WARMUP_SAMPLES=${TARGET_RAMP_SAMPLES}


# If checkpoint doesn't exist, just load (frozen) vision encoder checkpoint.
if [[ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    LOAD_OPTIONS="--load ${CHECKPOINT_DIR}"
else
    LOAD_OPTIONS="--load-vision-from ${VISION_CHECKPOINT_DIR}"
fi

# Megatron-LM training arguments.
OPTIONS=" \
    --use-mcore-models \
    --hybrid-layer-pattern M-M-M*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEM-M-/*E/*E \
    --spec megatron.core.models.hybrid.hybrid_layer_specs gated_delta_product_stack_spec \
    --hidden-size 3072 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 1 \
    --mamba-num-heads 24 \
    --mamba-head-dim 64 \
    --mamba-state-dim 128 \
    --mamba-num-groups 24 \
    --ffn-hidden-size 12288 \
    --kv-channels 256 \
    --squared-relu \
    --activation-func-tanh-clamp-scale 16 \
    --untie-embeddings-and-output-weights \
    --embedding-min-norm 1e-3 \
    --init-method-std 0.0162 \
    --position-embedding-type none \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --normalization RMSNorm \
    --output-logit-dtype fp32 \
    \
    --num-experts 512 \
    --moe-router-topk 10 \
    --moe-latent-size 768 \
    --moe-ffn-hidden-size 4096 \
    --moe-shared-expert-intermediate-size 4096 \
    --moe-shortcut-connection \
    --moe-shortcut-parallel \
    --moe-use-norm-before-up-proj \
    --moe-token-dispatcher-type flex \
    --moe-flex-dispatcher-backend hybridep \
    --moe-flex-dispatcher-num-sms 32 \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 0 \
    --moe-router-topk-scaling-factor 3.16 \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type quantile_balancing \
    --moe-router-quantile-balancing-estimation-scope global_batch \
    --moe-router-qb-num-bins 1000 \
    --moe-permute-fusion \
    --moe-router-fusion \
    \
    --wide-residual 3 \
    --wide-residual-streamwise-sigmoid-init-scale 0.01 \
    --wide-residual-learned-retention \
    --wide-residual-retention-init 0.999 \
    --wide-residual-retention-max-forget 0.10 \
    \
    --use-transformer-engine-op-fuser \
    --recompute-granularity selective \
    --recompute-modules residual_stream moe_act gdp_in_proj gdp_qkv \
    --residual-stream-recompute-num-layers 4 \
    --fine-grained-activation-offloading \
    --offload-modules fused_group_mlp \
    \
    --mtp-use-repeated-layer \
    --mtp-loss-scaling-factor 0.1 \
    --calculate-per-token-loss \
    --mtp-normalize-loss-by-main-tokens \
    --mtp-hsm \
    \
    --gdp-cutedsl-kernel \
    --use-fused-weighted-squared-relu \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    \
    --optimizer muon \
    --muon-momentum 0.9 \
    --muon-extra-scale-factor 0.2 \
    --muon-scale-mode spectral \
    --muon-num-ns-steps 16 \
    --muon-coefficient-type polar_express \
    --muon-scalar-optimizer adam \
    --muon-tp-mode auto \
    --muon-expert-tp-mode layer_sharded \
    --moe-router-skip-muon \
    \
    --bf16 \
    --moe-router-padding-for-quantization \
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --reuse-grad-buf-for-mxfp8-param-ag \
    --fp8-param-gather \
    --te-precision-config-file ${QUANT_CONFIG} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --train-samples ${TOTAL_TRAIN_SAMPLES} \
    --phase-transition-samples ${PHASE_TRANSITION_SAMPLES} \
    --lr-decay-style WSD \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    --lr-wsd-decay-style minus_sqrt \
    --lr-wsd-decay-samples ${LR_WSD_DECAY_SAMPLES} \
    --override-opt_param-scheduler \
    --micro-batch-size 3 \
    --step-batch-size-schedule "${STEP_BATCH_SIZE_SCHEDULE}" \
    --lr 8.0e-4 \
    --min-lr 2.0e-5 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-16 \
    \
    --eval-interval 2000 \
    --eval-iters 0 \
    --input-tokens-per-source ${INPUT_TOKENS_PER_SOURCE} \
    --eval-micro-batch-size 1 \
    --eval-global-batch-size ${MAX_BATCH_SIZE} \
    \
    --dataloader-type external \
    --pretokenized-config ${PRETOKENIZED_CONFIG} \
    --tokenizer-type NullTokenizer \
    --vocab-size 262272 \
    --padded-vocab-size 262272 \
    --null-tokenizer-eod-id 2 \
    --null-tokenizer-pad-id 3 \
    --num-workers 1 \
    --no-create-attention-mask-in-dataloader \
    \
    --no-save-tokenizer-assets \
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
    --mimo-bridge-skip-shape-exchange \
    --mimo-encoder-prefetch \
    --mimo-encoder-prefetch-depth 4 \
    --mistral-encoder-num-workers 2 \
    --mistral-image-group-workers 4 \
    \
    --mimo-encoder-tp 1 \
    --mimo-encoder-dp 64 \
    --mimo-llm-offset 64 \
    --mimo-llm-tp 1 \
    --mimo-llm-cp 1 \
    --mimo-llm-pp 1 \
    --mimo-llm-dp 24 \
    --mimo-llm-ep 64 \
    --mimo-llm-expt-tp 1 \
    --tensor-parallel-num-weight-shards 64 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --high-priority-stream-groups ep \
    --grad-reduce-in-bf16 \
    --ddp-reduce-scatter-with-fp32-accumulation \
    --ddp-param-name-patterns-for-fp32-local-accumulation *output_layer* *router* \
    --gtp-remat-reduce-scatter-with-fp32-accumulation \
    --gtp-remat-nccl-ub \
    --ddp-num-buckets 10 \
    --ddp-pad-buckets-for-high-nccl-busbw \
    --attention-backend flash \
    --flash-attention-version 4 \
    \
    --seed 1234 \
    --te-rng-tracker \
    --no-load-rng \
    --rerun-mode disabled \
    \
    --ckpt-format torch_dist \
    ${LOAD_OPTIONS} \
    --save ${CHECKPOINT_DIR} \
    --save-interval 200 \
    --save-retain-interval 2000 \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load \
    --ckpt-assume-constant-structure \
    --ckpt-fully-parallel-save-process-group ep_dp \
    --ckpt-fully-parallel-load-process-group ep_dp \
    --ckpt-fully-parallel-load-per-rank-objects \
    --ckpt-pg-tensors-cache-path ${CKPT_PG_TENSORS_CACHE_DIR} \
    ${CKPT_PG_TENSORS_CACHE_CREATE} \
    --ckpt-drop-redundant-extra-state \
    --no-ckpt-load-validate-sharding-integrity \
    --async-save \
    --use-persistent-ckpt-worker \
    --async-ckpt-use-cpu-shm \
    --result-rejected-tracker-filename ${CHECKPOINT_DIR}/result_rejected_tracker.txt \
    \
    --log-interval 10 \
    --log-memory-interval 1000 \
    --log-device-memory-used \
    --log-num-zeros-in-grad \
    --log-params-norm \
    --log-throughput \
    --log-progress \
    --log-energy \
    --log-quantization-types \
    --logging-level 20 \
    --timing-log-option minmax \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-queue-size 65536 \
    --tensor-metrics \
        layer-router-health:500 \
        layer-router-expert-state:500 \
        layer-router-seq-aux-decomposition:500 \
        param-l2:500 \
        wgrad-l2:500 \
        wgrad-abs-max:500 \
        wgrad-zero-fraction:500 \
        layer-residual-accumulator-l2:500 layer-residual-contribution-l2:500 \
    \
    --manual-gc \
    --manual-gc-interval 100 \
    --distributed-timeout-minutes 30 \
    --exit-duration-in-mins 1430 \
    --disable-gloo-process-groups \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
        "

	
RUN_CMD="set -f; cd ${REPO_DIR}; exec python -u -m examples.mimo.pretrain_mimo ${OPTIONS}"

# Megatron arguments, one flag per line with its values, matching how OPTIONS is
# written above. Logged here rather than in the paths block because OPTIONS is only
# assembled at this point. set -f keeps the glob patterns (*output_layer*, *router*)
# from being expanded against the working directory while word-splitting OPTIONS.
echo "<< START ARGS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
set -f
printf '%s\n' ${OPTIONS} | awk '
    /^--/ { if (started) printf "\n"; printf "    %s", $0; started = 1; next }
    started { printf " %s", $0 }
    END { if (started) printf "\n" }' |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
set +f
echo "<< END ARGS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

srun -l \
     --mpi=none \
     --container-image=${IMAGE_PATH} \
     --container-mounts="/scratch:/scratch,/lustre:/lustre" \
     --no-container-mount-home \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     bash -c "${RUN_CMD}"

