#!/bin/bash

#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH -A llmservice_nemo_mlops
#SBATCH -J llmservice_nemo_mlops:aegis-v2-nvdev-judge+hf-test-model
#SBATCH -t 04:00:00
#SBATCH --exclusive
#SBATCH --mem=0MB
#SBATCH --dependency=singleton
#SBATCH --partition=polar,polar3,polar4

# =================== ENV SETUP ====================
export NVIDIA_API_KEY=???
export HF_TOKEN=???

export LOGS_DIR="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/logs"
export HF_HOME="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/hf_cache"
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/trt_cache_dir/trt_cache_dir"
export LM_EVAL_DIR="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/codebases/lm-eval-adlr-aegis"
export ACCELERATE_CONFIG_FILE="${HF_HOME}/accelerate/default_config.yaml"

export TASK="aegis_v2_content_safety"
export TEST_MODEL="nvidia/Nemotron-H-8B-Base-8K"

export OUTPUT_PATH="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/evaluations/${TASK}/${TEST_MODEL}"
export CACHE_PATH="${OUTPUT_PATH}/cache.db"
mkdir -p "${OUTPUT_PATH}"

# =================== SINGLE JOB: HF MODEL EVALUATION ====================
srun \
    -l \
    --output "${LOGS_DIR}/test_model+eval_%x_%j.log" \
    --container-image="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/containers/nemotron5-111524_mamba_ssm_222_fix.sqsh" \
    --container-mounts="/lustre:/lustre,/home:/home" \
    bash -c "
huggingface-cli login --token \"${HF_TOKEN}\"
export TRITON_CACHE_DIR=\"${TRITON_CACHE_DIR}\"
export HF_HOME=\"${HF_HOME}\"
export NVIDIA_API_KEY=\"${NVIDIA_API_KEY}\"
export HF_DATASETS_TRUST_REMOTE_CODE=true

cd ${LM_EVAL_DIR}
pip install -e .[api]
pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger
pip install -U transformers
pip install wandb==0.16.6

accelerate launch --config_file ${ACCELERATE_CONFIG_FILE} lm_eval \
    --output_path ${OUTPUT_PATH} \
    --model hf \
    --model_args pretrained=${TEST_MODEL},trust_remote_code=True \
    --tasks ${TASK} \
    --num_fewshot 0 \
    --use_cache ${CACHE_PATH} \
    --log_samples \
    --write_out \
    --trust_remote_code"