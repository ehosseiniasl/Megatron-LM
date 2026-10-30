#!/bin/bash

#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH -A llmservice_nemo_mlops
#SBATCH -J llmservice_nemo_mlops:cs280-base-vllm-judge+hf-test-model-0
#SBATCH -t 04:00:00
#SBATCH --exclusive
#SBATCH --mem=0MB
#SBATCH --dependency=singleton
#SBATCH --partition=polar,polar3,polar4

#SBATCH hetjob
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH -A llmservice_nemo_mlops
#SBATCH -J llmservice_nemo_mlops:cs280-base-vllm-judge+hf-test-model-1
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

export TASK="cs_280_wildguard_judge"
export TEST_MODEL="nvidia/Nemotron-H-8B-Base-8K"
export JUDGE_MODEL="allenai/wildguard"

export OUTPUT_PATH="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/evaluations/${TASK}/${TEST_MODEL}_HF"
export CACHE_PATH="${OUTPUT_PATH}/cache.db"
mkdir -p "${OUTPUT_PATH}"

# =================== FUNCTION TO WAIT FOR JUDGE MODEL SERVER ====================
function wait_for_judge_server() {
    echo "Waiting for Judge Model server to be ready..."
    local RETRIES=60
    local SLEEP_TIME=5
    for ((i=1; i<=RETRIES; i++)); do
        if curl -s http://localhost:5000/v1/models | grep -q '"id":'; then
            echo "✅ Judge Model server is up!"
            return 0
        fi
        echo "⏳ Judge Model server not ready yet (attempt $i/$RETRIES)..."
        sleep $SLEEP_TIME
    done
    echo "❌ Timed out waiting for Judge Model server"
    exit 1
}

# =================== JOB 0: JUDGE MODEL ====================
srun \
    -l \
    --het-group=0 \
    --output "${LOGS_DIR}/judge_model_%x_%j.log" \
    --container-image="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/containers/vllm_v0.6.6.post1-deci.sqsh" \
    --container-mounts="/lustre:/lustre,/home:/home" \
    bash -c "
huggingface-cli login --token \"${HF_TOKEN}\"
export TRITON_CACHE_DIR=\"${TRITON_CACHE_DIR}\"
export HF_HOME=\"${HF_HOME}\"
export NVIDIA_API_KEY=\"${NVIDIA_API_KEY}\"
export HF_DATASETS_TRUST_REMOTE_CODE=true

python3 -m vllm.entrypoints.openai.api_server \
    --model '${JUDGE_MODEL}' \
    --served-model-name 'judge-model' \
    --trust-remote-code \
    --seed 1 \
    --host '0.0.0.0' \
    --port 5000 \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --download-dir \"${HF_HOME}\"" &

JUDGE_PID=$!
sleep 60

wait_for_judge_server

JUDGE_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "JUDGE_HOST: ${JUDGE_HOST}"

# =================== JOB 1: LOCAL HF MODEL EVALUATION ====================
srun \
    -l \
    --het-group=1 \
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
export JUDGE_NODE='${JUDGE_HOST}'

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

wait $JUDGE_PID