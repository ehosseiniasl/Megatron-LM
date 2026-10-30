#!/bin/bash

#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH -A llmservice_nemo_mlops
#SBATCH -J llmservice_nemo_mlops:llama-3b-base-aegis-v2-nvdev-judge+vllm-test-model
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

export TASK="aegis_v2_content_safety"
export TEST_MODEL="meta-llama/Llama-3.2-3B"

export OUTPUT_PATH="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/evaluations/${TASK}/${TEST_MODEL}"
export CACHE_PATH="${OUTPUT_PATH}/cache.db"
mkdir -p "${OUTPUT_PATH}"

# =================== FUNCTION TO WAIT FOR MODEL SERVER ====================
function wait_for_model_server() {
    echo "Waiting for Test Model server to be ready..."
    local RETRIES=60
    local SLEEP_TIME=5
    for ((i=1; i<=RETRIES; i++)); do
        if curl -s http://localhost:8000/v1/models | grep -q '"id":'; then
            echo "✅ Test Model server is up!"
            return 0
        fi
        echo "⏳ Test Model server not ready yet (attempt $i/$RETRIES)..."
        sleep $SLEEP_TIME
    done
    echo "❌ Timed out waiting for Test Model server"
    exit 1
}

# =================== SINGLE JOB: TEST MODEL + EVALUATION ====================
srun \
    -l \
    --output "${LOGS_DIR}/test_model+eval_%x_%j.log" \
    --container-image="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/containers/vllm_v0.6.6.post1-deci.sqsh" \
    --container-mounts="/lustre:/lustre,/home:/home" \
    bash -c "
huggingface-cli login --token \"${HF_TOKEN}\"
export TRITON_CACHE_DIR=\"${TRITON_CACHE_DIR}\"
export HF_HOME=\"${HF_HOME}\"
export NVIDIA_API_KEY=\"${NVIDIA_API_KEY}\"
export OPENAI_API_KEY=\"dummy-key\"
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Start the model server
python3 -m vllm.entrypoints.openai.api_server \
    --model '${TEST_MODEL}' \
    --served-model-name '${TEST_MODEL}' \
    --trust-remote-code \
    --seed 42 \
    --host '0.0.0.0' \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --download-dir \"${HF_HOME}\" &

TEST_MODEL_PID=\$!
sleep 60

# Wait for model server to be ready
$(declare -f wait_for_model_server)
wait_for_model_server

# Setup and run evaluation
cd ${LM_EVAL_DIR}
pip install -e .[api]
pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger

lm_eval \
    --output_path ${OUTPUT_PATH} \
    --model local-completions \
    --model_args model=${TEST_MODEL},base_url=http://0.0.0.0:8000/v1,tokenizer=${TEST_MODEL},tokenizer_backend=huggingface \
    --tasks ${TASK} \
    --num_fewshot 0 \
    --use_cache ${CACHE_PATH} \
    --log_samples \
    --write_out

# Wait for model server to finish
wait \$TEST_MODEL_PID"