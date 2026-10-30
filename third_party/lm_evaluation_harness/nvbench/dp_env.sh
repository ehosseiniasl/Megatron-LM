#!/bin/bash

# WARNING: Use of static engine + dp_env.sh is discouraged.
# Please pull megatron-lm closer to main and use the dynamic text generation
# server instead: nemotron-serve <checkpoint_path> --tier nano|super|ultra
echo "WARNING: Use of static engine + dp_env.sh is discouraged. Please pull megatron-lm closer to main and use the dynamic text generation server instead." >&2

VLLM=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --tp)
      TP="$2"
      shift 2
      ;;
    --pp)
      PP="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --launch)
      LAUNCH_CMD="$2"
      shift 2
      ;;
    --vllm)
      VLLM=1
      shift 1
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$TP" || -z "$PP" || -z "$PORT" ]]; then
    echo "Usage: $0 --tp <tensor_parallel> --pp <pipeline_parallel> --port <port> --launch <command>"
    exit 1
fi

# Calculate derived values
IFS=',' read -ra VISIBLE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS_PER_TASK=$((TP * PP))

# Validate GPU availability
if [[ $NUM_GPUS_PER_TASK -gt ${#VISIBLE_DEVICES[@]} ]]; then
    echo "Error: Not enough GPUs per task. Cannot handle multi-node yet"
    exit 1
fi

# Calculate group assignments
NUM_GPUS_PER_NODE=${#VISIBLE_DEVICES[@]}
MASTER_PORT=${MASTER_PORT:-12345}
GLOBAL_RANK=${SLURM_PROCID:-0}
GROUP_RANK=$((GLOBAL_RANK / NUM_GPUS_PER_TASK))
NUM_GROUPS_PER_NODE=$((NUM_GPUS_PER_NODE / NUM_GPUS_PER_TASK))
LOCAL_GROUP_RANK=$((GROUP_RANK % NUM_GROUPS_PER_NODE))

# Extract visible devices for this group
START_IDX=$((LOCAL_GROUP_RANK * NUM_GPUS_PER_TASK))
END_IDX=$(((LOCAL_GROUP_RANK + 1) * NUM_GPUS_PER_TASK - 1))
GROUP_DEVICES=""
for i in $(seq $START_IDX $END_IDX); do
    if [[ -n "$GROUP_DEVICES" ]]; then
        GROUP_DEVICES="$GROUP_DEVICES,${VISIBLE_DEVICES[$i]}"
    else
        GROUP_DEVICES="${VISIBLE_DEVICES[$i]}"
    fi
done

# Set environment variables and execute
export MASTER_PORT=$((MASTER_PORT + GROUP_RANK))
export SLURM_PROCID=$((GLOBAL_RANK % NUM_GPUS_PER_TASK))
export SLURM_NTASKS=$NUM_GPUS_PER_TASK
export SLURM_NODEID=0
export SLURM_NNODES=1
export SLURM_NODELIST=$(hostname)
export SLURM_LOCALID=$((GLOBAL_RANK % NUM_GPUS_PER_TASK))
export CUDA_VISIBLE_DEVICES="$GROUP_DEVICES"
export RANK=$((GLOBAL_RANK % NUM_GPUS_PER_TASK))
export LOCAL_RANK=$((GLOBAL_RANK % NUM_GPUS_PER_TASK))
export WORLD_SIZE=$NUM_GPUS_PER_TASK
export MASTER_ADDR=$(hostname)
PORT=$((PORT + GROUP_RANK))

# Signal to load_balancer.py that DP endpoint discovery should be used
export NVBENCH_DP_DISCOVER=1

echo "Environment overrides: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES RANK=$RANK GROUP_RANK=$GROUP_RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE PORT=$PORT"

# Remove trailing whitespace from LAUNCH_CMD and remove existing --port argument if present
LAUNCH_CMD=$(echo "$LAUNCH_CMD" | sed 's/[[:space:]]*$//')
LAUNCH_CMD=$(echo "$LAUNCH_CMD" | sed 's/--port[[:space:]]\+[^[:space:]]\+//g')
FULL_COMMAND="$LAUNCH_CMD --port $PORT"

if [[ "$VLLM" -eq 0 || ( "$VLLM" -eq 1 && "$LOCAL_RANK" -eq 0 ) ]]; then
    echo "Executing on $(hostname): $FULL_COMMAND"
    exec $FULL_COMMAND
fi
