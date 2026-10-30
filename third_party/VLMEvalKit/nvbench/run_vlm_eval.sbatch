#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=vlmeval-fewshot

set -euo pipefail
umask 027
SERVER_IMAGE=${NVBENCH_SERVER_IMAGE:?Export NVBENCH_SERVER_IMAGE before calling sbatch}
CLIENT_IMAGE=${NVBENCH_CLIENT_IMAGE:?Export NVBENCH_CLIENT_IMAGE before calling sbatch}
CLIENT_LD_LIBRARY_PATH=${NVBENCH_CLIENT_LD_LIBRARY_PATH:-/opt/hpcx/ucx/lib}
CLIENT_EXTRA_PYTHONPATH=${NVBENCH_CLIENT_EXTRA_PYTHONPATH:-}
MOUNTS=${NVBENCH_CONTAINER_MOUNTS:-/home:/home,/lustre:/lustre,/scratch:/scratch}
VENDORED_VLMEVALKIT_ROOT=${SLURM_SUBMIT_DIR}/third_party/VLMEvalKit
ARRAY_INDEX=${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}
NODES=${SLURM_NNODES:?The allocation must expose SLURM_NNODES}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:?The allocation must expose SLURM_GPUS_ON_NODE}
SERVER_CPUS=${NVBENCH_SERVER_CPUS:?The submission planner must set NVBENCH_SERVER_CPUS}
SERVER_MEMORY=${NVBENCH_SERVER_MEMORY:?The submission planner must set NVBENCH_SERVER_MEMORY}
CLIENT_CPUS=${NVBENCH_CLIENT_CPUS:?The submission planner must set NVBENCH_CLIENT_CPUS}
CLIENT_MEMORY=${NVBENCH_CLIENT_MEMORY:?The submission planner must set NVBENCH_CLIENT_MEMORY}
SERVER_HOST=127.0.0.1
SERVER_ARGS=()

if (( NODES > 1 )); then
  SERVER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p' | tr '[:upper:]' '[:lower:]')
  RENDEZVOUS_PORT=${NVBENCH_RENDEZVOUS_PORT:-29500}
  SERVER_ARGS=(--master-addr "${SERVER_HOST}" --rendezvous-port "${RENDEZVOUS_PORT}")
fi

# One supervisor per node starts either local torchrun or one cross-node group.
# The CPU-only client stays on node zero beside the HTTP rank.
srun --mpi=none --exact --overlap --nodes="${NODES}" --ntasks="${NODES}" \
  --ntasks-per-node=1 --cpus-per-task="${SERVER_CPUS}" \
  --gpus-per-task="${GPUS_PER_NODE}" --mem="${SERVER_MEMORY}" \
  --container-image="${SERVER_IMAGE}" --container-mounts="${MOUNTS}" \
  --container-workdir="${SLURM_SUBMIT_DIR}" \
  env PYTHONPATH="${VENDORED_VLMEVALKIT_ROOT}" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  python3 -m nvbench.run_server \
  "$@" --server-host "${SERVER_HOST}" "${SERVER_ARGS[@]}" \
  --benchmark-index "${ARRAY_INDEX}" &
SERVER_STEP_PID=$!
cleanup_server_step() { kill -TERM "${SERVER_STEP_PID}" 2>/dev/null || true; wait "${SERVER_STEP_PID}" || true; }
trap cleanup_server_step EXIT INT TERM

srun --mpi=none --exact --overlap --relative=0 --nodes=1 --ntasks=1 \
  --cpus-per-task="${CLIENT_CPUS}" --gpus=0 --mem="${CLIENT_MEMORY}" \
  --container-image="${CLIENT_IMAGE}" --container-mounts="${MOUNTS}" \
  --container-workdir="${SLURM_SUBMIT_DIR}" \
  env LD_LIBRARY_PATH="${CLIENT_LD_LIBRARY_PATH}" \
  PYTHONPATH="${VENDORED_VLMEVALKIT_ROOT}${CLIENT_EXTRA_PYTHONPATH:+:${CLIENT_EXTRA_PYTHONPATH}}" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  python3 -m nvbench.run_client \
  "$@" --server-host "${SERVER_HOST}" --benchmark-index "${ARRAY_INDEX}"

wait "${SERVER_STEP_PID}"
trap - EXIT INT TERM
