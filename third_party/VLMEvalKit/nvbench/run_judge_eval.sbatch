#!/bin/bash
#SBATCH --job-name=vlmeval-judge

set -euo pipefail
umask 027
CLIENT_IMAGE=${NVBENCH_CLIENT_IMAGE:?Export NVBENCH_CLIENT_IMAGE before calling sbatch}
CLIENT_LD_LIBRARY_PATH=${NVBENCH_CLIENT_LD_LIBRARY_PATH:-/opt/hpcx/ucx/lib}
CLIENT_EXTRA_PYTHONPATH=${NVBENCH_CLIENT_EXTRA_PYTHONPATH:-}
MOUNTS=${NVBENCH_CONTAINER_MOUNTS:-/home:/home,/lustre:/lustre,/scratch:/scratch}
VENDORED_VLMEVALKIT_ROOT=${SLURM_SUBMIT_DIR}/third_party/VLMEvalKit
ARRAY_INDEX=${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}

srun --mpi=none --exact --nodes=1 --ntasks=1 \
  --container-image="${CLIENT_IMAGE}" --container-mounts="${MOUNTS}" \
  --container-workdir="${SLURM_SUBMIT_DIR}" \
  env LD_LIBRARY_PATH="${CLIENT_LD_LIBRARY_PATH}" \
  PYTHONPATH="${VENDORED_VLMEVALKIT_ROOT}${CLIENT_EXTRA_PYTHONPATH:+:${CLIENT_EXTRA_PYTHONPATH}}" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  python3 -m nvbench.run_judge "$@" --benchmark-index "${ARRAY_INDEX}"
