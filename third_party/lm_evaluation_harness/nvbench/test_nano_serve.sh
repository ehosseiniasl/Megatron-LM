#!/bin/bash
# Quick test: launch nemotron-serve for nano (3B) on a single node.
# Usage: bash nvbench/test_nano_serve.sh [JOBID]

set -euo pipefail

JOBID=${1:-${SLURM_JOB_ID:-}}
if [[ -z "${JOBID}" ]]; then
    echo "Usage: bash nvbench/test_nano_serve.sh <JOBID>"
    exit 1
fi

LME=/scratch/fsw/portfolios/general/projects/general_cs_infra/users/sasatheesh/src/lm-evaluation-harness
CKPT=/scratch/fsw/portfolios/general/projects/general_cs_infra/users/sasatheesh/data/checkpoints/nemo-3-nano-base-mcore-8k
MLM=/scratch/fsw/portfolios/general/projects/general_cs_infra/users/sasatheesh/src/megatron-lm
IMAGE=/home/sasatheesh/data/images/m_lm.sqsh

srun --jobid="${JOBID}" --nodes=1 --ntasks-per-node=1 --overlap \
  --container-image "${IMAGE}" \
  --container-mounts "/lustre:/lustre/,/scratch:/scratch/" \
  bash -lc "cd ${LME} && python3 nemotron-serve ${CKPT} \
    --tier nano --port 5000 --megatron-path ${MLM}"
