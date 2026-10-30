#!/bin/bash

set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

if (( $# < 1 )); then
  echo "Usage: $0 RUN_CONFIG [submit_suite arguments...]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVBENCH_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${NVBENCH_ROOT}/../../.." && pwd)
RUN_CONFIG=$1
shift

if [[ ${RUN_CONFIG} != /* ]]; then
  RUN_CONFIG="${NVBENCH_ROOT}/${RUN_CONFIG}"
fi

export NCCL_NET=IB
export NVBENCH_SERVER_IMAGE=/home/kshih/kshih_lnk/workspace/images/m_lm_energon_cmh_gb300_pre_vlm_05_pre_energon_07_efa_pt2602_nvrx_0_6_0_v2.sqsh
export NVBENCH_CLIENT_IMAGE=/home/kshih/kshih_lnk/workspace/images/m_lm_vlmeval_20260621.sqsh
export NVBENCH_CLIENT_EXTRA_PYTHONPATH=/home/kshih/kshih_lnk/workspace/vlmevalkit_assets/client_python_packages_m_lm_vlmeval_20260621
export HF_HOME=/home/kshih/kshih_lnk/workspace/vlmevalkit_assets/huggingface
CACHE_ROOT=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/kshih/workspace/vlmevalkit_cache
WORLDVQA_EMBEDDED_CACHE=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/kshih/workspace/megatron-lm/logs/evals/contained_vlmevalkit/cache
export NVBENCH_CONTAINER_MOUNTS="/home:/home,/lustre:/lustre,/scratch:/scratch,${CACHE_ROOT}:${WORLDVQA_EMBEDDED_CACHE}:ro"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/third_party/VLMEvalKit${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 -m nvbench.submit_suite \
  --run-config "${RUN_CONFIG}" \
  --workspace "${NVBENCH_WORKSPACE:-/home/kshih/kshih_lnk/workspace}" \
  "$@"
