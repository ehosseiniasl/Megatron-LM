#!/bin/bash

set -euo pipefail

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
export NVBENCH_SERVER_IMAGE=/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/sasatheesh/data/images/m_lm_energon_cmh_gb300_pre_vlm_05_pre_energon_07_efa_pt2602_nvrx_0_6_0_v2.sqsh
export NVBENCH_CLIENT_IMAGE=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/antpeng/images/m_lm_vlmeval_20260621.sqsh
export NVBENCH_CLIENT_EXTRA_PYTHONPATH=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/kshih/workspace/megatron-lm/logs/evals/contained_vlmevalkit/assets/client_python_packages_m_lm_vlmeval_20260621

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/third_party/VLMEvalKit${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 -m nvbench.submit_suite \
  --run-config "${RUN_CONFIG}" \
  --workspace "${NVBENCH_WORKSPACE:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/kshih/workspace}" \
  "$@"
