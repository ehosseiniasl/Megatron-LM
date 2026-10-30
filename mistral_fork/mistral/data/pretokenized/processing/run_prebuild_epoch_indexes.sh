#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)

GROUP_NAME=${GROUP_NAME-nemotron_coalition_mistral}
SLURM_ACCOUNT=${SLURM_ACCOUNT-${GROUP_NAME}}

if [[ -n "${GROUP_NAME}" ]] && ! id -Gn | tr ' ' '\n' | grep -qx "${GROUP_NAME}"; then
  echo "Please run this from a ${GROUP_NAME} shell first:"
  echo "  newgrp ${GROUP_NAME}"
  exit 1
fi

usage() {
  echo "Usage: $0 CONFIG [NUM_TOTAL_TOKENS]"
  echo
  echo "NUM_TOTAL_TOKENS may also be set as an environment variable."
  echo
  echo "Optional env overrides:"
  echo "  WORKDIR CONTAINER_IMAGE CONTAINER_MOUNTS GROUP_NAME SLURM_ACCOUNT"
  echo "  SLURM_PARTITION SLURM_QOS SLURM_TIME SLURM_MEM MEM_PER_PROCESS_GB"
  echo "  JOB_NAME LOGS_DIR"
  echo "  SLURM_NODELIST SLURM_EXCLUDE"
  echo "  NUM_TASKS MAX_PARALLEL_TASKS PROCESSES_PER_TASK CPUS_PER_PROCESS DRY_RUN"
  echo "  IMAGE_TRANSFORM_WORKERS"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

CONFIG=$1
NUM_TOTAL_TOKENS=${2:-${NUM_TOTAL_TOKENS:-}}
if [[ -z "${NUM_TOTAL_TOKENS}" ]]; then
  echo "NUM_TOTAL_TOKENS is required."
  usage
  exit 1
fi
if [[ ! "${NUM_TOTAL_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_TOTAL_TOKENS must be a positive integer, got: ${NUM_TOTAL_TOKENS}"
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config does not exist: ${CONFIG}"
  exit 1
fi

CONFIG=$(cd -- "$(dirname -- "${CONFIG}")" && pwd)/$(basename -- "${CONFIG}")
CONFIG_STEM=$(basename "${CONFIG}")
CONFIG_STEM=${CONFIG_STEM%.yaml}
CONFIG_STEM=${CONFIG_STEM%.yml}

WORKDIR=${WORKDIR:-${REPO_ROOT}}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nemotron/users/szymon.ma/images/nvidia+pytorch+26.04-py3+dependencies+mamba.sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-${HOME}:${HOME},/scratch:/scratch,/lustre:/lustre}
SLURM_PARTITION=${SLURM_PARTITION:-cpu}
SLURM_QOS=${SLURM_QOS:-cpu-free}
SLURM_TIME=${SLURM_TIME:-48:00:00}
SLURM_NODELIST=${SLURM_NODELIST:-}
SLURM_EXCLUDE=${SLURM_EXCLUDE:-}
NUM_TASKS=${NUM_TASKS:-512}
MAX_PARALLEL_TASKS=${MAX_PARALLEL_TASKS:-256}
PROCESSES_PER_TASK=${PROCESSES_PER_TASK:-5}
# Match Mistral's per-task prebuild defaults for every local process. Five
# processes request 20 CPUs and 160 GB, so four tasks fit on a Coalition CPU
# node with 96 CPUs and 711479 MB of schedulable memory.
CPUS_PER_PROCESS=${CPUS_PER_PROCESS:-4}
MEM_PER_PROCESS_GB=${MEM_PER_PROCESS_GB:-32}
IMAGE_TRANSFORM_WORKERS=${IMAGE_TRANSFORM_WORKERS:-${CPUS_PER_PROCESS}}
DRY_RUN=${DRY_RUN:-false}

if [[ -n "${SLURM_NODELIST}" && -n "${SLURM_EXCLUDE}" ]]; then
  echo "Set only one of SLURM_NODELIST or SLURM_EXCLUDE."
  exit 1
fi
for node_spec_name in SLURM_NODELIST SLURM_EXCLUDE; do
  node_spec=${!node_spec_name}
  if [[ -n "${node_spec}" && ! "${node_spec}" =~ ^[A-Za-z0-9_,.-]+$ ]]; then
    echo "${node_spec_name} contains unsupported characters: ${node_spec}"
    exit 1
  fi
done

SLURM_NODE_DIRECTIVE=
if [[ -n "${SLURM_NODELIST}" ]]; then
  SLURM_NODE_DIRECTIVE="#SBATCH --nodelist=${SLURM_NODELIST}"
elif [[ -n "${SLURM_EXCLUDE}" ]]; then
  SLURM_NODE_DIRECTIVE="#SBATCH --exclude=${SLURM_EXCLUDE}"
fi

for value_name in NUM_TASKS MAX_PARALLEL_TASKS PROCESSES_PER_TASK CPUS_PER_PROCESS MEM_PER_PROCESS_GB IMAGE_TRANSFORM_WORKERS; do
  value=${!value_name}
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got: ${value}"
    exit 1
  fi
done
if ((MAX_PARALLEL_TASKS > NUM_TASKS)); then
  echo "MAX_PARALLEL_TASKS cannot exceed NUM_TASKS: ${MAX_PARALLEL_TASKS} > ${NUM_TASKS}"
  exit 1
fi
CPUS_PER_TASK=$((PROCESSES_PER_TASK * CPUS_PER_PROCESS))
SLURM_MEM=${SLURM_MEM:-$((PROCESSES_PER_TASK * MEM_PER_PROCESS_GB))G}

LOGS_DIR=${LOGS_DIR:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/epoch_index/logs/jobs}
timestamp=$(date +%y%m%d_%H%M%S)
JOB_NAME=${JOB_NAME:-epoch_index_${CONFIG_STEM}_${NUM_TASKS}tasks_${PROCESSES_PER_TASK}proc}
job_dir=${LOGS_DIR}/${timestamp}_${JOB_NAME}
mkdir -p "${job_dir}"
submission_path=${job_dir}/submission.sbatch

cat >"${submission_path}" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --array=0-$((NUM_TASKS - 1))%${MAX_PARALLEL_TASKS}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --qos=${SLURM_QOS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --output=${job_dir}/slurm-%A_%4a.out
${SLURM_NODE_DIRECTIVE}

echo "Compute node: \$(hostname)"
findmnt -T /raid -n -o TARGET,SOURCE,FSTYPE,OPTIONS || true
df -hT /raid /raid/enroot 2>/dev/null || true
df -i /raid /raid/enroot 2>/dev/null || true

newgrp ${GROUP_NAME} <<'NEWGRP_EOF'
set -euo pipefail
cd "${WORKDIR}"
srun \
  --ntasks=1 \
  --cpus-per-task=${CPUS_PER_TASK} \
  --container-image="${CONTAINER_IMAGE}" \
  --container-workdir="${WORKDIR}" \
  --no-container-remap-root \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-env=HOME,LANG,LANGUAGE,LC_ALL,SLURM_ARRAY_TASK_ID,SLURM_ARRAY_TASK_COUNT,SLURM_CPUS_PER_TASK \
  bash -lc 'export OMP_NUM_THREADS=${CPUS_PER_PROCESS} \
    MKL_NUM_THREADS=${CPUS_PER_PROCESS} \
    OPENBLAS_NUM_THREADS=${CPUS_PER_PROCESS} \
    PYTHONUNBUFFERED=1; \
  python3 -m mistral_fork.mistral.data.pretokenized.processing.prebuild_epoch_indexes_megatron \
    prebuild \
    --config "${CONFIG}" \
    --num_total_tokens "${NUM_TOTAL_TOKENS}" \
    --image-transform-workers "${IMAGE_TRANSFORM_WORKERS}" \
    --worker_id "\${SLURM_ARRAY_TASK_ID}" \
    --num_workers "\${SLURM_ARRAY_TASK_COUNT}" \
    --processes-per-worker "${PROCESSES_PER_TASK}"'
NEWGRP_EOF
EOF

echo "Configured ${NUM_TASKS} tasks (max ${MAX_PARALLEL_TASKS} concurrent) x "\
  "${PROCESSES_PER_TASK} processes x ${CPUS_PER_PROCESS} CPUs = ${CPUS_PER_TASK} CPUs/task"
echo "Memory: ${MEM_PER_PROCESS_GB} GB/process = ${SLURM_MEM}/task"
if [[ -n "${SLURM_NODE_DIRECTIVE}" ]]; then
  echo "Placement: ${SLURM_NODE_DIRECTIVE}"
fi
echo "Submission script: ${submission_path}"
echo "Logs: ${job_dir}"
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY_RUN=true; not submitting."
  exit 0
fi

job_id=$(sbatch --parsable "${submission_path}")
job_id=${job_id%%;*}
echo "Launched job ${job_id}"
