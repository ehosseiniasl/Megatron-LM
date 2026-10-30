#!/usr/bin/env bash

# Shared helpers for nvbench sbatch launch scripts.
# Usage:
#   source "${LME_SRC_DIR}/nvbench/sbatch_utils.sh"
#
# Typical sbatch structure:
#   1. Set config vars (MODEL_NAME, CHECKPOINT_DIR, IMAGE, etc.)
#   2. source sbatch_utils.sh
#   3. nvbench_derive_ports          # → HTTP_PORT, MASTER_ADDR, MASTER_PORT
#   4. srun ... nemotron-serve ... & # launch server
#   5. nvbench_wait_for_server ...   # poll until endpoint is reachable
#   6. srun ... launch_evals.py ...

# ---------------------------------------------------------------------------
# Port derivation
# ---------------------------------------------------------------------------

# Deterministic ports from SLURM_JOB_ID + SLURM_ARRAY_TASK_ID so concurrent
# jobs/array tasks don't collide.
# Sets: HTTP_PORT, MASTER_PORT, MASTER_ADDR
nvbench_derive_ports() {
    local port_base=${1:-42000}
    local port_window=${2:-2000}
    local job_hash=${SLURM_JOB_ID:-0}
    local array_hash=${SLURM_ARRAY_TASK_ID:-0}
    local offset=$(((job_hash + array_hash * 131) % port_window))
    HTTP_PORT=$((port_base + offset))
    MASTER_PORT=${MASTER_PORT:-$((RANDOM % 10000 + 20000))}
    MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"
    export HTTP_PORT MASTER_PORT MASTER_ADDR
    echo "Ports: HTTP=${HTTP_PORT} MASTER=${MASTER_PORT} ADDR=${MASTER_ADDR}"
}

# ---------------------------------------------------------------------------
# Server readiness
# ---------------------------------------------------------------------------

check_endpoint_socket() {
    local endpoint="$1"
    [[ -z "$endpoint" ]] && return 0
    local stripped=${endpoint#http://}
    local host=${stripped%%:*}
    local port=${stripped##*:}
    timeout 3 bash -c "exec 3<>/dev/tcp/${host}/${port}" >/dev/null 2>&1
}

nvbench_check_server_step_alive_or_fail() {
    local pid="${NVBENCH_SERVER_STEP_PID:-}"
    [[ -z "${pid}" ]] && return 0
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
        local rc=1
        if wait "${pid}" 2>/dev/null; then rc=0; else rc=$?; fi
        echo "Error: background server step (pid=${pid}) is no longer running (exit=${rc})." >&2
        return "${rc:-1}"
    fi
}

# Block until HTTP endpoint is reachable. Fails fast if the background server
# srun (NVBENCH_SERVER_STEP_PID) exits. If NO_EVAL=1, exits after server is up.
#   nvbench_wait_for_server <endpoint> [timeout_minutes]
nvbench_wait_for_server() {
    local endpoint="$1"
    local timeout_min=${2:-60}
    echo "Waiting for ${endpoint} (timeout=${timeout_min}m)..."
    local deadline=$(( $(date +%s) + timeout_min * 60 ))
    while true; do
        if ! nvbench_check_server_step_alive_or_fail; then exit 1; fi
        if check_endpoint_socket "${endpoint}"; then break; fi
        if (( $(date +%s) >= deadline )); then
            echo "Error: ${endpoint} not reachable after ${timeout_min} minutes." >&2
            exit 1
        fi
        sleep 5
    done
    echo "Server is ready."
    if [[ "${NO_EVAL:-0}" == "1" ]]; then
        echo "NO_EVAL=1: skipping evaluation. Exiting."
        sleep 5; exit 0
    fi
}
