#!/bin/bash
# Build the Megatron container for GB200 (aarch64) on Mistral infrastructure
#
# Usage:
#   ./build_gb200_container_on_mistral_infra.sh                    # Build docker image + import to enroot
#   ./build_gb200_container_on_mistral_infra.sh --step build       # Only build the Docker image
#   ./build_gb200_container_on_mistral_infra.sh --step enroot      # Only import to enroot (docker image must exist)
#   ./build_gb200_container_on_mistral_infra.sh --tag 260227v2     # Custom tag suffix
#
# Megatron-LM is NOT installed in the container. It is mounted and added to
# PYTHONPATH at runtime.
#
# Prerequisites:
#   - Run on a login node with access to an ARM docker service
#   - Set DOCKER_HOST to point to your ARM docker service

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${TAG:-$(date +%y%m%d)}"
DOCKERFILE="${SCRIPT_DIR}/gb200.Dockerfile"
IMAGE_NAME="megatron_image_gb200"
IMAGE_TAG="${IMAGE_NAME}:${TAG}"
CONTAINER_DIR="${CONTAINER_DIR:-/mnt/vast/shared/megatron/containers}"
SQSH_PATH="${CONTAINER_DIR}/${IMAGE_NAME}_${TAG}.sqsh"

# --- Parse arguments ---
STEP="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)
            STEP="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            IMAGE_TAG="${IMAGE_NAME}:${TAG}"
            SQSH_PATH="${CONTAINER_DIR}/${IMAGE_NAME}_${TAG}.sqsh"
            shift 2
            ;;
        --container-dir)
            CONTAINER_DIR="$2"
            SQSH_PATH="${CONTAINER_DIR}/${IMAGE_NAME}_${TAG}.sqsh"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--step build|enroot] [--tag TAG] [--container-dir DIR]"
            exit 1
            ;;
    esac
done

echo "=== GB200 Container Build (aarch64) ==="
echo "  Dockerfile:  ${DOCKERFILE}"
echo "  Image:       ${IMAGE_TAG}"
echo "  Sqsh:        ${SQSH_PATH}"
echo "  Step:        ${STEP}"
echo ""

# --- Step 1: Build Docker image ---
if [[ "$STEP" == "all" || "$STEP" == "build" ]]; then
    echo "=== Step 1: Building Docker image ==="

    if [[ -z "${DOCKER_HOST:-}" ]]; then
        echo "WARNING: DOCKER_HOST is not set. Set it to your ARM docker service."
        echo "Example: export DOCKER_HOST=your-arm-docker-service:2375"
        exit 1
    fi

    docker build \
        -f "${DOCKERFILE}" \
        -t "${IMAGE_TAG}" \
        --target base \
        "${SCRIPT_DIR}"

    echo "Docker image built: ${IMAGE_TAG}"
fi

# --- Step 2: Import to enroot as squashfs ---
if [[ "$STEP" == "all" || "$STEP" == "enroot" ]]; then
    echo "=== Step 2: Importing to enroot ==="

    mkdir -p "${CONTAINER_DIR}"

    # enroot on login node needs XDG_RUNTIME_DIR set
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/enroot_runtime_$(whoami)}"
    mkdir -p "${XDG_RUNTIME_DIR}"

    if [[ -f "${SQSH_PATH}" ]]; then
        echo "Removing existing ${SQSH_PATH}"
        rm -f "${SQSH_PATH}"
    fi

    enroot import -o "${SQSH_PATH}" "dockerd://${IMAGE_TAG}"

    echo "Squashfs created: ${SQSH_PATH}"
fi

echo ""
echo "=== Done ==="
echo "Container ready at: ${SQSH_PATH}"
echo ""
echo "Megatron-LM is added to PYTHONPATH at runtime via launch scripts."
