#!/usr/bin/env bash
set -euo pipefail

if ! id -Gn | tr ' ' '\n' | grep -qx nemotron_coalition_mistral; then
  echo "Run this from a nemotron_coalition_mistral group shell first:"
  echo "  newgrp nemotron_coalition_mistral"
  exit 1
fi

WORKDIR=${WORKDIR:-/lustre/fsw/portfolios/nemotron/users/thib.ma/code/nv-mistralai-megatron_pr206_config_check}
PYTHON=${PYTHON:-/lustre/fsw/portfolios/nemotron/users/thib.ma/code/nv-mistralai-megatron/.venv-pretok-launch/bin/python}
CONFIG=${CONFIG:-mistral_scratch/thib/260805_relaunch_missing_sources/missing_nvidia_crawl_dedup_sources.yaml}
PRETOKENIZED_DATA_DIR=${PRETOKENIZED_DATA_DIR:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/pretokenized_data}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nemotron/users/szymon.ma/images/nvidia+pytorch+26.04-py3+dependencies+mamba.sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-${HOME}:${HOME},/scratch:/scratch,/lustre:/lustre}

cd "$WORKDIR"

"$PYTHON" -m mistral_fork.mistral.data.pretokenized.processing.pretokenize distributed_process \
  --config "$CONFIG" \
  --pretokenized-data-dir "$PRETOKENIZED_DATA_DIR" \
  --num-shards 32 \
  --max-parallel-shards 32 \
  --cpus-per-shard 16 \
  --slurm-account nemotron_coalition_mistral \
  --slurm-partition cpu \
  --slurm-qos cpu-free \
  --slurm-time 48:00:00 \
  --group nemotron_coalition_mistral \
  --job-name pretok_nvidia_crawl_dedup_7src_32shards_16cpu \
  --workdir "$WORKDIR" \
  --container-image "$CONTAINER_IMAGE" \
  --container-name nemo_pytorch_2604_mamba_user \
  --no-container-remap-root \
  --container-mounts "$CONTAINER_MOUNTS" \
  --container-env HOME,LANG,LANGUAGE,LC_ALL
