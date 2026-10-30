#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

PYTHON=${PYTHON:-/lustre/fsw/portfolios/nemotron/users/thib.ma/code/nv-mistralai-megatron/.venv-pretok-launch/bin/python}
CONFIG=${CONFIG:-mistral_fork/local_data/configs/jsonl_pretokenization/old/60T-Aug04_mistral_dataloader_v0_JSONL.yaml}
OUTPUT=${OUTPUT:-mistral_adapter/example_configs/old/60T-Aug04_mistral_dataloader_v0_text_only.yaml}
PRETOKENIZED_DATA_DIR=${PRETOKENIZED_DATA_DIR:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/pretokenized_data}
EPOCH_INDEX_CACHE_PATH=${EPOCH_INDEX_CACHE_PATH:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/epoch_index}
ORIGINAL_SEQ_LEN=${ORIGINAL_SEQ_LEN:-8192}
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-1099511627776}

commit=$(git rev-parse HEAD)
tmp_output=$(mktemp "${OUTPUT}.tmp.XXXXXX")
trap 'rm -f "$tmp_output"' EXIT

cmd=(
  "$PYTHON"
  -m mistral_fork.mistral.data.pretokenized.processing.config
  --config "$CONFIG"
  --output "$tmp_output"
  --pretokenized-data-dir "$PRETOKENIZED_DATA_DIR"
  --epoch-index-cache-path "$EPOCH_INDEX_CACHE_PATH"
  --original-seq-len "$ORIGINAL_SEQ_LEN"
  --max-group-size "$MAX_GROUP_SIZE"
  --skip-images
)

"${cmd[@]}"

{
  printf '# Generated from commit: %s\n' "$commit"
  printf '# Command used to generate this file:\n'
  printf '#   '
  printf '%q ' "${cmd[@]/$tmp_output/$OUTPUT}"
  printf '\n'
  printf '# Text-only config: generated with --skip-images, so skip_images must stay true.\n'
  printf '# With interdocument_masking: true, Megatron'\''s --seq-length must match\n'
  printf '# original_seq_len (%s). --micro-batch-size is a launch-time setting.\n' "$ORIGINAL_SEQ_LEN"
  cat "$tmp_output"
} > "$OUTPUT"

echo "Wrote $OUTPUT"
