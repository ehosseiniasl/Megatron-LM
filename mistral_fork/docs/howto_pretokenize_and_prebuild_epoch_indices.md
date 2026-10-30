# Pretokenize a JSONL mixture and prebuild epoch indexes

This guide uses the complete CrabMo v1 NVIDIA mixture as a concrete example:

- JSONL config: `mistral_fork/local_data/configs/jsonl_pretokenization/crabmo_v1_nvidia_260826v0_NO_WEIGHTS.yaml`
- Runtime config: `mistral_adapter/example_configs/crabmo_v1_nvidia_260826v0_megatron_pretokenized_v9_NO_WEIGHTS.yaml`
- Pretokenized data root: `/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/pretokenized_data`
- Epoch-index root: `/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/epoch_index`

Run every command from the repository root on the Coalition cluster.

## 1. Setup and venv

See [Annex A](#annex-a-setup-and-venv).

## 2. Validate the JSONL config

Each `jsonl_dir` is one logical datasource and must contain chunked `*.jsonl`
files directly. Do not point multiple config entries at the same directory, and
do not use a parent directory containing several logical datasources.

Each source must explicitly define its sample type, filter, masking strategy,
image transform, rare-token filter, and PDS filter. The config loader validates
these combinations and rejects duplicate `Path(str(jsonl_dir))` values.

The launch command performs this validation before submitting work. To validate
without submitting, load the config directly:

```bash
python - <<'PY'
from pathlib import Path

from mistral_fork.mistral.data.pretokenized.processing.config import (
    load_jsonl_pretokenization_config,
)

path = Path(
    "mistral_fork/local_data/configs/jsonl_pretokenization/"
    "crabmo_v1_nvidia_260826v0_NO_WEIGHTS.yaml"
)
config = load_jsonl_pretokenization_config(path)
print(f"Validated {len(config.sources)} unique sources")
PY
```

### Update PDS or rare-token filtering

Filter artifacts are immutable once used. **Do not modify an existing PDS
manifest or rare-token file in place.** Doing so would silently change the
meaning of an existing cache path.

To introduce an updated filter:

1. Create a new versioned filter artifact.
2. Add a new `PdsFilter` or `RareTokenFilter` enum entry in
   `mistral_fork/mistral/data/pretokenized/document_filters.py`, then map that
   entry to the new artifact in `PDS_FILTER_PATHS` or
   `RARE_TOKEN_FILTER_PATHS`.
3. Update the corresponding tokenizer policy in
   `mistral_fork/mistral/data/pretokenized/cache_key.py`.
4. Set the new enum value explicitly on each affected JSONL source:

   ```yaml
   rare_token_filter: rare_tokens_n4_v6_262k_v2
   pds_filter: nvidia_pds_quarantine_v2
   ```

5. Reload the JSONL config as shown above so validation catches incompatible
   tokenizer, filter-profile, and sample-type combinations.

**Warning:** changing either enum value changes the cache key and
repretokenizes every affected source into a different output path. Clean the
obsolete pretokenized version after identifying the old path precisely and
confirming that the replacement is complete.

## 3. Pretokenize with 2,048 CPUs

This submits 128 shards with 16 CPUs each: `128 * 16 = 2,048` CPUs at full
concurrency.

```bash
NUM_SHARDS=128 \
MAX_PARALLEL_SHARDS=128 \
CPUS_PER_SHARD=16 \
SLURM_PARTITION=cpu \
SLURM_QOS=cpu-free \
JOB_NAME=pretok_crabmo_v1_nvidia_260826v0_NO_WEIGHTS \
./mistral_fork/mistral/data/pretokenized/processing/run_pretok.sh \
  mistral_fork/local_data/configs/jsonl_pretokenization/crabmo_v1_nvidia_260826v0_NO_WEIGHTS.yaml \
  pretok
```

Wait for every shard to finish successfully before merging.

## 4. Merge with 2,048 CPUs

This submits 64 merge shards with 32 CPUs each: `64 * 32 = 2,048` CPUs at full
concurrency.

```bash
MERGE_NUM_SHARDS=64 \
MERGE_MAX_PARALLEL_SHARDS=64 \
MERGE_CPUS_PER_SHARD=32 \
SLURM_PARTITION=cpu \
SLURM_QOS=cpu-free \
MERGE_JOB_NAME=merge_crabmo_v1_nvidia_260826v0_NO_WEIGHTS \
./mistral_fork/mistral/data/pretokenized/processing/run_pretok.sh \
  mistral_fork/local_data/configs/jsonl_pretokenization/crabmo_v1_nvidia_260826v0_NO_WEIGHTS.yaml \
  merge
```

Wait for every merge shard to finish successfully before generating the
runtime config.

## 5. Repeat-aware training

Set `percentage_data` explicitly for every source in the runtime training
config. Use the same values for a proxy run and the real run so each source has
the same number of repeats in both runs.

## 6. Generate the Megatron runtime config

The generator fails if an expected merged datasource is missing. The flags
below reproduce the checked-in CrabMo runtime config.

```bash
python -m mistral_fork.mistral.data.pretokenized.processing.config \
  --config mistral_fork/local_data/configs/jsonl_pretokenization/crabmo_v1_nvidia_260826v0_NO_WEIGHTS.yaml \
  --output mistral_adapter/example_configs/crabmo_v1_nvidia_260826v0_megatron_pretokenized_v9_NO_WEIGHTS.yaml \
  --pretokenized-data-dir /scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/pretokenized_data \
  --epoch-index-cache-path /scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/epoch_index \
  --no-interdocument-masking \
  --no-skip-images
```

Review the generated diff. Existing sources and global settings should remain
unchanged unless the mixture change intentionally modifies them.

## 7. Prebuild epoch indexes with 2,048 CPUs

Each task launches four processes with four CPUs per process. With 128 tasks
running concurrently, the reservation is exactly
`128 * 4 * 4 = 2,048` CPUs.

Set the horizon to the total number of tokens the training run will consume:

```bash
export TRAINING_HORIZON_IN_TOKENS=REPLACE_WITH_INTEGER_TOKEN_HORIZON
```

```bash
NUM_TASKS=128 \
MAX_PARALLEL_TASKS=128 \
PROCESSES_PER_TASK=4 \
CPUS_PER_PROCESS=4 \
IMAGE_TRANSFORM_WORKERS=4 \
SLURM_PARTITION=cpu \
SLURM_QOS=cpu-free \
JOB_NAME=epoch_index_crabmo_v1_nvidia_260826v0_NO_WEIGHTS \
bash mistral_fork/mistral/data/pretokenized/processing/run_prebuild_epoch_indexes.sh \
  mistral_adapter/example_configs/crabmo_v1_nvidia_260826v0_megatron_pretokenized_v9_NO_WEIGHTS.yaml \
  "${TRAINING_HORIZON_IN_TOKENS}"
```

The final argument is the total token budget used to determine how many epoch
indexes to build.

### Warnings for production prebuilds

1. **Use the final training weights.** The builder derives the required number
   of epochs per source from the mixture weights and training horizon. If every
   source still has `weight: 1.0`, the prebuild will not produce the epoch
   counts required by the real training mixture.
2. **Epochs are sequential by default.** Without an explicit override, each
   source builds epoch 0, then epoch 1, and so on until it covers its share of
   the training horizon. This can leave a long-running source on the critical
   path.
3. **Use explicit parallel epoch counts for a large production build.** Add
   `n_epoches_to_prebuild` to each runtime-config source, set to the required
   number of epochs plus one safety epoch. For example:

   ```yaml
   - prefix_path: /path/to/merged/source/chunks_00000-00128
     weight: 0.25
     percentage_data: 1.0
     sample_type: pretrain
     image_transform_type: null
     n_epoches_to_prebuild: 4  # Three required epochs plus one safety epoch.
   ```

   Explicit epoch counts let different epochs enter the distributed worker
   pool independently. With enough concurrent CPU capacity, the wall time is
   then bounded approximately by the slowest single source epoch. For CrabMo,
   plan for roughly five hours based on the observed worst-case one-epoch
   prebuild time.

Do not start training until every array task has completed successfully.

## Annex A: Setup and venv

Prepare the checkout from the repository root on the Coalition cluster:

```bash
newgrp nemotron_coalition_mistral
git switch main
git pull --ff-only
```

Create the venv once from the repository root on a Lustre-backed checkout:

```bash
PRETOK_VENV="$(pwd)/.venv-pretok-launch"
python3.12 -m venv "${PRETOK_VENV}"
"${PRETOK_VENV}/bin/python" -m pip install --upgrade pip
"${PRETOK_VENV}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch
"${PRETOK_VENV}/bin/python" -m pip install \
  PyYAML \
  pydantic \
  -r mistral_adapter/containers/mistral_requirements.txt
source "${PRETOK_VENV}/bin/activate"
```

To reuse an existing launch venv instead, activate it explicitly:

```bash
source /path/to/.venv-pretok-launch/bin/activate
```

Verify the three login-node entry points before submitting anything:

```bash
python -m mistral_fork.mistral.data.pretokenized.processing.pretokenize --help >/dev/null
python -m mistral_fork.mistral.data.pretokenized.processing.merge --help >/dev/null
python -m mistral_fork.mistral.data.pretokenized.processing.config --help >/dev/null
```

The CPU-only PyTorch build is sufficient on the login node. Submitted pretok,
merge, and epoch-index tasks use the container configured by their launch
scripts; this venv is only for validation and Slurm submission.
