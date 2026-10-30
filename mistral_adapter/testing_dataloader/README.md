# Testing the pretokenized dataloader on the shared dataset

Context: we created a mixture from the shared datasource. The goal here is only
to check that Megatron can load a few batches from the pretokenized data.

For the dataloader format and cache layout, see `mistral_fork/README.md`.

For the NVIDIA-side multimodal bring-up checklist, see
`mistral_fork/docs/NVIDIA_MULTIMODAL.md`.

For smoke tests, keep the checked-in full-mixture YAML unchanged and pass the
small runtime knobs to both the epoch-index prebuild command and the dataloader
test command:

- `--percentage-data-multiplier 0.0001`
- `--epoch-index-cache-path /mnt/pretokenized/nvidia_pretokenized/epoch_indexes_DEBUG`

On NVIDIA machines, replace `/mnt/pretokenized/nvidia_pretokenized/epoch_indexes_DEBUG` with
an epoch-index cache path that matches an available writable disk on that
system.

## Multimodal NVIDIA Data Sharing Test

Use this section for the multimodal config with images:

`mistral_adapter/example_configs/nvidia_data_sharing_1t_4096_MISTRAL.yaml`

### Mistral Paths

1. Make sure your venv or container includes
   `mistral_adapter/containers/mistral_requirements.txt`.
2. Prebuild the prepacking cache (= epoch indexes):

```bash
python -u -m mistral_fork.mistral.data.pretokenized.processing.prebuild_epoch_indexes_megatron \
  prebuild \
  --config mistral_adapter/example_configs/nvidia_data_sharing_1t_4096_MISTRAL.yaml \
  --percentage-data-multiplier 0.0001 \
  --epoch-index-cache-path /mnt/pretokenized/nvidia_pretokenized/epoch_indexes_DEBUG \
  --num_total_tokens 20480 \
  --worker_id 0 \
  --num_workers 1
```

3. Check that the dataloader can read a few batches:

```bash
python -u -m mistral_adapter.testing_dataloader.test_pretokenized_config_dataloader \
  --pretokenized-config mistral_adapter/example_configs/nvidia_data_sharing_1t_4096_MISTRAL.yaml \
  --percentage-data-multiplier 0.0001 \
  --epoch-index-cache-path /mnt/pretokenized/nvidia_pretokenized/epoch_indexes_DEBUG \
  --num-total-tokens 20480 \
  --num-batches 5 \
  --no-allow-build-epoch-index
```

### NVIDIA Paths

1. Download pretokenized datasets from
   `s3:nvidia-mistral/mistral-data/pretokenized/pretokenized_data/` to
   `$NV_PRETOKENIZED_DATA`.
2. Copy
   `mistral_adapter/example_configs/nvidia_data_sharing_1t_4096_MISTRAL.yaml`
   to
   `mistral_adapter/example_configs/nvidia_data_sharing_1t_4096_NVIDIA.yaml`.
3. In `mistral_adapter/example_configs/nvidia_data_sharing_1t_4096_NVIDIA.yaml`,
   replace the `/mnt/pretokenized/pretokenized_data` prefix in `prefix_path`
   values with `$NV_PRETOKENIZED_DATA`.
4. Choose an NVIDIA-writable epoch-index cache path that matches the local disk
   layout, and pass it via `--epoch-index-cache-path` in both commands below.
   The example uses `$NV_EPOCH_INDEX_CACHE`.
5. Make sure your venv or container includes
   `mistral_adapter/containers/mistral_requirements.txt`.
6. Prebuild the prepacking cache (= epoch indexes):

```bash
python -u -m mistral_fork.mistral.data.pretokenized.processing.prebuild_epoch_indexes_megatron \
  prebuild \
  --config mistral_adapter/example_configs/nvidia_data_sharing_1t_4096_NVIDIA.yaml \
  --percentage-data-multiplier 0.0001 \
  --epoch-index-cache-path "$NV_EPOCH_INDEX_CACHE" \
  --num_total_tokens 20480 \
  --worker_id 0 \
  --num_workers 1
```

7. Check that the dataloader can read a few batches:

```bash
python -u -m mistral_adapter.testing_dataloader.test_pretokenized_config_dataloader \
  --pretokenized-config mistral_adapter/example_configs/nvidia_data_sharing_1t_4096_NVIDIA.yaml \
  --percentage-data-multiplier 0.0001 \
  --epoch-index-cache-path "$NV_EPOCH_INDEX_CACHE" \
  --num-total-tokens 20480 \
  --num-batches 5 \
  --no-allow-build-epoch-index
```

Known caveat: multimodal NVIDIA-side runs are expected to fail until image paths
are remapped. The pretokenized metadata still points to Mistral paths, so
NVIDIA-side runs need a small path-mapping hotfix or regenerated metadata.

## Temporary Text-Only NVIDIA Data Sharing Test

Use this section for the temporary text-only config:

`mistral_adapter/example_configs/old/tmp_nvidia_data_sharing_1t_4096_text_only_MISTRAL.yaml`

This config keeps the full-mixture source weights, but `skip_images: true`
causes multimodal sources to be read as text-only. The file is prefixed with
`tmp_` because source weights were set for multimodal training and skipping
images will probably change the effective weights.

### Mistral Paths

1. Make sure your venv or container includes
   `mistral_adapter/containers/mistral_requirements.txt`.
2. Prebuild the prepacking cache (= epoch indexes):

```bash
python -u -m mistral_fork.mistral.data.pretokenized.processing.prebuild_epoch_indexes_megatron \
  prebuild \
  --config mistral_adapter/example_configs/old/tmp_nvidia_data_sharing_1t_4096_text_only_MISTRAL.yaml \
  --percentage-data-multiplier 0.0001 \
  --epoch-index-cache-path /mnt/pretokenized/nvidia_pretokenized/epoch_indexes_DEBUG \
  --num_total_tokens 20480 \
  --worker_id 0 \
  --num_workers 1
```

3. Check that the dataloader can read a few batches:

```bash
python -u -m mistral_adapter.testing_dataloader.test_pretokenized_config_dataloader \
  --pretokenized-config mistral_adapter/example_configs/old/tmp_nvidia_data_sharing_1t_4096_text_only_MISTRAL.yaml \
  --percentage-data-multiplier 0.0001 \
  --epoch-index-cache-path /mnt/pretokenized/nvidia_pretokenized/epoch_indexes_DEBUG \
  --num-total-tokens 20480 \
  --num-batches 5 \
  --no-allow-build-epoch-index
```

### NVIDIA Paths

1. Download pretokenized datasets from
   `s3:nvidia-mistral/mistral-data/pretokenized/pretokenized_data/` to
   `$NV_PRETOKENIZED_DATA`.
2. Copy
   `mistral_adapter/example_configs/old/tmp_nvidia_data_sharing_1t_4096_text_only_MISTRAL.yaml`
   to
   `mistral_adapter/example_configs/tmp_nvidia_data_sharing_1t_4096_text_only_NVIDIA.yaml`.
3. In
   `mistral_adapter/example_configs/tmp_nvidia_data_sharing_1t_4096_text_only_NVIDIA.yaml`,
   replace the `/mnt/pretokenized/pretokenized_data` prefix in `prefix_path`
   values with `$NV_PRETOKENIZED_DATA`.
4. Choose an NVIDIA-writable epoch-index cache path that matches the local disk
   layout, and pass it via `--epoch-index-cache-path` in both commands below.
   The example uses `$NV_EPOCH_INDEX_CACHE`.
5. Make sure your venv or container includes
   `mistral_adapter/containers/mistral_requirements.txt`.
6. Prebuild the prepacking cache (= epoch indexes):

```bash
python -u -m mistral_fork.mistral.data.pretokenized.processing.prebuild_epoch_indexes_megatron \
  prebuild \
  --config mistral_adapter/example_configs/tmp_nvidia_data_sharing_1t_4096_text_only_NVIDIA.yaml \
  --percentage-data-multiplier 0.0001 \
  --epoch-index-cache-path "$NV_EPOCH_INDEX_CACHE" \
  --num_total_tokens 20480 \
  --worker_id 0 \
  --num_workers 1
```

7. Check that the dataloader can read a few batches:

```bash
python -u -m mistral_adapter.testing_dataloader.test_pretokenized_config_dataloader \
  --pretokenized-config mistral_adapter/example_configs/tmp_nvidia_data_sharing_1t_4096_text_only_NVIDIA.yaml \
  --percentage-data-multiplier 0.0001 \
  --epoch-index-cache-path "$NV_EPOCH_INDEX_CACHE" \
  --num-total-tokens 20480 \
  --num-batches 5 \
  --no-allow-build-epoch-index
```
