# Using the Multimodal Pretokenized Dataloader on NVIDIA

This note is for bringing up multimodal training with the Mistral pretokenized dataloader on NVIDIA machines. In this code, epoch indexes are the prepacking indexes: they define which document chunks and image-token spans are packed into each fixed-length training sequence.

## Debug Bring-Up

1. Read `mistral_fork/README.md` first for the overall concepts: pretokenization, prepacking / epoch-index building, sampling, image metadata, and why image transforms must match between prepacking and runtime.

2. Follow `mistral_adapter/testing_dataloader/README.md` for the concrete smoke-test commands. Start by copying the Mistral config to an NVIDIA-local config, for example:

```bash
cp mistral_adapter/example_configs/nvidia_data_sharing_1t_4096_MISTRAL.yaml \
  mistral_adapter/example_configs/nvidia_data_sharing_1t_4096_NVIDIA.yaml
```

3. Copy the raw JSONL files, image tar data, and pretokenized data from S3 to NVIDIA-local storage. Then edit the `_NVIDIA.yaml` so every `prefix_path` points to the NVIDIA-local pretokenized data root.

4. Build a small debug prepacking cache (= epoch indexes) with the debug parameters from `mistral_adapter/testing_dataloader/README.md`, for example `--percentage-data-multiplier 0.0001`, a temporary `--epoch-index-cache-path`, `--num_total_tokens 20480`, `--worker_id 0`, and `--num_workers 1`. See `mistral_fork/README.md` for the meaning of `percentage_data`.

5. Run `mistral_adapter.testing_dataloader.test_pretokenized_config_dataloader` with the same `_NVIDIA.yaml`, percentage multiplier, temporary epoch-index cache path, and a few batches.

⚠️ For multimodal NVIDIA-side tests, image paths will need a runtime path hotfix. The pretokenized metadata can still contain hardcoded Mistral image paths, even after `prefix_path` has been changed in the YAML. If image loading fails, map the metadata image paths to the NVIDIA-local image tar location or regenerate the metadata with NVIDIA paths.

## Real Training

1. Finalize the NVIDIA pretokenized config. The important fields are the source `prefix_path` values, `epoch_index_cache_path`, and `skip_images: false` for multimodal training.

2. Prebuild the full epoch indexes / prepacking indexes using the same config that training will use.

⚠️ Full epoch-index building is slow. Distribute it across SLURM or any job framework by splitting workers with `--worker_id` and `--num_workers`. Keep the config, sequence length, seed, `percentage_data`, and image transform settings fixed; these fields are part of the epoch-index cache identity.

3. Launch Megatron training with `--pretokenized-config` pointing to the same `_NVIDIA.yaml` used for prebuilding epoch indexes. Do not switch config files between prebuilding and training unless you intentionally rebuild the indexes.

4. Set `original_micro_batch_size` in the `_NVIDIA.yaml` to the real microbatch size you want for training, and set `original_seq_len` to the read sequence length. With interdocument masking, Megatron must launch with `--micro-batch-size 1` and `--seq-length original_micro_batch_size * original_seq_len`; the real microbatch shape therefore has to live in the pretokenized config.
