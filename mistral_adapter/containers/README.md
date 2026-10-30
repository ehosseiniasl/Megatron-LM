# Megatron Container Build (GB200)

## Building the Container

The build script produces a squashfs container ready for `srun --container-image`.

### GB200 (aarch64)

```bash
cd mistral_fork/containers/

# Set DOCKER_HOST to your ARM docker service
export DOCKER_HOST=your-arm-docker-service:2375

./build_gb200_container_on_mistral_infra.sh                    # full build (docker + enroot)
./build_gb200_container_on_mistral_infra.sh --step build       # docker build only
./build_gb200_container_on_mistral_infra.sh --step enroot      # enroot import only (image must exist)
./build_gb200_container_on_mistral_infra.sh --tag 260227v2     # custom tag
./build_gb200_container_on_mistral_infra.sh --container-dir /custom/path  # custom output directory
```

- Dockerfile: `gb200.Dockerfile` (arm64)
- Output: `<container-dir>/megatron_image_gb200_<tag>.sqsh`

## Container Architecture

```
Base: nvcr.io/nvidia/pytorch:26.01-py3
  + system packages (git, zsh, tmux, curl, yq, libfabric-dev)
  + python packages (wandb, einops, transformers, pytest, etc.)
  + mistral_requirements.txt (boto3, fire, safetensors, msgspec, etc.)
  + TransformerEngine (pre-installed in base, NOT rebuilt)
  + CuDNN (pre-installed in base, NOT upgraded)
```

Megatron-LM is **not** baked into the container. It is mounted and added to
`PYTHONPATH` at runtime.

## Known Issues

- **CuDNN conflict**: Do NOT install `libcudnn9-cuda-12` via apt. It removes the
  working `libcudnn9-cuda-13` from the base container and breaks TransformerEngine.
- **Enroot scheme**: Use `dockerd://` for enroot import (not `docker-archive://`).
- **Enroot permissions**: On login nodes, set `XDG_RUNTIME_DIR=/tmp/enroot_runtime_$USER`.
