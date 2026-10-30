# =========================
# Build the image for GB200 (aarch64)
# =========================
# docker build -f gb200.Dockerfile -t megatron_image_gb200:260327 --target base --network host .

# =========================
# Use the pytorch image (newer base for built-in cudnn improvements)
# =========================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS base

ENV SHELL=/bin/bash

# =========================
# Install system packages
# =========================
RUN rm -rf /opt/megatron-lm && \
    apt-get update && \
    apt-get install -y sudo gdb bash-builtins git zsh autojump tmux curl gettext libfabric-dev && \
    wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_arm64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

# =========================
# Install Python packages
# =========================
# NOTE: `unset PIP_CONSTRAINT` to install packages that do not meet the default constraint in the base image.
# one-logger is skipped (requires NVIDIA internal artifactory access)
RUN unset PIP_CONSTRAINT && pip install debugpy dm-tree torch_tb_profiler einops wandb \
    sentencepiece tokenizers transformers torchvision ftfy modelcards datasets tqdm pydantic \
    nvidia-pytriton py-spy yapf darker \
    tiktoken flask-restful \
    nltk wrapt pytest pytest_asyncio pytest-cov pytest_mock pytest-random-order \
    black==24.4.2 isort==5.13.2 flake8==7.1.0 pylint==3.2.6 coverage mypy \
    setuptools==69.5.1

# NOTE: CuDNN is already included in the base pytorch 25.09 container (cuda-13 variant).
# Do NOT install libcudnn9-cuda-12 as it removes the working cuda-13 version and breaks TE.
# For mxfp8/cuDNN upgrades on GB200, you can optionally install libcudnn9-cuda-13 via apt.

# =========================
# Install Mistral data loading dependencies
# =========================
# These are transitively imported when using Mistral's PretokenizedDataLoader
# inside Megatron. The mistral repo itself is added via PYTHONPATH at runtime.
COPY mistral_requirements.txt /tmp/mistral_requirements.txt
RUN unset PIP_CONSTRAINT && pip install -r /tmp/mistral_requirements.txt && rm /tmp/mistral_requirements.txt

# NOTE: TransformerEngine is already included in the base pytorch container.
# NOTE: DeepEP is not installed here; install it manually in the container if needed.

# =========================
# Clean cache
# =========================
RUN rm -rf /root/.cache /tmp/*

# =========================
# Change the workspace
# =========================
WORKDIR /home/
