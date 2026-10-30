# Training image for the scaling-ladder hybrid Mamba + MoE recipes.
#
# Layers the extra components the recipes need on top of the NVIDIA PyTorch NGC
# base (which already provides CUDA, cuDNN, NCCL, PyTorch, and TransformerEngine):
# Mamba SSM kernels, HybridEP/DeepEP, NVRx, Emerging Optimizers, and assorted
# training/inference packages. The Megatron-LM checkout itself is NOT baked in:
# the recipes run
# `python ${REPO_DIR}/pretrain_hybrid.py`, so mount your worktree at run time
# (REPO_DIR=${ROOT_DIR}/code in each script).
#
# Build (from the repo root):
#   docker build -t scaling-ladder scaling_ladder
#
# Override the base if you need a different NGC release:
#   docker build --build-arg FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:25.06-py3 ...

# Tracks the repo's NGC pin (docker/.ngc_version.dev) at the time of writing.
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${FROM_IMAGE_NAME}

# Clone and build all third-party sources under /opt.
WORKDIR /opt

# Mamba dependencies.
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git && \
    cd causal-conv1d && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install -vvv --no-build-isolation .
RUN git clone https://github.com/state-spaces/mamba.git && \
    cd mamba && \
    MAMBA_FORCE_BUILD=TRUE pip install -vvv --no-build-isolation --no-deps .

# Install NVRx (NVIDIA Resiliency Extension).
RUN git clone https://github.com/NVIDIA/nvidia-resiliency-ext && \
    cd nvidia-resiliency-ext && \
    pip install .

# Install Emerging Optimizers.
RUN git clone --branch v0.2.0 https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git emerging_optimizers && \
    cd emerging_optimizers && \
    pip install .

# Install HybridEP (DeepEP hybrid-ep branch, built against a local rdma-core).
ENV RDMA_CORE_HOME=/opt/rdma-core/build
ENV HYBRID_EP_MULTINODE=1
RUN git clone https://github.com/linux-rdma/rdma-core.git && \
    cd rdma-core && git checkout tags/v60.0 && sh build.sh
RUN apt-get update && \
    apt-get install -y --no-install-recommends libnvidia-ml-dev && \
    git clone -b hybrid-ep https://github.com/deepseek-ai/DeepEP.git && \
    cd DeepEP && \
    TORCH_CUDA_ARCH_LIST="9.0 10.0" MAX_JOBS=8 pip install --no-build-isolation . && \
    apt-get purge -y libnvidia-ml-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install miscellaneous packages necessary for pre-training and inference.
RUN pip install boto3 einops flash-attn flask flask-restful sentencepiece tiktoken transformers wandb

# Install Mistral data loading dependencies.
# These are transitively imported when using Mistral's PretokenizedDataLoader
# inside Megatron. The mistral repo itself is added via PYTHONPATH at runtime.
COPY mistral_requirements.txt /tmp/mistral_requirements.txt
RUN pip install -r /tmp/mistral_requirements.txt && rm /tmp/mistral_requirements.txt

WORKDIR /workspace
