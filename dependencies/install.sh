#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Installs the dependency set used for final model training runs, on top of an
# NGC PyTorch base image. Run this when you already have a container or a bare
# machine and do not want to build an image:
#
#     bash dependencies/install.sh
#
# Packages go into the interpreter that `pip` resolves to, which on the NGC base
# image is the system Python. This is deliberately not the uv-managed /opt/venv
# that the CI images use; see docker/Dockerfile.ci.dev for that path.
#
# dependencies/Dockerfile.training installs the same dependency set and must
# stay in sync with this file. tools/check_dependency_sync.py compares them, and
# the sections below are the units it compares. When you add, remove, or
# bump a dependency, change both files.

set -euo pipefail

# BEGIN DEPENDENCY PINS
# Bump this whenever any pin below changes. The build stamps it into the
# image, and install_sbatch.sh names the output squashfs after it.
DEPENDENCY_SET_VERSION="18"
# A ref may be a branch, a tag or a commit.
# Local artifacts rather than anything fetchable, so these name files, not
# refs. The filename carries the version; WHEEL_DIR says where to look and is
# set per route, since a Docker build cannot read the same paths a Slurm one
# can. Both are expected beside these scripts and are not committed.
CUTLASS_DSL_INTERNAL_WHEEL="nvidia_cutlass_dsl_internal-0.3.0+20260719043000.ce0cac0-cp312-cp312-manylinux_2_28_aarch64.whl"
GDP_ATTN_FUSED_WHEEL="gdp_attn_fused-0.5.5+cp.dkg43d3c5b-py3-none-any.whl"
CPYTHON_REPO="https://github.com/python/cpython.git"
CPYTHON_REF="v3.12.3"
# Upstream gh-142183 landed after 3.12.3 and is applied as a backport below.
CPYTHON_PATCH_COMMIT="706fd4ec08acbf1b1def3630017ebe55d224adfa"
# The internal fork, not upstream. This branch is the v2.31.2-1 tag the previous
# version pinned plus the Inspector profiler plugin and the device-side telemetry
# it reads, neither of which is upstream. Cloned over HTTPS rather than SSH,
# unlike the TransformerEngine fork below: the project is internal-visible, so
# this needs no key and no token, which a CI runner would otherwise have to be
# given.
NCCL_REPO="https://gitlab-master.nvidia.com/nccl/nccl.git"
# Head of sirshakd/v2.31.2-profiler-lossy-ts-v8-init-time as of 09/02/2026,
# branched from v2.31.2-1. A commit rather than the branch name, so the branch
# moving cannot change this version.
NCCL_REF="78b39cdb6bcd1568f7cebde80878606655cf06b1"
# What the built library has to report. A commit ref carries no version, so
# verify-pins cannot read one out of NCCL_REF the way it could out of a tag.
NCCL_EXPECTED_VERSION="2.31.2"
CAUSAL_CONV1D_REPO="https://github.com/Dao-AILab/causal-conv1d.git"
# Pinned to main as of 08/25/2026; causal-conv1d cuts no releases.
CAUSAL_CONV1D_REF="cd81f0413cad2fc1e6f17e785ac39f59aae690cd"
MAMBA_REPO="https://github.com/state-spaces/mamba.git"
# Pinned to main as of 08/25/2026; mamba cuts no releases.
MAMBA_REF="e9594ce1c732d97440f0332fdc43170a2294dbfa"
NVIDIA_RESILIENCY_EXT_REPO="https://github.com/NVIDIA/nvidia-resiliency-ext.git"
# Head of NVRx stable as of 10/21/2026. It carries PR #406, the
# Lens integration, a native writer for checkpoint .metadata files, reuse of
# cached save plans and metadata, profiling spans around the finalize
# functions, and write results gathered to the coordinator as tables rather
# than pickled objects. It includes PR #459: selectable CUDA-GDB batch/stdin
# captures, concise timeout diagnostics, and one-poll upload batching. PR #453
# removes the deprecated in-process restart package; ft_launcher is retained.
NVIDIA_RESILIENCY_EXT_REF="ba5cfb48058059d682d248d641cb95a4f3fab618"
NEMO_LENS_REPO="https://github.com/NVIDIA-NeMo/Lens.git"
NEMO_LENS_REF="b5f941f373eab88afeb8aa26a64a34cbf32b4365"
OTELCOL_VERSION="0.156.0"
OTELCOL_URL="https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.156.0/otelcol-contrib_0.156.0_linux_arm64.tar.gz"
OTELCOL_SHA256="1f9afe1d245b4babbb4bcb7d6b57ba2836b3b23c5f61b38abc00ab461f049288"
CUDNN_FRONTEND_REPO="ssh://git@gitlab-master.nvidia.com:12051/ADLR/cudnn_frontend.git"
# The nemotron4 branch of the internal fork, not upstream. It is upstream
# c012ba87 with three commits on top. The first cherry-picks #906, which cuts the
# instruction count of the tanh soft clamp srelu and dsrelu grouped GEMM
# epilogues. Version 9 took #906 by moving 47 commits along develop instead,
# which dragged in a required nvidia-cutlass-dsl>=4.6.2 and pushed that package
# off the 4.6.0 the flash-attn-4 section settles on. Cherry-picking leaves the
# dependency alone. c012ba87 is still wanted for #858, which adds
# tanh_clamp_scale to those wrappers; TransformerEngine below refuses to build a
# soft-clamped squared ReLU without it, and verify-pins checks the signature
# rather than the version, since this declares the same 1.28.0 the release does.
# The second rounds the srelu pre-activation to the saved C dtype before the
# forward activation, so the forward pass and the backward recomputation from
# the saved C produce the same FC2 input bytes and scales. The third takes the
# kernel half of upstream #1283: the block-scaled grouped GEMM wgrad kernel
# fences each TMA descriptor its helper wrote before using it, so reusing the
# descriptor workspace back to back, as CUDA Graph replay does, cannot read a
# stale cached descriptor.
CUDNN_FRONTEND_REF="7d7c071a1b5f41d2abf875d50cb0659c21d54183"
# The nemotron4 branch of the internal fork, not upstream. It carries
# unmerged upstream pull requests, so cloning it needs gitlab-master
# credentials. Its nccl-extensions submodule now points at the public mirror,
# so only the fork itself is gated.
TRANSFORMER_ENGINE_REPO="ssh://git@gitlab-master.nvidia.com:12051/ADLR/transformer_engine.git"
# Head of nemotron4 as of 10/20/2026, based on upstream main at ea1a165d. It
# adds ScaledTanhSReLU and routes it through the fused grouped MLP, which is
# what needs the cudnn-frontend commit pinned above, and carries three merged
# upstream context-parallel attention commits on top. Three cherry-picks sit on
# that, one commit each. Upstream pull request 3426, merged, exports
# mark_qb_bin_bounds_validated; without it the quantile-balancing router
# re-validates its bin bounds every step, and that validation reads them back
# to the host, so each router layer costs a device sync per iteration. Version
# 11 carried the pre-merge revision of it; this is the merged one, which also
# tracks whether the mark happened during graph capture. Upstream pull request
# 3517, merged on 09/25, threads sharded weights through GroupedLinear's
# grouped-tensor path, so a GTP-sharded expert layer no longer reads main_grad
# off a plain tensor in backward; version 12 carried the pre-merge revision,
# which differs only by a skip guard on its test. Upstream pull request 3545,
# still open, keeps the DistributedWeight objects on ctx rather than reading
# them back from the saved tensors, which saved-tensor hooks rebuild as plain
# tensors; without it fine-grained activation offloading around the experts
# makes backward take the plain-parameter branch and raise "weight was removed
# while fuse_wgrad_accumulation=True". Three fork commits follow. The first has
# the fused router read the selected sigmoid scores back directly rather than
# adding and then subtracting the expert bias, which can erase tiny scores.
# The second, upstream pull request 3653, keeps the router probabilities in FP32
# when the fused grouped MLP scales the activated FC1 output, and returns their
# gradient to the router in FP32. The third, upstream pull request 3652, lets
# FusedAdam keep its second moment in FP64 via exp_avg_sq_dtype=torch.float64;
# the default FP32 and BF16 kernels are unchanged. Two more follow. Upstream
# pull request 3417, merged, fixes TMA shared-memory reuse synchronization in
# the quantization kernels. The last keeps router probabilities in FP32 in the
# basic, unfused scaled activations, which had rounded them to the activation
# dtype in forward and returned their gradient in it. Five more follow: upstream
# pull requests 3595 and 3540, both merged, which skip FA4 for deterministic
# backward on SM12x and select FA4 for D=256 THD attention; a fork commit that
# removes the host syncs from the THD context-parallel P2P backward zero-fill;
# a fork commit that adds an opt-in logit-space expert bias to the fused
# sigmoid router, selecting on logits plus bias and accumulating quantile-
# balancing histograms in logit units; and a fork commit that keeps the
# autocast precision state per thread, so an autocast entered in one thread no
# longer changes the precision of modules running in another.
TRANSFORMER_ENGINE_REF="d73cb8050d4a947b48533185622953b26221958d"
TRANSFORMER_ENGINE_CUDA_ARCHS="100a;103a"
EMERGING_OPTIMIZERS_REPO="https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git"
EMERGING_OPTIMIZERS_REF="v0.4.0"
DEEPEP_REPO="https://github.com/deepseek-ai/DeepEP.git"
# Pinned to head of tree on the hybrid-ep branch as of 08/25/2026.
DEEPEP_REF="58397eac493b075b0baea5861ac6e9793d8d7a21"
ONE_LOGGER_INDEX_URL="https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple"
# Architectures the CUDA sources are compiled for. This cluster is GB300, which
# is sm_103; the earlier "9.0 10.0" list omitted it entirely, so DeepEP carried
# no native GB300 kernels. Both lists match the reference container, which drops
# sm_90: nothing here targets H100 any more.
DEEPEP_TORCH_CUDA_ARCH_LIST="10.0 10.3 10.3a"
DEEPEP_MAX_JOBS="8"
# END DEPENDENCY PINS

# Optional CI source overrides; plain bash install.sh retains the training pins.
NCCL_REPO="${NCCL_REPO_URL:-${NCCL_REPO}}"
NCCL_REF="${NCCL_REPO_SHA:-${NCCL_REF}}"
CUDNN_FRONTEND_REPO="${CUDNN_FRONTEND_REPO_URL:-${CUDNN_FRONTEND_REPO}}"
CUDNN_FRONTEND_REF="${CUDNN_FRONTEND_REPO_SHA:-${CUDNN_FRONTEND_REF}}"
TRANSFORMER_ENGINE_REPO="${TRANSFORMER_ENGINE_REPO_URL:-${TRANSFORMER_ENGINE_REPO}}"
TRANSFORMER_ENGINE_REF="${TRANSFORMER_ENGINE_REPO_SHA:-${TRANSFORMER_ENGINE_REF}}"

# Where the wheels named in the pin block are expected. Defaults to this
# script's own directory, so staging them beside it is enough.
WHEEL_DIR="${WHEEL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Where the source checkouts land. They are left in place after installation.
SOURCE_DIR="${SOURCE_DIR:-/opt}"

# Where the Inspector profiler plugin is installed. Jobs name this path in
# NCCL_PROFILER_PLUGIN; see the nccl section for why it is not on the loader path.
export NCCL_INSPECTOR_DIR="${NCCL_INSPECTOR_DIR:-/opt/nccl_inspector}"

export DEBIAN_FRONTEND=noninteractive

TELEMETRY_ASSETS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/telemetry"

# The frozen-path check the cpython section runs, beside this script so the
# Docker route finds it in the folder it copies.
FROZEN_STDLIB_CHECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_frozen_stdlib.py"

# ==== cpython ====
# Backport gh-142183, which caches one popped interpreter data-stack chunk per
# thread state instead of repeatedly mapping and unmapping chunks.
#
# Before everything else, so that the pip installs and source builds below
# run under the rebuilt interpreter rather than only the last section
# exercising it, and so a broken swap fails ten minutes in rather than an
# hour in.
cd "${SOURCE_DIR}"
git clone --depth 1 --branch "${CPYTHON_REF}" "${CPYTHON_REPO}" cpython
cd "${SOURCE_DIR}/cpython"
git fetch --depth 2 origin "${CPYTHON_PATCH_COMMIT}"
git show --format= --binary "${CPYTHON_PATCH_COMMIT}" | patch -p1
for _stdlib_module in genericpath posixpath ntpath; do
    cp "/usr/lib/python3.12/${_stdlib_module}.py" "Lib/${_stdlib_module}.py"
done
grep -q "class ALLOW_MISSING" Lib/genericpath.py
./configure --prefix=/usr --enable-shared --without-ensurepip
make -j"$(nproc)"
LD_LIBRARY_PATH="${SOURCE_DIR}/cpython" ./python -m test -j 8 test_call test_sys test_threading

CPYTHON_EXTENSION_DIR=$(find build -maxdepth 1 -type d -name 'lib.linux-*-3.12' -print -quit)
test -n "${CPYTHON_EXTENSION_DIR}"
CPYTHON_ARCH_LIB=$(dpkg-architecture -qDEB_HOST_MULTIARCH)
cat >> /etc/python3.12/sitecustomize.py <<'PY'

import site as _datastack_site

for _datastack_path in (
    "/usr/local/lib/python3.12/dist-packages",
    "/usr/lib/python3/dist-packages",
    "/usr/lib/python3.12/dist-packages",
):
    _datastack_site.addsitedir(_datastack_path)

del _datastack_path, _datastack_site
PY

install -m 0755 python /usr/bin/python3.12
install -m 0644 libpython3.12.so.1.0 "/usr/lib/${CPYTHON_ARCH_LIB}/libpython3.12.so.1.0"
cp -L "${CPYTHON_EXTENSION_DIR}"/*.so /usr/lib/python3.12/lib-dynload/
ldconfig
# This build and the base image's both report 3.12.3, so the version cannot
# separate them, the same way it cannot separate the cudnn-frontend source build
# from PyPI's release. sys._git can: it reads ('CPython', '', '') on the base
# image, which is built from a source tarball, and ('CPython',
# 'tags/v3.12.3-dirty', 'f6650f9') here. The revision says this is the tag that
# was cloned and the -dirty says the tree was modified before it was built,
# which is the backport and the only evidence in the image that it is present.
# The hash ties the installed library to the tree built just above. PYTHONHOME
# and PYTHONPATH come off for this check because it asks what the interpreter
# is, not where the environment points it.
env -u PYTHONHOME -u PYTHONPATH python -c "import hashlib, pathlib, sys; assert sys.version_info[:3] == (3, 12, 3), sys.version; assert sys.prefix == '/usr', sys.prefix; assert sys._git[2] and '$(git rev-parse HEAD)'.startswith(sys._git[2]), f'interpreter was not built from ${CPYTHON_REF}: {sys._git}'; assert sys._git[1] == 'tags/${CPYTHON_REF}-dirty', f'built from an unmodified ${CPYTHON_REF}, so the backport is not in it: {sys._git}'; digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(); installed = digest('/usr/lib/${CPYTHON_ARCH_LIB}/libpython3.12.so.1.0'); assert installed == digest('${SOURCE_DIR}/cpython/libpython3.12.so.1.0'), 'libpython3.12 under /usr is not the one just built'; print('cpython', sys.version.split()[0], 'from', sys._git[2], '| libpython', installed[:12])"
# This one keeps both variables, because that is the environment training runs
# in. Importing torch walks the largest set of prebuilt extension modules in the
# image, so an interpreter that is missing one Ubuntu compiled in rather than
# shipped as a shared object fails here instead of at training time. That is how
# a revision of this section that dropped the lib-dynload copy above was caught.
python -c "import torch, numpy; print('rebuilt interpreter imported torch', torch.__version__, 'and numpy', numpy.__version__)"
# Ubuntu's tarfile security backport calls os.path.ALLOW_MISSING, which vanilla
# frozen path modules do not provide. The copies above put Ubuntu's sources in
# the tree the build freezes; this asserts what came out of it. It runs from
# outside the source tree, so the installed modules are the ones under test
# rather than anything the build directory shadows.
cd "${SOURCE_DIR}"
python "${FROZEN_STDLIB_CHECK}"

# ==== nccl ====
# Before transformer-engine, because it compiles libnccl_ep.a against these
# headers and links it statically. Upgrading NCCL later would leave
# transformer-engine built against one NCCL and running against another.
# The build replaces the base image's NCCL in place under /usr, which is where
# transformer-engine's own discovery looks, so nothing downstream needs telling.
cd "${SOURCE_DIR}"
git clone "${NCCL_REPO}" nccl
cd "${SOURCE_DIR}/nccl"
# Ask for the commit by name rather than taking whatever the clone brought. The
# branch it headed has since been force-pushed, which orphaned it: a clone
# fetches only what refs reach, so the checkout alone stopped finding it.
git fetch origin "${NCCL_REF}"
git checkout --detach "${NCCL_REF}"
test "$(git rev-parse HEAD)" = "${NCCL_REF}"
# NCCL_SUFFIX reaches the NCCL_SUFFIX define in nccl.h and the version string
# NCCL logs at init, though not the soname. Without it an image carrying this
# fork is indistinguishable from one carrying the upstream tag it branched off,
# since both report the same number from ncclGetVersion.
make -j"$(nproc)" src.build CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" NCCL_SUFFIX="-g$(git rev-parse --short=10 HEAD)"
# Derive the multiarch triplet rather than naming it: this image is aarch64.
ARCH_LIB=$(dpkg-architecture -qDEB_HOST_MULTIARCH)
cp -f build/include/nccl.h /usr/include/
# The device headers have to move with the library. nccl-ep's JIT compiles its
# kernels against /usr/include/nccl_device, so leaving the base image's copies
# there builds kernels that read ncclDevComm fields at 2.30.7 offsets against a
# library that writes them at 2.31.2 offsets: ncclResourceWindow_vidmem_t alone
# shrank from 64 bytes to 16, so every field after it lands in the wrong place
# and the intra-node LSA dispatch kernel dies on an illegal address. Drop the
# old directory rather than copying over it, so a header this version no longer
# ships cannot survive underneath.
rm -rf /usr/include/nccl_device
cp -a build/include/nccl_device /usr/include/
# The library announces its version at runtime and verify-pins asks it, but the
# headers carry no version to ask for, so tie them to the tree that produced the
# library instead. Byte identity is the whole guarantee: the headers and the
# library come out of this one make, so headers that match cannot describe a
# different library than the one installed beside them.
diff -r build/include/nccl_device /usr/include/nccl_device || { echo "error: /usr/include/nccl_device does not match the ${NCCL_REF} build" >&2; exit 1; }
echo "nccl_device headers match the ${NCCL_REF} build"
# Drop whatever version is there rather than naming one, then copy with -P so
# the libnccl.so.2 symlink is preserved rather than followed.
rm -f /usr/lib/${ARCH_LIB}/libnccl.so.2*
cp -P build/lib/libnccl.so.2* /usr/lib/${ARCH_LIB}/
ldconfig
# The Inspector profiler plugin, from the same tree as the library above: it
# reads device-side telemetry through a descriptor layout the two share, and a
# plugin built from a different commit misreads every record. Its Makefile is
# standalone and compiles against the profiler interface headers vendored beside
# it, so building from this checkout is what pairs the two; there is no header
# path to point at the library.
cd "${SOURCE_DIR}/nccl/plugins/profiler/inspector"
make -j"$(nproc)"
# Installed off the loader path on purpose. HPC-X ships a
# libnccl-profiler-inspector.so of its own in nccl_spectrum-x_plugin/lib, which
# is on the loader path, so a job that names the plugin by soname gets theirs.
# Jobs point NCCL_PROFILER_PLUGIN at this absolute path instead. Nothing here
# sets that variable, so a job that does not ask for the profiler pays nothing.
mkdir -p "${NCCL_INSPECTOR_DIR}"
install -m 0644 libnccl-profiler-inspector.so "${NCCL_INSPECTOR_DIR}/libnccl-profiler-inspector.so"

# ==== mamba ====
# causal-conv1d and mamba-ssm ship no wheel for this CUDA/PyTorch pair, so both
# are forced to build from source against the base image's torch.
cd "${SOURCE_DIR}"
git clone "${CAUSAL_CONV1D_REPO}" causal_conv1d
cd "${SOURCE_DIR}/causal_conv1d"
git checkout "${CAUSAL_CONV1D_REF}"
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install -vvv --no-build-isolation .

cd "${SOURCE_DIR}"
git clone "${MAMBA_REPO}" mamba
cd "${SOURCE_DIR}/mamba"
git checkout "${MAMBA_REF}"
# --no-deps keeps mamba-ssm from pulling its own causal-conv1d wheel over the
# source build above.
MAMBA_FORCE_BUILD=TRUE pip install -vvv --no-build-isolation --no-deps .

# ==== cudnn-frontend ====
# Installed ahead of transformer-engine, which builds against these headers.
cd "${SOURCE_DIR}"
git clone "${CUDNN_FRONTEND_REPO}" cudnn_frontend
cd "${SOURCE_DIR}/cudnn_frontend"
git checkout "${CUDNN_FRONTEND_REF}"
# Build isolation stays on here, unlike the source builds above: this one
# needs cmake, ninja and pybind11, which the base image does not carry.
pip install -v .
# transformer-engine locates these headers through the installed
# distribution and hard-errors when they are absent. Check the source build
# put them where the released wheel does, using the same lookup TE uses, so
# a packaging difference fails here instead of an hour into the TE compile.
python -c "from importlib.metadata import distribution; from pathlib import Path; h = Path(distribution('nvidia-cudnn-frontend').locate_file('include')) / 'cudnn_frontend.h'; assert h.is_file(), f'cudnn-frontend headers missing at {h}'; print('cudnn-frontend headers at', h.parent)"

# ==== flash-attn-4 ====
# Installed in three steps so cutlass-dsl settles on 4.6.0 with its CUDA 13
# libraries. The cu13 extra comes first because the metapackage otherwise
# installs only CUDA 12 libraries; the later commands repeat the 4.6.0 pin so
# quack-kernels and flash-attn-4 cannot drag it back. quack-kernels 0.6.1
# is the release that pins cutlass-dsl to exactly 4.6.0.
# Hold quack-kernels at 0.6.1. 0.6.4 adds quack/dsl/mixed_constexpr_if.py, which
# quack/dsl/__init__.py imports and which monkey-patches
# cutlass.base_dsl.ast_preprocessor for the whole process. Its rewrite of
# `if const_expr(S) or D:` drops the body's name bindings when the statement has
# no else, so gdp_attn's chunked kernel fails to compile with DSLUserCodeError
# TYPE_UNSTABLE_JOIN. gdp_attn never imports quack; it is reached because
# megatron/core/transformer/attention.py imports flash_attn.cute, which does.
# Verified 2026-08-27: v3 (0.6.4) never reaches iteration 1, v4 (0.6.1) trains.
# cutlass-dsl 4.6.0 caps protobuf below 7, so this drops the base image's
# protobuf 7.35.1 to 6.33.6 and leaves grpcio-tools asking for 7.35.1 or newer.
# The two cannot both be satisfied. grpc_tools and protoc still import on 6.33.6;
# nothing further about grpcio-tools has been tested. 4.5.2 declares no protobuf
# bound, so this arrived with the upgrade.
pip install 'nvidia-cutlass-dsl[cu13]==4.6.0'
pip install quack-kernels==0.6.1 'nvidia-cutlass-dsl==4.6.0'
pip install flash-attn-4==4.0.0b19 'nvidia-cutlass-dsl==4.6.0'
# flash-attn-4 lands under flash_attn/cute/ and does not own
# flash_attn/__init__.py, so it sits alongside the base image's flash-attn 2
# rather than replacing it. Import the subpackage: flash_attn.__version__ is
# flash-attn 2's and says nothing about whether this install worked.
python -c "import flash_attn.cute; import flash_attn; print('flash-attn-4 present under flash_attn.cute; flash_attn is', flash_attn.__version__)"

# ==== transformer-engine ====
cd "${SOURCE_DIR}"
git clone "${TRANSFORMER_ENGINE_REPO}" transformer_engine
cd "${SOURCE_DIR}/transformer_engine"
git checkout "${TRANSFORMER_ENGINE_REF}"
NVTE_CUDA_ARCHS="${TRANSFORMER_ENGINE_CUDA_ARCHS}" \
    NVTE_FRAMEWORK=pytorch pip install -vvv --no-build-isolation .

# ==== emerging-optimizers ====
cd "${SOURCE_DIR}"
git clone "${EMERGING_OPTIMIZERS_REPO}" emerging_optimizers
cd "${SOURCE_DIR}/emerging_optimizers"
git checkout "${EMERGING_OPTIMIZERS_REF}"
pip install .

# ==== hybrid-ep ====
# rdma-core is deliberately not built here. The base image already ships the
# infiniband headers at /usr/include/infiniband, so the source build was never
# needed, and building it left a libibverbs that could not find a matching
# provider: every rank warned that it could not load libmlx5-rdmav59.so.
# HYBRID_EP_MULTINODE is not set either. HybridEP is used only within an NVL
# domain here, so DeepEP's internode transport is not exercised.
apt-get update
apt-get install -y --no-install-recommends libnvidia-ml-dev
cd "${SOURCE_DIR}"
git clone "${DEEPEP_REPO}" deep_ep
cd "${SOURCE_DIR}/deep_ep"
git checkout "${DEEPEP_REF}"
TORCH_CUDA_ARCH_LIST="${DEEPEP_TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${DEEPEP_MAX_JOBS}" \
    pip install --no-build-isolation .
apt-get purge -y libnvidia-ml-dev
apt-get autoremove -y
rm -rf /var/lib/apt/lists/*

# ==== training-and-inference ====
# flash-attn is deliberately absent below. The NGC base image ships a tuned
# build carrying a local version segment, and installing over it from PyPI
# would replace it with a slower stock source build. Read the version instead
# of importing it, which needs no GPU, so a base image that stops shipping
# flash-attn fails here rather than at training time.
python -c "import importlib.metadata as m; print('flash-attn', m.version('flash-attn'))"

# einops and wandb also ship in the base image. These pins match what it
# carries, so pip leaves them alone until the base image moves.
# aiofiles, h2, hpack, hyperframe, priority and wsproto are reached through
# quart and hypercorn rather than named by them. Nothing else pins them, so a
# later rebuild would take whatever PyPI carries by then and two images of the
# same version could differ. They install in this one command so pip resolves
# the whole set at once. h11 is absent deliberately: the base image ships it at
# a version hypercorn accepts.
pip install \
    boto3==1.43.80 \
    einops==0.8.2 \
    msgpack==1.1.1 \
    sentencepiece==0.2.2 \
    tiktoken==0.14.0 \
    wandb==0.28.1 \
    aiofiles==25.1.0 \
    flask==3.1.3 \
    flask-restful==0.3.10 \
    h2==4.4.1 \
    hpack==4.2.0 \
    hypercorn==0.18.0 \
    hyperframe==6.1.0 \
    priority==2.0.0 \
    quart==0.20.0 \
    transformers==5.15.1 \
    wsproto==1.3.2

# ==== one-logger ====
if [[ ${INSTALL_INTERNAL_DEPS:-1} != 0 ]]; then
    pip install --index-url="${ONE_LOGGER_INDEX_URL}" one-logger==1.5.5
fi

# ==== debugging ====
apt-get update
apt-get install -y gdb strace
rm -rf /var/lib/apt/lists/*
pip install py-spy==0.4.2 pynvml==13.0.1

# ==== flash-linear-attention ====
pip install flash-linear-attention==0.5.1

# ==== cutlass-dsl-internal ====
# gdp-attn-fused imports cutlass.tensor_utils, which no published
# nvidia-cutlass-dsl provides: 4.5.2, 4.6.2 and 4.7.1 all lack it. This internal
# wheel is a separate distribution that ships its own cutlass through a .pth, so
# it installs alongside the PyPI one rather than replacing it, and wins the
# import. It is a superset: cutlass.cute still resolves, so flash-attn-4 keeps
# working. Installed after flash-attn-4 and before gdp-attn-fused.
# The PyPI nvidia-cutlass-dsl is deliberately left installed. The internal wheel
# ships its own cutlass through a .pth and wins the import anyway, verified on an
# image carrying both: cutlass resolves to nvidia_cutlass_dsl/dsl_packages and
# gdp_attn gets the cutlass.tensor_utils no published release provides. Removing
# the PyPI distribution also removes its metadata, and flash-attn-4 fails to load
# with "No package metadata was found for nvidia-cutlass-dsl".
# An earlier revision swept it away on the theory that the two coexisting caused
# gdp_attn's DSLUserCodeError TYPE_UNSTABLE_JOIN. That was wrong; see the
# quack-kernels pin above for the real cause.
if [[ ${INSTALL_INTERNAL_DEPS:-1} != 0 ]]; then
    test -f "${WHEEL_DIR}/${CUTLASS_DSL_INTERNAL_WHEEL}" || { echo "missing wheel: ${CUTLASS_DSL_INTERNAL_WHEEL}" >&2; exit 1; }
    pip install "${WHEEL_DIR}/${CUTLASS_DSL_INTERNAL_WHEEL}"
    python -c "import cutlass, cutlass.tensor_utils, cutlass.cute; print('cutlass from', cutlass.__file__)"
fi

# ==== gdp-attn-fused ====
# Custom fused GDP attention kernels, installed from a wheel on scratch. The
# wheel is pure Python and wants only torch, so it needs no compiler and can sit
# anywhere after torch exists.
if [[ ${INSTALL_INTERNAL_DEPS:-1} != 0 ]]; then
    test -f "${WHEEL_DIR}/${GDP_ATTN_FUSED_WHEEL}" || { echo "missing wheel: ${GDP_ATTN_FUSED_WHEEL}" >&2; exit 1; }
    pip install "${WHEEL_DIR}/${GDP_ATTN_FUSED_WHEEL}"
    python -c "import gdp_attn, importlib.metadata as m; print('gdp-attn-fused', m.version('gdp-attn-fused'))"
fi

# ==== mistral-dataloader ====
# Transitive imports of Mistral's PretokenizedDataLoader, and of mistral_common
# through masking.py -> mistral_common.tokens.tokenizers -> protocol -> chunk.
# Kept in step with scaling_ladder/mistral_requirements.txt, minus boto3 and
# tiktoken, which install.sh pins earlier and an unpinned entry here would be
# free to upgrade past, and minus that file's numpy<2.0. Nothing here asks
# for numpy 1, and opencv-python-headless requires numpy 2, so the cap only
# forced numpy back to 1.26 and opencv back a major version.
pip install \
    av==18.1.0 \
    cloudpickle==3.1.2 \
    dacite==1.9.2 \
    fasttext==0.9.3 \
    fire==0.7.1 \
    imagehash==4.3.2 \
    msgspec==0.21.1 \
    opencv-python-headless==5.0.0.93 \
    orjson==3.12.0 \
    psutil==7.2.2 \
    pybase64==1.5.0 \
    safetensors==0.8.0 \
    soundfile==0.14.0 \
    soxr==1.1.0 \
    sympy==1.14.0 \
    tenacity==9.1.4 \
    zstandard==0.25.0 \
    fastjsonschema==2.19.1 \
    jsonschema==4.26.0 \
    openapi-pydantic==0.5.1 \
    partial-json-parser==0.2.1.1.post7 \
    'pydantic-extra-types[pycountry]==2.11.1' \
    regex==2026.7.19 \
    structlog==26.1.0

# These carry compiled extensions built against a numpy ABI. A wheel built
# for numpy 1 fails at import under numpy 2 without declaring any constraint,
# which resolution cannot catch, so import them here rather than discovering it
# mid-training.
python -c "import importlib; mods = ['fasttext', 'cv2', 'imagehash', 'soundfile', 'soxr', 'av']; [importlib.import_module(m) for m in mods]; import numpy; print('imported', ', '.join(mods), 'against numpy', numpy.__version__)"

# ==== nemo-lens ====
# Preserve runtime package versions without constraining isolated builds.
install -d /opt/nemo_telemetry
python - <<'PYTHON' > /opt/nemo_telemetry/constraints.txt
import importlib.metadata as m

from packaging.utils import canonicalize_name

names = {canonicalize_name(d.metadata['Name']) for d in m.distributions() if d.metadata['Name']}
for name in sorted(names):
    if name in {'nemo-lens', 'nvidia-resiliency-ext', 'grpcio-tools'} or name.startswith('opentelemetry-'):
        continue
    print(f'{name}=={m.version(name)}')
PYTHON
cd "${SOURCE_DIR}"
git clone "${NEMO_LENS_REPO}" nemo_lens
git -C nemo_lens fetch origin "${NEMO_LENS_REF}"
git -C nemo_lens checkout --detach "${NEMO_LENS_REF}"
test "$(git -C nemo_lens rev-parse HEAD)" = "${NEMO_LENS_REF}"
pip install -c /opt/nemo_telemetry/constraints.txt \
    -r "${TELEMETRY_ASSETS_DIR}/requirements.txt" "${SOURCE_DIR}/nemo_lens[sdk]"

# ==== nvidia-resiliency-ext ====
cd "${SOURCE_DIR}"
git clone "${NVIDIA_RESILIENCY_EXT_REPO}" nvidia_resiliency_ext
git -C nvidia_resiliency_ext fetch origin "${NVIDIA_RESILIENCY_EXT_REF}"
git -C nvidia_resiliency_ext checkout --detach "${NVIDIA_RESILIENCY_EXT_REF}"
test "$(git -C nvidia_resiliency_ext rev-parse HEAD)" = "${NVIDIA_RESILIENCY_EXT_REF}"
# --no-deps retains the pinned Lens source despite NVRx requiring Lens >=0.3.
pip install --no-deps "${SOURCE_DIR}/nvidia_resiliency_ext[otel]"
# The native .metadata writer is an optional extension, so a failed compile
# would leave the slower Python writer in place without failing the build.
python -c "import nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler.native"
python - "${TELEMETRY_ASSETS_DIR}/requirements.txt" <<'PYTHON'
import importlib.metadata as m
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

requirements = [Requirement(raw) for raw in m.requires('nvidia-resiliency-ext')]
requirements = [str(r) for r in requirements if canonicalize_name(r.name) != 'nemo-lens'
                and (not r.marker or r.marker.evaluate({'extra': 'otel'}))]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-c', '/opt/nemo_telemetry/constraints.txt',
                '-c', sys.argv[1], *requirements], check=True)
PYTHON

# ==== otelcol-contrib ====
test "$(uname -m)" = aarch64 || { echo "otelcol pin requires Linux ARM64" >&2; exit 1; }
OTELCOL_DOWNLOAD_DIR=$(mktemp -d)
trap 'rm -rf "$OTELCOL_DOWNLOAD_DIR"' EXIT
curl --fail --location --proto '=https' --proto-redir '=https' --retry 3 \
    --connect-timeout 20 --max-time 300 "${OTELCOL_URL}" -o "${OTELCOL_DOWNLOAD_DIR}/collector.tar.gz"
printf '%s  %s\n' "${OTELCOL_SHA256}" "${OTELCOL_DOWNLOAD_DIR}/collector.tar.gz" | sha256sum --check -
tar -xzf "${OTELCOL_DOWNLOAD_DIR}/collector.tar.gz" -C "${OTELCOL_DOWNLOAD_DIR}" otelcol-contrib
install -m 0755 "${OTELCOL_DOWNLOAD_DIR}/otelcol-contrib" /usr/local/bin/otelcol-contrib
install -d /etc/otelcol-contrib
install -m 0644 "${TELEMETRY_ASSETS_DIR}"/otelcol-{platform,json-platform}.yaml /etc/otelcol-contrib/
rm -rf "${OTELCOL_DOWNLOAD_DIR}"
trap - EXIT

# ==== verify-pins ====
# Sections install in sequence and a later pip run can move a version an
# earlier one pinned, which is how flash-attn-4's cutlass-dsl pin could slip
# now that it runs before transformer-engine. Assert the versions rather than
# trusting the ordering. This runs after every install and before the stamp, so
# a drifted image carries no version.
python -c "import importlib.metadata as m; want = {'flash-attn-4': '4.0.0b19', 'quack-kernels': '0.6.1', 'nvidia-cudnn-frontend': '1.28.0', 'flash-linear-attention': '0.5.1', 'nvidia-cutlass-dsl': '4.6.0'}; got = {k: m.version(k) for k in want}; drift = {k: (v, got[k]) for k, v in want.items() if got[k] != v}; assert not drift, f'version drift, wanted vs got: {drift}'; import numpy; assert numpy.__version__.startswith('2.'), 'numpy moved off 2.x: ' + numpy.__version__; print('pins held:', got, '| numpy', numpy.__version__)"
# PyPI's 1.28.0 release declares the same version as the pinned develop commit
# but predates cudnn-frontend #858, so the version above cannot tell a released
# wheel from this source build. Check the wrapper signatures instead, which is
# the same test TransformerEngine runs before it will construct a soft-clamped
# squared ReLU. Both directions are required: a frontend carrying only the
# forward clamp would train against an unclamped backward.
# These CuTeDSL wrappers require APIs supplied by the internal CUTLASS wheel.
if [[ ${INSTALL_INTERNAL_DEPS:-1} != 0 ]]; then
    python -c "import inspect, cudnn; missing = [w.__name__ for w in (cudnn.grouped_gemm_srelu_wrapper_sm100, cudnn.grouped_gemm_dsrelu_wrapper_sm100) if 'tanh_clamp_scale' not in inspect.signature(w).parameters]; assert not missing, f'cudnn-frontend predates #858; no tanh_clamp_scale in: {missing}'; print('cudnn-frontend carries tanh_clamp_scale in both srelu wrappers')"
else
    python -c "import cudnn; print('public cuDNN frontend import passed')"
fi
# Two distributions ship a cutlass package and only one can win sys.path.
# Assert both consumers still resolve rather than trusting install order.
if [[ ${INSTALL_INTERNAL_DEPS:-1} != 0 ]]; then
    python -c "import importlib.metadata as m, cutlass, cutlass.tensor_utils, cutlass.cute, gdp_attn; import flash_attn.cute; assert 'nvidia_cutlass_dsl' in cutlass.__file__, 'cutlass resolved to ' + cutlass.__file__ + ', not the internal wheel'; print('cutlass from', cutlass.__file__); print('cutlass-dsl PyPI', m.version('nvidia-cutlass-dsl'), '| internal', m.version('nvidia-cutlass-dsl-internal'), '| gdp-attn-fused', m.version('gdp-attn-fused'))"
else
    python -c "import cutlass.cute; import flash_attn.cute; print('public CUTLASS DSL and flash-attn-4 imports passed')"
fi
# Ask the loaded library its own version. torch.cuda.nccl.version() reports the
# NCCL torch was compiled against, which stays at the base image's version
# however the library is replaced, so it cannot answer this.
# Reads NCCL_EXPECTED_VERSION rather than parsing NCCL_REF, which no longer
# carries a version now that it names a commit on the fork.
python -c "import ctypes; want = '${NCCL_EXPECTED_VERSION}'; v = ctypes.c_int(); ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(v)); got = '%d.%d.%d' % (v.value // 10000, (v.value // 100) % 100, v.value % 100); assert got == want, f'nccl {got} does not match the pinned {want}'; print('nccl', got)"
# ncclGetVersion returns the same number for this fork as for the upstream tag it
# branched from, so it cannot tell the two apart. The NCCL_SUFFIX the build
# passed can: it reaches nccl.h, which is what transformer-engine compiles
# against. Read the commit back out of the checkout rather than from NCCL_REF,
# which may name a branch or a tag instead.
grep -q -- "-g$(git -C "${SOURCE_DIR}/nccl" rev-parse --short=10 HEAD)" /usr/include/nccl.h || { echo "error: /usr/include/nccl.h carries no -g<commit> suffix, so it is not the build from this tree" >&2; exit 1; }
# A plugin that cannot be loaded is the failure mode this build has actually hit,
# and NCCL reports it only as a warning at communicator init, hours later.
python -c "import ctypes, os; p = os.path.join('${NCCL_INSPECTOR_DIR}', 'libnccl-profiler-inspector.so'); ctypes.CDLL(p); print('inspector plugin loads:', p)"
# The device headers nccl-ep's JIT compiles against have to stay paired with
# that library. Losing the pairing is what versions 1 to 8 did: they upgraded
# the library and left these at the base image's, so the kernels read
# ncclDevComm at the wrong offsets and the dispatch kernel died on an illegal
# address. Nothing below the nccl section installs libnccl-dev today, so this
# guards against one that starts to.
diff -r "${SOURCE_DIR}/nccl/build/include/nccl_device" /usr/include/nccl_device || { echo "error: /usr/include/nccl_device no longer matches the ${NCCL_REF} build" >&2; exit 1; }
echo "nccl_device headers match the ${NCCL_REF} build"
# transformer-engine embeds the short hash of the commit it was built from, so
# this catches a build that picked up the wrong fork commit.
python -c "import importlib.metadata as m, sys; want = '${TRANSFORMER_ENGINE_REF:0:8}'; got = m.version('transformer_engine'); sys.exit(0) if got.endswith('+' + want) else sys.exit(f'transformer_engine {got} was not built from {want}')"
# The interpreter is part of this set too. The cpython section rebuilt it at the
# top and every install since has run under it, so assert here as well. The
# debugging section is the one to watch: gdb depends on libpython3.12, so an apt
# run that pulls a newer one in writes over the library built above, and a pip
# install that puts a python of its own ahead of this one does the same thing.
python -c "import hashlib, pathlib, sys; assert sys.version_info[:3] == (3, 12, 3), sys.version; assert sys._git[1] == 'tags/${CPYTHON_REF}-dirty', f'not the patched interpreter the cpython section built: {sys._git}'; digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(); got = digest('/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)/libpython3.12.so.1.0'); assert got == digest('${SOURCE_DIR}/cpython/libpython3.12.so.1.0'), 'libpython3.12 is no longer the one the cpython section built'; print('cpython', sys.version.split()[0], 'from', sys._git[2], '| libpython', got[:12])"

python - "${TELEMETRY_ASSETS_DIR}/requirements.txt" "${NEMO_LENS_REF}" "${NVIDIA_RESILIENCY_EXT_REF}" <<'PYTHON'
import importlib.metadata as m
import sys
from pathlib import Path

from nvidia_resiliency_ext.shared_utils import telemetry

assert telemetry._AVAILABLE, 'NVRx telemetry import failed'
for name, ref in zip(('nemo-lens', 'nvidia-resiliency-ext'), sys.argv[2:]):
    version = m.version(name)
    revision = version.partition('+')[2]
    # Git abbreviations can grow beyond seven characters as a repository grows.
    assert (7 <= len(revision) <= len(ref)
            and all(char in '0123456789abcdef' for char in revision)
            and ref.startswith(revision)), f'{name} {version} does not match {ref}'
for line in Path(sys.argv[1]).read_text().splitlines():
    if line.strip() and not line.startswith('#'):
        name, version = line.split('==')
        assert m.version(name) == version, f'{name} does not match {version}'
PYTHON
test "$(otelcol-contrib --version)" = "otelcol-contrib version ${OTELCOL_VERSION}"
OTELCOL_CHECK_DIR=$(mktemp -d)
trap 'rm -rf "$OTELCOL_CHECK_DIR"' EXIT
mkdir -p "${OTELCOL_CHECK_DIR}/job-validation"
for config in otelcol-platform.yaml otelcol-json-platform.yaml; do
    OTEL_ARCHIVE_DIR="$OTELCOL_CHECK_DIR" SLURM_JOB_ID=validation SLURMD_NODENAME=validation \
        OTEL_NODE_LOCAL_OTLP_ENDPOINT=http://localhost:44327 \
        timeout 30 otelcol-contrib validate --config "/etc/otelcol-contrib/${config}"
done
rm -rf "$OTELCOL_CHECK_DIR"
trap - EXIT

# ==== version-stamp ====
# Record which dependency set this environment carries, so a container can be
# identified after the fact. This runs last, so the stamp is only present when
# every section above succeeded.
echo "${DEPENDENCY_SET_VERSION}" > /etc/megatron_dependency_set_version
