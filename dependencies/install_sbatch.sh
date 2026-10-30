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
# Runs dependencies/install.sh inside the NGC base image on a Slurm node and
# saves the result as a new squashfs image:
#
#     sbatch install_sbatch.sh
#
# Written for oci-aga, where enroot is absent from the login node, so the image
# is built through pyxis: --container-image supplies the base, and
# --container-save writes the finished filesystem back out.

#SBATCH --job-name=install
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 4:00:00
# Slurm parses this before any shell runs, so it cannot use CODE_DIR below and
# has to repeat the path. Keep the two in step.
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron4/container_setup/logs/%j.out

set -euo pipefail

# The symlink that names the build directory, so a build can be pointed at a
# different checkout by moving the link rather than editing this script.
# install.sh reads the wheels named in its pin block from its own directory, so
# they belong here too.
CODE_DIR=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron4/container_setup
# Where built images are written.
IMAGE_DIR=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron4/images

INSTALL_SCRIPT="${CODE_DIR}/install.sh"
BASE_IMAGE="${IMAGE_DIR}/nvidia+pytorch+26.07-py3.sqsh"

test -f "${INSTALL_SCRIPT}"
test -f "${BASE_IMAGE}"

# Read the version out of the script being run rather than repeating it here, so
# the image name cannot claim a version the script did not install.
DEPENDENCY_SET_VERSION=$(
    sed -n 's/^DEPENDENCY_SET_VERSION="\(.*\)"$/\1/p' "${INSTALL_SCRIPT}"
)
test -n "${DEPENDENCY_SET_VERSION}"

OUTPUT_IMAGE="${IMAGE_DIR}/nvidia+pytorch+26.07-py3+deps+v${DEPENDENCY_SET_VERSION}.sqsh"
# --container-save writes the container filesystem out whatever the step exited
# with, so a build that fails or is cancelled produces an image too. Save under
# a name that says so and rename only once install.sh has returned zero, so
# OUTPUT_IMAGE exists only when every section of it ran. The job id keeps two
# builds of the same version from colliding and says which job left the file.
PARTIAL_IMAGE="${OUTPUT_IMAGE%.sqsh}.partial-${SLURM_JOB_ID}.sqsh"

# --container-save overwrites its target. Refuse to clobber an image that jobs
# may already be running against; pick a new name instead.
if [ -e "${OUTPUT_IMAGE}" ]; then
    echo "error: ${OUTPUT_IMAGE} already exists. Choose a new OUTPUT_IMAGE." >&2
    exit 1
fi

# A cancelled build cannot clean up after itself, since Slurm kills this script
# and pyxis writes the image during step teardown. Those files are harmless and
# block nothing, but say they are there rather than letting them accumulate
# unnoticed. They are safe to delete once no job is reading them.
for LEFTOVER_IMAGE in "${IMAGE_DIR}"/*.partial-*.sqsh; do
    test -e "${LEFTOVER_IMAGE}" || break
    echo "warning: ${LEFTOVER_IMAGE} is a partial image from an earlier build" >&2
done

INSTALL_STATUS=0
srun \
    --container-image="${BASE_IMAGE}" \
    --container-mounts=/scratch:/scratch \
    --container-save="${PARTIAL_IMAGE}" \
    bash "${INSTALL_SCRIPT}" || INSTALL_STATUS=$?

if [ "${INSTALL_STATUS}" -ne 0 ]; then
    echo "error: ${INSTALL_SCRIPT} exited ${INSTALL_STATUS}; removing ${PARTIAL_IMAGE}" >&2
    rm -f "${PARTIAL_IMAGE}"
    exit "${INSTALL_STATUS}"
fi

# Same directory, so this is a link rather than a copy and OUTPUT_IMAGE never
# exists half written. link() fails when the target exists, which is what makes
# this safe: the check above ran before a build that takes hours, so a second
# build of the same version can have claimed the name since, and mv would
# replace an image that jobs may be reading.
if ! ln "${PARTIAL_IMAGE}" "${OUTPUT_IMAGE}"; then
    echo "error: ${OUTPUT_IMAGE} appeared while this build was running." >&2
    echo "This build finished; its image is ${PARTIAL_IMAGE}. Compare the two" >&2
    echo "and either delete it or move it into place once no job is reading" >&2
    echo "${OUTPUT_IMAGE}." >&2
    exit 1
fi
rm -f "${PARTIAL_IMAGE}"

echo "Wrote ${OUTPUT_IMAGE}"
