#!/bin/bash
# Package one node's write-mode cache on node-local storage, then atomically
# publish the tarballs for offline consolidation.
#
# Run exactly one copy per node after training (and any requested evaluation)
# has completed successfully. The Python workers write small files only to the
# node-local MCORE_CACHE_LOCAL directory. This script creates and verifies the
# tarballs locally before copying them to shared storage.

set -euo pipefail

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${SLURM_NODEID:?SLURM_NODEID is required}"
: "${MCORE_CACHE_DIR:?MCORE_CACHE_DIR is required}"

LOCAL_CACHE="${MCORE_CACHE_PACKAGE_LOCAL:-/tmp/mcore_cache_${SLURM_JOB_ID}}"
SCOPES="${MCORE_CACHE_SCOPES:-triton inductor cuda_ptx cutedsl hybrid_ep cudnn_fe megatron_fused_kernels}"
PUBLISH_ROOT="${MCORE_CACHE_DIR}/cache_write/_node"
FINAL_DIR="${PUBLISH_ROOT}/node-${SLURM_NODEID}"
PUBLISH_TMP="${PUBLISH_ROOT}/.node-${SLURM_NODEID}.part-${SLURM_JOB_ID}"
LOCAL_PACKAGE_TMP="${MCORE_CACHE_PACKAGE_TMP:-/tmp/mcore_cache_package_${SLURM_JOB_ID}_${SLURM_NODEID}_$$}"
CLEAN_LOCAL="${MCORE_CACHE_PACKAGE_CLEAN_LOCAL:-1}"

case "${CLEAN_LOCAL}" in
  0|1) ;;
  *) echo "[CACHE PACKAGE] MCORE_CACHE_PACKAGE_CLEAN_LOCAL must be 0 or 1" >&2; exit 2 ;;
esac

[[ -d "${LOCAL_CACHE}" ]] || {
  echo "[CACHE PACKAGE] missing local cache: ${LOCAL_CACHE}" >&2
  exit 2
}
[[ "$(cat "${LOCAL_CACHE}/.mcore_ready" 2>/dev/null || true)" == "write ${SLURM_JOB_ID}" ]] || {
  echo "[CACHE PACKAGE] ${LOCAL_CACHE}/.mcore_ready is not 'write ${SLURM_JOB_ID}'" >&2
  exit 2
}
[[ ! -e "${FINAL_DIR}" ]] || {
  echo "[CACHE PACKAGE] refusing to overwrite existing shard: ${FINAL_DIR}" >&2
  exit 2
}

rm -rf "${LOCAL_PACKAGE_TMP}"
mkdir -p "${LOCAL_PACKAGE_TMP}"
cleanup() {
  rm -rf "${LOCAL_PACKAGE_TMP}"
  if [[ -e "${PUBLISH_TMP}" ]]; then
    rm -rf "${PUBLISH_TMP}"
  fi
}
trap cleanup EXIT

for scope in ${SCOPES}; do
  [[ "${scope}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "[CACHE PACKAGE] invalid scope name: ${scope}" >&2
    exit 2
  }
  source_dir="${LOCAL_CACHE}/${scope}"
  [[ -d "${source_dir}" ]] || {
    echo "[CACHE PACKAGE] missing scope: ${source_dir}" >&2
    exit 2
  }
  if [[ -z "$(find "${source_dir}" -mindepth 1 -print -quit)" ]]; then
    # Some consumers legitimately produce no entries for a scope in a bounded
    # seed (for example, cuDNN frontend heuristics may stay unused). Preserve
    # the scope as a valid directory-only archive so read mode can distinguish
    # an accepted empty cache from a missing/corrupt publication.
    echo "[CACHE PACKAGE] empty scope: ${source_dir} (publishing directory-only archive)"
  fi
  tarball="${LOCAL_PACKAGE_TMP}/${scope}.tar.zst"
  if command -v zstd >/dev/null 2>&1; then
    tar --zstd -cf "${tarball}" -C "${source_dir}" .
  else
    # The suffix is an interface name. Readers use tar -xf, which detects gzip.
    tar -czf "${tarball}" -C "${source_dir}" .
  fi
  [[ -s "${tarball}" ]] && tar -tf "${tarball}" >/dev/null
done

mkdir -p "${PUBLISH_ROOT}"
rm -rf "${PUBLISH_TMP}"
mkdir "${PUBLISH_TMP}"
cp "${LOCAL_PACKAGE_TMP}"/*.tar.zst "${PUBLISH_TMP}/"
(
  cd "${PUBLISH_TMP}"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum ./*.tar.zst > MANIFEST
    sha256sum -c MANIFEST >/dev/null
  else
    shasum -a 256 ./*.tar.zst > MANIFEST
    shasum -a 256 -c MANIFEST >/dev/null
  fi
)

# Same-filesystem rename makes a node shard visible only when every scope and
# the checksum manifest are complete.
mv "${PUBLISH_TMP}" "${FINAL_DIR}"
if [[ "${CLEAN_LOCAL}" == "1" ]]; then
  rm -rf "${LOCAL_CACHE}"
fi
echo "[CACHE PACKAGE] published node-${SLURM_NODEID}: ${FINAL_DIR}"
