#!/bin/bash
# tools/persistent_cache/lib.sh — shared config for the offline (host-side)
# cache scripts. Source me; don't run me.
#
# Provides: SCOPES (array from MCORE_CACHE_SCOPES), the tar compressor choice
# and tar_safe().

# shellcheck disable=SC2206
SCOPES=(${MCORE_CACHE_SCOPES:-triton inductor cuda_ptx cutedsl hybrid_ep cudnn_fe megatron_fused_kernels})

# Multi-threaded zstd by default (decompress 4-8x faster on multi-core nodes).
# tar --zstd reads ZSTD_NBTHREADS for both compress/decompress.
: "${ZSTD_NBTHREADS:=0}"   # 0 = use all available cores
export ZSTD_NBTHREADS

# Compressor for cache tarballs. zstd is fastest (multi-threaded) but isn't in
# every host/container -- gzip is universal. CACHE_COMPRESSOR=auto picks zstd
# if on PATH, else gzip. Tarball filenames stay *.tar.zst regardless (cosmetic):
# readers use `tar -xf`, which auto-detects the actual format (and fall back to
# Python's zstandard where the zstd CLI is absent).
: "${CACHE_COMPRESSOR:=auto}"
if [[ "${CACHE_COMPRESSOR}" == "zstd" ]] \
   || { [[ "${CACHE_COMPRESSOR}" == "auto" ]] && command -v zstd >/dev/null 2>&1; }; then
  _MCORE_TAR_COMPRESS=(--zstd)
  MCORE_TAR_COMPRESSOR=zstd
else
  _MCORE_TAR_COMPRESS=(-z)
  MCORE_TAR_COMPRESSOR=gzip
fi
export MCORE_TAR_COMPRESSOR

# Atomic tar+compress. tmp+rename so readers never see a partial tarball.
# NOTE: don't use --exclude='.*' — that matches the '.' source arg and emits an empty tar.
tar_safe() {
  local src="$1" tarball="$2" name="$3" allow_empty="${4:-0}"
  [[ -d "$src" ]] || return 1
  if [[ "${allow_empty}" != "1" && -z "$(ls -A "$src" 2>/dev/null)" ]]; then
    return 1
  fi
  local tmp="${tarball}.tmp.$$"
  if tar "${_MCORE_TAR_COMPRESS[@]}" -cf "$tmp" --blocking-factor=8192 \
       -C "$src" --exclude='tmp*' --exclude='.tmp_*' \
       --exclude='.mcore_ready' --exclude='.mcore_cache_dirty' \
       --exclude='.mcore_cache_writeback_stamp' . ; then
    mv "$tmp" "$tarball"
    echo "[CACHE] ${name}: tarball (${MCORE_TAR_COMPRESSOR}, $(du -sh "$tarball" 2>/dev/null | cut -f1))"
    return 0
  fi
  rm -f "$tmp"
  echo "[CACHE] ${name}: tar failed" >&2
  return 1
}
