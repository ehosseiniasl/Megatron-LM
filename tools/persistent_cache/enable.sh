#!/bin/bash
# enable.sh — the first-iteration kernel-cache interface. SOURCE this, then run
# the training command as usual; nothing wraps the command. The PYTHONPATH
# shim's sitecustomize does the per-node work lazily inside every Python
# worker (see _triton_patch_shim/mcore_cache_node_setup.py).
#
# Source it inside the container command (simplest):
#   RUN_CMD="cd ${REPO_DIR}; source tools/persistent_cache/enable.sh && exec python ..."
# or in the sbatch body, then forward ${MCORE_CACHE_ENV_FORWARD} via --container-env.
#
# Cache source (read mode):
#   image   default when MCORE_CACHE_DIR is unset: the cache baked into the
#           container image at ${MCORE_CACHE_IMAGE_DIR:-/opt/mcore_cache} by
#           bake_cache_image.sh. Used in place on a writable rootfs; copied
#           node-locally otherwise. No shared-filesystem traffic.
#   lustre  export MCORE_CACHE_DIR=<namespace dir> holding consolidated/<scope>.tar.zst,
#           extracted once per node. MCORE_CACHE_PREFER_IMAGE=1 makes the image
#           cache win even when MCORE_CACHE_DIR is set.
#
# Modes (MCORE_CACHE_MODE, default read):
#   read   warm consume.
#   write  seed run (needs MCORE_CACHE_DIR): caches land under
#          ${MCORE_CACHE_DIR}/cache_write/_node/node-<rank>/ (or node-locally with
#          MCORE_CACHE_LOCAL set, for package_node_cache.sh). Seal them offline:
#            MCORE_CACHE_DIR=<ns> bash consolidate_cache.sh
#
# A namespace is only valid for the image + code + model geometry it was
# seeded on; encode those in its directory name and bump a -vN suffix to rebuild.

export MCORE_CACHE_MODE="${MCORE_CACHE_MODE:-read}"
# Repository loaders default to /opt/mcore_cache. Retain location discovery
# for copies bundled in older images: <root>/tools/persistent_cache/enable.sh
# beside <root>/MANIFEST uses that root unless the caller overrides it.
if [[ -z "${MCORE_CACHE_IMAGE_DIR:-}" ]]; then
  _mcen_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  if [[ -f "${_mcen_root}/MANIFEST" ]]; then
    MCORE_CACHE_IMAGE_DIR="${_mcen_root}"
  else
    MCORE_CACHE_IMAGE_DIR="/opt/mcore_cache"
  fi
fi
export MCORE_CACHE_IMAGE_DIR
export MCORE_CACHE_SCOPES="${MCORE_CACHE_SCOPES:-triton inductor cuda_ptx cutedsl hybrid_ep cudnn_fe megatron_fused_kernels}"
# Cache status lines print from rank 0 only; set to 1 for full per-rank output.
export MCORE_CACHE_VERBOSE="${MCORE_CACHE_VERBOSE:-0}"

if [[ -n "${MCORE_CACHE_DIR:-}" ]]; then
  export MCORE_CACHE_DIR
  _mcen_source="lustre"
  _mcen_ns="$(basename "${MCORE_CACHE_DIR}")"
elif [[ "${MCORE_CACHE_MODE}" == "write" ]]; then
  echo "[CACHE ENABLE] error: MCORE_CACHE_MODE=write needs MCORE_CACHE_DIR=<cache namespace dir>" >&2
  return 2 2>/dev/null || exit 2
else
  _mcen_source="image"
  # The seed's key travels inside the cache (cutedsl/.aot_namespace); the shim
  # overrides this placeholder from it.
  _mcen_ns="image"
fi
if [[ "${MCORE_CACHE_MODE}" != "read" && "${MCORE_CACHE_MODE}" != "write" ]]; then
  echo "[CACHE ENABLE] error: MCORE_CACHE_MODE=${MCORE_CACHE_MODE}; expected read|write" >&2
  return 2 2>/dev/null || exit 2
fi

_mcen_shim="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_triton_patch_shim"
# The selected checkout must win even if another image/repository shim is
# inherited ahead of an existing entry for this one.
case "${PYTHONPATH:-}:" in
  "${_mcen_shim}:"*) ;;
  *) export PYTHONPATH="${_mcen_shim}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# Intra-job reuse wanted in both modes.
export TRITON_CACHE_AUTOTUNING="${TRITON_CACHE_AUTOTUNING:-1}"
export MCORE_CUTE_DSL_FILE_CACHE="${MCORE_CUTE_DSL_FILE_CACHE:-1}"
export MCORE_TRITON_GROUP_REBASE="${MCORE_TRITON_GROUP_REBASE:-1}"
# CuTe AOT object export/reuse is wanted in both modes: a seed must produce the
# objects, a warm run loads them. Kill switch: MCORE_CUTE_DSL_AOT_CACHE=0.
export MCORE_CUTE_DSL_AOT_CACHE="${MCORE_CUTE_DSL_AOT_CACHE:-1}"
if [[ -z "${MCORE_CUTE_DSL_AOT_NAMESPACE:-}" ]]; then
  export MCORE_CUTE_DSL_AOT_NAMESPACE="${_mcen_ns}" MCORE_CUTE_DSL_AOT_NS_FROM=default
fi
# Record runtime Inductor choices while seeding and recover them after reload.
# Both modes need the hooks; explicit 0 remains the kill switch.
export MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE="${MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE:-1}"

# Reuse/recompile counters at process exit (CuTe AOT, Inductor and Triton autotune); zero cost when off.
export MCORE_CACHE_STATS="${MCORE_CACHE_STATS:-0}"
if [[ "${MCORE_CACHE_STATS}" == "1" ]]; then
  export MCORE_CUTE_DSL_AOT_DEBUG="${MCORE_CUTE_DSL_AOT_DEBUG:-1}"
  export TRITON_AUTOTUNE_PATCH_DEBUG="${TRITON_AUTOTUNE_PATCH_DEBUG:-1}"
fi

# Everything a containerized launcher must forward (--container-env) when this
# file is sourced on the HOST side. Unnecessary when sourced inside the container.
export MCORE_CACHE_ENV_FORWARD="PYTHONPATH,MCORE_CACHE_DIR,MCORE_CACHE_MODE,MCORE_CACHE_IMAGE_DIR,MCORE_CACHE_PREFER_IMAGE,MCORE_CACHE_IMAGE_COPY,MCORE_CACHE_SCOPES,MCORE_CACHE_LOCAL,MCORE_CACHE_VERBOSE,TRITON_CACHE_AUTOTUNING,MCORE_CUTE_DSL_FILE_CACHE,MCORE_TRITON_GROUP_REBASE,MCORE_INDUCTOR_AUTOTUNE_RESTART_CACHE,MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE,MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU,MCORE_CUTE_DSL_AOT_CACHE,MCORE_CUTE_DSL_AOT_NAMESPACE,MCORE_CUTE_DSL_AOT_NS_FROM,MCORE_CACHE_STATS,MCORE_CUTE_DSL_AOT_DEBUG,MCORE_CUTE_DSL_AOT_DEBUG_RANKS,TRITON_AUTOTUNE_PATCH_DEBUG,TRITON_AUTOTUNE_PREHOOK_PATCH_DISABLE"

# Optional topology defaults come from the image. Keep this setting unset
# unless the caller overrides it (0 disables reuse); stock NCCL is unchanged.
MCORE_CACHE_ENV_FORWARD="${MCORE_CACHE_ENV_FORWARD},MCORE_NCCL_TOPO_CACHE"

echo "[CACHE ENABLE] mode=${MCORE_CACHE_MODE} source=${_mcen_source} dir=${MCORE_CACHE_DIR:-${MCORE_CACHE_IMAGE_DIR}} scopes='${MCORE_CACHE_SCOPES}'"
