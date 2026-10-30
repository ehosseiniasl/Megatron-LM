#!/bin/bash
# consolidate_cache.sh — merge the per-node output of a seed run into ONE
# deduplicated "consolidated" tarball per scope, so every node loads the same
# small base instead of reading N per-node shards from Lustre.
#
# Rationale (measured on a 64-node seed): the 64 node shards are ~8.4x
# redundant (110,091 files -> 13,159 unique). The same-path differences across
# nodes are cosmetic (embedded /tmp debug paths baked into cubins); the
# genuinely node-specific entries are Inductor fxgraph shards that miss on a
# checkpoint restart anyway. A single consolidated base therefore yields the
# same warm-start hits at ~1/8 the size.
#
# Input:   ${MCORE_CACHE_DIR}/cache_write/_node/node-<rank>/  -- either plain
#          <scope>/ dirs (write mode without MCORE_CACHE_LOCAL) or <scope>.tar.zst
#          per node (package_node_cache.sh)
# Output:  ${MCORE_CACHE_DIR}/consolidated/<scope>.tar.zst  -- the one layout the
#          sitecustomize shim (lustre source) and install_image_cache.py read
#
# Run OFFLINE on any host after the seed job ended (login node or cpu job):
#   MCORE_CACHE_DIR=<namespace> bash tools/persistent_cache/consolidate_cache.sh
#
# Optional frequency filter: MCORE_CACHE_CONSOLIDATE_MIN_FRACTION=f keeps only
# files present on >= f of the source shards (0..1). Default 0 = full lossless
# union. Set it to, e.g., 0.9, to keep only the shared, shape-static base and drop the
# node-specific tail (which does not hit on a restart).
#
# Overrides: MCORE_CACHE_CONSOLIDATE_SRC, MCORE_CACHE_CONSOLIDATE_OUT,
#            MCORE_CACHE_SCOPES (default from lib.sh), CACHE_COMPRESSOR.

set -euo pipefail
: "${MCORE_CACHE_DIR:?MCORE_CACHE_DIR=<cache namespace> is required}"
# shellcheck disable=SC1091
source "$(dirname "$0")/lib.sh"

# Seed runs leave per-node DIRS (direct write mode) or per-node TARBALLS
# (package_node_cache.sh) under cache_write/_node/; both shapes are accepted.
SRC_ROOT="${MCORE_CACHE_CONSOLIDATE_SRC:-${MCORE_CACHE_DIR}/cache_write/_node}"
OUT_DIR="${MCORE_CACHE_CONSOLIDATE_OUT:-${MCORE_CACHE_DIR}/consolidated}"
MIN_FRACTION="${MCORE_CACHE_CONSOLIDATE_MIN_FRACTION:-0}"

[[ -d "${SRC_ROOT}" ]] || { echo "[CONSOLIDATE] missing shard root: ${SRC_ROOT}" >&2; exit 2; }
mkdir -p "${OUT_DIR}"

# Enumerate the source node dirs (node-<rank>, tarballs or plain scope dirs).
# Any first-level directory counts; entries with neither a scope tarball nor a
# scope dir are skipped naturally in the per-scope loop.
_is_node_source() {
  local d="$1" sc
  for sc in "${SCOPES[@]}"; do
    [[ -f "${d}/${sc}.tar.zst" || -d "${d}/${sc}" ]] && return 0
  done
  return 1
}
node_dirs=()
while IFS= read -r _d; do
  _is_node_source "${_d}" && node_dirs+=("${_d}")
done < <(find "${SRC_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort -V)
num_nodes=${#node_dirs[@]}
(( num_nodes > 0 )) || { echo "[CONSOLIDATE] no per-node sources under ${SRC_ROOT}" >&2; exit 2; }

# min_count = ceil(MIN_FRACTION * num_nodes), floored at 1.
min_count="$(awk -v f="${MIN_FRACTION}" -v n="${num_nodes}" \
  'BEGIN{ c=int(f*n); if (c < f*n) c++; if (c < 1) c=1; print c }')"
echo "[CONSOLIDATE] nodes=${num_nodes} min_fraction=${MIN_FRACTION} -> keep files present on >= ${min_count} node(s)"
echo "[CONSOLIDATE] scopes: ${SCOPES[*]}"

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

had_error=0
for scope in "${SCOPES[@]}"; do
  stage="${work}/${scope}/stage"
  merged="${work}/${scope}/merged"
  mkdir -p "${stage}" "${merged}"

  # Extract every node's shard for this scope into an isolated per-node dir so
  # identical relative paths from different roles do not overwrite during the
  # frequency count.
  present=0
  for nd in "${node_dirs[@]}"; do
    tarball="${nd}/${scope}.tar.zst"
    scope_dir="${nd}/${scope}"
    dest="${stage}/$(basename "${nd}")"
    if [[ -f "${tarball}" ]]; then
      mkdir -p "${dest}"
      # tar -xf auto-detects zstd or gzip, so either compressor round-trips.
      if tar -xf "${tarball}" -C "${dest}" 2>/dev/null; then
        present=$((present + 1))
      else
        # A partial extraction must not enter the union: its truncated files
        # could be picked as the representative copy and sealed.
        rm -rf "${dest}"
        echo "[CONSOLIDATE] ${scope}: extract failed for ${tarball} (skipped)" >&2
      fi
    elif [[ -d "${scope_dir}" && -n "$(ls -A "${scope_dir}" 2>/dev/null)" ]]; then
      # Durable-Lustre seed: plain directory, no tar round-trip needed.
      mkdir -p "${dest}"
      if cp -a "${scope_dir}/." "${dest}/" 2>/dev/null; then
        present=$((present + 1))
      else
        rm -rf "${dest}"
        echo "[CONSOLIDATE] ${scope}: copy failed for ${scope_dir} (skipped)" >&2
      fi
    fi
  done
  if (( present == 0 )); then
    echo "[CONSOLIDATE] ${scope}: no shards, skipping"
    continue
  fi

  # Count how many node dirs contain each relative path. Cache relpaths carry no
  # spaces (Triton/Inductor hash dirs), so the "count relpath" split is safe.
  : > "${work}/${scope}.relpaths"
  for sd in "${stage}"/*/; do
    [[ -d "${sd}" ]] || continue
    ( cd "${sd}" && find . -type f -print ) >> "${work}/${scope}.relpaths"
  done

  # Keep each path present on >= min_count nodes; copy one representative. When
  # roles disagree only in embedded debug paths (verified cosmetic), keeping any
  # one copy is correct; a real miss just recompiles at load time.
  kept=0; dropped=0
  while IFS= read -r line; do
    count="${line%% *}"
    relpath="${line#* }"
    rel="${relpath#./}"
    if (( count >= min_count )); then
      for sd in "${stage}"/*/; do
        src="${sd}/${rel}"
        if [[ -f "${src}" ]]; then
          mkdir -p "${merged}/$(dirname "${rel}")"
          cp -a "${src}" "${merged}/${rel}"
          kept=$((kept + 1))
          break
        fi
      done
    else
      dropped=$((dropped + 1))
    fi
  done < <(sort "${work}/${scope}.relpaths" | uniq -c | sed -E 's/^[[:space:]]*//')

  if [[ -z "$(ls -A "${merged}" 2>/dev/null)" ]]; then
    # A scope can be valid yet contain no files on any node (for example,
    # cudnn_fe in a bounded seed). Emit a directory-only archive rather than
    # omitting the configured scope: readers then report scope=ok and safely
    # compile a miss into node-local storage without mutating the publication.
    if tar_safe "${merged}" "${OUT_DIR}/${scope}.tar.zst" "${scope}" 1; then
      echo "[CONSOLIDATE] ${scope}: shards=${present} kept=0 dropped=${dropped} empty_scope=1"
    else
      echo "[CONSOLIDATE] ${scope}: empty-scope tar failed" >&2
      had_error=1
    fi
    continue
  fi
  # tar_safe: atomic tmp+rename, zstd if available else gzip (see lib.sh).
  if tar_safe "${merged}" "${OUT_DIR}/${scope}.tar.zst" "${scope}"; then
    echo "[CONSOLIDATE] ${scope}: shards=${present} kept=${kept} dropped=${dropped}"
  else
    echo "[CONSOLIDATE] ${scope}: tar failed" >&2
    had_error=1
  fi
done

echo "[CONSOLIDATE] done: ${OUT_DIR} ($(du -sh "${OUT_DIR}" 2>/dev/null | cut -f1 || echo '?'))"
exit "${had_error}"
