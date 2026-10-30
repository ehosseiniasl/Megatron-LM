#!/bin/bash
# Rebase Triton cache-group child paths after extracting a restart tarball.
#
# Triton's FileCacheManager stores absolute paths in __grp__*.json. A tarball
# preserves those paths from the producing job, so a later job otherwise sees
# an empty group and recompiles despite having every cached artifact locally.

set -uo pipefail

ROOT=${1:?usage: rebase_triton_groups.sh TRITON_CACHE_DIR}
if [[ ! -d "${ROOT}" ]]; then
    echo "[TRITON-GROUP REBASE] missing root=${ROOT}" >&2
    exit 1
fi

# A shell loop previously launched sed/grep/cmp/chmod once per manifest. A
# production role has ~175 manifests, and an InJob node restores four roles, so
# process launch overhead alone added ~15 seconds to restart setup. Keep this as
# one public script, but do the identical atomic rewrite in one Python process.
python3 - "${ROOT}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rewritten = 0
unchanged = 0
invalid = 0
seen = 0

for manifest in sorted(root.rglob("__grp__*.json")):
    seen += 1
    group_dir = manifest.parent
    cache_key = group_dir.name
    tmp = manifest.with_name(f"{manifest.name}.rebase.{os.getpid()}")
    try:
        original = manifest.read_text()
        escaped_dir = json.dumps(str(group_dir))[1:-1]
        pattern = re.compile(r'(:\s*")[^"]*/' + re.escape(cache_key) + r'/')
        updated = pattern.sub(lambda match: f'{match.group(1)}{escaped_dir}/', original)
        paths = re.findall(r':\s*"([^"]+)"', updated)
        expected_prefix = f"{escaped_dir}/"
        if not paths or any(not path.startswith(expected_prefix) for path in paths):
            invalid += 1
            continue
        if updated == original:
            unchanged += 1
            continue
        tmp.write_text(updated)
        os.chmod(tmp, manifest.stat().st_mode)
        os.replace(tmp, manifest)
        rewritten += 1
    except Exception:
        invalid += 1
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass

print(
    f"[TRITON-GROUP REBASE] root={root} seen={seen} rewritten={rewritten} "
    f"unchanged={unchanged} invalid={invalid}"
)
sys.exit(0 if invalid == 0 and seen > 0 else 1)
PY
