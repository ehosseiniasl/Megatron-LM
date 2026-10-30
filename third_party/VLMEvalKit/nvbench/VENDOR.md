# Vendored VLMEvalKit Snapshot

The tree above `nvbench/` is a snapshot of VLMEvalKit with upstream `.github/` omitted.
Those CI workflows and helpers are not part of the benchmark runtime and must not be imported:
the parent repository's unanchored `.github/` CODEOWNERS rule also matches nested directories and
would request its automation team. All retained upstream files are unmodified. NVIDIA-authored
integration code is confined to `nvbench/` and is excluded from the upstream-integrity digest.

## Upstream identity

- Repository: `https://github.com/open-compass/VLMEvalKit.git`
- Commit: `bdb6d429f3f6804eaa6cd899c341486c8a42aed8`
- Git tree: `aea0204bd1accfb020c86b6bf27011e0fc3c7cf2`
- Snapshot date: 2026-09-03
- License: Apache-2.0; retained verbatim at `third_party/VLMEvalKit/LICENSE`
- Notice: the upstream snapshot contains no separate `NOTICE` file
- Upstream tracked files: 842
- Retained upstream files: 838 (four `.github/` files omitted)

Upstream tree SHA-256: `e2253ebb91fed4dd3f95dbc8838b9f12b7ae985bfbe4dd4f91adc94ff2d15470`

The SHA-256 above is the `vlmevalkit-vendored-tree-v1` digest of the retained tree, produced by
`nvbench/verify_vendored_tree.py`. It covers each relative path, Git-compatible file mode, and file
content outside `nvbench/`.

## Reproducing the import

Start with a clean checkout at the recorded commit. From the nv-mistralai-megatron repository root:

```bash
git -C /path/to/VLMEvalKit status --short
STAGED_VLMEVALKIT=$(mktemp -d)
git -C /path/to/VLMEvalKit archive --format=tar \
  bdb6d429f3f6804eaa6cd899c341486c8a42aed8 \
  | tar -xf - -C "${STAGED_VLMEVALKIT}"
rm -r -- "${STAGED_VLMEVALKIT}/.github"
# The destination must contain no stale upstream files from a previous import.
tar -C "${STAGED_VLMEVALKIT}" -cf - . | tar -xf - -C third_party/VLMEvalKit
python3 third_party/VLMEvalKit/nvbench/verify_vendored_tree.py \
  --source "${STAGED_VLMEVALKIT}"
rm -r -- "${STAGED_VLMEVALKIT}"
```

The status command must produce no output. The archive operation preserves upstream executable bits
and omits repository metadata by construction.

## Exclusions and update policy

- `.git/` is source-control metadata and is not copied or digested.
- Upstream `.github/` is deliberately not copied; remove it from the staged source before
  importing or comparing. It has no role in the benchmark runtime and must stay absent.
- `nvbench/` is the single NVIDIA-owned overlay and is not part of the upstream digest.
- Untracked caches, build products, logs, and local environments are not copied because the import
  is produced from `git archive`, not from the source checkout's filesystem. Generated-looking
  paths that upstream itself tracks, including two bytecode files in this snapshot, remain part of
  the exact vendor import.
- No other upstream path is excluded. The verifier still hashes every non-overlay file present,
  so accidentally restoring `.github/` causes a digest mismatch rather than being silently ignored.

An upstream refresh must update the commit, Git tree, file count, digest, and snapshot date
together. It must pass both the verifier and a direct comparison with a clean staged source
after the documented `.github/` exclusion, before review.
