# Training dependencies

The dependency set used for final model training runs, on top of an NGC PyTorch
base image. This is separate from the CI images under `docker/`, which install
into a uv-managed `/opt/venv`; everything here installs into the base image's
system Python.

| File | Purpose |
|---|---|
| `install.sh` | The dependency set as a script. Run it against a container or machine you already have. |
| `Dockerfile.training` | The image build route. Copies this folder and runs `install.sh`. |
| `install_sbatch.sh` | Runs `install.sh` on a Slurm node through pyxis and saves a new squashfs image. |
| `check_frozen_stdlib.py` | Asserts the rebuilt interpreter kept Ubuntu's frozen path modules. Run by the `cpython` section. |

`install.sh` is the sole dependency recipe. `Dockerfile.training` copies this
folder and runs the installer; it no longer duplicates pins and installation
commands. `tools/check_dependency_sync.py` checks that thin-wrapper contract and
that `install_sbatch.sh` uses the same base image. Run it after changing either
installation route.

The Docker build accepts optional BuildKit secrets `gitlab_netrc` and
`gitlab_config`, mounted only during the installer RUN. They provide PAT
credentials and SSH-to-HTTPS URL mappings for private Git dependencies without
putting credentials into arguments, source files, or image layers. The dependency
versions and installer commands are otherwise the same for Docker and Slurm.

The base image is named two ways: `Dockerfile.training` as a registry reference
(`ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.07-py3`) and `install_sbatch.sh`
as an already-imported squashfs (`nvidia+pytorch+26.07-py3.sqsh`). The check
converts the former into the filename enroot writes for it, so bumping one
without the other fails.

The check reads text, not behavior. It confirms the Dockerfile still delegates
instead of growing its own recipe, and that both routes name the same base
image. It cannot prove the two produce the same environment.

## Prerequisites

Two dependencies come from wheels rather than from a repository or an index, and
neither is committed:

| Wheel | Size | Provides |
|---|---|---|
| `nvidia_cutlass_dsl_internal-0.3.0+20260719043000.ce0cac0-cp312-cp312-manylinux_2_28_aarch64.whl` | 343 MB | `cutlass.tensor_utils`, which no published `nvidia-cutlass-dsl` has |
| `gdp_attn_fused-0.5.5+cp.dkg43d3c5b-py3-none-any.whl` | 260 KB | the fused GDP attention kernels, as `gdp_attn` |

**Stage both in this directory before building.** `install.sh` looks for them in
its own directory, and `Dockerfile.training` copies them from
`dependencies/` in the build context, so putting them beside these files
satisfies both routes. `install.sh` checks each exists first and fails naming the
missing one. Set `WHEEL_DIR` to look somewhere else.

The pin block names only the filenames, which carry the versions, so a wheel
cannot change underneath a build without the pin changing. Where to find them is
per route, because a Docker build step runs in an isolated filesystem and cannot
read an absolute host path even when that path is mounted on the build machine.

`.gitignore` carries `dependencies/*.whl`. The 343 MB wheel is why: git keeps
every version forever, so committing it would grow a 587 MB repository by more
than half, again on each bump.

### Advisory GitHub CI build

`cicd-training-container-build` runs alongside the existing CI image build for
maintainer runs on the ARM64 GB300 runner. Tests and the final merge gate do not
wait for it; failures are advisory and excluded from the gate's failed-job scan.
The job builds without publishing an image or exporting a shared registry cache.

GitHub CI passes `INSTALL_INTERNAL_DEPS=0` to skip
`nvidia-cutlass-dsl-internal`, `gdp-attn-fused`, and `one-logger`, including wheel
existence checks and imports that require the skipped wheels. This includes
cuDNN grouped-GEMM API checks: those wrappers need CUTLASS DSL APIs supplied by
the internal wheel. cuDNN's version and core import remain checked. One Logger is only
available from an internal package index that the GitHub runner cannot resolve.
Public CUTLASS DSL and flash-attn-4 remain installed and verified. The flag defaults to `1`, so normal Docker and
Slurm training builds still require and install both wheels and One Logger. No
wheel storage, source mirror, or additional credentials are needed for the skipped packages.
A GitHub warning annotation and job summary name all skipped packages and
state that the advisory build does not validate the full production dependency set.

GitHub CI uses public upstream sources instead of the GitLab forks. These paired
overrides work as Docker build arguments and installer environment variables:

| Repository URL | Commit SHA |
|---|---|
| `NCCL_REPO_URL` | `NCCL_REPO_SHA` |
| `CUDNN_FRONTEND_REPO_URL` | `CUDNN_FRONTEND_REPO_SHA` |
| `TRANSFORMER_ENGINE_REPO_URL` | `TRANSFORMER_ENGINE_REPO_SHA` |

When changing repositories, supply a commit available in that repository as well.
Each unset or empty override falls back to the original training pin. CI's build
arguments, including the public URLs and SHAs, live in `ci-public-build.yaml`.
The workflow reads that file and passes the arguments to Docker. A public build
does not include the private forks' additional changes. Existing version and API
checks unrelated to the skipped internal dependencies still run; incompatible
overrides fail the advisory build. No GitLab token
or mirrored Git source is required.

With no overrides, `bash install.sh` retains the original private pins, installs
both internal wheels and One Logger, runs the internal-wheel verification checks,
and writes the original integer version stamp. The installer's version-stamp behavior is unchanged.

Check option handling without compiling the dependency set:

```bash
python -m unittest discover -s dependencies/tests -v
python tools/check_dependency_sync.py
```

## Versioning

The whole dependency set carries one integer version, `DEPENDENCY_SET_VERSION`,
declared in the pin block of `install.sh`. Both the Docker and Slurm routes run
that installer, so they consume the same version and pins.

**Bump it whenever any pin changes.** It is a label for a set of pins, not a
semantic version: there is no major/minor distinction, just "these exact
versions".

It shows up in three places:

- **In the image.** The last section of the build writes it to
  `/etc/megatron_dependency_set_version`. That section runs last, so the stamp
  exists only if every install above it succeeded. To identify a container:
  `cat /etc/megatron_dependency_set_version`.
- **In the image name.** `install_sbatch.sh` reads the version out of
  `install.sh` rather than repeating it, and names its output
  `nvidia+pytorch+26.07-py3+deps+v<N>.sqsh`. An image name cannot claim a
  version the script did not install.
- **In the table below**, against the versions that build produced.

**Bumping it is a code review responsibility.**
`tools/check_dependency_sync.py` checks the wrapper contract and the base image;
it does not compare against the base branch, so it cannot tell that a pin moved
while the version stayed. Reviewers of any change to `install.sh` or
`Dockerfile.training` should confirm `DEPENDENCY_SET_VERSION` moved with it.

| Version | Date | Notes |
|---|---|---|
| 1 | 2026-08-25 | First pinned set. Every git ref a tag or commit, every `pip install` version-pinned. Built by Slurm job 568861. |
| 2 | 2026-08-26 | Adds flash-attn-4, the GDP wheels and `verify-pins`. The GDP kernel does not compile. |
| 3 | 2026-08-26 | Uninstalls the PyPI `nvidia-cutlass-dsl` to try to fix that. It does not, and it breaks flash-attn-4's metadata. |
| 4 | 2026-08-27 | Reorders the sections after a reference container and changes several things at once. Trains, but more changed than needed. |
| 5 | 2026-08-28 | `quack-kernels` 0.6.1 with cutlass-dsl 4.6.0, no rdma-core build, no `HYBRID_EP_MULTINODE`, sm_103 for TransformerEngine and DeepEP, TransformerEngine at `47f2e05b`. Built by Slurm job 606848. |
| 6 | 2026-09-02 | Moves cudnn-frontend to `c012ba87` and TransformerEngine to `c0460eb7` for `ScaledTanhSReLU` in the fused grouped MLP, with its fused activation recompute. Built by Slurm job 649456. |
| 7 | 2026-09-03 | Moves `flash-linear-attention` to 0.5.1 and adds it to `verify-pins`. Built by Slurm job 651948. |
| 8 | 2026-09-08 | Rebuilds CPython 3.12.3 with the gh-142183 backport, ahead of every other section so the rest of the build runs under it. Built by Slurm job 682956. |
| 9 | 2026-09-08 | Installs NCCL's `nccl_device` headers alongside its library, which nccl-ep's JIT needs and which stayed at the base image's version through versions 1 to 8. Moves TransformerEngine to `2efa03ea`, cudnn-frontend to `01858595` for #906, which also takes the declared version to 1.29.0, and `gdp-attn-fused` to 0.5.0. Built by Slurm job 685601. |
| 10 | 2026-09-09 | Takes cudnn-frontend #906 as a cherry-pick onto `c012ba87` from the internal fork, rather than the 47 commit jump version 9 made, so `nvidia-cutlass-dsl` stays at 4.6.0. Also moves `nvidia-resiliency-ext` to `d23f0a11`. |
| 11 | 2026-09-14 | Moves TransformerEngine to `5edda3fc`, the previous `nemotron4` head plus upstream #3426, still open, squashed into one commit. It exports `mark_qb_bin_bounds_validated`, which `finalize_model_grads.py` already calls behind an ImportError guard; no build had exported it, so the call was silently skipped and the quantile-balancing router re-validated its bin bounds every step, reading them back to the host once per router layer per iteration. Built by Slurm job 747443. |
| 12 | 2026-09-17 | Moves TransformerEngine to `21b0fb7b`. The `nemotron4` branch was rebuilt from the same `2efa03ea` base with two cherry-picks, one commit each: upstream #3426, now merged, replacing the pre-merge revision version 11 carried, and upstream #3517, still open, which threads sharded weights through `GroupedLinear`'s grouped-tensor path. Without #3517 a GTP-sharded expert layer on that path reads `main_grad` off a plain tensor in backward and raises `AttributeError`, which is reachable with `--te-precision-config-file` overriding MTP-MoE to bf16 plus EGTP > 1. The version 11 head is kept reachable as the `nemotron4/v11` tag. Also moves NCCL from the `v2.31.2-1` tag to the internal fork at `78b39cdb`, which is that tag plus the Inspector profiler, and builds the Inspector plugin from the same checkout into `/opt/nccl_inspector`. Built by Slurm job 781192. |
| 13 | 2026-09-21 | Adds Lens with OTel exporters and otelcol-contrib 0.156.0 ARM64. Updates NVRx to PR #406. Also fetches the NCCL pin by sha instead of taking whatever the clone carried: `sirshakd/v2.31.2-profiler-lossy-ts-v8-init-time` was force-pushed past `78b39cdb`, which orphaned the commit and failed the first build of this version at the `nccl` section. Lens sits at `b5f941f`, the `main` head carrying PR #73 and PR #75. PR #73 caches default tracers: without it Lens resolved a tracer per span, and the OTel SDK 1.44 lookup mutates the warning filters, clearing every module's `__warningregistry__` so already-shown warnings reprint once per span. A phase1 run logged 613 repeats of one torch deprecation across 94 iterations with telemetry on, against 1 with it off. Also adds `quart`, `hypercorn` and `msgpack` for the inference server, with the six packages they reach pinned alongside them so a later rebuild cannot resolve different versions under the same set number. Built by Slurm job 826917. Jobs 813883 and 816328 built earlier revisions of this set and their images are kept beside the current one as `v13.build-813883.sqsh` and `v13.build-816328.sqsh`. |
| 14 | 2026-09-25 | Moves TransformerEngine to `1ac66e5a`: the same `2efa03ea` base with upstream #3545, still open, added to the two existing picks. Saved-tensor hooks make `SavedVariable::unpack` rebuild a plain tensor, dropping the `DistributedWeight` subclass, so backward takes the plain-parameter branch and raises `weight was removed while fuse_wgrad_accumulation=True`; #3545 keeps those objects on `ctx` instead. Reachable wherever `--fine-grained-activation-offloading` wraps bf16 GTP-sharded `GroupedLinear` experts, as `ultra/phase1` does. The #3517 pick also moves to the revision merged on 09/25, which adds only a test skip guard. The version 12 head, shared with version 13, is kept as the `nemotron4/v12` tag. Built by Slurm job 871753. |
| 15 | 2026-10-08 | Moves TransformerEngine to `0ded11a3`, the version 14 head plus three fork commits: the fused router reads the selected sigmoid scores back directly instead of adding and subtracting the expert bias, which could erase tiny scores; upstream #3653, still open, keeps router probabilities and their gradient in FP32 through the fused grouped MLP; and upstream #3652, still open, lets `FusedAdam` keep its second moment in FP64, leaving the default kernels unchanged. Moves cudnn-frontend to `58d08f71`, which rounds the srelu pre-activation to the saved `C` dtype before the forward activation so the forward pass and the backward recomputation produce identical FC2 inputs. Moves NVRx to `9c667def`, the head of `tmp/fast-metadata-fixes`, which adds a native `.metadata` writer and reuses cached save plans and metadata; the build now fails if that optional extension did not compile. Built by Slurm job 1154546. |
| 16 | 2026-10-10 | Moves TransformerEngine to `b0c6c97b`, the version 15 head plus two commits: upstream #3417, merged, which fixes TMA shared-memory reuse synchronization in the quantization kernels, and a fork commit that keeps router probabilities in FP32 in the basic, unfused scaled activations, which had rounded them to the activation dtype before scaling and returned their gradient in it; this is the unfused counterpart of #3653. Moves cudnn-frontend to `7d7c071a`, which takes the kernel half of upstream #1283: the block-scaled grouped GEMM wgrad kernel, which the fused grouped MLP uses, fences each TMA descriptor its helper wrote before using it, so back-to-back reuse of the descriptor workspace, as on CUDA Graph replay, cannot read a stale descriptor. Moves NVRx to `402d0a8e`, the head of `tmp/fast-metadata-fixes`, 22 commits ahead of the version 15 pin with nothing dropped. It adds profiling spans in `execute_finalize_fns`, and has each rank send its write results to the coordinator as a table rather than pickled objects, which the native `.metadata` writer reads directly to write `storage_data`. Moves `gdp-attn-fused` to 0.5.5. Its backward state recompute now stores the coarse saved-state checkpoints in FP32, the forward's carry dtype, rather than BF16, and places its roundings where the forward does, so the recomputed states match the forward's; this doubles the size of those checkpoints. It also reads each chunk's total decay from the chunk's last token in the forward, where some tilings had read another row, and takes the CP summary leaf width as a runtime argument, so a new packed-document count no longer recompiles the summary kernels. The build's package check now accepts any abbreviated hash of at least seven characters that prefixes the pin, since NVRx reports eight (`402d0a8e`); job 1184338 failed on that check. Built by Slurm job 1185540. |
| 17 | 2026-10-17 | Moves NVRx to `b49f48fb`, the head of `tmp/fast-metadata-fixes`, 13 commits ahead of the version 16 pin with nothing dropped. The branch's own checkpointing changes are unchanged; the new commits merge NVRx `main`, which adds opt-in timeout diagnostics: with `--ft-timeout-dump-enabled=true`, the rank monitor captures the worker's stacks with py-spy, gdb and cuda-gdb when a selected section times out, before restarting it. It also stops the in-job attribution client polling attrsvc after three consecutive failed requests, and fixes the attribution progress guards, which had recommended stopping jobs 4234179 and 4125659 after a single stalled or unverifiable cycle. Built by Slurm job 1304373. |
| 18 | 2026-10-20 | Current. Moves TransformerEngine to `d73cb805`, the version 17 head plus five commits. Upstream #3595 and #3540, both merged, skip FA4 for deterministic backward on SM12x and select FA4 for D=256 THD attention. A fork commit removes the host syncs from the THD context-parallel P2P backward zero-fill, which sliced on a CUDA tensor and so copied it to the host three times per attention layer per microbatch; gradients are bitwise unchanged. A fork commit adds `qb_bias_space="logit"` to the fused sigmoid router, which selects experts on logits plus bias rather than sigmoid scores plus bias, keeps the unbiased sigmoid scores as mixture weights, and accumulates quantile-balancing histograms in logit units; the default score-space behavior is unchanged. A fork commit keeps the autocast precision state (enabled, recipe, group, calibration and graph capturing) per thread, so an autocast entered in the frozen-encoder prefetch thread no longer changes the precision of modules running in the training thread; the delayed-scaling bookkeeping stays process-wide. Moves NVRx to `ba5cfb48`, the upstream `stable` head, six commits ahead of the version 17 pin with nothing dropped; its checkpointing code is unchanged. PR #459 adds opt-in CUDA-GDB stdin command delivery, keeping batch as the default, reduces startup and capture logging, and gives incomplete upload batches one poll to collect late rank captures. PR #453 removes the deprecated in-process restart support; the in-job `ft_launcher` is unaffected. Built by Slurm job 1356440. |

Versions 2 to 4 were built and used, and each had problems. 2 and 3 could not
compile the GDP kernel; 3 additionally broke flash-attn-4's metadata. 4 fixed
both, but bundled in unrelated changes while the cause was still unknown. 5
keeps only what turned out to be needed. The flash-attn-4 section records what
the failure was.

## Pinned refs

| Package | Ref | Provenance |
|---|---|---|
| causal-conv1d | `cd81f0413cad2fc1e6f17e785ac39f59aae690cd` | main as of 08/25/2026; no releases cut |
| mamba | `e9594ce1c732d97440f0332fdc43170a2294dbfa` | main as of 08/25/2026; no releases cut |
| nvidia-resiliency-ext | `ba5cfb48058059d682d248d641cb95a4f3fab618` | upstream `stable` head as of 10/21/2026. Preserves the native `.metadata` writer, cached save plans and metadata, finalize spans and table-gathered write results; adds PR #459 timeout-diagnostic stdin mode, logging and upload batching, and PR #453 in-process restart removal |
| nemo-lens | `b5f941f373eab88afeb8aa26a64a34cbf32b4365` | SDK and OTLP exporters, at `main`. Carries PR #73, which stops Lens resolving a tracer per span, since in OTel SDK 1.44 that lookup mutates the warning filters and reprints every already-shown warning once per span, and PR #75, which preserves explicit spans across clock inversions |
| cudnn-frontend | `7d7c071a1b5f41d2abf875d50cb0659c21d54183` | head of `nemotron4` on the internal `ADLR/cudnn_frontend` fork: upstream `c012ba87` with #906 cherry-picked on top, then the srelu saved-activation precision fix, then the TMA descriptor acquire fence from #1283 |
| NCCL | `78b39cdb6bcd1568f7cebde80878606655cf06b1` | the `v2.31.2-1` tag plus the Inspector profiler, on the internal `nccl/nccl` fork. Was the head of `sirshakd/v2.31.2-profiler-lossy-ts-v8-init-time` as of 09/02/2026; that branch has since been force-pushed past this commit, which is now held reachable by the `nccl-v2.31.2-inspector-lossy-mode-nemo-v12` tag |
| TransformerEngine | `d73cb8050d4a947b48533185622953b26221958d` | head of `nemotron4` on the internal `ADLR/transformer_engine` fork as of 10/20/2026: the `2efa03ea` base plus upstream #3426, #3517, #3417, #3595 and #3540, all now merged, upstream #3545, #3653 and #3652, still open, the fused-router sigmoid score fix, FP32 router probabilities in the basic scaled activations, the THD CP P2P zero-fill without host syncs, the logit-space router bias and per-thread autocast state, each one commit; its `nccl-extensions` submodule points at the public mirror at `546723ff` |
| Emerging-Optimizers | `v0.4.0` | tag |
| DeepEP | `58397eac493b075b0baea5861ac6e9793d8d7a21` | head of tree on the `hybrid-ep` branch as of 08/25/2026 |
| CPython | `v3.12.3` with `706fd4ec08acbf1b1def3630017ebe55d224adfa` applied | the tag is whatever the base image ships and cannot be chosen freely; the commit is upstream gh-142183, which landed after 3.12.3 |

Every ref is a tag or a commit, and every `pip install` carries a version, so
everything above is pinned: a commit hash pins more tightly than a version
string, since a release can be yanked and re-uploaded but a hash cannot change
contents. Nothing here tracks a moving branch.

What is genuinely unpinned is everything nobody named: transitive dependencies
resolved by pip at build time, the apt packages in the `debugging` section, and
whatever the base image ships. Two builds of the same `DEPENDENCY_SET_VERSION`
can therefore differ in those, which is how `botocore` moved between two
otherwise identical images.

`cudnn-frontend` is built from source rather than installed from PyPI, because
the wanted commits are ahead of any release. The version string cannot separate
the two: the 1.28.0 release was tagged on a release branch a day before
`c012ba87` landed on `develop`, and both declare 1.28.0. `verify-pins` therefore
checks the wrapper signatures rather than the version, since a released 1.28.0
wheel would satisfy a version assertion and then fail the TransformerEngine
build.

This source build uses pip build isolation because the base image lacks
`cmake`, `ninja` and `pybind11`.

It is installed before `transformer-engine` because TE compiles against it.
TE vendors no copy of its own: its `setup.py` passes
`-DCUDNN_FRONTEND_INCLUDE_DIR` resolved from the installed distribution, so
whatever `nvidia-cudnn-frontend` is present is what TE builds against. TE also
does not overwrite it: `nvidia-cudnn-frontend>=1.25.0` is a build-system
requirement rather than a runtime dependency, and TE is installed with
`--no-build-isolation`, so pip leaves the environment's copy alone.

Because TE hard-errors when it cannot find `include/cudnn_frontend.h` in that
distribution, `install.sh` checks for the header with the same lookup right
after building it. A source build that packages headers differently from the
released wheel then fails immediately instead of an hour into the TE compile.

What can still shift between builds is the base image: `flash_attn`, `einops`
and `wandb` come from it rather than from `install.sh`.

The `mistral-dataloader` section runs before Lens, NVRx and the collector.
Its packages are included in the runtime constraints used by those installs.

The list is inlined from `scaling_ladder/mistral_requirements.txt` rather than
read from it, so the two can drift; nothing checks `install.sh` against that
file. Three entries are deliberately omitted. `boto3` and
`tiktoken` are pinned earlier in `install.sh`, and an unpinned entry in this
section would be free to upgrade past those pins. `numpy<2.0` is dropped
because nothing in the list asks for numpy 1, while current
`opencv-python-headless` requires numpy 2: keeping the cap resolved numpy back
to 1.26.4 under a PyTorch built against numpy 2, walked opencv back from 5.0 to
4.11, and pulled in scipy. `fasttext` is a compiled extension declaring only a
bare `numpy`, so a numpy 2 ABI problem there would show up at import rather
than in resolution. Running the section in a throwaway container on 2026-08-25
showed `fasttext` building from source into a `cp312-linux_aarch64` wheel and
importing, along with `cv2`, `imagehash`, `soundfile`, `soxr` and `av`, against
numpy 2.1.0. The section ends by importing exactly those six, so a future wheel
with a numpy 1 ABI fails the build instead of surfacing mid-training.

That run also downgraded `fastjsonschema` from the base image's 2.21.2 to the
pinned 2.19.1, the only exact version this section carries.

### nccl

The `nccl` section replaces the base image's NCCL with a source build of the
internal fork at `78b39cdb`, which is the `v2.31.2-1` tag plus the Inspector
profiler, and the library is not the only thing that has to move. NCCL ships
device-side headers under `/usr/include/nccl_device`, and nccl-ep's JIT compiles
its kernels against those, so they describe the layout the kernels assume for
`ncclDevComm` and its members.

**Versions 1 through 8 replaced `libnccl.so` and `nccl.h` but not
`nccl_device/`.** The device headers stayed at whatever the base image's
`libnccl-dev` shipped, which is 2.30.7 era. `ncclResourceWindow_vidmem_t` shrank
from 64 bytes to 16 between the two releases, so every field after it in
`ncclDevComm` sits at a different offset: kernels compiled against the stale
headers read `lsaMultimem`, `lsaBarrier` and the rest from the wrong bytes, and
the intra-node LSA dispatch kernel dies on an illegal or misaligned address. It
is deterministic, and it looks like an NCCL version problem rather than a header
problem, which is what made it hard to place.

Confirmed by fixing it in an already-built image rather than by reasoning:
copying the real 2.31.2 `nccl_device/` headers over the stale ones and rebuilding
`ep_bench` makes the crash go away and the run validate, repeatably, and the
untouched image crashes again immediately afterwards.

It went unnoticed for eight versions because the training workloads dispatch MoE
through HybridEP rather than NCCL EP, so nothing exercised those kernels outside
`ep_bench`. Nothing in the build noticed either, which is the part worth fixing:
the library announces its version and `verify-pins` asks it, but a header carries
no version to ask for.

The section therefore ties the headers to the tree that produced the library,
and `verify-pins` checks the same thing again after every later install. Byte
identity is the whole guarantee: the headers and the library come out of one
`make`, so headers that match cannot describe a different library than the one
installed beside them. The check catches a copy that did not happen, a stale
header left in place, and an orphan a future release stops shipping.

### NCCL fork and the Inspector plugin

The fork exists for the NCCL Inspector: a profiler plugin that reports per-collective
telemetry, plus the device-side instrumentation it reads. Neither is upstream, so
an image that wants the Inspector cannot get it from a released NCCL. Until this
version, every inspector container was built by hand on top of a finished image,
which is a second build nobody else can reproduce.

The plugin is built from the same checkout as the library, not fetched
separately, because the two share a descriptor layout: the plugin reads
device-side records whose fields the library writes, and a plugin built from a
different commit misreads every one of them. Its Makefile is standalone and
compiles against the profiler interface headers vendored beside it, so the
checkout is the only thing tying the two together; there is no header path to
pass it. Building both from one clone makes that pairing structural rather than
something a reader has to verify.

It installs to `/opt/nccl_inspector`, deliberately off the loader path. HPC-X
ships a `libnccl-profiler-inspector.so` of its own under
`nccl_spectrum-x_plugin/lib`, which *is* on the loader path, so a job that names
the plugin by soname silently gets theirs instead of this one. Jobs point
`NCCL_PROFILER_PLUGIN` at the absolute path.

Nothing in the build sets `NCCL_PROFILER_PLUGIN`, so a job that does not ask for
the profiler loads nothing and pays nothing: the plugin is an artifact in the
image, not a default. Neither build route leaves `NCCL_INSPECTOR_DIR` in the
image environment. `install.sh` exports it for its own run and cannot do more,
since it runs inside a container somebody else started, and the Docker route now
runs that same script in one `RUN` rather than setting `ENV`, so the variable
does not outlive the layer. Consumers should name the path rather than read it
from the environment.

`ncclGetVersion` reports `2.31.2` for the fork and for the tag it branched from
alike, so it cannot distinguish an image built from this fork from one built
from upstream. The build therefore passes `NCCL_SUFFIX=-g<commit>`, which reaches
the `NCCL_SUFFIX` define in `nccl.h` and the version string NCCL logs at init
(not the soname), and `verify-pins` greps the installed header for it. The
version assertion reads `NCCL_EXPECTED_VERSION` rather than parsing `NCCL_REF`,
which no longer carries a version now that it names a commit.

Cloning uses HTTPS rather than SSH, unlike the two `ADLR` forks. `nccl/nccl` is
internal-visible, so an unauthenticated read works: with no key and no token in
the container, `ls-remote` succeeds against it and is refused for
`ADLR/transformer_engine`. That leaves the credential requirement on the
TransformerEngine and cudnn-frontend sections alone rather than adding a third
reason a CI runner needs a key.

### TransformerEngine fork

TE is pinned to `nemotron4` on the internal `ADLR/transformer_engine` fork
rather than to upstream. That branch adds ten commits on top of upstream
`main` at `ea1a165d`:

| Commit | Contents |
|---|---|
| `368173cd` | Squash of unmerged upstream PR #3407, cuDNN deterministic dprob in the fused grouped MLP. Pairs with cudnn-frontend PR #521, which the pinned `01858595` already contains. |
| `60e5876c` | Squash of unmerged upstream PR #3355, MXFP8 dispatch returning a plain torch tensor. Upstream marks it "[Do not merge]"; it is a temporary workaround. |
| `5e69011f` | Repoints the `3rdparty/nccl-extensions` submodule at the internal repository at `4d3461ec`. |
| `d1b2dcf8` | Squash of unmerged upstream PR #3397, typing the grouped wgrad from `main_grad` for distributed weights. |
| `47f2e05b` | Repoints `3rdparty/nccl-extensions` at the public mirror at `546723ff`, which snapshots the pull-push `topk_idx` per handle so the nccl-ep combine reduce reads per-handle routing race-free. |
| `b93d2550` | Squash of unmerged upstream PR #3463, `ScaledTanhSReLU` wired into the fused grouped MLP. Needs the `tanh_clamp_scale` wrappers from cudnn-frontend #858, which the pinned `01858595` carries and the 1.28.0 release does not. |
| `c0460eb7` | Squash of unmerged upstream PR #3473, fused activation recompute for `ScaledTanhSReLU`. The cuDNN dsrelu kernel's `d_srelu` regeneration applies the clamp, so the clamped `fc2_x` no longer has to be saved. |
| `39e61a1` | Upstream PR #3221, no-load-balance THD all-gather context parallelism. Merged upstream. |
| `523a29c` | Upstream PR #3149, flash-attn-4 for context-parallel attention. Merged upstream. |
| `2efa03e` | Upstream PR #3438, follow-up hardening of the #3221 changes. Merged upstream. |

The five pull requests squashed into the first seven commits are open and can be
force-pushed, so the upstream commits they were taken from may cease to exist.
Drop each commit once its change lands upstream and arrives by another route.

The last three are different: they are merged upstream and were cherry-picked
ahead of the fork moving to a newer upstream base. They go away by themselves
the next time `nemotron4` rebases past them, rather than needing to be dropped.

**Building this branch needs `gitlab-master` credentials inside the container**,
for the fork itself. Since version 10 the `cudnn-frontend` fork needs them too,
where that clone used to be public. Its submodules do not: `47f2e05b` moved
`nccl-extensions` to the public mirror, and `cutlass`, `googletest` and the
submodule's own `third_party/nccl` were already on public GitHub. TE's
`setup.py` runs `git submodule update --init --recursive` during the build and
statically links `libnccl_ep.a` into `libtransformer_engine`.

That static link is why `nccl` installs before `transformer-engine`: TE compiles
`libnccl_ep.a` against whatever NCCL headers exist at its build time, so
upgrading NCCL afterwards would leave TE built against one NCCL and running
against another.

Note that the workloads this set is built for dispatch MoE through HybridEP
(`--moe-flex-dispatcher-backend hybridep`), not NCCL EP, so the nccl-ep code
path is built but not exercised.

### flash-attn-4

Installed in three steps so `nvidia-cutlass-dsl` settles on exactly 4.6.0 with
its CUDA 13 libraries. The `cu13` extra comes first because the metapackage
otherwise installs only CUDA 12 libraries, and the later commands repeat the
4.6.0 pin so `quack-kernels` and `flash-attn-4` cannot move it. `quack-kernels`
0.6.1 is the release that wants 4.6.0; 0.6.2 wants 4.6.1, 0.6.3 wants `~=4.6.0`
and 0.6.4 pins 4.6.2.

**Hold `quack-kernels` at 0.6.1.** 0.6.4 adds `quack/dsl/mixed_constexpr_if.py`,
which `quack/dsl/__init__.py` imports and which monkey-patches
`cutlass.base_dsl.ast_preprocessor` for the whole process at import. Its rewrite
of `if const_expr(S) or D:` drops the body's name bindings when the statement has
no `else`, and `gdp_attn`'s chunked kernel is exactly that shape, so it fails to
compile with `DSLUserCodeError TYPE_UNSTABLE_JOIN`. `gdp_attn` never imports
quack; it is reached because `megatron/core/transformer/attention.py` imports
`flash_attn.cute`, which does. Verified 2026-08-27: v3 (0.6.4) never reaches
iteration 1, v5 (0.6.1) trains. `gdp_repro/gdp_fa_repro.py` reproduces it on one
node in about a minute.

The section runs before transformer-engine, so later installs could move these
versions. The `verify-pins` section asserts them after every install rather
than relying on the ordering.

**`nvidia-cutlass-dsl` stays at 4.6.0, and the cudnn-frontend pin is what
keeps it there.** Version 9 took cudnn-frontend #906 by moving 47 commits along
`develop`, which crossed a commit making `nvidia-cutlass-dsl[cu13]>=4.6.2` a
required dependency where it had been `>=4.5.0` in an optional extra. That floor
is above the 4.6.0 this section settles on, so pip resolved to 4.7.1 and left
`quack-kernels` 0.6.1's `==4.6.0` unsatisfied. Version 10 takes #906 as a
cherry-pick onto `c012ba87` instead, which carries the epilogue change without
the dependency floor. `verify-pins` asserts the version either way.

Version 9 shipped 4.7.1 and worked: Slurm job 686999 compiled `gdp_attn`'s
chunked kernel on that image with `flash_attn.cute` imported first, so quack's
AST patching was active, and got finite output. So the pairing is not known to be
harmful, only undeclared.

TransformerEngine reads `flash-attn-4` and
`nvidia-cutlass-dsl` at runtime rather than at build time, so the position does
not affect what TE compiles.

`flash-attn-4` installs into `flash_attn/cute/` and does not own
`flash_attn/__init__.py`, so it sits alongside the base image's flash-attn 2
rather than replacing it. `flash_attn.__version__` still reports flash-attn 2's
version and says nothing about whether flash-attn-4 is present, which is why the
section imports `flash_attn.cute` instead. TE imports the same path,
`flash_attn.cute.interface`.

**This upgrade downgrades protobuf.** The cutlass-dsl 4.6.0 libraries declare
`protobuf<7,>=6.30.2`, so protobuf drops from the base image's 7.35.1 to 6.33.6.
`grpcio-tools` declares `protobuf<8.0.0,>=7.35.1`, and the two constraints
cannot both hold; 4.6.2 carries the same cap, so no newer cutlass-dsl avoids it.
cutlass-dsl 4.5.2, which the base image ships, declares no protobuf bound, so
this arrived with the upgrade. `grpc_tools` and `protoc` still import on 6.33.6.
The Lens section pins grpcio-tools to 1.76.0, which supports protobuf 6.33.6.

### cutlass-dsl-internal

`gdp-attn-fused` imports `cutlass.tensor_utils`, which no published
`nvidia-cutlass-dsl` provides. The internal wheel is a separate distribution
that ships its own `cutlass` through a `.pth`, so it installs alongside the PyPI
one and wins the import.

**The PyPI `nvidia-cutlass-dsl` is deliberately left installed.** Version 3 of
this set uninstalled it, on the theory that the two coexisting caused
`gdp_attn`'s `TYPE_UNSTABLE_JOIN`. That was wrong: the cause was `quack-kernels`
0.6.4, and removing the PyPI distribution also removed its metadata, after which
flash-attn-4 reports `No package metadata was found for nvidia-cutlass-dsl`.
Confirmed on an image carrying both that `cutlass` still resolves to
`nvidia_cutlass_dsl/dsl_packages` and `gdp_attn` still gets its `tensor_utils`,
which is what `verify-pins` now asserts.

### hybrid-ep

`rdma-core` is not built from source. An earlier revision did, to supply headers
for DeepEP; the base image already ships them at `/usr/include/infiniband`, so
the build was redundant, and it left a `libibverbs` that could not find a
matching provider. Every rank then warned that it could not load
`libmlx5-rdmav59.so`.

`HYBRID_EP_MULTINODE` is not set either. HybridEP is used only within an NVL
domain here, so DeepEP's internode transport is not exercised.

`DEEPEP_TORCH_CUDA_ARCH_LIST` and `TRANSFORMER_ENGINE_CUDA_ARCHS` both name
`sm_103`. The cluster is GB300; an earlier `9.0 10.0` list omitted it entirely,
so DeepEP carried no native GB300 kernels. Both lists drop `sm_90`, so nothing
here targets H100.

### cpython

The interpreter itself is rebuilt from source and swapped in, which nothing else
in this set does. `CPYTHON_REF` names the version the base image already ships,
so this is not a version upgrade. The `706fd4ec` backport of gh-142183 keeps one
popped data-stack chunk cached per thread state
instead of unmapping it and mapping another on the next call. The pathology it
removes is a call depth that repeatedly lands on a chunk boundary, which costs a
map and an unmap per call.

The installer also copies the base image's `genericpath.py`, `posixpath.py` and
`ntpath.py` into the source tree before `./configure`, so the build freezes those
sources rather than vanilla ones. Ubuntu's tarfile security backport calls
`os.path.ALLOW_MISSING`, which vanilla frozen path modules do not provide, and
the `__file__` values are no help in telling the two apart: they can point at the
patched files on disk while the frozen code is what runs. `genericpath` is the
one that matters, since it defines `ALLOW_MISSING` and `posixpath` imports it.
The `grep` beside the copy rejects a base image that ships no such backport
rather than building another inconsistent interpreter. The error this avoids
appears when pip extracts an sdist, not when Python or Torch first imports.
Nothing here disables frozen modules or relaxes the extraction filter, and it
takes no build argument or Docker `COPY` input; the modules come from the base
image.

`check_frozen_stdlib.py` asserts the result at the end of the section, run from
outside the CPython source tree so that the installed modules are the ones under
test. It requires `FrozenImporter`, exercises `realpath(..., strict=ALLOW_MISSING)`
along with normal and pip archive extraction, and checks that escaping paths and
links are still rejected. It prints `FROZEN_STDLIB_CHECK_OK`.

**`CPYTHON_REF` is tied to the base image and nothing enforces that.** Rebuilding
3.12.3 under a base that ships a different 3.12 patch release would put one
interpreter under extensions compiled against another. The section asserts the
version it produced rather than trusting the pin to have moved with the image.

The section runs first, ahead of `nccl`. It does not change what any later
section compiles against, since the headers under `/usr/include/python3.12` stay
the base image's. What running first buys is exercise: every pip install and
source build below then runs under the rebuilt interpreter, so a broken swap
fails ten minutes into the build rather than an hour in or at training time.

That is safe because of where the patch puts its field. `datastack_cached_chunk`
lands after `datastack_limit` in `struct _ts`, and the only members after it are
`exc_state` and `root_cframe`, both documented as not to be touched outside init
and both reached through the `exc_info` and `cframe` pointers the interpreter
fills in itself. Every member an extension reads directly keeps its offset,
including `py_recursion_remaining`, `cframe`, `current_exception`, `exc_info`,
`dict`, `gilstate_counter` and `context`. Wheels compiled against the unpatched
header, `torch` among them, are unaffected.

**Copying the built extension modules into `lib-dynload` is load-bearing; do not
drop it.** It looks redundant, because the backport only touches core interpreter
code and the modules therefore need no rebuild. The reason it is required is that
Ubuntu and a stock CPython build split modules differently between built-in and
shared: Ubuntu compiles `math`, `_blake2` and others into the interpreter, while
a stock build ships them as `.so` files. Replacing `libpython` without supplying
those `.so` files leaves them unavailable by either route, and the first thing to
notice is `ModuleNotFoundError: No module named 'math'` out of `urllib.parse`.
Removing the copy was tried on 2026-09-08 and failed exactly that way, in Slurm
job 682740.

The build environment has no development headers for sqlite3, ssl, lzma,
readline or uuid, so those modules do not build here and the distribution's own
copies stay in place. That is fine for the ones Ubuntu ships as `.so` files and
is worth remembering if a future base image moves any of them in-tree.

Both interpreters report 3.12.3, so the version cannot tell them apart, the same
problem `cudnn-frontend` has. `sys._git` can: it reads `('CPython', '', '')` on
the base image, which is built from a source tarball, and names the checkout for
a build from a git tree. The section asserts that against the tag it built,
along with the hash of `libpython3.12.so.1.0`, and `verify-pins` asserts both
again after every later install. That second check is not redundant now that
`cpython` runs first: `gdb` in the `debugging` section depends on
`libpython3.12`, so an `apt-get install` that pulls a newer one in would write
over the library this section built.

The section then imports `torch` in the environment training will see, rather
than the stripped one the identity check uses. That is the check that catches a
swap which left the interpreter unable to load a module, and it is how the
`lib-dynload` removal described above was found.

The payoff is not asserted anywhere, because a timing threshold inside a build is
a flake waiting to happen. It was measured instead, in Slurm job 683551, by
recursing to a fixed depth repeatedly and sweeping that depth, since the cost
appears only where a chunk boundary falls:

| | floor | worst depth | penalty |
|---|---|---|---|
| v7, unpatched | 24.8 ns/call | 167.9 ns/call at depth 2500 | 6.76x |
| v8, patched | 25.1 ns/call | 78.3 ns/call at depth 3100 | 3.12x |

The floors match, so the rebuilt interpreter costs nothing in the common case,
and the penalty at the bad depths drops by roughly two thirds. It does not
disappear: the patch caches one chunk per thread state, so a stack deep enough
to straddle several boundaries still maps and unmaps the rest.

A first attempt swept depths 90 to 130 and found nothing, because the root chunk
is never popped and holds far more frames than that. Sweep into the thousands.

```bash
python - <<'PY'
import sys, time
sys.setrecursionlimit(20000)

def deep(n):
    return 0 if n == 0 else deep(n - 1)

rows = []
for depth in range(100, 4000, 50):
    deep(depth)
    start = time.perf_counter()
    for _ in range(4000):
        deep(depth)
    rows.append((depth, (time.perf_counter() - start) / 4000 / depth * 1e9))
floor = min(r[1] for r in rows)
worst = max(rows, key=lambda r: r[1])
print(f"floor={floor:.1f}ns/call worst=depth{worst[0]} {worst[1]:.1f}ns/call ratio={worst[1] / floor:.2f}")
PY
```

`sys._git` reads `('CPython', 'tags/v3.12.3-dirty', 'f6650f9')` in the built
image, against `('CPython', '', '')` in the base. Both halves are asserted, in
the section and again in `verify-pins`. The revision says the pinned tag is what
was cloned, and the `-dirty` says the tree was modified before it was built,
which now includes the performance backport and the copied Ubuntu path modules.
The dirty marker alone cannot distinguish those modifications; explicit patch
application and the frozen-path runtime check establish their respective behavior.
An unmodified rebuild of the same tag would report `tags/v3.12.3` and fail.

That is a marker for the tree having been changed, not for this specific change.
What pins it to gh-142183 is `patch` exiting non-zero on a failed hunk earlier in
the section, which stops the build. Version 8 was built before these assertions
were added, in Slurm job 682956; the values above are read from that image and
are what the assertions were written against.

## NVRx

NVRx uses `--no-deps` to keep the pinned Lens 0.2.0 source despite its Lens >=0.3
requirement.

## Lens

Lens is installed with its SDK and OTLP exporters.

Update the MLM Lens API calls before using this image. Lens no longer accepts
`span_group_cls` in `NemoLensConfig.from_env()` or `rank` and `world_size` in
`setup_telemetry()`. These calls also fail when telemetry is disabled.

## Collector

`/usr/local/bin/otelcol-contrib` is installed with two configs:

- `/etc/otelcol-contrib/otelcol-platform.yaml`: forward traces to `OTEL_NODE_LOCAL_OTLP_ENDPOINT`.
- `/etc/otelcol-contrib/otelcol-json-platform.yaml`: also write `${OTEL_ARCHIVE_DIR}/job-${SLURM_JOB_ID}/node-${SLURMD_NODENAME}.otlp.zstd` using zstd level 3 and file rotation.

The launcher must start one collector per node, create the archive directory, and
route workers to OTLP/gRPC port 4317. With rotation enabled, the file exporter
appends to an existing archive on restart when the archive directory, job ID,
and node name are unchanged, rotating at the size limit.

Each node/job rotates at 16 MiB and retains every rotated file. Both
`max_backups: 0` and `max_days: 0` are explicit: there is no count-based or
age-based deletion. Do not omit `max_backups`, because the collector defaults
to keeping 100 backups when rotation is enabled.

The files contain OTLP JSON records compressed individually with zstd, each
preceded by a four-byte big-endian compressed-length prefix. Decode each frame
before parsing JSON; they are not plain JSONL or standalone `.zst` streams.
Keep `exporter.file.nativeCompression` disabled: in collector 0.156.0 its
file-level stream can split JSON records across rotated files. Default framing
keeps each record intact during rotation and requires no extra launcher flag.

## Building

As an image:

```bash
docker build -f dependencies/Dockerfile.training -t megatron-lm:training .
```

On oci-aga, where enroot is absent from the login node and pyxis does the build.
`install_sbatch.sh` reads `install.sh` from
`nemotron4/code/dependencies/`, so pull this branch there, stage the two
wheels beside it, and submit:

```bash
ssh oci-aga "cd /scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron4/code \
  && git pull && sbatch dependencies/install_sbatch.sh"
```

The NGC base image is read from `nemotron4/images/` and built images are
written there too. Job logs go to `dependencies/logs/` beside these scripts;
`.gitignore` keeps them out of the repository.

`install_sbatch.sh` refuses to start if its `OUTPUT_IMAGE` already exists, since
`--container-save` overwrites and other jobs may be running against that image.
Set a new name rather than deleting the old one.

**A build that does not finish still produces an image.** `--container-save`
writes the container filesystem out whatever the step exited with, so a failure
partway through, or an `scancel`, leaves a file that looks like a finished image
and is not one: the only thing separating them is the version stamp, which the
last section of `install.sh` writes and which reading requires mounting the
image. `install_sbatch.sh` therefore saves to
`...+deps+v<N>.partial-<jobid>.sqsh` and renames to `...+deps+v<N>.sqsh` only
after `install.sh` returns zero, so the name a job is given is a name that
finished. On a non-zero exit it deletes the partial itself. The rename is a
hard link, which fails when the name is taken, so a build that finishes after
another one claimed the same version stops instead of replacing an image that
jobs may be reading; its own image is left under the partial name.

A cancelled build cannot: Slurm kills the script, and pyxis writes the image
during step teardown, so anything the script tried to do about it would be
racing that write. Those files are left behind on purpose, named after the job
that produced them. They block nothing, and the next run lists them as a
warning. Delete them once no job is reading them.

## Generated image

Versions below are read from inside the built image, not from the build log.

**`images/nvidia+pytorch+26.07-py3+deps+v12.sqsh`** (26.9 GiB)

| | |
|---|---|
| Base | `nvidia+pytorch+26.07-py3.sqsh` |
| Built | 2026-09-17, Slurm job 781192 on oci-aga, 54m |
| Stamp | `/etc/megatron_dependency_set_version` reads `12` |
| Path | `/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron4/images/` |

Runtime:

| | |
|---|---|
| Python | 3.12.3, rebuilt by the `cpython` section; `sys._git` reads `('CPython', 'tags/v3.12.3-dirty', 'f6650f9')` |
| PyTorch | 2.13.0a0+9186a08b2c.nv26.07 |
| CUDA | 13.3 |
| NCCL | 2.31.2 from the internal fork at `78b39cdb`, built with `NCCL_SUFFIX=-g78b39cdb6b`, with its `nccl_device` headers (the base image ships 2.30.7) |
| NCCL Inspector | `/opt/nccl_inspector/libnccl-profiler-inspector.so`, off the loader path; jobs name it in `NCCL_PROFILER_PLUGIN` |

Installed by `install.sh` (290 packages total in the image):

| Package | Version |
|---|---|
| causal_conv1d | 1.7.0 |
| mamba_ssm | 2.3.2.post1 |
| nvidia-resiliency-ext | 0.8.0.dev204+ba5cfb48 |
| nvidia-cudnn-frontend | 1.28.0 |
| transformer_engine | 2.20.0.dev0+d73cb805 |
| emerging-optimizers | 0.4.0 |
| deep_ep | 1.2.1+58397ea |
| transformers | 5.15.1 |
| one-logger | 1.5.5 |
| py-spy | 0.4.2 |
| pynvml | 13.0.1 |
| quack-kernels | 0.6.1 |
| nvidia-cutlass-dsl | 4.6.0 |
| nvidia-cutlass-dsl-internal | 0.3.0+20260719043000.ce0cac0 |
| flash-attn-4 | 4.0.0b19 |
| gdp-attn-fused | 0.5.5+cp.dkg43d3c5b |
| flash-linear-attention | 0.5.1 |
| fla-core | 0.5.1 |
| opencv-python-headless | 5.0.0.93 |
| fasttext | 0.9.3 |
| numpy | 2.1.0 (unchanged from the base image) |

Against version 9: `nvidia-cutlass-dsl` back to 4.6.0 and `nvidia-cudnn-frontend`
back to 1.28.0, both from taking #906 as a cherry-pick rather than a jump along
`develop`, and `nvidia-resiliency-ext` forward to `0.7.0.dev125+d23f0a1`.

Every guard fired during the build:

```
cpython 3.12.3 from f6650f9 | libpython fa8a91238060
rebuilt interpreter imported torch 2.13.0a0+9186a08b2c.nv26.07 and numpy 2.1.0
nccl_device headers match the 78b39cdb6bcd1568f7cebde80878606655cf06b1 build
cudnn-frontend headers at /usr/local/lib/python3.12/dist-packages/include
flash-attn 2.7.4.post1+git5231d95fe1.54157505
gdp-attn-fused 0.5.5+cp.dkg43d3c5b
imported fasttext, cv2, imagehash, soundfile, soxr, av against numpy 2.1.0
pins held: {'flash-attn-4': '4.0.0b19', 'quack-kernels': '0.6.1', 'nvidia-cudnn-frontend': '1.28.0', 'flash-linear-attention': '0.5.1', 'nvidia-cutlass-dsl': '4.6.0'} | numpy 2.1.0
cudnn-frontend carries tanh_clamp_scale in both srelu wrappers
cutlass-dsl PyPI 4.6.0 | internal 0.3.0+20260719043000.ce0cac0 | gdp-attn-fused 0.5.5+cp.dkg43d3c5b
nccl 2.31.2
inspector plugin loads: /opt/nccl_inspector/libnccl-profiler-inspector.so
nccl_device headers match the 78b39cdb6bcd1568f7cebde80878606655cf06b1 build
cpython 3.12.3 from f6650f9 | libpython fa8a91238060
```

`nvidia-resiliency-ext` is not in that `pins held` set even though its version
string carries the commit hash the way `transformer_engine`'s does, so the bump
to `d23f0a11` was confirmed by reading the built image rather than by the build.

Slurm job 692113 ran 250 iterations of the 32 node, 128 GPU sanity config on this
image at 2247.7ms per step, against 2249.0ms for the same job before the
resiliency-ext bump and 2262.5ms on version 9. It uses `--mock-data`, so it says
nothing about numerics.

Inherited from the base image rather than installed:

| Package | Version | How |
|---|---|---|
| flash_attn | 2.7.4.post1+git5231d95fe1.54157505 | not installed; presence checked |
| einops | 0.8.2 | pinned to the version the base image carries |
| wandb | 0.28.1 | pinned to the version the base image carries |

`install.sh` does not install `flash-attn`. The base image ships an NVIDIA git
build, and any PyPI pin would eventually replace it with a slower stock source
build. The script reads its version instead, which needs no GPU, so a base image
that stops shipping `flash-attn` fails the build rather than surfacing at
training time.

### Known warnings from this build

`mamba_ssm 2.3.2.post1` declares `quack-kernels>=0.3.4` and `tilelang==0.1.8`.
`tilelang` is not installed; `quack-kernels` is, but the `flash-attn-4` section
installs it after mamba, so pip reports it unmet at mamba's build time and
satisfied in the finished image. `install.sh` passes `--no-deps` to the mamba
build on purpose, to stop it pulling its own `causal-conv1d` wheel over the
source build, and that suppresses these two as well. pip reports the unmet
requirements as a resolver error and the build continues. Any mamba code path
reaching for those kernels will fail at import.

The build replaces four things the base image already carried, which is the
intent of pinning them:

| Package | Base image | v6 |
|---|---|---|
| `transformer_engine` | 2.17.0+2e559f06 | 2.20.0.dev0+c0460eb7 |
| `nvidia-resiliency-ext` | 0.6.0 | 0.7.0.dev105+4ab4315 |
| `fastjsonschema` | 2.21.2 | 2.19.1 |
| NCCL (the shared library, not a wheel) | 2.30.7 | 2.31.2 |

`transformers 5.15.1` also downgrades `tokenizers` from 0.23.1 to 0.22.2, and
pip notes that `torch` wants `setuptools<82` while the environment carries 84.
Neither stopped the build.
