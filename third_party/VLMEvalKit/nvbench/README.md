# Megatron VLM evaluation with VLMEvalKit

`nvbench` runs the checked-in VLMEvalKit benchmark suite against an explicit
Megatron checkpoint. A single submission command groups benchmarks by their
configured Slurm resources and adds CPU-only scoring jobs where an external LLM
judge is required. Model and suite settings are shared across clusters; Slurm
topology, container locations, and site-specific runtime environment stay in a
cluster config and a short dated launcher.

## Quick start

Run these commands from the `nv-mistralai-megatron` repository root.

### 1. Check the prerequisites

You need:

- a checkpoint root visible from every allocated node;
- authenticated access to the pinned Hugging Face tokenizer;
- a shared operator workspace for the dataset cache and results;
- cluster-qualified Megatron server and VLMEvalKit client SquashFS images; and
- a cluster config matching the local Slurm account, partitions, node shape,
  and resource limits.

Container paths are site-specific. `nvbench` consumes existing images; it does
not build or download them.

### 2. Create a run config

Copy the generated-answer example:

```bash
NVBENCH=third_party/VLMEvalKit/nvbench
cp ${NVBENCH}/configs/run_configs/n4_v6_example.yaml \
  ${NVBENCH}/configs/run_configs/my_checkpoint.yaml
```

Edit only the run identity, checkpoint, cache, and output fields for a normal
checkpoint evaluation:

```yaml
schema_version: 1
run:
  name: my_checkpoint
model_config: ../models/n4_v6.yaml
suite_config: ../suites/n4_v6.yaml
cluster_config: ../clusters/cmh.yaml
checkpoint:
  path: ${workspace}/training_run/checkpoints
  iteration: 8398
paths:
  cache_root: ${workspace}/vlmevalkit_cache
output:
  root: ${workspace}/eval_runs/my_checkpoint
  resume: false
```

Use `../clusters/cmh.yaml` with the CMH helper and
`../clusters/aga.yaml` with the AGA helper. The selected profile must match the
cluster where the command is submitted.

`checkpoint.path` is the checkpoint root, not an `iter_*` directory. The
evaluator selects the configured iteration explicitly rather than deriving the
requested endpoint from `latest_checkpointed_iteration.txt`. Some Megatron
versions still open that tracker during checkpoint bootstrap, so when it is
present it must be readable and should agree with the configured iteration.

Symlinked checkpoint roots are common on shared clusters and are supported:
nvbench resolves the configured root before validating it. A per-iteration
`iter_*` symlink must currently resolve beneath that same root, however. An
iteration symlink that points outside the resolved checkpoint root is rejected
as a provenance guard. Do not assume an external one-iteration symlink view is
accepted merely because the target checkpoint itself is readable.

Current consolidated DCP checkpoints store common state inside the distributed
checkpoint and need no additional setting. Some legacy and merge-produced
checkpoints instead contain a standalone `common.pt`. If and only if that
checkpoint comes from a trusted source, opt in explicitly:

```yaml
checkpoint:
  path: ${workspace}/trusted_checkpoints
  iteration: 40000
  trusted_standalone_common_state: true
```

This setting enables PyTorch's full-pickle compatibility mode in the server
process so it can restore the trusted Python objects in `common.pt`. Full pickle
loading can execute code; never enable it for an untrusted checkpoint. Omitting
the setting preserves PyTorch's secure weights-only default. Both layouts may
exist at the same time, so choose based on the selected `iter_*` directory: a
standalone `common.pt` uses the explicit trusted opt-in, while consolidated DCP
(`.metadata` plus `__*.distcp`, without `common.pt`) uses the default.

`--workspace` supplies `${workspace}` interpolation and defines the permitted
output root. It is not the repository checkout. Absolute input paths are also
accepted, but `output.root` must remain below the workspace.

Shot count and demonstration construction belong to each dataset in the suite,
so one suite can mix them without another run config:

```yaml
benchmarks:
  - dataset: OCRBench
    class: OCRBench
    num_shot: 2
    demonstration:
      type: benchmark
      include_images: true
    # result, judge, and resource fields follow
  - dataset: MMMU_DEV_VAL
    class: MMMUDataset
    num_shot: 4
    demonstration:
      type: trivial
      include_images: false
    # result, judge, and resource fields follow
```

Supported demonstration types are:

- `benchmark`: native benchmark rows; by default the first eligible rows are
  used, and an optional `indices` list can pin exact rows in declaration order.
  Selected rows are removed from scoring, including pointing examples whose
  labels are boxes rather than textual answers;
- `rich`: packaged curated examples, currently supported by MMMU;
- `trivial`: packaged synthetic examples, supported by MMMU, MMMU-Pro,
  DocVQA, OCRBench, and RealWorldQA;
- `reasoning`: packaged worked examples, currently supported by MMMU-Pro COT.

For reasoning demonstrations, the optional boolean
`prepend_reasoning_cue: true` adds “Let's reason step by step.” to the start of
each assistant demonstration. It does not modify the scored user query or
prefill the model's scored response. The canonical MMMU-Pro COT profile enables
the cue; set it to `false` only for an explicitly labeled no-cue prompt
ablation:

```yaml
demonstration:
  type: reasoning
  include_images: true
  prepend_reasoning_cue: true
```

`include_images: false` makes only the demonstration turns text-only; the
scored query keeps its image. Unsupported dataset/type combinations fail in
dataset preflight instead of falling back silently. `answer_format` remains a
separate per-dataset setting controlling whether the assistant demonstration
answer is represented as `raw`, `label`, `value`, or a benchmark-specific
format such as `screenspot_click`.

Use explicit indices when benchmark order would produce redundant examples or
when a comparison must preserve the exact same demonstrations:

```yaml
num_shot: 4
demonstration:
  type: benchmark
  include_images: true
  indices: [94, 9, 24, 66]
```

The indices must be unique native row IDs, their count must equal `num_shot`,
and they are valid only with `type: benchmark`. Missing or ineligible rows fail
before inference instead of falling back to other examples.

### ScreenSpot pointing

The canonical `n4_v6.yaml` suite evaluates ScreenSpot Mobile with four pinned,
distinct benchmark screenshots spanning text/icon targets and different screen
regions. Each demonstration's bounding box is converted deterministically to a
center-point answer of the form `pyautogui.click(x=..., y=...)`; the four
demonstration rows are excluded from scoring. Explicit indices prevent
consecutive rows from the same screenshot from silently defining the prompt.
The suite retains a 16K request limit as headroom for the five images in each
request; this does not change checkpoint model shape or prefill the scored
answer.

It is part of the normal push-button suite; no second run config or submission
is needed. To run only a bounded ScreenSpot smoke from the repository root:

```bash
bash third_party/VLMEvalKit/nvbench/launchers/launch_n4_v6_0916_cmh.sh \
  configs/run_configs/my_checkpoint.yaml \
  --benchmark-index 17 \
  --limit 100
```

The cache root must contain `ScreenSpot_Mobile.tsv`. ScreenSpot's native system
instruction is preserved in each request, but is flattened into the user turn
to satisfy the contained client's strict alternating user/assistant transport
contract. Normalized results report overall, text-target, and icon-target
accuracy together with format-error rate and sample counts. The underlying
answer and result contracts also support the standard Desktop and Web subsets
when they are declared in a custom suite.

`nvbench` owns row selection and conversation assembly, but delegates each
demonstration and scored-query prompt to the dataset's native VLMEvalKit
`build_prompt` implementation. Option filtering, hints, and image layout
therefore follow the pinned upstream snapshot instead of a second nvbench copy.

When a reference evaluation requires a deliberate benchmark-specific prompt
contract, the suite can name a small adapter explicitly. For example,
CV-Bench uses `prompt_adapter: cvbench_explicit_options` to render labeled
`A. value` options for both demonstrations and scored queries. The adapter is
recorded in the run manifest and does not affect other suite entries.

The server normally uses the checkpoint's sequence length. A benchmark whose
valid multi-image prompts exceed that default can declare a narrow exception:

```yaml
- dataset: MMMU_Pro_10c
  server: {context_length: 16384}
  # remaining dataset fields follow
```

Because every suite entry runs in its own server allocation, this override
applies only to that dataset and is recorded in the resolved manifest. Do not
raise the model-wide default merely to accommodate one benchmark.

`context_length` is the maximum size of one request after text tokens, expanded
image tokens, and generated output tokens are combined. With no override, the
server uses the checkpoint's `seq_length`. With an override, nvbench passes
`--inference-max-seq-length` to Megatron and sizes
`--inference-dynamic-batching-max-tokens` to at least the larger of 16K tokens
or that request limit. This guarantees that one maximum-length request fits in
a prefill step while retaining capacity for two full 8K requests. The benchmark's
`request.max_new_tokens` is a separate upper bound on generated output and must
fit inside the total request limit.

When no override is configured, nvbench explicitly requests checkpoint-derived
context with `--inference-context-from-checkpoint`. The server otherwise retains
its existing default, so unrelated inference and lm-eval-harness launches are
unchanged.

Do not copy this value into `max-position-embeddings` or `decoder-seq-length`;
they are model-shape and training-sequence settings, not request limits. When no
benchmark override is present, nvbench leaves all three settings to the
checkpoint and server defaults.

### 3. Confirm the tokenizer

The N4 model profile loads `nvidia/n4-v6-262k` at the pinned revision recorded
in `configs/models/n4_v6.yaml`. It does not contain a user-specific tokenizer
path. The tokenizer repository owns the vocabulary and special-token IDs;
nvbench selects prompt serialization independently in the suite request.

Export a Hugging Face token with access to that repository before rendering or
submitting the suite. A shell profile such as `.bashrc` may provide it:

```bash
test -n "${HF_TOKEN:-}" || { echo "HF_TOKEN is not configured" >&2; exit 1; }
```

Do not print the token, store it in run YAML, or pass it as a command-line
argument. Slurm inherits the submitting shell environment, and the server and
client use the standard Hugging Face authentication and cache behavior. If the
default cache is not shared by all allocated nodes, set `HF_HOME` to a private,
shared filesystem location before submission. Changing the tokenizer revision
is an explicit model-contract change and requires a new run name or output.

For few-shot generation, use the checked-in minimal chat template:

```yaml
suite:
  request:
    chat_template: minimal
```

Every generation suite must set `chat_template` explicitly; configuration
loading fails if it is omitted. This prevents a newly added few-shot suite
from silently inheriting an unrelated tokenizer template.

It joins system and user content with one newline and assistant content with
two newlines. It does not add role labels, `question:`, `answer:`, BOS/EOS, or
tokenizer-specific special tokens. This keeps the demonstrations themselves as
the complete task contract and prevents an unrelated tokenizer template from
silently changing prompts.

Tokenizer identity and chat serialization are separate contracts. Keep the
checkpoint-compatible tokenizer for vocabulary, image-token IDs, and special
tokens, even when the tokenizer embeds a different chat template. For the
canonical few-shot suite, do not remove `chat_template: minimal` merely because
the tokenizer defines its own template; doing so can change the prompt seen by
the model and invalidate score comparisons.

Set `chat_template: tokenizer` only when the model's tokenizer-provided chat
contract is intentionally required, such as a future zero-shot instruction
suite. A checked-in custom Jinja file can instead be selected by its safe path
relative to `third_party/VLMEvalKit/nvbench/`, for example
`chat_template: assets/my_model_chat_template.jinja`. The resolved template is
recorded in the run manifest. Conditional perplexity uses `/v1/completions`
and does not apply a chat template.

### 4. Select the containers

```bash
export NVBENCH_SERVER_IMAGE=/path/to/megatron-vlm-server.sqsh
export NVBENCH_CLIENT_IMAGE=/path/to/vlmevalkit-client.sqsh
```

These locations are intentionally cluster-neutral. Use images available on a
filesystem shared by the nodes at your site. The server image must support the
checkpoint and configured Megatron serving options. The client image must
contain the pinned VLMEvalKit snapshot's benchmark dependencies; the snapshot
itself is loaded from this checkout.

The launcher defaults to mounting `/home`, `/lustre`, and `/scratch`. Sites
with different shared roots should export a Pyxis mount list before submission:

```bash
export NVBENCH_CONTAINER_MOUNTS=/shared:/shared,/datasets:/datasets
```

`NVBENCH_CLIENT_EXTRA_PYTHONPATH` can add site-provided client packages, and
`NVBENCH_CLIENT_LD_LIBRARY_PATH` can replace the client container's default
`/opt/hpcx/ucx/lib`. Neither is needed when the selected client image already
contains all runtime dependencies.

### CMH and AGA deployment

The model and suite YAMLs are cluster-independent. Keep these site-specific
settings out of them:

- Slurm account, GPU and CPU partitions, node shape, memory, and time limits;
- server and client SquashFS paths, mounts, and dependency overlays; and
- shared workspace, Hugging Face cache, and result locations.

Put scheduler and resource-class values in `configs/clusters/<cluster>.yaml`,
and put container paths plus environment variables in a short dated helper
under `launchers/`. A run moves between CMH and AGA by selecting the matching
cluster config and helper; its checkpoint iteration, tokenizer, suite, prompt,
decoding, and scoring settings should not change.

This snapshot includes both qualified pairs:

- CMH: `configs/clusters/cmh.yaml` and
  `launchers/launch_n4_v6_0916_cmh.sh`;
- AGA: `configs/clusters/aga.yaml` and
  `launchers/launch_n4_v6_0916_aga.sh`.

The AGA helper sets `NCCL_NET=IB`, `PYTHONNOUSERSITE=1`, and
`PYTHONDONTWRITEBYTECODE=1` while still invoking the same
`nvbench.submit_suite` entrypoint and shared science configs. Do not reuse the
CMH node shape or filesystem paths on AGA. Qualify a bounded OCRBench run before
submitting a full suite after changing an AGA image, mount, or resource profile.

### 5. Install judge credentials

The canonical `n4_v6` suite includes LLM-judged benchmarks, so install an
OpenAI-compatible credential once before launching it:

```bash
${NVBENCH}/setup_judge_credentials.sh
```

The installer prompts for:

1. an OpenAI-compatible base URL, ending in `/v1/` or
   `/chat/completions`;
2. a judge model ID that the supplied key is allowed to access; and
3. a hidden API key.

It sends one authenticated chat-completion request and installs the verified
base URL and key at `~/.config/nvbench/judge_credentials.json`. The directory is
user-owned mode `0700`, and the file is user-owned mode `0600`. The key is not
written to run configs, Slurm arguments, manifests, results, or logs.

The model entered during installation is used only for the connectivity test.
Evaluation uses the exact `judge:` model ID configured for each benchmark in
`configs/suites/n4_v6.yaml`. The installed key must be authorized for that
model. If another endpoint requires a different model, update both `judge:` and
its matching model-labeled `result_suffix`; path separators and punctuation in
the suffix label are represented as underscores.

To use another credential file, set `NVBENCH_JUDGE_CREDENTIALS_FILE` to its
absolute path. Managed environments may instead inject `OPENAI_API_KEY` and
optional `OPENAI_API_BASE`. Injected credentials conflict with, rather than
silently override, different installed credentials.

### 6. Dry-run the complete suite

Render the exact Slurm plan without submitting any jobs:

```bash
PYTHONPATH=third_party/VLMEvalKit python3 -m nvbench.submit_suite \
  --run-config ${NVBENCH}/configs/run_configs/my_checkpoint.yaml \
  --workspace /shared/operator_workspace \
  --dry-run
```

For either checked-in cluster profile, one canonical N4 V6 launch currently
plans:

- `n1`, `n4`, and `n8` GPU arrays covering all 18 benchmarks; and
- dependent CPU scoring arrays for WorldVQA and MathVista.

Review the checkpoint, iteration, output root, datasets, resources, and array
concurrency in the rendered JSON before submitting.

### 7. Submit

Run the same command without `--dry-run`:

```bash
PYTHONPATH=third_party/VLMEvalKit python3 -m nvbench.submit_suite \
  --run-config ${NVBENCH}/configs/run_configs/my_checkpoint.yaml \
  --workspace /shared/operator_workspace
```

The command prints the returned Slurm job IDs and resolved submission plan.
Exact-match benchmarks finish in their GPU jobs. LLM-judged benchmarks generate
validated predictions on GPUs, release the GPU allocations, and then score the
predictions in dependent CPU-only jobs. The dependency uses `afterany`, while
artifact validation prevents scoring incomplete generation output.

The dated login-node helpers pin each cluster's validated server image, client
image, package overlay, workspace, mounts, and environment. From the `nvbench`
directory, run the helper matching `cluster_config`:

```bash
cd ${NVBENCH}

# CMH with cluster_config: ../clusters/cmh.yaml
bash launchers/launch_n4_v6_0916_cmh.sh \
  configs/run_configs/my_checkpoint.yaml

# AGA with cluster_config: ../clusters/aga.yaml
bash launchers/launch_n4_v6_0916_aga.sh \
  configs/run_configs/my_checkpoint.yaml
```

Each helper renders the plan on its login node and directly submits the required
GPU arrays and dependent CPU judge jobs. Pass `--benchmark-index`, `--limit`, or
`--dry-run` after the run config when needed. Set `NVBENCH_WORKSPACE` only when
the operator workspace differs from that helper's default. A new cluster should
provide the same profile/helper pair rather than editing the suite or model
configuration.

## Results

Each generated-answer suite element writes to:

```text
<output.root>/<run.name>__array-<index>-of-<count>/
```

A completed element contains:

- `megatron_dynamic_server_vlm_eval_manifest.json`, the resolved run manifest;
- `demonstration_and_scored_row_partitions.json`, the exact demo/scored split;
- `vlmeval_predictions_ready.v1.json`, the validated prediction handoff;
- `vlmeval_dataset_status.json`, the terminal dataset status;
- `vlmeval_metrics.v1.json`, normalized metrics; and
- the native VLMEvalKit predictions and score artifacts.

Exact-match benchmarks consume the prediction handoff and finalize in the GPU
job. For an LLM-judged benchmark, the dependent CPU stage consumes the same
handoff and writes the final score, metrics, status, and completed manifest.

Treat a result as complete only when prediction coverage is complete, metrics
are finite, the requested checkpoint was loaded, and server shutdown is clean.
Slurm wrapper logs use the scheduler default unless the rendered command is
given an explicit `--output` path.

## Run a bounded debug evaluation

Select one zero-based suite entry and limit the number of scored rows without
editing the suite:

```bash
PYTHONPATH=third_party/VLMEvalKit python3 -m nvbench.submit_suite \
  --run-config ${NVBENCH}/configs/run_configs/my_checkpoint.yaml \
  --workspace /shared/operator_workspace \
  --benchmark-index 0 \
  --limit 10
```

For `benchmark` demonstrations, the limit is applied after demonstration rows
are removed. Packaged `rich`, `trivial`, and `reasoning` demonstrations do not
consume target rows. The limit is included in the
resolved run fingerprint. Give debug runs a distinct run name or output root
instead of reusing a full-run destination.

## Run conditional perplexity

Use the same launcher with the conditional-perplexity suite:

```yaml
schema_version: 1
run:
  name: my_checkpoint_surge_mm_ppl
model_config: ../models/n4_v6.yaml
suite_config: ../suites/surge_mm_ppl.yaml
cluster_config: ../clusters/cmh.yaml
checkpoint:
  path: ${workspace}/training_run/checkpoints
  iteration: 8398
paths:
  cache_root: ${workspace}/vlmevalkit_cache
output:
  root: ${workspace}/eval_runs/my_checkpoint
  resume: false
```

Perplexity does not use demonstrations. The run config selects the model,
cluster, checkpoint, and output paths. The suite config owns the dataset path
and scoring instruction:

```yaml
schema_version: 1
suite:
  name: surge_mm_ppl
  kind: conditional_perplexity
  request:
    instruction: "Describe the image in detail.\n"
    temperature: 0
    top_p: 1
    retry: 10
  benchmarks:
    - dataset: surge_mm_ppl
      format: packed_jsonl_tar
      data: /path/to/surge_mm_ppl/data.jsonl
      category: image_type
      resource: n1
```

Update `data` when the dataset is staged elsewhere. The supported
`packed_jsonl_tar` format requires `data.jsonl` and a sibling `data.tar`. Each
JSONL record supplies a unique `uid`, the reference text in `data.answer[0]`,
and an image category in `tags[0]` such as `image_type_chart`. The outer tar
member named by `uid` contains a nested tar whose same-named member is the
image. The checked-in Surge-MM adapter validates this structure before
scoring.

At the HTTP boundary the client submits the complete tokenized prompt and
image to `/v1/completions` with `echo=true`, prompt log probabilities enabled,
and zero generated tokens. This is a conditional-perplexity workflow, not a
generation run. The selected server image must expose that prompt-logprob
contract.

Start with `--benchmark-index 0 --limit 2`. A complete run writes an
append-only sample ledger, an empty failures ledger, terminal status, and
`surge_mm_ppl.metrics.json`. The metrics pool log likelihood, token, UTF-8 byte,
and whitespace-word counts into perplexity per word/byte, bits per byte, and
nats per token, both overall and by image type.

## Configuration reference

```text
configs/
├── run_configs/  # checkpoint, output, cache, and component choices
├── models/       # pinned Hugging Face tokenizer and serving overrides
├── suites/       # request defaults plus per-benchmark scoring and resources
└── clusters/     # Slurm identity and reusable resource classes
```

The run config is the only file normally copied for a new checkpoint. Users on
another cluster should copy a cluster config and update its Slurm identity,
resource classes, GPUs per node, memory, and CPU judge profile.

Model configs contain only the pinned tokenizer identity and settings not owned
by the checkpoint or tokenizer.
Suite configs describe shared request behavior and per-benchmark scoring.
Cluster configs own allocation sizes and step resources.

Each benchmark selects its dataset class, shot count, demonstration type and
image policy, demonstration answer format, result reader, result suffix and
unit, judge, and resource class. One suite can therefore mix shot counts,
demonstration contracts, exact matching, LLM judging, and node counts.

The suite-level `request` block supplies generation defaults such as chat
template policy, temperature, maximum output length, stop strings, and image
request details.
A benchmark may include a partial `request` block to override only the fields
it needs; for example, a reasoning benchmark can request a longer response
while inheriting deterministic decoding and image settings. Likelihood-based
multiple-choice scoring is a different evaluation workflow and is not
represented as a generation option here. Conditional perplexity likewise uses
its explicit suite kind shown above.

### Resources and topology

A cluster resource defines the allocation and its server/client step sizes. A
one-node resource uses local torchrun; a larger resource uses one rendezvoused
torchrun group across all allocated GPUs. Server topology is derived from the
selected resource and cannot be restated in run YAML.

External judge scoring uses the cluster-level `judge_scoring` CPU profile and
does not depend on model topology.

### Model settings and determinism

Checkpoint arguments are authoritative for model architecture. Add a server
override only when evaluation must intentionally differ from the checkpoint or
the serving interface requires it. The N4 model config deliberately selects
Mistral prompt formatting and preprocessing, the inference-optimized language
implementation, FP32 MoE routing, and NCCL MoE dispatch.

For kernel-level repeatability, add this optional model override:

```yaml
model:
  server:
    overrides:
      deterministic-mode: true
```

This is separate from `temperature: 0`, which requests greedy decoding but does
not make every kernel deterministic.

### Resume behavior

Set `output.resume: true` only when resuming the same resolved checkpoint,
tokenizer, suite, and server settings. Any changed input requires a new run name
or output root.

## Directory layout and provenance

```text
nvbench/
├── README.md
├── VENDOR.md
├── launchers/                    # dated, cluster-qualified submission examples
├── submit_suite.py
├── setup_judge_credentials.{sh,py}
├── run_{vlm_eval,judge_eval}.sbatch
├── run_{server,client,judge}.py
├── preflight_datasets.py
├── configs/
├── orchestration/
│   ├── dynamic_server_http_client/  # HTTP transport and server API clients
│   ├── fewshot_dataset_adaptation/  # Selection plus packaged predefined examples
│   └── perplexity_datasets/         # Conditional-PPL dataset adapters
└── tests/
```

All NVIDIA integration code stays inside `nvbench/`. The surrounding
VLMEvalKit snapshot omits upstream `.github/` CI workflows and helpers; all retained
upstream files are unchanged. Keep `.github/` excluded during vendor refreshes to avoid
the parent repository's automation-team review request. The vendored snapshot is commit
`bdb6d429f3f6804eaa6cd899c341486c8a42aed8`; see `VENDOR.md` for its
reproducible import record.

## ScreenSpot four-shot answer format

The `n4_v6` ScreenSpot Mobile entry uses `screenspot_click_normalized`: compute
the standard `[x,y,width,height]` bbox center as `(x+(width-1)/2, y+(height-1)/2)`,
divide by **that demo's original image width/height**, and format six decimal
places in `pyautogui.click(x=..., y=...)`. Image dimensions come from the native
demo prompt before image filtering or transport/server resizing. A missing or
ambiguous image, invalid dimensions, or out-of-image center fails explicitly.
The formatter supports arbitrary standard ScreenSpot demos; it contains no
lookup of the four configured IDs. The existing `screenspot_click` pixel format
remains selectable. Pro corner bboxes are excluded by the configuration gate.

The dedicated `assets/screenspot_shared_gui_end_marker_template.jinja` renders
the shared GUI instruction and task prefix once, followed by four screenshot /
question / normalized-click demos and the final screenshot / question. Each demo
answer ends with ten underscores on its own line; the matching stop string
terminates generation before another turn. The canonical entry retains demo
IDs `[94,9,24,66]`, excludes them from scoring, and keeps temperature 0,
128 generated tokens and 16K context. This marker template requires few-shot
demos: marker-only empty replies were observed in zero-shot experiments.

**Why this format:** a bounded audit of actual midtraining targets found all 134
sampled coordinate pairs in `[0,1]` across 60 sampled documents from three GUI
sources. GroundCUA used `pyautogui` actions; the other sources also used JSON or
XML tool parameters. This supports matching the learned coordinate convention;
it does not establish that every training example has the same syntax or units.
On the tested VLM-midtraining checkpoint, changing only four demo answer lines
from pixels to normalized coordinates improved a 20-target smoke from 1/20 to
16/20 and reduced full demo-coordinate copies from 19/20 to 0/20. The subsequent
full normalized four-shot Mobile run scored 454/498 (91.16%), compared with
220/498 (44.18%) zero-shot on the identical non-demo targets. These observations
are consistent with format alignment; they do not isolate a causal training
source or guarantee the same gains for another model.

The native dataset, bbox/scorer semantics, target partition and image
preprocessing are unchanged. Tests cover arbitrary original image dimensions,
invalid/ambiguous images, demo selection, canonical request settings, and exact
rendered parity with the successful four-shot prompt fixture.
