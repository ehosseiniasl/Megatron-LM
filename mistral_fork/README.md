# Pretokenized Dataloader (Mistral Fork)

Self-contained fork of the Mistral pretokenized data loading pipeline for Megatron-LM.

**Upstream Mistral commit**: `1f65b36b71e679f89f4557b34ae837979ab9e1d7` (2026-04-13, `main`)

## Overview

The pretokenized dataloader reads pre-tokenized binary data and builds training batches with good shuffling properties. The data pipeline has three stages:

1. **Pretokenization** (offline): tokenize raw text, extract image metadata (tar path, tar offset, image size), apply document-level filtering. Produces `.bin` + `.idx` + `.metadata` files. This fork has a config-driven implementation for the supported JSONL source shapes.
2. **Epoch index building** (offline or at training start): prepacking — build a mapping `(epoch_id, sequence_id) → [DocumentChunk, ...]` that defines how documents are concatenated and shuffled into fixed-length sequences. **Important**: it also determines the size in tokens that each image will have **after the image transformation**. Because image transforms are random (random crop, resize, etc.), the epoch index builder must sample the transform at build time using a deterministic seed to predict the exact token count for each image.
3. **Sampling** (at training time): given a `sequence_id`, load the corresponding document chunks and images via random access and assemble the batch.

This fork contains the **sampling** code (stage 3), the **epoch index building** code (stage 2, in `datasource.py`), and a small fork-local JSONL pretokenization/merge path for the supported NVIDIA source shapes.

For NVIDIA-side multimodal bring-up, see `mistral_fork/docs/NVIDIA_MULTIMODAL.md`.

## Fork-Local Pretokenization

This fork also has a small config-driven pretokenization/merge script for the
NVIDIA datamix source shapes currently supported here. It does not call or
import the sibling Mistral repo at runtime.

The pretokenization script takes a JSONL pretokenization YAML whose
`sources[*].jsonl_dir` points at raw JSONL data. The output source folder is
derived automatically from the raw path, tokenizer name, filter profile, and
masking strategy under `--pretokenized-data-dir`. A source may independently
select a versioned rare-token filter, PDS filter, or bad-document filter.
`rare_token_filter` and `pds_filter` are required source fields: set them
explicitly to an enum value or to `null`. The configured values are validated
against the tokenizer, filter profile, and sample type without being changed.
The same policy checks run again when constructing cache paths.
`bad_document_filter` remains optional.

```yaml
sources:
  - jsonl_dir: /path/to/source
    # ... weight, percentage_data, sample_type, and filter ...
    rare_token_filter: rare_tokens_DEBUG
    pds_filter: nvidia_pds_quarantine_v2
    bad_document_filter: nvidia_bad_documents_DEBUG
```

The bad-document filter matches the exact JSONL path and the byte offset of the
first byte in each JSONL record. The rare-token filter runs after tokenization.

### Rare token filtering

Rare token filtering must be configured **before pretokenization**. It removes
documents while they are being tokenized; it is not a postprocessing step that
can be applied to an existing pretokenized dataset.

`n4-v6-262k.json` and `ml5_tokenizer_hf` are the same tokenizer in tiktoken
JSON and Hugging Face formats, respectively. Both accept exactly the
`multimodal_v4__lobstral` and `multimodal_v1__grounding` filter profiles and
require the explicit value `rare_tokens_n4_v6_262k_v1`. The shared rare-token
enum loads token IDs from
`/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/tokenizers/n4-v6-262k/filtered_tokens.json`.
Other filter profiles and unregistered tokenizer paths are rejected instead of
falling back to a default. The versioned artifact must not be changed in place.

### PDS filtering

PDS filtering removes a multimodal document when one of its images occurs in a
registered quarantine manifest. The `nvidia_pds_quarantine_v2` enum maps to all
four NVIDIA PDS manifests used by this pipeline. The immutable v1 value remains
registered for existing cache paths. Sources using `n4-v6-262k.json` or
`ml5_tokenizer_hf` must explicitly set it when `sample_type.is_multimodal()` is
true and must set `pds_filter: null` for text-only sample types. Other
tokenizers can opt in on multimodal sources explicitly.

The PDS enum is stored in datasource metadata and included in the cache path.
An image without complete `tar_path`, `offset`, and `size` annotations cannot
be matched against the manifests; pretokenization logs a warning and continues
without filtering that image.

1. Add a new `RareTokenFilter` enum value such as
   `rare_tokens_v{my_version}` in `document_filters.py`, together with the token
   IDs that version should remove. Do not mutate an existing version after it
   may have been used: its name is part of the cache key and must retain the
   same meaning. Only edit an existing entry when you are certain it has never
   been used.
2. Select the new version in the source config:

   ```yaml
   rare_token_filter: rare_tokens_v{my_version}
   ```

3. Launch pretokenization normally. The filter version is included in the
   output path, for example:

   ```text
   /path/to/pretokenized_data/pretrain_v3_lobstenizer/pretrain_v4__lobstral_rare_tokens_v{my_version}/wikibooks/chunks
   ```

Pretokenize raw JSONL sources into a pretokenized data root:

```bash
uv run python -m mistral_fork.mistral.data.pretokenized.processing.pretokenize \
  --config /path/to/jsonl_pretokenization_config.yaml \
  --pretokenized-data-dir /path/to/pretokenized_data \
  --validation-fraction 0.001 \
  --guarantee-validation-per-chunk
```

Merge generated chunks after pretokenization:

```bash
uv run python -m mistral_fork.mistral.data.pretokenized.processing.merge \
  --config /path/to/jsonl_pretokenization_config.yaml \
  --pretokenized-data-dir /path/to/pretokenized_data
```

Build a Megatron-side `PretokenizedConfig` from the JSONL config. Use absolute
paths here: epoch-index cache paths are keyed by absolute pretokenized source
paths, and the conversion script rejects relative roots.

```bash
uv run python -m mistral_fork.mistral.data.pretokenized.processing.config \
  --config /path/to/jsonl_pretokenization_config.yaml \
  --output /path/to/megatron_pretokenized_config.yaml \
  --pretokenized-data-dir /path/to/pretokenized_data \
  --epoch-index-cache-path /path/to/epoch_index
```

The output source directory has this shape:

```text
{pretokenized_data_dir}/{tokenizer_name}/{filter_profile}[_{masking_strategy}][_{rare_token_filter}][_{pds_filter}][_{bad_document_filter}]/{relative_jsonl_path}
```

The config `tokenizer_path` must identify a registered tiktoken JSON file or
Hugging Face tokenizer directory; its registered name forms the first cache-key
component.

## How it works

### Data format on disk

```
{pretokenized_data_dir}/{cache_key}/{relative_jsonl_path}/
  merged_{size}b/
    chunks_XXXXX-YYYYY.bin       # concatenated tokens (int32)
    chunks_XXXXX-YYYYY.idx       # (offset, length) per document
    chunks_XXXXX-YYYYY.metadata  # image metadata: ImageIndexEntry per image (optional)
    chunks_XXXXX-YYYYY.mask      # loss mask per document (optional)

{epoch_index_cache_path}/{data_path}/
  chunks_XXXXX-YYYYY__{epoch_hash}/
    epoch_index.jsonl             # packed sequences: list[DocumentChunk]
    json_index.npy                # byte offsets into epoch_index.jsonl
    epoch_index_infos.json        # build specs (seq_len, seed, etc.)
    epoch_stats.json              # token/sequence counts

# Images are stored separately in TAR archives (referenced by tar_path + TarOffset
# in each ImageIndexEntry). They are loaded via random access at sampling time.
```

The **cache key** is derived from the tokenizer, filter profile, optional masking
strategy, optional rare-token filter, optional PDS filter, and optional
bad-document filter (for example,
`pretrain_v3_lobstenizer/pretrain_v4__lobstral_rare_tokens_DEBUG_nvidia_bad_documents_DEBUG`),
ensuring reproducible paths.

### Epoch indexes and blending

Epoch indexes are prepacking indexes. They are built for each source individually: each source gets its own packed sequence list under `epoch_index_cache_path`. The number of epochs to build for a source is determined by the mixture and the training horizon, so high-weight sources and small sources may need different numbers of source-local epochs.

The mixture is not materialized as one combined epoch index. Blending happens at runtime: the dataloader samples between sources using the mixture weights, then consumes the next sequence from the selected source's own epoch index.

### Sampling

All ranks share the same global state (`global_sequence_id`). For each step:

1. Sample `datasource_id` for each sequence in the global batch (given source weights)
2. Increment per-datasource `sequence_id`
3. Each rank selects its local sequences based on `(rank, world_size)`
4. For each sequence, `Datasource.get_sequence()` assembles the data:
   - **Epoch index read** (2 random accesses): first look up the byte offset in `json_index.npy`, then read the JSONL entry at that offset for this `sequence_id` → returns a list of `DocumentChunk` descriptors
   - **Text read** (1 random access per document): read tokens from the `.bin` file at the offset/size specified by each chunk's `TextIndexEntry`
   - **Mask read** (1 random access per document, if masks exist): read the loss mask from the `.mask` file
   - **Image loading** (1 random access per image, for multimodal sources): each `ImageIndexEntry` stores a `tar_path` + `TarOffset` pointing to the image inside a TAR archive (or S3). The image is loaded via random access into the TAR, decoded with PIL, then the image encoder applies transforms and produces image tokens. The transform was already "resolved" during epoch index building (seed + metadata → predicted token count), so the image token count is deterministic.

This design makes topology changes (resharding) trivial — only the `(rank, world_size)` mapping changes, the global state stays the same.

### Interdocument masking (THD format)

When `interdocument_masking=True` (default), the iterator flattens the micro-batch into `[1, B*S]` and provides `cu_seqlens` (cumulative segment lengths) for block-diagonal causal attention. This prevents cross-document attention leakage.

Output tensors:
- `tokens`, `labels`, `loss_mask`, `position_ids`: `[1, B*S]`
- `cu_seqlens`: `[1, num_segments+1]` (int32)
- `max_seqlen`: `[1]` (int32)

Without interdocument masking, output shape is `[micro_batch_size, seq_len]`.

### Image transforms and deterministic token counts

For multimodal sources, each image goes through a random transform (crop, resize, etc.) that determines how many tokens the image will occupy. Since sequences must have a fixed total token count, the epoch index builder must predict the exact number of image tokens at build time. It does this by sampling the transform using a deterministic seed derived from the image metadata, so that the same transform is reproduced at training time when the image is actually loaded. This coupling between epoch index building and image loading is why the epoch index builder is written in Python rather than a faster language.

## Usage in Megatron

### Launching from a Mistral sweep

The usual path is to launch Megatron from Mistral with
`tools/megatron/launch_megatron_sweep.py`. That launcher reads the Mistral sweep
YAML, builds the normal `TrainArgs`, then converts `train_args.data.pretokenized`
into a Megatron-side `PretokenizedConfig`.

Relevant Mistral-side code:

- `tools/megatron/launch_megatron_sweep.py::build_pretokenized_config(...)`
  converts `DataArgs.pretokenized` into `mistral_adapter.PretokenizedConfig`
- `tools/megatron/launch_megatron_sweep.py::_write_pretokenized_yaml(...)`
  writes it to `{run_dir}/pretokenized_config.yaml`
- `tools/megatron/launch_megatron_sweep.py::generate_sbatch(...)` passes it to
  Megatron as `--pretokenized-config {run_dir}/pretokenized_config.yaml`

Example launcher command:

```bash
python -m tools.megatron.launch_megatron_sweep \
  --sweep_path tools/megatron/sweeps/my_pretokenized_sweep.yaml \
  --megatron_dir /mnt/vast/home/thiblav/code/Megatron-LM-fork_claude_3 \
  --mistral_dir /mnt/vast/home/thiblav/code/mistral_claude_3 \
  --qos dev \
  --submit true
```

The sweep must set `data.pretokenized`. The launcher resolves source paths,
source weights, `percentage_data`, tokenizer IDs, `epoch_index_cache_path`, and
the original Mistral batch shape into the generated YAML.

If the generated YAML sets `skip_images: true`, multimodal sources are treated as
text-only during epoch-index building and sampling. Image metadata is ignored,
image encoders are not required, and sampled batches contain no image tensors.

### PretokenizedConfig schema

The generated YAML is validated by
`mistral_adapter/pretokenized_config.py::PretokenizedConfig`. It is the runtime
contract between Mistral's launcher and Megatron's external dataloader.

Top-level fields include:

- `version`: schema version checked by Megatron at startup.
- `sources`: list of per-source entries, described below.
- `bos_id`, `eos_id`, `pad_id`: text tokenizer special-token IDs.
- `img_id`, `img_break_id`, `img_end_id`: image special-token IDs, required
  when images are enabled.
- `seed`: sampling and epoch-index seed.
- `epoch_index_cache_path`: root directory for prebuilt or runtime-built epoch
  indexes.
- `interdocument_masking`: whether to emit THD batches with `cu_seqlens`.
- `original_seq_len`: sequence length used to build the pretokenized epoch
  indexes. With interdocument masking, Megatron launches with
  `--seq-length original_seq_len`; `--micro-batch-size` is a launch-time
  setting.
- `tokenizer_name`: tokenizer/cache-key identity.
- `optimized_datasource_id_sampling`: enables the faster datasource sampling
  path.
- `skip_images`: if true, multimodal sources are treated as text-only; image
  metadata is ignored and image encoders are not required.
- `drop_bos`: required boolean. If true, document BOS inputs are replaced by
  EOS when samples are loaded. Epoch indexes, document sizes, and document IDs
  are unchanged.
  The adapter currently requires this exactly when `interdocument_masking` is
  false; use Megatron's `--eod-mask-loss` to mask EOS-to-next-token loss. This
  runtime transformation is not equivalent to removing BOS while building
  epoch indexes; see [the design note](docs/RUNTIME_BOS_TO_EOS_DESIGN.md).

Each `sources` entry includes:

- `prefix_path`: path to the merged pretokenized source chunk prefix.
- `weight`: sampling weight for the source before normalization.
- `percentage_data`: fraction of that source to use when building epoch indexes;
  see the `percentage_data` section below.
- `sample_type`: source modality/type, for example `pretrain` or
  `pretrain_image`.
- `image_transform_type`: per-source image transform, such as
  `crawl_default_no_high_res`; set to `null` for text-only sources.

### percentage_data

`percentage_data` is set per source and controls the fraction of that source used when building epoch indexes / prepacking indexes. For faster experiments, it is OK to lower `percentage_data` or use `--percentage-data-multiplier`; this builds less data into the prepacking indexes, but training will revisit the smaller subset for more epochs.

⚠️ If editing a YAML config by hand to reduce the data fraction, multiply every existing per-source `percentage_data` by the reduction factor. Do not replace all values with the same number, because the existing values already encode source-specific fractions. We will introduce a parameter to make this easier.

TODO:

- Make the config more path agnostic so the same config can be used for both
  Mistral and NVIDIA environments.
- Move `num_microbatches` out of the generated config contract.

### Vocabulary size

With `--tokenizer-type NullTokenizer`, pass `--vocab-size` matching the
tokenizer that produced the pretokenized data.
`launch_megatron_sweep.py` resolves this via `tokenizer.n_words`.

| Tokenizer | File | `--vocab-size` |
|-----------|------|---------------|
| tekken v2 (`pretrain_v2`) | `pretrain_v2.tekken.json` | 131072 |
| lobstenizer (`pretrain_v3`) | `lobstenizer.tekken.json` | 204800 |

### Manual Megatron launch contract

If launching `pretrain_gpt.py` directly, pass the generated YAML with
`--pretokenized-config` and use Megatron's external dataloader path:

```bash
python -u pretrain_gpt.py \
  --dataloader-type external \
  --pretokenized-config /path/to/run_dir/pretokenized_config.yaml \
  --dataloader-inter-document-masking \
  --tokenizer-type NullTokenizer \
  --vocab-size <TOKENIZER_VOCAB_SIZE> \
  --micro-batch-size <MICRO_BATCH_SIZE> \
  --seq-length <ORIGINAL_SEQ_LEN> \
  --global-batch-size <NUM_MICROBATCHES * DATA_PARALLEL_SIZE> \
  --num-workers 4 \
  ...
```

Replace the angle-bracket expressions with integers before running. For example,
if the Mistral sweep uses `batch_size=2`, `seq_len=4096`,
`num_microbatches=8`, and `data_parallel_size=16`, Megatron must see:

```bash
--micro-batch-size 2
--seq-length 4096
--global-batch-size 128
```

In THD format, the adapter emits per-sample `cu_seqlens` and Megatron's
`flatten_batch_for_packed_sequences()` merges the microbatch to the flat
`[1, B*S]` format expected by attention. The generated YAML keeps the original
sequence length:

```yaml
interdocument_masking: true
drop_bos: false
original_seq_len: 4096
```

`pretrain_gpt.py` validates this contract at startup:

- `--tensor-model-parallel-size` must be `1` until TP support lands.
- `--seq-length` must be `original_seq_len`.
- `seq_aux_loss` with a positive coeff currently requires
  `--micro-batch-size 1`.
- `--global-batch-size` should be
  `micro_batch_size * num_microbatches * data_parallel_size`.

With those settings, the token count per optimizer step is unchanged:

```text
Megatron tokens/step
  = global_batch_size * seq_length
  = (original_B * num_microbatches * DP) * original_S
  = original_B * original_S * num_microbatches * DP
```

This is the same token count as the original Mistral training setup.

### Programmatic iterator

The low-level entry point is `MistralPretokenizedIterator` in
`mistral_adapter/pretokenized_iterator.py`.

```python
from pathlib import Path

from mistral_adapter.pretokenized_config_utils import load_pretokenized_config
from mistral_adapter.pretokenized_iterator import MistralPretokenizedIterator

config = load_pretokenized_config(Path("/path/to/pretokenized_config.yaml"))

iterator = MistralPretokenizedIterator(
    config=config,
    num_total_tokens=10_000_000,  # training horizon
    rank=0,
    world_size=8,
    micro_batch_size=2,
    num_workers=4,
)

for batch in iterator:
    # batch["tokens"], batch["labels"], batch["loss_mask"], ...
    pass
```

### State save/restore

The iterator supports checkpointing via `save_state()` / `set_state()`. On resume, the dataloader restores its exact position (which documents have been served) so no data is replayed.

```python
state = iterator.save_state()   # dict with global_sequence_id, per-source state
# ... save to disk ...
iterator.set_state(state)       # restore on resume
```

## Directory structure

```
mistral_fork/
├── configs/                                  # Example JSONL pretokenization configs
├── mistral/
│   ├── data/
│   │   ├── pretokenized/                    # Core data loading
│   │   │   ├── pretokenized_data_loader.py  # PretokenizedDataLoader — batch assembly
│   │   │   ├── datasource.py                # Datasource — per-source reading + epoch index building
│   │   │   ├── distributed_prebuild_of_epoch_indexes.py  # Worker-sharded epoch-index prebuild helpers
│   │   │   ├── pretokenized_utils.py        # SourceItem, weight scheduling, sampling
│   │   │   ├── pretokenized_args.py         # PretokenizedArgs config
│   │   │   ├── megatron_dataloader_config.py # Megatron dataloader config adapter
│   │   │   ├── megatron_index_reader.py     # Binary .bin/.idx reader and writer
│   │   │   ├── metadata.py                  # Document/image/text index entries
│   │   │   ├── indexed_jsonl.py             # Random-access JSONL (epoch index)
│   │   │   ├── indexed_mask.py              # Loss mask reader/writer
│   │   │   ├── masking.py                   # DocumentMask dataclass
│   │   │   ├── cache_key.py                 # TokenizerName enum, cache key derivation
│   │   │   ├── filter_profiles.py           # FilterArgsName enum
│   │   │   ├── image_transform_metadata.py  # TransformImageMetadata
│   │   │   ├── simple_image_encoder.py      # Lightweight image encoder used by epoch-index prebuild
│   │   │   ├── multiprocessed_pretokenized.py  # Multi-worker wrapper
│   │   │   ├── components/                  # Fork-local pretok components
│   │   │   │   ├── filters.py               # Supported JSONL filters and image filtering
│   │   │   │   ├── masking.py               # Supported loss-mask strategies
│   │   │   │   └── image_transforms.py      # Image transform registry/factory
│   │   │   └── processing/                  # Fork-local offline commands
│   │   │       ├── config.py                # JSONL config schema and Megatron config conversion CLI
│   │   │       ├── datasource_metadata.py   # PretokenizedDatasourceMetadataV2
│   │   │       ├── pretokenize.py           # Raw JSONL -> chunked pretokenized data CLI
│   │   │       ├── merge.py                 # Chunk merge CLI
│   │   │       ├── paths_helpers.py         # Path construction helpers
│   │   │       └── prebuild_epoch_indexes_megatron.py # PretokenizedConfig -> epoch-index CLI
│   │   ├── multiprocessed.py                # MultiprocessedDataLoader (subprocess wrapper)
│   │   ├── data_loader.py                   # Base DataLoader + Batch
│   │   ├── args.py                          # TwoPhaseArgs
│   │   ├── source.py                        # SampleType enum
│   │   ├── source_args.py                   # Source argument parsing
│   │   ├── utils.py                         # Seed utilities
│   │   ├── iterator/utils.py                # SequenceOfSamples
│   │   └── interleaved_pretokenized/
│   │       └── selection_args.py            # SelectionStrategy ABC
│   ├── bs_rampup.py                         # Batch size scheduling
│   ├── distributed.py                       # Distributed utilities
│   ├── monitoring/profiling.py              # Performance logging
│   └── utils.py                             # General utilities
├── mistral_common/                          # Minimal subset
│   ├── base.py                              # MistralBase (Pydantic model)
│   └── tokens/tokenizers/
│       ├── base.py                          # Span
│       └── image.py                         # ImageEncoding, image_size_to_num_tokens
└── shared/                                  # Minimal stubs
    ├── args.py                              # HasArgs Protocol
    ├── common_utils.py                      # TarOffset / TAR constants
    ├── image_utils.py                       # Image loading (PIL)
    ├── s3_utils.py                          # S3 file reading
    ├── safer_dict.py                        # Safe dict access
    └── tartar.py                            # TAR file handling
```


## Versioning

The file `mistral_fork/__init__.py` defines a `__version__` string and a
`MAIN_VERSION` integer. The Mistral launcher
(`tools/megatron/launch_megatron_sweep.py`) checks both values before launching a
job. If there is a mismatch, the launcher refuses to start.

**When to bump the version:** whenever you change code in `mistral_fork/` or
`mistral_adapter/` in a way that requires a matching change in the Mistral launcher
or sweep configs (e.g. adding a new parameter, changing the iterator API, updating
upstream Mistral code). Both sides must be updated together:

1. Bump `__version__` and `MAIN_VERSION` in `mistral_fork/__init__.py` (this repo)
2. Bump `EXPECTED_MEGATRON_FORK_VERSION` and `EXPECTED_MEGATRON_MAIN_VERSION` in `tools/megatron/launch_megatron_sweep.py` (Mistral repo)

## Known limitations

### Epoch index building is slow

The epoch index building code (`_build_epoch_index` in `datasource.py`) is written in Python and is slow for large datasets. It runs at training start when indexes are missing (with `allow_epoch_index_building=True`), or can be prebuilt offline using the Mistral repo's distributed `prebuild_epoch_indexes.py` script. A standalone prebuilding script for this repo will come in a later PR. We would like to rewrite the building code in Rust but the image transform resolution logic (predicting the number of tokens per image based on sampled transformations) is tightly coupled with the Python image processing code.

### Epoch indexes consume significant disk space

The epoch index uses a JSONL-based format (`epoch_index.jsonl` + `json_index.npy`) rather than a compact binary packing. This was chosen for flexibility during initial development (easy to inspect, debug, and iterate on the format). Now that the system is more stable, we want to migrate to a more compact binary format.

### Epoch index building requires a lot of memory

The current shuffling strategy (full epoch index precomputation) requires materializing the entire document-to-sequence mapping upfront. We plan to switch to an alternative shuffling strategy that avoids this full materialization.

### Multiprocessed dataloader worker synchronization is fragile

When using `num_workers > 1` in `MultiprocessedPretokenizedDataLoader`, each worker must skip the other workers' batches to stay in sync. This is done by calling `get_batch(skip=True)` in a loop at startup and after state restore, advancing the internal state without loading data. This approach is fragile: it is easy to break when adding schedules (batch size rampup, weight interpolation), because the skip logic must exactly mirror the batch production logic at every step. A mismatch causes silent state desynchronization between workers, leading to duplicate or missing data.

### Random access I/O at scale

The dataloader relies on random access reads for both text (`.bin` files) and images (TAR archives). At large scale this becomes the bottleneck:

- **Text**: ~4,000 random reads/s per GPU for a DS3-class model for 1k gb200
- **Images**: ~1,600 random reads/s per GPU for a DS3-class model for 1k gb200
- These numbers scale roughly linearly with model compute — a 10x smaller model needs ~10x more QPS from storage

A potential mitigation is to **pre-pack** text and images in the exact order they will be read (i.e. materialize the full packing before the run, turning random access into sequential reads). We estimated this would require ~40 TB of storage per 1T tokens of training data, so roughly ~3 PB for a full-scale training run.
