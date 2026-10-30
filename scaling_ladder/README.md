# Scaling-ladder recipes

SLURM sbatch recipes for ladders of hybrid Mamba/GDP + attention MoE models,
spanning 315M–3B active params (1B–30B total) and 88B–1T training tokens, used
to fit scaling laws. Layout:

```
scaling_ladder/
├── Dockerfile        # build the training image (see "Building the image")
├── README.md
└── scripts/          # one sbatch recipe per model size and recipe family
```

Recipe families:

- `scripts/adamw/`: AdamW baseline.
- `scripts/muon_lion/`: Muon with Lion for scalar parameters.
- `scripts/muon_lion_latent_moe_gdp/`: Muon/Lion latent-MoE variants using
  Gated Delta Product (GDP) sequence-mixer layers.

All recipe variants share the same scaffolding:

- bf16, WSD LR schedule, TikTokenizer
  (`multiMixV8.gpt4o_nc_sd.500000.128k`), distributed optimizer with
  `--overlap-grad-reduce` / `--overlap-param-gather`, `--ddp-num-buckets 8`,
  `--attention-backend flash`, `pretrain_hybrid.py` entrypoint.
- Each script runs inside a container built from `Dockerfile`
  (see [Building the image](#building-the-image)).

Before submitting, set `ROOT_DIR` at the top of each script to a path you have
write permission on; it must contain `code/` (your Megatron-LM checkout),
`tokenizers/`, and `blend_files/`. The SBATCH header assumes 8 GPUs/node; on
GB200/GB300 (4 GPUs/node) set `--ntasks-per-node=4`, `--gpus-per-node=4`, and
add `--segment=4` (or `--segment=16`).

## File-name convention

`a{active_params}_{total_params}_moe_{token_horizon}.sh` — all lowercase.
The `a` prefix marks the active-parameter count, total params follow, token
horizon comes last (e.g. `a3b_30b_moe_1t.sh` = 3B active / 30B total / 1T
tokens).

## GDP latent-MoE variants

The six scripts in `scripts/muon_lion_latent_moe_gdp/` form a parallel ladder to
`scripts/muon_lion/`. They select
`gated_delta_product_stack_spec`, use 512 latent experts with top-22 routing,
and keep the Muon/Lion optimizer and the same six active/total-parameter and
token-horizon targets.

Lion is selected for scalar parameters with `--muon-scalar-optimizer lion`.
The scaling-ladder recipes intentionally omit local CUDA graph capture arguments.

| Script | Nodes | GDP heads / groups | Latent size | Train samples | TP / EP | Micro batch |
|---|---:|---:|---:|---:|---|---:|
| `a315m_1b_moe_88b.sh` | 8 | 6 / 6 | 192 | 10,752,000 | 1 / 8 | 4 |
| `a500m_2b_moe_140b.sh` | 16 | 8 / 8 | 256 | 17,088,000 | 1 / 8 | 1 |
| `a770m_4b_moe_215b.sh` | 16 | 10 / 10 | 320 | 26,208,000 | 1 / 8 | 3 |
| `a1b_7b_moe_310b.sh` | 16 | 12 / 12 | 384 | 37,824,000 | 1 / 8 | 1 |
| `a2b_14b_moe_560b.sh` | 32 | 16 / 16 | 512 | 68,359,375 | 1 / 16 | 1 |
| `a3b_30b_moe_1t.sh` | 32 | 16 / 16 | 672 | 122,016,000 | 2 / 32 | 1 |

All GDP variants use a head dimension of 64, state dimension of 128,
`--moe-router-topk-scaling-factor 5.0`, and the per-size hybrid layer patterns,
learning rates, expert FFN sizes, and shared-expert sizes from the corresponding
latent-MoE scaling runs. The 30B variant retains the main ladder's two-depth,
shared-weight MTP configuration.

## Models

| Script | Nodes | Active / total | Tokens | Hidden | Attn heads | Mamba heads | FFN (per expert) | Init std | Shared expert | Pattern (main) | LR | Train samples | TP / EP | MTP |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| `a315m_1b_moe_88b.sh` | 8 | 315 M / 1 B | 88 B | 768 | 6 | 24 | 512 | 0.0325 | 960 | `MEMEM*EMEME*EMEM*EMEME` (22) | 2.2e-3 | 10.77 M | 1 / 8 | — |
| `a500m_2b_moe_140b.sh` | 32 | 500 M / 2 B | 140 B | 1024 | 8 | 32 | 640 | 0.028 | 1280 | `MEMEM*EMEMEM*EMEMEM*EMEMEME` (26) | 2.0e-3 | 17.09 M | 1 / 8 | — |
| `a770m_4b_moe_215b.sh` | 32 | 770 M / 4 B | 215 B | 1280 | 12 | 40 | 768 | 0.025 | 1536 | `MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEME` (32) | 1.8e-3 | 26.25 M | 1 / 8 | — |
| `a1b_7b_moe_310b.sh` | 32 | 1 B / 7 B | 310 B | 1536 | 12 | 48 | 1024 | 0.0229 | 2048 | `MEMEM*EMEMEM*EMEMEMEM*EMEMEM*EMEMEME` (35) | 1.6e-3 | 37.84 M | 1 / 8 | — |
| `a2b_14b_moe_560b.sh` | 32 | 2 B / 14 B | 560 B | 2048 | 16 | 64 | 1280 | 0.0198 | 2560 | `MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEME` (43) | 1.4e-3 | 68.36 M | 1 / 16 | — |
| `a3b_30b_moe_1t.sh` | 32 | 3 B / 30 B | 1 T | 2688 | 32 | 64 | 1856 | 0.0173 | 3712 | `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` (52) + `/*E/*E` | 1.2e-3 | 122.07 M | 2 / 32 | yes (2 depths, shared weights) |

Notes:

- `--num-experts 128`, `--moe-router-topk 6`, `--global-batch-size 768`,
  `--seq-length 8192`, `--lr-warmup-samples 1024000`, `--num-query-groups 2`,
  `--kv-channels 128`, `--pipeline-model-parallel-size 1`,
  `--expert-tensor-parallel-size 1` are identical across all six.
- `--micro-batch-size` is 4 for `a315m_1b_moe_88b.sh` and 1 for the rest.
- `--lr-decay-samples` equals `--train-samples` (full WSD decay); the
  `lr-wsd-decay` tail is 15% of train samples in each case.
- Only `a3b_30b_moe_1t.sh` enables MTP: pattern gets a `/*E/*E` suffix
  (two depths, identical `*E` pattern per depth), plus
  `--mtp-use-repeated-layer` (shared weights across depths),
  `--mtp-loss-scaling-factor 0.1`, `--calculate-per-token-loss`.

## Building the image

The image layers the components the recipes need on top of the NVIDIA PyTorch
NGC base (which already provides CUDA, NCCL, PyTorch, and TransformerEngine):
Mamba SSM kernels, HybridEP/DeepEP, NVRx, Emerging Optimizers, and assorted
training/inference packages. Build it from the bundled Dockerfile (run from the
repo root):

```bash
docker build -t scaling-ladder scaling_ladder
```

The Megatron-LM checkout is not baked in — mount your worktree at run time so
`REPO_DIR=${ROOT_DIR}/code` resolves (the scripts run
`python ${REPO_DIR}/pretrain_hybrid.py`). On Slurm with pyxis/enroot, import the
image to a `.sqsh` and point `IMAGE_PATH` at it. See `Dockerfile` for the
base-image build arg and the source revisions it builds.
