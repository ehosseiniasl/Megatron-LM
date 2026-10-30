# Nemotron-4 derisking runs

The sbatch scripts behind the Nemotron-4 derisking runs, as submitted on aws-cmh
from `/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron4/derisking/`.

Both are GB300 at 4 GPUs per node, sequence length 8192, Muon, a hybrid gated
delta product stack with a 512 expert latent MoE, multi-token prediction, and
the multimodal pretokenized dataloader.

## Layout

`<codename>/<phase>.sh`, mirroring `derisking/<codename>/phase1/run.sh` on the
cluster. The codename is the directory so later phases sit beside each other,
and no model size is baked into a filename.

## Reading the sample counts

`nano_21t` names two figures apart:

| | |
|---|---|
| `PHASE1_TRAIN_SAMPLES` | what this phase trains, 10T tokens; `--train-samples` reads it |
| `TOTAL_TRAIN_SAMPLES` | the whole run, 21T tokens |

The learning rate schedule and the batch size ramp are sized against the total,
so a later phase continues them rather than restarting. `super_3t` has a single
`TRAIN_SAMPLES`; its run is one phase.

## Differences between the two

| | `nano_21t` | `super_3t` |
|---|---|---|
| Nodes | 400 | 608 |
| Tokens this phase | 10.0T | 3.1T |
| Layers | 62 (32 M, 30 E) | 78 (40 M, 38 E) |
| Hidden size | 3072 | 5632 |
| Attention heads | 16 | 32 |
| Mamba heads / groups | 24 / 24 | 40 / 40 |
| FFN | 12288 | 22528 |
| Init method std | 0.0162 | 0.0119 |
| MoE latent size | 768 | 1536 |
| MoE FFN | 4096 | 7168 |
| Shared expert intermediate | 4096 | 7168 |
| Micro batch | 3 | 1 |
| Max global batch | 4608 | 6912 |
| Batch size ramp | 1536 to 3072 to 4608 | 2304 to 4608 to 6912 |
| LR / min LR | 8.0e-4 / 2.0e-5 | 6.0e-4 / 1.5e-5 |
| MIMO encoder DP | 64 | 128 |
| MIMO LLM offset | 64 | 128 |
| MIMO LLM DP | 24 | 36 |
| Fraction of blend | 1.0 | 0.051666 |
