#!/bin/bash

# A3B/30B GDP + Muon/Adam N4-v6 VLM launcher: CMH, full data.

#SBATCH -p batch
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=136
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --segment=4
#SBATCH --overcommit
#SBATCH --dependency=singleton
#SBATCH --job-name=a3b30b_n4v6_vlm_300b_cmh_full_nobinhvq_fa2_skipshape
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/sasatheesh/data/nemo3_nano/slurm/%x_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/sasatheesh/data/nemo3_nano/slurm/%x_%j.err

RUN_NAME="${RUN_NAME_OVERRIDE:-a3b30b_n4v6_vlm_300b_cmh_full_nobinhvq_fa2_skipshape}"
EXPECTED_CLUSTER_NAME=aws-cmh-slurm-1
IMAGE_PATH="/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/sasatheesh/data/images/m_lm_energon_cmh_gb300_pre_vlm_05_pre_energon_07_efa_pt2602.sqsh"
PYTHON_PACKAGES="/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/sasatheesh/data/m-lm-mistralai/container-python-packages-nvrx-0.6.0-py312"
NEW_VISION_WEIGHTS="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/kshih/workspace/weights/mistral_vit/vision_encoder_260811_mcore"
OLD_VISION_WEIGHTS="/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/sasatheesh/data/m-lm-mistralai/encoder_checkpoints/vision_encoder_mcore"
SOURCE_PRETOKENIZED_CONFIG="${SOURCE_PRETOKENIZED_CONFIG_OVERRIDE:-${SLURM_SUBMIT_DIR}/mistral_adapter/strawman_60t_v0.6_20260901_phase1_80_phase2_20_smart/combined.yaml}"
PERCENTAGE_DATA=1.0
EVAL_INTERVAL_OVERRIDE=100
EPOCH_INDEX_CACHE="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/sasatheesh/data/m-lm-mistralai/epoch_indexes_vlm_8k_3t_n4_v6_262k_strawman_v06_20260901_combined_fast_rle_shallow_cmh_full_epoch"
VLM_SFT_IMAGE_ROOT="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/shared/jsonl/n4/nvidia/multimodal-sft/vlm_sft_25b_jsonl_0825"
EXTRA_REQUIRED_PATHS=(
    "${VLM_SFT_IMAGE_ROOT}"
    /home/sasatheesh/.aws/config
    "${EPOCH_INDEX_CACHE}/.prepack_complete"
)
RUNTIME_ENVIRONMENT=(
    AWS_PROFILE=nvidia-sandia-cmh
    MISTRAL_S3_ENDPOINT_URL=https://s3.us-east-2.amazonaws.com
)
UNSET_ENVIRONMENT=(MISTRAL_LOCAL_IMAGE_ROOT)
source "${SLURM_SUBMIT_DIR}/scaling_ladder/scripts/muon_adam_latent_moe_gdp_mistral_dataloader_vlm/a3b_30b_moe_1t_n4_v6_common.sh"
