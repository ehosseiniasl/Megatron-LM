#!/bin/bash

# A3B/30B GDP + Muon/Adam N4-v6 VLM launcher: AGA, 1_60_subsampled data.

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
#SBATCH --job-name=a3b30b_n4v6_vlm_300b_aga_1_60_subsampled_nobinhvq_fa2_skipshape
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/sasatheesh/data/nemo3_nano/slurm/%x_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/sasatheesh/data/nemo3_nano/slurm/%x_%j.err

RUN_NAME="${RUN_NAME_OVERRIDE:-a3b30b_n4v6_vlm_300b_aga_1_60_subsampled_nobinhvq_fa2_skipshape}"
EXPECTED_CLUSTER_NAME=oci-aga-slurm-1
IMAGE_PATH="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron4/images/nvidia+pytorch+26.07-py3+deps+v5.sqsh"
PYTHON_PACKAGES="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/pengx/projects/nv-mistralai-megatron/examples/multimodal/migration/running_text_chain/outputs/container-python-packages"
NEW_VISION_WEIGHTS="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/pengx/projects/mistral_encoder_tasks/vision_encoder_260811_mcore"
OLD_VISION_WEIGHTS="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/pengx/projects/mistral/vision_encoder_mcore"
SOURCE_PRETOKENIZED_CONFIG="${SLURM_SUBMIT_DIR}/mistral_adapter/strawman_60t_v0.6_20260901_phase1_80_phase2_20_smart/phase1.yaml"
PERCENTAGE_DATA=0.016667
EPOCH_INDEX_CACHE="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/jkamalu/mistral-derisking/nv-mistralai-megatron.rwaleffe.main/workdir/epoch-index-8020smartp1-1over60-1p2t-nobinhvq"
LOCAL_IMAGE_ROOT="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron-v4/sandia/image_buckets"
EXTRA_REQUIRED_PATHS=("${LOCAL_IMAGE_ROOT}")
RUNTIME_ENVIRONMENT=("MISTRAL_LOCAL_IMAGE_ROOT=${LOCAL_IMAGE_ROOT}")
UNSET_ENVIRONMENT=(AWS_PROFILE MISTRAL_S3_ENDPOINT_URL)
source "${SLURM_SUBMIT_DIR}/scaling_ladder/scripts/muon_adam_latent_moe_gdp_mistral_dataloader_vlm/a3b_30b_moe_1t_n4_v6_common.sh"
