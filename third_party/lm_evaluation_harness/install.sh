#!/bin/bash
# Wrapper script to install lm-eval with HuggingFace token validation

if [ -z "$1" ]; then
    echo "Error: HuggingFace token required."
    echo ""
    echo "Usage: ./install.sh YOUR_HF_TOKEN"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

HF_TOKEN="$1"

# Basic validation - HF tokens start with "hf_"
if [[ ! "$HF_TOKEN" =~ ^hf_ ]]; then
    echo "Error: Invalid HuggingFace token format (should start with 'hf_')"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

ACCOUNT=$(sacctmgr -nP show assoc where user=$(whoami) format=account | head -1)
if [ -z "$ACCOUNT" ]; then
    echo "Error: No SLURM account found for user $(whoami)"
    exit 1
fi

echo "Submitting install job with HuggingFace token (account: $ACCOUNT)..."
sbatch -A "$ACCOUNT" --export=ALL,HF_TOKEN="$HF_TOKEN" install_env.sbatch
