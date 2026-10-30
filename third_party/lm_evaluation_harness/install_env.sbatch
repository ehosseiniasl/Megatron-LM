#!/bin/bash
#SBATCH --job-name=install_env
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=install_env_%j.out
#SBATCH --error=install_env_%j.err

# This script should be run via ./install.sh YOUR_HF_TOKEN
# which validates the token before submitting

set -e

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set. Use ./install.sh YOUR_HF_TOKEN"
    exit 1
fi

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH (it installs to ~/.local/bin or ~/.cargo/bin)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Create virtual environment with isolated Python (no dependency on system Python)
# uv will download and manage Python 3.11 in the virtualenv
uv venv env --python 3.11

source env/bin/activate

# Install dependencies using uv
uv pip install -e .

# Configure HuggingFace token
hf auth login --token "$HF_TOKEN"

# Add HF_TOKEN to activate script so it's always available
echo "" >> env/bin/activate
echo "# HuggingFace token for dataset downloads" >> env/bin/activate
echo "export HF_TOKEN=\"$HF_TOKEN\"" >> env/bin/activate

echo "Installation complete. Environment created in 'env' directory."
echo "HuggingFace token configured and added to env/bin/activate."
