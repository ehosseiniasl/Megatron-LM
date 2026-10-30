#! /bin/bash

set -e

echo "Changing to repository root directory..."
cd "$(dirname "$0")/.."

# Parse command line arguments
CLEAN_CHECK=false
UPDATE_LATEST=false
USE_CACHE=false

function retag_image() {
    local old_tag=$1
    local new_tag=$2
    echo "Retagging $repo:$old_tag to $repo:$new_tag"
    docker pull $repo:$old_tag
    docker tag $repo:$old_tag $repo:$new_tag
    docker push $repo:$new_tag
    echo "Successfully retagged image to $repo:$new_tag"
}

function print_usage() {
  echo "Usage: $0 [-c] [-u] [-q]"
  echo "  -c    Ensure working directory is clean (exit if not clean)"
  echo "  -u    Also push to tag without date suffix"
  echo "  -q    Use Docker cached layers (default: false)"
  exit 1
}

echo "Parsing command line arguments..."
while getopts "cuq" opt; do
  case $opt in
  c) CLEAN_CHECK=true ;;
  u) UPDATE_LATEST=true ;;
  q) USE_CACHE=true ;;
  *) print_usage ;;
  esac
done

echo "Checking git status..."
git status
if [ -n "$(git status --porcelain)" ]; then
  echo "Error: Working directory is not clean. Please commit or stash your changes."
  if [ "$CLEAN_CHECK" = true ]; then
    exit 1
  fi
fi

if [ -n "$(git diff)" ]; then
  echo "Error: There are uncommitted changes. Please commit or stash your changes."
  if [ "$CLEAN_CHECK" = true ]; then
    exit 1
  fi
fi

echo "Checking out adlr/nemo5 branch..."
git checkout adlr/nemo5

if [ "$CLEAN_CHECK" = true ]; then
  echo "Pulling latest changes from remote..."
  git pull
fi

echo "Setting up Docker repository and tag information..."
repo=gitlab-master.nvidia.com:5005/swdl-nemollm-mlops/evals/lm-evaluation-harness
suffix=$(date +%m%d%Y)

# Define an array of Dockerfile configurations
declare -A dockerfiles
dockerfiles=(
  ["Dockerfile.vllm"]="adlr-nemo5-vllm-$suffix"
  ["Dockerfile"]="adlr-nemo5-$suffix"
)

# Loop through each Dockerfile and build
for dockerfile in "${!dockerfiles[@]}"; do
  tag=${dockerfiles[$dockerfile]}
  echo "Building Docker image from $dockerfile with tag $repo:$tag..."
  
  # Set cache option based on USE_CACHE flag
  cache_option=""
  if [ "$USE_CACHE" != true ]; then
    cache_option="--no-cache"
  fi
  
  docker build -f $dockerfile --network=host $cache_option -t $repo:$tag .
  echo "Pushing Docker image $repo:$tag to registry..."
  docker push $repo:$tag
  
  # If -u flag is set, also push to tag without suffix
  if [ "$UPDATE_LATEST" = true ]; then
    latest_tag=${tag%-$suffix}
    echo "Tagging $repo:$tag as $repo:$latest_tag..."
    docker tag $repo:$tag $repo:$latest_tag
    echo "Pushing Docker image $repo:$latest_tag to registry..."
    docker push $repo:$latest_tag
    echo "Also pushed to $repo:$latest_tag"
  fi
done
