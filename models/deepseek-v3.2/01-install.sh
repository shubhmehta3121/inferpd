#!/usr/bin/env bash
# Full dependency installation for DeepSeek-V3 on 8xH200 (Ubuntu 24 + NVIDIA driver 580+).
# Run once after provisioning the instance.
#
# Expected runtime: 15-30 min (model download happens later on first server start).
#
# Usage:
#   source 00-env.sh
#   bash 01-install.sh
set -euo pipefail
INITIAL_DIR="$(pwd)"

echo "=== [1/8] System packages ==="
apt-get update -y
apt-get install -y \
    python3 python3-pip python3-dev \
    git curl wget tmux htop nvtop nano \
    build-essential unzip

echo "=== [2/8] uv package manager ==="
pip install uv
export PATH="$HOME/.local/bin:$PATH"

echo "=== [3/8] Python virtual environment ==="
uv venv "$PYTHON_ENV_PATH" --python 3.12
# shellcheck source=/dev/null
source "$PYTHON_ENV_PATH/bin/activate"

echo "=== [4/8] PyTorch nightly (CUDA 12.8 / driver 550+) ==="
# cu124 works with driver 550+. Use cu130 only if you have driver 580+.
uv pip install --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Sanity check
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('PyTorch OK:', torch.__version__)"

echo "=== [5/8] vLLM stable + DeepGEMM from source ==="
uv pip install --upgrade vllm

# DeepGEMM required for MQA logits (VLLM_USE_DEEP_GEMM=0 disables only MoE for stability)
BUILD_DIR="${BUILD_DIR:-/tmp}"
cd "$BUILD_DIR"
git clone --depth 1 https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM
git fetch --tags && git checkout v2.1.1.post3
git submodule update --init --recursive
pip install . --no-build-isolation
cd "$INITIAL_DIR"

echo "=== [6/8] LMCache + NIXL ==="
uv pip install lmcache
uv pip install nixl

echo "=== [7/8] Benchmark / monitoring helpers ==="
uv pip install \
    openai \
    transformers \
    accelerate \
    numpy \
    tqdm \
    lm-eval \
    hf_transfer

echo ""
echo "=== Verification ==="
python -c "
import torch, vllm, transformers, numpy, lmcache
print('torch        :', torch.__version__)
print('vllm         :', vllm.__version__)
print('transformers :', transformers.__version__)
print('lmcache      :', lmcache.__file__)   # shows where it's installed
print('GPUs         :', torch.cuda.device_count())
print('CUDA         :', torch.version.cuda)
print('All OK')
"

echo ""
echo "Installation complete. Start a new shell or run:"
echo "  source $PYTHON_ENV_PATH/bin/activate"
echo "  source 00-env.sh"
