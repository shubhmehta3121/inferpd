#!/bin/bash
# ============================================================
# 00_setup.sh — vLLM 0.18.0 + LMCache 0.4.2 + DeepGEMM
# ============================================================
set -euo pipefail
INITIAL_DIR="$(pwd)"

echo "============================================================"
echo " DeepSeek-V2-Lite — LMCache MP stack (1P + 2D)"
echo " GPU: 3× A100 SXM 80GB (recommended) | CUDA 12.8 | PyTorch 2.8.0"
echo "============================================================"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "[INFO] Found $NUM_GPUS GPU(s)"
if [ "$NUM_GPUS" -lt 3 ]; then
  echo "[ERROR] This layout needs at least 3 GPUs (1 prefiller + 2 decoders)."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 1. System packages ----
echo "[STEP 1] Installing system packages..."
apt-get update -y
apt-get install -y \
  python3 python3-pip python3-dev \
  git curl wget tmux htop nvtop nano \
  build-essential unzip jq

# ---- 2. UV package manager ----
echo "[STEP 2] Installing UV..."
if ! command -v uv &> /dev/null; then
  pip install uv
  export PATH="$HOME/.local/bin:$PATH"
  echo "[INFO] UV installed"
else
  export PATH="$HOME/.local/bin:$PATH"
  echo "[INFO] UV already installed"
fi

# ---- 3. Virtual environment ----
echo "[STEP 3] Creating virtual environment at /workspace/.venv..."
if [ ! -d "/workspace/.venv" ]; then
  uv venv /workspace/.venv --python 3.12
  echo "[INFO] Virtual environment created"
else
  echo "[INFO] Virtual environment already exists"
fi
source /workspace/.venv/bin/activate

# ---- 4. Python packages ----
echo "[STEP 4] Installing Python packages with UV..."

# PyTorch — use the version that came with the RunPod template
# (already installed system-wide; reinstalling here into venv)
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

uv pip install vllm==0.18.0
uv pip install lmcache==0.4.2

uv pip install fastapi uvicorn httpx aiohttp
uv pip install numpy pandas matplotlib tqdm rich openai transformers accelerate
uv pip install hf_transfer huggingface_hub
uv pip install prometheus-client

echo "[INFO] Verifying core packages..."
python -c "import torch; print(f'  PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | GPUs: {torch.cuda.device_count()}')"
python -c "import vllm; print(f'  vLLM: {vllm.__version__}')"
python -c "import importlib.metadata as m; print('  LMCache:', m.version('lmcache'))"

# ---- 5. DeepGEMM from source (NOT on PyPI — must build) ----
echo "[STEP 5] Building DeepGEMM from source..."
# DeepGEMM is required for DeepSeek MLA fused matrix multiply kernels.
# It is NOT published to PyPI — uv pip install deepgemm will always fail.
DEEPGEMM_DIR="/workspace/DeepGEMM"

if [ ! -d "$DEEPGEMM_DIR" ]; then
  echo "[INFO] Cloning DeepGEMM..."
  cd /workspace
  git clone --depth 1 https://github.com/deepseek-ai/DeepGEMM.git
  cd DeepGEMM

  git fetch --tags
  # Try v2.1.1.post3 first (vLLM DeepSeek MLA path)
  if git checkout v2.1.1.post3 2>/dev/null; then
    echo "[INFO] DeepGEMM v2.1.1.post3"
  else
    echo "[INFO] DeepGEMM: using default branch"
  fi
  git submodule update --init --recursive
  uv pip install . --no-build-isolation
else
  cd "$DEEPGEMM_DIR"
  uv pip install . --no-build-isolation
fi

cd "$INITIAL_DIR"

# Verify DeepGEMM
if python -c "import deep_gemm; print('  DeepGEMM: OK')" 2>/dev/null; then
  echo "[INFO] DeepGEMM OK"
else
  echo "[WARN] DeepGEMM import failed — MLA may fall back"
fi

echo "[STEP 6] Prometheus + Grafana (optional)..."
if ! command -v prometheus &> /dev/null; then
  cd /tmp
  wget -q https://github.com/prometheus/prometheus/releases/download/v2.51.0/prometheus-2.51.0.linux-amd64.tar.gz
  tar xzf prometheus-2.51.0.linux-amd64.tar.gz
  cp prometheus-2.51.0.linux-amd64/prometheus /usr/local/bin/
  cp prometheus-2.51.0.linux-amd64/promtool /usr/local/bin/
  mkdir -p /etc/prometheus /var/lib/prometheus
fi

if ! command -v grafana-server &> /dev/null; then
  apt-get install -y software-properties-common
  wget -q -O /usr/share/keyrings/grafana.key https://apt.grafana.com/gpg.key
  echo "deb [signed-by=/usr/share/keyrings/grafana.key] https://apt.grafana.com stable main" \
    > /etc/apt/sources.list.d/grafana.list
  apt-get update -y
  apt-get install -y grafana
fi

cd "$INITIAL_DIR"

echo "[STEP 7] Directories..."
mkdir -p /workspace/models
mkdir -p /workspace/logs/prefiller /workspace/logs/decoder /workspace/logs/proxy /workspace/logs/lmcache_mp
mkdir -p /workspace/benchmark_results

echo ""
echo "[STEP 8] Final check..."
python -c "
import importlib.metadata as md
import torch, vllm
print('  torch   :', torch.__version__)
print('  vllm    :', vllm.__version__)
print('  lmcache :', md.version('lmcache'))
print('  GPUs    :', torch.cuda.device_count())
"

echo ""
echo "============================================================"
echo " Done. source /workspace/.venv/bin/activate"
echo " Next: bash scripts/01_download_model.sh"
echo "============================================================"
