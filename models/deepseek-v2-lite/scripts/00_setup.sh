#!/bin/bash
# ============================================================
# 00_setup.sh — Install ALL dependencies on RunPod
# Run this ONCE after spinning up your pod.
# Template: PyTorch 2.8.0 + CUDA 12.8
# GPU: 2x A100 SXM 80GB
# ============================================================
set -euo pipefail
INITIAL_DIR="$(pwd)"

echo "============================================================"
echo " DeepSeek-V2-Lite PD Stack Setup"
echo " GPU: 2x A100 SXM 80GB | CUDA 12.8 | PyTorch 2.8.0"
echo "============================================================"

# ---- 0. Verify GPU count ----
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "[INFO] Found $NUM_GPUS GPU(s)"
if [ "$NUM_GPUS" -lt 2 ]; then
  echo "[ERROR] Need at least 2 GPUs. Please rent a 2x A100 SXM pod."
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

# Core inference stack
uv pip install vllm==0.16.0

# NIXL (transport layer for KV transfer — NixlConnector built into vLLM)
uv pip install nixl

# Proxy server dependencies
uv pip install fastapi uvicorn httpx aiohttp

# Benchmarking and analysis
uv pip install numpy pandas matplotlib tqdm rich openai transformers accelerate

# HuggingFace transfer acceleration + CLI
uv pip install hf_transfer huggingface_hub

# Monitoring
uv pip install prometheus-client

echo "[INFO] Verifying core packages..."
python -c "import torch; print(f'  PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | GPUs: {torch.cuda.device_count()}')"
python -c "import vllm; print(f'  vLLM: {vllm.__version__}')"
python -c "import nixl; print(f'  NIXL: OK')"

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

  # Pin to the version that matches vLLM 0.16.0
  # Fetch tags first so we can checkout the specific release
  git fetch --tags
  # Try v2.1.1.post3 first (same as V3.2 setup), fall back to latest if not found
  if git checkout v2.1.1.post3 2>/dev/null; then
    echo "[INFO] Checked out DeepGEMM v2.1.1.post3"
  else
    echo "[INFO] Tag v2.1.1.post3 not found — using latest main"
  fi

  git submodule update --init --recursive
  uv pip install . --no-build-isolation
  echo "[INFO] DeepGEMM built and installed from source"
else
  echo "[INFO] DeepGEMM already exists at $DEEPGEMM_DIR — skipping clone"
  cd "$DEEPGEMM_DIR"
  uv pip install . --no-build-isolation
fi

cd "$INITIAL_DIR"

# Verify DeepGEMM
if python -c "import deep_gemm; print(f'  DeepGEMM: OK ({deep_gemm.__file__})')" 2>/dev/null; then
  echo "[INFO] DeepGEMM verified successfully"
else
  echo "[WARN] DeepGEMM import failed — vLLM will fall back to standard kernels"
  echo "       This is non-fatal but MLA performance will be reduced."
  echo "       Try: cd /workspace/DeepGEMM && uv pip install . --no-build-isolation"
fi

# ---- 6. Install monitoring tools ----
echo "[STEP 6] Installing Prometheus and Grafana..."

# Prometheus
if ! command -v prometheus &> /dev/null; then
  cd /tmp
  wget -q https://github.com/prometheus/prometheus/releases/download/v2.51.0/prometheus-2.51.0.linux-amd64.tar.gz
  tar xzf prometheus-2.51.0.linux-amd64.tar.gz
  cp prometheus-2.51.0.linux-amd64/prometheus /usr/local/bin/
  cp prometheus-2.51.0.linux-amd64/promtool /usr/local/bin/
  mkdir -p /etc/prometheus /var/lib/prometheus
  echo "[INFO] Prometheus installed"
else
  echo "[INFO] Prometheus already installed"
fi

# Grafana
if ! command -v grafana-server &> /dev/null; then
  apt-get install -y software-properties-common
  wget -q -O /usr/share/keyrings/grafana.key https://apt.grafana.com/gpg.key
  echo "deb [signed-by=/usr/share/keyrings/grafana.key] https://apt.grafana.com stable main" \
    > /etc/apt/sources.list.d/grafana.list
  apt-get update -y
  apt-get install -y grafana
  echo "[INFO] Grafana installed"
else
  echo "[INFO] Grafana already installed"
fi

cd "$INITIAL_DIR"

# ---- 7. Create workspace directories ----
echo "[STEP 7] Creating workspace directories..."
mkdir -p /workspace/models
mkdir -p /workspace/logs/prefiller
mkdir -p /workspace/logs/decoder
mkdir -p /workspace/logs/proxy
mkdir -p /workspace/benchmark_results

# ---- 8. Final verification ----
echo ""
echo "[STEP 8] Final verification..."
python -c "
import torch, vllm, nixl
print('  torch        :', torch.__version__)
print('  vllm         :', vllm.__version__)
print('  nixl         : OK')
print('  GPUs         :', torch.cuda.device_count())
print('  CUDA         :', torch.version.cuda)
try:
    import deep_gemm
    print('  deep_gemm    : OK')
except ImportError:
    print('  deep_gemm    : NOT FOUND (non-fatal, check /workspace/DeepGEMM)')
print('All OK')
"

echo ""
echo "============================================================"
echo " Setup complete!"
echo " Virtual env: source /workspace/.venv/bin/activate"
echo " Next:        bash scripts/01_download_model.sh"
echo "============================================================"