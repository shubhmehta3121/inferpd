#!/bin/bash
# ============================================================
# 01_download_model.sh — Download nvidia/DeepSeek-V3.2-NVFP4
# Saves to /workspace/models (persistent volume) so you don't
# re-download on pod restart.
# ============================================================
set -e

source "$(dirname "$0")/../.env"
source /workspace/.venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

MODEL_DIR="${MODEL_PATH:-/workspace/models/deepseek-v3.2-lmcmp-1p3d}"

echo "============================================================"
echo " Downloading: nvidia/DeepSeek-V3.2-NVFP4"
echo " Destination: $MODEL_DIR"
echo " Precision:   NVFP4 (Hub)"
echo " Approx size: ~415 GB (enable HF_HUB_ENABLE_HF_TRANSFER=1 in .env)"
echo "============================================================"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[ERROR] Set HF_TOKEN in .env before running this script."
  echo "        Get your token at: https://huggingface.co/settings/tokens"
  exit 1
fi

if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
  echo "[INFO] Model already exists at $MODEL_DIR — skipping download."
  echo "       Delete $MODEL_DIR to force re-download."
  exit 0
fi

mkdir -p "$MODEL_DIR"

echo "[INFO] Starting download via huggingface-cli..."
huggingface-cli download \
  nvidia/DeepSeek-V3.2-NVFP4 \
  --local-dir "$MODEL_DIR" \
  --token "$HF_TOKEN" \
  --repo-type model

echo ""
echo "[INFO] Download complete!"
echo "[INFO] Model size on disk:"
du -sh "$MODEL_DIR"

echo ""
echo "[INFO] Model files:"
ls -lh "$MODEL_DIR"/*.safetensors 2>/dev/null | head -20 || \
ls -lh "$MODEL_DIR"/*.bin 2>/dev/null | head -20

echo ""
echo "============================================================"
echo " Model ready at: $MODEL_DIR"
echo " Next: Run 02_launch_prefiller.sh"
echo "============================================================"
