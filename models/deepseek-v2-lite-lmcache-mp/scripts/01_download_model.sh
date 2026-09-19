#!/bin/bash
# ============================================================
# 01_download_model.sh — Download DeepSeek-V2-Lite
# Saves to /workspace/models (persistent volume) so you don't
# re-download on pod restart.
# ============================================================
set -e

source "$(dirname "$0")/../.env"
source /workspace/.venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER

MODEL_DIR="/workspace/models/deepseek-v2-lite"

echo "============================================================"
echo " Downloading: deepseek-ai/DeepSeek-V2-Lite"
echo " Destination: $MODEL_DIR"
echo " Precision:   BF16 (native from HuggingFace)"
echo " Approx size: ~31 GB"
echo "============================================================"

if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "hf_YOUR_TOKEN_HERE" ]; then
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
  deepseek-ai/DeepSeek-V2-Lite \
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
