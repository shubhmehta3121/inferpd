#!/bin/bash
# ============================================================
# 03_launch_decoder.sh — Start vLLM Decoder (KV Consumer)
# GPU 1 | Port 8200 | NixlConnector
#
# --no-enable-prefix-caching is correct on the decoder.
# The decoder never runs prefill — prefix caching would waste
# VRAM for zero benefit.
#
# START ORDER: proxy → decoder → prefiller
#   This MUST be up before the prefiller starts.
#   Prefiller connects to decoder's NIXL side channel at startup.
# ============================================================

source "$(dirname "$0")/../.env"
source /workspace/.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/workspace/logs/decoder/decoder_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /workspace/logs/decoder

echo "============================================================"
echo " Launching DECODER (kv_consumer) — NixlConnector"
echo " GPU:               $DECODER_GPU"
echo " vLLM port:         $DECODER_PORT"
echo " NIXL side channel: ${NIXL_SIDE_CHANNEL_PORT_DECODER:-5560}"
echo " Model:             $MODEL_PATH"
echo " Max Seq Len:       $MAX_MODEL_LEN"
echo " Mem Util:          $GPU_MEM_UTIL"
echo " Prefix cache:      DISABLED (decoder never prefills — correct)"
echo " KV fail policy:    fail"
echo " Log:               $LOG_FILE"
echo "============================================================"

export UCX_TLS="$UCX_TLS"
export CUDA_VISIBLE_DEVICES="$DECODER_GPU"
export VLLM_NIXL_SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT_DECODER:-5560}"
export VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD"
export VLLM_USE_DEEP_GEMM="$VLLM_USE_DEEP_GEMM"
export PYTHONHASHSEED="$PYTHONHASHSEED"

KV_TRANSFER_CONFIG='{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_both",
  "kv_buffer_device": "cuda",
  "kv_load_failure_policy": "fail"
}'

echo "[INFO] Starting decoder... (logs → $LOG_FILE)"

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$DECODER_PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --block-size "$BLOCK_SIZE" \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --trust-remote-code \
  --enable-prefix-caching \
  --enforce-eager \
  --disable-log-requests \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  2>&1 | tee "$LOG_FILE"