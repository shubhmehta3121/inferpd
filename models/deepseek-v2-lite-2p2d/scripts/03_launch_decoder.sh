#!/bin/bash
# ============================================================
# 03_launch_decoder.sh — Start ONE Decoder instance
#
# Usage: bash 03_launch_decoder.sh <INDEX>
#   INDEX 0 → GPU 1, port 8200, NIXL side-channel 5560
#   INDEX 1 → GPU 3, port 8201, NIXL side-channel 5562
#
# Run BEFORE launching any prefillers.
# Prefillers connect to decoder NIXL side-channels at startup.
#
# Each decoder runs in its own tmux pane or terminal.
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../.env"

IDX=${1:-0}

GPU=${DECODER_GPUS[$IDX]}
PORT=${DECODER_PORTS[$IDX]}
NIXL_PORT=${NIXL_PORTS_DECODER[$IDX]}

LOG_FILE="/workspace/logs/decoder/decoder${IDX}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /workspace/logs/decoder

echo "============================================================"
echo " Launching DECODER $IDX"
echo " GPU:          $GPU"
echo " HTTP port:    $PORT"
echo " NIXL port:    $NIXL_PORT  ← prefillers will connect here"
echo " Model:        $MODEL_PATH"
echo " Log:          $LOG_FILE"
echo "============================================================"

export UCX_TLS="$UCX_TLS"
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_NIXL_SIDE_CHANNEL_PORT="$NIXL_PORT"
export VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD"
export VLLM_USE_DEEP_GEMM="$VLLM_USE_DEEP_GEMM"
export PYTHONHASHSEED="$PYTHONHASHSEED"

source /workspace/.venv/bin/activate

KV_TRANSFER_CONFIG='{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_both",
  "kv_buffer_device": "cuda",
  "kv_load_failure_policy": "fail"
}'

echo "[INFO] Starting decoder $IDX... (logs → $LOG_FILE)"

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
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