#!/bin/bash
# ============================================================
# 02_launch_prefiller.sh — Start ONE Prefiller instance
#
# Usage: bash 02_launch_prefiller.sh <INDEX>
#   INDEX 0 → GPU 0, port 8100, NIXL side-channel 5559
#   INDEX 1 → GPU 2, port 8101, NIXL side-channel 5561
#
# Run AFTER all decoders are up and their NIXL ports are bound.
# Check: ss -tlnp | grep 5560 && ss -tlnp | grep 5562
#
# Each prefiller runs in its own tmux pane or terminal.
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../.env"

IDX=${1:-0}

GPU=${PREFILLER_GPUS[$IDX]}
PORT=${PREFILLER_PORTS[$IDX]}
NIXL_PORT=${NIXL_PORTS_PREFILLER[$IDX]}

LOG_FILE="/workspace/logs/prefiller/prefiller${IDX}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /workspace/logs/prefiller

echo "============================================================"
echo " Launching PREFILLER $IDX"
echo " GPU:          $GPU"
echo " HTTP port:    $PORT"
echo " NIXL port:    $NIXL_PORT"
echo " Model:        $MODEL_PATH"
echo " Log:          $LOG_FILE"
echo "============================================================"
echo ""
echo "[IMPORTANT] Both decoders must already be running."
echo "  Check: ss -tlnp | grep 5560 && ss -tlnp | grep 5562"
echo ""

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

echo "[INFO] Starting prefiller $IDX... (logs → $LOG_FILE)"

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
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --trust-remote-code \
  --enable-prefix-caching \
  --enforce-eager \
  --disable-log-requests \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  2>&1 | tee "$LOG_FILE"