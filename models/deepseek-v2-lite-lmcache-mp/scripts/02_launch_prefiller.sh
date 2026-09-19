#!/bin/bash
# ============================================================
# 02_launch_prefiller.sh — Prefiller (LMCacheMPConnector)
# START ORDER: 07 (LMCache MP) → 04 (proxy) → 03 (decoder) → this
# ============================================================

set -euo pipefail
source "$(dirname "$0")/../.env"
source /workspace/.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/workspace/logs/prefiller/prefiller_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /workspace/logs/prefiller

echo "============================================================"
echo " PREFILLER — LMCacheMPConnector"
echo " GPU:               $PREFILLER_GPU"
echo " Port:              $PREFILLER_PORT"
echo " LMCache MP:        ${LMCACHE_MP_PORT:-6000} (ZMQ)"
echo " Telemetry POST:    http://localhost:${TELEMETRY_PORT:-5768}/api/v1/telemetry"
echo " Model:             $MODEL_PATH"
echo " Log:               $LOG_FILE"
echo "============================================================"

export CUDA_VISIBLE_DEVICES="$PREFILLER_GPU"
export VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD"
export VLLM_USE_DEEP_GEMM="$VLLM_USE_DEEP_GEMM"
export PYTHONHASHSEED="$PYTHONHASHSEED"

# LMCache: scheduler adapter posts here when KV store finishes (disagg_prefill_mp)
export LMCACHE_REQUEST_TELEMETRY_TYPE=fastapi
export LMCACHE_REQUEST_TELEMETRY_ENDPOINT="http://localhost:${TELEMETRY_PORT}/api/v1/telemetry"
export LMCACHE_SAVE_UNFULL_CHUNK="${LMCACHE_SAVE_UNFULL_CHUNK:-true}"

MP_PORT="${LMCACHE_MP_PORT:-6000}"
KV_TRANSFER_CONFIG="$(printf '%s\n' "{
  \"kv_connector\": \"LMCacheMPConnector\",
  \"kv_role\": \"kv_both\",
  \"kv_connector_extra_config\": {
    \"lmcache.mp.host\": \"tcp://localhost\",
    \"lmcache.mp.port\": \"${MP_PORT}\"
  }
}")"

echo "[INFO] Starting prefiller... (logs → $LOG_FILE)"

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PREFILLER_PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --block-size "$BLOCK_SIZE" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --enforce-eager \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  2>&1 | tee "$LOG_FILE"
