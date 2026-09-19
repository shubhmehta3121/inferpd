#!/bin/bash
# ============================================================
# 02_launch_prefiller.sh — Prefiller (LMCacheMPConnector)
# Cross-node: set LMCACHE_MP_CLIENT_HOST + TELEMETRY_HTTP_HOST (or full
# LMCACHE_REQUEST_TELEMETRY_ENDPOINT) in .env — see README § Two-node 1P3D.
# START ORDER: 07 (LMCache MP on Node D) → … → prefiller after proxy/decoders per README
# ============================================================

set -euo pipefail
source "$(dirname "$0")/../.env"
source /workspace/.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/workspace/logs/prefiller/prefiller_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /workspace/logs/prefiller

PREF_GPUS="${PREFILLER_GPUS:-0,1,2,3}"
MP_PORT="${LMCACHE_MP_PORT:-6000}"
MP_CLIENT="${LMCACHE_MP_CLIENT_HOST:-tcp://127.0.0.1}"

if [ -n "${LMCACHE_REQUEST_TELEMETRY_ENDPOINT:-}" ]; then
  TEL_EP="$LMCACHE_REQUEST_TELEMETRY_ENDPOINT"
else
  TELEMETRY_HTTP_HOST="${TELEMETRY_HTTP_HOST:-127.0.0.1}"
  TEL_EP="http://${TELEMETRY_HTTP_HOST}:${TELEMETRY_PORT:-5768}/api/v1/telemetry"
fi

echo "============================================================"
echo " PREFILLER — LMCacheMPConnector"
echo " GPUs (CUDA_VISIBLE): $PREF_GPUS"
echo " Port:              $PREFILLER_PORT"
echo " LMCache MP client: ${MP_CLIENT} (port ${MP_PORT} ZMQ)"
echo " Telemetry POST:    ${TEL_EP}"
echo " Weights:           $MODEL_PATH"
echo " API model name:    $MODEL_NAME"
echo " Log:               $LOG_FILE"
echo "============================================================"

export CUDA_VISIBLE_DEVICES="$PREF_GPUS"
export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD"
export VLLM_USE_DEEP_GEMM="$VLLM_USE_DEEP_GEMM"
export PYTHONHASHSEED="$PYTHONHASHSEED"

export LMCACHE_REQUEST_TELEMETRY_TYPE=fastapi
export LMCACHE_REQUEST_TELEMETRY_ENDPOINT="$TEL_EP"
export LMCACHE_SAVE_UNFULL_CHUNK="${LMCACHE_SAVE_UNFULL_CHUNK:-true}"

KV_TRANSFER_CONFIG="$(printf '%s\n' "{
  \"kv_connector\": \"LMCacheMPConnector\",
  \"kv_role\": \"kv_both\",
  \"kv_connector_extra_config\": {
    \"lmcache.mp.host\": \"${MP_CLIENT}\",
    \"lmcache.mp.port\": \"${MP_PORT}\"
  }
}")"

echo "[INFO] Starting prefiller... (logs → $LOG_FILE)"

vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PREFILLER_PORT" \
  --dtype "$DTYPE" \
  --kv-cache-dtype fp8 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --block-size "$BLOCK_SIZE" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v32 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --enforce-eager \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  2>&1 | tee "$LOG_FILE"
