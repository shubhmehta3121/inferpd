#!/bin/bash
# ============================================================
# 03_launch_decoder.sh — Decoder (LMCacheMPConnector)
# One vLLM process, TP+EP across DECODER_GPUS.
# Cross-node: set LMCACHE_MP_CLIENT_HOST (Node D → tcp://127.0.0.1, Node P → tcp://NodeD_IP).
# START ORDER: 07 → 04 → all decoders → 02 (prefiller last)
# ============================================================

set -euo pipefail
source "$(dirname "$0")/../.env"
source /workspace/.venv/bin/activate

# Instance B: 0/1 → D0/D1 (never use :-$DECODER_PORT with set -u when DECODER_PORT may be unset).
if [[ "${1:-}" == "0" ]]; then
  [[ -n "${DECODER_PORT_D0:-}" ]] && DECODER_PORT="$DECODER_PORT_D0"
  [[ -n "${DECODER_GPUS_D0:-}" ]] && DECODER_GPUS="$DECODER_GPUS_D0"
  shift
elif [[ "${1:-}" == "1" ]]; then
  [[ -n "${DECODER_PORT_D1:-}" ]] && DECODER_PORT="$DECODER_PORT_D1"
  [[ -n "${DECODER_GPUS_D1:-}" ]] && DECODER_GPUS="$DECODER_GPUS_D1"
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/workspace/logs/decoder/decoder_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /workspace/logs/decoder

DEC_GPUS="${DECODER_GPUS:-4,5,6,7}"
MP_PORT="${LMCACHE_MP_PORT:-6000}"
MP_CLIENT="${LMCACHE_MP_CLIENT_HOST:-tcp://127.0.0.1}"

if [[ -z "${DECODER_PORT:-}" ]]; then
  echo "[ERROR] DECODER_PORT is unset. Instance A: set DECODER_PORT in .env. Instance B: cp .env.b .env and run with 0 or 1 (DECODER_PORT_D0/D1)." >&2
  exit 1
fi

echo "============================================================"
echo " DECODER — LMCacheMPConnector"
echo " GPUs (CUDA_VISIBLE): $DEC_GPUS"
echo " Port:              $DECODER_PORT"
echo " LMCache MP client: ${MP_CLIENT} (port ${MP_PORT} ZMQ)"
echo " Weights:           $MODEL_PATH"
echo " API model name:    $MODEL_NAME"
echo " Log:               $LOG_FILE"
echo "============================================================"

export CUDA_VISIBLE_DEVICES="$DEC_GPUS"
export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD"
export VLLM_USE_DEEP_GEMM="$VLLM_USE_DEEP_GEMM"
export PYTHONHASHSEED="$PYTHONHASHSEED"
export LMCACHE_SAVE_UNFULL_CHUNK="${LMCACHE_SAVE_UNFULL_CHUNK:-true}"

KV_TRANSFER_CONFIG="$(printf '%s\n' "{
  \"kv_connector\": \"LMCacheMPConnector\",
  \"kv_role\": \"kv_both\",
  \"kv_connector_extra_config\": {
    \"lmcache.mp.host\": \"${MP_CLIENT}\",
    \"lmcache.mp.port\": \"${MP_PORT}\"
  }
}")"

echo "[INFO] Starting decoder... (logs → $LOG_FILE)"

vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$DECODER_PORT" \
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
