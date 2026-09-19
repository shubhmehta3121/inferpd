#!/bin/bash
# ============================================================
# 02_launch_prefiller.sh — Start vLLM Prefiller (KV Producer)
# GPU 0 | Port 8100 | NixlConnector
#
# KNOWN BUG — vLLM 0.16.0 + NixlConnector + prefix caching:
#   --enable-prefix-caching crashes the EngineCore on first request:
#   nixl_connector.py:725: assert num_external_tokens == 0 — AssertionError
#   NixlConnector's scheduler misreads locally-cached prefix blocks
#   as "external tokens pending transfer", violating its invariant.
#
# FIX for now:    --no-enable-prefix-caching (system works, no crash)
# FIX permanently: uv pip install "vllm>=0.8.5" (bug is fixed upstream)
#
# START ORDER: proxy → decoder → prefiller
# ============================================================

source "$(dirname "$0")/../.env"
source /workspace/.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/workspace/logs/prefiller/prefiller_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /workspace/logs/prefiller

echo "============================================================"
echo " Launching PREFILLER (kv_producer) — NixlConnector"
echo " GPU:               $PREFILLER_GPU"
echo " vLLM port:         $PREFILLER_PORT"
echo " NIXL side channel: ${NIXL_SIDE_CHANNEL_PORT_PREFILLER:-5559}"
echo " Model:             $MODEL_PATH"
echo " Max Seq Len:       $MAX_MODEL_LEN"
echo " Mem Util:          $GPU_MEM_UTIL"
echo " Prefix cache:      DISABLED (vLLM 0.16.0 + NixlConnector bug)"
echo " KV fail policy:    fail"
echo " Log:               $LOG_FILE"
echo "============================================================"
echo ""
echo "[IMPORTANT] Decoder must already be running."
echo "  Check: ss -tlnp | grep ${NIXL_SIDE_CHANNEL_PORT_DECODER:-5560}"
echo ""

export UCX_TLS="$UCX_TLS"
export CUDA_VISIBLE_DEVICES="$PREFILLER_GPU"
export VLLM_NIXL_SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT_PREFILLER:-5559}"
export VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD"
export VLLM_USE_DEEP_GEMM="$VLLM_USE_DEEP_GEMM"
export PYTHONHASHSEED="$PYTHONHASHSEED"

# kv_load_failure_policy: "fail" — request errors hard if KV transfer fails.
# Never use "recompute" — it silently runs full prefill on the decoder,
# hiding transfer failures and destroying TTFT with no visible error.
KV_TRANSFER_CONFIG='{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_both",
  "kv_buffer_device": "cuda",
  "kv_load_failure_policy": "fail"
}'

echo "[INFO] Starting prefiller... (logs → $LOG_FILE)"

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PREFILLER_PORT" \
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