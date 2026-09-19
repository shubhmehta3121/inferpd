#!/usr/bin/env bash
# Starts the PREFILLER vLLM instance on Node A (8xH200).
#
# Run this BEFORE starting the decoder.  The decoder's NIXL handshake
# connects back to the prefiller's side-channel port, so the prefiller
# must be fully up first.
#
# Usage (inside tmux on Node A):
#   source 00-env.sh
#   source $PYTHON_ENV_PATH/bin/activate
#   bash 03-start-prefiller.sh 2>&1 | tee $VLLM_LOGS_DIR/prefiller.log
#
# Startup time: 10-20 min on first run (FP8 GEMM warmup + CUDA graph capture).
# Wait for "Uvicorn running on http://0.0.0.0:8000" before starting decoder.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/00-env.sh"
source "$PYTHON_ENV_PATH/bin/activate"

# NIXL side-channel must advertise THIS node's IP so the decoder can reach it.
export VLLM_NIXL_SIDE_CHANNEL_HOST="${PREFILLER_IP}"
export VLLM_NIXL_SIDE_CHANNEL_PORT="5600"
# With TP=8, ports 5600-5607 are allocated (base + gpu_rank).
# Ensure these are open in your firewall / security group on Node A.

# LMCache config for the prefiller role.
export LMCACHE_CONFIG_FILE="${LMCACHE_PREFILLER_CONFIG}"

echo "Starting PREFILLER on ${PREFILLER_IP}:${VLLM_SERVE_PORT}"
echo "Model: ${MODEL_REPO}  TP=${TENSOR_PARALLEL_SIZE}"
echo "LMCache config: ${LMCACHE_CONFIG_FILE}"

# ------------------------------------------------------------------
# KV-transfer config:
#   kv_producer  = this instance only performs prefill and sends KV.
#   discard_partial_chunks: false = keep boundary chunks for accurate cache hits.
#   lmcache_rpc_port: "producer1" = unique ID for this prefiller instance.
#     If you later add a second prefiller node, use "producer2" there.
# ------------------------------------------------------------------
KV_TRANSFER_CONFIG='{
  "kv_connector": "LMCacheConnectorV1",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "discard_partial_chunks": false,
    "lmcache_rpc_port": "producer1"
  }
}'

exec vllm serve "$MODEL_REPO" \
    \
    `# --- Parallelism ---` \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size 1 \
    \
    `# --- Precision ---` \
    --dtype bfloat16 \
    \
    `# --- Context & batching ---` \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    \
    `# --- Prefix caching ---` \
    `# vLLM's built-in APC is DISABLED when using LMCacheConnectorV1 in disagg` \
    `# mode — LMCache manages prefix caching itself via its CPU cache.` \
    --no-enable-prefix-caching \
    `# Chunked prefill still helps interleave work and manage TTFT spikes.` \
    --enable-chunked-prefill \
    --max-num-chunked-tokens 8192 \
    \
    `# --- LMCache KV connector ---` \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    \
    `# --- Eager mode ---` \
    `# enforce-eager disables CUDA graph capture; required for LMCacheConnectorV1` \
    `# in current vLLM nightly.  Remove if a future release supports graphs here.` \
    --enforce-eager \
    \
    `# --- Networking ---` \
    --host 0.0.0.0 \
    --port "$VLLM_SERVE_PORT" \
    \
    `# --- Metrics ---` \
    --enable-metrics \
    --metrics-port "$VLLM_METRICS_PORT" \
    \
    `# --- Logging ---` \
    --disable-log-requests \
    --uvicorn-log-level warning
