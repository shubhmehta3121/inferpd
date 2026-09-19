#!/usr/bin/env bash
# Starts the DECODER vLLM instance on Node B (8xH200).
#
# Start the PREFILLER first and wait for it to be fully up before running this.
# Check prefiller readiness: curl http://<PREFILLER_IP>:8000/health
#
# Usage (inside tmux on Node B):
#   source 00-env.sh
#   source $PYTHON_ENV_PATH/bin/activate
#   bash 04-start-decoder.sh 2>&1 | tee $VLLM_LOGS_DIR/decoder.log
#
# Startup time: same as prefiller (~10-20 min first run).
# After both are up, start the proxy (05-start-proxy.sh).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/00-env.sh"
source "$PYTHON_ENV_PATH/bin/activate"

# NIXL side-channel must advertise THIS node's IP so the prefiller can push KV.
export VLLM_NIXL_SIDE_CHANNEL_HOST="${DECODER_IP}"
export VLLM_NIXL_SIDE_CHANNEL_PORT="5600"
# With TP=8, ports 5600-5607 are allocated (base + gpu_rank).
# Ensure these are open in your firewall / security group on Node B.

# LMCache config for the decoder role.
export LMCACHE_CONFIG_FILE="${LMCACHE_DECODER_CONFIG}"

echo "Starting DECODER on ${DECODER_IP}:${VLLM_SERVE_PORT}"
echo "Model: ${MODEL_REPO}  TP=${TENSOR_PARALLEL_SIZE}"
echo "LMCache config: ${LMCACHE_CONFIG_FILE}"
echo "Expecting KV from prefiller at: ${PREFILLER_IP}"

# ------------------------------------------------------------------
# KV-transfer config:
#   kv_consumer = this instance only runs decode and receives KV from prefiller.
#   skip_last_n_tokens: 1 = the decoder generates the final token itself;
#     critical for correct autoregressive output (prefiller stops 1 token short).
#   lmcache_rpc_port: "consumer1" = unique ID for this decoder.
#     Use "consumer2" for a second decoder node if you scale out.
# ------------------------------------------------------------------
KV_TRANSFER_CONFIG='{
  "kv_connector": "LMCacheConnectorV1",
  "kv_role": "kv_consumer",
  "kv_connector_extra_config": {
    "discard_partial_chunks": false,
    "lmcache_rpc_port": "consumer1",
    "skip_last_n_tokens": 1
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
    `# Decoder handles many concurrent long-decode requests.` \
    `# More sequences = more parallel streams = better token throughput.` \
    --max-num-seqs 512 \
    --max-num-batched-tokens 32768 \
    \
    `# --- Prefix caching ---` \
    `# Disabled: LMCache on the prefiller handles caching; the decoder` \
    `# just receives and generates.` \
    --no-enable-prefix-caching \
    \
    `# --- LMCache KV connector ---` \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    \
    `# --- Eager mode (required for LMCacheConnectorV1) ---` \
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
