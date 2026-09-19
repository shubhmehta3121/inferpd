#!/usr/bin/env bash
# Launches the vLLM OpenAI-compatible server for DeepSeek-V3 on 8xH200.
#
# Stack:
#   vLLM (TP=8, BF16) → LMCacheConnectorV1 (kv_both)
#   LMCache: GPU APC + CPU spill (200 GB) + CacheGen compression
#
# Usage (inside a tmux session):
#   source 00-env.sh
#   source $PYTHON_ENV_PATH/bin/activate
#   bash 02-start-server.sh 2>&1 | tee $VLLM_LOGS_DIR/server.log
#
# First start: 30-60 min total (model download + FP8 GEMM warmup + CUDA graph capture).
# Subsequent restarts: ~5-10 min (all caches warm).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/00-env.sh"
source "$PYTHON_ENV_PATH/bin/activate"

echo "Starting vLLM server — model: $MODEL_REPO"
echo "TP=$TENSOR_PARALLEL_SIZE  max_model_len=$MAX_MODEL_LEN  port=$VLLM_SERVE_PORT"

# ------------------------------------------------------------------
# KV-transfer config: LMCacheConnectorV1 in kv_both mode
#
# kv_both = this single instance handles both prefill and decode.
# LMCache persists hot KV blocks to CPU RAM (CacheGen compressed)
# so repeated prefixes skip GPU recomputation on future requests.
# ------------------------------------------------------------------
KV_TRANSFER_CONFIG='{
  "kv_connector": "LMCacheConnectorV1",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "discard_partial_chunks": false
  }
}'

exec vllm serve "$MODEL_REPO" \
    \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    # --pipeline-parallel-size 1 \
    \
    --dtype float8 \
    --tokenizer-mode deepseek_v32 \
    --tool-call-parser deepseek_v32 \
    --disable-custom-all-reduce \
    --disable-cuda-graph \
    --enable-auto-tool-choice \
    \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    `# Reduced from 256 to 128: limits concurrent request sequences to prevent memory spikes` \
    --max-num-seqs 128 \
    `# Reduced from 32768 to 16384: limits batched token processing to reduce peak memory usage` \
    --max-num-batched-tokens 16384 \
    \
    `# APC caches hot KV blocks in GPU VRAM — critical for 80% prefix hit rate` \
    --enable-prefix-caching \
    `# Chunked prefill interleaves prefill+decode steps, reduces TTFT spikes` \
    --enable-chunked-prefill \
    \
    # --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    \
    --enforce-eager \
    --host 0.0.0.0 \
    --port "$VLLM_SERVE_PORT" \
    \
    --uvicorn-log-level warning
