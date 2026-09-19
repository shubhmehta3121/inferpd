#!/usr/bin/env bash
# Common environment variables for the single-node (v1) deployment.
# Source this before running any script:
#   source /workspace/deepseek-deploy/00-env.sh

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
set -a
# shellcheck source=.env
source "$(dirname "${BASH_SOURCE[0]}")/.env"
set +a

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
export DEPLOY_DIR="/workspace/deepseek-deploy"
export PYTHON_ENV_PATH="/workspace/.venv"
export HF_HOME="/workspace/.cache/huggingface"
export VLLM_LOGS_DIR="/workspace/logs"
export LMCACHE_DIR="/workspace/lmcache_storage"
export BENCHMARK_OUTPUT_DIR="${DEPLOY_DIR}/outputs/benchmarks"

# Vast.ai /workspace is persistent; use /root or /home/ubuntu on RunPod if needed.

# ---------------------------------------------------------------------------
# Driver / CUDA (fix vLLM 0.16 Error 803 on H200)
# ---------------------------------------------------------------------------
# Prepend PyTorch's bundled cuDNN to avoid system cuDNN version mismatch (9.8 vs 9.10)
# TORCH_LIB="$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)"
# export LD_LIBRARY_PATH="${TORCH_LIB:+$TORCH_LIB:}/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu"
# export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu"
# Add PyTorch's library directory to LD_LIBRARY_PATH (critical for DeepGEMM)
TORCH_LIB="$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)"
if [ -n "$TORCH_LIB" ] && [ -d "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu"
fi

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Stable model (not -Exp). https://huggingface.co/deepseek-ai/DeepSeek-V3.2
export MODEL_REPO="deepseek-ai/DeepSeek-V3.2"

# ---------------------------------------------------------------------------
# GPU / Parallelism
# ---------------------------------------------------------------------------
# TP=8 is the stable choice for a single 8xH200 node.
# EP/DP has NCCL instability on single-node H200 per hands-on testing.
export TENSOR_PARALLEL_SIZE="8"
# Reduced from 0.85 to 0.8 to prevent CUDA OOM errors during peak inference.
# This leaves 20% headroom for memory fragmentation and CUDA Graphs overhead.
export GPU_MEMORY_UTILIZATION="0.8"

# Max context length.  P95 input (15840) + P95 output (561) = ~16401 tokens.
# 16384 frees more VRAM for KV cache slots (= more concurrent requests).
# Raise to 32768 if P95 requests are getting truncated.
export MAX_MODEL_LEN="16384"

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
export VLLM_SERVE_PORT="8000"
export VLLM_METRICS_PORT="8100"   # Prometheus scrapes this
export PROMETHEUS_PORT="9090"
export GRAFANA_PORT="3000"
export NODE_EXPORTER_PORT="9100"
export DCGM_EXPORTER_PORT="9400"

# ---------------------------------------------------------------------------
# LMCache
# ---------------------------------------------------------------------------
export LMCACHE_CONFIG_FILE="${DEPLOY_DIR}/lmcache_config.yaml"

# ---------------------------------------------------------------------------
# vLLM runtime flags
# ---------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_ENABLE_V1_MULTIPROCESSING="1"
export PYTHONHASHSEED="0"
export VLLM_NO_USAGE_STATS="1"
# DeepGEMM: required for MQA logits; disable MoE part for stability on H200
export VLLM_USE_DEEP_GEMM=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export OMP_NUM_THREADS=4
# PyTorch memory allocator: reduces fragmentation to prevent CUDA OOM errors
# expandable_segments allows PyTorch to grow memory segments dynamically
# export PYTORCH_ALLOC_CONF="expandable_segmentsFalse"
export NCCL_P2P_DISABLE=0

# Disable InfiniBand if your node has none (prevents broken-pipe errors).
# Comment this out if your cluster has IB.
export NCCL_IB_DISABLE="1"

# ---------------------------------------------------------------------------
mkdir -p "$VLLM_LOGS_DIR" "$LMCACHE_DIR" "$BENCHMARK_OUTPUT_DIR"
