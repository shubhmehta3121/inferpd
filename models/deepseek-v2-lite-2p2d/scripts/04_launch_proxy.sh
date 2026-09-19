#!/bin/bash
# ============================================================
# 04_launch_proxy.sh — Start the Proxy (2P + 2D)
#
# Start order: proxy → both decoders → both prefillers
# Run this FIRST before anything else.
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../.env"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/workspace/logs/proxy/proxy_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /workspace/logs/proxy

ulimit -n 65536
echo "[INFO] File descriptor limit: $(ulimit -n)"

# Build comma-separated URL strings from the arrays in .env
PREFILLER_URL_STR=$(printf "http://localhost:%s," "${PREFILLER_PORTS[@]}" | sed 's/,$//')
DECODER_URL_STR=$(printf "http://localhost:%s," "${DECODER_PORTS[@]}" | sed 's/,$//')

echo "============================================================"
echo " Launching PROXY v5.0 (2P + 2D)"
echo " HTTP:        0.0.0.0:$PROXY_PORT"
echo " Prefillers:  $PREFILLER_URL_STR"
echo " Decoders:    $DECODER_URL_STR"
echo " Log:         $LOG_FILE"
echo "============================================================"
echo ""
echo "[IMPORTANT] Start order: proxy → decoders → prefillers"
echo ""

source /workspace/.venv/bin/activate

python "$PROJECT_DIR/proxy/disagg_proxy_server.py" \
  --host 0.0.0.0 \
  --port "$PROXY_PORT" \
  --prefiller-urls "$PREFILLER_URL_STR" \
  --decoder-urls   "$DECODER_URL_STR" \
  2>&1 | tee "$LOG_FILE"