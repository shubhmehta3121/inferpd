#!/bin/bash
# ============================================================
# 04_launch_proxy.sh — Start the Disaggregated Prefill Proxy
#
# NixlConnector: Single HTTP proxy (port 9000)
#   - No telemetry server — prefiller HTTP response IS the KV-done signal
#   - Prefiller and decoder do NIXL handshake at startup (ports 5559/5560)
#
# Start order: Proxy → Decoder → Prefiller
#   (Decoder must bind NIXL 5560 before prefiller connects)
# ============================================================

source "$(dirname "$0")/../.env"
source /workspace/.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/workspace/logs/proxy/proxy_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /workspace/logs/proxy

# ---- Raise file descriptor limit BEFORE starting the proxy ----
ulimit -n 65536
echo "[INFO] File descriptor limit set to $(ulimit -n)"

echo "============================================================"
echo " Launching PROXY SERVER v3.0 (NixlConnector)"
echo " HTTP Proxy:    0.0.0.0:$PROXY_PORT   (client requests)"
echo " Prefiller:     localhost:$PREFILLER_PORT"
echo " Decoder:       localhost:$DECODER_PORT"
echo " Log:           $LOG_FILE"
echo "============================================================"
echo ""
echo "[IMPORTANT] Start order: proxy → decoder → prefiller"
echo ""

python "$PROJECT_DIR/proxy/disagg_proxy_server.py" \
  --host 0.0.0.0 \
  --port "$PROXY_PORT" \
  --prefiller-host localhost \
  --prefiller-port "$PREFILLER_PORT" \
  --decoder-host localhost \
  --decoder-port "$DECODER_PORT" \
  2>&1 | tee "$LOG_FILE"
