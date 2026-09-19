#!/bin/bash
# ============================================================
# 04_launch_proxy.sh — LMCache disagg_prefill_mp proxy + telemetry
# START ORDER: 07 → this → 03 (decoder) → 02 (prefiller)
# ============================================================

set -euo pipefail
source "$(dirname "$0")/../.env"
source /workspace/.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/workspace/logs/proxy/proxy_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /workspace/logs/proxy

ulimit -n 65536
echo "[INFO] File descriptor limit set to $(ulimit -n)"

NUM_DECODERS="${NUM_DECODERS:-1}"

echo "============================================================"
echo " PROXY (HTTP + telemetry thread)"
echo " HTTP:       0.0.0.0:$PROXY_PORT"
echo " Telemetry:  0.0.0.0:${TELEMETRY_PORT:-5768}  (/api/v1/telemetry)"
echo " Prefiller:  localhost:$PREFILLER_PORT"
echo " Decoders:   $NUM_DECODERS × base port $DECODER_PORT (round-robin)"
echo " Log:        $LOG_FILE"
echo "============================================================"

# Max wait for LMCache request_store_finished after prefill (see doc/disagg-proxy-kv-store-wait-timeout.md)
KV_WAIT_SEC="${DISAGG_PROXY_KV_STORE_WAIT_TIMEOUT_SEC:-120}"

python "$PROJECT_DIR/proxy/disagg_proxy_server.py" \
  --host 0.0.0.0 \
  --port "$PROXY_PORT" \
  --telemetry-port "${TELEMETRY_PORT:-5768}" \
  --prefiller-host localhost \
  --prefiller-port "$PREFILLER_PORT" \
  --decoder-host localhost \
  --decoder-port "$DECODER_PORT" \
  --num-decoders "$NUM_DECODERS" \
  --kv-store-wait-timeout-sec "$KV_WAIT_SEC" \
  2>&1 | tee "$LOG_FILE"
