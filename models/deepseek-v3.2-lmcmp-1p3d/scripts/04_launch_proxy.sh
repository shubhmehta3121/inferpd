#!/bin/bash
# ============================================================
# 04_launch_proxy.sh — LMCache disagg_prefill_mp proxy + telemetry
# Two-node: set DISAGG_PROXY_PREFILLER_HOST + DISAGG_PROXY_DECODER_HOST_CSV
#           (and optional DISAGG_PROXY_DECODER_PORT_CSV) in .env on Node D.
# START ORDER: 07 → 04 → 03 (all decoders) → 02
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

NUM_DECODERS="${NUM_DECODERS:-3}"
PREF_HOST="${DISAGG_PROXY_PREFILLER_HOST:-localhost}"
DEC_HOST_CSV="${DISAGG_PROXY_DECODER_HOST_CSV:-}"
DEC_PORT_CSV="${DISAGG_PROXY_DECODER_PORT_CSV:-}"
BASE_DEC_PORT="${DECODER_PORT:-8200}"
KV_WAIT_SEC="${DISAGG_PROXY_KV_STORE_WAIT_TIMEOUT_SEC:-120}"

if [ -n "$DEC_HOST_CSV" ]; then
  if [ -z "$DEC_PORT_CSV" ]; then
    DEC_PORT_CSV="${BASE_DEC_PORT}"
    for ((i = 1; i < NUM_DECODERS; i++)); do
      DEC_PORT_CSV+=",$((BASE_DEC_PORT + i))"
    done
  fi
  IFS=',' read -ra HARR <<< "$DEC_HOST_CSV"
  IFS=',' read -ra PARR <<< "$DEC_PORT_CSV"
  if [ "${#HARR[@]}" -ne "$NUM_DECODERS" ]; then
    echo "[ERROR] DISAGG_PROXY_DECODER_HOST_CSV must list exactly NUM_DECODERS=$NUM_DECODERS hosts (got ${#HARR[@]})."
    exit 1
  fi
  if [ "${#PARR[@]}" -ne "$NUM_DECODERS" ]; then
    echo "[ERROR] DISAGG_PROXY_DECODER_PORT_CSV must list exactly NUM_DECODERS=$NUM_DECODERS ports (got ${#PARR[@]})."
    exit 1
  fi
  DECODER_HOST_ARG="$DEC_HOST_CSV"
  DECODER_PORT_ARG="$DEC_PORT_CSV"
else
  DECODER_HOST_ARG="localhost"
  DECODER_PORT_ARG="${BASE_DEC_PORT}"
fi

echo "============================================================"
echo " PROXY (HTTP + telemetry thread)"
echo " HTTP:       0.0.0.0:$PROXY_PORT"
echo " Telemetry:  0.0.0.0:${TELEMETRY_PORT:-5768}  (/api/v1/telemetry)"
echo " Prefiller:  ${PREF_HOST}:$PREFILLER_PORT"
if [ -n "$DEC_HOST_CSV" ]; then
  echo " Decoders:   ${DECODER_HOST_ARG} ports ${DECODER_PORT_ARG}"
else
  echo " Decoders:   localhost × $NUM_DECODERS × base port $BASE_DEC_PORT"
fi
echo " Log:        $LOG_FILE"
echo "============================================================"

python "$PROJECT_DIR/proxy/disagg_proxy_server.py" \
  --host 0.0.0.0 \
  --port "$PROXY_PORT" \
  --telemetry-port "${TELEMETRY_PORT:-5768}" \
  --prefiller-host "$PREF_HOST" \
  --prefiller-port "$PREFILLER_PORT" \
  --decoder-host "$DECODER_HOST_ARG" \
  --decoder-port "$DECODER_PORT_ARG" \
  --num-decoders "$NUM_DECODERS" \
  --kv-store-wait-timeout-sec "$KV_WAIT_SEC" \
  2>&1 | tee "$LOG_FILE"
