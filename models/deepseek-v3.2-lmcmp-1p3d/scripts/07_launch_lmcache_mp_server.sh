#!/bin/bash
# ============================================================
# 07_launch_lmcache_mp_server.sh — LMCache ZMQ server (run FIRST on MP node)
# --host defaults to 0.0.0.0 so another node can open TCP to this port.
# Override with LMCACHE_MP_BIND_HOST=localhost for loopback-only tests.
# ============================================================

set -euo pipefail
source "$(dirname "$0")/../.env"
source /workspace/.venv/bin/activate

mkdir -p /workspace/logs/lmcache_mp
LOG_FILE="/workspace/logs/lmcache_mp/lmcache_mp_$(date +%Y%m%d_%H%M%S).log"

MP_PORT="${LMCACHE_MP_PORT:-6000}"
MP_BIND_HOST="${LMCACHE_MP_BIND_HOST:-0.0.0.0}"
L1_GB="${LMCACHE_L1_GB:-200}"
WORKERS="${LMCACHE_MP_MAX_WORKERS:-12}"

echo "============================================================"
echo " LMCache multiprocess server (ZMQ)"
echo " Bind:     ${MP_BIND_HOST}:${MP_PORT} (clients use LMCACHE_MP_CLIENT_HOST + port)"
echo " L1:       ${L1_GB} GB"
echo " Workers:  $WORKERS | Eviction: LRU"
echo " Log:      $LOG_FILE"
echo "============================================================"

python3 -m lmcache.v1.multiprocess.server \
  --host "$MP_BIND_HOST" \
  --l1-size-gb "$L1_GB" \
  --eviction-policy LRU \
  --port "$MP_PORT" \
  --max-workers "$WORKERS" \
  2>&1 | tee "$LOG_FILE"
