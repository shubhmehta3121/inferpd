#!/bin/bash
# ============================================================
# 07_launch_lmcache_mp_server.sh — LMCache ZMQ server (run FIRST)
# ============================================================

set -euo pipefail
source "$(dirname "$0")/../.env"
source /workspace/.venv/bin/activate

mkdir -p /workspace/logs/lmcache_mp
LOG_FILE="/workspace/logs/lmcache_mp/lmcache_mp_$(date +%Y%m%d_%H%M%S).log"

MP_PORT="${LMCACHE_MP_PORT:-6000}"
L1_GB="${LMCACHE_L1_GB:-80}"
WORKERS="${LMCACHE_MP_MAX_WORKERS:-4}"

echo "============================================================"
echo " LMCache multiprocess server (ZMQ)"
echo " Port:     $MP_PORT"
echo " L1:       ${L1_GB} GB"
echo " Workers:  $WORKERS | Eviction: LRU"
echo " Log:      $LOG_FILE"
echo "============================================================"

python3 -m lmcache.v1.multiprocess.server \
  --l1-size-gb "$L1_GB" \
  --eviction-policy LRU \
  --port "$MP_PORT" \
  --max-workers "$WORKERS" \
  2>&1 | tee "$LOG_FILE"
