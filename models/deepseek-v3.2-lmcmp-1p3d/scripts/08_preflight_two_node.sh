#!/bin/bash
# ============================================================
# 08_preflight_two_node.sh — TCP checks before starting vLLM (saves $)
# Requires NODE_P_PRIVATE_IP and NODE_D_PRIVATE_IP in .env.
# Run from either node after both VMs are up.
# ============================================================

set -euo pipefail
source "$(dirname "$0")/../.env.b"

P="${NODE_P_PRIVATE_IP:-}"
D="${NODE_D_PRIVATE_IP:-}"
MP_PORT="${LMCACHE_MP_PORT:-6000}"
TEL_PORT="${TELEMETRY_PORT:-5768}"

if [ -z "$P" ] || [ -z "$D" ]; then
  echo "[ERROR] Set NODE_P_PRIVATE_IP and NODE_D_PRIVATE_IP in .env for this script."
  exit 1
fi

echo "============================================================"
echo " Preflight (two-node)  P=$P  D=$D"
echo "============================================================"

try_nc() {
  local label=$1
  local host=$2
  local port=$3
  if command -v nc >/dev/null 2>&1; then
    if nc -zvw3 "$host" "$port" 2>/dev/null; then
      echo "  ✅ $label  $host:$port"
    else
      echo "  ❌ $label  $host:$port (blocked or closed)"
    fi
  else
    echo "  ⚠️  $label  (install nc for port checks)"
  fi
}

echo ""
echo "[ From Node P toward Node D — LMCache + telemetry ]"
echo "  (Run this block ON Node P, or rely on symmetric routing)"
try_nc "P → D ZMQ/MP" "$D" "$MP_PORT"
try_nc "P → D telemetry" "$D" "$TEL_PORT"

echo ""
echo "[ From Node D toward Node P — prefiller + decoder0 ]"
try_nc "D → P prefiller" "$P" "${PREFILLER_PORT:-8100}"
try_nc "D → P decoder0" "$P" "${DECODER_PORT:-8200}"

echo ""
echo " Fix cloud firewall / security groups if any check fails."
echo "============================================================"
