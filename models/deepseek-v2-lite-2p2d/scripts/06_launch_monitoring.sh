#!/bin/bash
# ============================================================
# 06_launch_monitoring.sh — Start Prometheus + Grafana
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3000  (admin/admin)
# ============================================================

source "$(dirname "$0")/../.env"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export GF_SECURITY_ALLOW_EMBEDDING="true"
export GF_AUTH_ANONYMOUS_ENABLED="true"
export GRAFANA_ROOT_URL="https://94pa36dxm2wpfa-3000.proxy.runpod.net"
export GF_SERVER_ROOT_URL="${GRAFANA_ROOT_URL:-http://localhost:3000}"
export GF_SECURITY_COOKIE_SAME_SITE="lax"

echo "============================================================"
echo " Launching Monitoring Stack"
echo " Prometheus: http://localhost:$PROMETHEUS_PORT"
echo " Grafana:    http://localhost:$GRAFANA_PORT  (admin/admin)"
echo "============================================================"

# ---- Stop existing (free ports, avoid duplicate processes) ----
if pgrep -x prometheus > /dev/null 2>&1; then
  echo "[INFO] Stopping existing Prometheus..."
  pkill -x prometheus 2>/dev/null || true
  sleep 2
fi
if pgrep -f "grafana-server" > /dev/null 2>&1; then
  echo "[INFO] Stopping existing Grafana..."
  pkill -f "grafana-server" 2>/dev/null || true
  sleep 2
fi

# ---- Start Prometheus ----
echo "[INFO] Starting Prometheus..."
prometheus \
  --config.file="$PROJECT_DIR/configs/prometheus.yml" \
  --storage.tsdb.path=/var/lib/prometheus \
  --web.listen-address="0.0.0.0:$PROMETHEUS_PORT" \
  --log.level=warn \
  > /workspace/logs/prometheus.log 2>&1 &
PROM_PID=$!
echo "  Prometheus PID: $PROM_PID (log: /workspace/logs/prometheus.log)"

# ---- Start Grafana ----
echo "[INFO] Starting Grafana..."
mkdir -p /workspace/logs
GRAFANA_LOG="/workspace/logs/grafana.log"

# Try systemd/service first; on RunPod/Docker often not available
if command -v service &> /dev/null; then
  service grafana-server start > /dev/null 2>&1
fi
sleep 3

# If Grafana not running (e.g. no systemd on RunPod), start binary directly
if ! curl -sf "http://localhost:$GRAFANA_PORT/api/health" > /dev/null 2>&1; then
  if command -v grafana-server &> /dev/null; then
    export GF_SERVER_HTTP_PORT="$GRAFANA_PORT"
    nohup grafana-server --homepath=/usr/share/grafana \
      > "$GRAFANA_LOG" 2>&1 &
    echo "  Grafana started via binary (log: $GRAFANA_LOG)"
    sleep 5
  else
    echo "  ⚠️  grafana-server not found — run scripts/00_setup.sh first"
  fi
fi

# Wait for Grafana to be ready, then configure Prometheus datasource
echo "[INFO] Waiting for Grafana (up to 2 min)..."
GRAFANA_READY=0
for i in $(seq 1 24); do
  if curl -sf "http://localhost:$GRAFANA_PORT/api/health" > /dev/null 2>&1; then
    GRAFANA_READY=1
    break
  fi
  [ $((i % 4)) -eq 0 ] && echo "  ... still waiting ($((i*5))s)"
  sleep 5
done

if [ "$GRAFANA_READY" -eq 1 ]; then
  if curl -sf -X POST \
    -H "Content-Type: application/json" \
    -u admin:admin \
    -d '{
      "name": "Prometheus",
      "type": "prometheus",
      "url": "http://localhost:9090",
      "access": "proxy",
      "isDefault": true
    }' \
    "http://localhost:$GRAFANA_PORT/api/datasources" > /dev/null 2>&1; then
    echo "  ✅ Grafana datasource configured"
  else
    # Datasource may already exist (e.g. 409)
    echo "  ✅ Grafana ready (datasource may already exist)"
  fi
else
  echo "  ⚠️  Grafana did not become ready — check: cat $GRAFANA_LOG"
fi

echo ""
echo "============================================================"
echo " Access Grafana at: http://localhost:$GRAFANA_PORT"
echo " Username: admin | Password: admin"
echo " Import dashboard JSON from: monitoring/grafana_dashboard.json"
echo " In RunPod: forward port $GRAFANA_PORT to access from browser"
echo " To stop:   pkill -x prometheus; pkill -f grafana-server"
echo "============================================================"
