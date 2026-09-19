#!/usr/bin/env bash
# Starts the HTTP disaggregated-prefill proxy.
#
# Start this AFTER both prefiller (03-start-prefiller.sh) and decoder
# (04-start-decoder.sh) are fully up and responding to /health.
#
# The proxy can run on either node or a cheap separate VM.
# If running on Node A or B, just source 00-env.sh and run this.
#
# Usage:
#   source 00-env.sh
#   source $PYTHON_ENV_PATH/bin/activate
#   bash 05-start-proxy.sh 2>&1 | tee $VLLM_LOGS_DIR/proxy.log
#
# Client sends all inference requests to:
#   http://<THIS_MACHINE_IP>:$PROXY_PORT/v1/chat/completions
#   http://<THIS_MACHINE_IP>:$PROXY_PORT/v1/completions
#
# Health check:
#   curl http://localhost:$PROXY_PORT/health
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/00-env.sh"
source "$PYTHON_ENV_PATH/bin/activate"

# Verify prefiller and decoder are up before starting proxy
echo "Checking prefiller at ${PREFILLER_IP}:${VLLM_SERVE_PORT}..."
timeout 30 bash -c "until curl -sf http://${PREFILLER_IP}:${VLLM_SERVE_PORT}/health > /dev/null 2>&1; do sleep 2; done" \
    && echo "  ✅  Prefiller healthy" \
    || { echo "  ❌  Prefiller not responding — start 03-start-prefiller.sh first"; exit 1; }

echo "Checking decoder at ${DECODER_IP}:${VLLM_SERVE_PORT}..."
timeout 30 bash -c "until curl -sf http://${DECODER_IP}:${VLLM_SERVE_PORT}/health > /dev/null 2>&1; do sleep 2; done" \
    && echo "  ✅  Decoder healthy" \
    || { echo "  ❌  Decoder not responding — start 04-start-decoder.sh first"; exit 1; }

echo ""
echo "Starting proxy on port ${PROXY_PORT}"
echo "  Prefiller: ${PREFILLER_IP}:${VLLM_SERVE_PORT}"
echo "  Decoder:   ${DECODER_IP}:${VLLM_SERVE_PORT}"
echo "  Client endpoint: http://$(hostname -I | awk '{print $1}'):${PROXY_PORT}/v1/"

exec python "$(dirname "${BASH_SOURCE[0]}")/proxy_server.py" \
    --host 0.0.0.0 \
    --port "${PROXY_PORT}" \
    --prefiller-hosts "${PREFILLER_IP}" \
    --prefiller-ports "${VLLM_SERVE_PORT}" \
    --decoder-hosts   "${DECODER_IP}" \
    --decoder-ports   "${VLLM_SERVE_PORT}"

# ── Scaling out to 2P2D ──────────────────────────────────────────────────────
# When you add a second prefiller (Node C) and second decoder (Node D):
#
# export PREFILLER_IP_2="<NODE_C_IP>"
# export DECODER_IP_2="<NODE_D_IP>"
#
# exec python proxy_server.py \
#   --prefiller-hosts "${PREFILLER_IP}" "${PREFILLER_IP_2}" \
#   --prefiller-ports "${VLLM_SERVE_PORT}" "${VLLM_SERVE_PORT}" \
#   --decoder-hosts   "${DECODER_IP}"   "${DECODER_IP_2}" \
#   --decoder-ports   "${VLLM_SERVE_PORT}" "${VLLM_SERVE_PORT}"
# ─────────────────────────────────────────────────────────────────────────────
