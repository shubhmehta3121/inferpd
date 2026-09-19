#!/bin/bash
# ============================================================
# launch_all.sh — Launch all 5 processes in tmux (2P + 2D)
#
# Creates a tmux session "pd" with 5 windows:
#   0: proxy
#   1: decoder-0  (GPU 1, port 8200, NIXL 5560)
#   2: decoder-1  (GPU 3, port 8201, NIXL 5562)
#   3: prefiller-0 (GPU 0, port 8100, NIXL 5559)
#   4: prefiller-1 (GPU 2, port 8101, NIXL 5561)
#
# START ORDER (enforced by sleep delays):
#   proxy starts immediately
#   decoders start 5s later  (need proxy up first is optional, but fine)
#   prefillers start 60s after decoders (decoders need to bind NIXL ports first)
#
# Usage:
#   bash launch_all.sh          # start everything
#   tmux attach -t pd           # attach to session
#   Ctrl-B + 0..4               # switch between windows
#   bash launch_all.sh kill     # kill the session
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="pd"

if [ "${1:-}" = "kill" ]; then
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "Session '$SESSION' killed" || echo "No session to kill"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[ERROR] tmux session '$SESSION' already exists."
  echo "  Attach:  tmux attach -t $SESSION"
  echo "  Kill:    bash launch_all.sh kill"
  exit 1
fi

source "$SCRIPT_DIR/../.env"

echo "============================================================"
echo " Launching 2P + 2D PD stack in tmux session '$SESSION'"
echo "============================================================"

# Window 0: proxy (start immediately)
tmux new-session -d -s "$SESSION" -n "proxy" \
  "bash $SCRIPT_DIR/04_launch_proxy.sh; read"

# Window 1: decoder-0 (start after 5s)
tmux new-window -t "$SESSION" -n "decoder-0" \
  "sleep 5 && bash $SCRIPT_DIR/03_launch_decoder.sh 0; read"

# Window 2: decoder-1 (start after 5s)
tmux new-window -t "$SESSION" -n "decoder-1" \
  "sleep 5 && bash $SCRIPT_DIR/03_launch_decoder.sh 1; read"

# Window 3: prefiller-0 (start after 75s — decoders need ~60s to bind NIXL)
tmux new-window -t "$SESSION" -n "prefiller-0" \
  "sleep 75 && bash $SCRIPT_DIR/02_launch_prefiller.sh 0; read"

# Window 4: prefiller-1 (start after 75s)
tmux new-window -t "$SESSION" -n "prefiller-1" \
  "sleep 75 && bash $SCRIPT_DIR/02_launch_prefiller.sh 1; read"

echo ""
echo " Processes starting in background:"
echo "   proxy      → now        (window 0)"
echo "   decoder-0  → 5s         (window 1)"
echo "   decoder-1  → 5s         (window 2)"
echo "   prefiller-0→ 75s        (window 3)"
echo "   prefiller-1→ 75s        (window 4)"
echo ""
echo " Attach:    tmux attach -t $SESSION"
echo " Switch:    Ctrl-B + 0-4"
echo " Kill all:  bash launch_all.sh kill"
echo ""
echo " After ~3 minutes, run:"
echo "   bash scripts/05_health_check.sh"