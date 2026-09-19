#!/usr/bin/env bash
# Environment variables for the v2 disaggregated prefill deployment.
# This sources the common vars from the parent deploy/ folder and adds
# multi-node specifics.
#
# Source this (not the parent 00-env.sh) before running any v2 script:
#   source /workspace/deepseek-v3.2/v2/env.sh

# Load common vars (model, paths, vLLM flags, etc.)
source "$(dirname "${BASH_SOURCE[0]}")/../00-env.sh"

# ---------------------------------------------------------------------------
# Multi-node IPs  ← UPDATE THESE before deploying
# ---------------------------------------------------------------------------
# Run `hostname -I | awk '{print $1}'` on each machine to find the IP.
export PREFILLER_IP="REPLACE_WITH_PREFILLER_NODE_IP"   # Node A (~$30/hr)
export DECODER_IP="REPLACE_WITH_DECODER_NODE_IP"       # Node B (~$30/hr)

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
# HTTP proxy port — clients send all inference requests here
export PROXY_PORT="9100"

# NIXL side-channel base port.  With TP=8, vLLM uses ports 5600-5607
# (base + gpu_rank).  Open these in your firewall on both nodes.
export NIXL_SIDE_CHANNEL_PORT="5600"

# ---------------------------------------------------------------------------
# LMCache configs (node-specific)
# ---------------------------------------------------------------------------
export LMCACHE_PREFILLER_CONFIG="$(dirname "${BASH_SOURCE[0]}")/lmcache-prefiller.yaml"
export LMCACHE_DECODER_CONFIG="$(dirname "${BASH_SOURCE[0]}")/lmcache-decoder.yaml"

# ---------------------------------------------------------------------------
# UCX transport for NIXL
# ---------------------------------------------------------------------------
export UCX_TLS="cuda_ipc,cuda_copy,tcp"
export UCX_NET_DEVICES="all"
