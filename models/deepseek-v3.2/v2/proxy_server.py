#!/usr/bin/env python3
"""
Disaggregated prefill proxy server.

Routes each request through two phases:
  1. PREFILL  → sends request to prefiller with max_tokens=1.
               Gets back kv_transfer_params (tells decoder where KV lives).
  2. DECODE   → sends original request + kv_transfer_params to decoder.
               Streams the full response back to the client.

The proxy uses round-robin across multiple prefiller/decoder instances.
For the initial 1P1D (1 prefiller, 1 decoder) deployment:
  --prefiller-hosts <PREFILLER_IP>  --prefiller-ports 8000
  --decoder-hosts   <DECODER_IP>    --decoder-ports   8000

To scale out later (e.g. 2P2D):
  --prefiller-hosts <IP_A> <IP_B>  --prefiller-ports 8000 8000
  --decoder-hosts   <IP_C> <IP_D>  --decoder-ports   8000 8000

Usage:
  source 00-env.sh
  source $PYTHON_ENV_PATH/bin/activate
  python proxy_server.py \
    --port $PROXY_PORT \
    --prefiller-hosts $PREFILLER_IP --prefiller-ports 8000 \
    --decoder-hosts   $DECODER_IP   --decoder-ports   8000

Client sends all requests to http://<proxy-host>:$PROXY_PORT/v1/...
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("proxy")

# ---------------------------------------------------------------------------
# Global args (set in main)
# ---------------------------------------------------------------------------
global_args: argparse.Namespace


# ---------------------------------------------------------------------------
# Lifespan: create connection pools
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_clients = []
    app.state.decode_clients = []

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )
        logger.info("Registered prefiller %d → %s", i, url)

    for i, (host, port) in enumerate(global_args.decoder_instances):
        url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )
        logger.info("Registered decoder  %d → %s", i, url)

    app.state.prefill_rr = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_rr  = itertools.cycle(range(len(app.state.decode_clients)))

    logger.info(
        "Proxy ready — %d prefiller(s), %d decoder(s)",
        len(app.state.prefill_clients),
        len(app.state.decode_clients),
    )

    yield

    for info in app.state.prefill_clients + app.state.decode_clients:
        await info["client"].aclose()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Round-robin client selection
# ---------------------------------------------------------------------------
def _next_client(app: FastAPI, role: str) -> dict:
    if role == "prefill":
        return app.state.prefill_clients[next(app.state.prefill_rr)]
    return app.state.decode_clients[next(app.state.decode_rr)]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
_AUTH_HEADER = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"}


async def _prefill_request(client_info: dict, endpoint: str, body: dict, req_id: str) -> dict:
    """Send to prefiller with max_tokens=1, return the kv_transfer_params."""
    payload = dict(body)
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload.pop("max_completion_tokens", None)
    payload.pop("stream_options", None)

    headers = {**_AUTH_HEADER, "X-Request-Id": req_id}
    resp = await client_info["client"].post(endpoint, json=payload, headers=headers)
    resp.raise_for_status()
    await resp.aread()   # consume body so the connection is released
    return resp.json()


async def _stream_decode(client_info: dict, endpoint: str, body: dict, req_id: str):
    """Stream decode response from decoder."""
    headers = {**_AUTH_HEADER, "X-Request-Id": req_id}
    async with client_info["client"].stream("POST", endpoint, json=body, headers=headers) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            yield chunk


# ---------------------------------------------------------------------------
# Core handler (shared by completions + chat/completions)
# ---------------------------------------------------------------------------
async def _handle(endpoint: str, request: Request) -> StreamingResponse:
    body = await request.json()
    req_id = str(uuid.uuid4())

    prefill_info = _next_client(request.app, "prefill")
    decode_info  = _next_client(request.app, "decode")

    logger.debug("req=%s  prefiller=%s  decoder=%s", req_id, prefill_info["id"], decode_info["id"])

    # Phase 1: prefill
    prefill_resp = await _prefill_request(prefill_info, endpoint, body, req_id)
    kv_params = prefill_resp.get("kv_transfer_params")
    if kv_params:
        body["kv_transfer_params"] = kv_params

    # Phase 2: decode + stream
    return StreamingResponse(
        _stream_decode(decode_info, endpoint, body, req_id),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle("/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle("/chat/completions", request)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Disaggregated prefill proxy")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9100)

    p.add_argument("--prefiller-hosts", nargs="+", default=["localhost"])
    p.add_argument("--prefiller-ports", nargs="+", type=int, default=[8000])
    p.add_argument("--decoder-hosts",   nargs="+", default=["localhost"])
    p.add_argument("--decoder-ports",   nargs="+", type=int, default=[8000])

    args = p.parse_args()

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        p.error("--prefiller-hosts and --prefiller-ports must have the same length")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        p.error("--decoder-hosts and --decoder-ports must have the same length")

    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances   = list(zip(args.decoder_hosts,   args.decoder_ports))
    return args


if __name__ == "__main__":
    import uvicorn

    global_args = parse_args()
    uvicorn.run(app, host=global_args.host, port=global_args.port, log_level="info")
