#!/usr/bin/env python3
"""
disagg_proxy_server.py — PD Disaggregated Prefill Proxy v4.0
============================================================
NixlConnector — CORRECT kv_transfer_params flow.

THE KEY FIX (v4.0 vs v3.0):
  v3.0 sent the WRONG params to each instance:
    prefiller ← {do_remote_prefill: True}   ← WRONG (means "fetch KV from someone")
    decoder   ← {do_remote_decode: True}    ← WRONG (means "I already prefilled")

  v4.0 sends the CORRECT params:
    prefiller ← {do_remote_decode: True, remote_block_ids: None, ...}
                 means "prefill this, hold the KV blocks, decoder will pull"
    prefiller response JSON contains kv_transfer_params with:
                 {do_remote_prefill: True, remote_block_ids: [...],
                  remote_engine_id: ..., remote_host: ..., remote_port: ...}
    decoder   ← those EXACT params extracted from prefiller response
                 means "go fetch KV from prefiller at these specific blocks"

Reference: vLLM disagg_encoder_proxy.py (process_prefill_stage / maybe_prefill)
  github.com/vllm-project/vllm / examples/online_serving/disaggregated_encoder/

Port: 9000 (HTTP proxy for all client requests)

Usage:
  python disagg_proxy_server.py \
    --host 0.0.0.0 --port 9000 \
    --prefiller-host localhost --prefiller-port 8100 \
    --decoder-host localhost --decoder-port 8200
============================================================
"""

import argparse
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import (
    Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, generate_latest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("disagg_proxy")

PREFILLER_URL: str = ""
DECODER_URL: str = ""

_http_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError("HTTP client not initialized — lifespan not started")
    return _http_client


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNTER = Counter(
    "proxy_requests_total", "Total requests received by proxy", ["status"]
)
PREFILL_LATENCY = Histogram(
    "proxy_prefill_latency_seconds", "Time spent waiting for prefiller HTTP response",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
TTFT_HISTOGRAM = Histogram(
    "proxy_ttft_seconds", "Time to first token (end-to-end from proxy)",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 10.0, 30.0],
)
E2E_LATENCY = Histogram(
    "proxy_e2e_latency_seconds", "End-to-end request latency",
    buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 25.0, 60.0],
)
ACTIVE_REQUESTS = Gauge("proxy_active_requests", "Currently active requests")
TOKENS_GENERATED = Counter("proxy_tokens_generated_total", "Total output tokens generated")


# ---------------------------------------------------------------------------
# THE CORRECT kv_transfer_params for NixlConnector
# ---------------------------------------------------------------------------

def _prefill_kv_params() -> dict:
    """
    Sent to the PREFILLER.

    do_remote_decode=True means: "you are the prefiller — compute KV,
    hold the blocks in your GPU memory, the decoder will come and READ them."

    The other fields are None/False here because the prefiller doesn't
    need to know where the decoder is — the decoder initiates the pull
    using the block addresses that come back in the prefiller's response.
    """
    return {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }


# _decode_kv_params() does NOT exist as a static function anymore.
# The decode params come from the prefiller's HTTP response JSON.
# See _do_prefill() — it returns (latency, kv_transfer_params_for_decoder).


# ---------------------------------------------------------------------------
# Lifespan: single shared client
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=512,
            max_keepalive_connections=256,
            keepalive_expiry=30,
        ),
        http2=False,
        verify=False,
    )
    logger.info("Shared httpx.AsyncClient created (max_connections=512)")
    yield
    await _http_client.aclose()
    logger.info("Shared httpx.AsyncClient closed")


app = FastAPI(title="DeepSeek PD Disagg Proxy", version="4.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
async def _do_prefill(body: dict) -> tuple[float, dict]:
    """
    Send request to prefiller with max_tokens=1 and do_remote_decode=True.

    The prefiller computes KV for the full prompt, holds the blocks,
    and returns in the response JSON a `kv_transfer_params` field containing:
      {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": [[...]], ← actual GPU block IDs on prefiller
        "remote_engine_id": "...",   ← prefiller's engine UUID
        "remote_request_id": "...",  ← request ID on the prefiller
        "remote_host": "localhost",  ← prefiller's NIXL side channel host
        "remote_port": 5559,         ← prefiller's NIXL side channel port
        "tp_size": 1,
      }

    These params are then forwarded verbatim to the decoder so it knows
    exactly which GPU blocks on the prefiller to pull via NIXL.

    Returns: (prefill_latency_seconds, kv_transfer_params_for_decoder)
    """
    prefill_body = dict(body)
    prefill_body["max_tokens"] = 1
    prefill_body["stream"] = False
    prefill_body["n"] = 1
    prefill_body.pop("stream_options", None)
    prefill_body["kv_transfer_params"] = _prefill_kv_params()

    t0 = time.perf_counter()
    try:
        resp = await get_client().post(
            f"{PREFILLER_URL}/v1/completions",
            json=prefill_body,
            timeout=120.0,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Prefiller HTTP error: {e}")
        raise
    finally:
        latency = time.perf_counter() - t0
        PREFILL_LATENCY.observe(latency)

    # Extract the kv_transfer_params the decoder needs from the prefiller response.
    # The prefiller's request_finished() callback populates these with the actual
    # GPU block addresses, engine ID, and NIXL side channel port.
    resp_json = resp.json()
    kv_params_for_decoder = resp_json.get("kv_transfer_params", {})

    if not kv_params_for_decoder:
        logger.warning(
            "Prefiller response had no kv_transfer_params — "
            "decoder will not know where to fetch KV. "
            "Check that kv_load_failure_policy=fail is set and decoder logs."
        )

    logger.debug(f"Prefiller kv_transfer_params for decoder: {kv_params_for_decoder}")
    return time.perf_counter() - t0, kv_params_for_decoder


async def _stream_decode(
    body: dict,
    kv_params_for_decoder: dict,
    request_start: float,
) -> AsyncIterator[bytes]:
    """
    Stream decode from decoder. The decoder receives the kv_transfer_params
    extracted from the prefiller's response — it uses these to pull KV from
    the prefiller's GPU memory via NIXL, then generates tokens.
    """
    decode_body = dict(body)
    decode_body["stream"] = True

    # Pass the params from the prefiller response to the decoder.
    # If empty (prefiller didn't return them), decoder will recompute (bad).
    if kv_params_for_decoder:
        decode_body["kv_transfer_params"] = kv_params_for_decoder
    else:
        logger.error(
            "No kv_transfer_params from prefiller — "
            "decoder cannot locate prefiller KV blocks. Request will likely fail."
        )

    first_token = True
    token_count = 0
    t0 = time.perf_counter()

    async with get_client().stream(
        "POST",
        f"{DECODER_URL}/v1/completions",
        json=decode_body,
        timeout=300.0,
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if chunk:
                if first_token:
                    TTFT_HISTOGRAM.observe(time.perf_counter() - request_start)
                    first_token = False
                for line in chunk.decode("utf-8", errors="ignore").split("\n"):
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            for c in data.get("choices", []):
                                if c.get("text"):
                                    token_count += 1
                        except Exception:
                            pass
                yield chunk

    E2E_LATENCY.observe(time.perf_counter() - t0)
    TOKENS_GENERATED.inc(token_count)
    logger.info(
        f"Done | tokens={token_count} | e2e={time.perf_counter() - t0:.3f}s"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    REQUEST_COUNTER.labels(status="received").inc()
    ACTIVE_REQUESTS.inc()
    request_start = time.perf_counter()

    try:
        body = await request.json()

        # Phase 1: Prefill — returns kv_transfer_params the decoder needs
        try:
            prefill_latency, kv_params = await _do_prefill(body)
            PREFILL_LATENCY.observe(prefill_latency)
            logger.info(f"Prefill done in {prefill_latency:.3f}s")
        except Exception as e:
            REQUEST_COUNTER.labels(status="prefill_error").inc()
            logger.error(f"Prefill failed: {e}")
            return Response(
                content=json.dumps({"error": f"Prefill failed: {str(e)}"}),
                status_code=502,
                media_type="application/json",
            )

        # Phase 2: Decode — decoder uses kv_params to pull KV from prefiller
        if body.get("stream", False):
            return StreamingResponse(
                _stream_decode(body, kv_params, request_start),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no"},
            )
        else:
            decode_body = dict(body)
            decode_body["stream"] = False
            if kv_params:
                decode_body["kv_transfer_params"] = kv_params
            try:
                resp = await get_client().post(
                    f"{DECODER_URL}/v1/completions",
                    json=decode_body,
                    timeout=300.0,
                )
                resp.raise_for_status()
                E2E_LATENCY.observe(time.perf_counter() - request_start)
                REQUEST_COUNTER.labels(status="success").inc()
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type="application/json",
                )
            except Exception as e:
                REQUEST_COUNTER.labels(status="decode_error").inc()
                return Response(
                    content=json.dumps({"error": f"Decode failed: {str(e)}"}),
                    status_code=502,
                    media_type="application/json",
                )

    except Exception as e:
        REQUEST_COUNTER.labels(status="error").inc()
        logger.error(f"Proxy error: {e}")
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json",
        )
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    REQUEST_COUNTER.labels(status="received").inc()
    ACTIVE_REQUESTS.inc()
    request_start = time.perf_counter()

    try:
        body = await request.json()

        # Phase 1: Prefill
        prefill_body = dict(body)
        prefill_body["max_tokens"] = 1
        prefill_body.pop("max_completion_tokens", None)
        prefill_body["stream"] = False
        prefill_body.pop("stream_options", None)
        prefill_body["kv_transfer_params"] = _prefill_kv_params()

        t0 = time.perf_counter()
        kv_params = {}
        try:
            resp = await get_client().post(
                f"{PREFILLER_URL}/v1/chat/completions",
                json=prefill_body,
                timeout=120.0,
            )
            resp.raise_for_status()
            resp_json = resp.json()
            kv_params = resp_json.get("kv_transfer_params", {})
        except Exception as e:
            REQUEST_COUNTER.labels(status="prefill_error").inc()
            return Response(
                content=json.dumps({"error": f"Prefill failed: {str(e)}"}),
                status_code=502,
                media_type="application/json",
            )
        finally:
            PREFILL_LATENCY.observe(time.perf_counter() - t0)

        # Phase 2: Decode
        if body.get("stream", False):
            stream_body = dict(body)
            if kv_params:
                stream_body["kv_transfer_params"] = kv_params

            async def _stream_chat():
                first = True
                async with get_client().stream(
                    "POST",
                    f"{DECODER_URL}/v1/chat/completions",
                    json=stream_body,
                    timeout=300.0,
                ) as r:
                    r.raise_for_status()
                    async for chunk in r.aiter_bytes():
                        if chunk:
                            if first:
                                TTFT_HISTOGRAM.observe(time.perf_counter() - request_start)
                                first = False
                            yield chunk
                E2E_LATENCY.observe(time.perf_counter() - request_start)

            return StreamingResponse(
                _stream_chat(),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no"},
            )
        else:
            decode_body = dict(body)
            decode_body["stream"] = False
            if kv_params:
                decode_body["kv_transfer_params"] = kv_params
            resp = await get_client().post(
                f"{DECODER_URL}/v1/chat/completions",
                json=decode_body,
                timeout=300.0,
            )
            REQUEST_COUNTER.labels(status="success").inc()
            E2E_LATENCY.observe(time.perf_counter() - request_start)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )

    except Exception as e:
        REQUEST_COUNTER.labels(status="error").inc()
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json",
        )
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/health")
async def health():
    prefiller_ok = decoder_ok = False
    try:
        r = await get_client().get(f"{PREFILLER_URL}/health", timeout=5.0)
        prefiller_ok = r.status_code == 200
    except Exception:
        pass
    try:
        r = await get_client().get(f"{DECODER_URL}/health", timeout=5.0)
        decoder_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "status": "ok" if (prefiller_ok and decoder_ok) else "degraded",
        "prefiller": "ok" if prefiller_ok else "unreachable",
        "decoder": "ok" if decoder_ok else "unreachable",
        "connector": "NixlConnector",
        "proxy_version": "4.0.0",
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
async def list_models():
    resp = await get_client().get(f"{DECODER_URL}/v1/models", timeout=10.0)
    return Response(content=resp.content, media_type="application/json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PD Disagg Proxy v4.0 (NixlConnector)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--prefiller-host", default="localhost")
    p.add_argument("--prefiller-port", type=int, default=8100)
    p.add_argument("--decoder-host", default="localhost")
    p.add_argument("--decoder-port", type=int, default=8200)
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    PREFILLER_URL = f"http://{args.prefiller_host}:{args.prefiller_port}"
    DECODER_URL = f"http://{args.decoder_host}:{args.decoder_port}"

    logger.info("============================================================")
    logger.info(" DeepSeek PD Disagg Proxy v4.0 (NixlConnector)")
    logger.info(f" Listening:   {args.host}:{args.port}")
    logger.info(f" Prefiller:   {PREFILLER_URL}")
    logger.info(f" Decoder:     {DECODER_URL}")
    logger.info(" Flow: prefiller←{do_remote_decode:True} → response→kv_params → decoder←kv_params")
    logger.info("============================================================")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level,
        access_log=False,
    )