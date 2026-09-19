#!/usr/bin/env python3
"""
disagg_proxy_server.py — PD Disaggregated Prefill Proxy v5.0
============================================================
Multi-instance: N prefillers + M decoders (NixlConnector)

Routing:
  Prefiller selection: round-robin (atomic counter)
  Decoder selection:   least-inflight (pick decoder with fewest active requests)

Why no P→D pinning:
  The NixlConnector uses a pull model. The prefiller holds the KV blocks and
  returns their addresses (remote_block_ids, remote_host, remote_port) in the
  HTTP response. The decoder uses those addresses to pull the KV directly via
  NIXL, regardless of which decoder it is — as long as the N×M NIXL handshake
  mesh was established at startup (which happens automatically when all
  prefillers start up and connect to all decoder NIXL side-channels).

  So: pick any prefiller for prefill, pick any decoder for decode. The
  kv_transfer_params in the prefiller response tells the decoder exactly
  where to pull from.

Usage (2P + 2D):
  python disagg_proxy_server.py \
    --host 0.0.0.0 --port 9000 \
    --prefiller-urls http://localhost:8100,http://localhost:8101 \
    --decoder-urls   http://localhost:8200,http://localhost:8201

Usage (1P + 1D, backward-compatible):
  python disagg_proxy_server.py \
    --host 0.0.0.0 --port 9000 \
    --prefiller-urls http://localhost:8100 \
    --decoder-urls   http://localhost:8200
============================================================
"""

import argparse
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from itertools import cycle
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

# ---------------------------------------------------------------------------
# Instance pool
# ---------------------------------------------------------------------------
PREFILLER_URLS: list[str] = []
DECODER_URLS: list[str] = []

# Round-robin counter for prefillers (atomic via asyncio lock)
_prefiller_idx = 0
_prefiller_lock = asyncio.Lock()

# Per-decoder inflight counter for least-inflight routing
_decoder_inflight: list[int] = []
_decoder_lock = asyncio.Lock()

_http_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError("HTTP client not initialized")
    return _http_client


async def pick_prefiller() -> str:
    """Round-robin across prefillers."""
    global _prefiller_idx
    async with _prefiller_lock:
        url = PREFILLER_URLS[_prefiller_idx % len(PREFILLER_URLS)]
        _prefiller_idx += 1
    return url


async def pick_decoder() -> tuple[int, str]:
    """Least-inflight: pick decoder with fewest active requests.
    Returns (decoder_index, decoder_url)."""
    async with _decoder_lock:
        idx = _decoder_inflight.index(min(_decoder_inflight))
        _decoder_inflight[idx] += 1
    return idx, DECODER_URLS[idx]


async def release_decoder(idx: int):
    async with _decoder_lock:
        _decoder_inflight[idx] = max(0, _decoder_inflight[idx] - 1)


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNTER = Counter(
    "proxy_requests_total", "Total requests received", ["status"]
)
PREFILL_LATENCY = Histogram(
    "proxy_prefill_latency_seconds", "Prefiller HTTP response time",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
TTFT_HISTOGRAM = Histogram(
    "proxy_ttft_seconds", "Time to first token from proxy",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 10.0, 30.0],
)
E2E_LATENCY = Histogram(
    "proxy_e2e_latency_seconds", "End-to-end request latency",
    buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 25.0, 60.0],
)
ACTIVE_REQUESTS = Gauge("proxy_active_requests", "Currently active requests")
TOKENS_GENERATED = Counter("proxy_tokens_generated_total", "Total output tokens")
PREFILLER_REQUESTS = Counter(
    "proxy_prefiller_requests_total", "Requests routed per prefiller", ["prefiller_url"]
)
DECODER_REQUESTS = Counter(
    "proxy_decoder_requests_total", "Requests routed per decoder", ["decoder_url"]
)


# ---------------------------------------------------------------------------
# kv_transfer_params sent to the PREFILLER
# ---------------------------------------------------------------------------
def _prefill_kv_params() -> dict:
    """
    Tells the prefiller: "compute KV, hold the blocks, decoder will pull."
    The block addresses come back in the prefiller's HTTP response JSON.
    """
    return {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=1024,
            max_keepalive_connections=512,
            keepalive_expiry=30,
        ),
        http2=False,
        verify=False,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
    )
    logger.info(f"Proxy v5.0 started — {len(PREFILLER_URLS)} prefiller(s), {len(DECODER_URLS)} decoder(s)")
    for i, url in enumerate(PREFILLER_URLS):
        logger.info(f"  Prefiller {i}: {url}")
    for i, url in enumerate(DECODER_URLS):
        logger.info(f"  Decoder   {i}: {url}")
    yield
    await _http_client.aclose()


app = FastAPI(title="DeepSeek PD Disagg Proxy", version="5.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Core: prefill one request on a chosen prefiller
# ---------------------------------------------------------------------------
async def _do_prefill(body: dict, prefiller_url: str) -> tuple[float, dict]:
    """
    Send request to the chosen prefiller (max_tokens=1, do_remote_decode=True).
    Returns (latency_seconds, kv_transfer_params_for_decoder).

    The kv_transfer_params from the response contain:
      remote_block_ids, remote_engine_id, remote_host, remote_port (NIXL port)
    These tell any decoder exactly where to pull the KV from via NIXL.
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
            f"{prefiller_url}/v1/completions",
            json=prefill_body,
            timeout=120.0,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Prefiller {prefiller_url} error: {e}")
        raise
    finally:
        latency = time.perf_counter() - t0
        PREFILL_LATENCY.observe(latency)
        PREFILLER_REQUESTS.labels(prefiller_url=prefiller_url).inc()

    resp_json = resp.json()
    kv_params = resp_json.get("kv_transfer_params", {})

    if not kv_params:
        logger.warning(
            f"Prefiller {prefiller_url} returned no kv_transfer_params — "
            "decoder cannot pull KV. Check prefiller logs."
        )
    else:
        logger.debug(f"kv_params from {prefiller_url}: {kv_params}")

    return time.perf_counter() - t0, kv_params


# ---------------------------------------------------------------------------
# Core: stream decode from a chosen decoder
# ---------------------------------------------------------------------------
async def _stream_decode(
    body: dict,
    kv_params: dict,
    decoder_url: str,
    decoder_idx: int,
    request_start: float,
) -> AsyncIterator[bytes]:
    decode_body = dict(body)
    decode_body["stream"] = True
    if kv_params:
        decode_body["kv_transfer_params"] = kv_params
    else:
        logger.error(f"No kv_transfer_params — decoder {decoder_url} cannot locate KV blocks")

    DECODER_REQUESTS.labels(decoder_url=decoder_url).inc()
    first_token = True
    token_count = 0

    try:
        async with get_client().stream(
            "POST",
            f"{decoder_url}/v1/completions",
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
    finally:
        await release_decoder(decoder_idx)
        E2E_LATENCY.observe(time.perf_counter() - request_start)
        TOKENS_GENERATED.inc(token_count)
        logger.info(
            f"Done | decoder={decoder_idx} | tokens={token_count} | "
            f"e2e={time.perf_counter() - request_start:.3f}s"
        )


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------
@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    REQUEST_COUNTER.labels(status="received").inc()
    ACTIVE_REQUESTS.inc()
    request_start = time.perf_counter()

    try:
        body = await request.json()

        # --- Phase 1: Prefill ---
        prefiller_url = await pick_prefiller()
        try:
            prefill_latency, kv_params = await _do_prefill(body, prefiller_url)
            logger.info(
                f"Prefill done | prefiller={prefiller_url.split(':')[-1]} | "
                f"latency={prefill_latency:.3f}s"
            )
        except Exception as e:
            REQUEST_COUNTER.labels(status="prefill_error").inc()
            return Response(
                content=json.dumps({"error": f"Prefill failed: {str(e)}"}),
                status_code=502,
                media_type="application/json",
            )

        # --- Phase 2: Decode ---
        decoder_idx, decoder_url = await pick_decoder()
        logger.info(f"Decode routing | decoder={decoder_idx} ({decoder_url.split(':')[-1]})")

        if body.get("stream", False):
            return StreamingResponse(
                _stream_decode(body, kv_params, decoder_url, decoder_idx, request_start),
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
                    f"{decoder_url}/v1/completions",
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
            finally:
                await release_decoder(decoder_idx)

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


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    REQUEST_COUNTER.labels(status="received").inc()
    ACTIVE_REQUESTS.inc()
    request_start = time.perf_counter()

    try:
        body = await request.json()

        # Phase 1: Prefill
        prefiller_url = await pick_prefiller()
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
                f"{prefiller_url}/v1/chat/completions",
                json=prefill_body,
                timeout=120.0,
            )
            resp.raise_for_status()
            kv_params = resp.json().get("kv_transfer_params", {})
            PREFILLER_REQUESTS.labels(prefiller_url=prefiller_url).inc()
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
        decoder_idx, decoder_url = await pick_decoder()
        DECODER_REQUESTS.labels(decoder_url=decoder_url).inc()

        if body.get("stream", False):
            stream_body = dict(body)
            if kv_params:
                stream_body["kv_transfer_params"] = kv_params

            async def _stream_chat():
                first = True
                try:
                    async with get_client().stream(
                        "POST",
                        f"{decoder_url}/v1/chat/completions",
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
                finally:
                    await release_decoder(decoder_idx)

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
            try:
                resp = await get_client().post(
                    f"{decoder_url}/v1/chat/completions",
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
            finally:
                await release_decoder(decoder_idx)

    except Exception as e:
        REQUEST_COUNTER.labels(status="error").inc()
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json",
        )
    finally:
        ACTIVE_REQUESTS.dec()


# ---------------------------------------------------------------------------
# /health — checks all instances
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    results = {"prefillers": {}, "decoders": {}, "connector": "NixlConnector", "version": "5.0.0"}
    all_ok = True

    for url in PREFILLER_URLS:
        try:
            r = await get_client().get(f"{url}/health", timeout=5.0)
            results["prefillers"][url] = "ok" if r.status_code == 200 else f"status_{r.status_code}"
            if r.status_code != 200:
                all_ok = False
        except Exception as e:
            results["prefillers"][url] = f"error: {e}"
            all_ok = False

    for i, url in enumerate(DECODER_URLS):
        try:
            r = await get_client().get(f"{url}/health", timeout=5.0)
            inflight = _decoder_inflight[i] if i < len(_decoder_inflight) else "?"
            results["decoders"][url] = {
                "status": "ok" if r.status_code == 200 else f"status_{r.status_code}",
                "inflight": inflight,
            }
            if r.status_code != 200:
                all_ok = False
        except Exception as e:
            results["decoders"][url] = {"status": f"error: {e}", "inflight": 0}
            all_ok = False

    results["status"] = "ok" if all_ok else "degraded"
    return results


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
async def list_models():
    resp = await get_client().get(f"{DECODER_URLS[0]}/v1/models", timeout=10.0)
    return Response(content=resp.content, media_type="application/json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PD Disagg Proxy v5.0 — Multi-instance")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument(
        "--prefiller-urls",
        default="http://localhost:8100",
        help="Comma-separated prefiller URLs, e.g. http://localhost:8100,http://localhost:8101",
    )
    p.add_argument(
        "--decoder-urls",
        default="http://localhost:8200",
        help="Comma-separated decoder URLs, e.g. http://localhost:8200,http://localhost:8201",
    )
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    PREFILLER_URLS.extend(u.strip().rstrip("/") for u in args.prefiller_urls.split(",") if u.strip())
    DECODER_URLS.extend(u.strip().rstrip("/") for u in args.decoder_urls.split(",") if u.strip())
    _decoder_inflight.extend(0 for _ in DECODER_URLS)

    logger.info("============================================================")
    logger.info(f" PD Disagg Proxy v5.0  |  {args.host}:{args.port}")
    logger.info(f" Prefillers ({len(PREFILLER_URLS)}): {PREFILLER_URLS}")
    logger.info(f" Decoders   ({len(DECODER_URLS)}): {DECODER_URLS}")
    logger.info(" Routing: round-robin prefillers | least-inflight decoders")
    logger.info("============================================================")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level,
        access_log=False,
    )