#!/usr/bin/env python3
"""Aggregate vLLM bench sweeps into site/data/results.json for the Results page."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "data" / "results.json"

# Chronological experiment order (phases 4–9 are measured P/D stacks).
STACKS = [
    {
        "phase": 4,
        "id": "v2-lite-nixl-1p1d",
        "title": "V2-Lite · NIXL · 1P1D",
        "topology": "1P1D",
        "model": "DeepSeek V2-Lite",
        "gpu_model": "A100",
        "hardware": "2× A100 80GB (1 prefill + 1 decode)",
        "connector": "NixlConnector",
        "status": "success",
        "summary": "First stable disaggregated P/D reference on cheap 2-GPU hardware.",
        "repo_path": "models/deepseek-v2-lite",
        "report": "models/deepseek-v2-lite/benchmark_results/benchmark_report.html",
        "highlight": False,
    },
    {
        "phase": 5,
        "id": "v2-lite-2p2d",
        "title": "V2-Lite · NIXL · 2P2D",
        "topology": "2P2D",
        "model": "DeepSeek V2-Lite",
        "gpu_model": "A100",
        "hardware": "4× A100 (2 prefill + 2 decode)",
        "connector": "NixlConnector (per-request KV)",
        "status": "success",
        "summary": "Scale-out prefiller/decoders. Throughput up; still no shared cross-request L1.",
        "repo_path": "models/deepseek-v2-lite-2p2d",
        "report": "models/deepseek-v2-lite-2p2d/benchmark_results/benchmark_report.html",
        "highlight": False,
    },
    {
        "phase": 6,
        "id": "v2-lite-lmcache-mp",
        "title": "V2-Lite · LMCache MP · 1P1D",
        "topology": "1P1D",
        "model": "DeepSeek V2-Lite",
        "gpu_model": "A100",
        "hardware": "2× A100 80GB (1 prefill + 1 decode)",
        "connector": "LMCacheMPConnector + central ZMQ L1",
        "status": "success",
        "summary": "Shared ZMQ L1 validated. Prefix-heavy traffic starts to win on TTFT.",
        "repo_path": "models/deepseek-v2-lite-lmcache-mp",
        "report": "models/deepseek-v2-lite-lmcache-mp/benchmark_results/benchmark_report.html",
        "highlight": False,
    },
    {
        "phase": 7,
        "id": "v2-lite-lmcmp-1p2d",
        "title": "V2-Lite · LMCache MP · 1P2D",
        "topology": "1P2D",
        "model": "DeepSeek V2-Lite",
        "gpu_model": "A100",
        "hardware": "3× A100 80GB (1 prefill + 2 decode)",
        "connector": "LMCacheMPConnector + central ZMQ L1",
        "status": "success",
        "summary": "Best match for production-like prefix-heavy chat at moderate QPS.",
        "repo_path": "models/deepseek-v2-lite-lmcmp-1p2d",
        "report": "models/deepseek-v2-lite-lmcmp-1p2d/benchmark_results/benchmark_report.html",
        "highlight": True,
    },
    {
        "phase": 8,
        "id": "v3.2-lmcmp-1p1d",
        "title": "V3.2 NVFP4 · LMCache MP · 1P1D",
        "topology": "1P1D",
        "model": "DeepSeek V3.2 NVFP4",
        "gpu_model": "B200",
        "hardware": "8× B200",
        "topology_note": "1 prefill engine + 1 decode engine (TP=4 each)",
        "connector": "LMCacheMPConnector + central ZMQ L1",
        "status": "success",
        "summary": "Scaled the shared-L1 architecture to the production-class model on Blackwell.",
        "repo_path": "models/deepseek-v3.2-lmcmp-1p1d",
        "report": "models/deepseek-v3.2-lmcmp-1p1d/benchmark_results/benchmark_report.html",
        "highlight": True,
    },
    {
        "phase": 9,
        "id": "v3.2-lmcmp-1p3d",
        "title": "V3.2 NVFP4 · LMCache MP · 1P3D",
        "topology": "1P3D",
        "model": "DeepSeek V3.2 NVFP4",
        "gpu_model": "B200",
        "hardware": "2 rentals · 8× B200 each (16 GPUs)",
        "topology_note": "Cross-node: 1 prefill cluster + 3 decode engines (planned)",
        "connector": "LMCacheMPConnector + central ZMQ L1",
        "status": "blocked",
        "summary": "Runbook complete. Vast.ai confirmed no private network between separate multi-GPU rentals.",
        "repo_path": "models/deepseek-v3.2-lmcmp-1p3d",
        "report": None,
        "highlight": False,
    },
]


def ms2s(v: float | None) -> float | None:
    if v is None:
        return None
    v = float(v)
    return round(v / 1000 if v > 100 else v, 4)


def extract_row(data: dict) -> dict:
    rate = data.get("request_rate")
    achieved = data.get("request_throughput") or data.get("actual_request_rate")
    p50_ttft = data.get("p50_ttft_ms")
    if p50_ttft is None and data.get("ttft_percentile_50") is not None:
        p50_ttft = float(data["ttft_percentile_50"]) * 1000
    p95_ttft = data.get("p95_ttft_ms")
    if p95_ttft is None and data.get("ttft_percentile_95") is not None:
        p95_ttft = float(data["ttft_percentile_95"]) * 1000
    p50_e2e = data.get("p50_e2el_ms") or data.get("p50_e2e_ms")
    p95_e2e = data.get("p95_e2el_ms") or data.get("p95_e2e_ms")
    out_tput = data.get("output_throughput")
    return {
        "qps_target": rate,
        "qps_achieved": round(float(achieved), 2) if achieved is not None else None,
        "ttft_p50": ms2s(p50_ttft),
        "ttft_p95": ms2s(p95_ttft),
        "e2e_p50": ms2s(p50_e2e),
        "e2e_p95": ms2s(p95_e2e),
        "output_tok_s": round(float(out_tput), 1) if out_tput is not None else None,
        "success": data.get("completed") or data.get("successful_requests"),
        "failed": data.get("failed", 0),
    }


def load_sweep(model_rel: str, patterns: tuple[str, ...] = ("p50_sweep_*", "p50_2p2d_*")) -> tuple[str | None, list[dict]]:
    base = ROOT / model_rel / "benchmark_results"
    if not base.exists():
        return None, []
    for pattern in patterns:
        dirs = sorted(base.glob(pattern), reverse=True)
        if not dirs:
            continue
        sweep_dir = dirs[0]
        rows: list[dict] = []
        for path in sorted(sweep_dir.glob("qps*.json")):
            if "warmup" in path.name:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                rows.append(extract_row(payload))
            except (json.JSONDecodeError, OSError):
                continue
        if rows:
            rows.sort(key=lambda r: r["qps_target"] or 0)
            return sweep_dir.name, rows
    return None, []


def main() -> None:
    payload: dict = {
        "targets": {
            "traffic": {
                "peak_qps": 60,
                "input_p50_tokens": 7870,
                "input_p95_tokens": 15840,
                "output_p50_tokens": 252,
                "output_p95_tokens": 561,
                "prefix_cache_hit_pct": 80,
                "streaming": True,
            },
            "latency": {
                "ttft_p50_s": 2.0,
                "ttft_p95_s": 3.5,
                "e2e_p50_s": 11.0,
                "e2e_p95_s": 25.0,
                "note": "Production ballparks from contract workload; TTFT weighted higher than E2E.",
            },
            "benchmark_profile": {
                "name": "P50 load profile",
                "shared_prefix_tokens": 6000,
                "avg_input_tokens": 7870,
                "prompts_per_step": 600,
                "qps_sweep": "1, 2, 5, 10, 20, 30, 50, 60 req/s",
            },
        },
        "stacks": [],
        "baseline_v32_single_node": None,
    }

    for stack in STACKS:
        sweep_id, points = load_sweep(stack["repo_path"])
        payload["stacks"].append({**stack, "sweep_id": sweep_id, "points": points})

    v32_path = ROOT / "scripts" / "benchmark_analysis_output.json"
    if v32_path.exists():
        v32 = json.loads(v32_path.read_text(encoding="utf-8"))
        payload["baseline_v32_single_node"] = {
            "title": "V3.2 BF16 · single node · 8× H200 TP=8",
            "note": "Early baseline before disagg + shared L1. Long-prompt runs show prefill-bound TTFT.",
            "single_request": v32.get("single_request"),
            "load_tests": v32.get("load_tests", []),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
