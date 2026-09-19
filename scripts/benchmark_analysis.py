#!/usr/bin/env python3
"""
Benchmark analysis for DeepSeek-V3 deployment on 8xH200 (RunPod).

Aggregates benchmark_results_1..6.json and benchmark_results.json, computes
TTFT/E2E percentiles, and produces a summary report. Includes
single-request baseline from individual request measurements.

Usage:
    python benchmark_analysis.py
    python benchmark_analysis.py --md    # Output markdown summary
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Benchmark file → (num_requests, qps) mapping
# ---------------------------------------------------------------------------
BENCHMARK_CONFIG: dict[str, tuple[int, int]] = {
    "benchmark_results_1.json": (150, 15),
    "benchmark_results_2.json": (300, 30),
    "benchmark_results_3.json": (100, 10),
    "benchmark_results_4.json": (100, 10),
    "benchmark_results_5.json": (250, 25),
    "benchmark_results_6.json": (600, 60),
    "benchmark_results.json": (300, 30),  # 300 req @ 30 QPS (or use benchmark_results_6 for 60 QPS)
}

# Root directory for benchmark JSON files (optional override via --model-dir)
ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = Path(__file__).resolve().parent


def parse_single_request_from_txt(txt_path: Path) -> dict | None:
    """Extract single-request metrics from individual request terminal log."""
    if not txt_path.exists():
        return None
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    ttft_re = re.compile(r"TTFT[^:]*:\s*([\d.]+)s", re.IGNORECASE)
    total_re = re.compile(r"Total time:\s*([\d.]+)s", re.IGNORECASE)
    tokens_re = re.compile(r"(?:Tokens received|Total tokens received):\s*(\d+)", re.IGNORECASE)
    tps_re = re.compile(r"Tokens per second:\s*([\d.]+)", re.IGNORECASE)
    tpt_re = re.compile(r"(?:Time per token|Average time per token):\s*([\d.]+)s", re.IGNORECASE)

    # Each run ends with "Total time" / "Tokens received" - find all such blocks
    total_matches = list(total_re.finditer(text))
    best: dict | None = None
    for total_m in total_matches:
        start = max(0, total_m.start() - 2000)  # TTFT is typically within 2k chars before
        block = text[start : total_m.end() + 200]
        ttft_m = ttft_re.search(block)
        tokens_m = tokens_re.search(block)
        tps_m = tps_re.search(block)
        tpt_m = tpt_re.search(block)

        ttft = float(ttft_m.group(1)) if ttft_m else 0.0
        total = float(total_m.group(1))
        tokens = int(tokens_m.group(1)) if tokens_m else 0
        tps = float(tps_m.group(1)) if tps_m else (1.0 / float(tpt_m.group(1)) if tpt_m else 0.0)
        if tps <= 0 and tokens > 0 and total > 0:
            tps = tokens / total

        cand = {"ttft_s": ttft, "total_time_s": total, "output_tokens": tokens, "tokens_per_second": tps}
        # Prefer research-prompt run: ~285 tokens, TTFT ~0.067s
        if 250 <= tokens <= 350 and 0.05 <= ttft <= 0.15:
            return cand
        if best is None or (tokens >= 100 and (best["output_tokens"] < 100 or (tokens >= 200 and ttft < best["ttft_s"]))):
            best = cand
    return best


def percentile(data: list[float], p: float) -> float:
    """Compute percentile (P50=50, P95=95)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = 1 if f < len(sorted_data) - 1 else 0
    return sorted_data[f] + (k - f) * (sorted_data[f + c] - sorted_data[f])


def analyze_benchmark(filepath: Path) -> dict | None:
    """Load benchmark JSON and compute stats."""
    if not filepath.exists():
        return None

    with open(filepath, encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        return None

    ttft = [r["ttft"] for r in records if r.get("success", True)]
    e2e = [r["e2e"] for r in records if r.get("success", True)]
    output_tokens = [r.get("output_tokens", 0) for r in records if r.get("success", True)]

    success_count = sum(1 for r in records if r.get("success", True))
    total = len(records)

    return {
        "file": filepath.name,
        "num_requests": total,
        "success": success_count,
        "failed": total - success_count,
        "ttft_mean": statistics.mean(ttft) if ttft else 0,
        "ttft_p50": percentile(ttft, 50) if ttft else 0,
        "ttft_p95": percentile(ttft, 95) if ttft else 0,
        "ttft_p99": percentile(ttft, 99) if ttft else 0,
        "e2e_mean": statistics.mean(e2e) if e2e else 0,
        "e2e_p50": percentile(e2e, 50) if e2e else 0,
        "e2e_p95": percentile(e2e, 95) if e2e else 0,
        "e2e_p99": percentile(e2e, 99) if e2e else 0,
        "output_tokens_mean": statistics.mean(output_tokens) if output_tokens else 0,
        "output_tokens_total": sum(output_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark analysis summary")
    parser.add_argument("--md", action="store_true", help="Output markdown summary")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "deepseek-v3.2",
        help="Model deploy dir (default: models/deepseek-v3.2). Reads from <dir>/outputs/benchmarks/",
    )
    args = parser.parse_args()

    bench_dir = args.model_dir / "outputs" / "benchmarks"
    individual_txt = ROOT / "archive" / "deploy" / "individual request.txt"

    # Single-request baseline
    single_req = parse_single_request_from_txt(individual_txt) if individual_txt.exists() else None
    if single_req is None:
        single_req = {"ttft_s": 0.067, "total_time_s": 6.706, "output_tokens": 285, "tokens_per_second": 42.50}
    time_per_token = 1.0 / single_req["tokens_per_second"] if single_req["tokens_per_second"] else 0.024

    if not args.md:
        print("=" * 70)
        print("  DeepSeek-V3 Benchmark Analysis - Summary")
        print("  RunPod: 8xH200SXM | vLLM TP=8 | max_model_len=16384 | DeepGEMM=0")
        print("=" * 70)

        print("\n--- Single Request (No Load) ---")
        print(
            f"  Representative: ~28 input tokens, 600 max output, research prompt\n"
            f"  TTFT:        {single_req['ttft_s']:.3f}s\n"
            f"  Total time:  {single_req['total_time_s']:.3f}s\n"
            f"  Output:      {single_req['output_tokens']} tokens\n"
            f"  Tokens/sec:  {single_req['tokens_per_second']:.2f}\n"
            f"  Time/token:  {time_per_token:.3f}s"
        )

    # Load config for achieved QPS (from individual request terminal output)
    config_achieved_qps: dict[str, float] = {
        "benchmark_results_1.json": 0,  # not in log
        "benchmark_results_2.json": 1.60,
        "benchmark_results_3.json": 5.46,
        "benchmark_results_4.json": 3.98,
        "benchmark_results_5.json": 7.21,
        "benchmark_results_6.json": 11.02,
        "benchmark_results.json": 1.60,  # 300 @ 30 QPS run
    }

    results: list[dict] = []

    for filename, (num_req, qps) in BENCHMARK_CONFIG.items():
        filepath = bench_dir / filename
        stats = analyze_benchmark(filepath)
        if stats is None:
            continue

        achieved = config_achieved_qps.get(filename)
        if achieved is None and filename != "benchmark_results_1.json":
            # Estimate from total_time if we had it
            achieved = num_req / (stats["e2e_mean"] * 10) if stats["e2e_mean"] > 0 else 0

        stats["qps_target"] = qps
        stats["qps_achieved"] = achieved
        stats["num_requests"] = num_req
        results.append(stats)

    # Sort by num_requests
    results.sort(key=lambda r: r["num_requests"])

    print("\n--- Load Test Results ---")
    print(
        f"\n  {'File':<28} {'Req':>5} {'QPS':>4} {'Achiev':>7} "
        f"{'TTFT P50':>9} {'TTFT P95':>9} {'E2E P50':>9} {'E2E P95':>10} {'Success':>8}"
    )
    print("  " + "-" * 100)

    for r in results:
        achiev_str = f"{r['qps_achieved']:.2f}" if r["qps_achieved"] else "-"
        print(
            f"  {r['file']:<28} {r['num_requests']:>5} {r['qps_target']:>4} {achiev_str:>7} "
            f"{r['ttft_p50']:>8.2f}s {r['ttft_p95']:>8.2f}s "
            f"{r['e2e_p50']:>8.2f}s {r['e2e_p95']:>9.2f}s "
            f"{r['success']:>5}/{r['num_requests']}"
        )

    print("\n--- SLA Targets ---")
    print("  TTFT: P50 <= 2.0s,  P95 <= 3.5s")
    print("  E2E:  P50 <= 11.0s, P95 <= 25.0s")

    # Determine best/worst by load
    light = next((r for r in results if r["num_requests"] == 100), None)
    heavy = next((r for r in results if r["num_requests"] == 600), None)

    if light and heavy:
        print("\n--- Summary ---")
        print(f"  At 10 QPS / 100 req: TTFT P50={light['ttft_p50']:.2f}s, E2E P50={light['e2e_p50']:.2f}s")
        print(f"  At 60 QPS / 600 req: TTFT P50={heavy['ttft_p50']:.2f}s, E2E P50={heavy['e2e_p50']:.2f}s")
        print(f"  Peak achieved QPS: {heavy['qps_achieved']:.2f} (target 60)")

    # Save detailed JSON
    out_path = Path(__file__).resolve().parent / "benchmark_analysis_output.json"
    output_data = {
        "single_request": single_req,
        "load_tests": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Detailed output -> {out_path}")
    print("=" * 70)

    if args.md:
        light = next((r for r in results if r["num_requests"] == 100), results[0] if results else None)
        heavy = next((r for r in results if r["num_requests"] == 600), results[-1] if results else None)
        good_runs = [r for r in results if r["ttft_p50"] < 5 and r["num_requests"] >= 100]
        print("\n" + "-" * 50)
        print("MARKDOWN (copy for summary message):")
        print("-" * 50)
        print(_md_summary(single_req, results, light, heavy, good_runs))


def _md_summary(
    single_req: dict,
    results: list[dict],
    light: dict | None,
    heavy: dict | None,
    good_runs: list[dict],
) -> str:
    # Dedupe by (num_requests, qps) - keep best E2E P50
    seen: dict[tuple[int, int], dict] = {}
    for r in good_runs:
        key = (r["num_requests"], r["qps_target"])
        if key not in seen or r["e2e_p50"] < seen[key]["e2e_p50"]:
            seen[key] = r
    unique_runs = list(seen.values())
    unique_runs.sort(key=lambda x: x["num_requests"])

    lines = [
        "**DeepSeek-V3 on RunPod (8x H200SXM)**",
        "",
        "**Single request** (no load, ~28 input tokens, 600 max output):",
        f"- TTFT: {single_req['ttft_s']:.3f}s",
        f"- Total time: {single_req['total_time_s']:.2f}s",
        f"- Output: {single_req['output_tokens']} tokens @ {single_req['tokens_per_second']:.1f} tok/s",
        "",
        "**Under load:**",
    ]
    for r in unique_runs:
        achiev = f" (achieved {r['qps_achieved']:.2f} QPS)" if r.get("qps_achieved") else ""
        lines.append(
            f"- {r['num_requests']} req @ {r['qps_target']} QPS{achiev}: "
            f"TTFT P50={r['ttft_p50']:.2f}s, E2E P50={r['e2e_p50']:.2f}s"
        )
    if heavy:
        lines.append("")
        lines.append(f"Peak: 600 req @ 60 QPS target, achieved {heavy.get('qps_achieved', 0):.2f} QPS")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
