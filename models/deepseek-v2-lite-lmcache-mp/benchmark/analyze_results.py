#!/usr/bin/env python3
"""
analyze_results.py  --  inferpd benchmark analysis
Usage:
  python analyze_results.py /workspace/benchmark_results/p50_sweep_*/
  python analyze_results.py /workspace/benchmark_results/p50_sweep_20260315_*/qps5_*.json
"""

import json
import sys
import os
import glob
from pathlib import Path

# ---- Reference numbers (not hard limits, just for comparison) ----
REF = {
    "ttft_p50_s": 2.0,    # current prod P50 TTFT
    "ttft_p95_s": 3.5,    # current prod P95 TTFT
    "e2e_p50_s":  11.0,   # current prod P50 E2E
    "e2e_p95_s":  25.0,   # current prod P95 E2E
}


def load(path):
    with open(path) as f:
        return json.load(f)


def get_ms(data, pct, metric):
    """Return latency in ms. Handles p50_ttft_ms and metric_percentile_50 formats."""
    v = data.get(f"p{pct}_{metric}_ms")
    if v is not None:
        return v
    v = data.get(f"{metric}_percentile_{pct}")
    if v is not None:
        return v * 1000.0 if v < 100 else v
    if pct == 50:
        v = data.get(f"mean_{metric}_ms")
        if v is not None:
            return v
    return None


def get_s(data, pct, metric):
    """Return latency in seconds."""
    v = get_ms(data, pct, metric)
    return v / 1000.0 if v is not None else None


def fmt_ms(val, in_seconds=False):
    """Format latency: s for large values, ms for small."""
    if val is None:
        return "N/A"
    if in_seconds or val >= 1000:
        return f"{val/1000:.3f}s"
    return f"{val:.1f}ms"


def show_latency_block(data, metric, label, ref_p50=None, ref_p95=None, in_seconds=True):
    """Print mean, median, P50/P90/P95/P99 for a latency metric."""
    mean_v = data.get(f"mean_{metric}_ms")
    median_v = data.get(f"median_{metric}_ms")
    p50 = get_ms(data, 50, metric)
    p90 = get_ms(data, 90, metric)
    p95 = get_ms(data, 95, metric)
    p99 = get_ms(data, 99, metric)

    print(f"\n  -- {label} --")
    print(f"  {'mean':<10} {fmt_ms(mean_v, in_seconds):>12}  |  {'median':<8} {fmt_ms(median_v, in_seconds):>12}")
    ref_str = lambda r: f"  (prod: {r}s)" if r else ""
    print(f"  {'P50':<10} {fmt_ms(p50, in_seconds):>12}{ref_str(ref_p50)}  |  {'P90':<8} {fmt_ms(p90, in_seconds):>12}")
    print(f"  {'P95':<10} {fmt_ms(p95, in_seconds):>12}{ref_str(ref_p95)}  |  {'P99':<8} {fmt_ms(p99, in_seconds):>12}")


def analyze_file(path):
    d = load(path)
    name = Path(path).stem
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")

    rate = d.get("request_rate", "?")
    completed = d.get("completed", d.get("successful_requests", "?"))
    failed = d.get("failed", 0)
    achieved = d.get("request_throughput", d.get("actual_request_rate"))
    duration = d.get("duration")
    total_input = d.get("total_input_tokens")
    total_output = d.get("total_output_tokens")
    out_tput = d.get("output_throughput")
    total_tok_tput = d.get("total_token_throughput")
    max_concurrent = d.get("max_concurrent_requests")

    print(f"\n  -- Run summary --")
    print(f"  configured rate         {rate} req/s")
    if achieved is not None:
        print(f"  achieved rate           {achieved:.2f} req/s")
    print(f"  successful requests     {completed}")
    print(f"  failed requests         {failed}")
    if duration is not None:
        print(f"  benchmark duration      {duration:.1f}s")
    if total_input is not None:
        print(f"  total input tokens      {total_input:,}")
    if total_output is not None:
        print(f"  total generated tokens  {total_output:,}")
    if achieved is not None:
        print(f"  request throughput      {achieved:.2f} req/s")
    if out_tput is not None:
        print(f"  output token throughput {out_tput:.1f} tok/s")
    if total_tok_tput is not None:
        print(f"  total token throughput  {total_tok_tput:.1f} tok/s")
    if max_concurrent is not None:
        print(f"  max concurrent requests {max_concurrent:.0f}")
    peak_tput = d.get("max_output_tokens_per_s")
    if peak_tput is not None:
        print(f"  peak output tok/s       {peak_tput:.1f} tok/s")

    show_latency_block(d, "ttft", "Time to First Token (TTFT)", REF["ttft_p50_s"], REF["ttft_p95_s"])
    show_latency_block(d, "tpot", "Time per Output Token (TPOT)", in_seconds=False)
    show_latency_block(d, "itl", "Inter-Token Latency (ITL)", in_seconds=False)
    show_latency_block(d, "e2el", "End-to-End Latency (E2E)", REF["e2e_p50_s"], REF["e2e_p95_s"])

    t50 = get_s(d, 50, "ttft")
    t95 = get_s(d, 95, "ttft")
    if t50 and t50 > 0:
        print(f"\n  -- vs current prod --")
        print(f"  TTFT P50  {REF['ttft_p50_s']:.1f}s prod  vs  {t50:.3f}s here  ({REF['ttft_p50_s']/t50:.2f}x faster)")
        if t95 and t95 > 0:
            print(f"  TTFT P95  {REF['ttft_p95_s']:.1f}s prod  vs  {t95:.3f}s here  ({REF['ttft_p95_s']/t95:.2f}x faster)")


def sweep_table(directory):
    files = sorted(glob.glob(os.path.join(directory, "qps*.json")))
    if not files:
        return

    print(f"\n{'='*72}")
    print(f" QPS SWEEP  --  {directory}")
    print(f"{'='*72}")
    print(f"  {'QPS':>5} | {'TTFT P50':>10} | {'TTFT P95':>10} | {'E2E P50':>10} | {'tok/s':>7}")
    print(f"  {'---':>5}-+-{'--------':>10}-+-{'--------':>10}-+-{'-------':>10}-+-{'-----':>7}")

    for path in files:
        d = load(path)
        rate = d.get("request_rate", "?")
        t50  = get_s(d, 50, "ttft")
        t95  = get_s(d, 95, "ttft")
        e50  = get_s(d, 50, "e2el")
        tput = d.get("output_throughput", 0) or 0
        ns   = lambda v: f"{v:.3f}s" if v is not None else "  N/A  "
        print(f"  {rate:>5} | {ns(t50):>10} | {ns(t95):>10} | {ns(e50):>10} | {tput:>6.0f}t")

    print(f"\n  Reference (current prod):  TTFT P50 {REF['ttft_p50_s']}s  |  TTFT P95 {REF['ttft_p95_s']}s  |  E2E P50 {REF['e2e_p50_s']}s")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <dir_or_file_or_glob>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        sweep_table(target)
        # Only analyze files that look like benchmark results, not warmup discards
        files = sorted(
            f for f in glob.glob(os.path.join(target, "*.json"))
            if "warmup" not in Path(f).name and "infqps" not in Path(f).name
        )
        seen = set()
        for f in files:
            real = os.path.realpath(f)
            if real not in seen:
                seen.add(real)
                analyze_file(f)
    elif os.path.isfile(target):
        analyze_file(target)
    else:
        seen = set()
        for f in sorted(glob.glob(target)):
            real = os.path.realpath(f)
            if real not in seen:
                seen.add(real)
                analyze_file(f)
    print()


if __name__ == "__main__":
    main()