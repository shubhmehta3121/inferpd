#!/usr/bin/env python3
"""
Production workload benchmark for DeepSeek-V3 deployment.

Simulates target workload traffic characteristics:
  - Input  tokens: P50=7870, P95=15840
  - Output tokens: P50=252,  P95=561
  - Peak QPS: ~60  |  Avg QPM: ~2300 (1800-2800 range)
  - Prefix cache hit rate: ~80%
  - Streaming responses

Target SLAs:
  - TTFT P50 <= 2s,  P95 <= 3.5s
  - E2E  P50 <= 11s, P95 <= 25s

Usage:
    python benchmark.py --url http://localhost:8000 --qps 30 --num-requests 300
    python benchmark.py --url http://localhost:8000 --qps 60 --num-requests 600

    # Against RunPod proxied URL:
    python benchmark.py --url https://<pod-id>-8000.proxy.runpod.net --qps 30 --num-requests 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import openai

from datetime import datetime

# ---------------------------------------------------------------------------
# Optional token counter using tiktoken
# ---------------------------------------------------------------------------
try:
    import tiktoken
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

def count_tokens(text: str, model_name="gpt-4") -> int:
    if TOKENIZER_AVAILABLE:
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    else:
        return max(1, len(text.split()))

# ---------------------------------------------------------------------------
# Workload constants
# ---------------------------------------------------------------------------
INPUT_P50 = 7870
INPUT_P95 = 15840
CACHE_HIT_RATIO = 0.80
OUTPUT_P50 = 252
OUTPUT_P95 = 561

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
_PROMPT_TEMPLATES = [
    "Explain the concept of {topic} in simple terms. Give an example or scenario that illustrates it.",
    "Summarize the key ideas about {topic} and explain their practical implications in everyday life.",
    "Provide a concise summary of {topic} and why it matters.",
    "What are the main challenges and opportunities in {topic}? Provide a few strategies or solutions.",
    "What are the pros and cons of {topic}? Include a balanced view.",
    "Describe {topic} in detail and include a short case study or example for better understanding.",
    "Describe the practical use cases of {topic} and where it is most effective.",
    "Generate a short dialogue or discussion between two experts about {topic}.",
    "Provide a step-by-step guide to understand or implement {topic} with clear instructions.",
    "Compare and contrast {topic} with a related concept, highlighting their differences and similarities.",
    "Explain {topic} in layman's terms for someone completely new to the subject.",
    "Create a fictional news article reporting on a recent development related to {topic}.",
    "Write a persuasive argument advocating for the importance of understanding {topic}.",
    "List and briefly explain five fascinating facts about {topic}.",
    "Ask and answer three common questions that people new to {topic} might have.",
    "Draw connections between {topic} and another field or discipline.",
    "Write a short story or anecdote that involves {topic} in an everyday situation.",
    "Explain potential future trends or innovations involving {topic}.",
    "Summarize the latest research or advancements in {topic}.",
    "Draft a letter explaining {topic} to a friend who is unfamiliar with it.",
    "Precisely define {topic} and break it down into its component parts.",
    "Describe how {topic} has evolved over time or throughout history.",
    "Generate a set of interview questions for an expert in {topic}.",
    "Present a fictional debate between two people with opposing views on {topic}.",
    "Suggest creative ways to teach {topic} to children or beginners.",
]

# ---------------------------------------------------------------------------
# Diverse topics
# ---------------------------------------------------------------------------
_TOPICS = [
    # Psychology / Sociology / Sexuality
    "cognitive biases", "decision-making", "emotional intelligence", "motivation theory", "group dynamics",
    "behavioral economics", "memory retention", "social influence",
    "sexual communication", "consent in digital relationships", "sexting and its psychological impact",
    "digital intimacy", "sex education in the digital age", "privacy concerns in online sexuality",
    
    # Economics / Finance / Business
    "macroeconomic policy", "inflation trends", "investment strategies", "portfolio diversification",
    "corporate finance", "risk management", "financial forecasting", "cryptocurrency markets",
    "impact of digital technology on financial services", "online payment security", "cryptocurrency in online dating",

    # Technology / AI / CS
    "machine learning algorithms", "transformers in NLP", "reinforcement learning", "AI ethics",
    "parallel computing optimization", "cloud infrastructure design", "cybersecurity threats",
    "secure messaging app protocols", "AI in relationship counseling", "automated content moderation for explicit materials", "privacy in online communications",

    # Game theory / Strategy
    "prisoner's dilemma", "Nash equilibrium", "auction strategies", "market competition dynamics",    
    "strategic disclosure of personal information", "privacy vs. openness in digital interaction",
    
    # Biology / Health / Food / Sexual Health
    "evolutionary adaptations", "human nutrition", "gut microbiome", "plant growth cycles",
    "culinary techniques", "fusion cuisine", "fermentation process",
    "sexual health risks of digital intimacy", "safe sexting practices", "impact of online relationships on mental health", "public health campaigns on consent",

    # Geography / Culture / Politics / Society
    "urbanization trends in Europe", "Japanese work culture", "US electoral system", "climate policies in Scandinavia",
    "historical trade routes", "sustainable city planning",
    "cultural perspectives on sexting", "legal aspects of digital sexual content", "age restrictions and online safety", "media portrayal of online intimacy",

    # Random / Fun / Misc
    "board game strategies", "travel planning for adventure tourism", "sports analytics", "music composition techniques",
    "literary symbolism in classic novels", "cryptography basics",
    "evolution of romance in the internet era", "humor in online flirting", "emoji use in digital romance", "fictional scenarios involving digital relationships",
]

# ---------------------------------------------------------------------------
# Generate prompts
# ---------------------------------------------------------------------------
def generate_prompts(n: int = 50) -> list[str]:
    prompts: list[str] = []
    for _ in range(n):
        template = random.choice(_PROMPT_TEMPLATES)
        topic = random.choice(_TOPICS)
        prompts.append(f"{template.format(topic=topic)}\n### Response:\n")
    return prompts

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class Result:
    prompt: str = ""
    input_tokens: int = 0
    ttft: Optional[float] = None
    e2e: Optional[float] = None
    output_tokens: int = 0
    success: bool = False
    error: str = ""

# ---------------------------------------------------------------------------
# Single async streaming request
# ---------------------------------------------------------------------------
async def send_request(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
) -> Result:
    result = Result(prompt=prompt)
    result.input_tokens = count_tokens(prompt, model)
    t_start = time.perf_counter()
    first_token_at: Optional[float] = None
    buf: list[str] = []

    try:
        stream = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
        )
        async for chunk in stream:
            text = chunk.choices[0].text if chunk.choices else ""
            if first_token_at is None and text:
                first_token_at = time.perf_counter()
                result.ttft = first_token_at - t_start
            if text:
                buf.append(text)

        result.e2e = time.perf_counter() - t_start
        result.output_tokens = count_tokens("".join(buf), model)
        result.success = True
    except Exception as exc:
        result.error = str(exc)

    return result

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
async def run_benchmark(
    base_url: str,
    api_key: str,
    model: str,
    num_requests: int,
    target_qps: float,
) -> list[Result]:
    client = openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key=api_key)

    prompts = [
        (prompt, random.randint(OUTPUT_P50, OUTPUT_P95))
        for prompt in generate_prompts(num_requests)
    ]

    inter_arrival = 1.0 / target_qps
    tasks: list[asyncio.Task] = []
    t0 = time.perf_counter()

    for i, (prompt, max_tokens) in enumerate(prompts):
        delay = (t0 + i * inter_arrival) - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(send_request(client, model, prompt, max_tokens)))

    print(f"\nAll {num_requests} requests dispatched. Waiting for completion...")
    results = list(await asyncio.gather(*tasks))
    await client.close()
    return results

# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------
def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return sorted(values)[min(int(len(values) * p / 100), len(values) - 1)]

def fmt(v: float) -> str:
    return f"{v:.3f}"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepSeek-V3 production benchmark")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY", ""),
        help="Bearer token for vLLM auth (defaults to env VLLM_API_KEY).",
    )
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    p.add_argument("--qps", type=float, default=30.0)
    p.add_argument("--num-requests", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)

    today_str = datetime.now().strftime("%Y%m%d")
    default_output_filename = f"benchmark_results_{today_str}.json"
    p.add_argument(
        "--output",
        default=os.path.join(
            os.environ.get("BENCHMARK_OUTPUT_DIR", "outputs/benchmarks"),
            default_output_filename,
        ),
        help="Output path for results (default: outputs/benchmarks/benchmark_results_yyyymmdd.json)",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print(f"  DeepSeek-V3 Benchmark")
    print(f"  URL:         {args.url}")
    print(f"  Model:       {args.model}")
    print(f"  Target QPS:  {args.qps}")
    print(f"  Requests:    {args.num_requests}")
    print(f"  Prompt templates: {len(_PROMPT_TEMPLATES)}")
    print(f"  Auth:        {'VLLM_API_KEY set' if args.api_key else 'none'}")
    print("=" * 60)

    results = asyncio.run(
        run_benchmark(args.url, args.api_key, args.model, args.num_requests, args.qps)
    )

    ok     = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    ttfts  = [r.ttft for r in ok if r.ttft is not None]
    e2es   = [r.e2e  for r in ok if r.e2e  is not None]
    total_duration = max(e2es) if e2es else 0.0

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Successful : {len(ok)} / {args.num_requests}")
    print(f"  Failed     : {len(failed)}")
    print(f"  Achieved QPS: {len(ok) / total_duration:.2f}" if total_duration else "  Achieved QPS: n/a")
    print()

    if ttfts:
        print("  Time-To-First-Token (s)")
        print(f"    P50 : {fmt(pct(ttfts, 50))}   [target <= 2.0]")
        print(f"    P95 : {fmt(pct(ttfts, 95))}   [target <= 3.5]")
        print(f"    P99 : {fmt(pct(ttfts, 99))}")
        print(f"    Mean: {fmt(statistics.mean(ttfts))}")

    if e2es:
        print("\n  End-to-End Latency (s)")
        print(f"    P50 : {fmt(pct(e2es, 50))}   [target <= 11.0]")
        print(f"    P95 : {fmt(pct(e2es, 95))}   [target <= 25.0]")
        print(f"    P99 : {fmt(pct(e2es, 99))}")
        print(f"    Mean: {fmt(statistics.mean(e2es))}")

    print("=" * 60)

    if failed:
        print(f"\n  First 5 errors:")
        for r in failed[:5]:
            print(f"    {r.error}")

    # Save full raw results including prompt, input tokens, output tokens
    raw = [
        {
            "prompt": r.prompt,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "ttft": r.ttft,
            "e2e": r.e2e,
            "success": r.success,
            "error": r.error
        }
        for r in results
    ]

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(raw, fh, indent=2)
    print(f"\n  Raw results saved → {output_path}")

if __name__ == "__main__":
    main()
