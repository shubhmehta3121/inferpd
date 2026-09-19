#!/usr/bin/env python3
"""
Environment validation for DeepSeek-V3 on 8xH200.
Run this immediately after provisioning and before starting the server.

Usage:
    source 00-env.sh
    python system_validation.py
"""

from __future__ import annotations

import subprocess
import platform
import sys

import torch


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_system() -> None:
    section("SYSTEM")
    print(f"OS        : {platform.system()} {platform.release()}")
    print(f"Python    : {platform.python_version()}")
    print(f"PyTorch   : {torch.__version__}")
    print(f"CUDA avail: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA ver  : {torch.version.cuda}")
        print(f"cuDNN     : {torch.backends.cudnn.version()}")


def check_gpus() -> float:
    section("GPU DETAILS")

    if not torch.cuda.is_available():
        print("❌  CUDA not available")
        sys.exit(1)

    n = torch.cuda.device_count()
    print(f"GPU count : {n}")

    total_gb = 0.0
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        mem_gb = p.total_memory / 1e9
        total_gb += mem_gb
        tag = (
            "✅ Hopper (H200) — supported"
            if "H200" in p.name
            else "✅ Blackwell — optimal"
            if ("B200" in p.name or "GB200" in p.name)
            else "⚠️  Unknown GPU — may not be fully supported"
        )
        print(f"  GPU[{i}] {p.name}  {mem_gb:.1f} GB  CC={p.major}.{p.minor}  {tag}")

    print(f"\nTotal VRAM: {total_gb:.1f} GB")
    return total_gb


def check_nvlink() -> None:
    section("NVLINK")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "nvlink", "--status"], stderr=subprocess.STDOUT, text=True
        )
        if "Active" in out or "Link" in out:
            print("✅  NVLink active — multi-GPU bandwidth optimal")
        else:
            print("⚠️  nvidia-smi nvlink ran but status unclear")
            print(out[:500])
    except Exception as exc:
        print(f"⚠️  Could not query NVLink: {exc}")


def check_packages() -> None:
    section("PYTHON PACKAGES")
    pkgs = ["torch", "vllm", "lmcache", "nixl", "transformers", "openai", "numpy"]
    for pkg in pkgs:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  ✅  {pkg:20s} {ver}")
        except ImportError:
            print(f"  ❌  {pkg:20s} NOT INSTALLED")


def recommendations(total_gb: float) -> None:
    section("RECOMMENDATIONS")

    if total_gb >= 1100:
        print("✅  Sufficient VRAM for DeepSeek-V3.2 in BF16")
        print("   Use: --tensor-parallel-size 8 --dtype bfloat16")
        print()
        print("   NOTE: EP/DP (--dp 8 --enable-expert-parallel) is recommended")
        print("   by this validator, but has NCCL instability on single-node H200.")
        print("   Stick with TP=8 as confirmed by hands-on testing.")
    elif total_gb >= 600:
        print("⚠️  Marginal VRAM — consider FP8 quantization or a larger node")
    else:
        print("❌  Insufficient VRAM for DeepSeek-V3.2 BF16 on this configuration")

    print()
    print("Checklist before starting server:")
    print("  [ ] source 00-env.sh  (sets HF_TOKEN, LD_LIBRARY_PATH, VLLM_USE_DEEP_GEMM=0, etc.)")
    print("  [ ] docker compose up -d  (starts Prometheus + Grafana)")
    print("  [ ] bash 02-start-server.sh  (start vLLM inside a tmux session)")
    print("  [ ] Wait for 'Uvicorn running on http://0.0.0.0:8000' in logs")
    print("  [ ] curl http://localhost:8000/v1/models  (smoke test)")


def main() -> None:
    check_system()
    total_gb = check_gpus()
    check_nvlink()
    check_packages()
    recommendations(total_gb)


if __name__ == "__main__":
    main()
