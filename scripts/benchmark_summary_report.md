# DeepSeek-V3.2 Deployment and Benchmark Summary

## Executive Summary
This report summarizes the deployment configuration and benchmark behavior for the `deepseek-ai/DeepSeek-V3.2` deployment on a single `8x NVIDIA H200` RunPod node.

The headline finding is:

- The later benchmarks are fast without `LMCache` KV transfer because the current serving setup still has `vLLM` prefix caching enabled.
- The large TTFT gap between the early and later runs is best explained by workload shape, especially prompt length and prefill cost, plus a warmer runtime state.
- There is no strong evidence of anything "fishy" in the later batch runs; the data is consistent with a shorter-prompt workload and cache-friendly repeated prefixes.

## Deployment Configuration

| Item | Value | Source |
| --- | --- | --- |
| Model | `deepseek-ai/DeepSeek-V3.2` | `models/deepseek-v3.2/00-env.sh` |
| Hardware | `8x NVIDIA H200` | `models/deepseek-v3.2` |
| CPU | `224 cores` | `models/deepseek-v3.2` |
| RAM | `2013 GB` | `models/deepseek-v3.2` |
| Tensor parallelism | `8` | `models/deepseek-v3.2/00-env.sh` |
| Max model length | `16384` | `models/deepseek-v3.2/00-env.sh` |
| GPU memory utilization | `0.8` | `models/deepseek-v3.2/00-env.sh` |
| Serve port | `8000` | `models/deepseek-v3.2/00-env.sh` |
| Metrics port | `8100` | `models/deepseek-v3.2/00-env.sh` |
| Dtype in current start script | `float8` | `models/deepseek-v3.2/02-start-server.sh` |
| Tokenizer mode | `deepseek_v32` | `models/deepseek-v3.2/02-start-server.sh` |
| Prefix caching | `enabled` | `models/deepseek-v3.2/02-start-server.sh` |
| Chunked prefill | `enabled` | `models/deepseek-v3.2/02-start-server.sh` |
| CUDA graph | `disabled` in current script | `models/deepseek-v3.2/02-start-server.sh` |
| Eager execution | `enabled` via `--enforce-eager` | `models/deepseek-v3.2/02-start-server.sh` |
| KV transfer / LMCache connector | `disabled` in current script | `models/deepseek-v3.2/02-start-server.sh` |

## Framework and Runtime Versions

| Component | Version / Status | Confidence |
| --- | --- | --- |
| Python | `3.12.x` | High. `01-install.sh` creates a Python 3.12 venv; repo validation examples show `3.12.3`. |
| vLLM | `0.16.0` | High for the benchmarked runtime. Historical server logs in the repo show `version 0.16.0`. |
| PyTorch | `2.8.0+cu128` | Medium. Repo validation examples show this exact version, but `01-install.sh` installs PyTorch nightly rather than pinning it. |
| CUDA | `12.8.1` | High. Present in `GPU_files/env`. |
| cuBLAS | `12.8.4.1-1` | High. Present in `GPU_files/env`. |
| cuDNN | `9.8.0.87-1` | High. Present in `GPU_files/env`. |
| NCCL package env | `2.25.1-1` | High. Present in `GPU_files/env`. |
| NCCL observed at runtime | `2.27.5` | High for the historical run captured in `server.log`. |
| DeepGEMM | `v2.1.1.post3` | High. Explicitly checked out in `01-install.sh`. |
| Transformers | Installed, exact version not pinned in setup script | Medium. `01-install.sh` installs latest available at install time. |
| LMCache | Installed, exact version not pinned in setup script | Medium. `01-install.sh` installs it, but the current start script does not enable KV transfer. |
| NIXL | Installed, exact version not pinned in setup script | Medium. Installed in `01-install.sh`. |

## KV Cache and Caching Status

This is the key distinction:

| Feature | Status | What it means |
| --- | --- | --- |
| Standard per-request KV cache inside vLLM | Active | Every in-flight generation still uses normal KV cache during decoding. |
| Prefix caching (`--enable-prefix-caching`) | Enabled | Repeated prompt prefixes can reuse already-built KV blocks in GPU memory. |
| Chunked prefill (`--enable-chunked-prefill`) | Enabled | Helps smooth prefill behavior and reduce TTFT spikes under load. |
| LMCache KV transfer (`--kv-transfer-config`) | Disabled in the current script | No external KV transfer connector is being used. |
| CPU spill / persistent external KV reuse | Not enabled in the current script | There is no evidence that the current script is offloading or reloading KV blocks through LMCache. |

## Single-Request Reference Points

The `deploy/individual request.txt` file shows why prompt length matters so much:

| Scenario | Approx input tokens | Output tokens | TTFT | Total time | Notes |
| --- | --- | --- | --- | --- | --- |
| Short research prompt | `~28` | `285` | `0.067s` | `6.706s` | Very fast first token on a short prompt. |
| Medium explicit-content summary prompt | `~853` | `129` | `0.182s` | `3.173s` | Still fast TTFT. |
| Long prompt example | `~16372` | `6` | `1.762s` | `1.906s` | Prefill is visibly higher even for a tiny completion. |
| Very long prompt example | `~11228` | `12` | `1.172s` | `1.459s` | Same pattern: prompt length drives TTFT. |

These single-request tests show that long prompts raise TTFT even before adding concurrency.

## Benchmark Results

Primary table:

| Run | Requests | Target QPS | Achieved QPS | Avg Input Tokens | Avg Output Tokens | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | E2E P95 | E2E P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `benchmark_results_1.json` | 150 | 15 | n/a | n/a | 29.5 | 46.84s | 91.25s | 94.40s | 68.02s | 103.03s | 105.49s |
| `benchmark_results_2.json` | 300 | 30 | 1.60 | 11644.1 | 66.6 | 87.48s | 160.46s | 167.05s | 120.40s | 182.42s | 185.69s |
| `benchmark_results_3.json` | 100 | 10 | 5.46 | 5625.9 | 90.0 | 0.20s | 1.12s | 1.18s | 1.33s | 14.57s | 16.77s |
| `benchmark_results_4.json` | 100 | 10 | 3.98 | 20.1 | 292.2 | 0.18s | 0.42s | 0.46s | 15.12s | 23.07s | 24.64s |
| `benchmark_results_5.json` | 250 | 25 | 7.21 | 19.0 | 333.8 | 0.30s | 0.73s | 0.75s | 23.17s | 33.04s | 34.10s |
| `benchmark_results_6.json` | 600 | 60 | 11.02 | 18.8 | 316.6 | 1.02s | 1.44s | 1.51s | 38.94s | 51.92s | 53.98s |

Additional note:

- `benchmark_results.json` appears to be another `300 requests @ 30 QPS` run, with `TTFT P50 = 111.86s`, `TTFT P95 = 210.71s`, `E2E P50 = 143.95s`, and `E2E P95 = 224.13s`.
- It is best treated as an additional poor `30 QPS` run rather than a separate benchmark tier.

## What the Benchmark Numbers Mean

| Metric | Meaning | Practical interpretation |
| --- | --- | --- |
| TTFT | Time to first token | Dominated by queueing, prompt prefill, scheduling, and any prefix-cache benefit before generation starts. |
| E2E | End-to-end latency | Total request time from dispatch to completion, including TTFT plus all decode time. |
| P50 | Median | What a typical request sees. |
| P95 | 95th percentile | Tail latency seen by the slowest 5% of requests. |
| P99 | 99th percentile | Worst tail behavior, useful for judging instability under load. |
| Achieved QPS | Actual served throughput | The effective throughput the system sustained, regardless of the requested target. |

## Interpretation of the Results

### 1. The later runs are not evidence of LMCache KV transfer

The current `02-start-server.sh` does **not** pass `--kv-transfer-config`, so the LMCache connector is not active in the checked-in serving command.

However, the script **does** enable:

- `--enable-prefix-caching`
- `--enable-chunked-prefill`

That means fast TTFT in the later runs is still plausible without LMCache KV transfer.

### 2. Prompt length is the biggest reason the early runs look terrible

The strongest evidence is `benchmark_results_2.json`:

- Average input length is about `11.6k` tokens.
- Average output length is only about `66.6` tokens.
- TTFT P50 is `87.48s`.

That is classic prefill-dominated behavior: the system spends most of its time chewing through the input before it can emit the first token.

By contrast:

- `benchmark_results_4.json`, `benchmark_results_5.json`, and `benchmark_results_6.json` all use roughly `19-20` input tokens on average.
- Their TTFT P50 ranges from `0.18s` to `1.02s`.

So the late-run workload is simply much easier.

### 3. Why `benchmark_results_6.json` is still fast on TTFT

`benchmark_results_6.json` reaches:

- `600` requests
- `60` target QPS
- `11.02` achieved QPS
- `1.02s` TTFT P50

This is credible because:

- the prompts are very short on average (`18.8` tokens),
- prefix caching is enabled,
- the model/runtime was likely already warm,
- and TTFT only measures time to the first token, not the full completion.

The same run still shows heavy total latency:

- `E2E P50 = 38.94s`
- `E2E P95 = 51.92s`

So the system is fast to *start* responding, but it is not actually serving the full completions cheaply under that load.

### 4. Warmup probably helped, but warmup alone does not explain everything

A warm runtime can help through:

- already-loaded weights,
- allocator stabilization,
- compiled kernels already present,
- a generally hotter serving process.

But warmup alone is not enough to explain the difference between:

- `87s` TTFT on average with `~11.6k` input tokens, and
- `0.18s` to `1.02s` TTFT with `~19` input tokens.

The bigger driver is the workload itself.

### 5. Why `benchmark_results_3.json` looks better than its average input length suggests

`benchmark_results_3.json` still reports a high average input length (`5625.9`), but the latency is much better than runs 1 and 2.

The most likely explanation is a combination of:

- mixed prompt lengths rather than a uniformly long workload,
- cache-friendly repeated prefixes,
- lighter request pressure than the `30 QPS` run,
- and a warmer server state.

So this run looks transitional rather than contradictory.

## Final Conclusion

The data does **not** support the idea that a hidden KV-transfer feature was making the later runs look artificially good.

What the data supports is:

1. `LMCache` KV transfer is not enabled in the current start script.
2. `vLLM` prefix caching is enabled, so repeated prefixes can still be fast.
3. Early bad runs were dominated by very large prompts and prefill cost, especially the `30 QPS` benchmark with `~11.6k` average input tokens.
4. Later runs used much shorter prompts, so their TTFT naturally dropped.
5. The runtime being warm likely helped, but the prompt mix is the main reason the numbers improved.

## Short Executive Summary

We deployed `DeepSeek-V3.2` on a single `8x H200` RunPod node with `TP=8`, `max_model_len=16384`, `CUDA 12.8.1`, `cuDNN 9.8.0.87`, and `vLLM 0.16.0` as captured in the available artifacts. The current serving script enables `prefix caching` and `chunked prefill`, but it does **not** enable `LMCache` KV transfer.

The benchmark story is straightforward: the very poor early TTFT results came from prefill-heavy workloads with extremely large prompts, while the later runs used much shorter prompts and a warmer runtime. That is why later tests can show sub-second or near-sub-second TTFT even without KV transfer. The fast later TTFT is real, but it reflects easier prompts and cache-friendly serving conditions rather than a hidden optimization.
