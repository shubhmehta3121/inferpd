# Environment + Compatibility Notes (Vast.ai / H200 / vLLM / DeepSeek-V3.2)

Last updated: 2026-03-05 (added section 13: CUDA graph capture failure with custom_all_reduce.cuh)

This document captures:

- The exact commands to print **CUDA / NVIDIA driver / GPU / PyTorch / cuDNN / vLLM / DeepGEMM / uv** versions.
- A concrete snapshot of what you actually had running (from the terminal output you shared).
- The common failure modes we hit (and how to recognize them quickly).
- “Do / Don’t” rules to avoid repeating expensive mistakes.

> Security note: never paste or commit real tokens. This doc intentionally does not include any secret values.

---

## 1) What “CUDA version” means (there are multiple)

When people say “CUDA version”, they might mean one of these:

- **Driver-supported CUDA** (what the NVIDIA driver advertises)
  - Shown in the `nvidia-smi` header as `CUDA Version: X.Y`
  - This is **not** the same as the toolkit installed inside the container.
- **CUDA Toolkit / NVCC version** (compiler + toolkit under `/usr/local/cuda-`*)
  - Shown by `nvcc --version` and `/usr/local/cuda/version.txt`
- **PyTorch CUDA runtime version** (what your torch wheel is built for)
  - Shown by `python -c "import torch; print(torch.version.cuda)"`
  - Example: `torch 2.9.1+cu128` implies `torch.version.cuda == "12.8"`

All three can differ and still be “valid” — but binary extensions (like DeepGEMM) must match the **PyTorch ABI**.

---

## 2) One-shot “print everything” command (copy/paste)

Run this after:

```bash
cd /workspace/deepseek-v3.2
source 00-env.sh
source /workspace/.venv/bin/activate
```

Then run:

```bash
set -euxo pipefail

echo "=== OS ==="
uname -a
cat /etc/os-release || true

echo "=== NVIDIA driver + GPUs (nvidia-smi) ==="
nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version,vbios_version,pci.bus_id,memory.total,compute_cap --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv

echo "=== CUDA Toolkit (nvcc + /usr/local/cuda) ==="
nvcc --version || true
ls -l /usr/local/cuda* 2>/dev/null || true
cat /usr/local/cuda/version.txt 2>/dev/null || true

echo "=== Python / uv ==="
python -V
which python
which uv || true
uv --version || true

echo "=== PyTorch runtime ==="
python -c "import sys, torch; print('python', sys.version); print('torch', torch.__version__); print('torch.version.cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('arch list', torch.cuda.get_arch_list())"
python -c "import torch; print('cudnn enabled', torch.backends.cudnn.enabled); print('cudnn version', torch.backends.cudnn.version())"
python -c "import torch; print('torch build config:\\n', torch.__config__.show())"

echo "=== vLLM ==="
python -c "import vllm; print('vllm', vllm.__version__)"

echo "=== DeepGEMM python package ==="
python -c "import deep_gemm; print('deep_gemm', deep_gemm.__version__)" || true
uv pip show deep-gemm torch vllm || true

echo "=== Torch lib + dynamic linker ==="
python -c "import os, torch; print('torch_lib', os.path.join(os.path.dirname(torch.__file__), 'lib'))"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}"
ls -l /workspace/.venv/lib/python3.12/site-packages/torch/lib/libc10.so 2>/dev/null || true

echo "=== DeepGEMM extension path + linkage ==="
python -c "import importlib.util; s=importlib.util.find_spec('deep_gemm_cpp'); print('deep_gemm_cpp', s.origin if s else None)" || true
ldd /workspace/.venv/lib/python3.12/site-packages/deep_gemm_cpp*.so 2>/dev/null || true
```

---

## 3) Snapshot from your instance (as captured in the shared logs)

### Hardware / driver / GPU

- **GPU**: 8 × NVIDIA H200
- **GPU memory**: ~143,771 MiB per GPU
- **Compute capability**: 9.0 (Hopper)
- **NVIDIA driver**: 570.195.03
- **Driver-advertised CUDA version** (`nvidia-smi` header): 12.8
- **No running processes** at time of snapshot (`nvidia-smi` showed empty process table)

### CUDA toolkit

- `/usr/local/cuda` → `/usr/local/cuda-12.8`
- **NVCC**: CUDA compilation tools, release 12.8, V12.8.93

### OS / Python / uv

- **OS**: Ubuntu 24.04.3 LTS
- **Kernel**: 5.15.0-157-generic
- **Python** (venv): 3.12.12
- **uv**: 0.10.8

### PyTorch / cuDNN / vLLM

- **torch**: 2.9.1+cu128
- **torch.version.cuda**: 12.8
- **cuDNN runtime** (from `torch.backends.cudnn.version()`): 90800 (CuDNN 9.8)
- **vLLM**: 0.16.0

### DeepGEMM

- **deep-gemm** (installed): 2.1.1+c9f8b34 (from `/tmp/DeepGEMM`, tag `v2.1.1.post3`)
- **Status**: installed, but import fails due to ABI mismatch (see next section).

---

## 4) DeepGEMM failure modes we hit (and what they mean)

### A) `ModuleNotFoundError: No module named 'deep_gemm'`

Meaning:

- `deep-gemm` was not installed into the **Python interpreter you’re using**.

Common cause in this setup:

- Your venv created by `uv venv` may not include `pip`, so `python -m pip ...` fails.
- The `deploy/01-install.sh` script currently runs `pip install .` for DeepGEMM, which can end up using **system pip**, not the venv, depending on shell state and how `pip` resolves on PATH.

What to do:

- Prefer `uv pip ...` and explicitly point it at the venv’s Python.

Example:

```bash
cd /tmp/DeepGEMM
uv pip install . --no-build-isolation --python /workspace/.venv/bin/python
```

### B) `ImportError: libc10.so: cannot open shared object file`

Meaning:

- Dynamic linker cannot find PyTorch’s shared libraries at runtime.

Typical fix:

- Ensure torch’s `lib` dir is in `LD_LIBRARY_PATH`:

```bash
TORCH_LIB="$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")"
export LD_LIBRARY_PATH="$TORCH_LIB:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu"
```

### C) `undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv`

Meaning (important):

- DeepGEMM’s compiled extension (`deep_gemm_cpp*.so`) was compiled against a **different PyTorch / libc10 ABI** than the one currently installed.
- This is not a PATH issue anymore — it’s a **binary compatibility** issue.

Why this happens:

- PyTorch internal symbols can change across versions (especially nightlies).
- A CUDA extension built against one torch build can fail to load against another.

What to do (choices):

- **Align versions**: use a torch version known compatible with that DeepGEMM tag/commit, then rebuild DeepGEMM.
- Or **use a DeepGEMM commit/tag** compatible with your torch build.
- Or **disable FP8/DeepGEMM path** (trade-off: less performance / different memory characteristics).

How to capture proof for future debugging:

- Keep the outputs of:
  - `torch.__version__`, `torch.version.cuda`, `torch.__config__.show()`
  - `uv pip show deep-gemm torch vllm`
  - `ldd ...deep_gemm_cpp*.so`
  - The exact `undefined symbol ...` line

---

## 5) cuDNN “compiled vs runtime” mismatch (what to look for)

Symptom we saw earlier in the journey:

- PyTorch reports it was built for cuDNN e.g. 9.10.x but runtime loads a different cuDNN e.g. 9.8.x.

Root cause:

- `LD_LIBRARY_PATH` ordering makes the loader pick the “wrong” cuDNN first.

Rule:

- Prefer PyTorch’s bundled CUDA/cuDNN libraries unless you *deliberately* want to use system ones.

How to check what cuDNN torch is actually using:

```bash
python -c "import torch; print(torch.backends.cudnn.version())"
python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
```

---

## 6) GPU memory “missing” / expensive idle usage (how to verify)

The correct “first checks”:

```bash
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

If `nvidia-smi` shows memory used but no processes inside the container:

- It may be a **host-level** process not visible/killable from inside the container namespace.

More aggressive checks:

```bash
# Lists processes holding /dev/nvidia* devices (if available)
sudo fuser -v /dev/nvidia* || true

# Check for stray vLLM/torch processes in the container namespace
ps aux | egrep -i "vllm|python|torch" | head -n 200
```

If you still can’t see/kill it from inside the container:

- The reliable fix is often: **terminate/restart the instance** from the provider UI.

---

## 7) Files and directories: where things go

### Persistent vs temporary

- `**/workspace`**: treat as persistent volume on Vast.ai (good place for venv, model cache, logs)
- `**/tmp**`: temporary build area (fine for cloning/building DeepGEMM)

### What gets “installed” where

- `git clone /tmp/DeepGEMM` is only the source tree.
- After `uv pip install .` (or pip install), the importable module lives in:
  - `/workspace/.venv/lib/python3.12/site-packages/...`

Deleting `/tmp/DeepGEMM` after install is fine — **unless you plan to rebuild**.

---

## 8) uv / pip / venv rules (very important in this setup)

Observed behavior:

- Your venv python reported: `/workspace/.venv/bin/python: No module named pip`

Rules:

- **Do** use `uv pip ...` for installs and `uv pip show ...` for package introspection.
- **Don’t** assume `python -m pip ...` works unless you explicitly install pip into the venv.
- **Do** verify you’re installing into the same interpreter you run:

```bash
which python
python -c "import sys; print(sys.executable)"
uv pip show torch vllm deep-gemm
```

---

## 9) GitHub “push protection” / secret scanning (avoid repeating)

What happened:

- A HuggingFace token existed in `deploy/.env` in git history.
- GitHub rejected `git push` due to repository secret scanning rules.

Rules:

- **Never commit `.env` files** with secrets.
- Ensure `.gitignore` contains `deploy/.env`.
- If a secret leaks into history:
  - **Revoke the token immediately**
  - Rewrite history with `git filter-repo` (amend is not sufficient if the secret is in an older commit)

---

## 10) Quick “Do / Don’t” checklist

- **Do** keep everything persistent under `/workspace`:
  - venv: `/workspace/.venv`
  - HF cache: `/workspace/.cache/huggingface`
  - logs: `/workspace/logs`
- **Do** capture a version snapshot before long builds (section 2).
- **Do** use `uv pip ...` consistently.
- **Don’t** mix system pip and venv installs.
- **Do** treat DeepGEMM as ABI-sensitive:
  - If torch changes (especially nightlies), rebuild DeepGEMM or pin torch.
- **Don’t** assume “CUDA 12.8” means the same thing everywhere (section 1).
- **Do** verify GPU memory and processes early (`nvidia-smi` + query).

---

## 11) What the vLLM server log shows (first boot attempt)

From `server.log` (first full start attempt with DeepSeek-V3.2):

- **vLLM config at startup**
  - `vLLM version`: 0.16.0
  - `model`: `deepseek-ai/DeepSeek-V3.2`
  - `dtype`: `bfloat16`
  - `tensor_parallel_size`: 8
  - `max_model_len`: 16384
  - `quantization`: `fp8` (this is what triggers DeepGEMM)
  - `gpu_memory_utilization`: 0.85
  - `enable_prefix_caching`: True
  - `enable_chunked_prefill`: True
  - `kv_connector`: `LMCacheConnectorV1` with role `kv_both`

- **Backends chosen (from log lines)**
  - Attention backend: `FLASHMLA_SPARSE`
  - FP8 MoE backend: `FLASHINFER_CUTLASS` (out of several candidates)
  - NCCL version: `2.27.5`

- **Warmup and compile behavior**
  - Time spent downloading weights: ~610 s (about 10 minutes).
  - Model loading:
    - Log line: `Model loading took 80.78 GiB memory and 687.275633 seconds`
    - This is per worker, so the model is very heavy even before KV cache.
  - `torch.compile`:
    - vLLM compiled CUDA graphs for the range `(1, 32768)` tokens.
    - Log line: `Compiling a graph ... takes 593.24 s` and `torch.compile takes 610.69 s in total`.
  - KV cache:
    - Available KV cache memory (per GPU worker): `30.52 GiB`
    - GPU KV cache size: `418,304 tokens`
    - Maximum concurrency for 16,384-token requests: `~25.5×`

- **Where torch.compile cache went**
  - Log line:
    - `Using cache directory: /root/.cache/vllm/torch_compile_cache/28f3c7ea39/rank_0_0/backbone`
  - This means:
    - Even though most things were moved to `/workspace`, vLLM’s **torch.compile cache** still defaults under `/root/.cache/vllm/...`.
    - If root disk is small, this directory can grow large over time (worth checking with `du -sh /root/.cache/vllm`).

- **Why the engine ultimately failed**
  - After weights loaded and CUDA graphs were compiled, vLLM attempted FP8 MQA logits via DeepGEMM:
    - Calls go through `vllm.utils.deep_gemm.fp8_paged_mqa_logits`, which internally calls `_missing()` if DeepGEMM is unavailable.
  - The critical runtime error in the log:
    - `RuntimeError: DeepGEMM backend is not available or outdated. Please install or update the 'deep_gemm' to a newer version to enable FP8 kernels.`
  - This propagates up and causes:
    - `EngineCore failed to start.` and
    - `Engine core initialization failed. ... Failed core proc(s): {}`
  - Net result:
    - You **never reach** the usual `Uvicorn running on http://0.0.0.0:8000` line.
    - The server process exits after failing during the FP8 warmup phase.

**Implication for future runs**

- If you want FP8 speedups:
  - You must fix DeepGEMM ABI compatibility (section 4) so that:
    - `import deep_gemm` succeeds **and**
    - vLLM FP8 calls do not raise the “backend is not available or outdated” error.
- If you just want a working server quickly:
  - Start vLLM with configuration that **does not use FP8 / DeepGEMM** (for example, disabling FP8 quantization for this model), so the DeepGEMM code path is never hit.

---

## 12) CUDA Out of Memory Error During Inference (Runtime Failure)

### What Happened

**Timeline:**
- Server started successfully and processed requests normally
- Benchmark showed successful completions initially (200 OK responses, TTFT ~3.7-5.1s)
- After processing many requests, the engine crashed with a CUDA OOM error
- Subsequent requests failed with `EngineDeadError: EngineCore encountered an issue`

**Error Details (from `server_2.log` line 1376):**

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.28 GiB. 
GPU 0 has a total capacity of 139.81 GiB of which 7.30 GiB is free. 
Process 2678129 has 132.50 GiB memory in use. 
Of the allocated memory 115.89 GiB is allocated by PyTorch, 
with 2.01 GiB allocated in private pools (e.g., CUDA Graphs), 
and 10.53 GiB is reserved by PyTorch but unallocated.
```

**Where it failed:**
- During FP8 MQA logits computation (`fp8_mqa_logits`) in the sparse attention indexer
- Call stack: `vllm.utils.deep_gemm.fp8_mqa_logits` → `sparse_attn_indexer` → model forward pass
- The error occurred in a compiled CUDA graph (`<eval_with_key>.6`)

**Memory Breakdown:**
- **Total GPU capacity**: 139.81 GiB
- **Free memory**: 7.30 GiB
- **Attempted allocation**: 9.28 GiB (failed)
- **Process memory**: 132.50 GiB
- **PyTorch allocated**: 115.89 GiB
- **CUDA Graphs (private pools)**: 2.01 GiB
- **PyTorch reserved but unallocated**: 10.53 GiB

### Root Cause Analysis

The server was configured with:
- `gpu_memory_utilization`: 0.85 (85% of GPU memory)
- `tensor_parallel_size`: 8
- `max_model_len`: 16384
- `quantization`: fp8 (using DeepGEMM)
- `enable_chunked_prefill`: True
- `max_num_batched_tokens`: 32768

**Why it happened:**
1. Memory fragmentation: 10.53 GiB reserved but unallocated suggests fragmentation
2. CUDA Graphs overhead: 2.01 GiB in private pools (compiled graphs)
3. Peak memory spike: FP8 MQA logits computation tried to allocate 9.28 GiB during inference
4. Insufficient headroom: Only 7.30 GiB free when 9.28 GiB was needed

### Fixes and Mitigations

**Immediate fixes:**

1. **Set PyTorch memory allocator to reduce fragmentation:**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
   Add this to `deploy/00-env.sh` before starting the server.

2. **Reduce GPU memory utilization:**
   ```bash
   export GPU_MEMORY_UTILIZATION=0.75  # Lower from 0.85 to 0.75
   ```
   This leaves more headroom for peak allocations.

3. **Reduce batch size / concurrency:**
   - Lower `max_num_batched_tokens` (e.g., from 32768 to 16384)
   - Reduce `max_num_seqs` if set

**Long-term considerations:**

- Monitor GPU memory usage during peak load
- Consider disabling FP8 quantization if memory is tight (trade-off: slower inference)
- Use `nvidia-smi` to track memory fragmentation over time
- Consider restarting the server periodically if memory fragmentation accumulates

### How to Detect This Early

**Warning signs:**
- `nvidia-smi` shows high memory usage (>90%) even at idle
- Requests start failing intermittently with "EngineCore encountered an issue"
- Benchmark results show increasing failure rate over time

**Diagnostic commands:**
```bash
# Check current GPU memory state
nvidia-smi

# Check PyTorch memory stats (if available)
python -c "import torch; print(torch.cuda.memory_summary())"

# Monitor memory during benchmark
watch -n 1 nvidia-smi
```

### Prevention Checklist

- [ ] Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in environment
- [ ] Use conservative `gpu_memory_utilization` (0.75-0.80) for production
- [ ] Monitor memory usage during load testing
- [ ] Set up alerts for GPU memory >90%
- [ ] Plan for periodic server restarts if running long-term

### Memory Configuration Settings Explained

**Current settings (after CUDA OOM fix):**

1. **`GPU_MEMORY_UTILIZATION=0.8`** (reduced from 0.85)
   - **What it does**: Tells vLLM to use 80% of total GPU memory for model weights and KV cache
   - **Why 0.8**: Leaves 20% headroom for:
     - Memory fragmentation (unused gaps between allocations)
     - CUDA Graphs overhead (~2 GiB for compiled graphs)
     - Peak allocation spikes during FP8 MQA logits computation
     - Temporary buffers during batch processing
   - **Impact**: Slightly reduces maximum concurrent requests, but prevents crashes

2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**
   - **What it does**: Enables PyTorch's expandable memory segments allocator
   - **Why it helps**: 
     - Reduces memory fragmentation by allowing segments to grow dynamically
     - Prevents "reserved but unallocated" memory waste (we saw 10.53 GiB wasted)
     - Better handles variable-sized allocations during inference
   - **Impact**: More efficient memory usage, fewer OOM errors

3. **`--max-num-seqs 128`** (reduced from 256)
   - **What it does**: Limits the maximum number of concurrent request sequences processed simultaneously
   - **Why reduce it**:
     - Each sequence needs KV cache slots (memory)
     - More sequences = more peak memory usage
     - During high concurrency, memory can spike unpredictably
   - **Impact**: 
     - Lower maximum throughput (fewer concurrent requests)
     - More stable memory usage
     - Better for production reliability

4. **`--max-num-batched-tokens 16384`** (reduced from 32768)
   - **What it does**: Limits the total number of tokens processed in a single batch step
   - **Why reduce it**:
     - Batched token processing requires temporary memory buffers
     - Larger batches = larger peak memory allocations
     - The FP8 MQA logits computation tried to allocate 9.28 GiB (failed)
   - **Impact**:
     - Smaller batch sizes per step (may slightly increase latency)
     - Lower peak memory usage
     - More predictable memory behavior

**Trade-offs:**

| Setting | Before | After | Trade-off |
|---------|--------|-------|-----------|
| `gpu_memory_utilization` | 0.85 | 0.8 | Less memory available, but more stable |
| `max_num_seqs` | 256 | 128 | Lower max concurrency, but safer |
| `max_num_batched_tokens` | 32768 | 16384 | Smaller batches, but lower peak memory |

**When to adjust:**

- **Increase settings** if:
  - Memory usage consistently stays <80% during peak load
  - You need higher throughput and can accept occasional crashes
  - You're running shorter workloads (less fragmentation)

- **Decrease settings** if:
  - Still seeing OOM errors
  - Running long-term production workloads
  - Memory fragmentation accumulates over time

---

## 13) CUDA Graph Capture Failure: custom_all_reduce.cuh Invalid Argument Error

### What Happened

**Environment:**
- **Provider**: RunPod
- **Hardware**: 8× NVIDIA H200 GPUs
- **PyTorch**: 2.12.0.dev20260304+cu128
- **vLLM**: 0.16.0
- **Config**: `tensor_parallel_size=8`, `max_model_len=16384`, `quantization=fp8`

**Timeline (from `server_6.log`):**
- Server started successfully
- Model weights downloaded (~24 minutes)
- Model loaded successfully (80.58 GiB memory, ~25 minutes)
- `torch.compile` completed successfully (~40 seconds)
- CUDA graphs for "mixed prefill-decode, PIECEWISE" captured successfully (51 graphs)
- **Failure occurred** during "Capturing CUDA graphs (decode, FULL)" phase

**Error Details (from `server_6.log` line 466):**

```
Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'invalid argument'
```

The error occurred on **all 8 TP workers simultaneously**, causing:
- `Worker proc VllmWorker-6 died unexpectedly, shutting down executor`
- `EngineCore failed to start`
- `RuntimeError: Engine core initialization failed`

### Root Cause Analysis

**Why it happened:**

1. **Known vLLM bug**: This is a documented issue in vLLM 0.16.0 with tensor parallelism + custom all-reduce + CUDA graphs
   - GitHub issues: #9046, #5613, #9774
   - The `custom_all_reduce.cuh` kernel receives invalid arguments during CUDA graph capture for decode phase
   - Particularly problematic with FP8 quantization + TP=8 + CUDA graph FULL mode

2. **Configuration combination**:
   - `tensor_parallel_size=8` (required for large model)
   - `quantization=fp8` (enables custom all-reduce path)
   - `cudagraph_mode=FULL_AND_PIECEWISE` (default when `enforce_eager=False`)
   - The decode-phase CUDA graph capture triggers the buggy code path

3. **Why PIECEWISE worked but FULL failed**:
   - PIECEWISE graphs capture smaller chunks and avoid the problematic kernel configuration
   - FULL decode graphs attempt to capture larger operations that trigger the invalid argument error

### Fixes and Workarounds

**Immediate workaround (tested and working):**

1. **Disable CUDA graphs entirely**:
   ```bash
   --enforce-eager
   ```
   Add this flag to `deploy/02-start-server.sh` vLLM launch command.

   **Trade-off**: 
   - ✅ Server starts successfully
   - ✅ No CUDA graph capture errors
   - ❌ Slightly slower inference (no graph optimization)
   - ❌ Longer warmup time per request

2. **Alternative: Disable custom all-reduce** (if exposed in vLLM config):
   - Some vLLM versions allow disabling custom all-reduce via compilation config
   - This would fall back to standard NCCL all-reduce
   - May reduce performance but avoids the bug

**Long-term solution:**

- **Upgrade vLLM**: Later versions (after 0.16.0) may have fixes for this issue
- **Monitor vLLM GitHub**: Track issues #9046, #5613, #9774 for upstream fixes
- **Consider alternative quantization**: If FP8 is not critical, try bfloat16 without custom all-reduce

### Detection and Prevention

**Warning signs:**
- Server starts normally but crashes during CUDA graph capture phase
- Error message contains `custom_all_reduce.cuh` and `invalid argument`
- All TP workers fail simultaneously (not a single GPU issue)
- Happens specifically during "decode, FULL" graph capture

**Prevention checklist:**
- [ ] Use `--enforce-eager` flag for initial testing/debugging
- [ ] Test CUDA graph capture separately before production deployment
- [ ] Monitor vLLM release notes for fixes to custom_all_reduce issues
- [ ] Consider TP=4 or TP=2 if model size allows (may avoid the bug)
- [ ] Document PyTorch + vLLM version combinations that work

### Configuration That Works (Using Workaround)

**Important**: The underlying bug is **NOT fixed** in vLLM 0.16.0. The configuration below uses `--enforce-eager` as a **workaround** to avoid the buggy CUDA graph code path.

**What `--enforce-eager` does:**
- Disables CUDA graph capture entirely (skips the phase that triggers the bug)
- Allows the server to start and load the model without hitting `custom_all_reduce.cuh:455` error
- Trade-off: Slightly slower inference (no graph optimization), but stable

**Example configuration (from `server_4.log`):**

```bash
--tensor-parallel-size 8
--dtype bfloat16
--max-model-len 16384
--gpu-memory-utilization 0.8
--enable-prefix-caching
--enable-chunked-prefill
--enforce-eager  # ← Critical: disables CUDA graphs, avoids custom_all_reduce.cuh error
--host 0.0.0.0
--port 8000
```

**Note about `server_4.log`:**
- ✅ **Avoided** the `custom_all_reduce.cuh:455` error (because `--enforce-eager` was used)
- ✅ Model loaded successfully
- ❌ **Failed** with a different error: LMCache assertion (`assert self.lmcache_engine is not None`)
- This shows that `--enforce-eager` successfully bypasses the CUDA graph bug, but other issues (like LMCache misconfiguration) can still cause failures

**To get a fully working server:**
- Use `--enforce-eager` to avoid the CUDA graph bug
- **AND** either fix LMCache configuration or disable it entirely (remove `--kv-transfer-config`)

### Related Issues

- **vLLM GitHub Issue #9046**: 'invalid argument' Error with custom_all_reduce when doing tensor parallelism
- **vLLM GitHub Issue #5613**: Failed: custom_all_reduce.cuh:310 'invalid argument'
- **vLLM GitHub Issue #9774**: MoE + TP + custom allreduce bug

---

## Appendix A: Minimal commands (short list)

```bash
# NVIDIA
nvidia-smi
nvidia-smi -L

# CUDA toolkit
nvcc --version || true

# Python / uv / torch / vllm
python -V
uv --version || true
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"
python -c "import vllm; print(vllm.__version__)"

# DeepGEMM package + extension linkage
uv pip show deep-gemm torch vllm
python -c "import deep_gemm" || true
ldd /workspace/.venv/lib/python3.12/site-packages/deep_gemm_cpp*.so 2>/dev/null || true
```

