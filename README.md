# inferpd

Self-taught R&D: disaggregated DeepSeek inference with a **shared L1 KV store** for prefix-heavy streaming chat.

**Headline** (DeepSeek V2-Lite, 3× A100, 1 prefiller + 2 decoders, LMCache MP): TTFT P50 **0.52s**, **8.5 / 10** req/s delivered. Production ballpark was ~2s TTFT and ~60 QPS on a V3.2-class model. On V3.2 NVFP4 (8× B200) the same architecture held sub-second TTFT only to ~2 req/s. This repo shows a shared L1 is the right shape for that traffic, not a production-capacity deploy.

- **Site:** [shubhmehta3121.github.io/inferpd](https://shubhmehta3121.github.io/inferpd/)
- **Write-ups:** [`docs/deep-dives/`](docs/deep-dives/README.md) (build, results, costs, insights, resources)
- **Study notes:** [`docs/learning/`](docs/learning/LEARNING_ROADMAP.md)

## Repository map

| Path | Role |
|------|------|
| **`models/`** | Per-experiment runbooks, env templates, and benchmark sweeps |
| **`docs/`** | Deep dives and learning notes |
| **`site/`** | One-page GitHub Pages pitch |
| **`scripts/`** | Helpers such as rebuilding `site/data/results.json` |

## Runbooks

| Use case | Folder |
|----------|--------|
| Headline V2-Lite 1P2D (shared L1) | [`models/deepseek-v2-lite-lmcmp-1p2d/`](models/deepseek-v2-lite-lmcmp-1p2d/) |
| V2-Lite MP 1P1D | [`models/deepseek-v2-lite-lmcache-mp/`](models/deepseek-v2-lite-lmcache-mp/) |
| V2-Lite NIXL 1P1D / 2P2D | [`models/deepseek-v2-lite/`](models/deepseek-v2-lite/) · [`models/deepseek-v2-lite-2p2d/`](models/deepseek-v2-lite-2p2d/) |
| V3.2 NVFP4 1P1D (8× B200) | [`models/deepseek-v3.2-lmcmp-1p1d/`](models/deepseek-v3.2-lmcmp-1p1d/) |
| V3.2 NVFP4 1P3D two-node (runbook; sweep blocked) | [`models/deepseek-v3.2-lmcmp-1p3d/`](models/deepseek-v3.2-lmcmp-1p3d/) |
| V3.2 BF16 single / dual node | [`models/deepseek-v3.2/`](models/deepseek-v3.2/) |

Copy `.env.example` to `.env` on the box. Do not commit `.env`.

## Run the site locally

Charts fetch `data/results.json`, which browsers block on `file://`. Serve the folder:

```bash
cd site
python -m http.server 8000
# then open http://localhost:8000
```

Pushing to `master` or `main` deploys `site/` via [`.github/workflows/pages.yml`](.github/workflows/pages.yml).

## License

[MIT](LICENSE).
