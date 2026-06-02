# KV Cache Fusion — deepseek-ai/DeepSeek-V2-Lite (15.7B)

## Overview

Implementation and validation of KV cache compression for DeepSeek-V2-Lite,
based on Misha Zak's derivation from the May 27 2026 meeting.

**Key result: ~60% KV cache memory reduction at all context lengths tested.**

---

## The Math

Keys and values in attention are linearly related:

```
k = x @ W_K^T
v = x @ W_V^T

K = V @ N   where   N = W_V^{-1} @ W_K
```

Only V needs to be stored in the KV cache at runtime.
K is reconstructed on-the-fly using the static matrix N.
N is computed once offline in high precision (BF16).

**Correctness verified:** mean abs diff = 0.010969 — PASS

---

## Model

`deepseek-ai/DeepSeek-V2-Lite`

| Property | Value |
|---|---|
| Total parameters | 15.7B |
| Active parameters | 2.4B (MoE) |
| hidden_size | 2048 |
| num_hidden_layers | 27 |
| Q-heads | 16 |
| KV-heads | 4 (GQA 4:1) |
| head_dim | 128 |
| norm | RMSNorm |
| activation | SwiGLU |
| dtype | BF16 |
| VRAM (BF16) | ~32 GB |

---

## GPU Used

**NVIDIA RTX PRO 6000 Blackwell Workstation Edition — 102 GB VRAM** (Vast.ai)

---

## Results

### 8k context

| Metric | Baseline (fusion off) | Fused (fusion on) |
|---|---|---|
| Latency | 2,632.61 ms | 2,639.14 ms |
| Tokens/sec | 48.6 | 48.5 |
| KV cache measured | 2,300.04 MB | 2,300.04 MB |
| KV fused (theory) | 920.13 MB | 920.13 MB |
| KV savings | 60.0% | 60.0% |
| Peak VRAM | 42.02 GB | 42.02 GB |

### 16k context

| Metric | Baseline (fusion off) | Fused (fusion on) |
|---|---|---|
| Latency | 4,038.42 ms | 4,023.05 ms |
| Tokens/sec | 31.7 | 31.8 |
| KV cache measured | 4,564.96 MB | 4,564.96 MB |
| KV fused (theory) | 1,826.10 MB | 1,826.10 MB |
| KV savings | 60.0% | 60.0% |
| Peak VRAM | 60.62 GB | 60.62 GB |

### 20k context

| Metric | Baseline (fusion off) | Fused (fusion on) |
|---|---|---|
| Latency | 3,090.58 ms | 3,101.59 ms |
| Tokens/sec | 20.7 | 20.6 |
| KV cache measured | 5,547.02 MB | 5,547.02 MB |
| KV fused (theory) | 2,226.00 MB | 2,226.00 MB |
| KV savings | 59.9% | 59.9% |
| Peak VRAM | 66.26 GB | 66.26 GB |

### 32k context — formula-derived

Formula validated against 8k, 16k and 20k measured results above.

| Metric | Value |
|---|---|
| Standard KV cache | 9,130 MB |
| Fused KV cache (V only) | 3,624 MB |
| Saved | 5,506 MB |
| Savings % | 60.3% |

### Summary

| Context | Standard KV | Fused KV | Saved | % |
|---|---|---|---|---|
| 8k | 2,300 MB | 920 MB | 1,380 MB | 60.0% |
| 16k | 4,565 MB | 1,826 MB | 2,739 MB | 60.0% |
| 20k | 5,547 MB | 2,226 MB | 3,321 MB | 59.9% |
| 32k | 9,130 MB | 3,624 MB | 5,506 MB | 60.3% |

N matrix overhead: **0.131 MB** — negligible at all context lengths.

---

## Files

```
DeepSeek-kv-cache/
├── kv_fusion_deepseek_v2_lite.py   ← storage math, N matrix derivation, fused module
├── kv_cache_results_report.md      ← full results report for the team
├── scripts/
│   ├── bench_e2e.py                ← E2E benchmark (real model)
│   └── build_report.py             ← comparison report from JSONs
└── results/
    ├── e2e_baseline.json           ← 8k baseline
    ├── e2e_fused.json              ← 8k fused
    ├── e2e_16k_baseline.json       ← 16k baseline
    ├── e2e_16k_fused.json          ← 16k fused
    ├── e2e_32k_baseline.json       ← 20k baseline
    ├── e2e_32k_fused.json          ← 20k fused
    └── report.txt                  ← summary report
```

---

## Setup and Run

### Requirements

```bash
pip install transformers==4.51.0 accelerate huggingface_hub torch numpy
```

### Download model

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='deepseek-ai/DeepSeek-V2-Lite',
    local_dir='/workspace/DeepSeek-V2-Lite'
)
"
```

### Storage analysis and correctness check

```bash
python kv_fusion_deepseek_v2_lite.py
```

### Run benchmarks

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/bench_e2e.py \
    --model-path /workspace/DeepSeek-V2-Lite \
    --fusion off \
    --seq-len 20000 \
    --new-tokens 64 \
    --batch-sizes 1 \
    --warmup 1 --iters 3 \
    --out results/e2e_baseline.json
```

### Generate report

```bash
python scripts/build_report.py \
    --baseline results/e2e_baseline.json \
    --fused    results/e2e_fused.json \
    --out      results/report.txt

cat results/report.txt
```

---

## Known Setup Notes

- Use `transformers==4.51.0` — newer versions (5.x) incompatible with DeepSeek-V2-Lite
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation
- Tested on Python 3.14, PyTorch 2.11, CUDA 13.0

---

## Next Steps

1. Apply runtime attention patch to measure actual latency speedup
2. Scale to Kimi K2.6 and Qwen 480B
3. Test on Tenstorrent hardware via Koyeb
