# KV Cache Fusion — deepseek-ai/DeepSeek-V2-Lite (15.7B)

## Why this model (from May 27 meeting)

The team agreed to validate the KV cache compression algorithm on a
small model that fits on one GPU, to measure latency and memory savings
when the cache hits chip capacity. DeepSeek-V2-Lite was selected.

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
| VRAM (BF16) | ~20 GB |
| Download size | ~32 GB |

---

## GPU — 1× RTX A6000 (48GB) on Koyeb

| Component | VRAM |
|---|---|
| Model weights | ~20 GB |
| KV cache at 128k (standard) | ~7 GB |
| KV cache at 128k (fused) | ~3.5 GB |
| Total peak | ~28 GB — fits in 48 GB |

Koyeb: select **RTX-A6000** ($0.75/h, high supply, no quota needed).

---

## KV cache savings

The meeting agreed to test with long context that exhausts on-chip memory.
128k context is the target — this is where the savings matter most.

| Context | Standard KV | Fused (V only) | Saved |
|---|---|---|---|
| 32k | 1,811 MB | 906 MB | ~50% |
| 64k | 3,623 MB | 1,811 MB | ~50% |
| 128k | 7,247 MB | 3,623 MB | ~50% |

N matrix overhead (static): **0.26 MB** total — negligible.

---

## Files

```
deepseek-v2-lite-kv/
├── kv_fusion_deepseek_v2_lite.py   ← storage math, N matrix, fused module
├── scripts/
│   ├── bench_e2e.py                ← E2E benchmark (real model)
│   └── build_report.py             ← comparison report from JSONs
└── results/                        ← created at runtime
    ├── e2e_baseline.json
    ├── e2e_fused.json
    └── report.txt
```

---

## Run order

```bash
# 1. Download model (~32GB)
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite \
    --local-dir /data/DeepSeek-V2-Lite

# 2. Storage analysis + correctness check (no GPU needed)
python kv_fusion_deepseek_v2_lite.py

# 3. Baseline — measure latency and KV cache WITHOUT fusion
python scripts/bench_e2e.py \
    --model-path /data/DeepSeek-V2-Lite \
    --fusion off \
    --seq-len 131072 \
    --new-tokens 128 \
    --batch-sizes 1 \
    --warmup 2 --iters 5 \
    --out results/e2e_baseline.json

# 4. Fused — measure latency and KV cache WITH fusion
python scripts/bench_e2e.py \
    --model-path /data/DeepSeek-V2-Lite \
    --fusion on \
    --seq-len 131072 \
    --new-tokens 128 \
    --batch-sizes 1 \
    --warmup 2 --iters 5 \
    --out results/e2e_fused.json

# 5. Compare results
python scripts/build_report.py
cat results/report.txt
```
