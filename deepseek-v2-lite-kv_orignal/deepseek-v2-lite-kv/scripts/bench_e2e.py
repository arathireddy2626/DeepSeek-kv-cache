"""
bench_e2e.py - DeepSeek-V2-Lite KV Cache Benchmark
"""

import os
import sys

# Force transformers to use our local model directory as the module cache
# This prevents it from copying and corrupting the file
os.environ["HF_MODULES_CACHE"] = "/workspace/DeepSeek-V2-Lite"

# Intercept transformers cache copy and fix the file before it is parsed
import importlib
import transformers.dynamic_module_utils as _dmu

_orig_check = _dmu.check_imports
def _patched_check(filename):
    try:
        with open(filename, 'r') as f:
            txt = f.read()
        # Fix broken try block if present
        if 'try:\ntry:' in txt or '    try:\n    try:' in txt or '    try:\nfrom' in txt:
            import re
            txt = re.sub(
                r'try:[\s\S]*?(?=from \.configuration_deepseek)',
                'try:\n    from transformers.utils.import_utils import is_torch_fx_available\nexcept ImportError:\n    def is_torch_fx_available():\n        return False\n',
                txt, count=1
            )
            with open(filename, 'w') as f:
                f.write(txt)
    except Exception:
        pass
    return _orig_check(filename)

_dmu.check_imports = _patched_check

import argparse
import json
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kv_fusion_deepseek_v2_lite import compute_kv_storage, DEEPSEEK_V2_LITE_CONFIG


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_info():
    if not torch.cuda.is_available():
        return []
    return [
        {
            "index":   i,
            "name":    torch.cuda.get_device_name(i),
            "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1),
        }
        for i in range(torch.cuda.device_count())
    ]


def load_model(model_path, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Fix DynamicCache.seen_tokens for newer transformers
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: None
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = lambda self, seq_len, layer_idx=0: self.get_seq_length()
        if not hasattr(DynamicCache, "reorder_cache"):
            DynamicCache.reorder_cache = lambda self, beam_idx: None
    except Exception:
        pass

    print(f"[bench] Loading {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        tok = None
        print("[bench] No tokenizer — using random inputs")

    mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[bench] Loaded  |  GPU memory: {mem:.2f} GB")
    return model, tok


def make_inputs(tok, batch_size, seq_len, device, vocab_size=32000):
    if tok is not None:
        text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10)
        enc = tok(
            [text] * batch_size,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding="max_length",
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return ids, torch.ones_like(ids)


def run_benchmark(model, tok, args, dtype):
    device  = next(model.parameters()).device
    cfg     = model.config
    vocab   = getattr(cfg, "vocab_size", 32000)
    results = []

    L = getattr(cfg, "num_hidden_layers",   DEEPSEEK_V2_LITE_CONFIG["num_hidden_layers"])
    H = getattr(cfg, "num_key_value_heads", DEEPSEEK_V2_LITE_CONFIG["num_key_value_heads"])
    D = DEEPSEEK_V2_LITE_CONFIG["head_dim"]

    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        ids, mask = make_inputs(tok, bs, args.seq_len, device, vocab)
        print(f"\n  batch={bs}  seq_len={args.seq_len}  new_tokens={args.new_tokens}")

        print(f"  Warming up ({args.warmup} runs)...")
        for _ in range(args.warmup):
            with torch.no_grad():
                model.generate(
                    ids, attention_mask=mask,
                    max_new_tokens=args.new_tokens,
                    do_sample=False, use_cache=True,
                )
        _sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print(f"  Benchmarking ({args.iters} runs)...")
        latencies = []
        kv_bytes  = 0

        for _ in range(args.iters):
            _sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    ids, attention_mask=mask,
                    max_new_tokens=args.new_tokens,
                    do_sample=False, use_cache=True,
                    return_dict_in_generate=True,
                )
            _sync()
            latencies.append(time.perf_counter() - t0)

        if hasattr(out, "past_key_values") and out.past_key_values is not None:
            try:
                for layer_kv in out.past_key_values:
                    for t in layer_kv:
                        kv_bytes += t.element_size() * t.nelement()
            except Exception:
                pass

        mean_s    = sum(latencies) / len(latencies)
        n_new     = args.new_tokens
        peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        theory    = compute_kv_storage(args.seq_len, L, H, D)

        row = {
            "batch_size":             bs,
            "seq_len":                args.seq_len,
            "new_tokens":             n_new,
            "latency_ms":             round(mean_s * 1e3, 2),
            "tokens_per_sec":         round((bs * n_new) / mean_s, 1),
            "kv_cache_measured_MB":   round(kv_bytes / 1e6, 2),
            "kv_standard_theory_MB":  round(theory["standard_MB"], 2),
            "kv_fused_theory_MB":     round(theory["fused_MB"], 2),
            "kv_savings_theory_pct":  round(theory["savings_pct"], 1),
            "peak_vram_gb":           round(peak_vram, 2),
        }
        results.append(row)

        print(f"  latency     : {row['latency_ms']} ms")
        print(f"  tokens/sec  : {row['tokens_per_sec']}")
        print(f"  KV measured : {row['kv_cache_measured_MB']} MB")
        print(f"  KV fused    : {row['kv_fused_theory_MB']} MB  (-{row['kv_savings_theory_pct']}%)")
        print(f"  peak VRAM   : {row['peak_vram_gb']} GB")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path",  default="deepseek-ai/DeepSeek-V2-Lite")
    ap.add_argument("--fusion",      default="off", choices=["on", "off"])
    ap.add_argument("--batch-sizes", default="1")
    ap.add_argument("--seq-len",     type=int, default=131_072)
    ap.add_argument("--new-tokens",  type=int, default=128)
    ap.add_argument("--warmup",      type=int, default=2)
    ap.add_argument("--iters",       type=int, default=5)
    ap.add_argument("--dtype",       default="bf16",
                    choices=["bf16", "fp32", "fp16"])
    ap.add_argument("--out",         default="results/e2e_baseline.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.dtype]

    print(f"\n{'='*65}")
    print(f"  DeepSeek-V2-Lite KV Cache Benchmark")
    print(f"  model  : {args.model_path}")
    print(f"  fusion : {args.fusion}")
    print(f"  seq_len: {args.seq_len}")
    print(f"{'='*65}")

    model, tok = load_model(args.model_path, dtype)
    results    = run_benchmark(model, tok, args, dtype)

    output = {
        "model":    args.model_path,
        "fusion":   args.fusion,
        "dtype":    args.dtype,
        "seq_len":  args.seq_len,
        "warmup":   args.warmup,
        "iters":    args.iters,
        "gpus":     _gpu_info(),
        "results":  results,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[bench] Results written to {args.out}")


if __name__ == "__main__":
    main()
