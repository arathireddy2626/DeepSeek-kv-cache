"""
build_report.py
===============
Reads e2e_baseline.json and e2e_fused.json, prints comparison.

Usage:
    python scripts/build_report.py \
        --baseline results/e2e_baseline.json \
        --fused    results/e2e_fused.json \
        --out      results/report.txt
"""

import argparse
import json
import os
from datetime import datetime


def _load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/e2e_baseline.json")
    ap.add_argument("--fused",    default="results/e2e_fused.json")
    ap.add_argument("--out",      default="results/report.txt")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    baseline = _load(args.baseline)
    fused    = _load(args.fused)

    b_map = {r["batch_size"]: r for r in baseline["results"]}
    f_map = {r["batch_size"]: r for r in fused["results"]}

    lines = []
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines += [
        "=" * 65,
        f"  DeepSeek-V2-Lite KV Cache Fusion Report  —  {ts}",
        f"  model   : {baseline.get('model')}",
        f"  seq_len : {baseline.get('seq_len')}",
        f"  dtype   : {baseline.get('dtype')}",
        f"  GPUs    : {[g['name'] for g in baseline.get('gpus', [])]}",
        "=" * 65,
    ]

    lines.append(
        f"\n  {'bs':<4}  {'base_lat':>10}  {'fused_lat':>10}  "
        f"{'lat_speedup':>12}  {'base_kv':>10}  {'fused_kv':>10}  {'kv_saved':>9}"
    )
    lines.append("  " + "-" * 72)

    lat_speedups, kv_savings = [], []

    for bs in sorted(set(list(b_map.keys()) + list(f_map.keys()))):
        b = b_map.get(bs, {})
        f = f_map.get(bs, {})

        b_lat = b.get("latency_ms", 0)
        f_lat = f.get("latency_ms", 0)
        b_kv  = b.get("kv_cache_measured_MB", 0)
        f_kv  = f.get("kv_cache_measured_MB", 0)

        lat_sp = round(b_lat / f_lat, 3) if f_lat else 0
        kv_sv  = round(100 * (b_kv - f_kv) / b_kv, 1) if b_kv else 0

        lat_speedups.append(lat_sp)
        kv_savings.append(kv_sv)

        lines.append(
            f"  {bs:<4}  {b_lat:>8.1f}ms  {f_lat:>8.1f}ms  "
            f"{lat_sp:>10.3f}x  {b_kv:>8.1f}MB  {f_kv:>8.1f}MB  {kv_sv:>7.1f}%"
        )

    if lat_speedups:
        avg_lat = sum(lat_speedups) / len(lat_speedups)
        avg_kv  = sum(kv_savings)  / len(kv_savings)
        lines += [
            "",
            f"  avg latency speedup  :  {avg_lat:.3f}x",
            f"  avg KV memory saved  :  {avg_kv:.1f}%",
        ]

    lines.append("\n" + "=" * 65 + "\n")

    report = "\n".join(lines)
    print(report)

    with open(args.out, "w") as f:
        f.write(report)
    print(f"[report] Written to {args.out}")


if __name__ == "__main__":
    main()
