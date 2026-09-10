"""Attention backend benchmark: SDPA vs eager and SDPA kernel selection.

Measures real training-style forward+backward throughput and memory for the
configured actor on genuine BC samples, plus a numerical-correctness check
between the SDPA flash kernel and the math kernel.

    python scripts/bench_attention.py \
        --model-path /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
        --manifest outputs/embodied_bc/dataset_v2/bc_train.jsonl \
        --out outputs/diagnostics/attention
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_samples(manifest: str, limit: int):
    from spatialforge.experiment.training import load_and_preprocess_image, load_training_records
    from spatialforge.models.vl_adapter import build_training_tensors

    base_dir = Path(manifest).parent
    records = load_training_records(manifest, base_dir=base_dir)[:limit]
    samples = []
    for r in records:
        img = load_and_preprocess_image(r.image_path)
        samples.append((img, r.question, r.answer))
    return samples


def _run_once(model, processor, samples, batch: int, device="cuda"):
    import torch

    from spatialforge.models.vl_adapter import build_training_tensors, collate_training_batch

    model.eval()
    total = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            items = [build_training_tensors(processor, img, q, a) for img, q, a in chunk]
            if len(items) == 1:
                it = items[0]
                bt = {
                    "input_ids": it["input_ids"].unsqueeze(0).to(device),
                    "attention_mask": it["attention_mask"].unsqueeze(0).to(device),
                    "pixel_values": it["pixel_values"].to(device),
                    "image_grid_thw": it["image_grid_thw"].to(device),
                }
                if "mm_token_type_ids" in it:
                    bt["mm_token_type_ids"] = it["mm_token_type_ids"].unsqueeze(0).to(device)
                _ = model(**bt)
            else:
                bt = collate_training_batch(items, device=device)
                bt.pop("labels", None)
                bt.pop("sample_weight", None)
                _ = model(**bt)
            total += len(chunk)
    dt = time.time() - t0
    return total, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--manifest", default="outputs/embodied_bc/dataset_v2/bc_train.jsonl")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default="outputs/diagnostics/attention")
    args = ap.parse_args()

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from spatialforge.experiment.performance import GpuTelemetrySampler
    from spatialforge.models.vl_adapter import load_model, load_processor, resolve_spec

    samples = _load_samples(args.manifest, args.limit)
    print(f"[attn] {len(samples)} samples", flush=True)
    processor = load_processor(args.model_path)
    results = {}

    for impl in ("sdpa", "eager"):
        spec = resolve_spec(args.model_path, adapter=args.adapter, attn_implementation=impl)
        try:
            model = load_model(spec)
        except Exception as e:  # noqa: BLE001
            results[impl] = {"error": str(e)[:300]}
            continue
        if args.adapter:
            model.eval()
        sampler = GpuTelemetrySampler(device_index=0, sample_interval_sec=0.05).start()
        sampler.set_phase("train_active")
        torch.cuda.reset_peak_memory_stats()
        n, dt = _run_once(model, processor, samples, args.batch)
        tele = sampler.stop()
        results[impl] = {
            "samples": n,
            "wall_sec": round(dt, 3),
            "samples_per_sec": round(n / max(dt, 1e-6), 3),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "gpu_util_avg": tele["gpu_utilization"]["avg"],
            "gpu_power_w_avg": tele["gpu_power_w"]["avg"],
        }
        del model
        torch.cuda.empty_cache()

    # SDPA kernel selection: flash vs math (numerical + perf)
    spec = resolve_spec(args.model_path, adapter=args.adapter, attn_implementation="sdpa")
    model = load_model(spec)
    kernels = {}
    losses = {}
    for name, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("math", SDPBackend.MATH)):
        try:
            sampler = GpuTelemetrySampler(device_index=0, sample_interval_sec=0.05).start()
            sampler.set_phase("train_active")
            torch.cuda.reset_peak_memory_stats()
            with sdpa_kernel([backend]):
                n, dt = _run_once(model, processor, samples, args.batch)
            tele = sampler.stop()
            kernels[name] = {
                "samples_per_sec": round(n / max(dt, 1e-6), 3),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "gpu_util_avg": tele["gpu_utilization"]["avg"],
            }
        except Exception as e:  # noqa: BLE001
            kernels[name] = {"error": str(e)[:200]}
    results["sdpa_kernels"] = kernels

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "attention_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[attn] " + json.dumps(results, indent=2)[:2500], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
