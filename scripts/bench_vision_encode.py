"""Batched vision-encoder throughput benchmark (serial vs batch 4..64).

Measures images/sec, peak VRAM, GPU utilization and power for the frozen
vision tower at increasing batch sizes, until OOM or throughput plateau. The
cache contract is unchanged: features are keyed by real image bytes.

    python scripts/bench_vision_encode.py \
        --model-path /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
        --image-dir outputs/embodied_bc/dataset_v2 --batches 1 4 8 16 32 64 \
        --out outputs/diagnostics/vision_encode
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--image-dir", default="outputs/embodied_bc/dataset_v2")
    ap.add_argument("--num-images", type=int, default=64)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64])
    ap.add_argument("--out", default="outputs/diagnostics/vision_encode")
    args = ap.parse_args()

    import torch

    from spatialforge.experiment.performance import GpuTelemetrySampler
    from spatialforge.experiment.training import load_and_preprocess_image
    from spatialforge.models.vl_adapter import build_training_tensors, load_model, load_processor, resolve_spec

    spec = resolve_spec(args.model_path, adapter=args.adapter)
    processor = load_processor(args.model_path)
    model = load_model(spec)
    model.eval()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model

    image_paths = sorted(Path(args.image_dir).glob("**/frames/frame_0000.png"))[: args.num_images]
    if not image_paths:
        print("[vision_bench] no images found")
        return 1
    print(f"[vision_bench] {len(image_paths)} images, family={spec.family}", flush=True)

    tensors = []
    for p in image_paths:
        img = load_and_preprocess_image(str(p))
        t = build_training_tensors(processor, img, "", "")
        tensors.append((t["pixel_values"], t["image_grid_thw"]))

    results = []
    for b in args.batches:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            sampler = GpuTelemetrySampler(device_index=0, sample_interval_sec=0.05).start()
            sampler.set_phase("vision_encode")
            t0 = time.time()
            n_done = 0
            with torch.no_grad():
                for i in range(0, len(tensors), b):
                    chunk = tensors[i:i + b]
                    pv = torch.cat([c[0] for c in chunk], dim=0).to("cuda")
                    grid = torch.cat([c[1] for c in chunk], dim=0).to("cuda")
                    if pv.dtype == torch.float32:
                        pv = pv.to(torch.bfloat16)
                    out = base.model.get_image_features(pv, grid)
                    _ = out.pooler_output
                    n_done += len(chunk)
            dt = time.time() - t0
            tele = sampler.stop()
            peak = torch.cuda.max_memory_allocated() / 1e9
            results.append({
                "batch": b,
                "images": n_done,
                "wall_sec": round(dt, 3),
                "images_per_sec": round(n_done / max(dt, 1e-6), 2),
                "peak_vram_gb": round(peak, 2),
                "gpu_util_avg": tele["gpu_utilization"]["avg"],
                "gpu_power_w_avg": tele["gpu_power_w"]["avg"],
            })
            print(f"[vision_bench] batch={b} img/s={results[-1]['images_per_sec']} "
                  f"peak={peak:.1f}GB", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results.append({"batch": b, "error": "CUDA OOM"})
                print(f"[vision_bench] batch={b} OOM", flush=True)
                torch.cuda.empty_cache()
                break
            raise

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "vision_encode_benchmark.v1",
        "spec": spec.to_dict(),
        "num_images": len(image_paths),
        "results": results,
    }
    with open(out / "vision_encode_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[vision_bench] " + json.dumps(report)[:2000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
