"""External actor model pre-flight (single image, strict action inference).

Loads the configured external model (base or + adapter) with the thin adapter,
runs the real BC question on one held-out frame, enforces the strict action
parser, and reports VRAM / latency / tokens / attention backend. No training.

    python scripts/model_preflight.py \
        --model-path /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
        --image outputs/embodied_bc/dataset_v2/<ep>/frames/frame_0000.png \
        --out outputs/embodied_bc/preflight --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialforge.embodied.action_parser import parse_action_output  # noqa: E402
from spatialforge.embodied.bc_dataset import build_bc_question  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--image", default=None)
    ap.add_argument("--question", default=None)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="outputs/embodied_bc/preflight")
    args = ap.parse_args()

    import torch

    from spatialforge.experiment.training import load_and_preprocess_image, run_inference_greedy
    from spatialforge.models.vl_adapter import load_model, load_processor, resolve_spec

    image_path = args.image
    if image_path is None:
        candidates = sorted(Path("outputs/embodied_bc").glob("**/frames/frame_0000.png"))
        if not candidates:
            print("no image found; pass --image")
            return 1
        image_path = str(candidates[0])
    question = args.question or build_bc_question(
        "Find a mug.", "mug", 0.0, [], max_steps=100, current_step=0,
    )

    spec = resolve_spec(args.model_path, adapter=args.adapter,
                        attn_implementation=args.attn)
    print(f"[preflight] spec={json.dumps(spec.to_dict())}", flush=True)
    t0 = time.time()
    processor = load_processor(args.model_path)
    model = load_model(spec)
    model.eval()
    load_s = time.time() - t0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    image = load_and_preprocess_image(image_path)

    runs = []
    for i in range(max(1, args.runs)):
        t = time.time()
        raw = run_inference_greedy(model, processor, image, question,
                                   max_new_tokens=12, device="cuda")
        latency = time.time() - t
        parsed = parse_action_output(raw)
        runs.append({"raw": raw, "parsed": parsed, "latency_ms": round(latency * 1000.0, 1)})
        print(f"[preflight] run {i}: {parsed['action']} ({parsed['reason']}) "
              f"{latency*1000:.0f}ms raw={raw!r}", flush=True)

    lat = sorted(r["latency_ms"] for r in runs)
    n = len(lat)
    report = {
        "spec": spec.to_dict(),
        "image": os.path.abspath(image_path),
        "question": question,
        "load_seconds": round(load_s, 1),
        "runs": runs,
        "latency_ms": {
            "mean": round(sum(lat) / n, 1),
            "p50": lat[int(0.5 * (n - 1))],
            "p95": lat[int(0.95 * (n - 1))],
        },
        "valid_action_rate": round(
            sum(1 for r in runs if r["parsed"]["action"]) * 100.0 / n, 2),
        "vram": {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2) if torch.cuda.is_available() else 0,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "attn_requested": args.attn,
        "attn_actual": str(getattr(getattr(model, "config", None), "_attn_implementation", None)
                           or getattr(getattr(model, "config", None), "attn_implementation", None)),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "preflight_" + (spec.family or "model") + ("_adapter" if args.adapter else "")
    with open(out / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[preflight] " + json.dumps(report, indent=2)[:2500], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
