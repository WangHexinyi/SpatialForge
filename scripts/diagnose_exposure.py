"""FPV exposure diagnostic (scientific sensor-fidelity check).

Renders one fixed house / pose / yaw / horizon under several AI2-THOR quality
settings and measures raw-frame luminance statistics, so over-exposure can be
localized to (A) the engine frame, (B) the persisted PNG, or (C) display only.

Run with the ai2thor venv and an NVIDIA X display:
    python scripts/diagnose_exposure.py \
        --house /root/autodl-tmp/val_house_0.json \
        --out outputs/diagnostics/exposure --width 256 --x-display :0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

QUALITIES = ("Low", "Medium", "High")


def luminance_stats(rgb: np.ndarray) -> dict:
    """Luminance + channel statistics of an HxWx3 uint8 frame."""
    arr = rgb.astype(np.float64)
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    flat = lum.reshape(-1)
    stats = {
        "shape": list(rgb.shape),
        "mean": round(float(flat.mean()), 3),
        "median": round(float(np.median(flat)), 3),
        "p05": round(float(np.percentile(flat, 5)), 3),
        "p50": round(float(np.percentile(flat, 50)), 3),
        "p95": round(float(np.percentile(flat, 95)), 3),
        "p99": round(float(np.percentile(flat, 99)), 3),
        "max": round(float(flat.max()), 3),
        "frac_ge_250": round(float((flat >= 250).mean()), 6),
        "frac_ge_245": round(float((flat >= 245).mean()), 6),
        "frac_ge_240": round(float((flat >= 240).mean()), 6),
        "frac_le_5": round(float((flat <= 5).mean()), 6),
        "mean_rgb": [round(float(arr[..., c].mean()), 3) for c in range(3)],
    }
    for c, name in enumerate("rgb"):
        ch = arr[..., c].reshape(-1)
        hist, edges = np.histogram(ch, bins=16, range=(0, 256))
        stats[f"hist_{name}"] = [int(v) for v in hist]
    stats["hist_edges"] = [int(v) for v in edges]
    return stats


def render_quality(house: str, quality: str, width: int, height: int,
                   x_display: str, out_dir: str) -> dict:
    from spatialforge.embodied.backends.procthor import ProcTHORBackend

    backend = ProcTHORBackend(
        house=house, width=width, height=height, quality=quality,
        x_display=x_display, require_nvidia=True,
    )
    try:
        info = backend.load_house()
        event = backend._event
        raw = np.asarray(event.frame)
        raw_png = os.path.join(out_dir, f"raw_{quality.lower()}.png")
        Image.fromarray(raw.astype(np.uint8)).save(raw_png)

        # persisted-PNG round trip (what training/God View actually read)
        reloaded = np.asarray(Image.open(raw_png).convert("RGB"))
        persist_stats = luminance_stats(reloaded)

        agent = event.metadata.get("agent", {})
        return {
            "quality": quality,
            "house_id": info.house_id,
            "resolution": f"{width}x{height}",
            "renderer": info.rendering_backend,
            "agent_pose": {
                "position": agent.get("position"),
                "rotation": agent.get("rotation"),
                "cameraHorizon": agent.get("cameraHorizon"),
            },
            "raw_frame": luminance_stats(raw),
            "persisted_png": persist_stats,
            "raw_png": os.path.relpath(raw_png, out_dir),
            "bitwise_identical_raw_vs_png": bool(np.array_equal(raw, reloaded)),
        }
    finally:
        backend.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", default="/root/autodl-tmp/val_house_0.json")
    ap.add_argument("--out", default="outputs/diagnostics/exposure")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--x-display", default=":0")
    ap.add_argument("--qualities", nargs="*", default=list(QUALITIES))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = []
    for q in args.qualities:
        print(f"[exposure] rendering quality={q} ...", flush=True)
        results.append(render_quality(
            args.house, q, args.width, args.height, args.x_display, args.out,
        ))

    report = {
        "schema": "fpv_exposure_diagnostic.v1",
        "house": os.path.abspath(args.house),
        "results": results,
    }
    with open(os.path.join(args.out, "exposure_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("EXPOSURE_REPORT " + json.dumps(
        [{k: r[k] for k in ("quality", "raw_frame", "persisted_png",
                            "bitwise_identical_raw_vs_png")} for r in results]
    )[:6000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
