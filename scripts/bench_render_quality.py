"""Benchmark AI2-THOR render quality: FPV exposure statistics + step latency.

For each quality setting this samples a fixed set of reachable poses (render on)
and measures luminance saturation plus genuine action latency, so the sensor
fidelity fix can be chosen on evidence (lowest visually-normal quality).

Run with the ai2thor venv and NVIDIA X display:
    python scripts/bench_render_quality.py --house /root/autodl-tmp/val_house_0.json \
        --out outputs/diagnostics/render_quality --x-display :0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from spatialforge.embodied.procthor_teacher import build_reachable_grid  # noqa: E402

QUALITIES = ("Low", "Medium", "High")


def frame_stats(rgb: np.ndarray) -> dict:
    arr = rgb.astype(np.float64)
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    flat = lum.reshape(-1)
    return {
        "mean": round(float(flat.mean()), 2),
        "median": round(float(np.median(flat)), 2),
        "p95": round(float(np.percentile(flat, 95)), 2),
        "p99": round(float(np.percentile(flat, 99)), 2),
        "frac_ge_250": round(float((flat >= 250).mean()), 5),
        "frac_ge_245": round(float((flat >= 245).mean()), 5),
        "frac_ge_240": round(float((flat >= 240).mean()), 5),
    }


def bench_quality(house, quality, out_dir, n_poses, width, x_display) -> dict:
    from spatialforge.embodied.backends.procthor import ProcTHORBackend

    backend = ProcTHORBackend(
        house=house, width=width, height=width, quality=quality,
        x_display=x_display, require_nvidia=True,
    )
    try:
        backend.load_house()
        grid = build_reachable_grid(backend.reachable_positions())
        cells = sorted(grid.keys())
        if not cells:
            return {"quality": quality, "error": "no reachable positions"}
        step = max(1, len(cells) // max(n_poses, 1))
        chosen = cells[::step][:n_poses]

        samples = []
        teleport_ms = []
        for i, cell in enumerate(chosen):
            y = grid[cell]
            yaw = float((i * 90) % 360)
            t0 = time.perf_counter()
            md = backend.set_agent_pose(
                (cell[0] * 0.25, y, cell[1] * 0.25), rotation_yaw_deg=yaw,
                horizon_deg=0.0, render=True,
            )
            teleport_ms.append((time.perf_counter() - t0) * 1000.0)
            raw = np.asarray(backend._event.frame)
            s = frame_stats(raw)
            s["yaw"] = yaw
            s["cell"] = list(cell)
            samples.append(s)

        # genuine action latency (RotateLeft x 10) on one pose
        act_ms = []
        for _ in range(10):
            t0 = time.perf_counter()
            backend.apply_action(
                __import__("spatialforge.embodied.contracts", fromlist=["AgentActionType"]).AgentActionType.ROTATE_LEFT,
                {}, 0,
            )
            act_ms.append((time.perf_counter() - t0) * 1000.0)

        agg = {
            "quality": quality,
            "renderer": backend.LABEL,
            "n_poses": len(samples),
            "samples": samples,
            "mean_of_mean": round(float(np.mean([s["mean"] for s in samples])), 2),
            "mean_of_median": round(float(np.mean([s["median"] for s in samples])), 2),
            "max_frac_ge_250": round(float(np.max([s["frac_ge_250"] for s in samples])), 5),
            "mean_frac_ge_250": round(float(np.mean([s["frac_ge_250"] for s in samples])), 5),
            "teleport_ms": {
                "mean": round(float(np.mean(teleport_ms)), 1),
                "p95": round(float(np.percentile(teleport_ms, 95)), 1),
            },
            "rotate_step_ms": {
                "mean": round(float(np.mean(act_ms)), 1),
                "p95": round(float(np.percentile(act_ms, 95)), 1),
            },
        }
        return agg
    finally:
        backend.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", default="/root/autodl-tmp/val_house_0.json")
    ap.add_argument("--out", default="outputs/diagnostics/render_quality")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--x-display", default=":0")
    ap.add_argument("--poses", type=int, default=12)
    ap.add_argument("--qualities", nargs="*", default=list(QUALITIES))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = []
    for q in args.qualities:
        print(f"[render_quality] quality={q} ...", flush=True)
        r = bench_quality(args.house, q, args.out, args.poses, args.width, args.x_display)
        results.append(r)
        print(f"[render_quality] {q}: mean_of_mean={r.get('mean_of_mean')} "
              f"max_frac_ge_250={r.get('max_frac_ge_250')} "
              f"rotate_ms={r.get('rotate_step_ms')}", flush=True)

    report = {
        "schema": "render_quality_benchmark.v1",
        "house": os.path.abspath(args.house),
        "resolution": f"{args.width}x{args.width}",
        "results": results,
    }
    with open(os.path.join(args.out, "render_quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("RENDER_QUALITY_REPORT " + json.dumps(report)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
