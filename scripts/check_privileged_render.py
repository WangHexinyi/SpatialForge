"""Privileged render equivalence: renderImage=True vs False.

Spawn/view-pose probing uses privileged TeleportFull. If the authoritative
object ``visible`` metadata is identical with and without RGB rendering, then
privileged planning can run renderImage=False (no wasted pixels), while genuine
student observations and God View frames keep rendering.

    python scripts/check_privileged_render.py \
        --house /root/autodl-tmp/val_house_0.json --poses 20 --x-display :0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialforge.embodied.procthor_teacher import build_reachable_grid  # noqa: E402


def _visible_map(md: dict) -> dict:
    return {
        str(o.get("objectId")): bool(o.get("visible", False))
        for o in md.get("objects", []) if o.get("objectId")
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", default="/root/autodl-tmp/val_house_0.json")
    ap.add_argument("--poses", type=int, default=20)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--x-display", default=":0")
    ap.add_argument("--out", default="outputs/diagnostics/privileged_render")
    args = ap.parse_args()

    from spatialforge.embodied.backends.procthor import ProcTHORBackend

    backend = ProcTHORBackend(
        house=args.house, width=args.width, height=args.width,
        x_display=args.x_display, require_nvidia=True,
    )
    try:
        backend.load_house()
        grid = build_reachable_grid(backend.reachable_positions())
        cells = sorted(grid.keys())
        step = max(1, len(cells) // max(args.poses, 1))
        chosen = cells[::step][: args.poses]

        mismatches = 0
        checked_objects = 0
        t_render = t_norender = 0.0
        per_pose = []
        for i, cell in enumerate(chosen):
            y = grid[cell]
            yaw = float((i * 45) % 360)
            t0 = time.perf_counter()
            md_on = backend.set_agent_pose(
                (cell[0] * 0.25, y, cell[1] * 0.25), yaw, 0.0, render=True)
            t_on = time.perf_counter() - t0
            t0 = time.perf_counter()
            md_off = backend.set_agent_pose(
                (cell[0] * 0.25, y, cell[1] * 0.25), yaw, 0.0, render=False)
            t_off = time.perf_counter() - t0
            t_render += t_on
            t_norender += t_off
            v_on, v_off = _visible_map(md_on), _visible_map(md_off)
            keys = set(v_on) & set(v_off)
            diff = [k for k in keys if v_on[k] != v_off[k]]
            checked_objects += len(keys)
            if diff:
                mismatches += 1
            per_pose.append({
                "cell": list(cell), "yaw": yaw,
                "objects": len(keys), "visibility_mismatches": len(diff),
                "render_ms": round(t_on * 1000, 1),
                "norender_ms": round(t_off * 1000, 1),
            })

        report = {
            "schema": "privileged_render_equivalence.v1",
            "house": os.path.abspath(args.house),
            "poses": len(chosen),
            "objects_compared": checked_objects,
            "poses_with_mismatch": mismatches,
            "visibility_equivalent": mismatches == 0,
            "timing_ms": {
                "render_total": round(t_render * 1000, 1),
                "norender_total": round(t_norender * 1000, 1),
                "speedup": round(t_render / max(t_norender, 1e-9), 2),
            },
            "per_pose": per_pose,
        }
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "privileged_render_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("[render_equiv] " + json.dumps(
            {k: report[k] for k in (
                "poses", "objects_compared", "poses_with_mismatch",
                "visibility_equivalent", "timing_ms")}, indent=2))
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    sys.exit(main())
