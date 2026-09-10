"""Generate one post-fix, Medium-quality, spawn-semantics-clean episode.

This is the reproducible entry point used to produce the episode the Inspector
loads by default (real Unity God View + clear FPV + truthful trajectory).

    DISPLAY=:0 SF_RENDER_QUALITY=Medium \
        /root/autodl-tmp/.venv-procthor/bin/python \
        scripts/generate_postfix_episode.py \
        --house /root/autodl-tmp/val_house_0.json \
        --out outputs/episodes_postfix --category mug

Requires the Python-3.10 ProcTHOR venv + an NVIDIA-backed X display.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialforge.embodied.rollout import ProcthorEpisodeGenerator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", default="/root/autodl-tmp/val_house_0.json")
    ap.add_argument("--out", default="outputs/episodes_postfix")
    ap.add_argument("--category", default="mug")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--x-display", default=os.environ.get("DISPLAY") or ":0")
    ap.add_argument("--require-nvidia", action="store_true")
    args = ap.parse_args()

    gen = ProcthorEpisodeGenerator(
        args.house, width=args.width, height=args.height,
        x_display=args.x_display, out_dir=args.out, seed=args.seed,
        require_nvidia=args.require_nvidia,
    )
    try:
        r = gen.generate_episode(args.category, max_steps=args.max_steps, allow_easy=False)
        rec = r["privileged_record"]
        report = {
            "episode_id": r["episode_id"],
            "status": r["status"],
            "success": r["success"],
            "steps": r["steps"],
            "render_quality": rec.get("render_quality"),
            "house_path": rec.get("house_path"),
            "setup_initial_pose": (rec.get("setup") or {}).get("initial_pose"),
            "first_step_pose": (rec.get("steps") or [{}])[0].get("agent"),
        }
        print(json.dumps(report, indent=2, default=str))
        return 0
    finally:
        gen.close()


if __name__ == "__main__":
    sys.exit(main())
