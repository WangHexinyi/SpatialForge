"""Embodied BC rollout worker (one process = one AI2-THOR runtime owner).

Runs under the Python-3.10 ai2thor venv with the NVIDIA X display set. Each
worker owns real ProcTHOR/AI2-THOR environments (one backend per distinct
house, reused across jobs of that house) and executes a deterministic job list
seeded per job. Appends compact JSONL result rows and (in persist mode) episode
records + first-person frames under the dataset root. On a per-job failure it
records the crash and -- when the backend is dead -- restarts it for the next
job, so one broken episode never sinks a shard.

Invocation:
    python scripts/embodied_bc_worker.py \
        --jobs <shard jobs json> --slot <int> \
        [--persist --out-root <root>] [--width 256 --height 256]
        [--x-display :0] [--require-nvidia] [--no-nvidia]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialforge.embodied.rollout import ProcthorEpisodeGenerator  # noqa: E402


def _write_result(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _heartbeat(hb_dir: str, slot: int, state: Dict[str, Any]) -> None:
    os.makedirs(hb_dir, exist_ok=True)
    with open(os.path.join(hb_dir, f"slot-{slot:02d}.json"), "w", encoding="utf-8") as f:
        json.dump({"slot": slot, "pid": os.getpid(), "ts": time.time(), **state}, f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="JSON file: {'jobs':[{house, category, seed, max_steps, job_id}]}")
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--results", required=True, help="JSONL file appended per job")
    ap.add_argument("--hb-dir", required=True)
    ap.add_argument("--persist", action="store_true")
    ap.add_argument("--out-root", default=None, help="episode artifacts root (persist mode)")
    ap.add_argument("--width", type=int, default=256)
    ap.set_defaults(require_nvidia=True)
    ap.add_argument("--require-nvidia", dest="require_nvidia", action="store_true",
                    help="fail loudly if the X display is not an NVIDIA renderer")
    ap.add_argument("--no-nvidia", dest="require_nvidia", action="store_false")
    ap.add_argument("--x-display", default=":0")
    ap.add_argument("--tag", default="bc")
    args = ap.parse_args()

    with open(args.jobs, "r", encoding="utf-8") as f:
        jobs: List[Dict[str, Any]] = json.load(f)["jobs"]
    if not jobs:
        return 0

    _heartbeat(args.hb_dir, args.slot, {"state": "starting", "job_id": None})
    gens: Dict[str, ProcthorEpisodeGenerator] = {}
    closed = False
    try:
        for job in jobs:
            house_path = job["house"]
            job_id = job["job_id"]
            _heartbeat(args.hb_dir, args.slot, {"state": "running", "job_id": job_id})
            t0 = time.time()
            gen = gens.get(house_path)
            if gen is None:
                gen = ProcthorEpisodeGenerator(
                    house_path,
                    width=args.width, height=args.width,
                    x_display=args.x_display,
                    out_dir=args.out_root if args.persist else None,
                    seed=0,
                    require_nvidia=args.require_nvidia,
                )
                gens[house_path] = gen
            try:
                r = gen.generate_episode(
                    job["category"],
                    max_steps=int(job.get("max_steps", 150)),
                    allow_easy=False,
                    tag=args.tag,
                    seed=int(job.get("seed", 0)),
                )
                steps_times = list(r.get("env_step_times_ms") or [])
                row = {
                    "job_id": job_id,
                    "slot": args.slot,
                    "house": house_path,
                    "house_id": r.get("house_id"),
                    "category": job["category"],
                    "status": r.get("status"),
                    "success": bool(r.get("success")),
                    "outcome_code": r.get("outcome_code") or (
                        "success" if r.get("success") else r.get("reason", "unknown")),
                    "steps": int(r.get("steps", 0)),
                    "elapsed_s": float(r.get("elapsed_s", time.time() - t0)),
                    "env_step_times": [round(x, 1) for x in steps_times],
                    "reason": r.get("reason"),
                    "initially_visible": r.get("initially_visible"),
                    "resample_attempts": r.get("spawn_resample_attempts"),
                    "episode_id": r.get("episode_id"),
                    "crashed": False,
                    "error": None,
                }
                _write_result(args.results, row)
            except Exception:
                exc = traceback.format_exc(limit=5)
                _write_result(args.results, {
                    "job_id": job_id, "slot": args.slot,
                    "house": house_path, "category": job["category"],
                    "status": "crashed", "success": False, "outcome_code": "worker_crash",
                    "steps": 0, "elapsed_s": round(time.time() - t0, 1),
                    "env_step_times": [],
                    "reason": exc[-600:], "crashed": True,
                })
                # backend may be dead -> drop it so the next job restarts cleanly
                try:
                    gen.close()
                except Exception:
                    pass
                gens.pop(house_path, None)
                _heartbeat(args.hb_dir, args.slot, {"state": "recovered_after_crash", "job_id": job_id})
            _heartbeat(args.hb_dir, args.slot, {"state": "job_done", "job_id": job_id})
        closed = True
    finally:
        for gen in gens.values():
            try:
                gen.close()
            except Exception:
                pass
        gens.clear()
        if not closed:
            _heartbeat(args.hb_dir, args.slot, {"state": "died", "job_id": None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
