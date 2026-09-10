"""Model-controlled rollout worker (one process = one real agent runtime).

Same isolation guarantees as embodied_bc_worker but drives episodes with the
*model's own* actions (scripts/model_action_server.py) instead of the teacher.

    python scripts/embodied_model_worker.py --jobs <json> --slot N \
        --server-url http://127.0.0.1:8765 --out-root <root> --width 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialforge.embodied.model_rollout import (  # noqa: E402
    ModelActionClient,
    ProcthorModelEpisodeGenerator,
)


def _write_result(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--hb-dir", required=True)
    ap.add_argument("--server-url", required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--x-display", default=":0")
    ap.set_defaults(require_nvidia=True)
    ap.add_argument("--require-nvidia", dest="require_nvidia", action="store_true")
    ap.add_argument("--no-nvidia", dest="require_nvidia", action="store_false")
    ap.add_argument("--persist", action="store_true",
                    help="persist episodes under --out-root (always on in model mode)")
    ap.add_argument("--max-consecutive-invalid", type=int, default=5)
    ap.add_argument("--setup-timeout", type=float, default=150.0)
    args = ap.parse_args()

    with open(args.jobs, "r", encoding="utf-8") as f:
        jobs = json.load(f)["jobs"]

    client = ModelActionClient(args.server_url)
    gens = {}
    try:
        for job in jobs:
            house = job["house"]
            t0 = time.time()
            hb = os.path.join(args.hb_dir, f"slot-{args.slot:02d}.json")
            os.makedirs(args.hb_dir, exist_ok=True)
            with open(hb, "w") as f:
                json.dump({"slot": args.slot, "pid": os.getpid(),
                           "state": "running", "job_id": job["job_id"]}, f)
            gen = gens.get(house)
            if gen is None:
                gen = ProcthorModelEpisodeGenerator(
                    house, width=args.width, height=args.width,
                    x_display=args.x_display,
                    out_dir=args.out_root, seed=0,
                    require_nvidia=args.require_nvidia,
                )
                gens[house] = gen
            try:
                r = gen.generate_episode(
                    job["category"], client,
                    max_steps=int(job.get("max_steps", 120)),
                    tag="model", seed=int(job.get("seed", 0)),
                    max_consecutive_invalid=args.max_consecutive_invalid,
                    setup_timeout_s=args.setup_timeout,
                )
                row = {
                    "job_id": job["job_id"], "slot": args.slot,
                    "house": house,
                    "house_id": r.get("house_id"),
                    "category": job["category"],
                    "status": r.get("status"),
                    "success": bool(r.get("success")),
                    "outcome_code": r.get("reason") or r.get("status"),
                    "steps": int(r.get("steps", 0)),
                    "elapsed_s": float(r.get("elapsed_s", time.time() - t0)),
                    "decision_count": r.get("decision_count", 0),
                    "invalid_decision_count": r.get("invalid_decision_count", 0),
                    "episode_id": r.get("episode_id"),
                    "crashed": False,
                    "error": None,
                }
                _write_result(args.results, row)
            except Exception:
                exc = traceback.format_exc(limit=5)
                _write_result(args.results, {
                    "job_id": job["job_id"], "slot": args.slot, "house": house,
                    "category": job["category"], "status": "crashed",
                    "success": False, "outcome_code": "worker_crash",
                    "steps": 0, "elapsed_s": round(time.time() - t0, 1),
                    "decision_count": 0, "invalid_decision_count": 0,
                    "crashed": True, "error": exc[-600:],
                })
                try:
                    gen.close()
                except Exception:
                    pass
                gens.pop(house, None)
    finally:
        for gen in gens.values():
            try:
                gen.close()
            except Exception:
                pass
        gens.clear()
    return 0


if __name__ == "__main__":
    sys.exit(main())
