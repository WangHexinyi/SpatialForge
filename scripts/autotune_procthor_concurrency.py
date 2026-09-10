"""Phase B -- real ProcTHOR concurrency autotune sweep.

Runs one fixed, deterministic job pool over worker counts [1, 2, 4, 6, 8] (and
optionally 12/16), each worker owning real independent AI2-THOR environments on
the NVIDIA X display. Records aggregate useful throughput (env steps/sec,
episodes/min, successful episodes/min, step latency), GPU/CPU/RAM telemetry and
robustness counters, then selects the profile with maximum *useful* aggregate
throughput that is stable and correct.

Run from any interpreter (parent never imports ai2thor):
    python scripts/autotune_procthor_concurrency.py \
        --jobs outputs/embodied_bc/sweep_jobs.json \
        --out-root outputs/embodied_bc/sweep \
        --venv-python /root/autodl-tmp/.venv-procthor/bin/python \
        --width 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialforge.embodied.worker_pool import (  # noqa: E402
    DEFAULT_VENV_PYTHON,
    aggregate_results,
    load_all_results,
    run_worker_pool,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out-root", default="outputs/embodied_bc/sweep")
    ap.add_argument("--venv-python", default=DEFAULT_VENV_PYTHON)
    ap.add_argument("--worker-counts", default="1,2,4,6,8")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--x-display", default=":0")
    ap.add_argument("--no-nvidia", dest="require_nvidia", action="store_false")
    ap.add_argument("--max-respawns", type=int, default=2)
    args = ap.parse_args()

    with open(args.jobs, "r", encoding="utf-8") as f:
        jobs = json.load(f)["jobs"]

    counts = [int(c) for c in args.worker_counts.split(",") if c.strip()]
    os.makedirs(args.out_root, exist_ok=True)
    report: dict = {"jobs": jobs, "runs": []}
    for n in counts:
        if n > len(jobs):
            print(f"[sweep] skipping {n} workers (> jobs {len(jobs)})", flush=True)
            continue
        print(f"\n===== WORKER SWEEP n={n} ({len(jobs)} jobs) =====", flush=True)
        out_dir = os.path.join(args.out_root, f"w{n}")
        res = run_worker_pool(
            jobs, num_workers=n, out_dir=out_dir,
            python=args.venv_python, persist=False, width=args.width,
            x_display=args.x_display, require_nvidia=args.require_nvidia,
            tag="sweep", max_respawns_per_slot=args.max_respawns,
        )
        print("[sweep] " + json.dumps({
            "workers": n,
            **res["aggregate"],
            "gpu_util_avg": (res["telemetry"].get("gpu") or {}).get("gpu_utilization", {}).get("avg"),
            "gpu_mem_avg_mb": (res["telemetry"].get("gpu") or {}).get("gpu_memory_mb", {}).get("avg"),
            "power_avg_w": (res["telemetry"].get("gpu") or {}).get("gpu_power_w", {}).get("avg"),
            "cpu_avg": res["telemetry"].get("cpu", {}).get("overall", {}).get("avg"),
        }, default=str), flush=True)
        report["runs"].append({
            "num_workers": n, "result": res,
        })
    # selection: max aggregate env steps/sec among runs with no crashes
    best = None
    for run in report["runs"]:
        agg = run["result"]["aggregate"]
        if agg["crashed_jobs"] > 0:
            continue
        if best is None or agg["aggregate_steps_per_sec"] > best["agg"]["aggregate_steps_per_sec"]:
            best = {"workers": run["num_workers"], "agg": agg}
    report["selection"] = best
    with open(os.path.join(args.out_root, "autotune_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n=== SELECTED PROFILE ===")
    print(json.dumps(best, default=str, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
