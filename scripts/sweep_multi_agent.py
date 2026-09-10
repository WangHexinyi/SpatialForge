"""Multi-agent batched closed-loop throughput sweep.

Starts one model action server (dynamic batcher) and runs the same fixed job
set with N agents (1/2/4/8/...) to find the throughput sweet spot. Reports
completed valid environment steps/sec, episodes/hour, per-agent decision
latency and GPU telemetry per N.

    python scripts/sweep_multi_agent.py \
        --houses /root/autodl-tmp/val_house_0.json /root/autodl-tmp/val_house_1.json \
        --agent-counts 1 2 4 8 --episodes-per-house 2 \
        --model-path /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
        --out-root outputs/embodied_bc/multi_agent
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialforge.embodied.worker_pool import (  # noqa: E402
    DEFAULT_VENV_PYTHON,
    run_worker_pool,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCH_PYTHON = "/root/miniconda3/bin/python"
MODEL_SERVER = os.path.join(REPO_ROOT, "scripts", "model_action_server.py")
MODEL_WORKER = os.path.join(REPO_ROOT, "scripts", "embodied_model_worker.py")


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def wait_health(url: str, timeout_s: float = 600.0) -> bool:
    op = _opener()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with op.open(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(3.0)
    return False


def server_stats(url: str) -> dict:
    try:
        with _opener().open(url + "/stats", timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:
        return {}


def plan_jobs(houses, episodes_per_house: int, max_steps: int, seed_base: int) -> list:
    from spatialforge.embodied.bc_manifest import house_stem

    jobs = []
    for house in houses:
        for k in range(episodes_per_house):
            jobs.append({
                "house": os.path.abspath(house),
                "category": None,  # resolved by worker? no: worker needs a category
                "seed": seed_base + len(jobs),
                "max_steps": max_steps,
                "job_id": f"ma-{house_stem(house)}-{k}",
            })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", nargs="+", required=True)
    ap.add_argument("--catalog", default="outputs/embodied_bc/catalog.json")
    ap.add_argument("--jobs-file", default=None)
    ap.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--agent-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--episodes-per-house", type=int, default=2)
    ap.add_argument("--categories", nargs="*", default=None,
                    help="fixed target categories (same jobs for every N)")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=70000)
    ap.add_argument("--server-port", type=int, default=8767)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--batch-window-ms", type=float, default=8.0)
    ap.add_argument("--venv-python", default=DEFAULT_VENV_PYTHON)
    ap.add_argument("--out-root", default="outputs/embodied_bc/multi_agent")
    ap.add_argument("--x-display", default=":0")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--no-nvidia", dest="require_nvidia", action="store_false")
    ap.set_defaults(require_nvidia=True)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    if args.jobs_file:
        jobs = json.load(open(args.jobs_file))["jobs"]
    else:
        catalog = json.load(open(args.catalog)) if os.path.exists(args.catalog) else {}
        from scripts.run_model_closed_loop import plan_model_jobs  # type: ignore

        jobs = plan_model_jobs(
            args.houses, catalog, args.episodes_per_house, 2,
            args.max_steps, args.seed_base,
        )
    if args.categories:
        keep = set(args.categories)
        jobs = [j for j in jobs if j["category"] in keep]
    if not jobs:
        print("[sweep] no jobs")
        return 1
    # freeze the job list for all N
    jobs_file = os.path.join(args.out_root, "sweep_jobs.json")
    with open(jobs_file, "w") as f:
        json.dump({"jobs": jobs}, f, indent=1)
    print(f"[sweep] {len(jobs)} fixed jobs; agent counts={args.agent_counts}", flush=True)

    ready = os.path.join(args.out_root, "server_ready.txt")
    try:
        os.remove(ready)
    except OSError:
        pass
    info_file = os.path.join(args.out_root, "model_info.json")
    cmd = [
        TORCH_PYTHON, MODEL_SERVER,
        "--port", str(args.server_port),
        "--model-path", args.model_path,
        "--ready-file", ready,
        "--info-file", info_file,
        "--attn", args.attn,
        "--max-batch", str(args.max_batch),
        "--batch-window-ms", str(args.batch_window_ms),
    ]
    if args.adapter:
        cmd += ["--adapter", args.adapter]
    log = open(os.path.join(args.out_root, "model_server.log"), "w")
    server = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
    url = f"http://127.0.0.1:{args.server_port}"
    results = []
    try:
        if not wait_health(url):
            print("[sweep] model server failed to start")
            return 2
        for n in args.agent_counts:
            stats_before = server_stats(url)
            run_dir = os.path.join(args.out_root, f"agents-{n}")
            t0 = time.time()
            res = run_worker_pool(
                jobs, num_workers=n, out_dir=run_dir,
                python=args.venv_python, persist=True,
                persist_root=run_dir, width=args.width,
                x_display=args.x_display, require_nvidia=args.require_nvidia,
                tag="ma", env={"SF_MODEL_URL": url},
                job_timeout_s=600.0, max_respawns_per_slot=3,
                worker_script=MODEL_WORKER,
                extra_worker_args=["--server-url", url, "--setup-timeout", "150"],
            )
            stats_after = server_stats(url)
            entry = {
                "agents": n,
                "wall_sec": round(time.time() - t0, 1),
                "aggregate": res.get("aggregate"),
                "telemetry": res.get("telemetry"),
                "server": {
                    "batches_served": stats_after.get("batches_served", 0)
                    - stats_before.get("batches_served", 0),
                    "items_served": stats_after.get("items_served", 0)
                    - stats_before.get("items_served", 0),
                    "mean_batch_size": stats_after.get("mean_batch_size"),
                },
                "jobs_missing": res.get("jobs_missing"),
            }
            results.append(entry)
            agg = entry["aggregate"] or {}
            print(f"[sweep] agents={n} steps/s={agg.get('aggregate_steps_per_sec')} "
                  f"ep/h={round(agg.get('episodes_per_min', 0)*60, 2)} "
                  f"batch={entry['server']['mean_batch_size']}", flush=True)
    finally:
        try:
            server.terminate()
            server.wait(timeout=10)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass
        log.close()

    report = {
        "schema": "multi_agent_sweep.v1",
        "model": args.model_path,
        "adapter": args.adapter,
        "max_batch": args.max_batch,
        "batch_window_ms": args.batch_window_ms,
        "jobs": len(jobs),
        "results": results,
    }
    with open(os.path.join(args.out_root, "sweep_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("[sweep] " + json.dumps(report, default=str)[:2000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
