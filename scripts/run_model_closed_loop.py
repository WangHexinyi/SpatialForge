"""Phase F -- real model-controlled closed-loop rollout on held-out houses.

Starts the torch action server (miniconda python) then runs N rollout workers
(venv python) that let the Qwen BC model control real AI2-THOR agents in real
houses. Launch from any interpreter:

    python scripts/run_model_closed_loop.py \
        --houses /root/autodl-tmp/val_house_0.json /root/autodl-tmp/val_house_1.json \
        --catalog outputs/embodied_bc/catalog.json \
        --adapter outputs/embodied_bc/train/run1/adapter \
        --workers 2 --episodes-per-house 3 \
        --out-root outputs/embodied_bc/model_rollout
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
    aggregate_results,
    load_all_results,
    run_worker_pool,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCH_PYTHON = "/root/miniconda3/bin/python"
MODEL_SERVER = os.path.join(REPO_ROOT, "scripts", "model_action_server.py")
MODEL_WORKER = os.path.join(REPO_ROOT, "scripts", "embodied_model_worker.py")


def wait_health(url: str, timeout_s: float = 600.0) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with opener.open(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(3.0)
    return False


def plan_model_jobs(houses, catalog, episodes_per_house: int, min_count: int,
                    max_steps: int, seed_base: int) -> list:
    from spatialforge.embodied.bc_manifest import house_stem

    jobs = []
    for house in houses:
        entry = catalog.get(os.path.abspath(house))
        if not entry or entry.get("error"):
            continue
        cats = sorted(
            ((c, n) for c, n in entry.get("category_counts", {}).items()
             if c not in _STRUCTURAL and n >= min_count),
            key=lambda kv: (-kv[1], kv[0]),
        )
        for cat, _ in cats[:3]:
            for k in range(episodes_per_house):
                jobs.append({
                    "house": os.path.abspath(house),
                    "category": cat,
                    "seed": seed_base + len(jobs),
                    "max_steps": max_steps,
                    "job_id": f"model-{house_stem(house)}-{cat}-{k}",
                })
    return jobs


_STRUCTURAL = {
    "wall", "floor", "ceiling", "doorway", "window", "cabinet", "dresser",
    "shelf", "countertop", "counter", "toilet", "bathtub", "fridge",
    "lightswitch", "houseplant", "table", "armchair", "sofa", "bed", "chair",
    "desk", "bookcase", "television", "kitchencounter", "drawer", "painting",
    "mirror", "sink", "bin", "trashcan", "diningtable", "coffeetable", "side",
    "stool", "tv", "dog", "basketball", "plunger", "laundry", "robothor",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", nargs="*", default=None)
    ap.add_argument("--catalog", default="outputs/embodied_bc/catalog.json")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--batch-window-ms", type=float, default=8.0)
    ap.add_argument("--godview-url", default=None,
                    help="optional Embodied Inspector URL to push display-only telemetry")
    ap.add_argument("--model-tag", default=None,
                    help="display label for the actor (e.g. qwen3-8b-weighted-bc)")
    ap.add_argument("--training-profile", default=None,
                    help="trainability profile label for the God View model card")
    ap.add_argument("--visual-tokens", type=int, default=None,
                    help="visual tokens per image (researcher metadata)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--episodes-per-house", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=90000)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--server-port", type=int, default=8765)
    ap.add_argument("--venv-python", default=DEFAULT_VENV_PYTHON)
    ap.add_argument("--torch-python", default=TORCH_PYTHON)
    ap.add_argument("--out-root", default="outputs/embodied_bc/model_rollout")
    ap.add_argument("--x-display", default=":0")
    ap.add_argument("--jobs-file", default=None,
                    help="explicit jobs JSON {'jobs':[...]} (bypasses auto-plan)")
    ap.add_argument("--no-nvidia", dest="require_nvidia", action="store_false")
    ap.set_defaults(require_nvidia=True)
    args = ap.parse_args()

    catalog = json.load(open(args.catalog)) if os.path.exists(args.catalog) else {}
    if args.jobs_file:
        jobs = json.load(open(args.jobs_file))["jobs"]
        for j in jobs:
            j["house"] = os.path.abspath(j["house"])
    elif not args.houses:
        ap.error("--houses (or --jobs-file) is required")
    else:
        jobs = plan_model_jobs(
            args.houses, catalog, args.episodes_per_house, args.min_count,
            args.max_steps, args.seed_base,
        )
    if not jobs:
        print("[model_closed_loop] no jobs planned", flush=True)
        return 1
    n_houses = len(args.houses) if args.houses else len({os.path.basename(j["house"]) for j in jobs})
    print(f"[model_closed_loop] {len(jobs)} episodes over {n_houses} houses",
          flush=True)

    ready = os.path.join(args.out_root, "server_ready.txt")
    os.makedirs(args.out_root, exist_ok=True)
    for f in (ready,):
        try:
            os.remove(f)
        except OSError:
            pass
    info_file = os.path.join(args.out_root, "model_info.json")
    cmd = [
        args.torch_python, MODEL_SERVER,
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
    server_log = open(os.path.join(args.out_root, "model_server.log"), "w")
    server = subprocess.Popen(cmd, stdout=server_log, stderr=server_log,
                              start_new_session=True)
    url = f"http://127.0.0.1:{args.server_port}"
    try:
        if not wait_health(url):
            print("[model_closed_loop] model server failed to become ready", flush=True)
            return 2
        print(f"[model_closed_loop] model server ready on {url}", flush=True)

        run_dir = os.path.join(args.out_root, f"runs/w{args.workers}-{int(time.time())}")
        res = run_worker_pool(
            jobs, num_workers=args.workers, out_dir=run_dir,
            python=args.venv_python,
            persist=True,
            persist_root=args.out_root,
            width=args.width, x_display=args.x_display,
            require_nvidia=args.require_nvidia, tag="model",
            env={"SF_MODEL_URL": url},
            job_timeout_s=420.0,
            max_respawns_per_slot=3,
            worker_script=MODEL_WORKER,
            extra_worker_args=["--server-url", url, "--setup-timeout", "150"],
        )
        print("[model_closed_loop] SUMMARY " + json.dumps(res, default=str)[:2000],
              flush=True)
        with open(os.path.join(run_dir, "pool_result.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)

        agg = res.get("aggregate", {}) or {}
        model_info = {}
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                model_info = json.load(f).get("model", {})
        except Exception:
            pass
        model_info = dict(model_info)
        if args.model_tag:
            model_info["label"] = args.model_tag
        if args.training_profile:
            model_info["training_profile"] = args.training_profile
        if args.visual_tokens:
            model_info["visual_tokens"] = args.visual_tokens
        model_info["visual_resolution"] = f"{args.width}x{args.width}"
        telemetry = {
            "ts": time.time(),
            "model": model_info,
            "runtime": {
                "agents": args.workers,
                "env_steps_per_sec": agg.get("aggregate_steps_per_sec"),
                "episodes_per_hour": round(agg.get("episodes_per_min", 0.0) * 60.0, 2),
                "successful_episodes_per_hour": round(
                    agg.get("successful_episodes_per_min", 0.0) * 60.0, 2),
                "episodes": agg.get("episodes"),
                "successful_episodes": agg.get("successful_episodes"),
                "total_env_steps": agg.get("total_env_steps"),
                "model_batch": args.max_batch,
                "decision_p50_ms": (agg.get("env_step_latency_ms") or {}).get("p50_ms"),
                "outcome_codes": agg.get("outcome_codes"),
            },
        }
        with open(os.path.join(args.out_root, "live_telemetry.json"), "w",
                  encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2, default=str)
        if args.godview_url:
            try:
                req = urllib.request.Request(
                    args.godview_url.rstrip("/") + "/api/telemetry",
                    data=json.dumps(telemetry).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                opener.open(req, timeout=5).read()
            except Exception:
                pass
        return 0
    finally:
        try:
            server.terminate()
            server.wait(timeout=10)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass
        server_log.close()


if __name__ == "__main__":
    sys.exit(main())
