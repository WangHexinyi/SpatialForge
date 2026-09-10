"""Held-out real closed-loop evaluation matrix (same jobs, multiple actors).

Runs the SAME fixed held-out ProcTHOR job set for each actor config
(label:model_path[:adapter]) through scripts/run_model_closed_loop.py and
aggregates the scientific metrics required to judge policy collapse:

success rate, avg steps, invalid rate, action entropy, MoveAhead / Done /
Rotate ratios, false-Done, timeout, decision latency, throughput.

    python scripts/eval_closed_loop_matrix.py \
        --jobs-file outputs/embodied_bc/eval_jobs.json \
        --config legacy3b:/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct:outputs/embodied_bc/train/run1/adapter \
        --config qwen3_8b_base:/root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
        --config qwen3_8b_bc:/root/autodl-tmp/models/Qwen3-VL-8B-Instruct:outputs/embodied_bc/train/qwen3_run1/adapter \
        --workers 4 --out-root outputs/embodied_bc/eval_closed_loop
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialforge.embodied.bc_dataset import BC_ACTION_VOCAB  # noqa: E402
from spatialforge.embodied.worker_pool import load_all_results  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCH_PYTHON = "/root/miniconda3/bin/python"
CLOSED_LOOP = os.path.join(REPO_ROOT, "scripts", "run_model_closed_loop.py")


def parse_config(s: str):
    parts = s.split(":")
    label = parts[0]
    model = parts[1] if len(parts) > 1 else ""
    adapter = parts[2] if len(parts) > 2 and parts[2] else None
    return label, model, adapter


def aggregate_run(out_root: str) -> dict:
    result_rows = []
    for rd in sorted(glob.glob(os.path.join(out_root, "runs", "*"))):
        result_rows.extend(load_all_results(os.path.join(rd, "results")).values())

    episodes = result_rows
    n = len(episodes)
    successes = [r for r in episodes if r.get("success")]
    total_steps = sum(int(r.get("steps", 0)) for r in episodes)
    invalid = sum(int(r.get("invalid_decision_count", 0)) for r in episodes)
    decisions = sum(int(r.get("decision_count", 0)) for r in episodes)

    action_counts = {a: 0 for a in BC_ACTION_VOCAB}
    invalid_decisions = 0
    latencies = []
    false_done = 0
    done_executed = 0
    for ep_dir in sorted(glob.glob(os.path.join(out_root, "ep-*"))):
        dec_file = os.path.join(ep_dir, "decisions.json")
        if not os.path.isfile(dec_file):
            continue
        try:
            decs = json.loads(open(dec_file).read())
        except Exception:
            continue
        for d in decs:
            a = d.get("model_action")
            if a in action_counts:
                action_counts[a] += 1
            if d.get("invalid"):
                invalid_decisions += 1
            if isinstance(d.get("latency_ms"), (int, float)):
                latencies.append(float(d["latency_ms"]))
        if decs and decs[-1].get("model_action") == "Done":
            done_executed += 1
    for r in episodes:
        code = str(r.get("outcome_code") or "")
        if "Done emitted but no target" in code:
            false_done += 1

    total_actions = sum(action_counts.values())
    probs = [c / total_actions for c in action_counts.values() if c > 0]
    entropy = -sum(p * math.log(p) for p in probs) if probs else 0.0
    entropy_norm = entropy / math.log(len(BC_ACTION_VOCAB)) if probs else 0.0

    def ratio(name):
        return round(action_counts[name] * 100.0 / max(total_actions, 1), 2)

    lat = sorted(latencies)
    return {
        "episodes": n,
        "successes": len(successes),
        "success_rate": round(len(successes) * 100.0 / max(n, 1), 2),
        "total_env_steps": total_steps,
        "avg_steps_per_episode": round(total_steps / max(n, 1), 2),
        "decisions": decisions,
        "invalid_decisions": invalid_decisions,
        "invalid_rate": round(invalid_decisions * 100.0 / max(decisions, 1), 2),
        "false_done_episodes": false_done,
        "done_executed_episodes": done_executed,
        "timeout_episodes": sum(
            1 for r in episodes if "step budget" in str(r.get("outcome_code") or "")),
        "invalid_exhausted_episodes": sum(
            1 for r in episodes if r.get("outcome_code") == "invalid_exhausted"),
        "action_counts": action_counts,
        "moveahead_ratio": ratio("MoveAhead"),
        "done_ratio": ratio("Done"),
        "rotate_ratio": round((action_counts["RotateLeft"] + action_counts["RotateRight"])
                              * 100.0 / max(total_actions, 1), 2),
        "action_entropy": round(entropy, 4),
        "action_entropy_normalized": round(entropy_norm, 4),
        "decision_latency_ms": {
            "mean": round(sum(lat) / max(len(lat), 1), 1),
            "p50": lat[len(lat) // 2] if lat else None,
            "p95": lat[min(len(lat) - 1, int(0.95 * (len(lat) - 1)))] if lat else None,
        } if lat else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-file", required=True)
    ap.add_argument("--config", action="append", required=True,
                    help="label:model_path[:adapter]")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--server-port", type=int, default=8768)
    ap.add_argument("--out-root", default="outputs/embodied_bc/eval_closed_loop")
    ap.add_argument("--venv-python", default=None)
    ap.add_argument("--x-display", default=":0")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--godview-url", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    report = {"schema": "closed_loop_eval_matrix.v1", "configs": {}}
    for cfg in args.config:
        label, model, adapter = parse_config(cfg)
        out = os.path.join(args.out_root, label)
        os.makedirs(out, exist_ok=True)
        cmd = [
            TORCH_PYTHON, CLOSED_LOOP,
            "--jobs-file", args.jobs_file,
            "--model-path", model,
            "--workers", str(args.workers),
            "--server-port", str(args.server_port),
            "--out-root", out,
            "--x-display", args.x_display,
            "--width", str(args.width),
            "--model-tag", label,
            "--visual-tokens", str((args.width // 16 // 2) ** 2),
        ]
        if adapter:
            cmd += ["--adapter", adapter]
        if args.venv_python:
            cmd += ["--venv-python", args.venv_python]
        if args.godview_url:
            cmd += ["--godview-url", args.godview_url]
        t0 = time.time()
        print(f"[matrix] running {label} ...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
        agg = aggregate_run(out)
        agg["wall_sec"] = round(time.time() - t0, 1)
        agg["model"] = model
        agg["adapter"] = adapter
        agg["steps_per_sec"] = round(
            agg["total_env_steps"] / max(agg["wall_sec"], 0.001), 3)
        report["configs"][label] = agg
        print(f"[matrix] {label}: success={agg['success_rate']}% "
              f"avg_steps={agg['avg_steps_per_episode']} invalid={agg['invalid_rate']}% "
              f"entropy={agg['action_entropy_normalized']} "
              f"MoveAhead={agg['moveahead_ratio']}% Done={agg['done_ratio']}%", flush=True)

    with open(os.path.join(args.out_root, "matrix_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("[matrix] " + json.dumps(report, indent=2, default=str)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
