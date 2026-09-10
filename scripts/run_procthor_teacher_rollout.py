"""Real ProcTHOR object-search teacher rollout validation.

Drives real AI2-THOR houses and runs teacher episodes across configurable
cases, covering success, failure / unreachable / invalid task handling, and the
initial-visible spawn *resampling* path. Writes per-episode artifacts (frames +
two records) under --out and prints a compact per-case summary.

Requires the Python 3.10 venv (ai2thor 5.0.0) + a running X server (Xvfb :99).

Example:
    python scripts/run_procthor_teacher_rollout.py \\
        --house /root/autodl-tmp/val_house_0.json \\
        --out outputs/episodes
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

from spatialforge.embodied.rollout import ProcthorEpisodeGenerator  # noqa: E402


def leak_free(student: dict) -> bool:
    s = json.dumps(student)
    forbidden = (
        "position", "target_object_ids", "reachable", "teacher_path",
        "world_position", "segmentation", "godview", "god_view",
    )
    return not any(f in s for f in forbidden)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", required=True)
    ap.add_argument("--out", default="outputs/episodes")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=160)
    ap.add_argument("--x-display", default=":0")
    ap.add_argument("--require-nvidia", action="store_true",
                    help="fail loudly if the X display is not an NVIDIA renderer")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    gen = ProcthorEpisodeGenerator(
        args.house, width=args.width, height=args.height,
        x_display=args.x_display, out_dir=args.out, seed=args.seed,
        require_nvidia=args.require_nvidia,
    )
    try:
        summary = {"house": gen.house_id, "cases": []}
        # success cases (non-trivial spawn, not initially visible)
        for cat in ("mug", "winebottle"):
            r = gen.generate_episode(cat, max_steps=150, allow_easy=False)
            _log(r)
            summary["cases"].append(_compact(r))
        # initial-visible spawn resampling demonstration
        r = gen.generate_episode("mug", max_steps=150, allow_easy=False,
                                 demo_resample=True)
        _log(r)
        summary["cases"].append(_compact(r))
        # step-budget timeout / failure
        r = gen.generate_episode("teddybear", max_steps=5, allow_easy=False)
        _log(r)
        summary["cases"].append(_compact(r))
        # invalid category (no real instances)
        r = gen.generate_episode("nonexistent_thing_xyz", max_steps=50)
        _log(r)
        summary["cases"].append(_compact(r))
        # unreachable / not-observable real object
        r = gen.generate_episode("apple", max_steps=60)
        _log(r)
        summary["cases"].append(_compact(r))

        print("=== EPISODE LEAK CHECK ===")
        for r in summary["cases"]:
            ok = leak_free(r["student_record"])
            print(f"{r['episode_id']:<22} domain={r['student_record']['domain']:<16} "
                  f"student_leak_free={ok}")
        print("ROLLOUT_SUMMARY " + json.dumps(summary, default=str))
        return 0
    finally:
        gen.close()


def _compact(r: dict) -> dict:
    return {
        "episode_id": r["episode_id"],
        "house_id": r["house_id"],
        "category": r["category"],
        "target_object_id": r.get("target_object_id"),
        "initial_pose": r.get("initial_agent_pose"),
        "initially_visible": r.get("initially_visible"),
        "resample_attempts": r.get("spawn_resample_attempts"),
        "status": r["status"],
        "success": r["success"],
        "outcome_code": r.get("outcome_code"),
        "steps": r["steps"],
        "reason": r.get("reason"),
        "student_record": r["student_record"],
        "privileged_record": r["privileged_record"],
    }


def _log(r: dict) -> None:
    print(f"[case] {r['episode_id']} cat={r['category']:<12} "
          f"status={r['status']} success={r['success']} steps={r['steps']} "
          f"initially_visible={r.get('initially_visible')} "
          f"resample={r.get('spawn_resample_attempts')} reason={r.get('reason')}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
