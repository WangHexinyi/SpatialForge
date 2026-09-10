"""Phase C -- real multi-house teacher BC dataset generation + manifest build.

Subcommands:
    gen        run real teacher rollouts over a planned job pool
    manifest   aggregate persisted episodes into train/val manifests + report

Example (after catalog probe):
    python scripts/run_bc_dataset.py gen \
        --houses-dir outputs/embodied_bc/houses \
        --train-list outputs/embodied_bc/train_houses.txt \
        --val-list outputs/embodied_bc/val_houses.txt \
        --catalog outputs/embodied_bc/catalog.json \
        --phase train --workers 6 --out-root outputs/embodied_bc/dataset
    python scripts/run_bc_dataset.py manifest \
        --train-list ... --val-list ... --out-root outputs/embodied_bc/dataset \
        --train-cap 6400 --val-cap 1500
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialforge.embodied import bc_manifest as bm  # noqa: E402
from spatialforge.embodied.bc_dataset import (  # noqa: E402
    load_manifest,
    scan_rows_leakage,
    write_manifest,
)
from spatialforge.embodied.worker_pool import (  # noqa: E402
    DEFAULT_VENV_PYTHON,
    load_all_results,
    run_worker_pool,
)


def resolve_houses(houses_dir: str, stems: str) -> list:
    if not stems:
        return []
    if os.path.exists(stems):
        with open(stems) as f:
            items = [ln.strip() for ln in f if ln.strip()]
    else:
        items = [s for s in stems.split(",") if s]
    candidates = [houses_dir, "/root/autodl-tmp", os.getcwd()]
    out = []
    for item in items:
        if os.path.exists(item):
            out.append(os.path.abspath(item))
            continue
        found = None
        for base in candidates:
            p = os.path.join(base, item if item.endswith(".json") else f"{item}.json")
            if os.path.exists(p):
                found = os.path.abspath(p)
                break
        if found is None:
            raise FileNotFoundError(f"house not found: {item}")
        out.append(found)
    return out


def cmd_gen(args) -> int:
    catalog = json.load(open(args.catalog)) if os.path.exists(args.catalog) else {}
    train_houses = resolve_houses(args.houses_dir, args.train_list)
    val_houses = resolve_houses(args.houses_dir, args.val_list)
    if args.phase == "train":
        houses = train_houses
        tag = "bc"
    else:
        houses = val_houses
        tag = "bcval"
    if not houses:
        print(f"[gen] no houses for phase {args.phase}", flush=True)
        return 1
    jobs = bm.plan_jobs(
        houses, catalog,
        min_count=args.min_count,
        categories_per_house=args.categories_per_house,
        episodes_per_category=args.episodes_per_category,
        max_steps=args.max_steps,
        seed_base=args.seed_base,
        tag=tag,
        k_offset=args.episodes_start,
    )
    print(f"[gen] phase={args.phase} houses={len(houses)} jobs={len(jobs)}", flush=True)
    jobs_path = os.path.join(args.out_root, "runs", f"jobs-{args.phase}.json")
    os.makedirs(os.path.dirname(jobs_path), exist_ok=True)
    with open(jobs_path, "w") as f:
        json.dump({"jobs": jobs}, f, indent=1)

    run_dir = os.path.join(args.out_root, "runs",
                           f"{args.phase}-w{args.workers}-{int(time.time())}")
    res = run_worker_pool(
        jobs, num_workers=args.workers, out_dir=run_dir,
        python=args.venv_python, persist=True, persist_root=args.out_root,
        width=args.width, x_display=args.x_display,
        require_nvidia=not args.no_nvidia, tag=tag,
        max_respawns_per_slot=args.max_respawns,
    )
    print("[gen] SUMMARY " + json.dumps(res, default=str)[:2000], flush=True)
    # write a copy of the chosen jobs into the run dir for provenance
    with open(os.path.join(run_dir, "jobs.json"), "w") as f:
        json.dump({"jobs": jobs}, f, indent=1)
    with open(os.path.join(run_dir, "pool_result.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    return 0


def cmd_manifest(args) -> int:
    os.makedirs(args.out_root, exist_ok=True)
    run_dirs = sorted(glob.glob(os.path.join(args.out_root, "runs", "*")))
    result_rows = []
    for rd in run_dirs:
        result_rows.extend(load_all_results(os.path.join(rd, "results")).values())
    house_map = bm.build_episode_house_map(result_rows)
    episode_rows = bm.load_student_records(args.out_root)
    rows = bm.collect_rows(episode_rows, house_map, args.out_root, "all")
    print(f"[manifest] result rows={len(result_rows)} episodes w/records={len(episode_rows)} "
          f"expanded rows={len(rows)}", flush=True)

    train_list = resolve_houses(args.houses_dir, args.train_list)
    val_list = resolve_houses(args.houses_dir, args.val_list)
    train_stems = [bm.house_stem(p) for p in train_list]
    val_stems = [bm.house_stem(p) for p in val_list]
    train_rows, val_rows = bm.split_rows_by_house(rows, train_stems, val_stems)
    print(f"[manifest] train rows={len(train_rows)} val rows={len(val_rows)}", flush=True)

    leak_bad = scan_rows_leakage(rows)
    if leak_bad:
        print(f"[manifest] LEAKAGE FOUND in {len(leak_bad)} rows: {leak_bad[:10]}", flush=True)
        return 2
    print("[manifest] leakage scan: 0 privileged leakage", flush=True)

    train_rows = bm.cap_rows_deterministic(train_rows, args.train_cap)
    val_rows = bm.cap_rows_deterministic(val_rows, args.val_cap)
    write_manifest(train_rows, os.path.join(args.out_root, "bc_train.jsonl"))
    write_manifest(val_rows, os.path.join(args.out_root, "bc_val.jsonl"))
    train_rep = bm.report_dataset(train_rows)
    val_rep = bm.report_dataset(val_rows)
    report = {
        "schema": "embodied_bc_manifest.v1",
        "version": "1.0.0",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "provenance": {
            "generator": "scripts/run_bc_dataset.py + spatialforge.embodied.rollout",
            "procthor_split_source": "official ProcTHOR-10K val.jsonl.gz (train-split houses "
                                     "unavailable: allenai/procthor-10k HF gated / S3 403; "
                                     "strict disjoint house-level split used instead)",
            "renderer": "NVIDIA GLX (Xorg :0) quality=Low",
            "resolution": "256x256",
            "teacher": "ThorObjectSearchTeacher (privileged visible-goal nav)",
            "action_vocabulary": list(bm.BC_ACTION_VOCAB),
            "train_house_stems": train_stems,
            "val_house_stems": val_stems,
            "house_level_split_verified": len(set(train_stems) & set(val_stems)) == 0,
        },
        "train": train_rep,
        "val": val_rep,
    }
    with open(os.path.join(args.out_root, "dataset_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[manifest] DATASET_REPORT " + json.dumps(report, default=str)[:2500], flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gen")
    p.add_argument("--phase", choices=["train", "val"], required=True)
    p.add_argument("--houses-dir", default="outputs/embodied_bc/houses")
    p.add_argument("--train-list", default="outputs/embodied_bc/train_houses.txt")
    p.add_argument("--val-list", default="outputs/embodied_bc/val_houses.txt")
    p.add_argument("--catalog", default="outputs/embodied_bc/catalog.json")
    p.add_argument("--out-root", default="outputs/embodied_bc/dataset")
    p.add_argument("--venv-python", default=DEFAULT_VENV_PYTHON)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--x-display", default=":0")
    p.add_argument("--no-nvidia", dest="no_nvidia", action="store_true")
    p.add_argument("--max-respawns", type=int, default=3)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--categories-per-house", type=int, default=6)
    p.add_argument("--episodes-per-category", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--seed-base", type=int, default=10000)
    p.add_argument("--episodes-start", type=int, default=0,
                   help="k offset for supplemental runs (unique job ids/seeds)")
    p.set_defaults(func=cmd_gen)

    p = sub.add_parser("manifest")
    p.add_argument("--houses-dir", default="outputs/embodied_bc/houses")
    p.add_argument("--train-list", default="outputs/embodied_bc/train_houses.txt")
    p.add_argument("--val-list", default="outputs/embodied_bc/val_houses.txt")
    p.add_argument("--out-root", default="outputs/embodied_bc/dataset")
    p.add_argument("--train-cap", type=int, default=6400)
    p.add_argument("--val-cap", type=int, default=1500)
    p.set_defaults(func=cmd_manifest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
