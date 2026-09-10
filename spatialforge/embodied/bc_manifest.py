"""BC dataset job planning + manifest aggregation (pure Python).

House identity is the *house file stem* (official ProcTHOR-10K houses do not
carry a unique ``houseId``), which makes the required scene/house-level
train/validation split explicit and verifiable.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from spatialforge.embodied.bc_dataset import (
    BC_ACTION_VOCAB,
    load_manifest,
    scan_rows_leakage,
    student_record_to_rows,
    write_manifest,
)

DEFAULT_STRUCTURAL = {
    "wall", "floor", "ceiling", "doorway", "window", "cabinet", "dresser",
    "shelf", "countertop", "counter", "toilet", "bathtub", "bathtubbasin",
    "fridge", "lightswitch", "houseplant", "table", "armchair", "sofa", "bed",
    "chair", "desk", "bookcase", "television", "kitchencounter", "drawer",
    "painting", "mirror", "sink", "bin", "trashcan", "blinds", "curtains",
    "diningtable", "coffeetable", "side", "laundry", "robothor", "stool",
    "tv", "dog", "basketball", "plunger", "toiletpaperholder",
}


def house_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def is_searchable_category(category: str) -> bool:
    return category.lower() not in DEFAULT_STRUCTURAL


def plan_jobs(
    houses: Sequence[str],
    catalog: Dict[str, Dict[str, Any]],
    min_count: int = 2,
    categories_per_house: int = 6,
    episodes_per_category: int = 3,
    max_steps: int = 120,
    seed_base: int = 1000,
    tag: str = "bc",
    k_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Deterministic per-house job list from the engine catalog."""
    jobs: List[Dict[str, Any]] = []
    for house in houses:
        key = os.path.abspath(house)
        entry = catalog.get(key)
        if entry is None or entry.get("error") or entry.get("objects", 0) == 0:
            continue
        cats = [
            c for c, n in entry.get("category_counts", {}).items()
            if is_searchable_category(c) and n >= min_count
        ]
        # deterministic, prefer richer categories; drop structural leftovers
        cats = sorted(cats, key=lambda c: (-entry["category_counts"][c], c))
        cats = cats[:categories_per_house]
        for cat in cats:
            for k in range(k_offset, k_offset + episodes_per_category):
                jobs.append({
                    "house": os.path.abspath(house),
                    "category": cat,
                    "seed": seed_base + len(jobs),
                    "max_steps": max_steps,
                    "job_id": f"{tag}-{house_stem(house)}-{cat}-{k}",
                })
    return jobs


def load_student_records(out_root: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Load every student_training_record.json under out_root/<episode>/.json"""
    found: List[Tuple[str, Dict[str, Any]]] = []
    if not os.path.isdir(out_root):
        return found
    for name in sorted(os.listdir(out_root)):
        if name in ("runs", "heartbeats", "results"):
            continue
        ep_dir = os.path.join(out_root, name)
        rec_path = os.path.join(ep_dir, "student_training_record.json")
        if os.path.isdir(ep_dir) and os.path.exists(rec_path):
            with open(rec_path, "r", encoding="utf-8") as f:
                found.append((name, json.load(f)))
    return found


def build_episode_house_map(result_rows: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in result_rows:
        ep = row.get("episode_id")
        house = row.get("house")
        if ep and house:
            mapping[ep] = house_stem(str(house))
    return mapping


def collect_rows(
    episode_rows: Iterable[Tuple[str, Dict[str, Any]]],
    house_map: Dict[str, str],
    run_root: str,
    split: str,
) -> List[Dict[str, Any]]:
    """Expand records into manifest rows; drop missing/terminal-only episodes."""
    out: List[Dict[str, Any]] = []
    missing_images = 0
    for ep_id, record in episode_rows:
        house = house_map.get(ep_id)
        if house is None:
            continue
        for row in student_record_to_rows(record):
            img = row.get("image_path")
            if not img:
                missing_images += 1
                continue
            abs_img = os.path.join(run_root, img)
            if not os.path.exists(abs_img):
                missing_images += 1
                continue
            row["image_path"] = os.path.relpath(abs_img, run_root)
            row["scene_id"] = house
            row["split"] = split
            row["image_exists"] = True
            out.append(row)
    return out


def split_rows_by_house(
    rows: Sequence[Dict[str, Any]],
    train_houses: Sequence[str],
    val_houses: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_set = set(train_houses)
    val_set = set(val_houses)
    overlap = train_set & val_set
    if overlap:
        raise ValueError(f"House-level split violation: houses in both splits: {sorted(overlap)}")
    train_rows, val_rows = [], []
    unknown: List[str] = []
    for row in rows:
        scene = str(row.get("scene_id", ""))
        if scene in train_set:
            train_rows.append(row)
        elif scene in val_set:
            val_rows.append(row)
        else:
            unknown.append(scene)
    if unknown:
        raise ValueError(f"Rows from houses not assigned to any split: {sorted(set(unknown))}")
    return train_rows, val_rows


def cap_rows_deterministic(
    rows: Sequence[Dict[str, Any]], cap: int
) -> List[Dict[str, Any]]:
    """House-stratified deterministic cap (drop tail rows after sorting)."""
    if len(rows) <= cap:
        return list(rows)
    by_house: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_house.setdefault(str(row.get("scene_id", "?")), []).append(row)
    kept: List[Dict[str, Any]] = []
    total = len(rows)
    # integer quota via largest-remainder on deterministic order
    quotas = {
        h: (len(v) * cap) // total for h, v in sorted(by_house.items())
    }
    remainder = cap - sum(quotas.values())
    for h in sorted(by_house):
        if remainder > 0:
            quotas[h] += 1
            remainder -= 1
    for h in sorted(by_house):
        v = sorted(by_house[h], key=lambda r: r["id"])
        kept.extend(v[: quotas[h]])
    kept.sort(key=lambda r: r["id"])
    return kept


def report_dataset(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    cat_counts: Dict[str, int] = {}
    house_counts: Dict[str, int] = {}
    ep_counts: Dict[str, int] = {}
    done_target = 0
    for row in rows:
        a = str(row.get("answer", ""))
        counts[a] = counts.get(a, 0) + 1
        c = str(row.get("category", ""))
        cat_counts[c] = cat_counts.get(c, 0) + 1
        h = str(row.get("scene_id", ""))
        house_counts[h] = house_counts.get(h, 0) + 1
        e = str(row.get("episode_id", ""))
        ep_counts[e] = ep_counts.get(e, 0) + 1
        if a == "Done":
            done_target += 1
    total = len(rows)
    pct = {k: round(v * 100.0 / total, 2) for k, v in counts.items()} if total else {}
    return {
        "total_samples": total,
        "episodes": len(ep_counts),
        "houses": len(house_counts),
        "per_house_samples": dict(sorted(house_counts.items())),
        "action_counts": dict(sorted(counts.items())),
        "action_pct": {a: pct.get(a, 0.0) for a in BC_ACTION_VOCAB},
        "category_counts": dict(sorted(cat_counts.items())),
        "done_target_pct": round(done_target * 100.0 / total, 2) if total else 0.0,
    }
