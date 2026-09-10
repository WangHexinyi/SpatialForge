"""Probe real houses with the real engine and cache an object-type catalog.

For each house JSON, launch one genuine ProcTHORBackend, list authoritative
objectType counts (post CreateHouse), reachable count and agent metadata, then
close cleanly. Output: outputs/embodied_bc/catalog.json (plus optional house
extraction from a ProcTHOR-10K .jsonl.gz into per-house files).

Run under the Python-3.10 ai2thor venv:
    python scripts/procthor_catalog.py --houses val_house_0.json val_house_1.json
    python scripts/procthor_catalog.py --extract val.jsonl.gz --indexes 0,1,2,5 \
        --out outputs/embodied_bc/houses
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialforge.embodied.backends.procthor import ProcTHORBackend  # noqa: E402

STRUCTURAL = {
    "wall", "floor", "ceiling", "doorway", "window", "cabinet", "dresser",
    "shelf", "countertop", "counter", "toilet", "bathtub", "bathtubbasin",
    "fridge", "lightswitch", "houseplant", "table", "armchair", "sofa", "bed",
    "chair", "desk", "bookcase", "television", "kitchencounter", "drawer",
    "painting", "blinds", "curtains", "stoveburner", "stoveknob", "mirror",
    "showerhead", "showerglass", "towelholder", "toiletpaper", "topplight",
    "light", "switch", "diningtable", "coffeetable", "side", "trashcan",
    "robothor", "laundry", "sink", "bin",
}


def is_searchable(category: str) -> bool:
    return category.lower() not in STRUCTURAL


def probe_house(house_path: str, width: int = 256, x_display: str = ":0",
                require_nvidia: bool = True) -> dict:
    t0 = time.time()
    backend = ProcTHORBackend(house=house_path, width=width, height=width,
                              quality="Low", x_display=x_display,
                              require_nvidia=require_nvidia)
    try:
        info = backend.load_house(seed=0)
        counts = {k: len(v) for k, v in backend._category_object_ids.items()}
        reachable = len(backend._reachable)
        md = backend._event.metadata
        agent = md.get("agent", {})
        return {
            "house_path": house_path,
            "house_id": str(backend._house_data.get("houseId", "")),
            "rooms": info.room_count,
            "objects": info.object_count,
            "reachable_positions": reachable,
            "agent_pose": {
                "position": list(agent.get("position", {}).values()) or None,
                "yaw": (agent.get("rotation") or {}).get("y"),
                "horizon": agent.get("cameraHorizon"),
            },
            "category_counts": {k: v for k, v in sorted(counts.items(), key=lambda kv: -kv[1])},
            "searchable_counts": {k: v for k, v in sorted(
                ((k, v) for k, v in counts.items() if is_searchable(k)),
                key=lambda kv: -kv[1])},
            "probe_elapsed_s": round(time.time() - t0, 1),
        }
    finally:
        backend.close()


def extract_houses(jsonl_gz: str, indexes, out_dir: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    wanted = set(int(i) for i in indexes)
    written = []
    with gzip.open(jsonl_gz, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx not in wanted:
                continue
            house = json.loads(line)
            path = os.path.join(out_dir, f"house_{idx:04d}.json")
            with open(path, "w", encoding="utf-8") as out:
                json.dump(house, out)
            written.append({"index": idx, "path": path, "house_id": house.get("houseId")})
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", nargs="*", default=[])
    ap.add_argument("--extract", help="val.jsonl.gz to extract houses from")
    ap.add_argument("--indexes", help="comma separated indexes")
    ap.add_argument("--out", default="outputs/embodied_bc/houses")
    ap.add_argument("--catalog", default="outputs/embodied_bc/catalog.json")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--x-display", default=":0")
    ap.add_argument("--no-nvidia", dest="require_nvidia", action="store_false")
    ap.set_defaults(require_nvidia=True)
    args = ap.parse_args()

    written = []
    if args.extract:
        assert args.indexes, "--indexes required with --extract"
        written = extract_houses(args.extract,
                                 [int(i) for i in args.indexes.split(",")], args.out)
        print("extracted:", json.dumps(written), flush=True)
        for w in written:
            args.houses.append(w["path"])

    catalog_path = args.catalog
    catalog = {}
    if os.path.exists(catalog_path):
        with open(catalog_path) as f:
            catalog = json.load(f)
    for house_path in args.houses:
        key = os.path.abspath(house_path)
        print(f"[catalog] probing {house_path} ...", flush=True)
        try:
            entry = probe_house(house_path, width=args.width,
                                x_display=args.x_display,
                                require_nvidia=args.require_nvidia)
        except Exception as exc:
            entry = {"house_path": house_path, "error": str(exc)[-300:]}
            print(f"[catalog] FAILED {house_path}: {str(exc)[-200:]}", flush=True)
        catalog[key] = entry
        os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)
        print("[catalog] " + json.dumps(entry, default=str)[:500], flush=True)
    print(f"CATALOG_OK {catalog_path} entries={len(catalog)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
