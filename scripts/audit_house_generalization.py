#!/usr/bin/env python3
"""Multi-house Generalization Gate -- Phase 1 audit (AUDIT ONLY, no fixes).

Runs a canonical, house-agnostic audit of the SpatialForge God View pipeline
over a structurally diverse ProcTHOR house benchmark set:

    house JSON -> canonical scene3d -> shared truth -> 3D / 2D / Unity

Checks (pure Python, no engine required):
  * structure / schema variation
  * scene3d build, bounds, walls, openings, objects
  * opening truth (host resolution, real cut, no sealed doorway)
  * wall fragmentation (sliver segments)
  * cross-view coordinate truth (world_space.js / scene2d.js via node probes)

Checks (AI2-THOR, --unity):
  * CreateHouse success
  * real third-party God camera render is non-empty / non-background
  * overview + follow camera framing

Checks (episodes, when a house has one):
  * FPV frame / pose / yaw / horizon temporal agreement
  * trajectory (setup break, MoveAhead vs heading)

Usage:
    python scripts/audit_house_generalization.py            # fast, no engine
    python scripts/audit_house_generalization.py --unity    # + real renders
    python scripts/audit_house_generalization.py --unity --x-display :0

This script only *reads*; it never mutates a house JSON or an episode record.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spatialforge.inspector.scene3d import build_scene3d  # noqa: E402

# ---------------------------------------------------------------------------
# Benchmark set: >= 12 structurally diverse houses (complexity-tagged).
# ---------------------------------------------------------------------------
BENCHMARK: List[Dict[str, str]] = [
    # small / simple
    {"file": "outputs/embodied_bc/houses/house_0008.json", "complexity": "small"},
    {"file": "outputs/embodied_bc/houses/house_0024.json", "complexity": "small"},
    {"file": "outputs/embodied_bc/houses/house_0006.json", "complexity": "small"},
    # medium
    {"file": "outputs/embodied_bc/houses/house_0004.json", "complexity": "medium"},
    {"file": "outputs/embodied_bc/houses/house_0005.json", "complexity": "medium"},
    {"file": "outputs/embodied_bc/houses/house_0026.json", "complexity": "medium"},
    {"file": "outputs/embodied_bc/houses/house_0027.json", "complexity": "medium"},
    {"file": "outputs/embodied_bc/houses/house_0043.json", "complexity": "medium"},
    # large / multi-room
    {"file": "outputs/embodied_bc/houses/house_0009.json", "complexity": "large"},
    {"file": "outputs/embodied_bc/houses/house_0015.json", "complexity": "large"},
    {"file": "outputs/embodied_bc/houses/house_0034.json", "complexity": "large"},
    {"file": "/root/autodl-tmp/val_house_2.json", "complexity": "large"},
    # geometry-complex
    {"file": "outputs/embodied_bc/houses/house_0011.json", "complexity": "complex"},
    {"file": "outputs/embodied_bc/houses/house_0017.json", "complexity": "complex"},
    {"file": "outputs/embodied_bc/houses/house_0018.json", "complexity": "complex"},
    {"file": "outputs/embodied_bc/houses/house_0040.json", "complexity": "complex"},
    # known-good control
    {"file": "/root/autodl-tmp/val_house_0.json", "complexity": "control"},
]

SLIVER_M = 0.02
POS_TOL = 0.05


def _dist_xz(a, b):
    return math.hypot(a[0] - b[0], a[2] - b[2])


def _area(house: Dict[str, Any]) -> float:
    total = 0.0
    for room in house.get("rooms", []):
        poly = room.get("floorPolygon", [])
        s = 0.0
        for i in range(len(poly)):
            p, q = poly[i], poly[(i + 1) % len(poly)]
            s += p.get("x", 0) * q.get("z", 0) - q.get("x", 0) * p.get("z", 0)
        total += abs(s) / 2.0
    return total


def structural_stats(house: Dict[str, Any]) -> Dict[str, Any]:
    rooms = house.get("rooms", [])
    walls = house.get("walls", [])
    doors = house.get("doors", [])
    windows = house.get("windows", [])
    objects = house.get("objects", [])
    md = house.get("metadata") or {}
    agent = md.get("agent") or {}
    return {
        "schema": md.get("schema"),
        "room_count": len(rooms),
        "wall_count": len(walls),
        "door_count": len(doors),
        "window_count": len(windows),
        "object_count": len(objects),
        "floor_area_m2": round(_area(house), 1),
        "empty_walls": sum(1 for w in walls if w.get("empty")),
        "has_agent": bool(agent.get("position")),
        "room_types": sorted({str(r.get("roomType")) for r in rooms}),
    }


# ---------------------------------------------------------------------------
# scene3d audit
# ---------------------------------------------------------------------------
def _bounds_finite(b) -> bool:
    try:
        vals = list(b["min"]) + list(b["max"])
        return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)
    except Exception:
        return False


def audit_scene3d(house: Dict[str, Any]) -> Dict[str, Any]:
    codes: List[str] = []
    notes: List[str] = []
    try:
        scene = build_scene3d(house)
    except Exception as exc:  # noqa: BLE001
        return {"codes": ["SCENE3D_MISSING"], "notes": [f"build_scene3d raised: {exc}"],
                "scene": None, "metrics": {}}

    rooms = scene.get("rooms", [])
    walls = scene.get("walls", [])
    doors = scene.get("doors", [])
    windows = scene.get("windows", [])
    objects = scene.get("objects", [])

    if not rooms:
        codes.append("SCENE3D_MISSING")
    if len(objects) != len(house.get("objects", [])):
        codes.append("OBJECT_MAPPING_ERROR")
        notes.append(f"objects {len(house.get('objects', []))}->{len(objects)}")

    bounds = scene.get("bounds", {})
    if not _bounds_finite(bounds):
        codes.append("BOUNDS_ERROR")
        notes.append("non-finite bounds")
    else:
        span = (bounds["max"][0] - bounds["min"][0], bounds["max"][2] - bounds["min"][2])
        if min(span) <= 0.5:
            codes.append("BOUNDS_ERROR")
            notes.append(f"degenerate span {span}")

    # wall orientation / length
    bad_walls = [w for w in walls if not math.isfinite(w.get("length", 0)) or w["length"] <= 1e-6]
    if bad_walls:
        codes.append("WALL_ORIENTATION_ERROR")
        notes.append(f"{len(bad_walls)} zero/NaN-length walls")

    # fragmentation: sliver segments
    total_segments = 0
    slivers = 0
    for w in walls:
        for s in w.get("segments", []):
            total_segments += 1
            if (s["end"] - s["start"]) < SLIVER_M:
                slivers += 1
    if slivers:
        codes.append("SCENE3D_FRAGMENTED")
        notes.append(f"{slivers}/{total_segments} sliver segments (<{SLIVER_M}m)")

    # opening truth
    walls_by_id = {str(w.get("id")): w for w in walls}
    sealed = 0
    unresolved = 0
    bad_dims = 0
    for o in doors + windows:
        host = walls_by_id.get(str(o.get("wall_id")))
        if o.get("position") is None or host is None or not o.get("local_x"):
            unresolved += 1
            continue
        if o.get("width") is None or o.get("height") is None or o["width"] <= 0 or o["height"] <= 0:
            bad_dims += 1
            continue
        # A doorway is sealed if a single host-wall segment covers its whole
        # local_x span over its whole y span.
        lo, hi = o["local_x"]
        y0 = float(o.get("position")[1])
        y1 = y0 + float(o["height"])
        for s in host.get("segments", []):
            if s["start"] <= lo + 1e-4 and s["end"] >= hi - 1e-4 \
                    and s["y0"] <= y0 + 1e-4 and s["y1"] >= y1 - 1e-4:
                sealed += 1
                break
    if unresolved:
        codes.append("OPENING_ERROR")
        notes.append(f"{unresolved} openings without host/position")
    if bad_dims:
        codes.append("OPENING_ERROR")
        notes.append(f"{bad_dims} openings with bad dimensions")
    if sealed:
        codes.append("OPENING_ERROR")
        notes.append(f"{sealed} openings sealed by a wall segment")

    # object positions
    obj_missing = sum(1 for o in objects if not o.get("position"))
    if obj_missing:
        codes.append("OBJECT_MAPPING_ERROR")
        notes.append(f"{obj_missing} objects without position")

    metrics = {
        "rooms": len(rooms),
        "walls": len(walls),
        "doors": len(doors),
        "windows": len(windows),
        "objects": len(objects),
        "segments": total_segments,
        "slivers": slivers,
        "sealed_openings": sealed,
        "unresolved_openings": unresolved,
        "bounds": bounds,
        "wall_height_min": min((w["height"] for w in walls), default=None),
        "wall_height_max": max((w["height"] for w in walls), default=None),
    }
    return {"codes": sorted(set(codes)), "notes": notes, "scene": scene, "metrics": metrics}


# ---------------------------------------------------------------------------
# cross-view coordinate truth (node probes over the shipped JS)
# ---------------------------------------------------------------------------
def audit_cross_view(scene: Dict[str, Any]) -> Dict[str, Any]:
    codes: List[str] = []
    notes: List[str] = []
    node = None
    try:
        node = subprocess.run(["which", "node"], capture_output=True, text=True).stdout.strip()
    except Exception:
        node = ""
    if not node:
        return {"codes": ["UNKNOWN"], "notes": ["node not available; skipped"], "metrics": {}}

    bounds = scene["bounds"]
    # asymmetric probe points: one object, one wall end, bounds corners
    pts = []
    for o in scene.get("objects", [])[:1]:
        if o.get("position"):
            pts.append(o["position"])
    for w in scene.get("walls", [])[:1]:
        pts.append(list(w["base"][0]))
    pts.append(list(bounds["min"]))
    pts.append(list(bounds["max"]))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        probe_in = td / "probe.json"
        probe_in.write_text(json.dumps({
            "points": pts,
            "dirs": [[1, 0, 0], [0, 0, 1]],
            "bounds": bounds,
            "yaws": [0, 90, 180, 270],
        }))
        ws_probe = REPO / "tests" / "tools" / "world_space_probe.mjs"
        r = subprocess.run(["node", str(ws_probe), str(probe_in)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            codes.append("UNKNOWN")
            notes.append(f"world_space probe failed: {r.stderr[-200:]}")
            return {"codes": codes, "notes": notes, "metrics": {}}
        out = json.loads(r.stdout)

        # round-trip identity (threeToWorld(worldToThree(p)) == p)
        for p, rt in zip(pts, out["roundtrip"]):
            if any(abs(p[i] - rt[i]) > 1e-6 for i in range(3)):
                codes.append("VIEW_MIRROR")
                notes.append("worldToThree/threeToWorld not involutive")
                break
        # +X must map to +X, +Z must map to -Z (canonical mirror)
        if out["dirs"][0] != [1, 0, 0]:
            codes.append("VIEW_MIRROR")
            notes.append("+X direction changed")
        if out["dirs"][1] != [0, 0, -1]:
            codes.append("VIEW_MIRROR")
            notes.append("+Z direction not mirrored")
        # yaw 0 -> (0,-1) in three xz; yaw 90 -> (1,0)
        yaw_dirs = out["yaw_dirs"]
        if abs(yaw_dirs[0][0]) > 1e-9 or abs(yaw_dirs[0][1] + 1) > 1e-9:
            codes.append("VIEW_MIRROR")
            notes.append(f"yaw0 forward wrong: {yaw_dirs[0]}")
        if abs(yaw_dirs[1][0] - 1) > 1e-9 or abs(yaw_dirs[1][1]) > 1e-9:
            codes.append("VIEW_MIRROR")
            notes.append(f"yaw90 forward wrong: {yaw_dirs[1]}")

        # 2D projection: +Z must go up (smaller screen Y)
        payload = {"scene": scene, "episode": {"trace": [], "agent": None,
                                                "targets": [], "spawn": None}}
        payload_f = td / "payload.json"
        payload_f.write_text(json.dumps(payload))
        s2_probe = REPO / "tests" / "tools" / "scene2d_probe.mjs"
        r2 = subprocess.run(["node", str(s2_probe), str(payload_f)],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            codes.append("UNKNOWN")
            notes.append(f"scene2d probe failed: {r2.stderr[-200:]}")
        else:
            m = json.loads(r2.stdout)
            if m.get("bounds_source") != "scene":
                codes.append("BOUNDS_ERROR")
                notes.append(f"2D fell back to {m.get('bounds_source')} bounds")

    return {"codes": sorted(set(codes)), "notes": notes, "metrics": {"probe_points": len(pts)}}


# ---------------------------------------------------------------------------
# episode pose / trajectory truth
# ---------------------------------------------------------------------------
def _episode_house_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for results in glob.glob(str(REPO / "outputs" / "**" / "results" / "*.jsonl"), recursive=True):
        try:
            with open(results, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ep, house = row.get("episode_id"), row.get("house")
                    if ep and house:
                        mapping[str(ep)] = str(house)
        except Exception:
            continue
    return mapping


def _first_episode_for(house_path: str, mapping: Dict[str, str]) -> Optional[str]:
    for ep, hp in mapping.items():
        if os.path.abspath(hp) == os.path.abspath(house_path):
            return ep
    return None


_EPISODE_ROOTS = [
    ("outputs/episodes_postfix", 4),
    ("outputs/embodied_bc/model_rollout", 3),
    ("outputs/embodied_bc/model_live", 3),
    ("outputs/embodied_bc/dataset_v2", 2),
    ("outputs/embodied_bc/dataset", 1),
    ("outputs/episodes", 1),
]


def _episode_candidates(house_path: str, mapping: Dict[str, str]) -> List[str]:
    """Episodes mapped to a house, best (post-fix) first."""
    cands: List[tuple] = []
    target = os.path.abspath(house_path)
    for rel, prio in _EPISODE_ROOTS:
        root = REPO / rel
        if not root.is_dir():
            continue
        for d in sorted(root.glob("ep-*")):
            p = d / "privileged_research_record.json"
            if not p.is_file():
                continue
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            hp = rec.get("house_path")
            if hp:
                if os.path.abspath(hp) == target:
                    cands.append((prio, d.name))
            else:
                mapped = mapping.get(d.name)
                if mapped and os.path.abspath(mapped) == target:
                    cands.append((prio, d.name))
    cands.sort(key=lambda x: (-x[0], x[1]))
    return [e for _, e in cands]


def audit_episode(episode_id: str) -> Dict[str, Any]:
    codes: List[str] = []
    notes: List[str] = []
    from spatialforge.inspector.embodied_session import EmbodiedSession

    sess = EmbodiedSession("deterministic", seed=0)
    roots = [p for p in [
        REPO / "outputs" / "embodied_bc" / "model_rollout",
        REPO / "outputs" / "embodied_bc" / "model_live",
        REPO / "outputs" / "embodied_bc" / "dataset",
        REPO / "outputs" / "embodied_bc" / "dataset_v2",
        REPO / "outputs" / "episodes_postfix",
        REPO / "outputs" / "episodes",
    ] if p.is_dir()]
    sess.set_episodes_roots(roots)
    try:
        out = sess.load_episode(episode_id)
    except Exception as exc:  # noqa: BLE001
        return {"codes": ["UNKNOWN"], "notes": [f"load failed: {exc}"], "metrics": {}}

    tl = out["timeline"]
    dec = out.get("decisions") or []
    metrics: Dict[str, Any] = {"timeline_len": len(tl), "decisions": len(dec)}

    # 1. temporal agreement: entry N frame_N is the same env event as steps[N-1]
    mismatches = 0
    for i, e in enumerate(tl):
        if i == 0:
            continue
        if e.get("frame") != f"frame_{i:04d}.png":
            mismatches += 1
    if mismatches:
        codes.append("POSE_MISMATCH")
        notes.append(f"{mismatches} entries whose frame != idx")

    # 2. initial pose must equal the authoritative pre-action pose
    if dec:
        init_pose = tl[0].get("agent") or {}
        d0 = dec[0].get("agent") or {}
        if init_pose.get("rotation_yaw_deg") != d0.get("rotation_yaw_deg") \
                or _dist_xz(init_pose.get("position", [0, 0, 0]),
                            d0.get("position", [0, 0, 0])) > POS_TOL:
            codes.append("POSE_MISMATCH")
            notes.append("timeline[0] != decisions[0] (initial pose off-by-one)")

    # 3. trajectory: MoveAhead displacement must agree with heading
    traj_bad = 0
    for i in range(1, len(tl)):
        e = tl[i]
        if e.get("action") != "MoveAhead" or e.get("action_success") is not True:
            continue
        prev = tl[i - 1].get("agent") or {}
        cur = e.get("agent") or {}
        if not prev.get("position") or not cur.get("position"):
            continue
        yaw = math.radians(float(prev.get("rotation_yaw_deg") or 0.0))
        fwd = (math.sin(yaw), math.cos(yaw))
        dx = cur["position"][0] - prev["position"][0]
        dz = cur["position"][2] - prev["position"][2]
        if dx * fwd[0] + dz * fwd[1] <= 0 and math.hypot(dx, dz) > POS_TOL:
            traj_bad += 1
    if traj_bad:
        codes.append("TRAJECTORY_MISMATCH")
        notes.append(f"{traj_bad} MoveAhead transitions against heading")

    # 4. legacy / sensor
    header = out.get("header") or {}
    if header.get("render_quality") in (None, "Low"):
        codes.append("LEGACY_SENSOR_ONLY")
        notes.append(f"render_quality={header.get('render_quality')}")

    metrics["initial_source"] = tl[0].get("agent_source")
    return {"codes": sorted(set(codes)), "notes": notes, "metrics": metrics}


# ---------------------------------------------------------------------------
# Unity God View render
# ---------------------------------------------------------------------------
def audit_unity_batch(house_paths: List[str], x_display: str,
                      width: int = 640, height: int = 480) -> Dict[str, Dict[str, Any]]:
    """Render each house through a real AI2-THOR third-party camera.

    Reuses one engine (CreateHouse swaps the house) to keep the audit fast.
    """
    os.environ.setdefault("DISPLAY", x_display)
    from spatialforge.embodied.backends.procthor import ProcTHORBackend
    from spatialforge.inspector.god_camera import GOD_VIEW_SKYBOX_COLOR
    import numpy as np

    backend = ProcTHORBackend(
        house=house_paths[0], width=width, height=height,
        x_display=x_display, require_nvidia=False,
    )
    backend.load_house()
    controller = backend.controller
    results: Dict[str, Dict[str, Any]] = {}
    bg = np.array([0x10, 0x14, 0x18], dtype=float)

    def _frame_stats(frame):
        arr = np.asarray(frame).astype(float)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        dist = np.abs(arr - bg).sum(axis=2)
        content = float((dist > 30).mean())
        return {
            "shape": list(arr.shape),
            "mean": round(float(arr.mean()), 2),
            "content_frac": round(content, 4),
            "nonblack_frac": round(float((arr.max(axis=2) > 12).mean()), 4),
        }

    for i, hp in enumerate(house_paths):
        rec: Dict[str, Any] = {"codes": [], "notes": [], "metrics": {}}
        print(f"[unity] ({i+1}/{len(house_paths)}) {Path(hp).stem}", flush=True)
        try:
            if i > 0:
                house = json.loads(Path(hp).read_text(encoding="utf-8"))
                ev = controller.step(dict(action="CreateHouse", house=house, renderImage=False))
                if not ev.metadata.get("lastActionSuccess", False):
                    rec["codes"].append("HOUSE_LOAD_FAIL")
                    rec["notes"].append(str(ev.metadata.get("errorMessage")))
                    results[hp] = rec
                    continue
                backend._house_data = house
                backend._event = ev
            # teleport to agent start
            house = backend._house_data
            agent = (house.get("metadata") or {}).get("agent") or {}
            pos = agent.get("position") or {"x": 0, "y": 0.9, "z": 0}
            rot = agent.get("rotation") or {"x": 0, "y": 0, "z": 0}
            tp = controller.step(dict(
                action="TeleportFull",
                position={"x": float(pos["x"]), "y": float(pos["y"]), "z": float(pos["z"])},
                rotation={"x": 0.0, "y": float(rot.get("y", 0.0)), "z": 0.0},
                horizon=float(agent.get("horizon", 0.0)),
                standing=True,
            ))
            if not tp.metadata.get("lastActionSuccess", False):
                rec["codes"].append("HOUSE_LOAD_FAIL")
                rec["notes"].append("TeleportFull failed")
                results[hp] = rec
                continue

            # bounds from rooms
            xs, zs = [], []
            for r in house.get("rooms", []):
                for p in r.get("floorPolygon", []):
                    xs.append(float(p.get("x", 0.0)))
                    zs.append(float(p.get("z", 0.0)))
            cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
            span = max(max(xs) - min(xs), max(zs) - min(zs), 4.0)

            # overview (matches GodCameraRenderer._camera_for overview preset)
            ov = controller.step(dict(
                action="AddThirdPartyCamera",
                position={"x": cx, "y": max(6.0, span * 0.6), "z": cz - 0.1},
                rotation={"x": 60.0, "y": 0.0, "z": 0.0},
                fieldOfView=60.0,
                skyboxColor=GOD_VIEW_SKYBOX_COLOR,
            ))
            if not ov.metadata.get("lastActionSuccess", False):
                rec["codes"].append("UNITY_RENDER_FAIL")
                rec["notes"].append(str(ov.metadata.get("errorMessage")))
            else:
                f = ov.third_party_camera_frames[0]
                st = _frame_stats(f)
                rec["metrics"]["overview"] = st
                if st["mean"] < 3 or st["nonblack_frac"] < 0.01:
                    rec["codes"].append("UNITY_RENDER_FAIL")
                    rec["notes"].append("overview frame blank/black")
                elif st["content_frac"] < 0.02:
                    rec["codes"].append("UNITY_CAMERA_BAD_BOUNDS")
                    rec["notes"].append(
                        f"overview content {st['content_frac']:.3f} (house not framed)")

            # follow camera (behind agent)
            yaw = math.radians(float(agent.get("rotation", {}).get("y", 0.0) or 0.0))
            fu = controller.step(dict(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                position={
                    "x": float(pos["x"]) - math.sin(yaw) * 2.2,
                    "y": float(pos["y"]) + 2.0,
                    "z": float(pos["z"]) - math.cos(yaw) * 2.2,
                },
                rotation={"x": 32.0, "y": math.degrees(yaw), "z": 0.0},
                fieldOfView=70.0,
                skyboxColor=GOD_VIEW_SKYBOX_COLOR,
            ))
            if fu.metadata.get("lastActionSuccess", False):
                st = _frame_stats(fu.third_party_camera_frames[0])
                rec["metrics"]["follow"] = st
                if st["mean"] < 3 or st["nonblack_frac"] < 0.01:
                    rec["codes"].append("UNITY_RENDER_FAIL")
                    rec["notes"].append("follow frame blank/black")
            else:
                rec["codes"].append("UNITY_RENDER_FAIL")
                rec["notes"].append("follow camera update failed")
            rec["metrics"]["span"] = round(span, 2)
        except Exception as exc:  # noqa: BLE001
            rec["codes"].append("UNITY_RENDER_FAIL")
            rec["notes"].append(f"exception: {exc}")
        results[hp] = rec

    try:
        controller.stop()
    except Exception:
        pass
    return results


# ---------------------------------------------------------------------------
# overall classification
# ---------------------------------------------------------------------------
def classify(house_codes: List[str], unity_codes: List[str],
             episode_codes: List[str]) -> str:
    allc = set(house_codes) | set(unity_codes) | set(episode_codes)
    hard = {"HOUSE_LOAD_FAIL", "UNITY_RENDER_FAIL", "SCENE3D_MISSING", "BOUNDS_ERROR",
            "VIEW_MIRROR", "OPENING_ERROR", "OBJECT_MAPPING_ERROR",
            "WALL_ORIENTATION_ERROR"}
    if allc & hard:
        return "FAIL"
    soft = {"SCENE3D_FRAGMENTED", "UNITY_CAMERA_BAD_BOUNDS", "POSE_MISMATCH",
            "TRAJECTORY_MISMATCH", "LEGACY_SENSOR_ONLY", "SCHEMA_VARIANT"}
    if allc & soft:
        return "DEGRADED"
    return "PASS"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unity", action="store_true", help="run real AI2-THOR renders")
    ap.add_argument("--x-display", default=os.environ.get("DISPLAY") or ":0")
    ap.add_argument("--out", default="outputs/diagnostics/generalization")
    args = ap.parse_args()

    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    mapping = _episode_house_map()
    rows: List[Dict[str, Any]] = []

    for spec in BENCHMARK:
        hp = spec["file"]
        path = REPO / hp if not os.path.isabs(hp) else Path(hp)
        row: Dict[str, Any] = {
            "house": path.stem,
            "path": str(path),
            "complexity": spec["complexity"],
        }
        if not path.is_file():
            row.update({"codes": ["HOUSE_LOAD_FAIL"], "overall": "FAIL",
                        "notes": ["house file not found"], "stats": {}, "metrics": {}})
            rows.append(row)
            continue
        try:
            house = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            row.update({"codes": ["HOUSE_LOAD_FAIL"], "overall": "FAIL",
                        "notes": [str(exc)], "stats": {}, "metrics": {}})
            rows.append(row)
            continue

        row["stats"] = structural_stats(house)
        scene_audit = audit_scene3d(house)
        cv = audit_cross_view(scene_audit["scene"]) if scene_audit["scene"] else {
            "codes": [], "notes": [], "metrics": {}}
        row["scene_metrics"] = scene_audit["metrics"]

        ep_id = _first_episode_for(str(path), mapping)
        candidates = _episode_candidates(str(path), mapping)
        if candidates:
            ep_id = candidates[0]
        row["episode_id"] = ep_id
        row["episode_candidates"] = candidates[:5]
        ep = audit_episode(ep_id) if ep_id else {"codes": [], "notes": [], "metrics": {}}

        row["codes"] = sorted(set(scene_audit["codes"] + cv["codes"] + ep["codes"]))
        row["notes"] = scene_audit["notes"] + cv["notes"] + ep["notes"]
        row["episode_metrics"] = ep["metrics"]
        row["overall"] = classify(scene_audit["codes"], cv["codes"], ep["codes"])
        rows.append(row)

    # Unity pass (separate so the fast audit is always available)
    if args.unity:
        print(f"[unity] rendering {len(rows)} houses on {args.x_display} ...", flush=True)
        paths = [r["path"] for r in rows if Path(r["path"]).is_file()]
        t0 = time.time()
        unity = audit_unity_batch(paths, args.x_display)
        print(f"[unity] done in {time.time() - t0:.1f}s", flush=True)
        for r in rows:
            u = unity.get(r["path"], {"codes": ["UNKNOWN"], "notes": ["not rendered"],
                                      "metrics": {}})
            r["unity"] = u
            r["codes"] = sorted(set(r["codes"]) | set(u["codes"]))
            r["notes"] = r["notes"] + u["notes"]
            r["overall"] = classify(r["codes"], u["codes"], [])

    report = {"schema": "generalization_audit.v1", "rows": rows,
              "episode_house_map_size": len(mapping)}
    (outdir / "audit.json").write_text(json.dumps(report, indent=2))

    # failure frequency
    freq: Dict[str, int] = {}
    for r in rows:
        for c in r["codes"]:
            freq[c] = freq.get(c, 0) + 1
    report["failure_frequency"] = dict(sorted(freq.items(), key=lambda kv: -kv[1]))
    (outdir / "audit.json").write_text(json.dumps(report, indent=2))

    # markdown matrix
    def cell(v):
        return {"PASS": "PASS", "DEGRADED": "DEGRADED", "FAIL": "FAIL"}.get(v, v)

    lines = ["| house | complexity | rooms | walls | doors | win | objs | area m2 | "
             "3D | unity | pose | overall | reasons |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        st = r.get("stats") or {}
        if "unity" not in r:
            u = "NOT_RUN"
        else:
            unity_codes = (r["unity"] or {}).get("codes", [])
            u = "PASS" if not unity_codes else ("FAIL" if any(
                c in ("UNITY_RENDER_FAIL", "HOUSE_LOAD_FAIL", "UNITY_CAMERA_BAD_BOUNDS")
                for c in unity_codes) else "DEGRADED")
        scene3d = "FAIL" if any(c in ("SCENE3D_MISSING", "BOUNDS_ERROR", "OPENING_ERROR",
                                      "OBJECT_MAPPING_ERROR", "WALL_ORIENTATION_ERROR",
                                      "VIEW_MIRROR") for c in r["codes"]) else (
            "DEGRADED" if "SCENE3D_FRAGMENTED" in r["codes"] else "PASS")
        pose = "FAIL" if any(c in ("POSE_MISMATCH", "TRAJECTORY_MISMATCH")
                             for c in r["codes"]) else "PASS"
        lines.append(
            f"| {r['house']} | {r['complexity']} | {st.get('room_count','-')} | "
            f"{st.get('wall_count','-')} | {st.get('door_count','-')} | "
            f"{st.get('window_count','-')} | {st.get('object_count','-')} | "
            f"{st.get('floor_area_m2','-')} | {scene3d} | {u} | {pose} | "
            f"{cell(r['overall'])} | {', '.join(r['codes']) or '-'} |")
    (outdir / "matrix.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\nFailure frequency:")
    for k, v in report["failure_frequency"].items():
        print(f"  {k}: {v}/{len(rows)}")
    print(f"\nReport: {outdir/'audit.json'}")
    print(f"Matrix: {outdir/'matrix.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
