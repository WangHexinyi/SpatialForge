"""3D semantic scene payload for the researcher God View.

Pure Python (no ai2thor / no torch). Converts an official ProcTHOR house dict
(+ optional live AI2-THOR object metadata) into a compact, render-ready 3D
scene description:

* rooms -> floor polygons
* walls -> base segment + height (extruded by the frontend)
* doors / windows -> hole geometry + world position when available
* objects -> position/rotation/category (+ real axis-aligned bounding box when
  AI2-THOR metadata is present; otherwise explicitly ``None``)

Fields that the source metadata does not provide are reported in
``unavailable`` instead of being fabricated.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

SCENE_SCHEMA = "spatialforge_scene3d.v1"

#: openings shorter/thinner than this are not rendered as cuts (numerical noise).
_WALL_EPS = 1e-4
#: lateral tolerance for deciding two wall records are the same physical plane.
_WALL_COLLINEAR_TOL = 0.05


def _pt(p: Dict[str, Any]) -> List[float]:
    return [float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))]


def _dist_xz(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[2] - b[2])


def _yaw_from_segment(a: Sequence[float], b: Sequence[float]) -> float:
    """Thor yaw (degrees) whose forward is (sin yaw, cos yaw) in the x,z plane."""
    return math.degrees(math.atan2(b[0] - a[0], b[2] - a[2])) % 360.0


def _wall_direction(wall: Dict[str, Any]) -> Optional[Sequence[float]]:
    """Unit xz direction of a wall from ``base[0]`` to ``base[1]``."""
    a, b = wall["base"]
    dx, dz = b[0] - a[0], b[2] - a[2]
    length = math.hypot(dx, dz)
    if length <= _WALL_EPS:
        return None
    return (dx / length, dz / length)


def _wall_local_span(
    wall: Dict[str, Any], p0: Sequence[float], p1: Sequence[float]
) -> Optional[tuple]:
    """Project a world xz segment onto a wall axis.

    ``p0`` / ``p1`` are ``(x, z)`` world points. Returns the clamped
    ``(start, end)`` interval in wall-local coordinates when the segment lies on
    the wall's plane (collinear within tolerance) and overlaps it; ``None``
    otherwise. Duplicated ProcTHOR wall records for the same physical plane may
    be reversed, so the projection -- not the raw hole coordinates -- is what
    maps one opening onto both sides.
    """
    direction = _wall_direction(wall)
    if direction is None:
        return None
    a = wall["base"][0]
    ux, uz = direction
    d0 = abs((p0[0] - a[0]) * uz - (p0[1] - a[2]) * ux)
    d1 = abs((p1[0] - a[0]) * uz - (p1[1] - a[2]) * ux)
    if max(d0, d1) > _WALL_COLLINEAR_TOL:
        return None
    t0 = (p0[0] - a[0]) * ux + (p0[1] - a[2]) * uz
    t1 = (p1[0] - a[0]) * ux + (p1[1] - a[2]) * uz
    lo, hi = sorted((t0, t1))
    lo = max(0.0, lo)
    hi = min(wall["length"], hi)
    if hi - lo <= _WALL_EPS:
        return None
    return lo, hi


def _opening_cut(
    opening: Dict[str, Any], walls_by_id: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """World-space vertical cut one door/window makes in its host wall plane."""
    host = walls_by_id.get(str(opening.get("wall_id")))
    if host is None:
        return None
    direction = _wall_direction(host)
    if direction is None:
        return None
    local_x = opening.get("local_x")
    position = opening.get("position")
    height = opening.get("height")
    if not local_x or position is None or height is None:
        return None
    u0, u1 = float(local_x[0]), float(local_x[1])
    a = host["base"][0]
    p0 = (a[0] + direction[0] * u0, a[2] + direction[1] * u0)
    p1 = (a[0] + direction[0] * u1, a[2] + direction[1] * u1)
    y0 = float(position[1])
    return {
        "opening_id": opening.get("id"),
        "host_id": str(host.get("id")),
        "p0": p0,
        "p1": p1,
        "y0": y0,
        "y1": y0 + float(height),
    }


def _merge_bands(covered: Sequence[tuple]) -> List[tuple]:
    merged: List[tuple] = []
    for a, b in sorted(covered):
        if b <= a:
            continue
        if merged and a <= merged[-1][1] + _WALL_EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _wall_segments(wall: Dict[str, Any], cuts: Sequence[tuple]) -> List[Dict[str, float]]:
    """Split a wall rectangle into solid wall-local segments around cuts.

    ``cuts`` are ``(start, end, y0, y1)`` wall-local rectangles. The result is
    the complement of their union inside ``[0, length] x [0, height]``:
    full-height side segments plus sill/lintel bands for partial-height windows.
    A wall with no cuts yields exactly one segment covering the whole wall.
    """
    length = float(wall.get("length") or 0.0)
    height = float(wall.get("height") or 0.0)
    if length <= _WALL_EPS or height <= _WALL_EPS:
        return []
    valid = [
        (max(0.0, c[0]), min(length, c[1]), max(0.0, c[2]), min(height, c[3]))
        for c in cuts
    ]
    valid = [c for c in valid if c[1] - c[0] > _WALL_EPS and c[3] - c[2] > _WALL_EPS]
    if not valid:
        return [{"start": 0.0, "end": round(length, 4), "y0": 0.0, "y1": round(height, 4)}]
    boundaries = sorted({0.0, length} | {c[0] for c in valid} | {c[1] for c in valid})
    segments: List[Dict[str, float]] = []
    for lo, hi in zip(boundaries, boundaries[1:]):
        if hi - lo <= _WALL_EPS:
            continue
        mid = (lo + hi) / 2.0
        covered = [(c[2], c[3]) for c in valid if c[0] - _WALL_EPS <= mid <= c[1] + _WALL_EPS]
        cursor = 0.0
        for band_lo, band_hi in _merge_bands(covered):
            if band_lo > cursor + _WALL_EPS:
                segments.append({
                    "start": round(lo, 4), "end": round(hi, 4),
                    "y0": round(cursor, 4), "y1": round(band_lo, 4),
                })
            cursor = max(cursor, band_hi)
        if cursor < height - _WALL_EPS:
            segments.append({
                "start": round(lo, 4), "end": round(hi, 4),
                "y0": round(cursor, 4), "y1": round(height, 4),
            })
    return segments


def _wall_entry(wall: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    poly = [_pt(p) for p in wall.get("polygon", [])]
    if len(poly) < 2:
        return None
    ys = [p[1] for p in poly]
    y_min, y_max = min(ys), max(ys)
    bottom = [p for p in poly if abs(p[1] - y_min) < 1e-6]
    if len(bottom) < 2:
        bottom = poly[:2]
    base = bottom[:2]
    height = max(y_max - y_min, 0.0)
    return {
        "id": wall.get("id"),
        "room_id": wall.get("roomId"),
        "base": base,
        "height": round(height, 4),
        "yaw_deg": round(_yaw_from_segment(base[0], base[1]), 2),
        "length": round(_dist_xz(base[0], base[1]), 4),
        "empty": bool(wall.get("empty", False)),
    }


def _opening_entry(opening: Dict[str, Any], kind: str,
                   walls_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Canonical door/window entry.

    ``holePolygon`` is in the *host wall's local frame*: x = distance along the
    wall from the host's ``base[0]``, y = height above the floor. The world
    ``position`` is derived from the host wall + hole (never from
    ``assetPosition``, which is wall-local in ProcTHOR JSON), so the 2D and 3D
    views consume the same world coordinates.
    """
    hole = [_pt(p) for p in opening.get("holePolygon", [])]
    wall0 = walls_by_id.get(str(opening.get("wall0")))
    wall1 = walls_by_id.get(str(opening.get("wall1")))
    host = wall0 or wall1
    width = height = base_y = None
    local_x = None
    if len(hole) >= 2:
        u0, u1 = sorted((hole[0][0], hole[1][0]))
        v0, v1 = sorted((hole[0][1], hole[1][1]))
        width = u1 - u0
        height = v1 - v0
        base_y = v0
        local_x = [round(u0, 4), round(u1, 4)]
    position = None
    if host is not None and local_x is not None:
        direction = _wall_direction(host)
        if direction is not None:
            a = host["base"][0]
            mid = (local_x[0] + local_x[1]) / 2.0
            position = [
                round(a[0] + direction[0] * mid, 4),
                round(base_y, 4),
                round(a[2] + direction[1] * mid, 4),
            ]
    yaw = host.get("yaw_deg") if host is not None else None
    return {
        "id": opening.get("id"),
        "kind": kind,
        "asset_id": opening.get("assetId"),
        "room0": opening.get("room0"),
        "room1": opening.get("room1"),
        "position": position,
        "width": round(width, 4) if width is not None else None,
        "height": round(height, 4) if height is not None else None,
        "yaw_deg": yaw,
        "openable": opening.get("openable"),
        "openness": opening.get("openness"),
        "hole": hole,
        "wall_id": host.get("id") if host is not None else None,
        "wall_ids": [w.get("id") for w in (wall0, wall1) if w is not None],
        "local_x": local_x,
        "asset_position": opening.get("assetPosition"),
    }


def _aabb_from_thor(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    aabb = obj.get("axisAlignedBoundingBox")
    if not isinstance(aabb, dict):
        return None
    center = aabb.get("center")
    size = aabb.get("size")
    if not center or not size:
        return None
    return {
        "center": [float(center.get("x", 0.0)), float(center.get("y", 0.0)), float(center.get("z", 0.0))],
        "size": [float(size.get("x", 0.0)), float(size.get("y", 0.0)), float(size.get("z", 0.0))],
    }


def _category_from_object_id(object_id: str) -> str:
    return str(object_id).split("|", 1)[0]


def build_scene3d(
    house: Dict[str, Any],
    thor_objects: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the researcher 3D scene payload for one house."""
    unavailable: List[str] = []
    rooms: List[Dict[str, Any]] = []
    for room in house.get("rooms", []):
        poly = [_pt(p) for p in room.get("floorPolygon", [])]
        if not poly:
            unavailable.append(f"room {room.get('id')}: empty floorPolygon")
            continue
        rooms.append({
            "id": room.get("id"),
            "type": room.get("roomType") or room.get("type") or "Unknown",
            "floor_polygon": poly,
            "floor_y": min(p[1] for p in poly),
        })

    walls = [w for w in (_wall_entry(w) for w in house.get("walls", [])) if w]
    walls_by_id = {str(w["id"]): w for w in walls}

    doors = [_opening_entry(d, "door", walls_by_id) for d in house.get("doors", [])]
    windows = [_opening_entry(w, "window", walls_by_id) for w in house.get("windows", [])]

    # Derive render-only wall segments: the canonical wall entries (id / base /
    # height / length) are never mutated, each gains a `segments` list that cuts
    # out the linked door/window openings. Walls with no openings keep exactly
    # one full-wall segment, so unrelated geometry is unchanged. Openings are
    # applied to every collinear wall record, which is what keeps ProcTHOR's
    # duplicated (often reversed) wall records from sealing the same doorway.
    openings = doors + windows
    for opening in openings:
        if opening.get("wall_id") is None:
            unavailable.append(
                f"{opening.get('kind')} {opening.get('id')}: no host wall "
                f"(wall0={opening.get('wall0')!r} wall1={opening.get('wall1')!r})"
            )
    cuts = [c for c in (_opening_cut(o, walls_by_id) for o in openings) if c]
    for wall in walls:
        wall_cuts = []
        for cut in cuts:
            span = _wall_local_span(wall, cut["p0"], cut["p1"])
            if span is not None:
                wall_cuts.append((span[0], span[1], cut["y0"], cut["y1"]))
        wall["segments"] = _wall_segments(wall, wall_cuts)

    thor_by_id = {str(o.get("objectId")): o for o in (thor_objects or [])}
    objects: List[Dict[str, Any]] = []
    for obj in house.get("objects", []):
        oid = str(obj.get("id"))
        rot = obj.get("rotation") or {}
        pos = obj.get("position") or {}
        thor = thor_by_id.get(oid)
        aabb = _aabb_from_thor(thor) if thor else None
        objects.append({
            "id": oid,
            "category": _category_from_object_id(oid),
            "asset_id": obj.get("assetId"),
            "position": _pt(pos) if pos else None,
            "rotation_yaw_deg": float(rot.get("y", 0.0)),
            "kinematic": bool(obj.get("kinematic", False)),
            "aabb": aabb,
            "visible": bool(thor.get("visible", False)) if thor else None,
        })
    if thor_objects is None:
        unavailable.append(
            "AI2-THOR per-object bounding boxes / visibility (house JSON only)"
        )
    elif any(o["aabb"] is None for o in objects):
        unavailable.append("bounding boxes for objects missing from live metadata")

    agent_meta = (house.get("metadata") or {}).get("agent") or {}
    agent_start = None
    if agent_meta.get("position"):
        agent_start = {
            "position": _pt(agent_meta["position"]),
            "yaw_deg": float((agent_meta.get("rotation") or {}).get("y", 0.0)),
            "horizon_deg": float(agent_meta.get("horizon", 0.0)),
        }

    xs = [p[0] for r in rooms for p in r["floor_polygon"]]
    zs = [p[2] for r in rooms for p in r["floor_polygon"]]
    ys = [p[1] for r in rooms for p in r["floor_polygon"]] + [w["height"] for w in walls]
    bounds = {
        "min": [min(xs), min(ys), min(zs)] if xs else [0.0, 0.0, 0.0],
        "max": [max(xs), max(ys), max(zs)] if xs else [0.0, 0.0, 0.0],
    }

    return {
        "schema": SCENE_SCHEMA,
        "source": "procthor_house_json+ai2thor_metadata" if thor_objects is not None
        else "procthor_house_json",
        "house_id": str(house.get("houseId") or ""),
        "units": "meters",
        "up_axis": "y",
        "bounds": bounds,
        "rooms": rooms,
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "objects": objects,
        "agent_start": agent_start,
        "unavailable": unavailable,
    }


def episode_scene_payload(
    scene: Dict[str, Any],
    *,
    spawn: Optional[Dict[str, Any]] = None,
    timeline: Optional[Sequence[Dict[str, Any]]] = None,
    target_positions: Optional[Sequence[Sequence[float]]] = None,
    target_object_ids: Optional[Sequence[str]] = None,
    teacher_plan: Optional[Dict[str, Any]] = None,
    terminal: bool = False,
    frames: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Attach episode truth (trajectory / spawn / targets / FOV poses) to a scene.

    Trajectory invariant: a drawn segment only connects *consecutive, real
    executed, valid* environment transitions. Environment setup (spawn / reset /
    teleport / teacher view-pose probing) and unexplained displacement break the
    line instead of drawing a wall-crossing straight segment. This is what makes
    the God View trajectory truthful: ``trace_segments`` never contains a
    setup->model or teleport->model edge.
    """
    trace: List[List[float]] = []
    agent = None
    for entry in timeline or []:
        a = entry.get("agent") or {}
        pos = a.get("position")
        if pos and (not trace or trace[-1] != list(pos)):
            trace.append(list(pos))
        if pos:
            agent = {
                "position": list(pos),
                "rotation_yaw_deg": a.get("rotation_yaw_deg"),
                "camera_horizon_deg": a.get("camera_horizon_deg"),
                "step": entry.get("step"),
            }
    if spawn and spawn.get("position"):
        if not trace or trace[0] != list(spawn["position"]):
            trace.insert(0, list(spawn["position"]))
        if agent is None:
            agent = {
                "position": list(spawn["position"]),
                "rotation_yaw_deg": spawn.get("yaw"),
                "camera_horizon_deg": spawn.get("horizon"),
                "step": 0,
            }

    segments, breaks = _trace_segments(timeline, spawn)

    targets = []
    for i, pos in enumerate(target_positions or []):
        targets.append({
            "position": list(pos),
            "object_id": (target_object_ids or [None] * (i + 1))[i] if target_object_ids else None,
        })
    return {
        "scene": scene,
        "episode": {
            "trace": trace,
            "trace_segments": segments,
            "trace_breaks": breaks,
            "agent": agent,
            "spawn": spawn,
            "targets": targets,
            "teacher_plan": teacher_plan,
            "terminal": bool(terminal),
            "frames": list(frames or []),
        },
    }


#: model actions that legitimately translate the agent in the x,z plane.
_TRANSLATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
#: non-translation actions must not move the agent at all.
_MOVE_MAX_M = 0.75
_POS_TOL_M = 0.05
#: action origins that are environment setup, never model trajectory edges.
_SETUP_ORIGINS = {"setup", "spawn", "reset", "teleport", "teacher"}


def _break_reason(
    prev_pos: Sequence[float],
    pos: Sequence[float],
    action: Optional[str],
    origin: Optional[str],
    prev_origin: Optional[str],
) -> Optional[str]:
    moved = _dist_xz(prev_pos, pos)
    if moved <= _POS_TOL_M:
        return None  # same pose; nothing to connect or break
    if origin in _SETUP_ORIGINS:
        return "setup_origin"
    if action is None:
        return "missing_action"
    limit = _MOVE_MAX_M if action in _TRANSLATION_ACTIONS else _POS_TOL_M
    if moved > limit:
        return "unexplained_displacement"
    return None


def _trace_segments(
    timeline: Optional[Sequence[Dict[str, Any]]],
    spawn: Optional[Dict[str, Any]],
) -> tuple:
    """Split the recorded poses into contiguous valid-transition segments."""
    entries: List[Dict[str, Any]] = []
    if spawn and spawn.get("position"):
        entries.append({
            "idx": -1, "step": 0, "agent": {"position": list(spawn["position"])},
            "action": None, "action_origin": "spawn",
        })
    for entry in timeline or []:
        a = entry.get("agent") or {}
        if a.get("position"):
            entries.append(entry)

    segments: List[List[List[float]]] = []
    breaks: List[Dict[str, Any]] = []
    cur: List[List[float]] = []
    prev_pos = None
    prev_action = None
    prev_origin = None
    prev_idx = None
    for entry in entries:
        pos = list((entry.get("agent") or {}).get("position"))
        action = entry.get("action")
        origin = entry.get("action_origin")
        if prev_pos is None:
            cur = [pos]
        else:
            reason = _break_reason(prev_pos, pos, action, origin, prev_origin)
            if reason:
                if cur:
                    segments.append(cur)
                breaks.append({"after_idx": prev_idx, "reason": reason})
                cur = [pos]
            elif cur and _dist_xz(cur[-1], pos) > 1e-4:
                cur.append(pos)
        prev_pos = pos
        prev_action = action
        prev_origin = origin
        prev_idx = entry.get("idx")
    if cur:
        segments.append(cur)
    return segments, breaks

