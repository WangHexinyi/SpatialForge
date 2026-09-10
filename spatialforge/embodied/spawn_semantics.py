"""Spawn/reset semantics invariants for real episodes.

Environment setup uses privileged ``TeleportFull`` (view-pose probing, spawn
probing, episode spawn). A researcher must never mistake that for a model
output. This module checks, from a persisted privileged research record, that:

* the recorded spawn pose equals the first-frame agent pose (no hidden move);
* every subsequent pose change is explainable by the executed action
  (MoveAhead <= ~0.75 m; Rotate = +/-90 deg; Look = +/-30 deg; etc.);
* no unexplained (teleport-like) displacement happens between genuine steps.

A violation is a hard blocker for closed-loop evaluation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

POS_TOL_M = 0.05
MOVE_MAX_M = 0.75
YAW_TOL_DEG = 6.0
HORIZON_TOL_DEG = 6.0
ROTATE_DEG = 90.0
LOOK_DEG = 30.0


def _pos(entry: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    if not entry:
        return None
    p = entry.get("position")
    if p is None:
        return None
    return [float(p[0]), float(p[1]), float(p[2])]


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _dist_xz(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[2] - b[2])


def _angle_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(((float(b) - float(a) + 540.0) % 360.0) - 180.0)


def _pose(agent: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    agent = agent or {}
    return {
        "position": _pos(agent),
        "yaw": agent.get("rotation_yaw_deg"),
        "horizon": agent.get("camera_horizon_deg"),
    }


def check_episode_semantics(record: Dict[str, Any]) -> Dict[str, Any]:
    """Run spawn/reset invariants over one privileged research record."""
    violations: List[Dict[str, Any]] = []
    spawn = record.get("spawn") or {}
    setup = record.get("setup") or {}
    steps = record.get("steps") or []

    spawn_pose = {
        "position": [float(x) for x in spawn["position"]] if spawn.get("position") else None,
        "yaw": spawn.get("yaw"),
        "horizon": spawn.get("horizon"),
    }
    initial = setup.get("initial_pose") or {}
    initial_pose = {
        "position": [float(x) for x in initial["position"]] if initial.get("position") else None,
        "yaw": initial.get("rotation_yaw_deg", initial.get("yaw")),
        "horizon": initial.get("horizon_deg", initial.get("horizon")),
    }

    if spawn_pose["position"] and initial_pose["position"]:
        d = _dist(spawn_pose["position"], initial_pose["position"])
        dy = _angle_delta(spawn_pose["yaw"], initial_pose["yaw"])
        dh = _angle_delta(spawn_pose["horizon"], initial_pose["horizon"])
        if d > POS_TOL_M or (dy is not None and dy > 1.0) or (dh is not None and dh > 1.0):
            violations.append({
                "kind": "spawn_initial_pose_mismatch",
                "detail": f"dist={d:.3f}m dyaw={dy} dhorizon={dh}",
            })

    # the pose before the first genuine action is the spawn/initial pose
    prev = dict(initial_pose) if initial_pose.get("position") else dict(spawn_pose)
    if not prev.get("position") and steps:
        prev = _pose(steps[0].get("agent"))
    for i, st in enumerate(steps):
        cur = _pose(st.get("agent"))
        action = str(st.get("action_type") or "")
        if prev is None or cur.get("position") is None:
            prev = cur
            continue
        moved = _dist_xz(prev["position"], cur["position"])
        dyaw = _angle_delta(prev.get("yaw"), cur.get("yaw"))
        dhor = abs(float(cur.get("horizon") or 0.0) - float(prev.get("horizon") or 0.0))
        unexplained = False
        if action in ("MoveAhead", "MoveBack"):
            if moved > MOVE_MAX_M or (dyaw is not None and dyaw > YAW_TOL_DEG):
                unexplained = True
        elif action in ("RotateLeft", "RotateRight"):
            if moved > POS_TOL_M or (dyaw is not None and abs(dyaw - ROTATE_DEG) > YAW_TOL_DEG):
                unexplained = True
        elif action in ("LookUp", "LookDown"):
            if moved > POS_TOL_M or abs(dhor - LOOK_DEG) > HORIZON_TOL_DEG:
                unexplained = True
        elif action in ("Crouch", "Stand"):
            if moved > POS_TOL_M:
                unexplained = True
        elif action in ("Done", ""):
            if moved > POS_TOL_M:
                unexplained = True
        else:
            if moved > MOVE_MAX_M:
                unexplained = True
        if unexplained:
            violations.append({
                "kind": "unexplained_displacement",
                "step": st.get("step", i + 1),
                "action": action,
                "detail": (
                    f"moved={moved:.3f}m dyaw={None if dyaw is None else round(dyaw, 2)} "
                    f"dhorizon={round(dhor, 2)}"
                ),
            })
        prev = cur

    return {
        "ok": not violations,
        "violations": violations,
        "checks": {
            "spawn": spawn_pose,
            "initial": initial_pose,
            "steps_checked": len(steps),
            "teleport_policy": setup.get(
                "teleport_policy",
                "TeleportFull only during reset/spawn (environment setup)",
            ),
        },
    }


def setup_block(
    spawn: Optional[Dict[str, Any]],
    initial_agent_state: Any,
    teleport_policy: str = "TeleportFull only during reset/spawn (environment setup)",
) -> Dict[str, Any]:
    """Build the privileged ``setup`` section stored with an episode."""
    pos = getattr(initial_agent_state, "position", None)
    return {
        "spawn_requested": {
            "position": list(spawn.get("position", [])) if spawn else None,
            "yaw": spawn.get("yaw") if spawn else None,
            "horizon": spawn.get("horizon") if spawn else None,
        },
        "initial_pose": {
            "position": list(pos) if pos is not None else None,
            "rotation_yaw_deg": getattr(initial_agent_state, "rotation_yaw_deg", None),
            "horizon_deg": getattr(initial_agent_state, "camera_horizon_deg", None),
        },
        "teleport_policy": teleport_policy,
    }
