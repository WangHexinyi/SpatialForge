"""Researcher-facing episode quality labels, anomaly detection, condensed replay.

Pure Python (no ai2thor / torch) so the CPU test suite exercises the exact logic
the Inspector uses to curate the replay list and flag useless rollouts.

The goal is *truthfulness*: the Inspector must never present a legacy,
over-exposed, teleport-contaminated, or policy-collapsed rollout as if it were a
clean post-fix episode. Every episode gets an explicit label set, and the default
replay list is ordered so a researcher sees the useful episodes first.

Labels
------
``VALID``           post-fix, sensor-valid, spawn-semantics-clean episode.
``LEGACY``          produced before the render-quality / spawn-semantics fix
                    (no ``setup`` block and/or no ``render_quality``).
``INVALID_SENSOR``  render quality missing/known-bad (e.g. AI2-THOR ``Low``).
``INVALID_SPAWN``   spawn/setup mismatch (spawn != first-frame pose).
``ANOMALY``         unexplained (teleport-like) displacement in the trajectory.
``CRASH``           inference/runtime failure.
``TIMEOUT``         step budget exhausted before Done.
``STUCK``           long run with no net progress.
``POLICY_COLLAPSE`` degenerate action distribution (one action dominates).
``FALSE_DONE``      Done emitted without the target visible / success false.
``SUCCESS`` / ``FAILURE``  verifier outcome.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from spatialforge.embodied.spawn_semantics import check_episode_semantics

#: render qualities whose sensor response is visually normal.
ACCEPTABLE_RENDER_QUALITIES = ("Medium", "High", "Very High", "Ultra")
#: AI2-THOR quality that blows out highlights (diagnosed: frac>=250 up to 0.38).
BAD_RENDER_QUALITIES = ("Low",)

POS_TOL_M = 0.05
#: a step that moves less than this in xz is "no progress" for stuck detection.
NO_PROGRESS_M = 0.02
STUCK_MIN_STEPS = 20
STUCK_UNIQUE_CELLS = 3
COLLAPSE_MIN_DECISIONS = 20
COLLAPSE_DOMINANCE = 0.8
NO_PROGRESS_RUN_MIN = 3
#: actions that legitimately translate the agent (collapsible no-progress runs).
_TRANSLATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}


def _pos(entry: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    if not entry:
        return None
    p = entry.get("position")
    if p is None:
        return None
    try:
        return [float(p[0]), float(p[1]), float(p[2])]
    except (TypeError, IndexError, KeyError):
        return None


def _dist_xz(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[2] - b[2])


def _cell(p: Optional[Sequence[float]]) -> Optional[tuple]:
    if p is None:
        return None
    return (round(float(p[0]) * 2) / 2, round(float(p[2]) * 2) / 2)


def action_sequence(record: Dict[str, Any]) -> List[str]:
    out = []
    for st in record.get("steps", []) or []:
        a = st.get("action_type")
        if a:
            out.append(str(a))
    return out


def detect_stuck(record: Dict[str, Any]) -> Dict[str, Any]:
    """Detect a long run that never makes spatial progress."""
    steps = record.get("steps", []) or []
    cells = []
    for st in steps:
        c = _cell(_pos(st.get("agent")))
        if c is not None:
            cells.append(c)
    unique = len(set(cells))
    net = 0.0
    if len(cells) >= 2:
        net = math.hypot(cells[0][0] - cells[-1][0], cells[0][1] - cells[-1][1])
    stuck = (
        len(steps) >= STUCK_MIN_STEPS
        and unique <= STUCK_UNIQUE_CELLS
        and net < 1.0
    )
    return {
        "stuck": stuck,
        "steps": len(steps),
        "unique_cells": unique,
        "net_displacement_m": round(net, 3),
    }


def detect_policy_collapse(record: Dict[str, Any],
                           decisions: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Detect a degenerate action distribution (one action dominates)."""
    seq: List[str] = []
    if decisions:
        seq = [str(d.get("model_action")) for d in decisions if d.get("model_action")]
    if not seq:
        seq = action_sequence(record)
    if not seq:
        return {"collapse": False, "dominant": None, "fraction": 0.0, "n": 0}
    counts: Dict[str, int] = {}
    for a in seq:
        counts[a] = counts.get(a, 0) + 1
    dominant, n = max(counts.items(), key=lambda kv: kv[1])
    frac = n / len(seq)
    collapse = len(seq) >= COLLAPSE_MIN_DECISIONS and frac >= COLLAPSE_DOMINANCE
    return {
        "collapse": collapse,
        "dominant": dominant,
        "fraction": round(frac, 3),
        "n": len(seq),
    }


def no_progress_runs(
    record: Dict[str, Any], min_run: int = NO_PROGRESS_RUN_MIN
) -> List[Dict[str, Any]]:
    """Runs of consecutive *translation* steps that repeat without moving.

    Returns researcher-facing segments so the UI can collapse ``MoveAhead x 40``
    into a single labelled row instead of wasting the whole timeline. Pure
    rotations are not collapsed here (they are handled by policy-collapse
    detection) so genuine turning behaviour is never hidden.
    """
    steps = record.get("steps", []) or []
    runs: List[Dict[str, Any]] = []
    i = 0
    while i < len(steps):
        a = str(steps[i].get("action_type") or "")
        j = i + 1
        while (
            j < len(steps)
            and a in _TRANSLATION_ACTIONS
            and str(steps[j].get("action_type") or "") == a
        ):
            prev = _pos(steps[j - 1].get("agent"))
            cur = _pos(steps[j].get("agent"))
            if prev is None or cur is None or _dist_xz(prev, cur) > NO_PROGRESS_M:
                break
            j += 1
        count = j - i
        if a and count >= min_run:
            first = _pos(steps[i].get("agent"))
            last = _pos(steps[j - 1].get("agent"))
            disp = _dist_xz(first, last) if first and last else 0.0
            runs.append({
                "action": a,
                "start_step": steps[i].get("step", i + 1),
                "end_step": steps[j - 1].get("step", j),
                "count": count,
                "displacement_m": round(disp, 3),
                "kind": "no_progress",
            })
        i = j
    return runs


def classify_episode(
    record: Dict[str, Any],
    *,
    decisions: Optional[Sequence[Dict[str, Any]]] = None,
    render_quality: Optional[str] = None,
    semantics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return ``{"labels": [...], "flags": {...}, "metrics": {...}}``."""
    record = record or {}
    decisions = list(decisions or record.get("decisions") or [])
    labels: List[str] = []

    episode = record.get("episode", {}) or {}
    status = str(episode.get("status") or record.get("outcome", {}).get("status") or "")
    success = episode.get("success")
    if success is None:
        success = record.get("outcome", {}).get("success")
    terminal_reason = str(
        record.get("terminal_reason")
        or episode.get("success_reason")
        or record.get("outcome", {}).get("reason")
        or ""
    )

    setup = record.get("setup") or {}
    has_setup = bool(setup.get("initial_pose"))
    quality = render_quality if render_quality is not None else record.get("render_quality")
    quality = str(quality) if quality else None

    # --- legacy / sensor validity ------------------------------------
    if quality in BAD_RENDER_QUALITIES:
        labels.append("INVALID_SENSOR")
    if quality is None and not has_setup:
        labels.append("LEGACY")
    if quality is None and has_setup:
        # post-fix but quality not persisted: conservative, not sensor-valid
        labels.append("LEGACY")
    if quality in ACCEPTABLE_RENDER_QUALITIES and has_setup:
        labels.append("VALID")

    # --- spawn / trajectory semantics --------------------------------
    if semantics is None:
        try:
            semantics = check_episode_semantics(record)
        except Exception:  # noqa: BLE001
            semantics = {"ok": False, "violations": []}
    kinds = {v.get("kind") for v in semantics.get("violations", [])}
    if "spawn_initial_pose_mismatch" in kinds:
        labels.append("INVALID_SPAWN")
    if "unexplained_displacement" in kinds:
        labels.append("ANOMALY")

    # --- runtime outcome ---------------------------------------------
    low = terminal_reason.lower()
    if "inference_failure" in low or "server error" in low or "crash" in low:
        labels.append("CRASH")
    if "budget" in low or "timeout" in low or "step budget" in low:
        labels.append("TIMEOUT")
    if status == "success" or success is True:
        labels.append("SUCCESS")
    elif status in ("failure", "failed") or success is False:
        labels.append("FAILURE")

    # --- false Done ---------------------------------------------------
    false_done = any(
        d.get("model_action") == "Done"
        and d.get("executed") is not False
        and d.get("authoritative_target_visible") is False
        for d in decisions
    )
    if false_done and success is not True:
        labels.append("FALSE_DONE")

    # --- degenerate behaviour ----------------------------------------
    stuck = detect_stuck(record)
    if stuck["stuck"]:
        labels.append("STUCK")
    collapse = detect_policy_collapse(record, decisions)
    if collapse["collapse"]:
        labels.append("POLICY_COLLAPSE")

    # de-duplicate preserving order
    seen = set()
    labels = [x for x in labels if not (x in seen or seen.add(x))]

    hard_bad = {"INVALID_SENSOR", "INVALID_SPAWN", "CRASH", "ANOMALY", "LEGACY"}
    is_valid = "VALID" in labels and not (hard_bad & set(labels))
    useful = is_valid and not ({"STUCK", "POLICY_COLLAPSE"} & set(labels))
    return {
        "labels": labels,
        "flags": {
            "valid": is_valid,
            "legacy": "LEGACY" in labels,
            "sensor_valid": "INVALID_SENSOR" not in labels,
            "spawn_clean": "INVALID_SPAWN" not in labels and "ANOMALY" not in labels,
            "researcher_useful": useful,
            "success": "SUCCESS" in labels,
            "stuck": stuck["stuck"],
            "policy_collapse": collapse["collapse"],
            "false_done": false_done,
        },
        "metrics": {
            "status": status or None,
            "success": success,
            "render_quality": quality,
            "has_setup": has_setup,
            "stuck": stuck,
            "policy_collapse": collapse,
            "no_progress_runs": no_progress_runs(record),
            "semantics_ok": bool(semantics.get("ok")),
        },
    }


def curated_priority(classification: Dict[str, Any]) -> int:
    """Higher is better; used to order the default replay list."""
    flags = (classification or {}).get("flags", {})
    labels = set((classification or {}).get("labels", []))
    score = 0
    if "VALID" in labels:
        score += 40
    if flags.get("sensor_valid"):
        score += 20
    if flags.get("spawn_clean"):
        score += 15
    if flags.get("success"):
        score += 20
    if flags.get("researcher_useful"):
        score += 30
    if "LEGACY" in labels:
        score -= 50
    if "INVALID_SENSOR" in labels:
        score -= 60
    if "INVALID_SPAWN" in labels or "ANOMALY" in labels:
        score -= 40
    if "CRASH" in labels:
        score -= 70
    if flags.get("policy_collapse"):
        score -= 25
    if flags.get("stuck"):
        score -= 25
    if "TIMEOUT" in labels:
        score -= 10
    return score


def condensed_view(
    timeline: Sequence[Dict[str, Any]], runs: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build a condensed timeline: collapse no-progress runs into segments.

    ``timeline`` entries carry ``idx`` (and optional ``step``). Returns
    ``{"indices": [...], "segments": [...]}`` where ``segments`` describe
    collapsed spans the UI can render as a single row. The first frame of a run
    is kept (so the researcher still sees the state); the redundant repeats are
    skipped in the display index. Raw data is never deleted.
    """
    step_to_idx: Dict[int, int] = {}
    for entry in timeline or []:
        if entry.get("step") is not None and entry.get("idx") is not None:
            step_to_idx[int(entry["step"])] = int(entry["idx"])

    skip_idx = set()
    segments: List[Dict[str, Any]] = []
    for run in runs or []:
        start = run.get("start_step")
        end = run.get("end_step")
        if start is None or end is None:
            continue
        from_idx = step_to_idx.get(int(start))
        to_idx = step_to_idx.get(int(end))
        if from_idx is None or to_idx is None or to_idx <= from_idx:
            continue
        segments.append({
            "kind": "no_progress",
            "action": run.get("action"),
            "count": run.get("count"),
            "from_idx": from_idx,
            "to_idx": to_idx,
            "start_step": start,
            "end_step": end,
            "displacement_m": run.get("displacement_m"),
        })
        for idx in range(from_idx + 1, to_idx + 1):
            skip_idx.add(idx)

    indices = [
        int(e["idx"]) for e in (timeline or [])
        if e.get("idx") is not None and int(e["idx"]) not in skip_idx
    ]
    return {"indices": indices, "segments": segments}
