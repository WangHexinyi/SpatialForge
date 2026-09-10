"""Behavior-cloning dataset construction from genuine teacher transitions.

Pure Python (no torch, no ai2thor) so it can run in the rollout venv, the
training env and the CPU test suite. It converts one genuine episode's
``student_training_record`` (strict allowlist / leakage-scanned upstream) into
BC sample rows:

    (first-person RGB_t, goal text, permitted recent action history)
        -> teacher next AgentAction

The teacher action history that may appear in the prompt is derived only from
the already-isolated ``model_input`` history records (action_type/step/success/
collision/blocked). Everything is re-scanned with the recursive forbidden-term
guard before a row is emitted.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from spatialforge.embodied.model_input import FORBIDDEN_TERMS, validate_model_input_isolation

#: Phase-1 BC action vocabulary (teacher emits only these in this vertical slice).
BC_ACTION_VOCAB = (
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "Crouch",
    "Stand",
    "Done",
)

ACTION_ORDER = {a: i for i, a in enumerate(BC_ACTION_VOCAB)}

HISTORY_WINDOW = 10

BC_MANIFEST_SCHEMA = "embodied_bc_manifest.v1"


def action_history_text(history: Sequence[Dict[str, Any]], window: int = HISTORY_WINDOW) -> str:
    """Compact permitted action history (oldest first, up to ``window`` items)."""
    items = []
    for h in history[-window:]:
        a = h.get("action_type")
        if not a:
            continue
        tok = a
        if h.get("blocked"):
            tok += "(blocked)"
        elif h.get("success") is False:
            tok += "(failed)"
        items.append(tok)
    if not items:
        return "None yet"
    return ", ".join(items)


def build_bc_question(
    instruction: str,
    target_category: str,
    horizon_deg: Optional[float],
    history: Sequence[Dict[str, Any]],
    max_steps: Optional[int] = None,
    current_step: Optional[int] = None,
) -> str:
    """Build the fixed-format BC prompt (strictly non-privileged content)."""
    horizon = f"{float(horizon_deg):.0f} deg" if horizon_deg is not None else "unknown"
    step_txt = "" if current_step is None else f"Step: {current_step} of {max_steps}." if max_steps else f"Step: {current_step}."
    lines = [
        "You are an embodied agent searching for an object in a home.",
        f"Goal: {instruction}" if instruction else f"Goal: Find a {target_category}.",
        f"Recent actions (oldest first): {action_history_text(history)}",
        f"Current camera horizon: {horizon}.",
        step_txt,
        "Look at the current first-person view.",
        "Choose exactly ONE next action to make progress toward the goal.",
        "Allowed actions: " + ", ".join(BC_ACTION_VOCAB) + ".",
        "Emit only the action name and nothing else.",
    ]
    return "\n".join(l for l in lines if l)


def _clean(mi_obs: Dict[str, Any]) -> Dict[str, Any]:
    obs = dict(mi_obs or {})
    obs.pop("rgb_base64", None)
    obs.pop("image_path", None)
    return obs


def student_record_to_rows(
    record: Dict[str, Any],
    image_dir_resolver=None,
) -> List[Dict[str, Any]]:
    """Expand one student_training_record into BC sample rows.

    A row pairs the *decision observation* (initial obs, or the observation
    after an executed teacher action) with the teacher action executed next:

        obs_0      -> a_1 (first action of the episode)
        obs(a_1)   -> a_2
        ...
        obs(a_{k}) -> Done   (teacher emits Done only while target is visible)

    The final terminal transition (Done itself has no following observation)
    never produces a row.
    """
    schema = record.get("schema")
    if schema and "student_training_record" not in str(schema):
        raise ValueError(f"Not a student training record: {schema!r}")
    goal = record.get("goal", {})
    instruction = goal.get("instruction") or f"Find a {goal.get('target_category')}."
    target_category = goal.get("target_category", "object")
    steps: List[Dict[str, Any]] = record.get("steps", [])
    action_entries = [s for s in steps if s.get("role") == "action"]
    context_entries = [s for s in steps if s.get("role") == "context"]
    samples: List[Dict[str, Any]] = []

    def make_row(step_no, ref, mi_obs, history, teacher_action, terminal_target):
        obs = _clean(mi_obs or {})
        horizon = obs.get("camera_horizon_deg")
        if horizon is None:
            horizon = (ref or {}).get("camera_horizon_deg")
        image_path = (ref or {}).get("image_path")
        q = build_bc_question(
            instruction, target_category, horizon, history,
            max_steps=goal.get("max_steps"), current_step=step_no,
        )
        row = {
            "id": f"{record.get('episode_id', 'ep')}-s{step_no}",
            "question": q,
            "answer": str(teacher_action),
            "image_path": image_path,
            "scene_id": goal.get("house_id", ""),
            "episode_id": record.get("episode_id", ""),
            "category": target_category,
            "step": step_no,
            "history_len": len(history),
            "family": "embodied-bc",
            "tags": ["embodied-bc", str(target_category), str(teacher_action)],
            "terminal_target": bool(terminal_target),
        }
        validate_model_input_isolation(
            {
                "goal": {"instruction": instruction,
                         "target_category": target_category},
                "observation": {k: v for k, v in obs.items()
                                if k in ("camera_horizon_deg", "step")},
                "action_history": history,
                "question_text": q,
            }
        )
        return row

    if context_entries and action_entries:
        c = context_entries[0]
        first_target = action_entries[0].get("teacher_action")
        if first_target:
            samples.append(make_row(
                0, c.get("observation_ref", {}), c.get("observation", {}),
                [], first_target, False,
            ))
    for idx, st in enumerate(action_entries):
        if idx + 1 >= len(action_entries):
            continue  # terminal transition: no following teacher action
        nxt = action_entries[idx + 1]
        teacher_action = nxt.get("teacher_action")
        if not teacher_action:
            continue
        # Done is only a valid learning target when the target is authoritatively
        # visible in the decision observation (teacher occasionally emits Done at
        # the step-budget edge while the verifier rightly reports invisible).
        if teacher_action == "Done" and not bool(st.get("verifier_target_visible")):
            continue
        samples.append(make_row(
            int(st.get("step", idx + 1)),
            st.get("observation_ref", {}),
            st.get("observation", {}),
            st.get("action_history") or [],
            teacher_action,
            bool(nxt.get("terminal")),
        ))
    return samples


def scan_text_leakage(text: str) -> Optional[str]:
    """Return the first forbidden term found in free text, else None."""
    t = text.lower()
    for term in FORBIDDEN_TERMS:
        if term in t:
            return term
    return None


def scan_rows_leakage(rows: Iterable[Dict[str, Any]]) -> List[str]:
    """Leak-scan row question/answer text; return list of offending row ids."""
    bad = []
    for row in rows:
        hit = scan_text_leakage(str(row.get("question", "")))
        if hit is not None or scan_text_leakage(str(row.get("answer", ""))):
            bad.append(str(row.get("id")))
    return bad


def action_distribution(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {a: 0 for a in BC_ACTION_VOCAB}
    total = 0
    for row in rows:
        a = str(row.get("answer", ""))
        if a in counts:
            counts[a] += 1
        else:
            counts.setdefault(a, 0)
            counts[a] += 1
        total += 1
    return {"counts": counts, "total": total}


def write_manifest(rows: Sequence[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_manifest(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
