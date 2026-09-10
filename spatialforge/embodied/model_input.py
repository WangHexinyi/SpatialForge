"""Strict model-input isolation.

Phase-1 contract: the model may observe only

* first-person RGB
* task goal/instruction
* permitted action history
* permitted observation summaries
* camera horizon (vestibular / self-sense)

It must **never** receive world coordinates, target position, reachable set,
shortest/teacher path, God View truth, or the full scene graph. This module is
the single gate that builds model input and validates that nothing leaks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Exact keys permitted inside the observation portion of model input.
PERMITTED_OBSERVATION_KEYS = frozenset(
    {
        "step",
        "timestamp_ms",
        "image_path",
        "frame_id",
        "camera_horizon_deg",
        "instruction",
        "history_summary",
        "cause_action",
        "rgb_shape",
        "rgb_base64",
    }
)

# Key names / substrings that must never appear in model input.
FORBIDDEN_KEYS = frozenset(
    {
        "target_object_ids",
        "target_positions",
        "reachable_positions",
        "teacher_path",
        "position",
        "rotation_yaw_deg",
        "privileged",
        "truth",
        "godview",
        "god_view",
        "segmentation",
        "shortest",
        "scene_graph",
        "navmesh_reachable",
    }
)

# Catch-all forbidden semantic terms (defense in depth against renaming).
FORBIDDEN_TERMS = (
    "target_positions",
    "target_position",
    "target_object_ids",
    "target position",
    "target coordinates",
    "object id",
    "world_position",
    "world position",
    "coordinates",
    "reachable",
    "shortest_path",
    "teacher_path",
    "segmentation",
    "navmesh_reachable",
)


class ModelInputLeakError(ValueError):
    """Raised when privileged data is about to enter model input."""


def _contains_forbidden_term(payload: Dict[str, Any], path: str = "") -> Optional[str]:
    for key, value in payload.items():
        kp = f"{path}.{key}" if path else str(key)
        key_l = str(key).lower()
        if key in FORBIDDEN_KEYS or key_l in FORBIDDEN_KEYS:
            return kp
        for term in FORBIDDEN_TERMS:
            if term in str(key).lower():
                return kp
        if isinstance(value, dict):
            hit = _contains_forbidden_term(value, kp)
            if hit:
                return hit
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    hit = _contains_forbidden_term(item, f"{kp}[{i}]")
                    if hit:
                        return hit
                elif isinstance(item, str):
                    for term in FORBIDDEN_TERMS:
                        if term in item.lower():
                            return f"{kp}[{i}]"
        elif isinstance(value, str):
            for term in FORBIDDEN_TERMS:
                if term in value.lower():
                    return f"{kp}"
    return None


def validate_model_input_isolation(payload: Dict[str, Any]) -> None:
    """Raise ModelInputLeakError if any forbidden key/term is present."""
    hit = _contains_forbidden_term(payload)
    if hit is not None:
        raise ModelInputLeakError(
            f"Model-input isolation violated: forbidden content at '{hit}'."
        )


def build_agent_model_input(
    observation: Any,
    task: Any,
    history: Optional[Any] = None,
    include_rgb: bool = True,
) -> Dict[str, Any]:
    """Build the *only* permitted model input for one decision step.

    Returns a JSON-safe dict ready to feed a policy pi(a_t | o_<=t, goal).
    Raises ModelInputLeakError if anything privileged slips in.
    """
    obs_dict = observation.to_dict(include_rgb=include_rgb)
    # restrict observation to the strict allowlist:
    filtered_obs = {k: v for k, v in obs_dict.items() if k in PERMITTED_OBSERVATION_KEYS}

    goal = {
        "task_id": task.task_id,
        "house_id": task.house_id,
        "target_category": task.target_category,
        "instruction": task.instruction,
        "max_steps": task.max_steps,
        "step": task.step,
    }

    permitted_action_history: List[Dict[str, Any]] = []
    if history is not None:
        for action in history.actions:
            d = action.to_dict()
            # environment setup / teleport bookkeeping is not agent behavior and
            # must never appear in the model's permitted action history.
            if d.get("origin") in ("setup", "spawn", "reset"):
                continue
            # keep only step + outcome semantics; drop nothing privileged anyway
            permitted_action_history.append(
                {
                    "action_type": d["action_type"],
                    "step": d["step"],
                    "success": d.get("success"),
                    "collision": d.get("collision"),
                    "blocked": d.get("blocked"),
                }
            )

    payload: Dict[str, Any] = {
        "goal": goal,
        "observation": filtered_obs,
        "action_history": permitted_action_history,
    }

    validate_model_input_isolation(payload)
    return payload


def build_privileged_debug_payload(
    observation: Any,
    task: Any,
    agent_state: Any,
    house: Optional[Any] = None,
) -> Dict[str, Any]:
    """Researcher/debug-only payload (NEVER passed to model input)."""
    d: Dict[str, Any] = {
        "_privileged": True,
        "task": task.privileged_view(),
        "agent_state": agent_state.to_dict(),
    }
    if house is not None:
        d["house"] = house.to_dict()
    return d
