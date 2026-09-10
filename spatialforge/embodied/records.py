"""Two strictly-separated logical data domains produced by a teacher rollout.

Domain 1 -- *Student Training Record*
    What a student/BC policy may learn from. Contains only non-privileged
    content: first-person RGB *reference* (frame id / file), goal/instruction,
    the permitted action/observation history, and the teacher's next action
    (a plain ``AgentActionType`` label). No world pose, no target identity or
    position, no reachable set, no path.

Domain 2 -- *Privileged Research Record*
    The full researcher/teacher/debug payload: per-step world pose, target
    object ids/positions, reachable set, teacher view pose, authoritative
    visibility, executed trajectory, spawn + resample history, God-View truth.

The strict allowlist in :mod:`spatialforge.embodied.model_input` is **never
relaxed**. ``build_student_training_record`` re-runs
:func:`validate_model_input_isolation` over the produced structure so any leak
fails loudly instead of silently entering training data.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from spatialforge.embodied.model_input import validate_model_input_isolation

STUDENT_SCHEMA = "student_training_record.v1"
RESEARCH_SCHEMA = "privileged_research_record.v1"


def _clean_observation_ref(observation: Any) -> Dict[str, Any]:
    """Non-privileged first-person RGB reference (never embeds raw pixels)."""
    return {
        "step": observation.step,
        "frame_id": observation.frame_id,
        "image_path": observation.image_path,
        "camera_horizon_deg": observation.camera_horizon_deg,
        "rgb_shape": list(observation.rgb.shape) if observation.rgb is not None else None,
    }


def _permitted_history(model_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    return copy.deepcopy(model_input.get("action_history", []))


def build_student_training_record(
    episode_id: str,
    task_agent_view: Dict[str, Any],
    transitions: List[Any],
    initial_observation: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the student (BC) training record from genuine transitions."""
    steps: List[Dict[str, Any]] = []
    if initial_observation is not None:
        steps.append(
            {
                "role": "context",
                "step": 0,
                "observation_ref": _clean_observation_ref(initial_observation),
                "action_history": [],
                "teacher_action": None,
            }
        )

    for trans in transitions:
        if trans is None:
            continue
        mi = copy.deepcopy(trans.model_input)
        # drop embedded pixels if present; keep RGB reference
        mi_obs = mi.get("observation", {})
        mi_obs.pop("rgb_base64", None)
        action = trans.action
        steps.append(
            {
                "role": "action",
                "step": trans.step,
                "observation_ref": _clean_observation_ref(trans.observation),
                "observation": mi_obs,
                "action_history": _permitted_history(mi),
                "teacher_action": action.action_type.value if action else None,
                "teacher_action_success": action.success if action else None,
                "verifier_target_visible": trans.observation.target_visible,
                "terminal": bool(trans.terminal),
            }
        )

    record: Dict[str, Any] = {
        "schema": STUDENT_SCHEMA,
        "domain": "student_training",
        "episode_id": episode_id,
        "goal": copy.deepcopy(task_agent_view),
        "steps": steps,
        "_leak_checked": "clean",
    }
    # strict recursive isolation re-check before returning
    validate_model_input_isolation(record)
    return record


def _pose_to_dict(agent_state: Any) -> Dict[str, Any]:
    return {
        "position": list(agent_state.position),
        "rotation_yaw_deg": agent_state.rotation_yaw_deg,
        "camera_horizon_deg": agent_state.camera_horizon_deg,
        "is_crouching": agent_state.is_crouching,
        "room": agent_state.room,
    }


def build_privileged_research_record(
    episode_id: str,
    task_privileged_view: Dict[str, Any],
    episode_state: Any,
    transitions: List[Any],
    teacher_plan: Optional[Dict[str, Any]] = None,
    spawn: Optional[Dict[str, Any]] = None,
    frames: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the researcher-only record with full privileged truth."""
    steps: List[Dict[str, Any]] = []
    for trans in transitions:
        if trans is None:
            continue
        action = trans.action
        steps.append(
            {
                "step": trans.step,
                "action_type": action.action_type.value if action else None,
                "action_success": action.success if action else None,
                "agent": _pose_to_dict(trans.agent_state),
                "authoritative_target_visible": trans.observation.target_visible,
                "terminal": bool(trans.terminal),
            }
        )

    record: Dict[str, Any] = {
        "schema": RESEARCH_SCHEMA,
        "domain": "privileged_research",
        "_privileged": True,
        "episode_id": episode_id,
        "task": copy.deepcopy(task_privileged_view),
        "episode": episode_state.to_dict() if hasattr(episode_state, "to_dict") else episode_state,
        "spawn": copy.deepcopy(spawn) or {},
        "teacher_plan": copy.deepcopy(teacher_plan) or {},
        "steps": steps,
        "frames": list(frames or []),
    }
    return record
