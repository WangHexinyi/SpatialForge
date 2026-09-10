"""Embodied runtime data contracts.

These are pure, serializable dataclasses describing the Agent's world and
task. They are split into two explicit planes:

* **Agent-visible** objects (what may reach the model) are built exclusively
  through :func:`spatialforge.embodied.model_input.build_agent_model_input`,
  which enforces a strict allowlist.
* **Researcher / teacher / debug** objects (privileged truth) are separate and
  never allowed into model input.

Coordinate system note: positions/rotations are expressed in whatever native
frame the active backend uses (the ProcTHOR / AI2-THOR backend uses Unity
coordinates). The embodied layer treats them as opaque world state and only the
verifier / teacher interpret them geometrically against backend metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

STEP_MS_DEFAULT = 100
EPISODE_MAX_STEPS_DEFAULT = 300

# --------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------


class AgentActionType(str, Enum):
    MOVE_AHEAD = "MoveAhead"
    MOVE_BACK = "MoveBack"
    MOVE_LEFT = "MoveLeft"
    MOVE_RIGHT = "MoveRight"
    ROTATE_LEFT = "RotateLeft"
    ROTATE_RIGHT = "RotateRight"
    LOOK_UP = "LookUp"
    LOOK_DOWN = "LookDown"
    CROUCH = "Crouch"
    STAND = "Stand"
    DONE = "Done"


# Primary navigation set. MoveBack/MoveLeft/MoveRight optional (stable env).
CORE_NAV_ACTIONS = [
    AgentActionType.MOVE_AHEAD,
    AgentActionType.ROTATE_LEFT,
    AgentActionType.ROTATE_RIGHT,
    AgentActionType.LOOK_UP,
    AgentActionType.LOOK_DOWN,
    AgentActionType.CROUCH,
    AgentActionType.STAND,
    AgentActionType.DONE,
]

FULL_NAV_ACTIONS = [
    AgentActionType.MOVE_AHEAD,
    AgentActionType.MOVE_BACK,
    AgentActionType.MOVE_LEFT,
    AgentActionType.MOVE_RIGHT,
    AgentActionType.ROTATE_LEFT,
    AgentActionType.ROTATE_RIGHT,
    AgentActionType.LOOK_UP,
    AgentActionType.LOOK_DOWN,
    AgentActionType.CROUCH,
    AgentActionType.STAND,
    AgentActionType.DONE,
]


# --------------------------------------------------------------------------
# AgentAction
# --------------------------------------------------------------------------


class AgentActionOrigin(str, Enum):
    """Where an action came from.

    Setup/reset/spawn are environment bookkeeping and must never be mistaken
    for a model output; the God View / research record surfaces this explicitly.
    """

    MODEL = "model"
    SETUP = "setup"
    SPAWN = "spawn"
    RESET = "reset"
    VERIFIER = "verifier"


@dataclass
class AgentAction:
    """A single agent action plus its executed outcome.

    Outcome fields are filled by the backend after execution, so this is not
    frozen."""


    action_type: AgentActionType
    step: int = 0
    # Provided at request time:
    parameters: Dict[str, Any] = field(default_factory=dict)
    #: "model" for genuine policy actions; setup/spawn/reset/verifier otherwise.
    origin: str = AgentActionOrigin.MODEL.value
    # Populated by the environment after execution:
    success: Optional[bool] = None
    error: Optional[str] = None
    collision: Optional[bool] = None
    blocked: Optional[bool] = None
    timestamp_ms: Optional[int] = None

    def request_dict(self) -> Dict[str, Any]:
        """Payload to hand to the backend for execution."""
        return {
            "action_type": self.action_type.value,
            "parameters": dict(self.parameters or {}),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "step": self.step,
            "parameters": dict(self.parameters or {}),
            "origin": self.origin,
            "success": self.success,
            "error": self.error,
            "collision": self.collision,
            "blocked": self.blocked,
            "timestamp_ms": self.timestamp_ms,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AgentAction":
        return AgentAction(
            action_type=AgentActionType(data["action_type"]),
            step=int(data.get("step", 0)),
            parameters=dict(data.get("parameters") or {}),
            origin=str(data.get("origin", AgentActionOrigin.MODEL.value)),
            success=data.get("success"),
            error=data.get("error"),
            collision=data.get("collision"),
            blocked=data.get("blocked"),
            timestamp_ms=data.get("timestamp_ms"),
        )

    def describe(self) -> str:
        return self.action_type.value


# --------------------------------------------------------------------------
# AgentState (researcher / debug plane; NOT auto model input)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentState:
    """The agent's kinematic state.

    This object is *privileged* (researcher/debug). It is intentionally not
    fed to the model automatically; the model sees only the permitted
    observation fields. World coordinates / reachable set / target live here
    or in the debug channel only.
    """

    position: Tuple[float, float, float]
    rotation_yaw_deg: float = 0.0  # agent heading
    camera_horizon_deg: float = 0.0  # camera pitch (up positive)
    is_crouching: bool = False
    is_standing: bool = True
    step_count: int = 0
    last_action: Optional[str] = None
    last_action_success: Optional[bool] = None
    blocked: bool = False
    collision: bool = False
    room: Optional[str] = None
    timestamp_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": list(self.position),
            "rotation_yaw_deg": self.rotation_yaw_deg,
            "camera_horizon_deg": self.camera_horizon_deg,
            "is_crouching": self.is_crouching,
            "is_standing": self.is_standing,
            "step_count": self.step_count,
            "last_action": self.last_action,
            "last_action_success": self.last_action_success,
            "blocked": self.blocked,
            "collision": self.collision,
            "room": self.room,
            "timestamp_ms": self.timestamp_ms,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AgentState":
        return AgentState(
            position=tuple(data["position"]),
            rotation_yaw_deg=float(data.get("rotation_yaw_deg", 0.0)),
            camera_horizon_deg=float(data.get("camera_horizon_deg", 0.0)),
            is_crouching=bool(data.get("is_crouching", False)),
            is_standing=bool(data.get("is_standing", True)),
            step_count=int(data.get("step_count", 0)),
            last_action=data.get("last_action"),
            last_action_success=data.get("last_action_success"),
            blocked=bool(data.get("blocked", False)),
            collision=bool(data.get("collision", False)),
            room=data.get("room"),
            timestamp_ms=data.get("timestamp_ms"),
        )


# --------------------------------------------------------------------------
# AgentObservation (the permitted sensor input)
# --------------------------------------------------------------------------


@dataclass
class AgentObservation:
    """What the agent actually sees.

    First-person RGB is carried as a numpy (or byte) image via ``rgb`` and
    referenced by ``frame_id``/``image_path`` for serialization. Only the
    fields listed in :func:`build_agent_model_input` may reach a model.
    """

    step: int
    timestamp_ms: int = 0
    rgb: Any = None  # HxWx3 uint8 ndarray (optional at runtime)
    image_path: Optional[str] = None
    frame_id: Optional[str] = None
    camera_horizon_deg: float = 0.0
    # self-motion perception: the action that produced this observation:
    cause_action: Optional[AgentAction] = None
    # free-text task instruction shown to the agent this step:
    instruction: Optional[str] = None
    # permitted episodic context (compressed summaries, not privileged):
    history_summary: Optional[Dict[str, Any]] = None
    # authoritative visibility flag computed by the verifier/backend
    # (target visible in current frame). This is the *verifier* judgement, not
    # a raw world coordinate, and is only surfaced when the environment opts in.
    target_visible: Optional[bool] = None

    def to_dict(self, include_rgb: bool = False) -> Dict[str, Any]:
        """JSON-safe dict. Raw image never serialized; referenced by id/path."""
        d: Dict[str, Any] = {
            "step": self.step,
            "timestamp_ms": self.timestamp_ms,
            "image_path": self.image_path,
            "frame_id": self.frame_id,
            "camera_horizon_deg": self.camera_horizon_deg,
            "instruction": self.instruction,
            "history_summary": self.history_summary,
        }
        if self.cause_action is not None:
            d["cause_action"] = self.cause_action.to_dict()
        if include_rgb and self.rgb is not None:
            try:
                import numpy as np

                d["rgb_shape"] = list(self.rgb.shape)
                d["rgb_base64"] = _array_to_base64(self.rgb, np)
            except Exception:
                d["rgb_shape"] = None
        return d


def _array_to_base64(arr: Any, np: Any) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# Episode state / history
# --------------------------------------------------------------------------


class EpisodeStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    DONE = "done"


@dataclass
class EpisodeState:
    episode_id: str
    house_id: str
    task_id: str
    status: EpisodeStatus = EpisodeStatus.RUNNING
    step: int = 0
    max_steps: int = EPISODE_MAX_STEPS_DEFAULT
    success: Optional[bool] = None
    success_reason: Optional[str] = None
    created_ms: int = 0
    completed_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "house_id": self.house_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "step": self.step,
            "max_steps": self.max_steps,
            "success": self.success,
            "success_reason": self.success_reason,
            "created_ms": self.created_ms,
            "completed_ms": self.completed_ms,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EpisodeState":
        return EpisodeState(
            episode_id=data["episode_id"],
            house_id=data["house_id"],
            task_id=data["task_id"],
            status=EpisodeStatus(data.get("status", "running")),
            step=int(data.get("step", 0)),
            max_steps=int(data.get("max_steps", EPISODE_MAX_STEPS_DEFAULT)),
            success=data.get("success"),
            success_reason=data.get("success_reason"),
            created_ms=int(data.get("created_ms", 0)),
            completed_ms=data.get("completed_ms"),
        )


@dataclass
class EpisodeHistory:
    """Recorded episode (used by teacher / debugging / rollouts)."""

    episode: EpisodeState
    observations: List[AgentObservation] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)

    def record_observation(self, obs: AgentObservation) -> None:
        self.observations.append(obs)

    def record_action(self, action: AgentAction) -> None:
        self.actions.append(action)

    def last_action(self) -> Optional[AgentAction]:
        return self.actions[-1] if self.actions else None


# --------------------------------------------------------------------------
# Object Search Task
# --------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class ObjectSearchTask:
    """Find-an-object task.

    *Agent* readable: task_id, house_id, target_category, instruction,
    max_steps, step, status, success.
    *Privileged* (never model input): target_object_ids, target_positions,
    reachable_positions, teacher_path, goal pose.
    """

    task_id: str
    house_id: str
    target_category: str
    instruction: str
    max_steps: int = EPISODE_MAX_STEPS_DEFAULT
    step: int = 0
    status: TaskStatus = TaskStatus.PENDING
    success: Optional[bool] = None
    success_reason: Optional[str] = None
    # ---- privileged truth (researcher / teacher / debug) ----
    target_object_ids: List[str] = field(default_factory=list)
    target_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    reachable_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    teacher_path: List[str] = field(default_factory=list)

    def agent_view(self) -> Dict[str, Any]:
        """Fields safe to give to the agent (goal + bookkeeping only)."""
        return {
            "task_id": self.task_id,
            "house_id": self.house_id,
            "target_category": self.target_category,
            "instruction": self.instruction,
            "max_steps": self.max_steps,
            "step": self.step,
            "status": self.status.value,
            "success": self.success,
            "success_reason": self.success_reason,
        }

    def privileged_view(self) -> Dict[str, Any]:
        d = self.agent_view()
        d.update(
            {
                "target_object_ids": list(self.target_object_ids),
                "target_positions": [list(p) for p in self.target_positions],
                "reachable_positions": [list(p) for p in self.reachable_positions],
                "teacher_path": list(self.teacher_path),
            }
        )
        return d

    def to_dict(self) -> Dict[str, Any]:
        return self.privileged_view()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ObjectSearchTask":
        t = ObjectSearchTask(
            task_id=data["task_id"],
            house_id=data["house_id"],
            target_category=data["target_category"],
            instruction=data.get("instruction", data["target_category"]),
            max_steps=int(data.get("max_steps", EPISODE_MAX_STEPS_DEFAULT)),
            step=int(data.get("step", 0)),
            status=TaskStatus(data.get("status", "pending")),
            success=data.get("success"),
            success_reason=data.get("success_reason"),
            target_object_ids=list(data.get("target_object_ids", [])),
            target_positions=[tuple(p) for p in data.get("target_positions", [])],
            reachable_positions=[tuple(p) for p in data.get("reachable_positions", [])],
            teacher_path=list(data.get("teacher_path", [])),
        )
        return t


# --------------------------------------------------------------------------
# House / environment descriptor (researcher plane)
# --------------------------------------------------------------------------


@dataclass
class HouseInfo:
    """Top-level descriptor of a loaded house / environment (researcher plane)."""

    house_id: str
    dataset: str = ""
    split: Optional[str] = None
    room_count: int = 0
    room_ids: List[str] = field(default_factory=list)
    object_category_counts: Dict[str, int] = field(default_factory=dict)
    object_count: int = 0
    seed: Optional[int] = None
    rendering_backend: str = ""
    navmesh_reachable_count: int = 0
    agent_start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "house_id": self.house_id,
            "dataset": self.dataset,
            "split": self.split,
            "room_count": self.room_count,
            "room_ids": list(self.room_ids),
            "object_category_counts": dict(self.object_category_counts),
            "object_count": self.object_count,
            "seed": self.seed,
            "rendering_backend": self.rendering_backend,
            "navmesh_reachable_count": self.navmesh_reachable_count,
            "agent_start": list(self.agent_start),
        }
