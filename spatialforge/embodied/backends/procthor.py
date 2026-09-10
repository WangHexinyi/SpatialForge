"""Real ProcTHOR / AI2-THOR backend (product path) -- VALIDATED runtime.

Drives an actual AI2-THOR Unity build inside a real ProcTHOR house loaded from
official ProcTHOR-10K data via the engine's ``CreateHouse`` action. Imports of
``ai2thor`` are lazy/guarded so the CPU-only test suite never needs a Unity
binary.

Dependency / compatibility pins (see docs/PROJECT_PLAN_v3.md dependency table):
* ``ai2thor==5.0.0``   -- current ProcTHOR-10K requires ai2thor 5.0+.
* Python 3.10 isolated venv (ai2thor requires Python <= 3.11).
* AI2-THOR Unity Linux build (auto-downloaded by the Controller); on AutoDL /
  Linux headless runs it needs an X server (Xvfb) providing GL, or a
  CloudRendering / EGL path.

Loading a house is the *real* runtime loop, not a replay:
    Controller(scene="Procedural")
      -> reset
      -> CreateHouse(house_json)      # instantiates a real ProcTHOR house
      -> TeleportFull(agent pose)     # spawn the agent
      -> action_t -> AI2-THOR step -> observation_{t+1}

Verifier visibility is taken from AI2-THOR authoritative metadata (object
``visible`` engine state), never from ``camera_depth``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from spatialforge.embodied.contracts import (
    AgentAction,
    AgentActionOrigin,
    AgentActionType,
    AgentObservation,
    AgentState,
    HouseInfo,
    ObjectSearchTask,
)
from spatialforge.embodied.environment import EmbodiedBackend
from spatialforge.embodied.rendering import (
    RendererNotAvailableError,
    ensure_nvidia_renderer,
    resolve_render_quality,
)

_THOR_ACTION = {
    AgentActionType.MOVE_AHEAD: "MoveAhead",
    AgentActionType.MOVE_BACK: "MoveBack",
    AgentActionType.MOVE_LEFT: "MoveLeft",
    AgentActionType.MOVE_RIGHT: "MoveRight",
    AgentActionType.ROTATE_LEFT: "RotateLeft",
    AgentActionType.ROTATE_RIGHT: "RotateRight",
    AgentActionType.LOOK_UP: "LookUp",
    AgentActionType.LOOK_DOWN: "LookDown",
    AgentActionType.CROUCH: "Crouch",
    AgentActionType.STAND: "Stand",
}


class ProcTHORError(RuntimeError):
    pass


def load_house_json(source: Union[str, Path, dict]) -> dict:
    """Load an official ProcTHOR house dict from a path or a raw dict."""
    if isinstance(source, dict):
        return source
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _agent_meta(house: dict) -> dict:
    return house.get("metadata", {}).get("agent", {})


class ProcTHORBackend(EmbodiedBackend):
    """Real ProcTHOR house driven by an actual AI2-THOR Unity build.

    Construct with an official house dict or a path to one; ``load_house`` starts
    the engine and calls ``CreateHouse`` so the real house exists in the scene.
    """

    LABEL = "ai2thor-unity (ProcTHOR)"

    def __init__(
        self,
        house: Union[str, Path, dict, None] = None,
        width: int = 256,
        height: int = 256,
        quality: Optional[str] = None,
        scene: str = "Procedural",
        x_display: Optional[str] = None,
        require_nvidia: bool = False,
    ):
        self.house_source = house
        self._house_data: Optional[dict] = None
        self.width = width
        self.height = height
        self.quality = resolve_render_quality(quality)
        self.scene = scene
        self.x_display = x_display
        self.require_nvidia = require_nvidia
        self.controller = None
        self._event = None
        self._house_info: Optional[HouseInfo] = None
        self._category_object_ids: Dict[str, List[str]] = {}
        self._object_positions: Dict[str, tuple] = {}
        self._reachable: List[Any] = []

    # -- lifecycle ----------------------------------------------------
    def _imports(self):
        try:
            from ai2thor.controller import Controller  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise ProcTHORError(
                "ai2thor is not importable here. Run under the Python-3.10 venv with "
                "ai2thor>=5.0.0 installed."
            ) from e
        return Controller

    def load_house(self, house_id: str = "", **opts) -> HouseInfo:
        """Start the engine and instantiate the real ProcTHOR house.

        When ``require_nvidia`` is set the engine is only started if the target
        X display serves an NVIDIA OpenGL renderer; otherwise a loud error is
        raised (no silent llvmpipe fallback).
        """
        if self.require_nvidia:
            try:
                ensure_nvidia_renderer(self.x_display or opts.get("x_display"))
            except RendererNotAvailableError as e:
                raise ProcTHORError(
                    "require_nvidia=True but GPU rendering is not available: "
                    f"{e}"
                ) from e
        src = opts.get("house") or self.house_source or house_id or None
        if src is None:
            raise ProcTHORError("A ProcTHOR house dict or path is required.")
        if isinstance(src, (str, Path)):
            # Remember the resolved path so the live Unity God View can reuse the
            # exact same house the running controller instantiated.
            self.house_source = str(src)
        self._house_data = load_house_json(src)

        Controller = self._imports()
        kwargs = dict(
            width=self.width,
            height=self.height,
            quality=self.quality,
            scene=self.scene,
        )
        if self.x_display:
            kwargs["x_display"] = self.x_display
        self.controller = Controller(**kwargs)
        self.controller.step(dict(action="ResetObjectFilter"))
        ev = self.controller.step(dict(action="CreateHouse", house=self._house_data, renderImage=False))
        self._event = ev
        md = ev.metadata
        if not md.get("lastActionSuccess"):
            raise ProcTHORError(f"CreateHouse failed: {md.get('errorMessage')}")

        # teleport the agent to the official start pose (renders first frame)
        agent = _agent_meta(self._house_data)
        self._teleport(agent)
        self._index_metadata(md)
        self._fetch_reachable()
        self._house_info = self._build_house_info()
        return self._house_info

    def _fetch_reachable(self) -> List[Any]:
        """Authoritative AI2-THOR GetReachablePositions -> world (x,z) points."""
        ev = self.controller.step(dict(action="GetReachablePositions", renderImage=False))
        pts = ev.metadata.get("actionReturn", []) or []
        self._reachable = []
        for p in pts:
            y = p.get("y", 0.9)
            if not isinstance(y, (int, float)) or y is None:
                y = 0.9
            self._reachable.append((float(p["x"]), float(y), float(p.get("z", 0.0))))
        return self._reachable

    @staticmethod
    def _as_position(p: Any) -> Dict[str, float]:
        if isinstance(p, dict):
            return {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0)), "z": float(p.get("z", 0.0))}
        return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}

    @staticmethod
    def _as_rotation(r: Any) -> Dict[str, float]:
        if isinstance(r, dict):
            return {"x": float(r.get("x", 0.0)), "y": float(r.get("y", 0.0)), "z": float(r.get("z", 0.0))}
        return {"x": 0.0, "y": float(r), "z": 0.0}

    def _teleport(self, agent: dict) -> None:
        params = dict(
            action="TeleportFull",
            position=self._as_position(agent["position"]),
            rotation=self._as_rotation(agent.get("rotation", {"x": 0, "y": 0, "z": 0})),
            horizon=float(agent.get("horizon", 0)),
            standing=bool(agent.get("standing", True)),
        )
        self._event = self.controller.step(params)
        md = self._event.metadata
        if not md.get("lastActionSuccess", False):
            raise ProcTHORError(
                "TeleportFull failed during episode spawn: "
                f"{md.get('errorMessage')} (requested position={params['position']}, "
                f"rotation={params['rotation']}, horizon={params['horizon']})"
            )
        actual = md.get("agent", {}).get("position", {})
        requested = params["position"]
        err = max(
            abs(float(actual.get(k, 0.0)) - requested[k]) for k in ("x", "y", "z")
        )
        if err > 0.05:
            raise ProcTHORError(
                f"TeleportFull did not reach the requested spawn: requested={requested} "
                f"actual={actual} err={err:.3f}m"
            )

    def set_agent_pose(
        self,
        position: Union[tuple, dict],
        rotation_yaw_deg: float = 0.0,
        horizon_deg: float = 0.0,
        standing: bool = True,
        render: bool = True,
    ) -> dict:
        """Privileged teacher/research helper: teleport to an arbitrary pose.

        This uses the engine's ``TeleportFull`` directly and is **never** a
        student action and never goes through model input. It is used to search
        for a suitable visible observation pose and to sample episode spawns.
        Returns the raw metadata of the teleport event.
        """
        if isinstance(position, dict):
            px, py, pz = position["x"], position["y"], position["z"]
        else:
            px, py, pz = position[0], position[1], position[2]
        params = dict(
            action="TeleportFull",
            position={"x": float(px), "y": float(py), "z": float(pz)},
            rotation={"x": 0.0, "y": float(rotation_yaw_deg), "z": 0.0},
            horizon=float(horizon_deg),
            standing=bool(standing),
        )
        if not render:
            params["renderImage"] = False
        self._event = self.controller.step(params)
        md = self._event.metadata
        self._index_metadata(md)
        return md

    def close(self) -> None:
        if self.controller is not None:
            try:
                self.controller.stop()
            except Exception:
                pass
            self.controller = None

    # -- metadata -----------------------------------------------------
    def _index_metadata(self, md: dict) -> None:
        self._category_object_ids.clear()
        self._object_positions.clear()
        for obj in md.get("objects", []):
            otype = obj.get("objectType")
            oid = obj.get("objectId")
            if otype and oid:
                self._category_object_ids.setdefault(otype.lower(), []).append(oid)
            pos = obj.get("position")
            if pos and oid:
                self._object_positions[oid] = (pos["x"], pos["y"], pos["z"])

    def _build_house_info(self) -> HouseInfo:
        rooms = self._house_data.get("rooms", [])
        return HouseInfo(
            house_id=str(self._house_data.get("houseId", "procthor-val")),
            dataset="procthor-10k",
            split="val",
            room_count=len(rooms),
            room_ids=[str(r.get("id", "")) for r in rooms],
            object_category_counts={k: len(v) for k, v in self._category_object_ids.items()},
            object_count=len(self._event.metadata.get("objects", [])) if self._event else 0,
            rendering_backend=self.LABEL,
            metadata={
                "split_file": self._house_data.get("_source", ""),
                "resolution": f"{self.width}x{self.height}",
                "quality": self.quality,
                "scene": self.scene,
            },
        )

    def target_category_object_ids(self, category: str) -> List[str]:
        return list(self._category_object_ids.get(category.lower(), []))

    def object_position(self, object_id: str) -> Optional[Any]:
        return self._object_positions.get(object_id)

    def reachable_positions(self) -> List[Any]:
        return list(self._reachable)

    def current_room(self) -> Optional[str]:
        if self._event is not None:
            return self._event.metadata.get("agent", {}).get("roomId")
        return None

    def house(self) -> Optional[HouseInfo]:
        return self._house_info

    def house_path(self) -> Optional[str]:
        """Filesystem path of the loaded house JSON, when one was supplied.

        Used by the live Unity God View to reuse the same real house geometry
        that the running controller already instantiated.
        """
        src = self.house_source
        if isinstance(src, (str, Path)):
            p = Path(src)
            return str(p) if p.is_file() else None
        return None

    def _objects(self) -> List[dict]:
        return self._event.metadata.get("objects", []) if self._event is not None else []

    # -- verifier: authoritative engine visibility -------------------
    def target_is_visible(self, object_id: str) -> bool:
        for obj in self._objects():
            if obj.get("objectId") == object_id:
                return bool(obj.get("visible", False))
        return False

    # -- reset / step ------------------------------------------------
    def reset(self, task: ObjectSearchTask, **opts) -> Dict[str, Any]:
        # re-instantiate the house fresh for the episode, then start pose.
        # The reset/spawn is environment setup -- it is NOT a model action, so
        # no synthetic AgentAction is fabricated (the old MoveAhead placeholder
        # made the first frame look like a model teleport).
        self.controller.step(dict(action="ResetObjectFilter"))
        ev = self.controller.step(dict(action="CreateHouse", house=self._house_data, renderImage=False))
        self._event = ev
        agent = opts.get("agent") or _agent_meta(self._house_data)
        self._teleport(agent)
        self._index_metadata(self._event.metadata)
        p = agent.get("position", {})
        if isinstance(p, dict):
            pos = [float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))]
        else:
            pos = [float(p[0]), float(p[1]), float(p[2])]
        rot = agent.get("rotation", {})
        yaw = float(rot.get("y", 0.0)) if isinstance(rot, dict) else 0.0
        self._last_setup = {
            "kind": "spawn",
            "origin": AgentActionOrigin.SPAWN.value,
            "position": pos,
            "rotation_yaw_deg": yaw,
            "horizon_deg": float(agent.get("horizon", 0.0)),
        }
        return self._snapshot(action=None)

    def apply_action(self, action_type, parameters, step) -> Dict[str, Any]:
        thor_name = _THOR_ACTION.get(action_type)
        if thor_name is None:
            raise ProcTHORError(f"Unsupported action for Thor: {action_type}")
        event = self.controller.step(dict(action=thor_name, **(parameters or {})))
        self._event = event
        md = event.metadata
        success = bool(md.get("lastActionSuccess", True))
        action = AgentAction(
            action_type=action_type,
            step=step,
            origin=AgentActionOrigin.MODEL.value,
            success=success,
            collision=bool(md.get("collided", False)),
            blocked=not success,
            error=None if success else md.get("errorMessage"),
            timestamp_ms=int(time.time() * 1000),
        )
        self._index_metadata(md)
        return self._snapshot(action=action)

    def _snapshot(self, action: Optional[AgentAction]) -> Dict[str, Any]:
        md = self._event.metadata
        ag = md.get("agent", {})
        pos = ag.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        rot = ag.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
        agent_state = AgentState(
            position=(float(pos["x"]), float(pos["y"]), float(pos["z"])),
            rotation_yaw_deg=float(rot.get("y", 0.0)),
            camera_horizon_deg=float(ag.get("cameraHorizon", 0.0)),
            is_crouching=bool(ag.get("isCrouching", False)),
            is_standing=bool(ag.get("isStanding", not bool(ag.get("isCrouching", False)))),
            step_count=int(md.get("step", action.step if action else 0)),
            last_action=action.describe() if action else None,
            last_action_success=action.success if action else None,
            collision=bool(action.collision) if action else False,
            room=ag.get("roomId"),
            timestamp_ms=int(time.time() * 1000),
        )
        obs = AgentObservation(
            step=action.step if action else 0,
            timestamp_ms=int(time.time() * 1000),
            rgb=self._event.frame,
            frame_id=f"procthor-{action.step if action else 0}",
            image_path=None,
            camera_horizon_deg=agent_state.camera_horizon_deg,
            cause_action=action,
        )
        return {"observation": obs, "agent_state": agent_state, "action": action}

    # -- researcher 3D scene geometry (God View) ---------------------
    def scene_geometry(self) -> Dict[str, Any]:
        """Compact 3D semantic scene from ProcTHOR house JSON + Thor metadata.

        Missing fields are reported as unavailable (``None`` / explicit
        ``unavailable`` list) instead of being fabricated.
        """
        from spatialforge.inspector.scene3d import build_scene3d

        md_objects = self._objects()
        return build_scene3d(self._house_data, thor_objects=md_objects)
