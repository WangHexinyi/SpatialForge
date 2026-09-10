"""Deterministic grid backend (CPU test harness, NOT a training source).

A tiny, fully-deterministic 2D grid "house" on which the closed-loop
environment, contracts, transitions and the Done verifier can be exercised
without any rendering binary. RGB frames are synthesized deterministically from
the grid so the observation pipeline (frame id, allowlisted model input,
history, isolation) is real end-to-end.

This backend is intentionally separated and labelled; it must never appear as a
ProcTHOR environment or as training data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from spatialforge.embodied.contracts import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    AgentState,
    HouseInfo,
    ObjectSearchTask,
)
from spatialforge.embodied.environment import EmbodiedBackend

HEADINGS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}  # +x, +y, -x, -y

_CAT_COLORS = {
    "mug": (0x30, 0x70, 0xC0),
    "couch": (0xC0, 0x50, 0x30),
    "chair": (0x40, 0xA0, 0x40),
    "lamp": (0xC0, 0xC0, 0x20),
    "plant": (0x20, 0x80, 0x40),
    "wall": (0x60, 0x60, 0x60),
    "floor": (0x34, 0x34, 0x38),
    "sky": (0x20, 0x28, 0x30),
    "self": (0xE0, 0xE0, 0xE0),
}


@dataclass
class GridObject:
    object_id: str
    category: str
    cell: tuple
    visible_from: List[tuple] = None


class DeterministicGridBackend(EmbodiedBackend):
    """Deterministic grid 'house'. Reachable cells + category objects."""

    LABEL = "deterministic-grid (test harness only)"

    def __init__(self, size: int = 8, seed: int = 0):
        self.size = size
        self.rng = np.random.RandomState(seed)
        self._house = None
        self._agent_cell = None
        self._heading = 0
        self._crouched = False
        self._objects = []
        self._rgb_cache = None

    # -- house construction -------------------------------------------
    def _build_house(self) -> HouseInfo:
        size = self.size
        reachable = set()
        for x in range(size):
            for y in range(size):
                reachable.add((x, y))
        objects = []
        # place objects deterministically along the perimeter-ish pattern
        for idx in range(6):
            cat = list(_CAT_COLORS.keys())[idx % len(_CAT_COLORS)]
            cell = (min(idx + 2, size - 1), (idx * 3 + 1) % size)
            objects.append(
                GridObject(object_id=f"det-{idx}", category=cat, cell=cell)
            )
        counts: Dict[str, int] = {}
        for o in objects:
            counts[o.category] = counts.get(o.category, 0) + 1
        self._objects = objects
        self._reachable = reachable
        return HouseInfo(
            house_id=f"det-grid-{size}-{self.seed}",
            dataset="deterministic-test",
            split="none",
            room_count=1,
            room_ids=["room_0"],
            object_category_counts=counts,
            object_count=len(objects),
            seed=self.seed,
            rendering_backend=self.LABEL,
            navmesh_reachable_count=len(reachable),
            agent_start=(1, 1, 0.0),
        )

    def load_house(self, house_id: str, **opts) -> HouseInfo:
        self.seed = int(opts.get("seed", 0))
        self.rng = np.random.RandomState(self.seed)
        self._house = self._build_house()
        return self._house

    # -- metadata ------------------------------------------------------
    def target_category_object_ids(self, category: str) -> List[str]:
        return [o.object_id for o in self._objects if o.category == category]

    def object_position(self, object_id: str) -> Optional[Any]:
        for o in self._objects:
            if o.object_id == object_id:
                x, y = o.cell
                return (float(x), float(y), 0.0)
        return None

    def reachable_positions(self) -> List[Any]:
        return [tuple(c) for c in sorted(self._reachable)]

    def current_room(self) -> Optional[str]:
        return "room_0"

    def house(self) -> HouseInfo:
        return self._house

    # -- reset / step --------------------------------------------------
    def reset(self, task: ObjectSearchTask, **opts) -> Dict[str, Any]:
        start = opts.get("start", self._house.agent_start[:2])
        self._agent_cell = (int(start[0]), int(start[1]))
        self._heading = int(opts.get("start_heading", 0))
        self._crouched = False
        return self._raw_result(_noop_action())

    def apply_action(self, action_type, parameters, step) -> Dict[str, Any]:
        action = AgentAction(action_type=action_type, step=step, success=False)
        if action_type == AgentActionType.MOVE_AHEAD:
            action.success, action.blocked = self._move()
        elif action_type == AgentActionType.MOVE_BACK:
            self._heading = (self._heading + 2) % 4
            action.success, action.blocked = self._move()
            self._heading = (self._heading + 2) % 4
        elif action_type == AgentActionType.ROTATE_LEFT:
            self._heading = (self._heading + 3) % 4
            action.success = True
        elif action_type == AgentActionType.ROTATE_RIGHT:
            self._heading = (self._heading + 1) % 4
            action.success = True
        elif action_type == AgentActionType.CROUCH:
            self._crouched = True
            action.success = True
        elif action_type == AgentActionType.STAND:
            self._crouched = False
            action.success = True
        elif action_type in (
            AgentActionType.LOOK_UP,
            AgentActionType.LOOK_DOWN,
            AgentActionType.DONE,
        ):
            action.success = True
        else:
            action.success = False
            action.error = "unsupported"
        return self._raw_result(action)

    def _move(self) -> tuple:
        dx, dy = HEADINGS[self._heading]
        nx, ny = self._agent_cell[0] + dx, self._agent_cell[1] + dy
        if (nx, ny) in self._reachable and 0 <= nx < self.size and 0 <= ny < self.size:
            self._agent_cell = (nx, ny)
            return True, False
        return False, True

    # -- visibility (authoritative, deterministic) ---------------------
    def target_is_visible(self, object_id: str) -> bool:
        for o in self._objects:
            if o.object_id != object_id:
                continue
            dx, dy = o.cell[0] - self._agent_cell[0], o.cell[1] - self._agent_cell[1]
            fx, fy = HEADINGS[self._heading]
            forward = dx * fx + dy * fy
            if forward <= 0:
                return False
            # object straight ahead within 3 cells (orthogonal corridor):
            return abs(dx * fy - dy * fx) == 0 and forward <= 3
        return False

    # -- synthetic deterministic first-person RGB ----------------------
    def _raw_result(self, action: AgentAction) -> Dict[str, Any]:
        rgb = self._render_frame()
        agent_state = AgentState(
            position=(float(self._agent_cell[0]), float(self._agent_cell[1]), 0.0),
            rotation_yaw_deg=float(self._heading * 90),
            camera_horizon_deg=0.0,
            is_crouching=self._crouched,
            is_standing=not self._crouched,
            room=self.current_room(),
            timestamp_ms=int(time.time() * 1000),
        )
        obs = AgentObservation(
            step=action.step,
            timestamp_ms=int(time.time() * 1000),
            rgb=rgb,
            frame_id=f"det-{action.step}",
            camera_horizon_deg=0.0,
            cause_action=action,
        )
        return {"observation": obs, "agent_state": agent_state, "action": action}

    def _render_frame(self) -> np.ndarray:
        W = H = 96
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :] = _CAT_COLORS["sky"]
        cx, cy = self._agent_cell
        fx, fy = HEADINGS[self._heading]
        # perspective-ish projection: rows toward forward
        for o in self._objects:
            vx, vy = o.cell[0] - cx, o.cell[1] - cy
            forward = vx * fx + vy * fy
            lateral = vx * fy - vy * fx
            if forward <= 0 or forward > 4:
                continue
            col = W // 2 + int(lateral / max(forward, 1) * (W / 4))
            if 0 <= col < W:
                row = int(H * 0.55 - forward * (H / 14))
                size = max(4, int(10 - forward * 1.5))
                color = _CAT_COLORS.get(o.category, (255, 255, 255))
                img[max(0, row - size): row + size, max(0, col - size): col + size] = color
        # floor band
        img[int(H * 0.7):, :] = _CAT_COLORS["floor"]
        return img

    def close(self) -> None:
        pass


def _noop_action() -> AgentAction:
    return AgentAction(
        action_type=AgentActionType.MOVE_AHEAD,
        step=0,
        success=True,
        timestamp_ms=int(time.time() * 1000),
    )
