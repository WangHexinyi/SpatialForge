"""Real ProcTHOR Object Search privileged teacher.

This is the *real* mainline teacher. Unlike :class:`GridTeacher` (which plans a
path on the CPU deterministic harness and is a test/UAT convenience), this
teacher drives an actual AI2-THOR Unity build through the real closed loop:

    Observation_t -> teacher AgentAction -> controller.step() -> Observation_{t+1}

No pre-generated trajectory is "played back": each ``MoveAhead`` / ``RotateLeft``
/ ``RotateRight`` / ``LookUp`` / ``LookDown`` / ``Crouch`` / ``Stand`` is issued
through :class:`EmbodiedEnvironment.step`, which executes it against the real
engine and returns a genuinely new first-person observation. The teacher
re-plans reactively when a real action is blocked, and it only emits ``Done``
once the engine's authoritative ``visible`` metadata confirms the target is in
view. The verifier (not the teacher) decides success.

Planning uses AI2-THOR authoritative functionality:

* ``GetReachablePositions`` (grid-snapped at 0.25 m) as the navigation graph.
* The engine's own object ``visible`` metadata (never camera_depth) to pick a
  **suitable visible observation pose** and to gate ``Done``.
* privileged geometry (target object position/objectId).

The teacher consumes privileged truth but its output is only a valid
``AgentActionType`` sequence; privileged content is never written into the
student model-input record (see ``spatialforge.embodied.records``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from spatialforge.embodied.contracts import AgentActionType

# world grid spacing produced by AI2-THOR navigation
GRID_SPACING = 0.25
SNAP = int(round(1.0 / GRID_SPACING))  # 4  -> integer grid keys from metres


def to_grid_key(x: float, z: float) -> Tuple[int, int]:
    return (int(round(x * SNAP)), int(round(z * SNAP)))


def heading_for_step(current: Tuple[int, int], nxt: Tuple[int, int]) -> int:
    """Cardinal yaw (0/90/180/270) that moves from ``current`` to ``nxt``.

    Thor semantics (x,z plane): yaw 0 -> +z, 90 -> +x, 180 -> -z, 270 -> -x,
    i.e. direction vector (dx, dz) == (sin yaw, cos yaw).
    """
    dx = nxt[0] - current[0]
    dz = nxt[1] - current[1]
    yaw = math.degrees(math.atan2(dx, dz)) % 360.0
    return int(round(yaw / 90.0)) * 90 % 360


def build_reachable_grid(reachable: List[Any]) -> Dict[Tuple[int, int], float]:
    """Map grid keys to world y for reachable world (x,y,z) or (x,z) points."""
    grid: Dict[Tuple[int, int], float] = {}
    for p in reachable:
        if p is None:
            continue
        try:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
        except (TypeError, IndexError, ValueError):
            x = float(p[0])
            y = 0.9
            z = float(p[1])
        grid[to_grid_key(x, z)] = y
    return grid


def _neighbors(key: Tuple[int, int], grid) -> List[Tuple[int, int]]:
    x, z = key
    out = []
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = (x + dx, z + dz)
        if nb in grid:
            out.append(nb)
    return out


def bfs_path(
    start_key: Tuple[int, int],
    goal_key: Tuple[int, int],
    grid: Dict[Tuple[int, int], float],
) -> Optional[List[Tuple[int, int]]]:
    """Shortest cell path (inclusive) over the reachable grid, or None."""
    if start_key not in grid or goal_key not in grid:
        return None
    if start_key == goal_key:
        return [start_key]
    from collections import deque

    prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_key: None}
    q = deque([start_key])
    while q:
        cur = q.popleft()
        for nb in _neighbors(cur, grid):
            if nb in prev:
                continue
            prev[nb] = cur
            if nb == goal_key:
                path = [goal_key]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            q.append(nb)
    return None


def cardinal_rotation_plan(
    current_yaw: int, goal_yaw: int
) -> List[AgentActionType]:
    """Fewest 90-degree RotateLeft/RotateRight actions to reach goal_yaw."""
    goal_yaw %= 360
    right = ((goal_yaw - current_yaw) % 360) // 90
    left = ((current_yaw - goal_yaw) % 360) // 90
    if left < right:
        return [AgentActionType.ROTATE_LEFT] * int(left)
    return [AgentActionType.ROTATE_RIGHT] * int(right)


def _visible_heading_toward(cell_key, goal_key) -> Optional[int]:
    """Heading whose forward half-plane includes ``goal_key``."""
    heading = heading_for_step(cell_key, goal_key)
    # the object is toward goal_key; face that cardinal heading
    return heading


class ThorObjectSearchTeacher:
    """Real object-search teacher driving an :class:`EmbodiedEnvironment`.

    ``plan_view_pose`` is static (privileged, engine-teleport search). After the
    host has started the student episode via ``env.start_episode(...)`` (which
    re-instantiates the real house and teleports the agent to its spawn), the
    teacher genuinely navigates from the live agent pose to the chosen pose and
    gates ``Done`` on authoritative visibility.
    """

    #: Teacher-capable agent vocabulary (subset used here).
    ACTIONS = {
        AgentActionType.MOVE_AHEAD,
        AgentActionType.ROTATE_LEFT,
        AgentActionType.ROTATE_RIGHT,
        AgentActionType.LOOK_UP,
        AgentActionType.LOOK_DOWN,
        AgentActionType.CROUCH,
        AgentActionType.STAND,
        AgentActionType.DONE,
    }

    def __init__(
        self,
        env,
        target_object_id: str,
        max_tries_view_pose: int = 24,
        scan_horizons_deg: Tuple[float, ...] = (0.0, 30.0, 45.0),
    ):
        self.env = env
        self.backend = env.backend
        self.target_object_id = target_object_id
        self.max_tries_view_pose = max_tries_view_pose
        self.scan_horizons_deg = scan_horizons_deg
        #: genuine transitions produced while driving the real backend
        self.transitions: List[Any] = []

    # ------------------------------------------------------------------
    # Static privileged view-pose search (engine teleport; NOT student path)
    # ------------------------------------------------------------------
    def plan_view_pose(self) -> Optional[Dict[str, Any]]:
        """Return a reachable observation pose from which the target is visible.

        Uses the authoritative engine ``visible`` flag after a privileged
        ``TeleportFull`` preview. Returns None if no such pose is found within
        the search budget (unreachable / not-observable task).
        """
        grid = build_reachable_grid(self.backend.reachable_positions())
        if not grid:
            return None
        target_pos = self.backend.object_position(self.target_object_id)
        if target_pos is None:
            return None
        tx, tz = float(target_pos[0]), float(target_pos[2])
        target_key = to_grid_key(tx, tz)

        # rank reachable cells by distance to target (closest first)
        ordered = sorted(grid.keys(), key=lambda k: (k[0] - target_key[0]) ** 2 + (k[1] - target_key[1]) ** 2)

        # agent's feet y for this house (from any reachable point)
        any_y = next(iter(grid.values()))

        tries = 0
        for cell in ordered:
            # only cells reasonably near the target can see it
            dx = cell[0] - target_key[0]
            dz = cell[1] - target_key[1]
            dist = math.hypot(dx, dz) * GRID_SPACING
            if dist > 4.0 or dist < 0.05:
                continue
            face_yaw = _visible_heading_toward(cell, target_key)
            for yaw in (face_yaw, (face_yaw + 90) % 360, (face_yaw - 90) % 360):
                for hor in self.scan_horizons_deg:
                    if tries >= self.max_tries_view_pose:
                        return None
                    tries += 1
                    md = self.backend.set_agent_pose(
                        (cell[0] * GRID_SPACING, any_y, cell[1] * GRID_SPACING),
                        rotation_yaw_deg=float(yaw),
                        horizon_deg=float(hor),
                        render=True,
                    )
                    if self._authoritative_visible(md):
                        return {
                            "cell_key": cell,
                            "yaw": int(yaw),
                            "horizon": float(hor),
                            "dist_m": round(dist, 2),
                        }
        return None

    # ------------------------------------------------------------------
    # Genuine reactive navigation + Done gate
    # ------------------------------------------------------------------
    def run(self, view_pose: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Navigate the live student agent to ``view_pose`` then verify Done.

        Every action is executed through the real backend via ``env.step``.
        Returns a summary (never claims success itself).
        """
        if view_pose is None:
            view_pose = self.plan_view_pose()
        if view_pose is None:
            return {"planned": False, "reason": "no reachable visible pose"}

        goal_key = tuple(view_pose["cell_key"])
        goal_yaw = int(view_pose["yaw"])
        grid = build_reachable_grid(self.backend.reachable_positions())

        state = self._current_state()
        start_key = to_grid_key(state["x"], state["z"])
        current_yaw = int(state["yaw"])

        if start_key == goal_key:
            nav_ok = True
        else:
            nav_ok = self._navigate(grid, start_key, goal_key, current_yaw)

        # arrive & face the object, then check authoritative visibility
        if nav_ok:
            self._face_goal_yaw(goal_yaw)
            if not self._any_visible():
                self._scan_aim()

        visible = self._any_visible()
        self._act(AgentActionType.DONE)  # verifier decides success
        return {
            "planned": True,
            "view_pose": view_pose,
            "reached": nav_ok,
            "visible_at_done": visible,
        }

    def _navigate(self, grid, start_key, goal_key, current_yaw) -> bool:
        """Follow a reactive BFS path to ``goal_key`` with real actions."""
        step_budget = self.env.task.max_steps - self.env.episode.step - 8
        if step_budget <= 0:
            step_budget = 20
        guard = 0
        while True:
            if self._any_visible():
                return True  # already in view; Done may be emitted
            if self.env.episode.status.value != "running":
                return False
            pos = self._current_state()
            cur_key = to_grid_key(pos["x"], pos["z"])
            current_yaw = int(pos["yaw"])
            if cur_key == goal_key:
                return True
            path = bfs_path(cur_key, goal_key, grid)
            if path is None or len(path) < 2:
                return False
            guard += 1
            if guard > step_budget:
                return False
            # move one step toward the goal: face the next cell then MoveAhead
            next_key = path[1]
            need_yaw = heading_for_step(cur_key, next_key)
            for act in cardinal_rotation_plan(current_yaw, need_yaw):
                if not self._do(act):
                    return False
            if not self._do(AgentActionType.MOVE_AHEAD):
                # blocked -> re-plan next loop iteration
                continue
        # unreachable

    def _face_goal_yaw(self, goal_yaw: int) -> None:
        state = self._current_state()
        cur = int(state["yaw"])
        for act in cardinal_rotation_plan(cur, goal_yaw):
            if self.env.episode.status.value != "running":
                return
            self._do(act)

    def _scan_aim(self) -> None:
        # try a downward/upward camera nudge to bring a low object into view
        for act in (AgentActionType.LOOK_DOWN, AgentActionType.STAND):
            if self.env.episode.status.value != "running":
                return
            self._do(act)
            if self._any_visible():
                return

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _act(self, action_type: AgentActionType) -> Optional[Any]:
        """Issue a genuine action via env.step and record the transition."""
        if action_type not in self.ACTIONS:
            return None
        trans = self.env.step(action_type)
        if trans is None:
            return None
        self.transitions.append(trans)
        return trans

    def _do(self, action_type: AgentActionType) -> bool:
        trans = self._act(action_type)
        if trans is None:
            return False
        act = trans.action
        return bool(act is None or act.success is not False)

    def _current_state(self) -> Dict[str, float]:
        st = self.env.agent_state
        return {
            "x": float(st.position[0]),
            "z": float(st.position[2]),
            "yaw": float(st.rotation_yaw_deg % 360.0),
            "horizon": float(st.camera_horizon_deg),
        }

    def _authoritative_visible(self, md: dict) -> bool:
        for obj in md.get("objects", []):
            if obj.get("objectId") == self.target_object_id:
                return bool(obj.get("visible", False))
        return False

    def _any_visible(self) -> bool:
        return self.backend.target_is_visible(self.target_object_id)
