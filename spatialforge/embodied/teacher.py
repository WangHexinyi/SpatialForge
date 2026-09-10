"""Teacher / reference trajectory generation (privileged channel).

The teacher is an **oracle baseline** used for behaviour-cloning data
generation, debugging, and future evaluation comparison. It may consume
privileged truth (reachable set, target location, navigation graph) but its
output must never enter model input.

For the deterministic grid backend this computes a true shortest path in the
state space (x, y, heading) and emits an :class:`AgentActionType` sequence
ending in ``Done``. The exact path is a real BFS shortest path, not a heuristic.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from spatialforge.embodied.contracts import AgentActionType

HEADINGS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


@dataclass
class TeacherPath:
    """A privileged reference trajectory (never model input)."""

    actions: List[str] = field(default_factory=list)
    visited_cells: List[Tuple[int, int]] = field(default_factory=list)
    planned: bool = False
    detail: str = ""

    def to_agent_action_list(self) -> List[AgentActionType]:
        return [AgentActionType(a) for a in self.actions]


class GridTeacher:
    """BFS shortest-path teacher over a deterministic grid backend."""

    def __init__(self, backend):
        self.backend = backend

    def plan_to_visible(
        self, task_target_cell: Tuple[int, int], start_cell: Tuple[int, int],
        start_heading: int,
    ) -> TeacherPath:
        reachable = {tuple(c) for c in self.backend.reachable_positions()}

        # Choose a heading that points at the target from an adjacent reachable
        # cell, so that after arriving the object is directly ahead (visible).
        best = None
        for d, (dx, dy) in HEADINGS.items():
            behind = (task_target_cell[0] - dx, task_target_cell[1] - dy)
            if behind in reachable:
                cost = self._bfs_dist(
                    start_cell, start_heading, behind, d, reachable
                )
                if cost is not None and (best is None or cost < best[0]):
                    best = (cost, behind, d)
        if best is None:
            return TeacherPath(planned=False, detail="no reachable view pose")
        _, goal_cell, goal_heading = best
        path = self._bfs_path(start_cell, start_heading, goal_cell, goal_heading, reachable)
        if path is None:
            return TeacherPath(planned=False, detail="unreachable")
        return TeacherPath(
            actions=path,
            planned=True,
            detail=f"shortest path -> view pose {goal_cell} heading {goal_heading*90}deg",
        )

    # -- BFS helpers ----------------------------------------------
    def _bfs_dist(self, s, sh, g, gh, reachable) -> Optional[int]:
        path = self._bfs_path(s, sh, g, gh, reachable)
        return None if path is None else len(path)

    def _bfs_path(self, s, sh, g, gh, reachable) -> Optional[List[str]]:
        start = (s[0], s[1], sh)
        goal = (g[0], g[1], gh)
        if start == goal:
            return []
        prev = {}
        q = deque([start])
        visited = {start}
        while q:
            x, y, h = q.popleft()
            for act in (
                AgentActionType.ROTATE_LEFT,
                AgentActionType.ROTATE_RIGHT,
                AgentActionType.MOVE_AHEAD,
            ):
                nh = h
                if act == AgentActionType.ROTATE_LEFT:
                    nh = (h + 3) % 4
                    nxt = (x, y, nh)
                elif act == AgentActionType.ROTATE_RIGHT:
                    nh = (h + 1) % 4
                    nxt = (x, y, nh)
                else:
                    dx, dy = HEADINGS[h]
                    nx, ny = x + dx, y + dy
                    nxt = (nx, ny, h) if (nx, ny) in reachable else None
                if nxt is None or nxt in visited:
                    continue
                visited.add(nxt)
                prev[nxt] = ((x, y, h), act)
                if nxt == goal:
                    return self._reconstruct(prev, start, goal)
                q.append(nxt)
        return None

    @staticmethod
    def _reconstruct(prev, start, goal) -> List[str]:
        actions = []
        cur = goal
        while cur != start:
            cur, act = prev[cur]
            actions.append(act.value)
        actions.reverse()
        return actions


class AStarGridTeacher(GridTeacher):
    """Optional A* variant (same interface); kept for future expansion."""

    pass
