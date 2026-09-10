import unittest

from spatialforge.embodied.contracts import AgentActionType
from spatialforge.embodied.procthor_teacher import (
    GRID_SPACING,
    bfs_path,
    build_reachable_grid,
    cardinal_rotation_plan,
    heading_for_step,
    to_grid_key,
)


def open_grid(world_cols, world_rows):
    pts = [
        (GRID_SPACING * i, 0.9, GRID_SPACING * j)
        for i in range(world_cols) for j in range(world_rows)
    ]
    return build_reachable_grid(pts)


class NavHelperTests(unittest.TestCase):
    def test_snap_and_heading(self):
        self.assertEqual(to_grid_key(0.25, 0.25), (1, 1))
        # yaw semantics in Thor x,z plane
        self.assertEqual(heading_for_step((0, 0), (0, 1)), 0)    # +z
        self.assertEqual(heading_for_step((0, 0), (1, 0)), 90)   # +x
        self.assertEqual(heading_for_step((0, 0), (0, -1)), 180)  # -z
        self.assertEqual(heading_for_step((0, 0), (-1, 0)), 270)  # -x

    def test_bfs_shortest_and_disconnected(self):
        grid = open_grid(6, 6)
        start = to_grid_key(0.25, 0.25)
        goal = to_grid_key(0.25, 1.25)
        path = bfs_path(start, goal, grid)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 5)  # 1.0 m straight line of 0.25 steps
        # unreachable / off-grid goal
        self.assertIsNone(bfs_path(start, to_grid_key(99, 99), grid))

    def test_rotation_plan_minimal(self):
        self.assertEqual([a.value for a in cardinal_rotation_plan(0, 90)], ["RotateRight"])
        self.assertEqual([a.value for a in cardinal_rotation_plan(0, 270)], ["RotateLeft"])
        self.assertEqual(cardinal_rotation_plan(0, 0), [])
        self.assertEqual([a.value for a in cardinal_rotation_plan(90, 270)], ["RotateRight", "RotateRight"])

    def test_all_generated_actions_are_valid_agent_actions(self):
        grid = open_grid(12, 12)
        start = to_grid_key(0.25, 0.25)
        goal = to_grid_key(2.0, 2.0)
        path = bfs_path(start, goal, grid)
        self.assertIsNotNone(path)
        yaw = 0
        for a, b in zip(path, path[1:]):
            need = heading_for_step(a, b)
            for act in cardinal_rotation_plan(yaw, need):
                self.assertIn(act, (AgentActionType.ROTATE_LEFT, AgentActionType.ROTATE_RIGHT))
            yaw = need
        self.assertGreater(len(path), 1)


if __name__ == "__main__":
    unittest.main()
