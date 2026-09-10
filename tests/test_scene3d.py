import unittest

from spatialforge.inspector.scene3d import (
    SCENE_SCHEMA,
    build_scene3d,
    episode_scene_payload,
)


def _house():
    return {
        "houseId": "house-test",
        "rooms": [{
            "id": "room|1", "roomType": "Kitchen",
            "floorPolygon": [
                {"x": 0, "y": 0, "z": 0}, {"x": 4, "y": 0, "z": 0},
                {"x": 4, "y": 0, "z": 4}, {"x": 0, "y": 0, "z": 4},
            ],
        }],
        "walls": [{
            "id": "wall|1", "roomId": "room|1", "empty": False,
            "polygon": [
                {"x": 0, "y": 0, "z": 0}, {"x": 4, "y": 0, "z": 0},
                {"x": 0, "y": 2.9, "z": 0}, {"x": 4, "y": 2.9, "z": 0},
            ],
        }],
        "doors": [{
            "id": "door|1", "assetId": "Door_1", "room0": "room|1", "room1": "room|2",
            "wall0": "wall|1", "holePolygon": [
                {"x": 0.1, "y": 0, "z": 0}, {"x": 1.0, "y": 2.0, "z": 0},
            ],
            "assetPosition": {"x": 2.0, "y": 1.0, "z": 0.0},
        }],
        "windows": [],
        "objects": [{
            "id": "Mug|1|0", "assetId": "Mug_1",
            "position": {"x": 1.0, "y": 0.9, "z": 2.0},
            "rotation": {"x": 0, "y": 90.0, "z": 0},
        }],
        "metadata": {"agent": {
            "position": {"x": 3.0, "y": 0.9, "z": 3.0},
            "rotation": {"x": 0, "y": 180, "z": 0}, "horizon": 30,
        }},
    }


class Scene3DBuildTests(unittest.TestCase):
    def test_build_geometry(self):
        scene = build_scene3d(_house())
        self.assertEqual(scene["schema"], SCENE_SCHEMA)
        self.assertEqual(scene["source"], "procthor_house_json")
        self.assertEqual(len(scene["rooms"]), 1)
        self.assertEqual(len(scene["walls"]), 1)
        self.assertAlmostEqual(scene["walls"][0]["height"], 2.9)
        self.assertEqual(len(scene["doors"]), 1)
        self.assertIsNotNone(scene["doors"][0]["position"])
        self.assertEqual(scene["objects"][0]["category"], "Mug")
        self.assertIsNone(scene["objects"][0]["aabb"])
        self.assertTrue(any("bounding boxes" in u for u in scene["unavailable"]))
        self.assertEqual(scene["agent_start"]["yaw_deg"], 180.0)

    def test_build_with_thor_metadata_includes_aabb(self):
        thor = [{
            "objectId": "Mug|1|0", "visible": True,
            "axisAlignedBoundingBox": {
                "center": {"x": 1.0, "y": 0.9, "z": 2.0},
                "size": {"x": 0.1, "y": 0.2, "z": 0.1},
            },
        }]
        scene = build_scene3d(_house(), thor_objects=thor)
        self.assertEqual(scene["source"], "procthor_house_json+ai2thor_metadata")
        self.assertIsNotNone(scene["objects"][0]["aabb"])
        self.assertTrue(scene["objects"][0]["visible"])

    def test_episode_payload_trace_and_targets(self):
        scene = build_scene3d(_house())
        timeline = [
            {"step": 0, "action_origin": "setup", "agent": {"position": [3.0, 0.9, 3.0], "rotation_yaw_deg": 0, "camera_horizon_deg": 0}},
            {"step": 1, "action": "MoveAhead", "action_origin": "model", "agent": {"position": [3.0, 0.9, 2.75], "rotation_yaw_deg": 0, "camera_horizon_deg": 0}},
            {"step": 2, "action": "RotateRight", "action_origin": "model", "agent": {"position": [3.0, 0.9, 2.75], "rotation_yaw_deg": 90, "camera_horizon_deg": 0}},
        ]
        payload = episode_scene_payload(
            scene,
            spawn={"position": [3.0, 0.9, 3.0], "yaw": 0, "horizon": 0},
            timeline=timeline,
            target_positions=[[1.0, 0.9, 2.0]],
            target_object_ids=["Mug|1|0"],
            terminal=False,
        )
        ep = payload["episode"]
        self.assertEqual(ep["trace"][0], [3.0, 0.9, 3.0])
        self.assertEqual(len(ep["trace"]), 2)
        self.assertEqual(ep["targets"][0]["object_id"], "Mug|1|0")
        self.assertEqual(ep["agent"]["rotation_yaw_deg"], 90)
        # a clean episode is one contiguous segment with no breaks
        self.assertEqual(len(ep["trace_segments"]), 1)
        self.assertEqual(ep["trace_breaks"], [])

    def test_trajectory_breaks_on_setup_displacement(self):
        """A recorded spawn that != the real first-frame pose must not be drawn
        as a straight (wall-crossing) segment into the first model action."""
        scene = build_scene3d(_house())
        timeline = [
            {"idx": 0, "step": 0, "action_origin": "setup",
             "agent": {"position": [9.25, 0.9, 8.25], "rotation_yaw_deg": 0, "camera_horizon_deg": 0}},
            {"idx": 1, "step": 1, "action": "RotateRight", "action_origin": "model",
             "agent": {"position": [11.5, 0.9, 6.5], "rotation_yaw_deg": 180, "camera_horizon_deg": 30}},
            {"idx": 2, "step": 2, "action": "MoveAhead", "action_origin": "model",
             "agent": {"position": [11.5, 0.9, 6.25], "rotation_yaw_deg": 180, "camera_horizon_deg": 30}},
        ]
        payload = episode_scene_payload(
            scene,
            spawn={"position": [9.25, 0.9, 8.25], "yaw": 0, "horizon": 0},
            timeline=timeline,
        )
        ep = payload["episode"]
        self.assertEqual(len(ep["trace_segments"]), 2)
        self.assertEqual(ep["trace_breaks"][0]["reason"], "unexplained_displacement")
        # the first segment holds only the setup pose; it never reaches the
        # first model pose, so no setup->model edge is drawn.
        self.assertEqual(ep["trace_segments"][0], [[9.25, 0.9, 8.25]])
        self.assertNotIn([11.5, 0.9, 6.5], ep["trace_segments"][0])

    def test_trajectory_breaks_on_teleport_origin(self):
        scene = build_scene3d(_house())
        timeline = [
            {"idx": 0, "step": 0, "action_origin": "setup",
             "agent": {"position": [1.0, 0.9, 1.0], "rotation_yaw_deg": 0, "camera_horizon_deg": 0}},
            {"idx": 1, "step": 1, "action": "MoveAhead", "action_origin": "model",
             "agent": {"position": [1.0, 0.9, 1.25], "rotation_yaw_deg": 0, "camera_horizon_deg": 0}},
            {"idx": 2, "step": 2, "action": None, "action_origin": "teleport",
             "agent": {"position": [8.0, 0.9, 8.0], "rotation_yaw_deg": 0, "camera_horizon_deg": 0}},
        ]
        payload = episode_scene_payload(
            scene,
            spawn={"position": [1.0, 0.9, 1.0], "yaw": 0, "horizon": 0},
            timeline=timeline,
        )
        ep = payload["episode"]
        self.assertEqual(len(ep["trace_segments"]), 2)
        self.assertEqual(ep["trace_breaks"][0]["reason"], "setup_origin")


if __name__ == "__main__":
    unittest.main()
