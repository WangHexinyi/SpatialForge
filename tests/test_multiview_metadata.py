"""Tests for multi-view camera generation and ObservationMetadata."""

import unittest
from pathlib import Path

from spatialforge.environment.camera import CameraPose
from spatialforge.environment.views import generate_cardinal_views, VIEW_IDS
from spatialforge.environment.observation import ObservationMetadata


class MultiViewCameraTests(unittest.TestCase):

    def test_generates_exactly_four_views(self):
        """The default multi-view camera generator produces exactly 4 views."""
        cameras = generate_cardinal_views()
        self.assertEqual(len(cameras), 4)

    def test_view_ids_order_is_deterministic(self):
        """View IDs and ordering are deterministic."""
        ids1 = list(VIEW_IDS)
        ids2 = list(VIEW_IDS)
        self.assertEqual(ids1, ids2)
        self.assertEqual(ids1, ["south", "east", "north", "west"])

    def test_all_cameras_have_distinct_positions(self):
        """All four cameras have distinct positions."""
        cameras = generate_cardinal_views()
        positions = [c.position for c in cameras]
        self.assertEqual(len(set(positions)), 4)

    def test_all_cameras_target_same_scene_target(self):
        """All four cameras target the same scene target."""
        cameras = generate_cardinal_views()
        targets = {c.look_at for c in cameras}
        self.assertEqual(len(targets), 1)

    def test_cameras_are_valid_camera_poses(self):
        """Each generated camera is a valid CameraPose."""
        cameras = generate_cardinal_views()
        for cam in cameras:
            self.assertIsInstance(cam, CameraPose)
            # No exception means valid


class ObservationMetadataTests(unittest.TestCase):

    def _make_obs(self) -> ObservationMetadata:
        camera = CameraPose(
            position=(0.0, -6.0, 3.0),
            look_at=(0.0, 0.0, 0.4),
            up=(0.0, 0.0, 1.0),
            fov_deg=60.0,
        )
        return ObservationMetadata(
            scene_id="scene_000",
            view_id="south",
            image_path="outputs/synth/scene_000_view_south.png",
            camera=camera,
        )

    def test_serialization_contains_complete_camera_metadata(self):
        """ObservationMetadata serialization contains position, look_at, up, fov_deg."""
        obs = self._make_obs()
        d = obs.to_dict()

        self.assertEqual(d["scene_id"], "scene_000")
        self.assertEqual(d["view_id"], "south")
        self.assertEqual(d["image_path"], "outputs/synth/scene_000_view_south.png")

        cam = d["camera"]
        self.assertIn("position", cam)
        self.assertIn("look_at", cam)
        self.assertIn("up", cam)
        self.assertIn("fov_deg", cam)

        self.assertEqual(cam["position"], [0.0, -6.0, 3.0])
        self.assertEqual(cam["look_at"], [0.0, 0.0, 0.4])
        self.assertEqual(cam["up"], [0.0, 0.0, 1.0])
        self.assertEqual(cam["fov_deg"], 60.0)

    def test_camera_not_serialized_as_opaque_string(self):
        """Camera field is a dict, not an opaque string."""
        obs = self._make_obs()
        d = obs.to_dict()
        self.assertIsInstance(d["camera"], dict)

    def test_serialization_is_json_compatible(self):
        """to_dict produces plain Python types (no tuples, no CameraPose)."""
        import json

        obs = self._make_obs()
        d = obs.to_dict()
        # Must not raise
        json.dumps(d)


if __name__ == "__main__":
    unittest.main()
