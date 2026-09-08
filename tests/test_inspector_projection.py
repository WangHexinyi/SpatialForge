"""Unit tests for ProjectionContract, legacy reconstruction, and Blender probe."""

import math
import unittest
from pathlib import Path

from spatialforge.environment.camera import CameraPose
from spatialforge.inspector.projection import (
    COORDINATE_CONVENTION_BLENDER,
    STATUS_EXACT,
    STATUS_RECONSTRUCTED_LEGACY,
    STATUS_UNKNOWN,
    compute_blender_sensor_dimensions,
    compute_frustum_corners_local,
    compute_perspective_matrix,
    make_unknown_projection,
    project_world_to_image_coordinates,
    reconstruct_legacy_projection,
    run_blender_projection_probe,
)


class TestInspectorProjection(unittest.TestCase):
    def setUp(self):
        self.camera = CameraPose(
            position=(0.0, -6.0, 3.0),
            look_at=(0.0, 0.0, 0.4),
            up=(0.0, 0.0, 1.0),
            fov_deg=60.0,  # nominal default
        )

    def test_reconstructed_legacy_square_aspect(self):
        """Historical 512x512 render with lens=35mm, sensor=36mm AUTO."""
        contract = reconstruct_legacy_projection(self.camera, res_x=512, res_y=512)

        self.assertEqual(contract.status, STATUS_RECONSTRUCTED_LEGACY)
        self.assertEqual(contract.camera_pose_nominal_fov_deg, 60.0)
        self.assertTrue(contract.is_frustum_available)
        self.assertIn("reconstructed_legacy", contract.provenance)
        self.assertIn("lens=35.0mm", contract.provenance)

        # In square 512x512 with 36mm sensor width and 35mm lens:
        # tan(fov/2) = (36 / 2) / 35 = 18 / 35 = 0.5142857...
        expected_fov = 2.0 * math.degrees(math.atan(18.0 / 35.0))
        self.assertAlmostEqual(contract.horizontal_fov_deg, expected_fov, places=5)
        self.assertAlmostEqual(contract.vertical_fov_deg, expected_fov, places=5)
        # Verify it is ~54.43 degrees, NOT 60.0 degrees!
        self.assertAlmostEqual(contract.horizontal_fov_deg, 54.4322, places=3)
        self.assertNotEqual(contract.horizontal_fov_deg, 60.0)

    def test_non_square_aspect_landscape(self):
        """Landscape 640x480 with AUTO fit fits sensor_width to X."""
        w_mm, h_mm, fov_h, fov_v = compute_blender_sensor_dimensions(
            lens_mm=35.0,
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
            sensor_fit="AUTO",
            res_x=640,
            res_y=480,
        )
        self.assertEqual(w_mm, 36.0)
        self.assertAlmostEqual(h_mm, 36.0 * (480 / 640), places=5)
        self.assertGreater(fov_h, fov_v)
        self.assertAlmostEqual(fov_h, 54.4322, places=3)
        self.assertAlmostEqual(fov_v, 2.0 * math.degrees(math.atan(13.5 / 35.0)), places=5)

    def test_non_square_aspect_portrait(self):
        """Portrait 480x640 with AUTO fit fits sensor_width to Y in Blender."""
        w_mm, h_mm, fov_h, fov_v = compute_blender_sensor_dimensions(
            lens_mm=35.0,
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
            sensor_fit="AUTO",
            res_x=480,
            res_y=640,
        )
        self.assertEqual(h_mm, 36.0)
        self.assertAlmostEqual(w_mm, 36.0 * (480 / 640), places=5)
        self.assertGreater(fov_v, fov_h)
        self.assertAlmostEqual(fov_v, 54.4322, places=3)
        self.assertAlmostEqual(fov_h, 2.0 * math.degrees(math.atan(13.5 / 35.0)), places=5)

    def test_unknown_projection(self):
        """Unknown intrinsics keeps nominal FOV but marks frustum unavailable."""
        unknown = make_unknown_projection(self.camera)

        self.assertEqual(unknown.status, STATUS_UNKNOWN)
        self.assertEqual(unknown.camera_pose_nominal_fov_deg, 60.0)
        self.assertIsNone(unknown.horizontal_fov_deg)
        self.assertIsNone(unknown.vertical_fov_deg)
        self.assertFalse(unknown.is_frustum_available)
        self.assertIn("unknown_intrinsics", unknown.provenance)

    def test_frustum_corners_local(self):
        """Local frustum corners structure."""
        corners = compute_frustum_corners_local(horizontal_fov_deg=54.43, vertical_fov_deg=54.43, near=0.2, far=5.0)
        self.assertIn("near", corners)
        self.assertIn("far", corners)
        self.assertEqual(len(corners["near"]), 4)
        self.assertEqual(len(corners["far"]), 4)

    def test_project_world_to_image_coordinates(self):
        """Verify projection of optical ray, boundary, and behind-camera points."""
        contract = reconstruct_legacy_projection(self.camera, res_x=512, res_y=512)

        # Optical ray (look_at) must project to center (0.5, 0.5)
        u, v, z, inside = project_world_to_image_coordinates(self.camera.look_at, self.camera, contract)
        self.assertTrue(inside)
        self.assertAlmostEqual(u, 0.5, places=5)
        self.assertAlmostEqual(v, 0.5, places=5)
        self.assertGreater(z, 0.0)

        # Point behind camera (e.g. at (0, -10, 3) when camera is at (0, -6, 3) facing +Y)
        behind = (0.0, -10.0, 3.0)
        u_b, v_b, z_b, inside_b = project_world_to_image_coordinates(behind, self.camera, contract)
        self.assertFalse(inside_b)
        self.assertIsNone(u_b)
        self.assertIsNone(v_b)
        self.assertLess(z_b, 0.0)

    def test_blender_projection_probe_if_available(self):
        """If Blender is installed, run probe and verify sub-millimeter match."""
        blender_path = "/root/autodl-tmp/tools/blender/blender"
        if not Path(blender_path).exists():
            self.skipTest("Blender executable not found at default path")

        res = run_blender_projection_probe(blender_executable=blender_path)
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["all_passed"], f"Blender probe failed cases: {res.get('cases')}")
        self.assertLess(res["max_error"], 1e-4)


if __name__ == "__main__":
    unittest.main()
