import unittest
from pathlib import Path

from spatialforge.environment.camera import CameraPose
from spatialforge.environment.geometry import camera_basis


class CameraPoseValidationTests(unittest.TestCase):

    def test_position_and_look_at_cannot_match(self):
        with self.assertRaises(ValueError):
            CameraPose(
                position=(0.0, 0.0, 0.0),
                look_at=(0.0, 0.0, 0.0),
            )

    def test_fov_must_be_strictly_between_zero_and_180(self):
        for bad_fov in (0.0, -1.0, 180.0, 181.0):
            with self.subTest(fov=bad_fov):
                with self.assertRaises(ValueError):
                    CameraPose(
                        position=(0.0, 0.0, 0.0),
                        look_at=(0.0, 1.0, 0.0),
                        fov_deg=bad_fov,
                    )

    def test_up_vector_cannot_be_parallel_to_view_direction(self):
        camera = CameraPose(
            position=(0.0, 0.0, 0.0),
            look_at=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
        )

        with self.assertRaises(ValueError):
            camera_basis(camera)

    def test_environment_core_has_no_blender_dependency(self):
        root = Path("spatialforge/environment")

        for path in root.glob("*.py"):
            source = path.read_text(encoding="utf-8-sig")

            with self.subTest(file=str(path)):
                self.assertNotIn("import bpy", source)
                self.assertNotIn("from bpy", source)


if __name__ == "__main__":
    unittest.main()
