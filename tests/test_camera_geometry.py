import math
import unittest

from spatialforge.environment.camera import CameraPose
from spatialforge.environment.geometry import (
    camera_basis,
    camera_depth,
    metric_distance,
    world_to_camera,
)


class CameraGeometryTests(unittest.TestCase):

    def test_basis_is_orthonormal(self):
        camera = CameraPose(
            position=(0.0, -6.0, 3.0),
            look_at=(0.0, 0.0, 4.0),
        )

        right, up, forward = camera_basis(camera)

        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        def norm(v):
            return math.sqrt(dot(v, v))

        self.assertAlmostEqual(norm(right), 1.0)
        self.assertAlmostEqual(norm(up), 1.0)
        self.assertAlmostEqual(norm(forward), 1.0)

        self.assertAlmostEqual(dot(right, up), 0.0)
        self.assertAlmostEqual(dot(right, forward), 0.0)
        self.assertAlmostEqual(dot(up, forward), 0.0)

    def test_v1_view_preserves_world_x_left_right(self):
        camera = CameraPose(
            position=(0.0, -6.0, 3.0),
            look_at=(0.0, 0.0, 4.0),
        )

        left = world_to_camera((-1.0, 0.0, 0.5), camera)
        right = world_to_camera((1.0, 0.0, 0.5), camera)

        self.assertLess(left[0], 0.0)
        self.assertGreater(right[0], 0.0)

    def test_opposite_view_flips_left_right(self):
        point_a = (-1.0, 0.0, 0.0)
        point_b = (1.0, 0.0, 0.0)

        south_camera = CameraPose(
            position=(0.0, -6.0, 0.0),
            look_at=(0.0, 0.0, 0.0),
        )

        north_camera = CameraPose(
            position=(0.0, 6.0, 0.0),
            look_at=(0.0, 0.0, 0.0),
        )

        south_a = world_to_camera(point_a, south_camera)[0]
        south_b = world_to_camera(point_b, south_camera)[0]

        north_a = world_to_camera(point_a, north_camera)[0]
        north_b = world_to_camera(point_b, north_camera)[0]

        self.assertLess(south_a, south_b)
        self.assertGreater(north_a, north_b)

    def test_metric_distance_is_orientation_invariant(self):
        point = (2.0, 4.0, 1.0)

        camera_a = CameraPose(
            position=(0.0, 0.0, 0.0),
            look_at=(0.0, 1.0, 0.0),
        )

        camera_b = CameraPose(
            position=(0.0, 0.0, 0.0),
            look_at=(1.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(
            metric_distance(point, camera_a),
            metric_distance(point, camera_b),
        )

    def test_camera_depth_depends_on_orientation(self):
        point = (0.0, 5.0, 0.0)

        facing_point = CameraPose(
            position=(0.0, 0.0, 0.0),
            look_at=(0.0, 1.0, 0.0),
        )

        facing_sideways = CameraPose(
            position=(0.0, 0.0, 0.0),
            look_at=(1.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(
            camera_depth(point, facing_point),
            5.0,
        )

        self.assertAlmostEqual(
            camera_depth(point, facing_sideways),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
