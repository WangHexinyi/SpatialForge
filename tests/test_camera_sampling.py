"""CPU-only tests for the G2.0-B.1 camera sampling system (no bpy)."""

import math
import random
import unittest

from spatialforge.environment.camera import CameraPose
from spatialforge.environment.geometry import camera_basis
from spatialforge.environment.views import VIEW_IDS, generate_cardinal_views
from spatialforge.environment.sampling import (
    DIRECTIONS_AXIS,
    DIRECTIONS_26,
    canonical_camera,
    canonical_directions,
    canonical_views,
    jitter_pose,
    sample_camera,
)

_TARGET = (0.0, 0.0, 0.4)
_RADIUS = 6.0

_AXIS_IDS = ("east", "west", "north", "south", "up", "down")

# Documented deterministic ordering, written out independently of the
# production generators (see the module docstring contract):
#   axes first; for 26 the 12 edges in xy/xz/yz order then the 8 corners;
#   for 14 the 8 corners in their documented order.
_EXPECTED_IDS_14 = (
    "east", "west", "north", "south", "up", "down",
    "east_north_up", "east_north_down", "east_south_up", "east_south_down",
    "west_north_up", "west_north_down", "west_south_up", "west_south_down",
)
_EXPECTED_IDS_26 = (
    "east", "west", "north", "south", "up", "down",
    "east_north", "east_south", "west_north", "west_south",
    "east_up", "east_down", "west_up", "west_down",
    "north_up", "north_down", "south_up", "south_down",
    "east_north_up", "east_north_down", "east_south_up", "east_south_down",
    "west_north_up", "west_north_down", "west_south_up", "west_south_down",
)


def _unit(v):
    length = math.sqrt(sum(c * c for c in v))
    return tuple(c / length for c in v)


def _angle_deg(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


class CanonicalDirectionTests(unittest.TestCase):

    def test_exact_counts(self):
        self.assertEqual(len(canonical_directions(6)), 6)
        self.assertEqual(len(canonical_directions(14)), 14)
        self.assertEqual(len(canonical_directions(26)), 26)

    def test_six_axis_ids_exact_order(self):
        self.assertEqual(tuple(i for i, _ in canonical_directions(6)), _AXIS_IDS)

    def test_14_leads_with_axes_then_8_corners(self):
        dirs = canonical_directions(14)
        self.assertEqual(tuple(i for i, _ in dirs[:6]), _AXIS_IDS)
        self.assertEqual(len(dirs), 14)
        corners = dirs[6:]
        # every corner has no zero component
        for _, d in corners:
            self.assertNotIn(0, d)
        # 8 unique corners
        self.assertEqual(len({tuple(d) for _, d in corners}), 8)

    def test_26_axes_edges_corners_layout(self):
        dirs = canonical_directions(26)
        self.assertEqual(len(dirs), 26)
        self.assertEqual(tuple(i for i, _ in dirs[:6]), _AXIS_IDS)
        edges = dirs[6:18]
        self.assertEqual(len(edges), 12)
        for _, d in edges:
            nonzero = sum(1 for c in d if c != 0)
            self.assertEqual(nonzero, 2)
        corners = dirs[18:]
        self.assertEqual(len(corners), 8)
        for _, d in corners:
            self.assertNotIn(0, d)

    def test_no_duplicate_directions_in_each_set(self):
        for kind in (6, 14, 26):
            vectors = [tuple(d) for _, d in canonical_directions(kind)]
            self.assertEqual(len(vectors), len(set(vectors)), msg=f"kind={kind}")

    def test_sets_deterministic_across_calls(self):
        for kind in (6, 14, 26):
            self.assertEqual(canonical_directions(kind), canonical_directions(kind))

    def test_14_explicit_order_locked(self):
        self.assertEqual(
            tuple(i for i, _ in canonical_directions(14)),
            _EXPECTED_IDS_14,
        )

    def test_26_explicit_order_locked(self):
        self.assertEqual(
            tuple(i for i, _ in canonical_directions(26)),
            _EXPECTED_IDS_26,
        )

    def test_semantic_direction_vectors(self):
        table = {
            "east": (1, 0, 0),
            "west": (-1, 0, 0),
            "north": (0, 1, 0),
            "south": (0, -1, 0),
            "up": (0, 0, 1),
            "down": (0, 0, -1),
        }
        for vid, vec in DIRECTIONS_AXIS:
            self.assertEqual(tuple(int(c) for c in vec), table[vid])

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            canonical_directions(7)


class CanonicalPoseTests(unittest.TestCase):

    def test_all_canonical_poses_valid_under_camera_basis(self):
        for kind in (6, 14, 26):
            for cam in canonical_views(kind, target=_TARGET, radius=_RADIUS):
                right, up, forward = camera_basis(cam)  # must not raise
                self.assertAlmostEqual(round(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2], 9), 1.0)

    def test_axis_position_placement(self):
        table = {
            "east": (1, 0, 0),
            "west": (-1, 0, 0),
            "north": (0, 1, 0),
            "south": (0, -1, 0),
            "up": (0, 0, 1),
            "down": (0, 0, -1),
        }
        for vid, dvec in DIRECTIONS_AXIS:
            cam = canonical_camera(_TARGET, _RADIUS, dvec)
            expect_pos = (
                _TARGET[0] + table[vid][0] * _RADIUS,
                _TARGET[1] + table[vid][1] * _RADIUS,
                _TARGET[2] + table[vid][2] * _RADIUS,
            )
            for got, exp in zip(cam.position, expect_pos):
                self.assertAlmostEqual(got, exp)
            self.assertEqual(cam.look_at, _TARGET)

    def test_up_and_down_do_not_hit_parallel_up_failure(self):
        for vid in ("up", "down"):
            cam = canonical_camera(_TARGET, _RADIUS, next(d for i, d in DIRECTIONS_AXIS if i == vid))
            camera_basis(cam)  # must not raise
        self.assertEqual(
            canonical_camera(_TARGET, _RADIUS, (0, 0, 1)).up,
            (0.0, 1.0, 0.0),
        )

    def test_near_vertical_pose_uses_stable_up_fallback(self):
        # Tiny tilt off exact vertical: within the near-parallel threshold,
        # so the fallback up=(0, 1, 0) is chosen and the basis stays stable.
        for direction in ((1e-5, 0.0, 1.0), (-1e-5, 0.0, -1.0)):
            cam = canonical_camera(_TARGET, _RADIUS, direction)
            self.assertEqual(cam.up, (0.0, 1.0, 0.0))
            right, up, forward = camera_basis(cam)  # must not raise
            for v in (right, up, forward):
                norm = math.sqrt(sum(c * c for c in v))
                self.assertAlmostEqual(norm, 1.0)

    def test_normal_horizontal_pose_keeps_world_up(self):
        east = canonical_camera(_TARGET, _RADIUS, (1.0, 0.0, 0.0))
        self.assertEqual(east.up, (0.0, 0.0, 1.0))
        camera_basis(east)  # must not raise

    def test_canonical_camera_accepts_unnormalised_direction(self):
        cam = canonical_camera(_TARGET, _RADIUS, (1, 1, 1))
        camera_basis(cam)

    def test_canonical_camera_rejects_nonpositive_radius(self):
        with self.assertRaises(ValueError):
            canonical_camera(_TARGET, 0.0, (1, 0, 0))
        with self.assertRaises(ValueError):
            canonical_camera(_TARGET, -1.0, (1, 0, 0))


class RegressionCompatibilityTests(unittest.TestCase):

    def test_old_view_ids_unchanged(self):
        self.assertEqual(VIEW_IDS, ("south", "east", "north", "west"))

    def test_old_cardinal_views_deterministic_and_unchanged(self):
        first = generate_cardinal_views()
        second = generate_cardinal_views()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        for cam in first:
            self.assertIsInstance(cam, CameraPose)


class JitterTests(unittest.TestCase):

    def _cam(self):
        return canonical_views(14, target=_TARGET, radius=_RADIUS)[0]

    def test_negative_angular_deg_rejected(self):
        with self.assertRaises(ValueError):
            jitter_pose(self._cam(), seed=0, angular_deg=-1.0)

    def test_negative_radius_frac_rejected(self):
        with self.assertRaises(ValueError):
            jitter_pose(self._cam(), seed=0, radius_frac=-0.1)

    def test_radius_frac_at_or_above_one_rejected(self):
        for bad in (1.0, 1.5):
            with self.assertRaises(ValueError):
                jitter_pose(self._cam(), seed=0, radius_frac=bad)

    def test_negative_target_radius_rejected(self):
        with self.assertRaises(ValueError):
            jitter_pose(self._cam(), seed=0, target_radius=-0.1)

    def test_angular_deviation_never_exceeds_angular_deg(self):
        cam = canonical_views(26, target=_TARGET, radius=_RADIUS)[0]
        original_forward = camera_basis(cam)[2]
        for seed in range(25):
            out = jitter_pose(cam, seed=seed, angular_deg=5.0)
            new_forward = camera_basis(out)[2]
            deviation = _angle_deg(original_forward, new_forward)
            self.assertLessEqual(deviation, 5.0 + 1e-6)

    def test_angular_deviation_bounded_with_all_perturbations(self):
        cam = self._cam()
        original_forward = camera_basis(cam)[2]
        for seed in range(25):
            out = jitter_pose(
                cam, seed=seed, angular_deg=3.0, radius_frac=0.1, target_radius=0.2
            )
            new_forward = camera_basis(out)[2]
            deviation = _angle_deg(original_forward, new_forward)
            self.assertLessEqual(deviation, 3.0 + 1e-6)

    def test_target_displacement_never_exceeds_target_radius(self):
        cam = self._cam()
        radius = 0.25
        for seed in range(40):
            out = jitter_pose(
                cam, seed=seed, angular_deg=1.0, radius_frac=0.05, target_radius=radius
            )
            disp = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(out.look_at, cam.look_at))
            )
            self.assertLessEqual(disp, radius + 1e-9)

    def test_zero_target_radius_preserves_target_exactly(self):
        cam = self._cam()
        for seed in (0, 7, 99):
            out = jitter_pose(
                cam, seed=seed, angular_deg=2.0, radius_frac=0.05, target_radius=0.0
            )
            self.assertEqual(out.look_at, cam.look_at)

    def test_zero_jitter_preserves_input_pose(self):
        for kind in (6, 26):
            for cam in canonical_views(kind, target=_TARGET, radius=_RADIUS):
                out = jitter_pose(cam, seed=123)
                self.assertEqual(out, cam)

    def test_seeded_jitter_reproducible(self):
        cam = canonical_views(14, target=_TARGET, radius=_RADIUS)[0]
        a = jitter_pose(cam, seed=7, angular_deg=5.0, radius_frac=0.1, target_radius=0.2)
        b = jitter_pose(cam, seed=7, angular_deg=5.0, radius_frac=0.1, target_radius=0.2)
        self.assertEqual(a, b)

    def test_zero_angular_preserves_viewing_direction(self):
        cam = self._cam()
        original_forward = camera_basis(cam)[2]
        for seed in (0, 11, 42):
            out = jitter_pose(cam, seed=seed, radius_frac=0.1, target_radius=0.2)
            new_forward = camera_basis(out)[2]
            self.assertAlmostEqual(_angle_deg(original_forward, new_forward), 0.0)

    def test_different_seeds_produce_different_poses(self):
        cam = canonical_views(6, target=_TARGET, radius=_RADIUS)[0]
        a = jitter_pose(cam, seed=1, angular_deg=8.0, radius_frac=0.1, target_radius=0.2)
        b = jitter_pose(cam, seed=2, angular_deg=8.0, radius_frac=0.1, target_radius=0.2)
        self.assertNotEqual(a, b)

    def test_jittered_poses_remain_valid(self):
        for kind in (6, 14, 26):
            for cam in canonical_views(kind, target=_TARGET, radius=_RADIUS):
                for seed in (0, 5, 99):
                    out = jitter_pose(cam, seed=seed, angular_deg=4.0, radius_frac=0.05, target_radius=0.1)
                    camera_basis(out)  # must not raise
                    self.assertIsInstance(out, CameraPose)


class ContinuousSamplingTests(unittest.TestCase):

    def test_reproducible_with_same_seed(self):
        a = sample_camera(target=_TARGET, radius=_RADIUS, seed=42)
        b = sample_camera(target=_TARGET, radius=_RADIUS, seed=42)
        self.assertEqual(a, b)

    def test_samples_not_constrained_to_canonical_lattice(self):
        canonical_units = [_unit(d) for _, d in DIRECTIONS_26]
        worst_min_angle = 0.0
        for seed in range(50):
            cam = sample_camera(target=_TARGET, radius=_RADIUS, seed=seed)
            offset = _unit(tuple(cam.position[i] - _TARGET[i] for i in range(3)))
            min_angle = min(_angle_deg(offset, unit) for unit in canonical_units)
            worst_min_angle = max(worst_min_angle, min_angle)
        # At least one deterministic sample lies at a nontrivial angular
        # distance (>= 2 degrees) from every canonical unit direction.
        self.assertGreaterEqual(worst_min_angle, 2.0)

    def test_continuous_samples_remain_valid(self):
        for seed in range(20):
            cam = sample_camera(target=_TARGET, radius=_RADIUS, seed=seed)
            camera_basis(cam)  # must not raise
            self.assertIsInstance(cam, CameraPose)

    def test_different_seeds_differ(self):
        samples = {sample_camera(target=_TARGET, radius=_RADIUS, seed=s) for s in range(50)}
        self.assertGreater(len(samples), 40)


class GlobalRngIsolationTests(unittest.TestCase):

    def _assert_global_rng_untouched(self, func):
        saved_state = random.getstate()
        try:
            random.seed(12345)
            before = random.getstate()
            func()
            after = random.getstate()
            self.assertEqual(before, after)
        finally:
            random.setstate(saved_state)

    def test_jitter_pose_does_not_mutate_global_rng(self):
        cam = canonical_views(14, target=_TARGET, radius=_RADIUS)[0]

        def call():
            jitter_pose(
                cam, seed=1, angular_deg=5.0, radius_frac=0.1, target_radius=0.2
            )

        self._assert_global_rng_untouched(call)

    def test_sample_camera_does_not_mutate_global_rng(self):
        def call():
            sample_camera(target=_TARGET, radius=_RADIUS, seed=1)

        self._assert_global_rng_untouched(call)


if __name__ == "__main__":
    unittest.main()
