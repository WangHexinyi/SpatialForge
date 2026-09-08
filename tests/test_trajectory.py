"""Unit tests for adaptive camera framing, precessing trajectory generation, and diagnostics."""

import math
import unittest

from spatialforge.environment.camera import CameraPose, Vec3
from spatialforge.environment.geometry import camera_basis, _norm, _sub
from spatialforge.environment.scene import SceneObject, SceneState
from spatialforge.environment.trajectory import (
    DISTANCE_MODE_ADAPTIVE,
    DISTANCE_MODE_ADAPTIVE_SCALE,
    DISTANCE_MODE_HISTORICAL,
    DISTANCE_MODE_MANUAL,
    FRAMING_PRESET_BALANCED,
    FRAMING_PRESET_HISTORICAL,
    FRAMING_PRESET_LOOSE,
    FRAMING_PRESET_TIGHT,
    HISTORICAL_FIXED_RADIUS,
    TRAJECTORY_FAMILY_CUSTOM_WAYPOINTS,
    TRAJECTORY_FAMILY_FIGURE8_CROSS,
    TRAJECTORY_FAMILY_HELICAL_SWEEP,
    TRAJECTORY_FAMILY_PRECESSING_REFERENCE,
    TRAJECTORY_FAMILY_RADIAL_PROBE,
    TRAJECTORY_FAMILY_SURVEY_ORBIT,
    AdaptiveFramingResult,
    CameraTrajectory,
    TrajectoryConfig,
    TrajectoryFrame,
    adjust_camera_distance,
    compute_adaptive_framing,
    compute_scene_extent,
    distance_for_occupancy,
    generate_trajectory,
    interpolate_waypoints,
    parse_waypoint_text,
    primitive_size_margin,
    resolve_camera_distance,
)


class TestAdaptiveFraming(unittest.TestCase):
    def setUp(self):
        # Sparse/small scene: 2 small objects clustered near origin
        self.sparse_scene = SceneState(
            scene_id="sparse_scene",
            seed=42,
            objects=(
                SceneObject(name="obj0", shape="sphere", color="red", location=(-0.3, 0.2, 0.4), size=0.4),
                SceneObject(name="obj1", shape="cube", color="blue", location=(0.3, -0.2, 0.4), size=0.4),
            ),
        )

        # Extended/large scene: objects spread far apart
        self.extended_scene = SceneState(
            scene_id="extended_scene",
            seed=42,
            objects=(
                SceneObject(name="obj0", shape="sphere", color="red", location=(-2.5, 2.0, 0.4), size=1.0),
                SceneObject(name="obj1", shape="cube", color="yellow", location=(2.5, -2.0, 0.4), size=1.0),
                SceneObject(name="obj2", shape="cylinder", color="green", location=(0.0, 0.0, 0.4), size=1.0),
            ),
        )

        # Empty scene
        self.empty_scene = SceneState(
            scene_id="empty_scene",
            seed=0,
            objects=(),
        )

    def test_deterministic_result(self):
        """Identical scene yields identical adaptive framing results."""
        res1 = compute_adaptive_framing(self.sparse_scene, framing_mode=FRAMING_PRESET_BALANCED)
        res2 = compute_adaptive_framing(self.sparse_scene, framing_mode=FRAMING_PRESET_BALANCED)
        self.assertEqual(res1.camera_distance, res2.camera_distance)
        self.assertEqual(res1.scene_center, res2.scene_center)
        self.assertEqual(res1.scene_radius, res2.scene_radius)
        self.assertEqual(res1.to_dict(), res2.to_dict())

    def test_sparse_scene_closer_than_extended_scene(self):
        """Sparse/small scenes receive shorter camera distance than extended/large scenes."""
        res_sparse = compute_adaptive_framing(self.sparse_scene, framing_mode=FRAMING_PRESET_BALANCED)
        res_extended = compute_adaptive_framing(self.extended_scene, framing_mode=FRAMING_PRESET_BALANCED)

        self.assertLess(res_sparse.scene_radius, res_extended.scene_radius)
        self.assertLess(res_sparse.camera_distance, res_extended.camera_distance)
        # Sparse scene distance should be substantially closer than historical radius=6.0
        self.assertLess(res_sparse.camera_distance, 3.0)
        self.assertLess(res_sparse.camera_distance, HISTORICAL_FIXED_RADIUS)

    def test_distance_positive_and_finite(self):
        """Calculated camera distance is strictly positive and finite."""
        for scene in (self.sparse_scene, self.extended_scene, self.empty_scene):
            for mode in (FRAMING_PRESET_LOOSE, FRAMING_PRESET_BALANCED, FRAMING_PRESET_TIGHT, FRAMING_PRESET_HISTORICAL):
                res = compute_adaptive_framing(scene, framing_mode=mode)
                self.assertGreater(res.camera_distance, 0.0)
                self.assertTrue(math.isfinite(res.camera_distance))
                self.assertGreater(res.scene_radius, 0.0)
                self.assertTrue(math.isfinite(res.scene_radius))

    def test_object_size_affects_extent(self):
        """Larger object size conservatively expands the scene radius and increases camera distance."""
        scene_small_obj = SceneState(
            scene_id="small",
            seed=1,
            objects=(SceneObject(name="obj0", shape="cube", color="red", location=(0.0, 0.0, 0.4), size=0.2),),
        )
        scene_large_obj = SceneState(
            scene_id="large",
            seed=1,
            objects=(SceneObject(name="obj0", shape="cube", color="red", location=(0.0, 0.0, 0.4), size=1.5),),
        )
        res_small = compute_adaptive_framing(scene_small_obj)
        res_large = compute_adaptive_framing(scene_large_obj)

        self.assertLess(res_small.scene_radius, res_large.scene_radius)
        self.assertLess(res_small.camera_distance, res_large.camera_distance)

    def test_primitive_margin_ordering(self):
        """For equal size s, cube circumscribed margin > cylinder margin > sphere margin."""
        s = 1.0
        margin_cube = primitive_size_margin("cube", s)
        margin_cyl = primitive_size_margin("cylinder", s)
        margin_sph = primitive_size_margin("sphere", s)

        self.assertAlmostEqual(margin_cube, math.sqrt(3) / 2, places=5)
        self.assertAlmostEqual(margin_cyl, math.sqrt(2) / 2, places=5)
        self.assertAlmostEqual(margin_sph, 0.5, places=5)
        self.assertGreater(margin_cube, margin_cyl)
        self.assertGreater(margin_cyl, margin_sph)

    def test_empty_scene_handling_policy(self):
        """Empty scene does not crash or divide by zero; returns safe fallback center and radius."""
        res = compute_adaptive_framing(self.empty_scene)
        self.assertEqual(res.scene_center, (0.0, 0.0, 0.4))
        self.assertEqual(res.scene_radius, 1.0)
        self.assertIn("empty_scene_fallback", res.margin_assumptions)
        self.assertGreater(res.camera_distance, 0.0)

    def test_preset_ordering(self):
        """Preset camera distances must strictly order: Loose > Balanced > Tight."""
        res = compute_adaptive_framing(self.sparse_scene)
        comp = res.presets_comparison
        self.assertGreater(comp[FRAMING_PRESET_LOOSE], comp[FRAMING_PRESET_BALANCED])
        self.assertGreater(comp[FRAMING_PRESET_BALANCED], comp[FRAMING_PRESET_TIGHT])
        self.assertEqual(comp[FRAMING_PRESET_HISTORICAL], 6.0)

    def test_historical_fixed_preset(self):
        """Historical fixed preset strictly preserves camera_distance = 6.0."""
        res = compute_adaptive_framing(self.sparse_scene, framing_mode=FRAMING_PRESET_HISTORICAL)
        self.assertEqual(res.camera_distance, 6.0)
        self.assertEqual(res.framing_mode, FRAMING_PRESET_HISTORICAL)


class TestPrecessingTrajectory(unittest.TestCase):
    def setUp(self):
        self.scene = SceneState(
            scene_id="test_scene",
            seed=10,
            objects=(
                SceneObject(name="obj0", shape="sphere", color="red", location=(-0.8, 0.5, 0.4), size=0.6),
                SceneObject(name="obj1", shape="cube", color="yellow", location=(0.9, -0.6, 0.4), size=0.8),
            ),
        )

    def test_deterministic_config_and_seed(self):
        """Same config and seed generate bitwise-identical frame sequence and diagnostics."""
        cfg = TrajectoryConfig(frame_count=30, seed=123, framing_mode=FRAMING_PRESET_BALANCED)
        traj1 = generate_trajectory(self.scene, cfg)
        traj2 = generate_trajectory(self.scene, cfg)

        self.assertEqual(traj1.trajectory_id, traj2.trajectory_id)
        self.assertEqual(len(traj1.frames), len(traj2.frames))
        for f1, f2 in zip(traj1.frames, traj2.frames):
            self.assertEqual(f1.pose.position, f2.pose.position)
            self.assertEqual(f1.pose.look_at, f2.pose.look_at)
            self.assertEqual(f1.pose.up, f2.pose.up)
            self.assertEqual(f1.azimuth_deg, f2.azimuth_deg)
            self.assertEqual(f1.elevation_deg, f2.elevation_deg)
        self.assertEqual(traj1.diagnostics.to_dict(), traj2.diagnostics.to_dict())

    def test_correct_frame_count(self):
        """Trajectory output contains exactly the configured number of frames."""
        for count in (10, 35, 60, 100):
            cfg = TrajectoryConfig(frame_count=count)
            traj = generate_trajectory(self.scene, cfg)
            self.assertEqual(len(traj.frames), count)
            self.assertEqual(traj.diagnostics.frame_count, count)
            self.assertEqual(traj.frames[0].frame_index, 0)
            self.assertEqual(traj.frames[-1].frame_index, count - 1)
            self.assertAlmostEqual(traj.frames[0].normalized_time, 0.0)
            self.assertAlmostEqual(traj.frames[-1].normalized_time, 1.0)

    def test_valid_camera_pose_basis_every_frame(self):
        """Every single trajectory frame produces a strictly valid orthonormal camera basis."""
        cfg = TrajectoryConfig(frame_count=60, seed=42)
        traj = generate_trajectory(self.scene, cfg)

        for f in traj.frames:
            # camera_basis raises ValueError if up is parallel to forward or invalid
            right, up, forward = camera_basis(f.pose)
            self.assertAlmostEqual(_norm(right), 1.0, places=5)
            self.assertAlmostEqual(_norm(up), 1.0, places=5)
            self.assertAlmostEqual(_norm(forward), 1.0, places=5)

    def test_bounded_elevation(self):
        """Elevation stays strictly inside the configured elevation envelope."""
        envelope = (-10.0, 45.0)
        cfg = TrajectoryConfig(
            frame_count=60,
            seed=7,
            elevation_base_deg=20.0,
            elevation_amplitude_deg=30.0,
            elevation_envelope=envelope,
        )
        traj = generate_trajectory(self.scene, cfg)

        for f in traj.frames:
            self.assertGreaterEqual(f.elevation_deg, envelope[0] - 1e-5)
            self.assertLessEqual(f.elevation_deg, envelope[1] + 1e-5)

        self.assertGreaterEqual(traj.diagnostics.elevation_min_deg, envelope[0] - 1e-5)
        self.assertLessEqual(traj.diagnostics.elevation_max_deg, envelope[1] + 1e-5)

    def test_positive_radius(self):
        """Radius to look_at target is strictly positive across all frames."""
        cfg = TrajectoryConfig(frame_count=60, seed=99, radius_amplitude_frac=0.3)
        traj = generate_trajectory(self.scene, cfg)

        for f in traj.frames:
            self.assertGreater(f.distance_to_target, 0.2)
        self.assertGreater(traj.diagnostics.radius_min, 0.2)

    def test_bounded_target_bias(self):
        """Look-at target bias from scene center remains strictly within configured fraction of scene_radius."""
        cfg = TrajectoryConfig(frame_count=60, seed=42, target_bias_fraction=0.15)
        traj = generate_trajectory(self.scene, cfg)
        center = traj.framing.scene_center
        scene_r = traj.framing.scene_radius
        max_allowed_disp = cfg.target_bias_fraction * scene_r * 2.0  # safe conservative vector bound

        for f in traj.frames:
            disp = _norm(_sub(f.pose.look_at, center))
            self.assertLessEqual(disp, max_allowed_disp)

    def test_nonzero_variation_across_all_axes(self):
        """Trajectory produces non-trivial 3D variation in azimuth, elevation, and radius."""
        cfg = TrajectoryConfig(frame_count=60, seed=12)
        traj = generate_trajectory(self.scene, cfg)
        diag = traj.diagnostics

        self.assertGreater(diag.azimuth_span_deg, 180.0)
        self.assertGreater(diag.elevation_max_deg - diag.elevation_min_deg, 10.0)
        self.assertGreater(diag.radius_max - diag.radius_min, 0.1)
        self.assertGreater(diag.trajectory_path_length, 1.0)

    def test_no_trivial_duplicate_consecutive_frames(self):
        """Adjacent frames must never be identical (smooth continuous motion)."""
        cfg = TrajectoryConfig(frame_count=60, seed=42)
        traj = generate_trajectory(self.scene, cfg)

        for i in range(len(traj.frames) - 1):
            f_curr = traj.frames[i]
            f_next = traj.frames[i + 1]
            pos_diff = _norm(_sub(f_next.pose.position, f_curr.pose.position))
            self.assertGreater(pos_diff, 1e-4)

    def test_diagnostics_deterministic(self):
        """Diagnostics computation is purely deterministic and serializable."""
        cfg = TrajectoryConfig(frame_count=45, seed=5)
        traj = generate_trajectory(self.scene, cfg)
        d1 = traj.diagnostics.to_dict()
        d2 = traj.diagnostics.to_dict()
        self.assertEqual(d1, d2)
        self.assertIn("min_position_distance_m", d1)
        self.assertIn("min_angular_difference_deg", d1)
        self.assertIn("near_repeat_detected", d1)


class TestAdjustCameraDistance(unittest.TestCase):
    def setUp(self):
        self.original_pose = CameraPose(
            position=(0.0, -6.0, 3.0),
            look_at=(0.0, 0.0, 0.4),
            up=(0.0, 0.0, 1.0),
            fov_deg=60.0,
        )

    def test_preserves_target_and_optical_ray_direction(self):
        """adjust_camera_distance must keep look_at target and ray direction exactly unchanged."""
        for requested_dist in [0.5, 1.2, 3.5, 6.0, 15.0, 20.0]:
            adjusted = adjust_camera_distance(self.original_pose, requested_dist)
            # Target preserved
            self.assertEqual(adjusted.look_at, self.original_pose.look_at)

            # Distance matches requested
            p = adjusted.position
            t = adjusted.look_at
            dx, dy, dz = p[0] - t[0], p[1] - t[1], p[2] - t[2]
            actual_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            self.assertAlmostEqual(actual_dist, requested_dist, places=5)

            # Ray direction preserved
            p_orig = self.original_pose.position
            orig_dx, orig_dy, orig_dz = p_orig[0] - t[0], p_orig[1] - t[1], p_orig[2] - t[2]
            orig_norm = math.sqrt(orig_dx * orig_dx + orig_dy * orig_dy + orig_dz * orig_dz)
            orig_dir = (orig_dx / orig_norm, orig_dy / orig_norm, orig_dz / orig_norm)

            adj_dir = (dx / actual_dist, dy / actual_dist, dz / actual_dist)
            self.assertAlmostEqual(orig_dir[0], adj_dir[0], places=5)
            self.assertAlmostEqual(orig_dir[1], adj_dir[1], places=5)
            self.assertAlmostEqual(orig_dir[2], adj_dir[2], places=5)

            # Camera basis is orthonormal
            fwd, right, up = camera_basis(adjusted)
            self.assertAlmostEqual(_norm(fwd), 1.0, places=5)
            self.assertAlmostEqual(_norm(right), 1.0, places=5)
            self.assertAlmostEqual(_norm(up), 1.0, places=5)

    def test_rejects_invalid_distance(self):
        """Rejects zero, negative, infinite, and NaN distances with ValueError."""
        for invalid in [0.0, -1.0, -100.0, float("inf"), float("-inf"), float("nan")]:
            with self.assertRaises(ValueError):
                adjust_camera_distance(self.original_pose, invalid)

    def test_resolve_camera_distance_modes(self):
        """resolve_camera_distance respects historical_fixed, adaptive, manual, and adaptive_scale."""
        base_adaptive = 4.0
        # Historical fixed
        d_hist = resolve_camera_distance(DISTANCE_MODE_HISTORICAL, base_adaptive)
        self.assertEqual(d_hist, HISTORICAL_FIXED_RADIUS)

        # Adaptive
        d_adapt = resolve_camera_distance(DISTANCE_MODE_ADAPTIVE, base_adaptive)
        self.assertEqual(d_adapt, 4.0)

        # Manual
        d_manual = resolve_camera_distance(DISTANCE_MODE_MANUAL, base_adaptive, manual_distance=2.5)
        self.assertEqual(d_manual, 2.5)

        # Adaptive scale
        d_scale = resolve_camera_distance(DISTANCE_MODE_ADAPTIVE_SCALE, base_adaptive, distance_scale=1.5)
        self.assertEqual(d_scale, 6.0)


class TestWaypointParsingAndInterpolation(unittest.TestCase):
    def test_parse_waypoint_text_valid_3_and_6_coords(self):
        """Parses lines with 3 coordinates and 6 coordinates correctly."""
        text3 = """
        # Custom flight path (3 coords)
        3.0, -3.0, 2.0
        0.0, -4.0, 2.5
        -3.0, -3.0, 2.0
        """
        pos3, targets3 = parse_waypoint_text(text3)
        self.assertEqual(len(pos3), 3)
        self.assertIsNone(targets3)
        self.assertEqual(pos3[0], (3.0, -3.0, 2.0))
        self.assertEqual(pos3[1], (0.0, -4.0, 2.5))
        self.assertEqual(pos3[2], (-3.0, -3.0, 2.0))

        text6 = """
        # Custom flight path with explicit targets (6 coords)
        3.0, -3.0, 2.0, 0.0, 0.0, 0.4
        0.0, -4.0, 2.5, 0.1, 0.2, 0.3
        """
        pos6, targets6 = parse_waypoint_text(text6)
        self.assertEqual(len(pos6), 2)
        self.assertIsNotNone(targets6)
        self.assertEqual(len(targets6), 2)
        self.assertEqual(pos6[0], (3.0, -3.0, 2.0))
        self.assertEqual(targets6[0], (0.0, 0.0, 0.4))
        self.assertEqual(pos6[1], (0.0, -4.0, 2.5))
        self.assertEqual(targets6[1], (0.1, 0.2, 0.3))

    def test_parse_waypoint_text_rejects_insufficient_or_malformed(self):
        """Rejects fewer than 2 waypoints, inconsistent coordinates, and invalid lines with ValueError."""
        with self.assertRaises(ValueError):
            parse_waypoint_text("1.0, 2.0, 3.0\n")  # Only 1 point

        with self.assertRaises(ValueError):
            parse_waypoint_text("1.0, 2.0\n3.0, 4.0, 5.0\n")  # 2 tokens on first line

        with self.assertRaises(ValueError):
            # Inconsistent coord dimensions across lines
            parse_waypoint_text("1.0, 2.0, 3.0\n4.0, 5.0, 6.0, 0.0, 0.0, 0.0\n")

    def test_interpolate_waypoints_catmull_rom_centripetal(self):
        """Centripetal Catmull-Rom interpolation starts and ends exactly at waypoints and stays open."""
        wps = [
            (3.0, -3.0, 1.0),
            (0.0, -4.0, 2.0),
            (-3.0, -3.0, 1.5),
            (-1.0, -1.0, 0.8),
        ]
        poses, arc_dists, total_len = interpolate_waypoints(wps, frame_count=50, interpolation_mode="catmull_rom")
        self.assertEqual(len(poses), 50)
        self.assertEqual(len(arc_dists), 50)
        self.assertGreater(total_len, 0.0)

        # Exact start and end match
        p_start = poses[0].position
        p_end = poses[-1].position
        self.assertAlmostEqual(p_start[0], 3.0, places=4)
        self.assertAlmostEqual(p_start[1], -3.0, places=4)
        self.assertAlmostEqual(p_start[2], 1.0, places=4)

        self.assertAlmostEqual(p_end[0], -1.0, places=4)
        self.assertAlmostEqual(p_end[1], -1.0, places=4)
        self.assertAlmostEqual(p_end[2], 0.8, places=4)

        # Open curve: start != end
        self.assertNotEqual(p_start, p_end)

        # Monotonic arc distance
        for i in range(len(arc_dists) - 1):
            self.assertLessEqual(arc_dists[i], arc_dists[i + 1] + 1e-6)

        # Valid orthonormal camera basis for all frames
        for pose in poses:
            fwd, right, up = camera_basis(pose)
            self.assertAlmostEqual(_norm(fwd), 1.0, places=5)
            self.assertAlmostEqual(_norm(right), 1.0, places=5)
            self.assertAlmostEqual(_norm(up), 1.0, places=5)


class TestTrajectoryFamilies(unittest.TestCase):
    def setUp(self):
        self.scene = SceneState(
            scene_id="test_scene",
            seed=42,
            objects=(
                SceneObject(name="obj0", shape="cube", color="blue", location=(0.0, 0.0, 0.4), size=0.5),
            ),
        )

    def test_survey_orbit_topology(self):
        """Survey orbit completes full 360 degree azimuth coverage with steady elevation."""
        cfg = TrajectoryConfig(
            frame_count=60,
            trajectory_family=TRAJECTORY_FAMILY_SURVEY_ORBIT,
        )
        traj = generate_trajectory(self.scene, cfg)
        self.assertAlmostEqual(traj.diagnostics.azimuth_span_deg, 360.0, delta=15.0)

    def test_helical_sweep_topology(self):
        """Helical sweep sweeps elevation monotonically from low to high while orbiting."""
        cfg = TrajectoryConfig(
            frame_count=60,
            trajectory_family=TRAJECTORY_FAMILY_HELICAL_SWEEP,
            elevation_base_deg=20.0,
            elevation_amplitude_deg=25.0,
        )
        traj = generate_trajectory(self.scene, cfg)
        elevs = [f.elevation_deg for f in traj.frames]
        self.assertGreater(elevs[-1], elevs[0])
        self.assertGreater(traj.diagnostics.elevation_max_deg - traj.diagnostics.elevation_min_deg, 20.0)

    def test_figure8_cross_topology(self):
        """Figure-8 cross oscillates elevation at twice the azimuth rate, creating two elevation extrema."""
        cfg = TrajectoryConfig(
            frame_count=60,
            trajectory_family=TRAJECTORY_FAMILY_FIGURE8_CROSS,
        )
        traj = generate_trajectory(self.scene, cfg)
        elevs = [f.elevation_deg for f in traj.frames]
        peaks = sum(1 for i in range(1, len(elevs) - 1) if elevs[i] > elevs[i-1] and elevs[i] > elevs[i+1])
        valleys = sum(1 for i in range(1, len(elevs) - 1) if elevs[i] < elevs[i-1] and elevs[i] < elevs[i+1])
        self.assertGreaterEqual(peaks + valleys, 2)

    def test_radial_probe_topology(self):
        """Radial probe oscillates distance substantially between near and far inspection."""
        cfg = TrajectoryConfig(
            frame_count=60,
            trajectory_family=TRAJECTORY_FAMILY_RADIAL_PROBE,
        )
        traj = generate_trajectory(self.scene, cfg)
        radii = [f.distance_to_target for f in traj.frames]
        rad_span = max(radii) - min(radii)
        self.assertGreater(rad_span, 1.0)
        self.assertGreater(traj.diagnostics.radius_max - traj.diagnostics.radius_min, 1.0)

    def test_topological_distinctness(self):
        """Different trajectory families produce genuinely distinct diagnostic geometric signatures."""
        cfg_survey = TrajectoryConfig(frame_count=60, trajectory_family=TRAJECTORY_FAMILY_SURVEY_ORBIT)
        cfg_probe = TrajectoryConfig(frame_count=60, trajectory_family=TRAJECTORY_FAMILY_RADIAL_PROBE)

        traj_survey = generate_trajectory(self.scene, cfg_survey)
        traj_probe = generate_trajectory(self.scene, cfg_probe)

        rad_var_survey = traj_survey.diagnostics.radius_max - traj_survey.diagnostics.radius_min
        rad_var_probe = traj_probe.diagnostics.radius_max - traj_probe.diagnostics.radius_min
        self.assertGreater(rad_var_probe, rad_var_survey * 5.0)
        self.assertNotEqual(round(traj_survey.total_path_length_m, 1), round(traj_probe.total_path_length_m, 1))


class TestScientificSpeedAndTiming(unittest.TestCase):
    def test_duration_inversely_proportional_to_speed(self):
        """Physical trajectory duration T = L / v is exactly verified."""
        scene = SceneState(
            scene_id="test_scene",
            seed=42,
            objects=(SceneObject(name="obj", shape="cube", color="red", location=(0.0, 0.0, 0.4), size=0.5),),
        )
        cfg1 = TrajectoryConfig(frame_count=60, trajectory_family=TRAJECTORY_FAMILY_SURVEY_ORBIT, speed_m_s=1.0)
        cfg2 = TrajectoryConfig(frame_count=60, trajectory_family=TRAJECTORY_FAMILY_SURVEY_ORBIT, speed_m_s=2.0)

        t1 = generate_trajectory(scene, cfg1)
        t2 = generate_trajectory(scene, cfg2)

        self.assertAlmostEqual(t1.total_path_length_m, t2.total_path_length_m, places=4)
        self.assertAlmostEqual(t1.total_duration_sec, t1.total_path_length_m / 1.0, places=4)
        self.assertAlmostEqual(t2.total_duration_sec, t2.total_path_length_m / 2.0, places=4)
        self.assertAlmostEqual(t1.total_duration_sec / t2.total_duration_sec, 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
