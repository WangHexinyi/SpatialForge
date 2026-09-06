"""CPU-only tests for the G2.0-C view-conditioned spatial truth engine (no bpy)."""

import json
import unittest

from spatialforge.environment.camera import CameraPose
from spatialforge.environment.geometry import camera_basis
from spatialforge.environment.scene import SceneObject, SceneState
from spatialforge.environment.sampling import canonical_views, jitter_pose, sample_camera
from spatialforge.environment.relation import (
    REL_ABOVE,
    REL_ALIGNED,
    REL_BEHIND,
    REL_BELOW,
    REL_EQUIDISTANT,
    REL_FARTHER,
    REL_FRONT,
    REL_LEFT,
    REL_NEARER,
    REL_RIGHT,
    REL_SAME_DEPTH,
    ObjectTruth,
    PairTruth,
    SpatialTruthRecord,
    compute_spatial_truth,
)

_TARGET = (0.0, 0.0, 0.4)
_RADIUS = 6.0


def _obj(name, location, size=1.0, shape="cube", color="gray") -> SceneObject:
    return SceneObject(name=name, shape=shape, color=color, location=location, size=size)


def _scene(objects, scene_id="scene_test", seed=0) -> SceneState:
    return SceneState(scene_id=scene_id, seed=seed, objects=tuple(objects))


def _pair(record, index_a, index_b) -> PairTruth:
    for pair in record.pairs:
        if pair.object_index_a == index_a and pair.object_index_b == index_b:
            return pair
    raise AssertionError(f"pair ({index_a}, {index_b}) not found")


class RelationSemanticsTests(unittest.TestCase):
    """A. Exact expected relation labels, including all neutral/tie states.

    Camera used below looks along world +Y from the origin with world +Z up:
        position=(0,0,0), look_at=(0,1,0)
    Its camera frame is (right_x, up_y, depth_z) = (world_x, world_z, world_y).
    """

    def setUp(self):
        self.camera = CameraPose(position=(0.0, 0.0, 0.0), look_at=(0.0, 1.0, 0.0))
        self.state = _scene(
            [
                _obj("left_far", (-2.0, 3.0, 0.0)),    # 0
                _obj("right_far", (2.0, 3.0, 0.0)),    # 1
                _obj("above", (0.0, 1.0, 2.0)),        # 2
                _obj("below", (0.0, 1.0, -2.0)),       # 3
            ]
        )

    def test_left_right_labels(self):
        record = compute_spatial_truth(self.state, self.camera)
        self.assertEqual(_pair(record, 0, 1).left_right, REL_LEFT)    # x -2 vs +2
        self.assertEqual(_pair(record, 0, 2).left_right, REL_LEFT)    # x -2 vs 0
        self.assertEqual(_pair(record, 1, 2).left_right, REL_RIGHT)   # x +2 vs 0
        self.assertEqual(_pair(record, 2, 3).left_right, REL_ALIGNED)  # x 0 vs 0

    def test_above_below_labels(self):
        record = compute_spatial_truth(self.state, self.camera)
        self.assertEqual(_pair(record, 2, 3).above_below, REL_ABOVE)   # y +2 vs -2
        self.assertEqual(_pair(record, 0, 2).above_below, REL_BELOW)   # y 0 vs +2
        self.assertEqual(_pair(record, 0, 1).above_below, REL_ALIGNED)  # y 0 vs 0

    def test_front_behind_labels(self):
        record = compute_spatial_truth(self.state, self.camera)
        self.assertEqual(_pair(record, 0, 1).front_behind, REL_SAME_DEPTH)  # z 3 vs 3
        self.assertEqual(_pair(record, 0, 2).front_behind, REL_BEHIND)      # z 3 vs 1
        self.assertEqual(_pair(record, 0, 3).front_behind, REL_BEHIND)      # z 3 vs 1
        self.assertEqual(_pair(record, 2, 3).front_behind, REL_SAME_DEPTH)  # z 1 vs 1

    def test_nearer_farther_labels(self):
        record = compute_spatial_truth(self.state, self.camera)
        self.assertEqual(_pair(record, 0, 1).near_far, REL_EQUIDISTANT)  # both sqrt(13)
        self.assertEqual(_pair(record, 0, 2).near_far, REL_FARTHER)      # sqrt(13) vs sqrt(5)
        self.assertEqual(_pair(record, 1, 3).near_far, REL_FARTHER)
        self.assertEqual(_pair(record, 2, 3).near_far, REL_EQUIDISTANT)  # both sqrt(5)

    def test_negative_eps_rejected(self):
        with self.assertRaises(ValueError):
            compute_spatial_truth(self.state, self.camera, eps=-1.0)

    def test_non_finite_and_negative_eps_rejected(self):
        for bad_eps in (-1.0, float("-inf"), float("inf"), float("nan")):
            with self.subTest(eps=bad_eps):
                with self.assertRaises(ValueError):
                    compute_spatial_truth(self.state, self.camera, eps=bad_eps)

    def test_zero_eps_accepted_and_exact_ties_preserved(self):
        record = compute_spatial_truth(self.state, self.camera, eps=0.0)
        # Objects 0 and 1 share camera up_y (0.0) and depth (3.0) exactly, so
        # their neutral states persist even with eps == 0.
        self.assertEqual(_pair(record, 0, 1).above_below, REL_ALIGNED)
        self.assertEqual(_pair(record, 0, 1).front_behind, REL_SAME_DEPTH)


class EpsilonBandTests(unittest.TestCase):
    """Epsilon tolerance semantics: differences inside [-eps, +eps] stay neutral,
    differences strictly beyond eps become directional/non-neutral labels.

    Camera looks along world +Y from the origin with world +Z up, so its camera
    frame is (right_x, up_y, depth_z) = (world_x, world_z, world_y). Expected
    values are derived by hand from these coordinates, never from production
    helpers.
    """

    EPS = 0.1

    def setUp(self):
        self.camera = CameraPose(position=(0.0, 0.0, 0.0), look_at=(0.0, 1.0, 0.0))

    def _truth(self, objects):
        return compute_spatial_truth(_scene(objects), self.camera, eps=self.EPS)

    def test_epsilon_band_left_right(self):
        # Frame x = world x. Pair (0,1): dx = -0.05 inside band.
        # Pair (0,2): dx = -0.1 exactly at band edge -> still ALIGNED.
        # Pair (0,3): dx = -0.21 strictly beyond band -> REL_LEFT.
        record = self._truth(
            [
                _obj("a", (0.0, 5.0, 0.0)),
                _obj("b", (0.05, 5.0, 0.0)),
                _obj("c", (0.1, 5.0, 0.0)),
                _obj("d", (0.21, 5.0, 0.0)),
            ]
        )
        self.assertEqual(_pair(record, 0, 1).left_right, REL_ALIGNED)
        self.assertEqual(_pair(record, 0, 2).left_right, REL_ALIGNED)
        self.assertEqual(_pair(record, 0, 3).left_right, REL_LEFT)

    def test_epsilon_band_above_below(self):
        # Frame y = world z. Pair (0,1): dy = -0.05 inside band.
        # Pair (0,2): dy = -0.1 at band edge -> still ALIGNED.
        # Pair (0,3): dy = +0.21 strictly beyond band -> REL_ABOVE.
        record = self._truth(
            [
                _obj("a", (0.0, 5.0, 0.0)),
                _obj("b", (0.0, 5.0, 0.05)),
                _obj("c", (0.0, 5.0, 0.1)),
                _obj("d", (0.0, 5.0, -0.21)),
            ]
        )
        self.assertEqual(_pair(record, 0, 1).above_below, REL_ALIGNED)
        self.assertEqual(_pair(record, 0, 2).above_below, REL_ALIGNED)
        self.assertEqual(_pair(record, 0, 3).above_below, REL_ABOVE)

    def test_epsilon_band_front_behind(self):
        # Frame z (depth) = world y. Pair (0,1): dz = -0.05 inside band.
        # Pair (0,2): dz = -0.1 at band edge -> still SAME_DEPTH.
        # Pair (0,3): dz = -0.21 strictly beyond band -> REL_FRONT.
        record = self._truth(
            [
                _obj("a", (0.0, 5.0, 0.0)),
                _obj("b", (0.0, 5.05, 0.0)),
                _obj("c", (0.0, 5.1, 0.0)),
                _obj("d", (0.0, 5.21, 0.0)),
            ]
        )
        self.assertEqual(_pair(record, 0, 1).front_behind, REL_SAME_DEPTH)
        self.assertEqual(_pair(record, 0, 2).front_behind, REL_SAME_DEPTH)
        self.assertEqual(_pair(record, 0, 3).front_behind, REL_FRONT)

    def test_epsilon_band_near_far(self):
        # Metric distance is Euclidean distance from the camera at the origin.
        # Pair (0,1): |1.0 - 1.05| = 0.05 inside band.
        # Pair (0,2): |1.0 - 1.09| = 0.09 inside band.
        # Pair (0,3): |1.0 - 1.25| = 0.25 strictly beyond band -> REL_NEARER.
        record = self._truth(
            [
                _obj("a", (1.0, 0.0, 0.0)),
                _obj("b", (1.05, 0.0, 0.0)),
                _obj("c", (1.09, 0.0, 0.0)),
                _obj("d", (1.25, 0.0, 0.0)),
            ]
        )
        self.assertEqual(_pair(record, 0, 1).near_far, REL_EQUIDISTANT)
        self.assertEqual(_pair(record, 0, 2).near_far, REL_EQUIDISTANT)
        self.assertEqual(_pair(record, 0, 3).near_far, REL_NEARER)


class OppositeViewCounterfactualTests(unittest.TestCase):
    """B. Same SceneState under opposite cameras: relations flip, world facts persist."""

    def setUp(self):
        self.state = _scene(
            [
                _obj("west_obj", (-1.0, 0.0, 0.0), size=0.5, shape="sphere", color="red"),    # 0
                _obj("east_obj", (1.0, 0.0, 0.0), size=2.0, shape="cube", color="blue"),      # 1
                _obj("near_obj", (0.0, -1.0, 0.0), size=1.0, shape="cylinder", color="green"),  # 2
                _obj("far_obj", (0.0, 1.0, 0.0), size=1.5, shape="sphere", color="yellow"),   # 3
            ]
        )
        self.cam_south = CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0))
        self.cam_north = CameraPose(position=(0.0, 6.0, 0.0), look_at=(0.0, 0.0, 0.0))

    def test_left_right_flips_across_opposite_cameras(self):
        south = compute_spatial_truth(self.state, self.cam_south)
        north = compute_spatial_truth(self.state, self.cam_north)
        self.assertEqual(_pair(south, 0, 1).left_right, REL_LEFT)
        self.assertEqual(_pair(north, 0, 1).left_right, REL_RIGHT)

    def test_front_behind_flips_across_opposite_cameras(self):
        south = compute_spatial_truth(self.state, self.cam_south)
        north = compute_spatial_truth(self.state, self.cam_north)
        self.assertEqual(_pair(south, 2, 3).front_behind, REL_FRONT)
        self.assertEqual(_pair(north, 2, 3).front_behind, REL_BEHIND)

    def test_world_object_facts_unchanged(self):
        south = compute_spatial_truth(self.state, self.cam_south)
        north = compute_spatial_truth(self.state, self.cam_north)
        for s_obj, n_obj in zip(south.objects, north.objects):
            self.assertEqual(s_obj.object_index, n_obj.object_index)
            self.assertEqual(s_obj.name, n_obj.name)
            self.assertEqual(s_obj.shape, n_obj.shape)
            self.assertEqual(s_obj.color, n_obj.color)
            self.assertEqual(s_obj.size, n_obj.size)
            self.assertEqual(s_obj.world_location, n_obj.world_location)

    def test_object_to_object_metric_distance_unchanged(self):
        south = compute_spatial_truth(self.state, self.cam_south)
        north = compute_spatial_truth(self.state, self.cam_north)
        for s_pair, n_pair in zip(south.pairs, north.pairs):
            self.assertAlmostEqual(s_pair.metric_distance, n_pair.metric_distance)
        self.assertAlmostEqual(_pair(south, 0, 1).metric_distance, 2.0)
        self.assertAlmostEqual(_pair(south, 2, 3).metric_distance, 2.0)
        self.assertAlmostEqual(_pair(south, 0, 2).metric_distance, (2.0) ** 0.5)


class RotationOnlyControlTests(unittest.TestCase):
    """C. Fixed camera.position, different orientation.

    Objects A=(-1,0,0), B=(1,0,0). Camera at origin.
    Looking +Y: camera frame x = world x -> A LEFT of B, same depth.
    Looking +X: camera frame x = -world y = 0 for both -> aligned in X,
                depths are world x (-1 vs +1) -> A FRONT of B.
    Metric distances and object-object distances cannot change (pure rotation).
    """

    def setUp(self):
        self.state = _scene([_obj("a", (-1.0, 0.0, 0.0)), _obj("b", (1.0, 0.0, 0.0))])
        self.cam_look_y = CameraPose(position=(0.0, 0.0, 0.0), look_at=(0.0, 1.0, 0.0))
        self.cam_look_x = CameraPose(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0))

    def test_both_poses_are_valid_cameras(self):
        camera_basis(self.cam_look_y)
        camera_basis(self.cam_look_x)

    def test_camera_frame_relations_can_change(self):
        y = compute_spatial_truth(self.state, self.cam_look_y)
        x = compute_spatial_truth(self.state, self.cam_look_x)
        self.assertEqual(_pair(y, 0, 1).left_right, REL_LEFT)
        self.assertEqual(_pair(y, 0, 1).front_behind, REL_SAME_DEPTH)
        self.assertEqual(_pair(x, 0, 1).left_right, REL_ALIGNED)
        self.assertEqual(_pair(x, 0, 1).front_behind, REL_FRONT)

    def test_metric_distance_unchanged_under_pure_rotation(self):
        y = compute_spatial_truth(self.state, self.cam_look_y)
        x = compute_spatial_truth(self.state, self.cam_look_x)
        for y_obj, x_obj in zip(y.objects, x.objects):
            self.assertAlmostEqual(y_obj.camera_metric_distance, x_obj.camera_metric_distance)
            self.assertAlmostEqual(y_obj.camera_metric_distance, 1.0)

    def test_near_far_ordering_unchanged_under_pure_rotation(self):
        y = compute_spatial_truth(self.state, self.cam_look_y)
        x = compute_spatial_truth(self.state, self.cam_look_x)
        self.assertEqual(_pair(y, 0, 1).near_far, REL_EQUIDISTANT)
        self.assertEqual(_pair(x, 0, 1).near_far, REL_EQUIDISTANT)

    def test_object_to_object_distance_unchanged(self):
        y = compute_spatial_truth(self.state, self.cam_look_y)
        x = compute_spatial_truth(self.state, self.cam_look_x)
        self.assertAlmostEqual(_pair(y, 0, 1).metric_distance, 2.0)
        self.assertAlmostEqual(_pair(x, 0, 1).metric_distance, 2.0)


class CameraRelocationTests(unittest.TestCase):
    """D. Move camera from one side of A/B to the other: near/far flips, distance fixed."""

    def setUp(self):
        self.state = _scene([_obj("a", (-1.0, 0.0, 0.0)), _obj("b", (1.0, 0.0, 0.0))])
        self.cam_west = CameraPose(position=(-3.0, 0.0, 0.0), look_at=(0.0, 0.0, 0.0))
        self.cam_east = CameraPose(position=(3.0, 0.0, 0.0), look_at=(0.0, 0.0, 0.0))

    def test_near_far_flips_when_camera_crosses(self):
        west = compute_spatial_truth(self.state, self.cam_west)
        east = compute_spatial_truth(self.state, self.cam_east)
        self.assertEqual(_pair(west, 0, 1).near_far, REL_NEARER)   # A is 2, B is 4
        self.assertEqual(_pair(east, 0, 1).near_far, REL_FARTHER)  # A is 4, B is 2

    def test_object_to_object_distance_cannot_change(self):
        west = compute_spatial_truth(self.state, self.cam_west)
        east = compute_spatial_truth(self.state, self.cam_east)
        self.assertAlmostEqual(_pair(west, 0, 1).metric_distance, 2.0)
        self.assertAlmostEqual(_pair(east, 0, 1).metric_distance, 2.0)

    def test_metric_distance_scalars_reflect_relocation(self):
        west = compute_spatial_truth(self.state, self.cam_west)
        east = compute_spatial_truth(self.state, self.cam_east)
        self.assertAlmostEqual(west.objects[0].camera_metric_distance, 2.0)
        self.assertAlmostEqual(west.objects[1].camera_metric_distance, 4.0)
        self.assertAlmostEqual(east.objects[0].camera_metric_distance, 4.0)
        self.assertAlmostEqual(east.objects[1].camera_metric_distance, 2.0)


class CanonicalAndSampledCameraTests(unittest.TestCase):
    """E. Truth generation stays valid and deterministic over many pose sources."""

    def setUp(self):
        self.state = _scene(
            [
                _obj("a", (-1.0, 0.0, 0.4)),
                _obj("b", (1.0, 0.0, 0.4)),
                _obj("c", (0.0, -1.0, 0.8)),
            ]
        )

    def _poses(self):
        poses = []
        poses.extend(canonical_views(6, target=_TARGET, radius=_RADIUS))
        poses.extend(canonical_views(14, target=_TARGET, radius=_RADIUS))
        poses.extend(canonical_views(26, target=_TARGET, radius=_RADIUS))
        base = canonical_views(14, target=_TARGET, radius=_RADIUS)[0]
        for seed in (0, 5, 99):
            poses.append(
                jitter_pose(
                    base,
                    seed=seed,
                    angular_deg=5.0,
                    radius_frac=0.1,
                    target_radius=0.2,
                )
            )
        for seed in (1, 2, 3):
            poses.append(sample_camera(target=_TARGET, radius=_RADIUS, seed=seed))
        return poses

    def test_all_pose_sources_generate_valid_deterministic_records(self):
        n_objects = len(self.state.objects)
        for camera in self._poses():
            first = compute_spatial_truth(self.state, camera)
            second = compute_spatial_truth(self.state, camera)
            self.assertEqual(first, second)
            self.assertIsInstance(first, SpatialTruthRecord)
            self.assertEqual(len(first.objects), n_objects)
            self.assertEqual(len(first.pairs), n_objects * (n_objects - 1) // 2)


class BehindCameraPolicyTests(unittest.TestCase):
    """F. Behind-camera objects are kept and flagged, never dropped."""

    def setUp(self):
        self.state = _scene(
            [
                _obj("behind", (0.0, -5.0, 0.0)),
                _obj("at_camera_plane", (0.0, 0.0, 0.0)),
                _obj("front", (0.0, 8.0, 0.0)),
            ]
        )
        self.camera = CameraPose(position=(0.0, 0.0, 0.0), look_at=(0.0, 1.0, 0.0))

    def test_in_front_of_camera_flags(self):
        record = compute_spatial_truth(self.state, self.camera)
        self.assertFalse(record.objects[0].in_front_of_camera)  # depth -5
        self.assertFalse(record.objects[1].in_front_of_camera)  # depth 0 <= eps
        self.assertTrue(record.objects[2].in_front_of_camera)   # depth +8

    def test_all_objects_still_present(self):
        record = compute_spatial_truth(self.state, self.camera)
        self.assertEqual([o.object_index for o in record.objects], [0, 1, 2])

    def test_pairwise_relations_still_emitted(self):
        # Pairwise relations follow the signed camera-frame rules even for
        # behind-camera objects. Index 0 has camera depth -5 and index 2 has
        # depth +8, so dz = -5 - 8 = -13 < -eps and index 0 is deterministically
        # FRONT of index 2 (shallower signed depth), not merely "any label".
        record = compute_spatial_truth(self.state, self.camera)
        self.assertEqual(len(record.pairs), 3)
        pair = _pair(record, 0, 2)
        self.assertEqual(pair.near_far, REL_NEARER)  # metric 5 vs 8
        self.assertEqual(pair.left_right, REL_ALIGNED)  # both on camera x = 0
        self.assertEqual(pair.front_behind, REL_FRONT)  # dz = -13 < -eps

    def test_duplicate_names_do_not_corrupt_identity(self):
        state = _scene([_obj("dup", (-1.0, 0.0, 0.0)), _obj("dup", (1.0, 0.0, 0.0))])
        record = compute_spatial_truth(state, self.camera)
        self.assertEqual(record.objects[0].object_index, 0)
        self.assertEqual(record.objects[1].object_index, 1)
        self.assertEqual(len(record.pairs), 1)
        self.assertEqual(record.pairs[0].object_index_a, 0)
        self.assertEqual(record.pairs[0].object_index_b, 1)


class OrderingAndIdentityTests(unittest.TestCase):
    """G. object_index follows tuple order; pair ordering is deterministic index combinations."""

    def setUp(self):
        self.camera = CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0))
        self.state = _scene(
            [
                _obj("dup", (-2.0, 0.0, 0.0)),
                _obj("dup", (0.0, 0.0, 0.0)),
                _obj("dup", (2.0, 0.0, 0.0)),
            ]
        )

    def test_object_index_follows_tuple_order(self):
        record = compute_spatial_truth(self.state, self.camera)
        self.assertEqual([o.object_index for o in record.objects], [0, 1, 2])
        self.assertEqual([o.world_location for o in record.objects],
                         [(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)])

    def test_pair_order_is_deterministic_combinations(self):
        record = compute_spatial_truth(self.state, self.camera)
        self.assertEqual(len(record.pairs), 3)
        self.assertEqual(
            [(p.object_index_a, p.object_index_b) for p in record.pairs],
            [(0, 1), (0, 2), (1, 2)],
        )

    def test_duplicate_names_do_not_corrupt_pairs(self):
        record = compute_spatial_truth(self.state, self.camera)
        for pair in record.pairs:
            self.assertEqual(pair.name_a, "dup")
            self.assertEqual(pair.name_b, "dup")
        self.assertNotEqual(
            record.pairs[0].left_right, REL_ALIGNED,
            msg="distinct geometries must not collapse to all-aligned under duplicates",
        )


class SerializationTests(unittest.TestCase):
    """H. Determinism, json compatibility, stable ordering."""

    def setUp(self):
        self.state = _scene(
            [
                _obj("a", (-1.0, 0.0, 0.4), size=0.5),
                _obj("b", (1.0, 0.0, 0.4), size=1.5),
                _obj("c", (0.0, 1.0, 0.4), size=1.0),
            ]
        )
        self.camera = CameraPose(
            position=(0.0, -6.0, 3.0),
            look_at=(0.0, 0.0, 0.4),
            up=(0.0, 0.0, 1.0),
            fov_deg=60.0,
        )

    def test_identical_calls_produce_equal_records(self):
        a = compute_spatial_truth(self.state, self.camera)
        b = compute_spatial_truth(self.state, self.camera)
        self.assertEqual(a, b)

    def test_to_dict_is_json_compatible(self):
        record = compute_spatial_truth(self.state, self.camera)
        data = record.to_dict()
        json.dumps(data)
        self.assertEqual(data["scene_id"], "scene_test")
        self.assertEqual(data["camera"]["fov_deg"], 60.0)

    def test_to_dict_deterministic_order(self):
        record = compute_spatial_truth(self.state, self.camera)
        data = record.to_dict()
        self.assertEqual([o["object_index"] for o in data["objects"]], [0, 1, 2])
        self.assertEqual(
            [(p["object_index_a"], p["object_index_b"]) for p in data["pairs"]],
            [(0, 1), (0, 2), (1, 2)],
        )

    def test_per_object_truth_exposes_required_fields(self):
        record = compute_spatial_truth(self.state, self.camera)
        obj = record.objects[0]
        self.assertIsInstance(obj, ObjectTruth)
        for field in (
            "object_index",
            "name",
            "shape",
            "color",
            "size",
            "world_location",
            "camera_right_x",
            "camera_up_y",
            "camera_depth",
            "camera_metric_distance",
            "in_front_of_camera",
        ):
            self.assertIn(field, obj.to_dict())

    def test_per_pair_truth_exposes_required_fields(self):
        record = compute_spatial_truth(self.state, self.camera)
        pair = record.pairs[0]
        self.assertIsInstance(pair, PairTruth)
        for field in (
            "object_index_a",
            "object_index_b",
            "name_a",
            "name_b",
            "left_right",
            "above_below",
            "front_behind",
            "near_far",
            "metric_distance",
        ):
            self.assertIn(field, pair.to_dict())


if __name__ == "__main__":
    unittest.main()
