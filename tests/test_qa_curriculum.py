"""CPU-only tests for the G2.0-D multi-view QA curriculum layer (no bpy)."""

import json
import unittest

from spatialforge.environment.camera import CameraPose
from spatialforge.environment.scene import SceneObject, SceneState
from spatialforge.environment.sampling import (
    canonical_directions,
    canonical_views,
    jitter_pose,
    sample_camera,
)
from spatialforge.environment.relation import (
    ObjectTruth,
    PairTruth,
    SpatialTruthRecord,
    compute_spatial_truth,
)
from spatialforge.environment.qa import (
    CurriculumView,
    QASample,
    generate_multiview_curriculum,
    generate_paired_view_qa,
    generate_single_view_qa,
)

_TARGET = (0.0, 0.0, 0.4)
_RADIUS = 6.0


def _obj(name, location, size=1.0, shape="cube", color="gray") -> SceneObject:
    return SceneObject(name=name, shape=shape, color=color, location=location, size=size)


def _scene(objects, scene_id="scene_test", seed=0) -> SceneState:
    return SceneState(scene_id=scene_id, seed=seed, objects=tuple(objects))


def _truth(state, camera):
    return compute_spatial_truth(state, camera)


def _single(record, view_id="south", **kwargs):
    return generate_single_view_qa(record, view_id, **kwargs)


def _samples_by_family(samples):
    out = {}
    for sample in samples:
        out.setdefault(sample.family, []).append(sample)
    return out


def _samples_by_id(samples):
    return {sample.sample_id: sample for sample in samples}


class SingleViewExactSemanticsTests(unittest.TestCase):
    """B. Controlled geometry yields exact left/right/above/below/front/behind/
    nearer/farther single-view answers.

    Camera at origin looking along +Y, so the camera frame is
    (right_x, up_y, depth_z) = (world_x, world_z, world_y).
    """

    CAMERA = CameraPose(position=(0.0, 0.0, 0.0), look_at=(0.0, 1.0, 0.0))

    def setUp(self):
        # One object per cardinal direction relative to the anchor. Camera
        # frame is (right_x, up_y, depth_z) = (world_x, world_z, world_y).
        self.state = _scene(
            [
                _obj("deep", (0.0, 8.0, 0.0)),
                _obj("shallow", (0.0, 2.0, 0.0)),
                _obj("left", (-3.0, 5.0, 0.0)),
                _obj("right", (3.0, 5.0, 0.0)),
                _obj("upper", (0.0, 5.0, 2.0)),
                _obj("lower", (0.0, 5.0, -2.0)),
            ]
        )
        self.record = _truth(self.state, self.CAMERA)
        self.samples = _single(self.record)

    def _answer(self, index_a, index_b, family):
        for sample in self.samples:
            if (
                sample.object_indices == (index_a, index_b)
                and sample.family == family
            ):
                return sample.answer
        return None

    def test_all_directional_words_present_exactly(self):
        self.assertEqual(self._answer(0, 3, "horizontal"), "left")    # x 0 vs 3
        self.assertEqual(self._answer(0, 2, "horizontal"), "right")   # x 0 vs -3
        self.assertEqual(self._answer(0, 5, "vertical"), "above")     # z 0 vs -2
        self.assertEqual(self._answer(0, 4, "vertical"), "below")     # z 0 vs +2
        self.assertEqual(self._answer(1, 3, "depth"), "front")        # y 2 vs 5
        self.assertEqual(self._answer(0, 1, "depth"), "behind")       # y 8 vs 2
        self.assertEqual(self._answer(1, 3, "near_far"), "nearer")    # 2 vs 5
        self.assertEqual(self._answer(0, 1, "near_far"), "farther")   # 8 vs 2

    def test_question_says_from_this_view_and_uses_index_plus_name(self):
        for sample in self.samples:
            self.assertIn("From this view", sample.question)
            index_a, index_b = sample.object_indices
            names = [o.name for o in self.state.objects]
            self.assertIn(f'object {index_a} ("{names[index_a]}")', sample.question)
            self.assertIn(f'object {index_b} ("{names[index_b]}")', sample.question)

    def test_single_view_samples_are_view_dependent(self):
        self.assertTrue(self.samples)
        for sample in self.samples:
            self.assertTrue(sample.is_view_dependent)
            self.assertEqual(sample.mode, "single")
            self.assertEqual(len(sample.view_ids), 1)


class NeutralPolicyTests(unittest.TestCase):
    """C. Neutral states absent by default; emitted verbatim when requested."""

    CAMERA = CameraPose(position=(0.0, 0.0, 0.0), look_at=(0.0, 1.0, 0.0))

    def setUp(self):
        # Same depth plane (world y), same height (world z), symmetric metric.
        self.state = _scene(
            [_obj("a", (-2.0, 5.0, 0.0)), _obj("b", (2.0, 5.0, 0.0))]
        )
        self.record = _truth(self.state, self.CAMERA)

    def test_neutral_samples_absent_by_default(self):
        samples = _single(self.record)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].family, "horizontal")
        self.assertEqual(samples[0].answer, "left")

    def test_include_neutral_emits_exact_labels(self):
        samples = _single(self.record, include_neutral=True)
        by_family = _samples_by_family(samples)
        self.assertEqual(
            {family: [s.answer for s in ss] for family, ss in by_family.items()},
            {
                "horizontal": ["left"],
                "vertical": ["aligned"],
                "depth": ["same_depth"],
                "near_far": ["equidistant"],
            },
        )


class BehindCameraFilteringTests(unittest.TestCase):
    """D. Behind-camera objects never leak into normal visual QA."""

    CAMERA = CameraPose(position=(0.0, 0.0, 0.0), look_at=(0.0, 1.0, 0.0))

    def setUp(self):
        self.state = _scene(
            [
                _obj("behind", (0.0, -5.0, 0.0)),   # 0 depth -5
                _obj("plane", (0.0, 0.0, 0.0)),     # 1 depth 0
                _obj("front_a", (-2.0, 8.0, 0.0)),  # 2
                _obj("front_b", (2.0, 8.0, 0.0)),   # 3
            ]
        )
        self.record = _truth(self.state, self.CAMERA)

    def test_only_pairs_with_both_objects_in_front_are_emitted(self):
        samples = _single(self.record)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.object_indices, (2, 3))
        self.assertEqual(sample.family, "horizontal")
        self.assertEqual(sample.answer, "left")

    def test_behind_filter_survives_include_neutral(self):
        samples = _single(self.record, include_neutral=True)
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(s.object_indices == (2, 3) for s in samples))

    def test_no_visibility_claim_in_questions(self):
        samples = _single(self.record, include_neutral=True)
        for sample in samples:
            self.assertNotIn("visible", sample.question.lower())
            self.assertNotIn("fov", sample.question.lower())


class OppositeViewCounterfactualTests(unittest.TestCase):
    """E. South vs north on the same scene: relations flip correctly."""

    def setUp(self):
        self.state = _scene(
            [
                _obj("west_obj", (-1.0, 0.0, 0.0), size=0.5, shape="sphere", color="red"),
                _obj("east_obj", (1.0, 0.0, 0.0), size=2.0, shape="cube", color="blue"),
                _obj("near_obj", (0.0, -1.0, 0.0)),
                _obj("far_obj", (0.0, 1.0, 0.0)),
            ]
        )
        self.south = _truth(self.state, CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0)))
        self.north = _truth(self.state, CameraPose(position=(0.0, 6.0, 0.0), look_at=(0.0, 0.0, 0.0)))

    def test_paired_qa_contains_exact_left_to_right_transition(self):
        samples = generate_paired_view_qa(self.south, "south", self.north, "north")
        sample = _samples_by_id(samples)[
            "scene_test:paired:south->north:0-1:horizontal"
        ]
        self.assertEqual(sample.answer, "left -> right")
        self.assertTrue(sample.is_view_dependent)

    def test_paired_qa_contains_exact_front_to_behind_transition(self):
        samples = generate_paired_view_qa(self.south, "south", self.north, "north")
        sample = _samples_by_id(samples)["scene_test:paired:south->north:2-3:depth"]
        self.assertEqual(sample.answer, "front -> behind")

    def test_single_view_answers_change_correctly(self):
        south_samples = _samples_by_id(_single(self.south, "south"))
        north_samples = _samples_by_id(_single(self.north, "north"))
        self.assertEqual(
            south_samples["scene_test:single:south:0-1:horizontal"].answer, "left"
        )
        self.assertEqual(
            north_samples["scene_test:single:north:0-1:horizontal"].answer, "right"
        )

    def test_opposite_relation_equivalence_pair_metrics_invariant(self):
        samples = generate_paired_view_qa(self.south, "south", self.north, "north")
        control = _samples_by_id(samples)[
            "scene_test:paired:south->north:0-1:metric_invariance"
        ]
        self.assertEqual(control.answer, "unchanged")
        self.assertFalse(control.is_view_dependent)
        self.assertIn("world_invariant", control.tags)


class RotationOnlyControlTests(unittest.TestCase):
    """F. Same camera.position, changed orientation.

    Objects stay in the front halfspace for both look directions. Metric
    distance / near-far ordering cannot change under pure rotation, while the
    depth relation flips front <-> behind.
    """

    def setUp(self):
        self.state = _scene([_obj("a", (1.0, 1.0, 0.0)), _obj("b", (0.5, 2.0, 0.0))])
        self.look_y = _truth(self.state, CameraPose(position=(0.0, 0.0, 0.0), look_at=(0.0, 1.0, 0.0)))
        self.look_x = _truth(self.state, CameraPose(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0)))

    def test_near_far_unchanged_under_rotation(self):
        y_ans = _samples_by_id(_single(self.look_y, "y"))["scene_test:single:y:0-1:near_far"].answer
        x_ans = _samples_by_id(_single(self.look_x, "x"))["scene_test:single:x:0-1:near_far"].answer
        self.assertEqual(y_ans, "nearer")
        self.assertEqual(x_ans, "nearer")

    def test_viewpoint_dependent_depth_changes(self):
        samples = generate_paired_view_qa(self.look_y, "y", self.look_x, "x")
        sample = _samples_by_id(samples)["scene_test:paired:y->x:0-1:depth"]
        self.assertEqual(sample.answer, "front -> behind")

    def test_unchanged_near_far_absent_by_default_and_present_when_requested(self):
        default = _samples_by_id(generate_paired_view_qa(self.look_y, "y", self.look_x, "x"))
        self.assertNotIn("scene_test:paired:y->x:0-1:near_far", default)
        with_unchanged = _samples_by_id(
            generate_paired_view_qa(
                self.look_y, "y", self.look_x, "x", include_unchanged=True
            )
        )
        near_far = with_unchanged["scene_test:paired:y->x:0-1:near_far"]
        self.assertEqual(near_far.answer, "nearer -> nearer")
        self.assertIn("unchanged", near_far.tags)

    def test_world_invariant_metric_control_present(self):
        samples = generate_paired_view_qa(self.look_y, "y", self.look_x, "x")
        control = _samples_by_id(samples)["scene_test:paired:y->x:0-1:metric_invariance"]
        self.assertEqual(control.answer, "unchanged")


class RelocationCounterfactualTests(unittest.TestCase):
    """G. Camera relocation produces nearer -> farther while both objects stay
    in front in both views."""

    def setUp(self):
        self.state = _scene([_obj("a", (-2.0, 0.0, 0.0)), _obj("b", (2.0, 0.0, 0.0))])
        self.west = _truth(self.state, CameraPose(position=(-6.0, 0.0, 0.0), look_at=(0.0, 0.0, 0.0)))
        self.east = _truth(self.state, CameraPose(position=(6.0, 0.0, 0.0), look_at=(0.0, 0.0, 0.0)))

    def test_nearer_to_farther_transition_emitted(self):
        samples = generate_paired_view_qa(self.west, "west", self.east, "east")
        sample = _samples_by_id(samples)["scene_test:paired:west->east:0-1:near_far"]
        self.assertEqual(sample.answer, "nearer -> farther")

    def test_metric_distance_between_objects_still_unchanged(self):
        samples = generate_paired_view_qa(self.west, "west", self.east, "east")
        control = _samples_by_id(samples)[
            "scene_test:paired:west->east:0-1:metric_invariance"
        ]
        self.assertEqual(control.answer, "unchanged")


class WorldInvariantControlTests(unittest.TestCase):
    """H. World-invariant metric-distance control: emits unchanged and rejects
    incompatible or broken truth."""

    def setUp(self):
        self.state = _scene(
            [
                _obj("west_obj", (-1.0, 0.0, 0.0)),
                _obj("east_obj", (1.0, 0.0, 0.0)),
                _obj("near_obj", (0.0, -1.0, 0.0)),
            ]
        )
        self.south = _truth(self.state, CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0)))
        self.north = _truth(self.state, CameraPose(position=(0.0, 6.0, 0.0), look_at=(0.0, 0.0, 0.0)))

    def test_invariant_controls_emitted_for_every_eligible_pair(self):
        samples = generate_paired_view_qa(self.south, "south", self.north, "north")
        invariants = [s for s in samples if s.family == "metric_invariance"]
        self.assertEqual(len(invariants), 3)
        for control in invariants:
            self.assertEqual(control.answer, "unchanged")
            self.assertFalse(control.is_view_dependent)
            self.assertIn("world_invariant", control.tags)

    def test_mismatched_scene_id_rejected(self):
        other = _scene([_obj("a", (0.0, 0.0, 0.0))], scene_id="other")
        other_truth = _truth(other, CameraPose(position=(0.0, 0.0, 0.0), look_at=(0.0, 1.0, 0.0)))
        with self.assertRaises(ValueError):
            generate_paired_view_qa(self.south, "south", other_truth, "north")

    def test_mismatched_object_count_rejected(self):
        fewer = _scene([_obj("a", (-1.0, 0.0, 0.0))])
        fewer_truth = _truth(fewer, CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0)))
        with self.assertRaises(ValueError):
            generate_paired_view_qa(self.south, "south", fewer_truth, "north")

    def test_incompatible_world_facts_rejected(self):
        moved = _scene([_obj("west_obj", (-1.0, 0.0, 0.0)), _obj("east_obj", (1.0, 0.0, 0.0)), _obj("near_obj", (0.0, -9.0, 0.0))])
        moved_truth = _truth(moved, CameraPose(position=(0.0, 6.0, 0.0), look_at=(0.0, 0.0, 0.0)))
        with self.assertRaises(ValueError):
            generate_paired_view_qa(self.south, "south", moved_truth, "north")


class BrokenTruthInvariantTests(unittest.TestCase):
    """H. Deliberately fabricated truth records whose world facts agree but
    whose pair metric_distance disagrees are rejected as broken invariants."""

    CAMERA = CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0))

    def _object_truth(self, index, name, location):
        return ObjectTruth(
            object_index=index,
            name=name,
            shape="cube",
            color="gray",
            size=1.0,
            world_location=location,
            camera_right_x=0.0,
            camera_up_y=0.0,
            camera_depth=5.0,
            camera_metric_distance=5.0,
            in_front_of_camera=True,
        )

    def _pair_truth(self, index_a, index_b, metric_distance):
        return PairTruth(
            object_index_a=index_a,
            object_index_b=index_b,
            name_a="a",
            name_b="b",
            left_right="left",
            above_below="aligned",
            front_behind="same_depth",
            near_far="nearer",
            metric_distance=metric_distance,
        )

    def test_metric_distance_disagreement_raises(self):
        objects = (self._object_truth(0, "a", (0.0, 0.0, 0.0)), self._object_truth(1, "b", (0.0, 2.0, 0.0)))
        truth_a = SpatialTruthRecord(
            scene_id="scene_test", camera=self.CAMERA, eps=1e-6,
            objects=objects, pairs=(self._pair_truth(0, 1, 2.0),),
        )
        truth_b = SpatialTruthRecord(
            scene_id="scene_test", camera=self.CAMERA, eps=1e-6,
            objects=objects, pairs=(self._pair_truth(0, 1, 3.0),),
        )
        with self.assertRaises(ValueError):
            generate_paired_view_qa(truth_a, "south", truth_b, "north")


class CanonicalCurriculumTests(unittest.TestCase):
    """I. Canonical curriculum: deterministic, canonical tag preserved."""

    def setUp(self):
        self.state = _scene(
            [
                _obj("a", (-1.0, 0.0, 0.4), size=0.5),
                _obj("b", (1.0, 0.0, 0.4), size=1.5),
                _obj("c", (0.0, 1.0, 0.4), size=1.0),
            ]
        )

    def _views(self, kind):
        directions = canonical_directions(kind)
        poses = canonical_views(kind, target=_TARGET, radius=_RADIUS)
        return tuple(
            CurriculumView(direction_id, pose, tags=("canonical",))
            for (direction_id, _), pose in zip(directions, poses)
        )

    def test_kind_6_deterministic_and_tagged(self):
        views = self._views(6)
        first = generate_multiview_curriculum(self.state, views)
        second = generate_multiview_curriculum(self.state, views)
        self.assertEqual(first, second)
        self.assertTrue(first)
        singles = [s for s in first if s.mode == "single"]
        paireds = [s for s in first if s.mode == "paired"]
        self.assertTrue(singles)
        self.assertTrue(paireds)
        for sample in first:
            self.assertIn("canonical", sample.tags)

    def test_kind_14_and_26_are_deterministic(self):
        for kind in (14, 26):
            with self.subTest(kind=kind):
                views = self._views(kind)
                first = generate_multiview_curriculum(self.state, views)
                second = generate_multiview_curriculum(self.state, views)
                self.assertEqual(first, second)
                self.assertTrue(first)

    def test_single_view_blocks_preserve_input_view_order(self):
        views = self._views(6)
        out = generate_multiview_curriculum(self.state, views)
        seen_view_order = []
        for sample in out:
            if sample.mode == "single":
                seen_view_order.append(sample.view_ids[0])
        expected = []
        for view in views:
            expected.extend([view.view_id] * len(
                [s for s in out if s.mode == "single" and s.view_ids[0] == view.view_id]
            ))
        self.assertEqual(seen_view_order, expected)


class JitterCurriculumTests(unittest.TestCase):
    """J. Seeded jitter: deterministic QA, jitter tag preserved."""

    def setUp(self):
        self.state = _scene(
            [_obj("a", (-1.0, 0.0, 0.4)), _obj("b", (1.0, 0.0, 0.4)), _obj("c", (0.0, 1.0, 0.4))]
        )
        self.base = canonical_views(6, target=_TARGET, radius=_RADIUS)[0]

    def _views(self, seed):
        pose = jitter_pose(
            self.base, seed=seed, angular_deg=5.0, radius_frac=0.1, target_radius=0.2
        )
        return (CurriculumView(f"jitter_{seed}", pose, tags=("jitter",)),)

    def test_jittered_curriculum_is_deterministic_and_tagged(self):
        views = self._views(7)
        first = generate_multiview_curriculum(self.state, views)
        second = generate_multiview_curriculum(self.state, views)
        self.assertEqual(first, second)
        self.assertTrue(first)
        for sample in first:
            self.assertIn("jitter", sample.tags)
            self.assertEqual(sample.view_ids[0], "jitter_7")


class HardAngleCurriculumTests(unittest.TestCase):
    """K. Continuous sampled (hard-angle) view needs no canonical assumption."""

    def setUp(self):
        self.state = _scene(
            [_obj("a", (0.4, 0.1, 0.4)), _obj("b", (-0.3, 0.2, 0.4)), _obj("c", (0.1, -0.4, 0.4))]
        )

    def _views(self, seed):
        pose = sample_camera(target=_TARGET, radius=_RADIUS, seed=seed)
        return (CurriculumView(f"continuous_{seed}", pose, tags=("hard_angle",)),)

    def test_sampled_view_generates_deterministic_qa(self):
        views = self._views(11)
        first = generate_multiview_curriculum(self.state, views)
        second = generate_multiview_curriculum(self.state, views)
        self.assertEqual(first, second)
        self.assertTrue(first)
        for sample in first:
            self.assertIn("hard_angle", sample.tags)

    def test_tag_is_callersupplied_not_inferred(self):
        # Distinct poses with identical floating-point-agnostic ids must not
        # gain canonical/symmetry provenance beyond what the caller declared.
        views = (CurriculumView("continuous_3", sample_camera(target=_TARGET, radius=_RADIUS, seed=3), tags=("hard_angle",)),)
        out = generate_multiview_curriculum(self.state, views)
        for sample in out:
            self.assertNotIn("canonical", sample.tags)
            self.assertIn("hard_angle", sample.tags)


class SymmetryTrapTests(unittest.TestCase):
    """L. Exact inverse paired transitions receive symmetry_flip."""

    def test_opposite_view_transitions_are_symmetry_flips(self):
        state = _scene([_obj("a", (-2.0, 0.0, 0.0)), _obj("b", (2.0, 0.0, 0.0))])
        west = _truth(state, CameraPose(position=(-6.0, 0.0, 0.0), look_at=(0.0, 0.0, 0.0)))
        east = _truth(state, CameraPose(position=(6.0, 0.0, 0.0), look_at=(0.0, 0.0, 0.0)))
        samples = generate_paired_view_qa(west, "west", east, "east")
        flips = [
            s for s in samples
            if s.mode == "paired"
            and s.family in ("depth", "near_far")
            and s.object_indices == (0, 1)
        ]
        self.assertTrue(flips)
        for sample in flips:
            self.assertIn("symmetry_flip", sample.tags)
            self.assertEqual(sample.answer, "front -> behind") if sample.family == "depth" else None
        answers = {s.family: s.answer for s in flips}
        self.assertEqual(answers["depth"], "front -> behind")
        self.assertEqual(answers["near_far"], "nearer -> farther")


class SizeDistanceConflictTests(unittest.TestCase):
    """M. large-far vs small-near traps receive size_distance_conflict."""

    CAMERA = CameraPose(position=(0.0, 0.0, 0.0), look_at=(0.0, 1.0, 0.0))

    def test_large_far_versus_small_near_flagged(self):
        state = _scene(
            [_obj("big_far", (0.0, 10.0, 0.0), size=3.0), _obj("small_near", (0.0, 2.0, 0.0), size=0.5)]
        )
        record = _truth(state, self.CAMERA)
        samples = _single(record)
        self.assertTrue(samples)
        for sample in samples:
            self.assertEqual(sample.object_indices, (0, 1))
            self.assertIn("size_distance_conflict", sample.tags)
        answers = {s.family: s.answer for s in samples}
        self.assertEqual(answers["near_far"], "farther")  # big_far farther

    def test_large_near_versus_small_far_not_flagged(self):
        state = _scene(
            [_obj("big_near", (0.0, 2.0, 0.0), size=3.0), _obj("small_far", (0.0, 10.0, 0.0), size=0.5)]
        )
        record = _truth(state, self.CAMERA)
        samples = _single(record)
        self.assertTrue(samples)
        for sample in samples:
            self.assertNotIn("size_distance_conflict", sample.tags)


class DuplicateNameTests(unittest.TestCase):
    """N. Duplicate SceneObject.name values stay unambiguous via index."""

    CAMERA = CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0))

    def setUp(self):
        self.state = _scene(
            [_obj("dup", (-2.0, 0.0, 0.0)), _obj("dup", (0.0, 0.0, 0.0)), _obj("dup", (2.0, 0.0, 0.0))]
        )
        self.record = _truth(self.state, self.CAMERA)

    def test_sample_ids_unique_and_index_based(self):
        samples = _single(self.record)
        ids = [s.sample_id for s in samples]
        self.assertEqual(len(ids), len(set(ids)))
        for sample in samples:
            index_a, index_b = sample.object_indices
            self.assertIn(f":{index_a}-{index_b}:", sample.sample_id)

    def test_question_contains_object_indices(self):
        samples = _single(self.record, include_neutral=True)
        self.assertTrue(samples)
        for sample in samples:
            index_a, index_b = sample.object_indices
            self.assertIn(f'object {index_a} ("dup")', sample.question)
            self.assertIn(f'object {index_b} ("dup")', sample.question)
            self.assertEqual(index_a, sample.object_indices[0])
            self.assertEqual(index_b, sample.object_indices[1])


class DeterminismTests(unittest.TestCase):
    """O. Identical inputs -> identical tuple, ids, and serialization."""

    def setUp(self):
        self.state = _scene(
            [_obj("a", (-1.0, 0.0, 0.4)), _obj("b", (1.0, 0.0, 0.4)), _obj("c", (0.0, 1.0, 0.4))]
        )

    def _views(self):
        directions = canonical_directions(6)
        poses = canonical_views(6, target=_TARGET, radius=_RADIUS)
        return tuple(
            CurriculumView(direction_id, pose, tags=("canonical",))
            for (direction_id, _), pose in zip(directions, poses)
        )

    def test_multiview_output_identical(self):
        views = self._views()
        first = generate_multiview_curriculum(self.state, views)
        second = generate_multiview_curriculum(self.state, views)
        self.assertEqual(first, second)
        self.assertEqual(
            [s.sample_id for s in first], [s.sample_id for s in second]
        )
        self.assertEqual(
            [s.to_dict() for s in first], [s.to_dict() for s in second]
        )

    def test_include_neutral_identical(self):
        views = self._views()
        first = generate_multiview_curriculum(self.state, views, include_neutral=True)
        second = generate_multiview_curriculum(self.state, views, include_neutral=True)
        self.assertEqual(first, second)


class JsonCompatibilityTests(unittest.TestCase):
    """P. json.dumps of sample dicts works and is deterministic."""

    def test_json_serialization(self):
        state = _scene([_obj("a", (-1.0, 0.0, 0.4)), _obj("b", (1.0, 0.0, 0.4))])
        south = _truth(state, CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0)))
        north = _truth(state, CameraPose(position=(0.0, 6.0, 0.0), look_at=(0.0, 0.0, 0.0)))
        samples = generate_single_view_qa(south, "south", include_neutral=True)
        samples += generate_paired_view_qa(south, "south", north, "north")
        payload_a = json.dumps([s.to_dict() for s in samples], sort_keys=True)
        payload_b = json.dumps([s.to_dict() for s in samples], sort_keys=True)
        self.assertEqual(payload_a, payload_b)
        loaded = json.loads(payload_a)
        self.assertEqual(len(loaded), len(samples))
        self.assertIsInstance(samples[0], QASample)


class OrderingLockTests(unittest.TestCase):
    """Q. Exact sample ordering locked for a small controlled scene."""

    def test_exact_multiview_sample_order(self):
        state = _scene([_obj("a", (-1.0, 0.0, 0.0)), _obj("b", (1.0, 0.0, 0.0))])
        views = (
            CurriculumView("south", CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0)), tags=("canonical",)),
            CurriculumView("north", CameraPose(position=(0.0, 6.0, 0.0), look_at=(0.0, 0.0, 0.0)), tags=("canonical",)),
        )
        out = generate_multiview_curriculum(state, views)
        self.assertEqual(
            [s.sample_id for s in out],
            [
                "scene_test:single:south:0-1:horizontal",
                "scene_test:single:north:0-1:horizontal",
                "scene_test:paired:south->north:0-1:horizontal",
                "scene_test:paired:south->north:0-1:metric_invariance",
            ],
        )

    def test_single_view_family_order_locked(self):
        state = _scene(
            [_obj("a", (-1.0, 0.0, 0.0), size=0.5), _obj("b", (1.0, 0.0, 0.0), size=0.5)]
        )
        south = _truth(state, CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0)))
        samples = generate_single_view_qa(south, "south", include_neutral=True)
        self.assertEqual(
            [(s.family, s.answer) for s in samples],
            [("horizontal", "left"), ("vertical", "aligned"), ("depth", "same_depth"), ("near_far", "equidistant")],
        )


class ValidationTests(unittest.TestCase):
    """R. Reject invalid paired / multiview inputs."""

    def setUp(self):
        self.state = _scene([_obj("a", (-1.0, 0.0, 0.0)), _obj("b", (1.0, 0.0, 0.0))])
        self.truth = _truth(self.state, CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0)))

    def test_duplicate_view_id_rejected(self):
        views = (
            CurriculumView("v", CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0))),
            CurriculumView("v", CameraPose(position=(0.0, 6.0, 0.0), look_at=(0.0, 0.0, 0.0))),
        )
        with self.assertRaises(ValueError):
            generate_multiview_curriculum(self.state, views)

    def test_empty_view_id_rejected(self):
        with self.assertRaises(ValueError):
            CurriculumView("", CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0)))
        with self.assertRaises(ValueError):
            generate_single_view_qa(self.truth, "")
        with self.assertRaises(ValueError):
            generate_paired_view_qa(self.truth, "", self.truth, "north")
        with self.assertRaises(ValueError):
            generate_paired_view_qa(self.truth, "south", self.truth, "")

    def test_empty_multiview_returns_empty(self):
        self.assertEqual(generate_multiview_curriculum(self.state, ()), ())


class SingleViewApiShapeTests(unittest.TestCase):
    """Public single-view behaviour: mode, ids, families, view ids."""

    def setUp(self):
        self.state = _scene(
            [_obj("a", (-1.0, 0.0, 0.0)), _obj("b", (1.0, 0.0, 0.0)), _obj("c", (0.0, 1.0, 0.0))]
        )
        self.south = _truth(self.state, CameraPose(position=(0.0, -6.0, 0.0), look_at=(0.0, 0.0, 0.0)))

    def test_sample_shape(self):
        samples = _single(self.south, "south")
        for sample in samples:
            self.assertEqual(sample.sample_id, f"scene_test:single:south:{sample.object_indices[0]}-{sample.object_indices[1]}:{sample.family}")
            self.assertEqual(sample.scene_id, "scene_test")
            self.assertEqual(sample.mode, "single")
            self.assertEqual(sample.view_ids, ("south",))
            self.assertEqual(len(sample.object_indices), 2)
            self.assertIn(sample.family, ("horizontal", "vertical", "depth", "near_far"))
            self.assertTrue(sample.is_view_dependent)


if __name__ == "__main__":
    unittest.main()
