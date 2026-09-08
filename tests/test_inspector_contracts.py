"""Unit tests for InspectorSnapshot contracts, channel separation, and presentation order adapter."""

import json
import unittest

from spatialforge.environment.camera import CameraPose
from spatialforge.environment.relation import (
    PairTruth,
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
    compute_spatial_truth,
)
from spatialforge.environment.scene import SceneObject, SceneState
from spatialforge.inspector.contracts import (
    SCHEMA_VERSION,
    CameraChannel,
    EnvironmentTruthChannel,
    InspectorIdentity,
    InspectorSnapshot,
    ModelInputChannel,
    ObservationChannel,
    PredictionChannel,
    RuntimeChannel,
    SupervisionChannel,
    adapt_presentation_order,
    validate_model_input_isolation,
)
from spatialforge.inspector.projection import reconstruct_legacy_projection


class TestInspectorContracts(unittest.TestCase):
    def setUp(self):
        # Scene with DUPLICATE object names to test authoritative object_index identity
        self.state = SceneState(
            scene_id="test_scene",
            seed=42,
            objects=(
                SceneObject(name="red_cube", shape="cube", color="red", location=(-1.0, 0.0, 0.5), size=0.6),
                SceneObject(name="red_cube", shape="cube", color="red", location=(1.0, 0.0, 0.5), size=0.6),
                SceneObject(name="blue_sphere", shape="sphere", color="blue", location=(0.0, 2.0, 1.0), size=0.4),
            ),
        )
        self.camera = CameraPose(
            position=(0.0, -5.0, 2.0),
            look_at=(0.0, 0.0, 0.5),
        )

    def test_identity_preserves_object_index_with_duplicate_names(self):
        """Even with identical names, objects are identified strictly by object_index."""
        truth = compute_spatial_truth(self.state, self.camera)
        self.assertEqual(len(truth.objects), 3)

        # Both obj 0 and obj 1 are named "red_cube"
        self.assertEqual(truth.objects[0].name, "red_cube")
        self.assertEqual(truth.objects[1].name, "red_cube")
        self.assertEqual(truth.objects[0].object_index, 0)
        self.assertEqual(truth.objects[1].object_index, 1)

        # But camera_right_x differs (-1.0 vs 1.0)
        self.assertLess(truth.objects[0].camera_right_x, truth.objects[1].camera_right_x)

    def test_presentation_order_adapter_directional(self):
        """Verify deterministic inversion of directional relations for order 1."""
        pair = PairTruth(
            object_index_a=0,
            object_index_b=1,
            name_a="objA",
            name_b="objB",
            left_right=REL_LEFT,
            above_below=REL_ABOVE,
            front_behind=REL_FRONT,
            near_far=REL_NEARER,
            metric_distance=2.5,
        )

        # Order 0: Canonical (A relative to B)
        adapted_0 = adapt_presentation_order(pair, presentation_order=0)
        self.assertEqual(adapted_0["object_index_first"], 0)
        self.assertEqual(adapted_0["object_index_second"], 1)
        self.assertEqual(adapted_0["left_right"], REL_LEFT)
        self.assertEqual(adapted_0["above_below"], REL_ABOVE)
        self.assertEqual(adapted_0["front_behind"], REL_FRONT)
        self.assertEqual(adapted_0["near_far"], REL_NEARER)
        self.assertEqual(adapted_0["metric_distance"], 2.5)

        # Order 1: Inverted (B relative to A)
        adapted_1 = adapt_presentation_order(pair, presentation_order=1)
        self.assertEqual(adapted_1["object_index_first"], 1)
        self.assertEqual(adapted_1["object_index_second"], 0)
        self.assertEqual(adapted_1["left_right"], REL_RIGHT)
        self.assertEqual(adapted_1["above_below"], REL_BELOW)
        self.assertEqual(adapted_1["front_behind"], REL_BEHIND)
        self.assertEqual(adapted_1["near_far"], REL_FARTHER)
        self.assertEqual(adapted_1["metric_distance"], 2.5)

    def test_presentation_order_adapter_neutral(self):
        """Neutral relations remain neutral in inverted presentation order."""
        neutral_pair = PairTruth(
            object_index_a=0,
            object_index_b=1,
            name_a="objA",
            name_b="objB",
            left_right=REL_ALIGNED,
            above_below=REL_ALIGNED,
            front_behind=REL_SAME_DEPTH,
            near_far=REL_EQUIDISTANT,
            metric_distance=1.0,
        )

        adapted = adapt_presentation_order(neutral_pair, presentation_order=1)
        self.assertEqual(adapted["left_right"], REL_ALIGNED)
        self.assertEqual(adapted["above_below"], REL_ALIGNED)
        self.assertEqual(adapted["front_behind"], REL_SAME_DEPTH)
        self.assertEqual(adapted["near_far"], REL_EQUIDISTANT)

    def test_inspector_snapshot_serialization_and_channel_separation(self):
        """InspectorSnapshot serializes deterministically and contains separated channels."""
        proj = reconstruct_legacy_projection(self.camera)

        snapshot = InspectorSnapshot(
            identity=InspectorIdentity(
                scene_id="scene_000",
                view_id="south",
                sample_id="scene_000:single:south:0-1:depth:order_0",
                run_id="test_run",
            ),
            environment_truth=EnvironmentTruthChannel(
                scene_id="scene_000",
                seed=0,
                objects=[{"object_index": 0, "name": "c", "shape": "cube", "color": "red", "size": 0.5, "location": [0,0,0]}],
                camera_object_truths=[],
            ),
            camera=CameraChannel(
                pose={"position": list(self.camera.position), "look_at": list(self.camera.look_at), "fov_deg": 60.0},
                projection=proj.to_dict(),
                provenance=proj.provenance,
            ),
            observation=ObservationChannel(
                image_path="outputs/rendered/scene_000/south.png",
                image_url="/api/image?path=outputs/rendered/scene_000/south.png",
                scene_id="scene_000",
                view_id="south",
            ),
            model_input=ModelInputChannel(
                image_path="outputs/rendered/scene_000/south.png",
                question="Is the red cube in front of the blue sphere?",
            ),
            supervision=SupervisionChannel(
                ground_truth="front",
                family="depth",
                is_view_dependent=True,
            ),
            prediction=PredictionChannel(
                status="available",
                raw_prediction="front",
                parsed_prediction="front",
                is_valid_prediction=True,
                is_correct=True,
                provenance="test",
            ),
            runtime=RuntimeChannel(
                training_profile=None,
                progress=None,
                telemetry=None,
                source="none",
                freshness="missing",
            ),
        )

        d = snapshot.to_dict()
        self.assertEqual(d["schema_version"], SCHEMA_VERSION)
        self.assertIn("identity", d)
        self.assertIn("environment_truth", d)
        self.assertIn("camera", d)
        self.assertIn("observation", d)
        self.assertIn("model_input", d)
        self.assertIn("supervision", d)
        self.assertIn("prediction", d)
        self.assertIn("runtime", d)

        # JSON serialization round trip
        json_str = snapshot.to_json()
        loaded = json.loads(json_str)
        self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)

    def test_model_input_isolation_guard(self):
        """validate_model_input_isolation enforces strict allowlist: image + question only."""
        valid_input = {
            "model_input": {
                "image_path": "rendered.png",
                "question": "Is object left of object?",
            }
        }
        # Clean dict passes
        validate_model_input_isolation(valid_input)

        # Disallowed key in model_input raises AssertionError
        leaked_coords = {
            "model_input": {
                "image_path": "rendered.png",
                "question": "Is object left of object?",
                "world_location": [0.0, 1.0, 2.0],
            }
        }
        with self.assertRaises(AssertionError):
            validate_model_input_isolation(leaked_coords)

        # Leaked ground truth raises AssertionError
        leaked_answer = {
            "model_input": {
                "image_path": "rendered.png",
                "question": "Is object left of object?",
                "ground_truth": "left",
            }
        }
        with self.assertRaises(AssertionError):
            validate_model_input_isolation(leaked_answer)


if __name__ == "__main__":
    unittest.main()
