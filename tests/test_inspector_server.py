"""Unit and integration tests for Inspector HTTP server, artifact access, and prediction matching."""

import json
from pathlib import Path
import threading
import unittest
import urllib.request
import urllib.error

from spatialforge.experiment.training import (
    PROFILE_BALANCED,
    PROFILE_MAX_PERFORMANCE,
    PROFILE_REFERENCE,
)
from spatialforge.inspector.artifacts import ArtifactRepository, resolve_safe_path
from spatialforge.inspector.contracts import SCHEMA_VERSION
from spatialforge.inspector.server import create_inspector_server


class TestInspectorServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start test server on localhost ephemeral port
        cls.server = create_inspector_server(host="127.0.0.1", port=0)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        # Direct local opener bypassing any environment proxy
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get_json(self, path: str):
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url)
        with self.opener.open(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, path: str, payload: dict):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode("utf-8"))

    def test_list_scenes_and_scene_detail(self):
        """Verify /api/scenes and /api/scene/<scene_id>."""
        data = self._get_json("/api/scenes")
        self.assertIn("scenes", data)
        self.assertIn("scene_000", data["scenes"])

        detail = self._get_json("/api/scene/scene_000")
        self.assertEqual(detail["scene_id"], "scene_000")
        self.assertIn("views", detail)
        self.assertIn("south", detail["views"])
        self.assertIn("objects", detail)

    def test_samples_endpoint(self):
        """Verify /api/samples query."""
        data = self._get_json("/api/samples?scene_id=scene_000&view_id=south")
        self.assertIn("samples", data)
        self.assertGreater(data["count"], 0)
        sample = data["samples"][0]
        self.assertEqual(sample["scene_id"], "scene_000")
        self.assertEqual(sample["view_id"], "south")

    def test_snapshot_endpoint(self):
        """Verify /api/snapshot returns full versioned InspectorSnapshot."""
        snap = self._get_json("/api/snapshot?scene_id=scene_000&view_id=south")
        self.assertEqual(snap["schema_version"], SCHEMA_VERSION)
        self.assertEqual(snap["identity"]["scene_id"], "scene_000")
        self.assertEqual(snap["identity"]["view_id"], "south")
        self.assertIn("model_input", snap)
        self.assertIn("image_path", snap["model_input"])
        self.assertIn("question", snap["model_input"])
        # Crucial: verify no coordinates in model_input
        self.assertNotIn("world_location", snap["model_input"])
        self.assertNotIn("camera_depth", snap["model_input"])
        self.assertNotIn("ground_truth", snap["model_input"])

    def test_profiles_endpoint(self):
        """Verify all 3 authoritative profiles are exposed."""
        data = self._get_json("/api/profiles")
        self.assertIn("profiles", data)
        p_ids = [p["profile_id"] for p in data["profiles"]]
        self.assertIn(PROFILE_REFERENCE, p_ids)
        self.assertIn(PROFILE_BALANCED, p_ids)
        self.assertIn(PROFILE_MAX_PERFORMANCE, p_ids)

        for p in data["profiles"]:
            self.assertEqual(p["microbatch_size"] * p["gradient_accumulation_steps"], 8)
            self.assertEqual(p["effective_batch_size"], 8)
            self.assertFalse(p["gradient_checkpointing"])
            self.assertTrue(p["use_vision_cache"])

    def test_profile_capability_validation_preflight(self):
        """Capability validation rejects insufficient VRAM with NO SILENT FALLBACK."""
        # 1. Force available memory to 22000 MB (enough for Reference, but not Max Performance ~30.9G)
        res = self._get_json(f"/api/profiles/validate?profile={PROFILE_MAX_PERFORMANCE}&available_memory_mb=22000")
        self.assertEqual(res["status"], "incompatible")
        self.assertEqual(res["requested_profile"], PROFILE_MAX_PERFORMANCE)
        self.assertEqual(res["recommended_profile"], PROFILE_REFERENCE)
        self.assertIn("cannot safely start", res["message"])

        # 2. Force available memory to 35000 MB (enough for Max Performance)
        res_ok = self._get_json(f"/api/profiles/validate?profile={PROFILE_MAX_PERFORMANCE}&available_memory_mb=35000")
        self.assertEqual(res_ok["status"], "compatible")

    def test_image_serving_and_traversal_rejection(self):
        """Images are served correctly and directory traversal is rejected."""
        # Valid image
        img_url = f"{self.base_url}/api/image?path=outputs/experiments/g2.0-e/rendered/scene_000/south.png"
        req = urllib.request.Request(img_url)
        with self.opener.open(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Content-Type"), "image/png")
            data = resp.read()
            self.assertGreater(len(data), 1000)

        # Directory traversal attempt
        traversal_url = f"{self.base_url}/api/image?path=../../etc/passwd"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.opener.open(traversal_url, timeout=5)
        self.assertEqual(ctx.exception.code, 403)

        # Disallowed extension
        py_url = f"{self.base_url}/api/image?path=scripts/render_curriculum.py"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.opener.open(py_url, timeout=5)
        self.assertEqual(ctx.exception.code, 403)

    def test_prediction_matching_and_alignment_error(self):
        """Verify prediction matching and alignment conflict detection."""
        repo = ArtifactRepository()

        # 1. Valid prediction on scene_080 east view
        sample_id = "scene_080:single:east:0-1:depth:order_1"
        pred_valid = repo.match_prediction(
            sample_id=sample_id,
            expected_scene_id="scene_080",
            expected_view_id="east",
            expected_ground_truth="front",
            run_id="g2.0-e3/group_a/seed_42",
        )
        self.assertEqual(pred_valid.status, "available")
        self.assertEqual(pred_valid.raw_prediction, "front")
        self.assertEqual(pred_valid.parsed_prediction, "front")
        self.assertTrue(pred_valid.is_correct)

        # 2. Scene mismatch: expected scene_081 instead of scene_080 -> alignment error
        pred_mismatch = repo.match_prediction(
            sample_id=sample_id,
            expected_scene_id="scene_081",
            expected_view_id="east",
            expected_ground_truth="front",
            run_id="g2.0-e3/group_a/seed_42",
        )
        self.assertEqual(pred_mismatch.status, "alignment_error")
        self.assertFalse(pred_mismatch.is_correct)
        self.assertIn("scene_id mismatch", pred_mismatch.error_message)

        # 3. Missing prediction
        pred_missing = repo.match_prediction(
            sample_id="non_existent_sample_xyz",
            expected_scene_id="scene_000",
            expected_view_id="south",
            expected_ground_truth="front",
            run_id="g2.0-e3/group_a/seed_42",
        )
        self.assertEqual(pred_missing.status, "unavailable")

    def test_telemetry_endpoint(self):
        """Verify /api/telemetry returns telemetry."""
        telem = self._get_json("/api/telemetry")
        self.assertIn("freshness", telem)
        self.assertIn("telemetry", telem)

    def test_trajectory_endpoint(self):
        """Verify /api/trajectory returns deterministic trajectory and diagnostics."""
        data = self._get_json("/api/trajectory?scene_id=scene_000&framing_mode=balanced&frame_count=30&seed=42")
        self.assertIn("trajectory_id", data)
        self.assertEqual(data["scene_id"], "scene_000")
        self.assertIn("framing", data)
        self.assertIn("scene_center", data["framing"])
        self.assertIn("camera_distance", data["framing"])
        self.assertIn("presets_comparison", data["framing"])
        self.assertIn("frames", data)
        self.assertEqual(len(data["frames"]), 30)
        self.assertIn("diagnostics", data)
        self.assertEqual(data["diagnostics"]["frame_count"], 30)
        self.assertGreater(data["diagnostics"]["trajectory_path_length"], 0.0)

    def test_challenge_endpoint(self):
        """Verify /api/challenge and /api/challenge_summary return difficulty axes metadata."""
        # 1. Legacy scene computed on the fly
        data_s0 = self._get_json("/api/challenge?scene_id=scene_000")
        self.assertEqual(data_s0["difficulty_tier"], "S0")
        self.assertEqual(data_s0["object_count"], 4)
        self.assertIn("clutter_score", data_s0)
        self.assertIn("depth_layers", data_s0)
        self.assertIn("packing_summary", data_s0)

        # 2. Challenge scene with precomputed/persisted metadata
        data_s2 = self._get_json("/api/challenge_summary?scene_id=scene_challenge_s2_000")
        self.assertEqual(data_s2["difficulty_tier"], "S2")
        self.assertGreaterEqual(data_s2["object_count"], 8)
        self.assertEqual(data_s2["depth_layer_count"], 3)
        self.assertGreaterEqual(data_s2["distractor_pair_count"], 4)
        self.assertGreater(data_s2["clutter_score"], data_s0["clutter_score"])

    def test_view_sets_endpoint(self):
        """Verify /api/view_sets exposes Historical 8, Canonical 6, 14, 26 and all 26 poses."""
        data = self._get_json("/api/view_sets?scene_id=scene_000")
        self.assertIn("view_sets", data)
        self.assertIn("canonical_poses", data)

        sets_map = {vs["id"]: vs for vs in data["view_sets"]}
        self.assertIn("historical_8", sets_map)
        self.assertIn("canonical_6", sets_map)
        self.assertIn("canonical_14", sets_map)
        self.assertIn("canonical_26", sets_map)

        self.assertEqual(sets_map["historical_8"]["count"], 8)
        self.assertEqual(sets_map["canonical_6"]["count"], 6)
        self.assertEqual(sets_map["canonical_14"]["count"], 14)
        self.assertEqual(sets_map["canonical_26"]["count"], 26)

        # Verify canonical poses
        poses = data["canonical_poses"]
        self.assertEqual(len(poses), 26)
        for view_key, pose in poses.items():
            self.assertIn("position", pose)
            self.assertIn("look_at", pose)
            self.assertIn("up", pose)
            self.assertIn("direction_id", pose)
            self.assertIn("name", pose)
            # Coordinates are valid numbers
            self.assertEqual(len(pose["position"]), 3)

    def test_snapshot_manual_distance(self):
        """Verify /api/snapshot respects manual distance parameter."""
        snap = self._get_json("/api/snapshot?scene_id=scene_000&view_id=south&distance=3.5")
        pose = snap["camera"]["pose"]
        p = pose["position"]
        l = pose["look_at"]
        dx, dy, dz = p[0] - l[0], p[1] - l[1], p[2] - l[2]
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        self.assertAlmostEqual(dist, 3.5, places=4)

    def test_snapshot_unrendered_canonical_preview_projection(self):
        """Verify unrendered canonical view yields preview_generated projection status without error."""
        # view_15 is one of the 26 canonical views not present in historical 8
        snap = self._get_json("/api/snapshot?scene_id=scene_000&view_id=view_15")
        self.assertEqual(snap["identity"]["view_id"], "view_15")
        proj = snap["camera"]["projection"]
        self.assertEqual(proj["status"], "preview_generated")
        self.assertTrue(proj["is_frustum_available"])

    def test_trajectory_custom_waypoints_post(self):
        """Verify POST /api/trajectory accepts custom waypoints JSON and generates smooth curve."""
        wps = [
            {"x": 3.0, "y": -3.0, "z": 2.0},
            {"x": 0.0, "y": -4.0, "z": 2.5},
            {"x": -3.0, "y": -3.0, "z": 2.0},
        ]
        payload = {
            "scene_id": "scene_000",
            "trajectory_family": "custom_waypoints",
            "frame_count": 30,
            "speed_m_s": 1.5,
            "waypoints_json": json.dumps(wps),
        }
        res = self._post_json("/api/trajectory", payload)
        self.assertEqual(res["trajectory_family"], "custom_waypoints")
        self.assertEqual(len(res["frames"]), 30)

        p_start = res["frames"][0]["pose"]["position"]
        p_end = res["frames"][-1]["pose"]["position"]
        self.assertAlmostEqual(p_start[0], 3.0, places=3)
        self.assertAlmostEqual(p_start[1], -3.0, places=3)
        self.assertAlmostEqual(p_start[2], 2.0, places=3)

        self.assertAlmostEqual(p_end[0], -3.0, places=3)
        self.assertAlmostEqual(p_end[1], -3.0, places=3)
        self.assertAlmostEqual(p_end[2], 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
