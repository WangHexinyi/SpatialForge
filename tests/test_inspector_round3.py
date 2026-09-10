"""Round 3 regression tests: Unity/3D camera control, 2D lifecycle, picker, manual.

These are a mix of backend unit tests and source-level guards. The source guards
assert that the *shipped* browser code wires the interactions that a headless
test cannot click; the backend tests cover camera-pose parsing, house
resolution and loud degradation.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "spatialforge" / "inspector" / "embodied_web"


class CameraQueryTests(unittest.TestCase):
    def test_camera_from_query_parses_full_pose(self):
        from spatialforge.inspector.embodied_server import _camera_from_query

        cam = _camera_from_query({
            "cam_mode": "free",
            "px": "1", "py": "2", "pz": "3",
            "rx": "10", "ry": "20", "rz": "0",
            "fov": "70", "tx": "1", "ty": "0", "tz": "2", "dist": "4",
        })
        self.assertEqual(cam["mode"], "free")
        self.assertEqual(cam["position"], {"x": 1.0, "y": 2.0, "z": 3.0})
        self.assertEqual(cam["rotation"], {"x": 10.0, "y": 20.0, "z": 0.0})
        self.assertEqual(cam["fieldOfView"], 70.0)

    def test_camera_from_query_rejects_incomplete_pose(self):
        from spatialforge.inspector.embodied_server import _camera_from_query

        self.assertIsNone(_camera_from_query({"cam_mode": "free"}))
        self.assertIsNone(_camera_from_query({"px": "1", "py": "2", "pz": "3"}))
        self.assertIsNone(_camera_from_query({"px": "1"}))

    def test_camera_from_query_absent_without_mode(self):
        from spatialforge.inspector.embodied_server import _camera_from_query

        self.assertIsNone(_camera_from_query({"px": "1", "py": "2", "pz": "3"}))


class ExplicitCameraTests(unittest.TestCase):
    def test_explicit_camera_validates_and_coerces(self):
        from spatialforge.inspector.god_camera import GodCameraRenderer

        cam = GodCameraRenderer._explicit_camera({
            "position": {"x": 1, "y": 2, "z": 3},
            "rotation": {"x": 4, "y": 5, "z": 6},
            "fieldOfView": 80,
        })
        self.assertEqual(cam["position"], {"x": 1.0, "y": 2.0, "z": 3.0})
        self.assertEqual(cam["fieldOfView"], 80.0)
        self.assertIsNone(GodCameraRenderer._explicit_camera({"position": {"x": 1}}))
        self.assertIsNone(GodCameraRenderer._explicit_camera(None))

    def test_pose_key_detects_change(self):
        from spatialforge.inspector.god_camera import GodCameraRenderer

        a = {"position": [1.0, 0.9, 2.0], "rotation_yaw_deg": 10.0, "camera_horizon_deg": 0.0}
        b = {"position": [1.0, 0.9, 2.0], "rotation_yaw_deg": 10.0, "camera_horizon_deg": 0.0}
        c = {"position": [1.1, 0.9, 2.0], "rotation_yaw_deg": 10.0, "camera_horizon_deg": 0.0}
        self.assertEqual(GodCameraRenderer._pose_key(a), GodCameraRenderer._pose_key(b))
        self.assertNotEqual(GodCameraRenderer._pose_key(a), GodCameraRenderer._pose_key(c))


class HouseResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from spatialforge.inspector.embodied_session import EmbodiedSession

        cls.s = EmbodiedSession("deterministic", 0)
        cls.s.set_episodes_roots([str(REPO / "outputs")])

    def test_resolve_house_path_from_stem(self):
        p = self.s.resolve_house_path("house_0007")
        self.assertIsNotNone(p, "real ProcTHOR house fixture should resolve")
        self.assertTrue(p.endswith("house_0007.json"))

    def test_list_houses_includes_real_houses(self):
        houses = self.s.list_houses()
        self.assertTrue(houses)
        self.assertTrue(any(h["house_id"] == "house_0007" for h in houses))

    def test_live_unity_requires_active_episode(self):
        with self.assertRaises(RuntimeError):
            self.s.live_unity_png()


class FrontendRound3SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (WEB / "app.js").read_text(encoding="utf-8")
        cls.god3d = (WEB / "god3d.js").read_text(encoding="utf-8")
        cls.html = (WEB / "index.html").read_text(encoding="utf-8")

    # P0-A Unity real camera control
    def test_unity_mouse_camera_is_wired(self):
        for token in ("bindUnityCamera", "unityOrbit", "unityPan",
                      "unityComputePose", "unityFrameUrl", "UNITY_CAM"):
            self.assertIn(token, self.app, token)
        self.assertIn("cam_mode=", self.app)
        self.assertIn("setPointerCapture", self.app)
        self.assertIn('addEventListener("wheel"', self.app)

    def test_unity_camera_state_is_explicit(self):
        for field in ("mode:", "preset:", "target:", "yaw:", "pitch:", "distance:", "position:", "rotation:"):
            self.assertIn(field, self.app, field)

    def test_follow_does_not_steal_free_camera(self):
        self.assertIn("unityFollowStep", self.app)
        self.assertIn('if (UNITY_CAM.mode !== "follow") return;', self.app)

    def test_unity_render_is_coalesced_single_flight(self):
        self.assertIn("UNITY_RENDER_INFLIGHT", self.app)
        self.assertIn("UNITY_RENDER_PENDING", self.app)
        self.assertIn("_fetchUnityFrame", self.app)

    def test_unity_engine_prewarm_wired(self):
        self.assertIn("/api/unity/prewarm", self.app)

    # P0-C Three.js OrbitControls
    def test_3d_controls_configured(self):
        for token in ("configureControls", "controls.enabled = true",
                      "mouseButtons", "cameraState", "setUserInteractHandler"):
            self.assertIn(token, self.god3d, token)
        self.assertIn('controls.addEventListener("start"', self.god3d)
        self.assertIn("followAgent = false;", self.god3d)

    # P0-D 2D lifecycle
    def test_2d_scene_cache_and_race_guard(self):
        for token in ("SCENE3D_CACHE", "SCENE3D_INFLIGHT", "SCENE3D_GEN",
                      "currentSceneKey", "god.scene_unavailable"):
            self.assertIn(token, self.app, token)

    def test_2d_exposes_explicit_unavailable_banner(self):
        self.assertIn("sceneMissing", self.app)
        self.assertIn("scene_expected", self.app)

    # P0-E episode picker
    def test_episode_picker_is_custom_not_native_select(self):
        self.assertIn('id="ep-select-btn"', self.html)
        self.assertIn('id="ep-select-menu"', self.html)
        self.assertNotIn('id="ep-select"', self.html)
        self.assertIn("renderEpisodePicker", self.app)
        self.assertIn("openEpisodePicker", self.app)

    # P1 manual control
    def test_manual_control_wired(self):
        for token in ("refreshHouses", "house_id", "live_godview",
                      "manual-state", "refreshHouses()"):
            self.assertIn(token, self.app, token)

    def test_unity_tools_have_reset(self):
        self.assertIn('data-uv="reset"', self.html)


if __name__ == "__main__":
    unittest.main()
