import unittest
from pathlib import Path

from spatialforge.inspector.god_camera import GodCameraRenderer

REPO = Path(__file__).resolve().parents[1]


class GodCameraTests(unittest.TestCase):
    def test_unavailable_reason_for_missing_house(self):
        cam = GodCameraRenderer("/definitely/not/a/house.json")
        reason = cam.unavailable_reason()
        self.assertIsNotNone(reason)
        self.assertTrue("not found" in reason or "ai2thor" in reason)

    def test_overview_camera_frames_house_center(self):
        cam = GodCameraRenderer("/tmp/whatever.json")
        cam._bounds = {"center": [5.0, 6.0], "span": [10.0, 8.0]}
        view = cam._camera_for("overview", None)
        self.assertAlmostEqual(view["position"]["x"], 5.0)
        self.assertAlmostEqual(view["position"]["z"], 5.9)
        self.assertGreater(view["position"]["y"], 0.0)
        self.assertGreater(view["rotation"]["x"], 0.0)

    def test_follow_camera_sits_behind_agent(self):
        cam = GodCameraRenderer("/tmp/whatever.json")
        cam._bounds = {"center": [0.0, 0.0], "span": [4.0, 4.0]}
        pose = {"position": [1.0, 0.9, 1.0], "rotation_yaw_deg": 0.0, "camera_horizon_deg": 0.0}
        view = cam._camera_for("follow", pose)
        # yaw 0 -> forward is +z, so camera is at z - distance behind the agent
        self.assertLess(view["position"]["z"], pose["position"][2])
        self.assertGreater(view["position"]["y"], pose["position"][1])
        self.assertAlmostEqual(view["rotation"]["y"], 0.0)


class GodViewBackgroundTests(unittest.TestCase):
    """The God View clears the Procedural city skybox per third-party camera.

    The AI2-THOR ``Procedural`` scene carries an aerial city photograph through
    Unity's skybox; ``AddThirdPartyCamera`` defaults to Skybox clear flags. The
    fix passes ``skyboxColor`` on both Add and Update so only the researcher
    camera gets a solid neutral background. The first-person camera and the
    scene skybox/lighting are never touched.
    """

    @staticmethod
    def _renderer_with_fake_engine():
        import numpy as np

        class FakeEvent:
            def __init__(self):
                self.metadata = {"lastActionSuccess": True}
                self.third_party_camera_frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        class FakeController:
            def __init__(self):
                self.actions = []

            def step(self, action):
                self.actions.append(dict(action))
                return FakeEvent()

        class FakeBackend:
            _house_data = {
                "rooms": [{"floorPolygon": [
                    {"x": 0, "y": 0, "z": 0}, {"x": 4, "y": 0, "z": 4},
                ]}]
            }

            def __init__(self):
                self.controller = FakeController()

            def set_agent_pose(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("God View must not teleport without a pose")

        backend = FakeBackend()
        return GodCameraRenderer("/tmp/house.json", backend=backend), backend

    def test_neutral_background_color_is_the_documented_near_black(self):
        from spatialforge.inspector.god_camera import GOD_VIEW_SKYBOX_COLOR

        self.assertEqual(GOD_VIEW_SKYBOX_COLOR, "#101418")

    def test_add_and_update_third_party_camera_carry_skybox_color(self):
        from spatialforge.inspector.god_camera import GOD_VIEW_SKYBOX_COLOR

        cam, backend = self._renderer_with_fake_engine()
        frame = cam.render(agent_pose=None, view="overview", teleport=False)
        # God View always ships JPEG (PNG encode dominates the frame budget).
        self.assertTrue(frame.startswith(b"\xff\xd8"))

        first = backend.controller.actions[0]
        self.assertEqual(first["action"], "AddThirdPartyCamera")
        self.assertEqual(first["skyboxColor"], GOD_VIEW_SKYBOX_COLOR)

        cam.render(agent_pose=None, view="overview", teleport=False)
        second = backend.controller.actions[1]
        self.assertEqual(second["action"], "UpdateThirdPartyCamera")
        self.assertEqual(second["skyboxColor"], GOD_VIEW_SKYBOX_COLOR)

    def test_render_supports_png_and_jpeg(self):
        cam, _ = self._renderer_with_fake_engine()
        jpeg = cam.render(agent_pose=None, view="overview", teleport=False, image_format="jpeg")
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        cam2, _ = self._renderer_with_fake_engine()
        png = cam2.render(agent_pose=None, view="overview", teleport=False, image_format="png")
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_owned_engine_resizes_but_shared_backend_never_does(self):
        from spatialforge.inspector.god_camera import GodCameraRenderer

        class FakeEvent:
            metadata = {"lastActionSuccess": True}

        class FakeController:
            def __init__(self):
                self.actions = []

            def step(self, action):
                self.actions.append(dict(action))
                return FakeEvent()

        owned = GodCameraRenderer("/tmp/h.json")
        owned._backend = type("B", (), {"controller": FakeController()})()
        owned._render_size = (640, 480)
        owned._owns_backend = True
        owned._ensure_resolution(512, 384)
        self.assertEqual(owned._render_size, (512, 384))
        action = owned._backend.controller.actions[0]
        self.assertEqual(action["action"], "ChangeResolution")
        self.assertEqual((action["x"], action["y"]), (512, 384))

        shared = GodCameraRenderer("/tmp/h.json")
        shared._backend = type("B", (), {"controller": FakeController()})()
        shared._render_size = (256, 256)
        shared._owns_backend = False
        shared._ensure_resolution(1280, 960)
        self.assertEqual(shared._render_size, (256, 256))
        self.assertEqual(shared._backend.controller.actions, [])

    def test_godview_selects_preview_and_full_resolutions(self):
        from spatialforge.inspector.embodied_session import EmbodiedSession

        calls = []

        class FakeCam:
            def render(self, **kwargs):
                calls.append(kwargs)
                return b"\xff\xd8jpeg"

        session = EmbodiedSession("deterministic", 0)
        session._loaded_episode = {
            "episode_id": "ep", "header": {}, "timeline": [{"agent": None}],
        }
        session._resolve_episode_house_path = lambda *a, **k: "/tmp/house.json"
        session._god_camera_for = lambda hp: FakeCam()

        session.unity_godview_png(episode_id="ep", quality="preview")
        self.assertEqual(calls[-1]["resolution"], (512, 384))
        self.assertEqual(calls[-1]["image_format"], "jpeg")

        session.unity_godview_png(episode_id="ep", quality="full", width=1000, height=700)
        self.assertEqual(calls[-1]["resolution"], (1000, 700))

        session.unity_godview_png(episode_id="ep", quality="full", width=99999, height=99999)
        self.assertEqual(calls[-1]["resolution"], (1600, 1200))

    def test_only_third_party_camera_actions_are_used(self):
        """No global skybox / FPV-camera action may be issued by the God View."""
        cam, backend = self._renderer_with_fake_engine()
        cam.render(agent_pose=None, view="overview", teleport=False)
        cam.render(agent_pose=None, view="overview", teleport=False)
        actions = [a["action"] for a in backend.controller.actions]
        self.assertEqual(actions, ["AddThirdPartyCamera", "UpdateThirdPartyCamera"])
        source = (REPO / "spatialforge" / "inspector" / "god_camera.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("ToggleMapView", "ToggleCeiling", "SetSkybox", "ChangeSkybox"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
