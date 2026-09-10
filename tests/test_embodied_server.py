import json
import re
import threading
import unittest
import urllib.error
import urllib.request

from spatialforge.inspector.embodied_server import create_embodied_server


def _local_module_imports(js_text: str):
    """Relative ES-module specifiers imported by a browser JS file."""
    out = []
    for m in re.finditer(r'(?:from|import)\s+["\'](\./[^"\']+)["\']', js_text):
        out.append(m.group(1)[2:])  # strip leading "./"
    return out


class EmbodiedServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_embodied_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _post(self, path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            return json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    def _get(self, path):
        return json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}").read()
        )

    def test_i18n_served(self):
        data = self._get("/api/i18n")
        self.assertEqual(data["default"], "bilingual")
        self.assertIn("zh", data["dictionary"])
        self.assertIn("en", data["dictionary"])

    def test_index_served(self):
        r = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/")
        self.assertEqual(r.status, 200)
        html = r.read().decode()
        self.assertIn("sidebar", html)

    def test_workbench_layout_served(self):
        html = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/").read().decode()
        for marker in (
            'id="god-canvas"',
            'id="obs-canvas"',
            'id="ep-timeline"',
            'id="drawer"',
            'id="sidebar"',
            'id="btn-mode-live"',
            'id="btn-mode-replay"',
            'id="d-raw"',
            'id="d-parsed"',
            'id="d-executed"',
            'id="d-verifier"',
            'id="d-terminal"',
            'id="btn-play"',
            'id="btn-prev"',
            'id="btn-next"',
            'id="play-speed"',
            'id="ep-scrub"',
            'id="step-counter"',
        ):
            self.assertIn(marker, html)
        self.assertNotIn('class="step"', html)

    def test_frontend_assets_served(self):
        for path in ("/app.js", "/styles.css"):
            r = urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}")
            self.assertEqual(r.status, 200)

    def test_scene2d_module_served(self):
        r = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/scene2d.js")
        self.assertEqual(r.status, 200)
        self.assertEqual(r.headers.get_content_type(), "application/javascript")
        body = r.read().decode()
        self.assertIn("buildScene2d", body)
        self.assertIn("projectScenePoint", body)

    def test_all_frontend_js_modules_are_served(self):
        """Regression: every ES module in the app.js import graph must be routed.

        A 404 on any module (e.g. the scene2d.js incident) aborts the whole
        module graph, so the UI never initializes and all buttons are dead.
        """
        from pathlib import Path

        web = Path(__file__).resolve().parents[1] / "spatialforge" / "inspector" / "embodied_web"
        app_src = (web / "app.js").read_text(encoding="utf-8")

        seen = set()
        queue = _local_module_imports(app_src)
        while queue:
            rel = queue.pop()
            if rel in seen:
                continue
            seen.add(rel)
            f = web / rel
            if f.is_file():
                queue.extend(_local_module_imports(f.read_text(encoding="utf-8")))

        self.assertIn("god3d.js", seen)
        self.assertIn("scene2d.js", seen)
        self.assertIn("vendor/OrbitControls.js", seen)

        for rel in sorted(seen):
            r = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/{rel}")
            self.assertEqual(r.status, 200, f"{rel} is not served")
            self.assertEqual(
                r.headers.get_content_type(), "application/javascript", rel
            )
            self.assertTrue(r.read(), f"{rel} served empty")

        # `three` is a bare specifier resolved by the importmap in index.html.
        html = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/").read().decode()
        importmap = re.search(r'"three"\s*:\s*"(\./[^"]+)"', html)
        self.assertIsNotNone(importmap, "three importmap entry missing")
        three_rel = importmap.group(1)[2:]
        r = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/{three_rel}")
        self.assertEqual(r.status, 200, f"{three_rel} is not served")
        self.assertTrue(r.read(), f"{three_rel} served empty")

    def test_head_static_assets_return_200(self):
        for path in ("/app.js", "/god3d.js", "/scene2d.js", "/styles.css", "/"):
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.port}{path}", method="HEAD"
            )
            r = urllib.request.urlopen(req)
            self.assertEqual(r.status, 200, f"HEAD {path}")

    def test_house_load_and_start(self):
        h = self._post("/api/house/load", {"backend": "deterministic", "seed": 0})
        self.assertIn("house", h)
        st = self._post(
            "/api/episode/start",
            {"target_category": "mug", "backend": "deterministic", "max_steps": 50},
        )
        self.assertTrue(st["active"])
        self.assertIn("model_input", st)
        self.assertIn("privileged", st)

    def test_step_and_frame(self):
        self._post(
            "/api/episode/start",
            {"target_category": "mug", "backend": "deterministic", "max_steps": 50},
        )
        st = self._post("/api/action/step", {"action_type": "MoveAhead"})
        self.assertEqual(st["step"], 1)
        png = urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/api/frame"
        ).read()
        self.assertEqual(png[:4], b"\x89PNG")

    def test_teacher_endpoint(self):
        self._post(
            "/api/episode/start",
            {"target_category": "mug", "backend": "deterministic", "max_steps": 50},
        )
        t = self._get("/api/teacher")
        self.assertIn("planned", t)
        self.assertIn("_privileged_teacher_only", t)

    def test_done_terminal(self):
        st = self._post(
            "/api/episode/start",
            {"target_category": "mug", "backend": "deterministic", "max_steps": 50},
        )
        st = self._post("/api/action/step", {"action_type": "Done"})
        self.assertTrue(st["terminal"])

    def test_god3d_assets_served(self):
        for path in ("/god3d.js", "/vendor/three.module.js", "/vendor/OrbitControls.js"):
            r = urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}")
            self.assertEqual(r.status, 200)
            self.assertTrue(r.read())

    def test_scene3d_live_endpoint_reports_unavailable_for_cpu_backend(self):
        data = self._get("/api/scene3d")
        self.assertIn("scene", data)
        self.assertIn("unavailable", data)
        self.assertTrue(data["_privileged_researcher_only"])

    def test_session_switch_preserves_episode_roots(self):
        # Regression: a backend/seed switch creates a new EmbodiedSession; it
        # must inherit the resolved episodes roots or the replay selector goes
        # empty after a manual ProcTHOR session.
        from spatialforge.inspector.embodied_server import _Handler

        before = [e["episode_id"] for e in self._get("/api/episodes/list")["episodes"]]
        self._post("/api/house/load", {"backend": "deterministic", "seed": 7})
        self.assertEqual(
            [str(p) for p in _Handler.session.episodes_roots],
            [str(p) for p in _Handler.episodes_roots],
        )
        after = [e["episode_id"] for e in self._get("/api/episodes/list")["episodes"]]
        self.assertEqual(before, after)

    def test_houses_endpoint_lists_real_houses(self):
        data = self._get("/api/houses")
        self.assertIn("houses", data)
        self.assertIsInstance(data["houses"], list)
        for h in data["houses"]:
            self.assertIn("house_id", h)
            self.assertIn("path", h)

    def test_telemetry_ingest_and_read(self):
        posted = self._post("/api/telemetry", {
            "model": {"model_id": "test-model", "backend": "transformers"},
            "runtime": {"env_steps_per_sec": 1.5, "episodes_per_hour": 10.0},
        })
        self.assertTrue(posted["ok"])
        tele = self._get("/api/telemetry")
        self.assertEqual(tele["runtime"]["env_steps_per_sec"], 1.5)
        self.assertEqual(tele["model"]["model_id"], "test-model")
        model = self._get("/api/model")
        self.assertEqual(model["model_id"], "test-model")


class EpisodeScene3DServerTests(unittest.TestCase):
    """3D God View + spawn semantics on a synthetic saved episode."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.tmp.name
        ep = f"{root}/ep-test-0001"
        import os

        os.makedirs(f"{ep}/frames", exist_ok=True)
        with open(f"{ep}/privileged_research_record.json", "w") as f:
            json.dump({
                "schema": "privileged_research_record.v1",
                "domain": "privileged_research",
                "episode_id": "ep-test-0001",
                "task": {"house_id": "house-test", "target_category": "mug",
                         "target_positions": [[1.0, 0.9, 2.0]],
                         "target_object_ids": ["Mug|1|0"]},
                "episode": {"status": "success", "success": True, "steps": 1},
                "spawn": {"position": [3.0, 0.9, 3.0], "yaw": 0.0, "horizon": 0.0,
                          "initially_visible": False},
                "setup": {"initial_pose": {"position": [3.0, 0.9, 3.0],
                                           "rotation_yaw_deg": 0.0, "horizon_deg": 0.0}},
                "teacher_plan": {},
                "steps": [{"step": 1, "action_type": "MoveAhead", "action_origin": "model",
                           "action_success": True,
                           "agent": {"position": [3.0, 0.9, 2.75],
                                     "rotation_yaw_deg": 0.0, "camera_horizon_deg": 0.0},
                           "authoritative_target_visible": True, "terminal": True}],
                "frames": ["ep-test-0001/frames/frame_0000.png"],
            }, f)
        # minimal 1x1 PNG so frame endpoints do not 500
        png = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c626001000000ffff03000006000557bfabd40000000049454e44ae426082"
        )
        with open(f"{ep}/frames/frame_0000.png", "wb") as f:
            f.write(png)
        os.makedirs(f"{root}/houses", exist_ok=True)
        with open(f"{root}/houses/house-test.json", "w") as f:
            json.dump({
                "houseId": "house-test",
                "rooms": [{"id": "r1", "roomType": "Kitchen", "floorPolygon": [
                    {"x": 0, "y": 0, "z": 0}, {"x": 4, "y": 0, "z": 0},
                    {"x": 4, "y": 0, "z": 4}, {"x": 0, "y": 0, "z": 4}]}],
                "walls": [], "doors": [], "windows": [], "objects": [],
                "metadata": {"agent": {"position": {"x": 3, "y": 0.9, "z": 3},
                                       "rotation": {"y": 0}, "horizon": 0}},
            }, f)
        cls.server = create_embodied_server("127.0.0.1", 0, episodes_root=root)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.tmp.cleanup()

    def _post(self, path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req).read())

    def _get(self, path):
        return json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}").read()
        )

    def test_list_and_load_episode(self):
        eps = self._get("/api/episodes/list")["episodes"]
        self.assertTrue(any(e["episode_id"] == "ep-test-0001" for e in eps))
        data = self._post("/api/episodes/load", {"episode_id": "ep-test-0001"})
        self.assertEqual(data["header"]["setup"]["initial_pose"]["position"][0], 3.0)

    def test_episode_scene3d_resolves_house(self):
        self._post("/api/episodes/load", {"episode_id": "ep-test-0001"})
        payload = self._get("/api/episodes/scene3d")
        self.assertIsNotNone(payload["scene"])
        self.assertEqual(payload["scene"]["rooms"][0]["type"], "Kitchen")
        self.assertTrue(payload["_privileged_researcher_only"])
        self.assertEqual(len(payload["episode"]["trace"]), 2)

    def test_episode_semantics_endpoint(self):
        data = self._get("/api/episodes/semantics?episode_id=ep-test-0001")
        self.assertTrue(data["ok"], data["violations"])
        self.assertEqual(data["episode_id"], "ep-test-0001")

    def test_episode_scene3d_carries_canonical_bounds_and_episode_truth(self):
        self._post("/api/episodes/load", {"episode_id": "ep-test-0001"})
        payload = self._get("/api/episodes/scene3d")
        self.assertIn("bounds", payload["scene"])
        for key in ("agent", "targets", "trace", "trace_segments"):
            self.assertIn(key, payload["episode"])
        # 2D and 3D both project these exact world coordinates.
        self.assertEqual(payload["episode"]["targets"][0]["position"], [1.0, 0.9, 2.0])

    def test_episode_scene3d_is_idempotent_across_calls(self):
        # Regression for the 2D disappearing-geometry bug: repeated mode
        # switches refetch the scene; it must be byte-stable, never degrading.
        self._post("/api/episodes/load", {"episode_id": "ep-test-0001"})
        a = self._get("/api/episodes/scene3d")
        b = self._get("/api/episodes/scene3d")
        self.assertEqual(a["scene"], b["scene"])
        self.assertEqual(len(a["scene"]["rooms"]), len(b["scene"]["rooms"]))
        self.assertEqual(len(a["episode"]["trace_segments"]), len(b["episode"]["trace_segments"]))


class CuratedReplayServerTests(unittest.TestCase):
    """Curated replay ordering + real Unity God View endpoint wiring."""

    @classmethod
    def setUpClass(cls):
        import os
        import tempfile

        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.tmp.name

        def write(ep_id, record):
            ep = f"{root}/{ep_id}"
            os.makedirs(f"{ep}/frames", exist_ok=True)
            with open(f"{ep}/privileged_research_record.json", "w") as f:
                json.dump(record, f)

        def agent(pos, yaw=0.0):
            return {"position": list(pos), "rotation_yaw_deg": yaw, "camera_horizon_deg": 0.0}

        valid = {
            "episode_id": "ep-valid-0001",
            "task": {"house_id": "house-test", "target_category": "mug",
                     "target_positions": [[1.0, 0.9, 2.0]], "target_object_ids": ["Mug|1|0"]},
            "episode": {"status": "success", "success": True, "steps": 3},
            "render_quality": "Medium",
            "house_path": f"{root}/houses/house-test.json",
            "spawn": {"position": [3.0, 0.9, 3.0], "yaw": 0.0, "horizon": 0.0},
            "setup": {"initial_pose": {"position": [3.0, 0.9, 3.0],
                                       "rotation_yaw_deg": 0.0, "horizon_deg": 0.0}},
            "steps": [
                {"step": 1, "action_type": "RotateRight", "action_origin": "model",
                 "agent": agent([3.0, 0.9, 3.0], 90.0), "authoritative_target_visible": False},
                {"step": 2, "action_type": "MoveAhead", "action_origin": "model",
                 "agent": agent([3.0, 0.9, 2.75], 90.0), "authoritative_target_visible": False},
                {"step": 3, "action_type": "Done", "action_origin": "model",
                 "agent": agent([3.0, 0.9, 2.75], 90.0), "terminal": True},
            ],
            "frames": [],
        }
        legacy = {
            "episode_id": "ep-legacy-0001",
            "task": {"house_id": "house-test", "target_category": "mug"},
            "episode": {"status": "failure", "success": False, "steps": 2},
            "spawn": {"position": [9.25, 0.9, 8.25], "yaw": 0.0, "horizon": 0.0},
            "setup": {},
            "steps": [
                {"step": 1, "action_type": "RotateRight", "action_origin": "model",
                 "agent": agent([11.5, 0.9, 6.5], 180.0), "authoritative_target_visible": False},
            ],
            "frames": [],
        }
        write("ep-valid-0001", valid)
        write("ep-legacy-0001", legacy)

        os.makedirs(f"{root}/houses", exist_ok=True)
        with open(f"{root}/houses/house-test.json", "w") as f:
            json.dump({
                "houseId": "house-test",
                "rooms": [{"id": "r1", "roomType": "Kitchen", "floorPolygon": [
                    {"x": 0, "y": 0, "z": 0}, {"x": 4, "y": 0, "z": 0},
                    {"x": 4, "y": 0, "z": 4}, {"x": 0, "y": 0, "z": 4}]}],
                "walls": [], "doors": [], "windows": [], "objects": [],
                "metadata": {"agent": {"position": {"x": 3, "y": 0.9, "z": 3},
                                       "rotation": {"y": 0}, "horizon": 0}},
            }, f)

        cls.server = create_embodied_server("127.0.0.1", 0, episodes_root=root)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.tmp.cleanup()

    def _get(self, path):
        return json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}").read()
        )

    def _post(self, path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req).read())

    def test_valid_episode_ranks_above_legacy(self):
        eps = self._get("/api/episodes/list")["episodes"]
        ids = [e["episode_id"] for e in eps]
        self.assertLess(ids.index("ep-valid-0001"), ids.index("ep-legacy-0001"))
        valid = next(e for e in eps if e["episode_id"] == "ep-valid-0001")
        legacy = next(e for e in eps if e["episode_id"] == "ep-legacy-0001")
        self.assertIn("VALID", valid["labels"])
        self.assertTrue(valid["flags"]["researcher_useful"])
        self.assertIn("LEGACY", legacy["labels"])
        self.assertFalse(legacy["flags"]["valid"])

    def test_load_valid_episode_exposes_labels_and_clean_trace(self):
        data = self._post("/api/episodes/load", {"episode_id": "ep-valid-0001"})
        self.assertIn("VALID", data["header"]["labels"])
        self.assertEqual(data["header"]["render_quality"], "Medium")
        payload = self._get("/api/episodes/scene3d")
        self.assertEqual(len(payload["episode"]["trace_segments"]), 1)
        self.assertEqual(payload["episode"]["trace_breaks"], [])

    def test_legacy_episode_trace_is_broken_not_wall_crossing(self):
        data = self._post("/api/episodes/load", {"episode_id": "ep-legacy-0001"})
        self.assertIn("LEGACY", data["header"]["labels"])
        payload = self._get("/api/episodes/scene3d")
        self.assertGreaterEqual(len(payload["episode"]["trace_segments"]), 2)
        self.assertTrue(payload["episode"]["trace_breaks"])

    def test_unity_status_endpoint(self):
        data = self._get("/api/unity/status")
        self.assertIn("available", data)
        self.assertIsInstance(data["available"], bool)

    def test_unity_godview_endpoint_degrades_loudly(self):
        # under the CPU test interpreter ai2thor is absent; the endpoint must
        # return a clear error, never a fabricated image.
        url = f"http://127.0.0.1:{self.port}/api/episodes/unity_godview?episode_id=ep-valid-0001&step=1"
        try:
            data = json.loads(urllib.request.urlopen(url).read())
            self.assertIn("error", data)
        except urllib.error.HTTPError as e:
            data = json.loads(e.read())
            self.assertIn("error", data)

    def test_live_unity_godview_degrades_loudly(self):
        # no live episode is running; the endpoint must return an error, not a
        # fabricated frame, and must not silently fall back.
        url = f"http://127.0.0.1:{self.port}/api/unity/live_godview"
        try:
            data = json.loads(urllib.request.urlopen(url).read())
            self.assertIn("error", data)
        except urllib.error.HTTPError as e:
            data = json.loads(e.read())
            self.assertIn("error", data)

    def test_unity_prewarm_endpoint(self):
        data = self._get("/api/unity/prewarm?episode_id=ep-valid-0001")
        self.assertTrue(data.get("prewarming"), data)

    def test_unity_godview_accepts_explicit_camera_query(self):
        # explicit camera params must be parsed without breaking the endpoint;
        # the frame still fails loudly here because ai2thor is absent.
        url = (f"http://127.0.0.1:{self.port}/api/episodes/unity_godview"
               "?episode_id=ep-valid-0001&step=1&cam_mode=free"
               "&px=1&py=2&pz=3&rx=10&ry=20&rz=0&fov=70")
        try:
            data = json.loads(urllib.request.urlopen(url).read())
            self.assertIn("error", data)
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 503)
            self.assertIn("error", json.loads(e.read()))


class FrontendUnitySourceTests(unittest.TestCase):
    """Regression guards for the Unity God View black-screen P0.

    These assert the *shipped* frontend no longer contains the failure pattern
    (hide the <img> immediately after assigning src, and only hide the overlay
    on load) and that a failed request surfaces its real cause.
    """

    @classmethod
    def setUpClass(cls):
        from pathlib import Path

        cls.src = (
            Path(__file__).resolve().parents[1]
            / "spatialforge" / "inspector" / "embodied_web" / "app.js"
        ).read_text(encoding="utf-8")

    def test_black_screen_pattern_is_gone(self):
        self.assertNotIn("function showUnityStatus", self.src)
        self.assertIn("function setUnityOverlay", self.src)
        self.assertIn("function loadUnityFrame", self.src)

    def test_successful_frame_is_actually_shown(self):
        # on load the image must be un-hidden (the old code only hid the overlay)
        self.assertIn('img.classList.remove("hidden")', self.src)
        self.assertIn('setUnityOverlay("hidden")', self.src)

    def test_failure_is_loud_not_silent(self):
        self.assertIn("god.unity_unavailable", self.src)
        self.assertIn('let detail = "HTTP " + res.status', self.src)
        self.assertIn("res.ok", self.src)

    def test_latest_request_wins(self):
        self.assertIn("UNITY_LOAD_GEN", self.src)
        self.assertIn("gen !== UNITY_LOAD_GEN", self.src)

    def test_replay_step_switching_is_coalesced_and_cached(self):
        self.assertIn("function queueScrub", self.src)
        self.assertIn("FRAME_CACHE_MAX", self.src)
        self.assertIn("PERF.replayCached", self.src)

    def test_interactive_camera_is_local_and_coalesced(self):
        for token in (
            "UNITY_INTERACTIVE",
            "unityScheduleRender",
            "unityEndInteraction",
            "unityViewportSize",
            "setUnityIndicator",
            "requestAnimationFrame",
        ):
            self.assertIn(token, self.src, token)

    def test_preview_and_settle_frames_are_requested(self):
        self.assertIn('"q=" + (UNITY_INTERACTIVE ? "preview" : "full")', self.src)
        self.assertIn("god-unity-indicator", self.src)
        # a visible frame must never be covered by the full-screen loading panel
        self.assertIn("const showing = !!(img && img.dataset.src)", self.src)


if __name__ == "__main__":
    unittest.main()
