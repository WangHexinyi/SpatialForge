import json
import threading
import unittest
import urllib.error
import urllib.request

from spatialforge.inspector.embodied_server import create_embodied_server


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


if __name__ == "__main__":
    unittest.main()
