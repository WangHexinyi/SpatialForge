import json
import threading
import tempfile
import unittest
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from spatialforge.inspector.embodied_session import EmbodiedSession


def _write_fake_episode(root: Path, eid: str):
    d = root / eid
    (d / "frames").mkdir(parents=True)
    # a tiny valid png frame
    img = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    for i in range(3):
        img.save(d / "frames" / f"frame_{i:04d}.png")

    task = {
        "house_id": "procthor-val", "target_category": "mug",
        "target_object_ids": ["Mug|surface|1|1"],
        "target_positions": [[11.0, 0.9, 8.0]],
    }
    priv = {
        "domain": "privileged_research", "_privileged": True,
        "episode_id": eid, "house_id": "procthor-val", "category": "mug",
        "task": task,
        "episode": {"status": "success", "success": True,
                    "success_reason": "target visible", "step": 2},
        "spawn": {"initially_visible": False},
        "teacher_plan": {},
        "steps": [
            {"step": 1, "action_type": "MoveAhead", "action_success": True,
             "agent": {"position": [11.0, 0.9, 6.5], "rotation_yaw_deg": 0,
                       "camera_horizon_deg": 0}, "authoritative_target_visible": False,
             "terminal": False},
            {"step": 2, "action_type": "Done", "action_success": True,
             "agent": {"position": [11.0, 0.9, 7.0], "rotation_yaw_deg": 0,
                       "camera_horizon_deg": 0}, "authoritative_target_visible": True,
             "terminal": True},
        ],
    }
    student = {
        "domain": "student_training", "episode_id": eid, "_leak_checked": "clean",
        "steps": [{"role": "context", "teacher_action": None}],
    }
    (d / "privileged_research_record.json").write_text(json.dumps(priv))
    (d / "student_training_record.json").write_text(json.dumps(student))


class EpisodeInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="ep-insp-"))
        _write_fake_episode(cls.root, "ep-fake1")
        cls.sess = EmbodiedSession("deterministic", seed=0)
        cls.sess.set_episodes_root(cls.root)

    def test_list_and_load(self):
        eps = self.sess.list_episodes()
        self.assertTrue(any(e["episode_id"] == "ep-fake1" for e in eps))
        loaded = self.sess.load_episode("ep-fake1")
        self.assertEqual(loaded["header"]["category"], "mug")
        # timeline = context + 2 actions
        self.assertEqual(len(loaded["timeline"]), 3)
        self.assertEqual(loaded["header"]["student_domain"], "student_training")

    def test_state_and_frame_at_selected_step(self):
        st = self.sess.episode_state(2)  # the Done step
        self.assertTrue(st["terminal"])
        self.assertTrue(st["authoritative_target_visible"])
        self.assertEqual(st["action"], "Done")
        self.assertTrue(st["_privileged_researcher_only"])
        png = self.sess.episode_frame_png(2)
        self.assertEqual(png[:4], b"\x89PNG")

    def test_unsafe_episode_id_rejected(self):
        with self.assertRaises(RuntimeError):
            self.sess.load_episode("../../etc")


class EpisodeInspectionHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="ep-insp-http-"))
        _write_fake_episode(cls.root, "ep-http1")
        from spatialforge.inspector.embodied_server import create_embodied_server
        cls.server = create_embodied_server("127.0.0.1", 0, episodes_root=str(cls.root))
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _get(self, path):
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}").read())

    def _post(self, path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req).read())

    def test_http_episode_flow(self):
        listing = self._get("/api/episodes/list")
        self.assertTrue(any(e["episode_id"] == "ep-http1" for e in listing["episodes"]))
        loaded = self._post("/api/episodes/load", {"episode_id": "ep-http1"})
        self.assertEqual(len(loaded["timeline"]), 3)
        st = self._get("/api/episodes/state?step=2")
        self.assertEqual(st["action"], "Done")
        png = urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/api/episodes/frame?step=2").read()
        self.assertEqual(png[:4], b"\x89PNG")


if __name__ == "__main__":
    unittest.main()
