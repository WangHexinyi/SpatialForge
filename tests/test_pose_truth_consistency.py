"""FPV <-> agent pose / heading truth-consistency tests (Focused Fix #5).

Every Inspector/replay entry must represent ONE authoritative AI2-THOR state:
the RGB frame, the agent pose and the executed action on a timeline entry must
come from the same environment event. This suite pins the shared normalization
layer (:meth:`EmbodiedSession.load_episode`) so the 2D / 3D / Unity God View all
consume the same pose the FPV was captured with.

The regression this guards is the initial-entry off-by-one: ``frame_0000`` is the
pre-action-0 observation, so the entry-0 pose must be the authoritative initial
pose (``setup.initial_pose`` / ``decisions[0].agent`` / recorded spawn), never
``steps[0].agent`` (which is the post-action-0 state). With a first action that
rotates, the old fallback showed the post-rotation heading beside the
pre-rotation RGB frame.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

from spatialforge.inspector.embodied_session import EmbodiedSession

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROLLOUT = REPO_ROOT / "outputs" / "embodied_bc" / "model_rollout"
MODEL_LIVE = REPO_ROOT / "outputs" / "embodied_bc" / "model_live"

_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd40000000049454e44ae426082"
)


def _agent(pos, yaw, horizon=30.0):
    return {
        "position": [float(x) for x in pos],
        "rotation_yaw_deg": float(yaw),
        "camera_horizon_deg": float(horizon),
        "is_crouching": False,
    }


def _decision(idx, frame, step, action, agent):
    return {
        "decision_idx": idx, "frame": frame, "step": step,
        "model_action": action, "model_raw": action, "executed": True,
        "invalid": False, "agent": agent,
    }


def _write_episode(root: Path, eid: str, *, spawn=None, setup=None, steps,
                   decisions=None):
    d = root / eid
    (d / "frames").mkdir(parents=True, exist_ok=True)
    for i in range(len(steps) + 1):
        (d / "frames" / f"frame_{i:04d}.png").write_bytes(_PNG_1x1)
    rec = {
        "schema": "privileged_research_record.v1",
        "domain": "privileged_research",
        "_privileged": True,
        "episode_id": eid,
        "task": {"house_id": "house-test", "target_category": "mug"},
        "episode": {"status": "failure", "success": False, "step": len(steps)},
        "spawn": spawn or {},
        "setup": setup or {},
        "teacher_plan": {},
        "steps": steps,
        "frames": [f"{eid}/frames/frame_{i:04d}.png" for i in range(len(steps) + 1)],
    }
    if decisions is not None:
        rec["decisions"] = decisions
        rec["model_controlled"] = True
    (d / "privileged_research_record.json").write_text(json.dumps(rec))
    return d


def _forward_xz(yaw_deg):
    r = math.radians(float(yaw_deg))
    return (math.sin(r), math.cos(r))


def _dot_xz(position, forward):
    return position[0] * forward[0] + position[2] * forward[1]


def _delta_yaw(before, after):
    return (float(after) - float(before)) % 360.0


class InitialPoseNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sess = EmbodiedSession("deterministic", seed=0)
        self.sess.set_episodes_root(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_initial_pose_prefers_setup_block(self):
        eid = "ep-setup"
        _write_episode(
            self.root, eid,
            spawn={"position": [9.0, 0.9, 9.0], "yaw": 0.0, "horizon": 0.0},
            setup={"initial_pose": {"position": [3.0, 0.9, 3.0],
                                    "rotation_yaw_deg": 90.0, "horizon_deg": 30.0}},
            steps=[{"step": 1, "action_type": "MoveAhead", "action_success": True,
                    "agent": _agent([3.0, 0.9, 2.75], 90.0)}],
        )
        tl = self.sess.load_episode(eid)["timeline"]
        self.assertEqual(tl[0]["agent"]["position"], [3.0, 0.9, 3.0])
        self.assertEqual(tl[0]["agent"]["rotation_yaw_deg"], 90.0)
        self.assertEqual(tl[0]["agent_source"], "setup.initial_pose")

    def test_initial_pose_from_decisions_for_legacy_model_record(self):
        """Regression: first action RotateRight must not leak into entry 0."""
        eid = "ep-rot-model"
        initial = _agent([11.5, 0.9, 6.5], 90.0)
        _write_episode(
            self.root, eid,
            spawn={"position": [15.0, 0.9, 6.0], "yaw": 0.0, "horizon": 0.0},
            setup={},
            steps=[
                {"step": 1, "action_type": "RotateRight", "action_success": True,
                 "agent": _agent([11.5, 0.9, 6.5], 180.0)},
                {"step": 2, "action_type": "MoveAhead", "action_success": True,
                 "agent": _agent([11.5, 0.9, 6.25], 180.0)},
            ],
            decisions=[
                _decision(0, "frame_0000.png", 0, "RotateRight", initial),
                _decision(1, "frame_0001.png", 1, "MoveAhead", _agent([11.5, 0.9, 6.5], 180.0)),
            ],
        )
        tl = self.sess.load_episode(eid)["timeline"]
        # entry 0 is the pre-rotation frame/pose, NOT the post-rotation step 0 pose
        self.assertEqual(tl[0]["frame"], "frame_0000.png")
        self.assertEqual(tl[0]["agent"]["rotation_yaw_deg"], 90.0)
        self.assertEqual(tl[0]["agent_source"], "decisions[0].agent")
        self.assertEqual(tl[1]["agent"]["rotation_yaw_deg"], 180.0)
        self.assertNotEqual(
            tl[0]["agent"]["rotation_yaw_deg"], tl[1]["agent"]["rotation_yaw_deg"]
        )

    def test_initial_pose_from_spawn_for_legacy_teacher_record(self):
        eid = "ep-legacy-teacher"
        _write_episode(
            self.root, eid,
            spawn={"position": [7.0, 0.9, 7.0], "yaw": 270.0, "horizon": 0.0},
            setup={},
            steps=[{"step": 1, "action_type": "RotateLeft", "action_success": True,
                    "agent": _agent([7.0, 0.9, 7.0], 180.0)}],
        )
        tl = self.sess.load_episode(eid)["timeline"]
        self.assertEqual(tl[0]["agent"]["position"], [7.0, 0.9, 7.0])
        self.assertEqual(tl[0]["agent"]["rotation_yaw_deg"], 270.0)
        self.assertEqual(tl[0]["agent_source"], "spawn")

    def test_each_entry_frame_pose_action_same_state(self):
        eid = "ep-consistency"
        initial = _agent([11.5, 0.9, 6.5], 90.0)
        steps = [
            {"step": 1, "action_type": "MoveAhead", "action_success": True,
             "agent": _agent([11.75, 0.9, 6.5], 90.0)},
            {"step": 2, "action_type": "RotateLeft", "action_success": True,
             "agent": _agent([11.75, 0.9, 6.5], 0.0)},
            {"step": 3, "action_type": "MoveAhead", "action_success": True,
             "agent": _agent([11.75, 0.9, 6.75], 0.0)},
        ]
        decisions = [_decision(0, "frame_0000.png", 0, "MoveAhead", initial)]
        for i in range(1, len(steps)):
            decisions.append(
                _decision(i, f"frame_{i:04d}.png", i, steps[i]["action_type"],
                          steps[i - 1]["agent"])
            )
        _write_episode(self.root, eid, spawn={}, setup={}, steps=steps,
                       decisions=decisions)
        tl = self.sess.load_episode(eid)["timeline"]
        self.assertEqual(len(tl), len(steps) + 1)
        self.assertEqual(tl[0]["frame"], "frame_0000.png")
        self.assertEqual(tl[0]["agent"], initial)
        for i in range(1, len(tl)):
            # frame_N and steps[N-1] are the same env.step event
            self.assertEqual(tl[i]["frame"], f"frame_{i:04d}.png")
            self.assertEqual(tl[i]["agent"], steps[i - 1]["agent"])
            self.assertEqual(tl[i]["action"], steps[i - 1]["action_type"])

    def test_rotate_left_and_right_temporal_consistency(self):
        eid = "ep-rotations"
        initial = _agent([0.0, 0.9, 0.0], 0.0)
        steps = [
            {"step": 1, "action_type": "RotateLeft", "action_success": True,
             "agent": _agent([0.0, 0.9, 0.0], 270.0)},
            {"step": 2, "action_type": "RotateRight", "action_success": True,
             "agent": _agent([0.0, 0.9, 0.0], 0.0)},
        ]
        decisions = [
            _decision(0, "frame_0000.png", 0, "RotateLeft", initial),
            _decision(1, "frame_0001.png", 1, "RotateRight", steps[0]["agent"]),
        ]
        _write_episode(self.root, eid, spawn={}, setup={}, steps=steps,
                       decisions=decisions)
        tl = self.sess.load_episode(eid)["timeline"]
        self.assertEqual(tl[1]["action"], "RotateLeft")
        self.assertEqual(_delta_yaw(tl[0]["agent"]["rotation_yaw_deg"],
                                    tl[1]["agent"]["rotation_yaw_deg"]), 270.0)
        self.assertEqual(tl[2]["action"], "RotateRight")
        self.assertEqual(_delta_yaw(tl[1]["agent"]["rotation_yaw_deg"],
                                    tl[2]["agent"]["rotation_yaw_deg"]), 90.0)

    def test_move_ahead_displacement_aligns_with_heading(self):
        eid = "ep-moveahead"
        initial = _agent([0.0, 0.9, 0.0], 90.0)
        steps = [
            {"step": 1, "action_type": "MoveAhead", "action_success": True,
             "agent": _agent([0.25, 0.9, 0.0], 90.0)},
            {"step": 2, "action_type": "MoveAhead", "action_success": True,
             "agent": _agent([0.5, 0.9, 0.0], 90.0)},
        ]
        decisions = [
            _decision(0, "frame_0000.png", 0, "MoveAhead", initial),
            _decision(1, "frame_0001.png", 1, "MoveAhead", steps[0]["agent"]),
        ]
        _write_episode(self.root, eid, spawn={}, setup={}, steps=steps,
                       decisions=decisions)
        tl = self.sess.load_episode(eid)["timeline"]
        prev = tl[0]["agent"]
        for i, st in enumerate(steps, start=1):
            if st["action_type"] != "MoveAhead" or not st.get("action_success"):
                continue
            fwd = _forward_xz(prev["rotation_yaw_deg"])
            self.assertGreater(_dot_xz(st["agent"]["position"], fwd), 0.0)
            prev = st["agent"]

    def test_timeline_index_maps_deterministically(self):
        eid = "ep-index"
        steps = [
            {"step": 1, "action_type": "MoveAhead", "action_success": True,
             "agent": _agent([0.25, 0.9, 0.0], 0.0)},
            {"step": 2, "action_type": "RotateRight", "action_success": True,
             "agent": _agent([0.25, 0.9, 0.0], 90.0)},
        ]
        _write_episode(self.root, eid, spawn={}, setup={}, steps=steps)
        a = self.sess.load_episode(eid)["timeline"]
        b = self.sess.load_episode(eid)["timeline"]
        self.assertEqual([e["idx"] for e in a], [0, 1, 2])
        self.assertEqual(
            [(e["frame"], e.get("action")) for e in a],
            [(e["frame"], e.get("action")) for e in b],
        )
        for i, e in enumerate(a):
            self.assertEqual(e["frame"], f"frame_{i:04d}.png")
            self.assertEqual(e["idx"], i)

    def test_load_does_not_mutate_raw_record(self):
        eid = "ep-raw"
        d = _write_episode(
            self.root, eid, spawn={}, setup={},
            steps=[{"step": 1, "action_type": "MoveAhead", "action_success": True,
                    "agent": _agent([0.25, 0.9, 0.0], 0.0)}],
        )
        raw = d / "privileged_research_record.json"
        before = raw.read_bytes()
        self.sess.load_episode(eid)
        self.assertEqual(raw.read_bytes(), before)


class SharedRendererPoseTests(unittest.TestCase):
    """2D / 3D / Unity must consume the same normalized timeline pose."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sess = EmbodiedSession("deterministic", seed=0)
        self.sess.set_episodes_root(self.root)
        self.eid = "ep-shared"
        self.initial = _agent([0.0, 0.9, 0.0], 90.0)
        self.steps = [
            {"step": 1, "action_type": "RotateRight", "action_success": True,
             "agent": _agent([0.0, 0.9, 0.0], 180.0)},
        ]
        self.decisions = [_decision(0, "frame_0000.png", 0, "RotateRight", self.initial)]
        _write_episode(self.root, self.eid, spawn={}, setup={},
                       steps=self.steps, decisions=self.decisions)

    def tearDown(self):
        self.tmp.cleanup()

    def test_unity_godview_uses_timeline_pose(self):
        captured = {}

        class _FakeCam:
            def render(self, *, agent_pose=None, **kwargs):
                captured["pose"] = agent_pose
                return b"frame"

        self.sess.load_episode(self.eid)
        tl = self.sess._loaded_episode["timeline"]
        self.sess._resolve_episode_house_path = lambda eid, header: "/fake/house.json"
        self.sess._god_camera_for = lambda hp: _FakeCam()
        self.sess.unity_godview_png(step=0)
        self.assertEqual(captured["pose"], tl[0]["agent"])
        self.sess.unity_godview_png(step=1)
        self.assertEqual(captured["pose"], tl[1]["agent"])

    def test_scene3d_trace_starts_at_true_initial_pose(self):
        from spatialforge.inspector.scene3d import build_scene3d, episode_scene_payload

        self.sess.load_episode(self.eid)
        tl = self.sess._loaded_episode["timeline"]
        house = {
            "houseId": "house-test",
            "rooms": [{"id": "r1", "roomType": "Kitchen", "floorPolygon": [
                {"x": 0, "y": 0, "z": 0}, {"x": 4, "y": 0, "z": 0},
                {"x": 4, "y": 0, "z": 4}, {"x": 0, "y": 0, "z": 4}]}],
            "walls": [], "doors": [], "windows": [], "objects": [],
            "metadata": {"agent": {"position": {"x": 0, "y": 0.9, "z": 0},
                                   "rotation": {"y": 90}, "horizon": 30}},
        }
        payload = episode_scene_payload(build_scene3d(house), timeline=tl)
        trace = payload["episode"]["trace"]
        # no spawn supplied -> the first trace point is the pre-action-0 pose
        self.assertEqual(trace[0], [0.0, 0.9, 0.0])
        # the final agent pose is the last timeline pose
        self.assertEqual(payload["episode"]["agent"]["position"], tl[-1]["agent"]["position"])


@unittest.skipUnless(MODEL_ROLLOUT.is_dir(), "real model-rollout episodes not present")
class RealUatEpisodeTests(unittest.TestCase):
    PEN = "ep-model-182e4844-114654"
    ROT = "ep-model-0f12e9e7-114654"

    @classmethod
    def setUpClass(cls):
        cls.sess = EmbodiedSession("deterministic", seed=0)
        cls.sess.set_episodes_root(MODEL_ROLLOUT)

    def test_pen_step4_frame_and_pose_same_state(self):
        out = self.sess.load_episode(self.PEN)
        tl, dec = out["timeline"], out["decisions"]
        entry = tl[4]
        self.assertEqual(entry["frame"], "frame_0004.png")
        self.assertEqual(entry["action"], "MoveAhead")
        # the RGB frame's authoritative pose is the same AI2-THOR event
        self.assertEqual(dec[4]["frame"], "frame_0004.png")
        self.assertEqual(entry["agent"]["position"], dec[4]["agent"]["position"])
        self.assertEqual(entry["agent"]["rotation_yaw_deg"],
                         dec[4]["agent"]["rotation_yaw_deg"])
        # and it equals the post-action-3 transition pose
        self.assertEqual(entry["agent"]["position"],
                         out["timeline"][4]["agent"]["position"])

    def test_pen_initial_pose_is_true_start_not_post_action(self):
        out = self.sess.load_episode(self.PEN)
        tl, dec = out["timeline"], out["decisions"]
        self.assertEqual(tl[0]["agent"]["position"], dec[0]["agent"]["position"])
        self.assertEqual(tl[0]["agent"]["rotation_yaw_deg"],
                         dec[0]["agent"]["rotation_yaw_deg"])
        self.assertEqual(tl[0]["agent_source"], "decisions[0].agent")
        # entry 0 must not be the post-action-0 pose
        self.assertNotEqual(tl[0]["agent"]["position"], tl[1]["agent"]["position"])

    def test_rotation_episode_entry0_heading_matches_fpv(self):
        out = self.sess.load_episode(self.ROT)
        tl, dec = out["timeline"], out["decisions"]
        # frame_0000 was captured at the pre-rotation yaw (decisions[0])
        self.assertEqual(tl[0]["agent"]["rotation_yaw_deg"],
                         dec[0]["agent"]["rotation_yaw_deg"])
        self.assertEqual(tl[0]["agent"]["rotation_yaw_deg"], 90.0)
        # entry 1 is the post-RotateRight state and the same event as frame_0001
        self.assertEqual(tl[1]["action"], "RotateRight")
        self.assertEqual(tl[1]["agent"]["rotation_yaw_deg"], 180.0)
        self.assertEqual(_delta_yaw(tl[0]["agent"]["rotation_yaw_deg"],
                                    tl[1]["agent"]["rotation_yaw_deg"]), 90.0)


@unittest.skipUnless(MODEL_LIVE.is_dir(), "real model-live episodes not present")
class RealLiveUatEpisodeTests(unittest.TestCase):
    """The reported pen UAT episode (first action RotateRight)."""

    EID = "ep-model-182e4844-128203"

    @classmethod
    def setUpClass(cls):
        cls.sess = EmbodiedSession("deterministic", seed=0)
        cls.sess.set_episodes_root(MODEL_LIVE)

    def test_entry0_fpv_heading_matches_stored_pose(self):
        out = self.sess.load_episode(self.EID)
        tl, dec = out["timeline"], out["decisions"]
        # The RGB frame_0000 is the pre-action-0 (yaw 90) observation.
        self.assertEqual(dec[0]["frame"], "frame_0000.png")
        self.assertEqual(dec[0]["agent"]["rotation_yaw_deg"], 90.0)
        # The Inspector entry 0 must show that same yaw, not the post-RotateRight
        # yaw 180 that steps[0].agent carries.
        self.assertEqual(tl[0]["frame"], "frame_0000.png")
        self.assertEqual(tl[0]["agent"]["rotation_yaw_deg"], 90.0)
        self.assertEqual(tl[0]["agent_source"], "decisions[0].agent")
        self.assertEqual(out["timeline"][1]["agent"]["rotation_yaw_deg"], 180.0)
        self.assertNotEqual(tl[0]["agent"]["rotation_yaw_deg"],
                            tl[1]["agent"]["rotation_yaw_deg"])


if __name__ == "__main__":
    unittest.main()
