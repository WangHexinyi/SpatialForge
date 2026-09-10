import json
import unittest

from spatialforge.embodied.backends.deterministic import DeterministicGridBackend
from spatialforge.embodied.contracts import AgentActionType
from spatialforge.embodied.environment import EmbodiedEnvironment
from spatialforge.embodied.records import (
    build_privileged_research_record,
    build_student_training_record,
)


def _drive(max_steps=20, target="mug"):
    b = DeterministicGridBackend()
    b.load_house(None, seed=0)
    env = EmbodiedEnvironment(b, episode_max_steps=max_steps)
    ini = env.start_episode("g", target, max_steps=max_steps)
    trans = []
    for a in (AgentActionType.MOVE_AHEAD, AgentActionType.ROTATE_LEFT, AgentActionType.DONE):
        t = env.step(a)
        if t is not None:
            trans.append(t)
    return b, env, ini, trans


class StudentRecordIsolationTests(unittest.TestCase):
    def test_student_record_is_clean_and_structured(self):
        b, env, ini, trans = _drive()
        rec = build_student_training_record("ep-x", env.task.agent_view(), trans, ini.observation)
        self.assertEqual(rec["domain"], "student_training")
        self.assertEqual(rec["_leak_checked"], "clean")
        s = json.dumps(rec)
        for f in ("position", "target_object_ids", "reachable", "teacher_path",
                  "world_position", "segmentation", "god_view"):
            self.assertNotIn(f, s)
        actions = [x["teacher_action"] for x in rec["steps"] if x["role"] == "action"]
        self.assertEqual(actions[-1], "Done")

    def test_student_record_rejects_privileged_key(self):
        with self.assertRaises(Exception):
            build_student_training_record(
                "ep-x", {"target_object_ids": ["Mug|1"]}, [], None)

    def test_domains_are_distinct(self):
        b, env, ini, trans = _drive()
        s = build_student_training_record("ep", env.task.agent_view(), trans, ini.observation)
        p = build_privileged_research_record(
            "ep", env.task.privileged_view(), env.episode, trans, {"x": 1})
        self.assertEqual(s["domain"], "student_training")
        self.assertEqual(p["domain"], "privileged_research")
        self.assertNotEqual(s["domain"], p["domain"])
        # privileged carries truth, student does not
        self.assertTrue(p.get("_privileged"))
        self.assertIn("target_object_ids", p["task"])
        self.assertNotIn("target_object_ids", json.dumps(s))

    def test_privileged_carries_pose_and_visibility(self):
        b, env, ini, trans = _drive()
        p = build_privileged_research_record(
            "ep", env.task.privileged_view(), env.episode, trans, {})
        self.assertGreater(len(p["steps"]), 0)
        self.assertIn("agent", p["steps"][0])


if __name__ == "__main__":
    unittest.main()
