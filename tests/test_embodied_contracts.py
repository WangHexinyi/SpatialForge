import json
import unittest

from spatialforge.embodied.contracts import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    AgentState,
    EpisodeHistory,
    EpisodeState,
    EpisodeStatus,
    ObjectSearchTask,
)
from spatialforge.embodied.model_input import (
    ModelInputLeakError,
    build_agent_model_input,
    build_privileged_debug_payload,
    validate_model_input_isolation,
)


def make_observation(step=0):
    return AgentObservation(step=step, timestamp_ms=1, frame_id=f"f-{step}")


def make_task(**kw):
    base = dict(
        task_id="t1",
        house_id="h1",
        target_category="mug",
        instruction="Find a mug.",
        max_steps=10,
        target_object_ids=["mug_1"],
        target_positions=[(1.0, 2.0, 0.0)],
    )
    base.update(kw)
    return ObjectSearchTask(**base)


class EmbodiedContractRoundTripTests(unittest.TestCase):
    def test_agent_action_round_trip(self):
        a = AgentAction(
            action_type=AgentActionType.ROTATE_LEFT,
            step=3,
            success=True,
            collision=False,
            timestamp_ms=5,
        )
        b = AgentAction.from_dict(a.to_dict())
        self.assertEqual(b.action_type, AgentActionType.ROTATE_LEFT)
        self.assertEqual(b.step, 3)
        self.assertIs(b.success, True)

    def test_agent_state_round_trip(self):
        s = AgentState(position=(1.0, 2.0, 0.5), rotation_yaw_deg=90.0)
        t = AgentState.from_dict(s.to_dict())
        self.assertEqual(t.position, (1.0, 2.0, 0.5))
        self.assertEqual(t.rotation_yaw_deg, 90.0)

    def test_episode_state_round_trip(self):
        e = EpisodeState(
            episode_id="e1", house_id="h1", task_id="t1",
            status=EpisodeStatus.SUCCESS, success=True,
        )
        r = EpisodeState.from_dict(e.to_dict())
        self.assertEqual(r.status, EpisodeStatus.SUCCESS)
        self.assertIs(r.success, True)

    def test_object_search_task_agent_view_has_no_privileged(self):
        t = make_task()
        av = json.dumps(t.agent_view())
        self.assertNotIn("target_object_ids", av)
        self.assertNotIn("target_positions", av)
        self.assertNotIn("teacher_path", av)
        # privileged view does contain them
        pv = json.dumps(t.privileged_view())
        self.assertIn("target_positions", pv)

    def test_action_history_round_trip_fields(self):
        h = EpisodeHistory(EpisodeState("e", "h", "t"))
        h.record_action(AgentAction(AgentActionType.MOVE_AHEAD, step=0, success=True))
        self.assertEqual(len(h.actions), 1)
        self.assertEqual(h.last_action().action_type, AgentActionType.MOVE_AHEAD)


class ModelInputIsolationTests(unittest.TestCase):
    def test_model_input_excludes_all_privileged_truth(self):
        task = make_task()
        obs = make_observation(step=4)
        mi = build_agent_model_input(obs, task, None)
        s = json.dumps(mi)
        for forbidden in (
            "target_object_ids", "target_positions", "reachable_positions",
            "teacher_path", "position", "privileged", "segmentation",
            "shortest_path", "rotation_yaw_deg",
        ):
            self.assertNotIn(forbidden, s, f"leaked {forbidden}")
        # goal is present (task instruction/category allowed)
        self.assertEqual(mi["goal"]["target_category"], "mug")

    def test_leak_detection_raises(self):
        bad = {"goal": {}, "observation": {}, "action_history": [],
               "target_positions": [[1, 2, 3]]}
        with self.assertRaises(ModelInputLeakError):
            validate_model_input_isolation(bad)

    def test_nested_leak_detection(self):
        bad = {"a": {"b": {"c": "teacher_path from oracle"}}}
        with self.assertRaises(ModelInputLeakError):
            validate_model_input_isolation(bad)

    def test_observation_only_allowed_keys(self):
        obs = make_observation(step=0)
        obs.rgb = None
        obs.instruction = "Find a mug."
        mi = build_agent_model_input(obs, make_task(), None)
        obs_keys = set(mi["observation"].keys())
        self.assertTrue(
            obs_keys.issubset(
                {"step", "timestamp_ms", "image_path", "frame_id",
                 "camera_horizon_deg", "instruction", "history_summary",
                 "cause_action", "rgb_shape", "rgb_base64"}
            )
        )

    def test_privileged_builder_is_separate_and_flagged(self):
        obs = make_observation()
        t = make_task()
        agent = AgentState(position=(0.0, 0.0, 0.0))
        priv = build_privileged_debug_payload(obs, t, agent)
        self.assertIs(priv.get("_privileged"), True)
        self.assertIn("target_positions", json.dumps(priv))
        self.assertIn("position", json.dumps(priv))


if __name__ == "__main__":
    unittest.main()
