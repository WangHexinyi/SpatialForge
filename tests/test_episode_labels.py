import unittest

from spatialforge.embodied.episode_labels import (
    classify_episode,
    condensed_view,
    curated_priority,
    detect_policy_collapse,
    detect_stuck,
    no_progress_runs,
)
from spatialforge.embodied.model_rollout import no_progress_update


def _agent(pos, yaw=0.0, horizon=0.0):
    return {"position": list(pos), "rotation_yaw_deg": yaw, "camera_horizon_deg": horizon}


def _step(i, action, agent, origin="model", success=True, visible=False, terminal=False):
    return {
        "step": i,
        "action_type": action,
        "action_origin": origin,
        "action_success": success,
        "agent": agent,
        "authoritative_target_visible": visible,
        "terminal": terminal,
    }


def _valid_record():
    spawn = {"position": [1.0, 0.9, 1.0], "yaw": 0.0, "horizon": 0.0, "initially_visible": False}
    return {
        "episode": {"status": "success", "success": True},
        "render_quality": "Medium",
        "spawn": spawn,
        "setup": {
            "initial_pose": {
                "position": [1.0, 0.9, 1.0], "rotation_yaw_deg": 0.0, "horizon_deg": 0.0,
            },
        },
        "steps": [
            _step(1, "RotateRight", _agent([1.0, 0.9, 1.0], yaw=90.0)),
            _step(2, "MoveAhead", _agent([1.0, 0.9, 1.25], yaw=90.0)),
            _step(3, "Done", _agent([1.0, 0.9, 1.25], yaw=90.0), terminal=True),
        ],
    }


def _legacy_record():
    # recorded spawn is a privileged sampling candidate != real first-frame pose
    return {
        "episode": {"status": "success", "success": True},
        "spawn": {"position": [5.0, 0.9, 5.0], "yaw": 0.0, "horizon": 0.0},
        "setup": {},
        "steps": [
            _step(1, "RotateRight", _agent([1.0, 0.9, 1.0], yaw=90.0)),
            _step(2, "MoveAhead", _agent([1.0, 0.9, 1.25], yaw=90.0)),
        ],
    }


class EpisodeLabelTests(unittest.TestCase):
    def test_valid_episode_labeled_valid_success(self):
        cls = classify_episode(_valid_record())
        self.assertIn("VALID", cls["labels"])
        self.assertIn("SUCCESS", cls["labels"])
        self.assertNotIn("LEGACY", cls["labels"])
        self.assertTrue(cls["flags"]["valid"])
        self.assertTrue(cls["flags"]["researcher_useful"])
        self.assertGreater(curated_priority(cls), 50)

    def test_legacy_episode_labeled_legacy_anomaly(self):
        cls = classify_episode(_legacy_record())
        self.assertIn("LEGACY", cls["labels"])
        self.assertIn("ANOMALY", cls["labels"])
        self.assertFalse(cls["flags"]["valid"])
        self.assertLess(curated_priority(cls), 0)

    def test_low_quality_is_invalid_sensor(self):
        rec = _valid_record()
        rec["render_quality"] = "Low"
        cls = classify_episode(rec)
        self.assertIn("INVALID_SENSOR", cls["labels"])
        self.assertFalse(cls["flags"]["valid"])

    def test_timeout_and_collapse(self):
        rec = _valid_record()
        rec["episode"] = {"status": "failure", "success": False}
        rec["terminal_reason"] = "step budget exhausted before Done."
        rec["steps"] = [
            _step(i, "MoveAhead", _agent([1.0, 0.9, 1.0]))
            for i in range(1, 41)
        ]
        # align setup/spawn to avoid a spurious anomaly
        rec["spawn"]["position"] = [1.0, 0.9, 1.0]
        rec["setup"]["initial_pose"]["position"] = [1.0, 0.9, 1.0]
        cls = classify_episode(rec)
        self.assertIn("TIMEOUT", cls["labels"])
        self.assertIn("STUCK", cls["labels"])
        self.assertIn("POLICY_COLLAPSE", cls["labels"])

    def test_false_done(self):
        rec = _valid_record()
        rec["episode"] = {"status": "failure", "success": False}
        decisions = [{
            "model_action": "Done", "executed": True,
            "authoritative_target_visible": False,
        }]
        cls = classify_episode(rec, decisions=decisions)
        self.assertIn("FALSE_DONE", cls["labels"])

    def test_no_progress_run_and_condensed_view(self):
        rec = _valid_record()
        rec["steps"] = [
            _step(i, "MoveAhead", _agent([1.0, 0.9, 1.0]))
            for i in range(1, 13)
        ]
        rec["spawn"]["position"] = [1.0, 0.9, 1.0]
        rec["setup"]["initial_pose"]["position"] = [1.0, 0.9, 1.0]
        runs = no_progress_runs(rec)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["count"], 12)
        self.assertEqual(runs[0]["action"], "MoveAhead")

        timeline = [{"idx": 0, "step": 0}] + [
            {"idx": i, "step": i} for i in range(1, 13)
        ]
        view = condensed_view(timeline, runs)
        # first entry of the run is kept; the 11 repeats are collapsed
        self.assertEqual(len(view["indices"]), 2)
        self.assertEqual(len(view["segments"]), 1)
        self.assertEqual(view["segments"][0]["to_idx"], 12)

    def test_stuck_and_collapse_helpers(self):
        rec = _valid_record()
        rec["steps"] = [_step(i, "MoveAhead", _agent([1.0, 0.9, 1.0])) for i in range(1, 30)]
        self.assertTrue(detect_stuck(rec)["stuck"])
        self.assertTrue(detect_policy_collapse(rec)["collapse"])

    def test_no_progress_update_counter(self):
        last, count = no_progress_update((1.0, 1.0), None, 0)
        self.assertEqual(count, 0)
        last, count = no_progress_update((1.0, 1.0), last, count)
        self.assertEqual(count, 1)
        last, count = no_progress_update((1.0, 1.0), last, count)
        self.assertEqual(count, 2)
        last, count = no_progress_update((1.5, 1.0), last, count)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
