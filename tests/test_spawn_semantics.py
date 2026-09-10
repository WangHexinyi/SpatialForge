import unittest

from spatialforge.embodied.spawn_semantics import check_episode_semantics, setup_block


def _step(step, action, pos, yaw=0.0, horizon=0.0, origin="model"):
    return {
        "step": step,
        "action_type": action,
        "action_origin": origin,
        "agent": {
            "position": list(pos),
            "rotation_yaw_deg": yaw,
            "camera_horizon_deg": horizon,
        },
    }


class SpawnSemanticsTests(unittest.TestCase):
    def _base_record(self):
        return {
            "spawn": {"position": [1.0, 0.9, 1.0], "yaw": 0.0, "horizon": 0.0},
            "setup": {
                "spawn_requested": {"position": [1.0, 0.9, 1.0], "yaw": 0.0, "horizon": 0.0},
                "initial_pose": {
                    "position": [1.0, 0.9, 1.0],
                    "rotation_yaw_deg": 0.0,
                    "horizon_deg": 0.0,
                },
                "teleport_policy": "TeleportFull only during reset/spawn (environment setup)",
            },
            "steps": [
                _step(1, "RotateRight", [1.0, 0.9, 1.0], yaw=90.0),
                _step(2, "MoveAhead", [1.0, 0.9, 1.25], yaw=90.0),
                _step(3, "MoveAhead", [1.0, 0.9, 1.5], yaw=90.0),
                _step(4, "LookDown", [1.0, 0.9, 1.5], yaw=90.0, horizon=-30.0),
                _step(5, "Done", [1.0, 0.9, 1.5], yaw=90.0, horizon=-30.0),
            ],
        }

    def test_clean_episode_ok(self):
        result = check_episode_semantics(self._base_record())
        self.assertTrue(result["ok"], result["violations"])
        self.assertEqual(result["checks"]["steps_checked"], 5)

    def test_spawn_initial_mismatch_flagged(self):
        rec = self._base_record()
        rec["setup"]["initial_pose"]["position"] = [5.0, 0.9, 5.0]
        result = check_episode_semantics(rec)
        self.assertFalse(result["ok"])
        self.assertEqual(result["violations"][0]["kind"], "spawn_initial_pose_mismatch")

    def test_teleport_between_steps_flagged(self):
        rec = self._base_record()
        rec["steps"].append(_step(6, "RotateRight", [7.0, 0.9, 7.0], yaw=180.0))
        result = check_episode_semantics(rec)
        self.assertFalse(result["ok"])
        kinds = {v["kind"] for v in result["violations"]}
        self.assertIn("unexplained_displacement", kinds)

    def test_blocked_move_allowed(self):
        rec = self._base_record()
        rec["steps"].append(_step(6, "MoveAhead", [1.0, 0.9, 1.5], yaw=90.0, horizon=-30.0))
        result = check_episode_semantics(rec)
        self.assertTrue(result["ok"], result["violations"])

    def test_setup_block_built_from_agent_state(self):
        class _State:
            position = (1.0, 0.9, 2.0)
            rotation_yaw_deg = 45.0
            camera_horizon_deg = 30.0

        block = setup_block(
            {"position": (1.0, 0.9, 2.0), "yaw": 45.0, "horizon": 30.0}, _State()
        )
        self.assertEqual(block["initial_pose"]["position"], [1.0, 0.9, 2.0])
        self.assertIn("teleport_policy", block)


if __name__ == "__main__":
    unittest.main()
