import unittest

import numpy as np

from spatialforge.embodied.backends.deterministic import DeterministicGridBackend
from spatialforge.embodied.contracts import AgentActionType
from spatialforge.embodied.environment import EmbodiedEnvironment


def fresh(max_steps=300, target="mug"):
    b = DeterministicGridBackend()
    b.load_house(None, seed=0)
    env = EmbodiedEnvironment(b, episode_max_steps=max_steps)
    tr = env.start_episode("det-grid", target, max_steps=max_steps)
    return b, env, tr


class EmbodiedTransitionTests(unittest.TestCase):
    def test_reset_returns_observation_and_state(self):
        b, env, tr = fresh()
        self.assertIsNotNone(tr.observation.rgb)
        self.assertEqual(tr.episode.status.value, "running")
        self.assertEqual(tr.agent_state.position[:2], (1.0, 1.0))

    def test_move_ahead_changes_position_and_frame(self):
        b, env, tr = fresh()
        frame0 = tr.observation.rgb.copy()
        pos0 = tr.agent_state.position
        tr = env.step(AgentActionType.MOVE_AHEAD)
        self.assertNotEqual(tr.agent_state.position[:2], pos0[:2])
        self.assertFalse(np.array_equal(frame0, tr.observation.rgb))

    def test_rotate_changes_heading_not_position(self):
        b, env, tr = fresh()
        pos0 = tr.agent_state.position
        tr = env.step(AgentActionType.ROTATE_LEFT)
        self.assertEqual(tr.agent_state.position, pos0)
        self.assertEqual(tr.agent_state.rotation_yaw_deg, -90.0 % 360)

    def test_action_success_recorded(self):
        b, env, tr = fresh()
        tr = env.step(AgentActionType.MOVE_AHEAD)
        self.assertIs(tr.action.success, True)

    def test_done_when_target_visible_succeeds(self):
        # mug at (2,1); agent spawns (1,1) facing +x => visible
        b, env, tr = fresh()
        tr = env.step(AgentActionType.DONE)
        self.assertTrue(tr.terminal)
        self.assertIs(tr.episode.success, True)

    def test_done_when_target_absent_fails(self):
        b, env, tr = fresh()
        tr = env.step(AgentActionType.DONE)
        # rerun with an unknown category => no target => failure
        b2, env2, tr2 = fresh(target="nonexistent_thing")
        tr2 = env2.step(AgentActionType.DONE)
        self.assertTrue(tr2.terminal)
        self.assertIs(tr2.episode.success, False)

    def test_step_budget_exhaustion_fails(self):
        b, env, tr = fresh(max_steps=2)
        tr = env.step(AgentActionType.MOVE_AHEAD)
        tr = env.step(AgentActionType.MOVE_AHEAD)
        self.assertTrue(tr.terminal)
        self.assertIs(tr.episode.success, False)
        self.assertIn("budget", tr.episode.success_reason)

    def test_no_steps_after_terminal(self):
        b, env, tr = fresh()
        tr = env.step(AgentActionType.DONE)
        self.assertTrue(tr.terminal)
        self.assertIsNone(env.step(AgentActionType.MOVE_AHEAD))

    def test_history_records_episode(self):
        b, env, tr = fresh()
        env.step(AgentActionType.ROTATE_LEFT)
        env.step(AgentActionType.ROTATE_RIGHT)
        env.step(AgentActionType.MOVE_AHEAD)
        # reset action + 3 = 4 recorded
        self.assertGreaterEqual(len(env.history.actions), 4)
        self.assertEqual(len(env.history.observations), len(env.history.actions))


if __name__ == "__main__":
    unittest.main()
