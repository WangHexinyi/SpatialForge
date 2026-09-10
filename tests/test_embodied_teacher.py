import unittest

from spatialforge.embodied.backends.deterministic import DeterministicGridBackend
from spatialforge.embodied.contracts import AgentActionType
from spatialforge.embodied.environment import EmbodiedEnvironment
from spatialforge.embodied.teacher import GridTeacher


class TeacherTests(unittest.TestCase):
    def _setup_facing_away(self):
        # mug at (2,1). Spawn (1,1) but heading 2 (-x) => facing away from it.
        b = DeterministicGridBackend()
        b.load_house(None, seed=0)
        env = EmbodiedEnvironment(b)
        tr = env.start_episode("det-grid", "mug", start_heading=2)
        return b, env, tr

    def test_teacher_plans_a_nonempty_path_when_facing_away(self):
        b, env, tr = self._setup_facing_away()
        target_pos = b.object_position(tr.task.target_object_ids[0])
        target_cell = (int(target_pos[0]), int(target_pos[1]))
        start = (int(env.agent_state.position[0]), int(env.agent_state.position[1]))
        heading = int(env.agent_state.rotation_yaw_deg // 90) % 4
        path = GridTeacher(b).plan_to_visible(target_cell, start, heading)
        self.assertTrue(path.planned)
        self.assertGreater(len(path.actions), 0)
        # teacher actions come from the permitted agent action vocabulary
        for a in path.actions:
            AgentActionType(a)  # must not raise

    def test_replaying_teacher_path_leads_to_done_success(self):
        b, env, tr = self._setup_facing_away()
        target_pos = b.object_position(tr.task.target_object_ids[0])
        target_cell = (int(target_pos[0]), int(target_pos[1]))
        start = (int(env.agent_state.position[0]), int(env.agent_state.position[1]))
        heading = int(env.agent_state.rotation_yaw_deg // 90) % 4
        path = GridTeacher(b).plan_to_visible(target_cell, start, heading)
        for a in path.actions:
            env.step(AgentActionType(a))
        tr = env.step(AgentActionType.DONE)
        self.assertTrue(tr.terminal)
        self.assertIs(tr.episode.success, True)

    def test_unreachable_returns_unplanned(self):
        b = DeterministicGridBackend()
        b.load_house(None, seed=0)
        path = GridTeacher(b).plan_to_visible((99, 99), (1, 1), 0)
        self.assertFalse(path.planned)

    def test_teacher_path_not_in_model_input(self):
        import json

        b, env, tr = self._setup_facing_away()
        self.assertNotIn("teacher_path", json.dumps(tr.model_input))


if __name__ == "__main__":
    unittest.main()
