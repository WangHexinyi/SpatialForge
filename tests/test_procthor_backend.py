"""Offline tests for the ProcTHOR backend (no Unity binary required).

These cover import safety, the action-space mapping, and error handling. Real
AI2-THOR integration is exercised by the standalone runtime smoke (needs the
Unity build + a real ProcTHOR house); it is not monkeypatched here.
"""

import unittest

from spatialforge.embodied.backends.procthor import (
    ProcTHORBackend,
    ProcTHORError,
    _THOR_ACTION,
)
from spatialforge.embodied.contracts import (
    CORE_NAV_ACTIONS,
    AgentActionType,
)


class ProcTHORBackendOfflineTests(unittest.TestCase):
    def test_module_imports_without_ai2thor(self):
        # importing must not require ai2thor (lazy import inside load_house)
        import spatialforge.embodied.backends.procthor as m  # noqa: F401

    def test_core_actions_are_mapped(self):
        b = ProcTHORBackend()
        mapped = set(_THOR_ACTION.keys())
        for act in CORE_NAV_ACTIONS:
            if act == AgentActionType.DONE:
                continue  # DONE is handled by the runner, never stepped to Thor
            self.assertIn(act, mapped, f"missing mapping for {act}")

    def test_unsupported_action_raises(self):
        b = ProcTHORBackend()
        # unsupported actions raise before any controller access
        with self.assertRaises(ProcTHORError):
            b.apply_action(AgentActionType.DONE, {}, 0)

    def test_load_house_without_source_raises(self):
        b = ProcTHORBackend()  # no house provided
        with self.assertRaises(ProcTHORError):
            b.load_house()  # raises before trying to import ai2thor

    def test_house_json_loader_from_dict_and_path(self):
        from spatialforge.embodied.backends.procthor import load_house_json
        import tempfile, json, os

        data = {"rooms": [], "objects": [], "metadata": {}}
        self.assertEqual(load_house_json(data), data)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            p = f.name
        try:
            self.assertEqual(load_house_json(p), data)
        finally:
            os.unlink(p)


if __name__ == "__main__":
    unittest.main()
