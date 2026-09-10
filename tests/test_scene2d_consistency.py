"""2D/3D canonical-scene consistency invariants.

These tests enforce that the 2D top-down diagnostic view and the Three.js 3D
semantic view derive from *one* canonical scene payload, and that the 2D
projection is a pure function of the shared world coordinates and the shared
scene bounds (no independent legacy coordinate normalization).

The heavy lifting is done by executing the real browser module
``spatialforge/inspector/embodied_web/scene2d.js`` under node via
``tests/tools/scene2d_probe.mjs``, so the assertions cover the shipped code
rather than a Python re-implementation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from spatialforge.inspector.scene3d import build_scene3d, episode_scene_payload

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "procthor_house_fixture.json"
PROBE = REPO / "tests" / "tools" / "scene2d_probe.mjs"
WEB = REPO / "spatialforge" / "inspector" / "embodied_web"
NODE = shutil.which("node")


def _canonical_payload() -> dict:
    """Build a canonical payload from a *real* ProcTHOR house fixture."""
    house = json.loads(FIXTURE.read_text(encoding="utf-8"))
    scene = build_scene3d(house)
    meta_agent = (house.get("metadata") or {}).get("agent") or {}
    pos = meta_agent.get("position") or {"x": 0.0, "y": 0.0, "z": 0.0}
    yaw = float((meta_agent.get("rotation") or {}).get("y", 0.0))
    start = [float(pos["x"]), float(pos["y"]), float(pos["z"])]
    step1 = [start[0], start[1], start[2] - 0.25]
    timeline = [
        {
            "idx": 1, "step": 1, "action": "MoveAhead", "action_origin": "model",
            "agent": {"position": step1, "rotation_yaw_deg": yaw, "camera_horizon_deg": 0.0},
            "authoritative_target_visible": False,
        },
        {
            "idx": 2, "step": 2, "action": "Done", "action_origin": "model",
            "agent": {"position": step1, "rotation_yaw_deg": yaw, "camera_horizon_deg": 0.0},
            "terminal": True,
        },
    ]
    targets = [
        [start[0] + 1.0, start[1], start[2]],
        [start[0] - 1.0, start[1], start[2] + 1.0],
    ]
    return episode_scene_payload(
        scene,
        spawn={"position": start, "yaw": yaw, "horizon": 0.0},
        timeline=timeline,
        target_positions=targets,
        target_object_ids=["Mug|1|0", "Mug|2|0"],
        terminal=True,
    )


def _run_probe(payload: dict) -> dict:
    if NODE is None:
        raise unittest.SkipTest("node is not available for the cross-language probe")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "payload.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        proc = subprocess.run(
            [NODE, str(PROBE), str(p)],
            capture_output=True, text=True, timeout=120,
        )
    if proc.returncode != 0:
        raise AssertionError(f"node probe failed: {proc.stderr}")
    return json.loads(proc.stdout)


class Scene2dConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = _canonical_payload()
        cls.model = _run_probe(cls.payload)

    def test_entity_counts_match_canonical_scene(self):
        scene = self.payload["scene"]
        counts = self.model["counts"]
        self.assertEqual(counts["rooms"], len(scene["rooms"]))
        self.assertEqual(counts["walls"], len(scene["walls"]))
        self.assertEqual(counts["doors"], len(scene["doors"]))
        self.assertEqual(counts["windows"], len(scene["windows"]))
        self.assertEqual(counts["objects"], len(scene["objects"]))

    def test_scene_bounds_are_shared(self):
        self.assertEqual(self.model["bounds"], self.payload["scene"]["bounds"])
        self.assertEqual(self.model["bounds_source"], "scene")

    def test_agent_world_coordinate_is_identical_before_projection(self):
        agent_world = self.model["projected"]["agent_world"]
        self.assertEqual(agent_world, self.payload["episode"]["agent"]["position"])

    def test_agent_projection_is_a_pure_function_of_world_coordinate(self):
        self.assertEqual(
            self.model["projected"]["agent"],
            self.model["recomputed"]["agent"],
        )

    def test_targets_world_coordinates_are_identical_before_projection(self):
        expected = [t["position"] for t in self.payload["episode"]["targets"]]
        got = [t["world"] for t in self.model["projected"]["targets"]]
        self.assertEqual(got, expected)
        self.assertEqual(
            [t["projected"] for t in self.model["projected"]["targets"]],
            self.model["recomputed"]["targets"],
        )

    def test_trajectory_segment_count_is_identical(self):
        self.assertEqual(
            self.model["counts"]["trajectory_segments"],
            len(self.payload["episode"]["trace_segments"]),
        )

    def test_no_independent_episode_renormalization(self):
        """Shifting episode points must shift the projection by dx*scale only.

        If the 2D view re-fit its viewport to the episode points, the projected
        agent would not move by exactly the world delta times the (fixed) scene
        scale. This is the invariant that forbids a second, independent 2D map
        normalization.
        """
        t = self.model["transform"]
        dx, dz = 3.0, -2.0
        shifted = json.loads(json.dumps(self.payload))
        for seg in shifted["episode"]["trace_segments"]:
            for p in seg:
                p[0] += dx
                p[2] += dz
        for p in shifted["episode"]["trace"]:
            p[0] += dx
            p[2] += dz
        shifted["episode"]["agent"]["position"][0] += dx
        shifted["episode"]["agent"]["position"][2] += dz
        model2 = _run_probe(shifted)
        self.assertEqual(model2["transform"]["scale"], t["scale"])
        a0 = self.model["projected"]["agent"]
        a1 = model2["projected"]["agent"]
        self.assertAlmostEqual(a1[0] - a0[0], dx * t["scale"], places=6)
        self.assertAlmostEqual(a1[1] - a0[1], -dz * t["scale"], places=6)


class FrontendCanonicalSourceTests(unittest.TestCase):
    """Static guard: the shipped 2D and 3D views must share one payload builder."""

    @classmethod
    def setUpClass(cls):
        cls.src = (WEB / "app.js").read_text(encoding="utf-8")

    def test_both_views_consume_canonical_scene(self):
        self.assertIn("function canonicalScene()", self.src)
        self.assertIn("function canonicalEpisode()", self.src)
        self.assertGreaterEqual(self.src.count("canonicalScene()"), 3)

    def test_2d_view_uses_shared_projection_module(self):
        self.assertIn("Scene2D.buildScene2d(", self.src)
        self.assertIn('import * as Scene2D from "./scene2d.js";', self.src)

    def test_legacy_2d_map_logic_is_gone(self):
        self.assertNotIn("function godData(", self.src)
        self.assertNotIn("function detectPlane(", self.src)
        self.assertNotIn("function currentEpisode3d(", self.src)

    def test_2d_module_is_served(self):
        from spatialforge.inspector.embodied_server import create_embodied_server

        self.assertTrue((WEB / "scene2d.js").is_file())
        self.assertTrue((WEB / "package.json").is_file())
        # route exists (server smoke tested elsewhere); assert the file name is routed
        server_src = (REPO / "spatialforge" / "inspector" / "embodied_server.py").read_text()
        self.assertIn('"/scene2d.js"', server_src)
        self.assertIsNotNone(create_embodied_server)  # import stays valid


if __name__ == "__main__":
    unittest.main()
