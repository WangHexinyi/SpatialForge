"""Regression tests for truthful door / window wall openings (3D God View).

ProcTHOR represents one physical wall plane as several wall records (one per
adjacent room, sometimes with reversed vertex order) and doors / windows as
``holePolygon`` rectangles in the host wall's local frame. These tests prove:

* every opening is linked to a real host wall and cut out of *all* collinear
  wall records, so the duplicate side cannot seal the doorway;
* door openings remove floor-to-door-top wall (with the lintel preserved) while
  windows keep the wall below the sill and above the head;
* a real agent doorway transition never intersects a rendered solid segment;
* unrelated walls, counts, bounds, objects and episode truth are unchanged.

The wall placement math is executed through the shipped browser module
``wall_geometry.js`` via ``tests/tools/wall_geometry_probe.mjs`` (node).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from spatialforge.inspector.scene3d import build_scene3d, episode_scene_payload

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "spatialforge" / "inspector" / "embodied_web"
FIXTURE = REPO / "tests" / "fixtures" / "procthor_house_fixture.json"
HOUSE_0034 = REPO / "outputs" / "embodied_bc" / "houses" / "house_0034.json"
EPISODE_DIR = REPO / "outputs" / "embodied_bc" / "dataset" / "ep-bc-05ca85bf"
PROBE = REPO / "tests" / "tools" / "wall_geometry_probe.mjs"
NODE = shutil.which("node")

# Real doorway crossing from episode ep-bc-05ca85bf (house_0034, door|2|4):
# steps 26 -> 27 (timeline idx 27 -> 28) walk through the doorway at x=8.5.
DOOR_CROSSING = ([8.5, 0.9009997844696045, 10.75], [8.5, 0.9009997844696045, 11.0])
DOOR_CROSSING_POINT = [8.5, 0.9009997844696045, 10.881]


def _house(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scene(path: Path) -> dict:
    return build_scene3d(_house(path))


def _wall(scene: dict, wall_id: str) -> dict:
    for w in scene["walls"]:
        if w["id"] == wall_id:
            return w
    raise AssertionError(f"wall {wall_id} not in scene")


def _opening(scene: dict, opening_id: str) -> dict:
    for o in scene["doors"] + scene["windows"]:
        if o["id"] == opening_id:
            return o
    raise AssertionError(f"opening {opening_id} not in scene")


def _to_three(p: list) -> list:
    """Test-side mirror of the shipped canonical world_space.worldToThree."""
    return [p[0], p[1], -p[2]]


def _local_to_world(wall: dict, u: float) -> list:
    a, b = wall["base"]
    length = math.hypot(b[0] - a[0], b[2] - a[2])
    return [
        a[0] + (b[0] - a[0]) / length * u,
        a[2] + (b[2] - a[2]) / length * u,
    ]


def _world_to_local(wall: dict, x: float, z: float) -> float:
    a, b = wall["base"]
    length = math.hypot(b[0] - a[0], b[2] - a[2])
    ux, uz = (b[0] - a[0]) / length, (b[2] - a[2]) / length
    return (x - a[0]) * ux + (z - a[2]) * uz


def _segments_cover(segments: list, u: float, y: float, eps: float = 1e-6) -> bool:
    return any(
        s["start"] - eps <= u <= s["end"] + eps
        and s["y0"] - eps <= y <= s["y1"] + eps
        for s in segments
    )


def _point_in_box(point: list, box: dict, eps: float = 1e-6) -> bool:
    cx, cy, cz = box["center"]
    sx, sy, sz = box["size"]
    theta = box["rotationY"]
    dx, dy, dz = point[0] - cx, point[1] - cy, point[2] - cz
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    lx = dx * cos_t - dz * sin_t
    lz = dx * sin_t + dz * cos_t
    return (
        abs(lx) <= sx / 2 + eps
        and abs(dy) <= sy / 2 + eps
        and abs(lz) <= sz / 2 + eps
    )


def _run_probe(payload: dict) -> dict:
    if NODE is None:
        raise unittest.SkipTest("node is not available for the wall geometry probe")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "payload.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        proc = subprocess.run(
            [NODE, str(PROBE), str(p)], capture_output=True, text=True, timeout=120,
        )
    if proc.returncode != 0:
        raise AssertionError(f"node probe failed: {proc.stderr}")
    return json.loads(proc.stdout)


def _full_wall_boxes(scene: dict) -> dict:
    """Probe boxes when every wall renders as its canonical full rectangle."""
    stripped = json.loads(json.dumps({"scene": scene}))
    for w in stripped["scene"]["walls"]:
        w.pop("segments", None)
    probed = _run_probe(stripped)
    return {w["id"]: w["boxes"] for w in probed["walls"]}


class RealHouseDoorOpeningTests(unittest.TestCase):
    """house_0034 door|2|4: a real interior doorway between two rooms."""

    @classmethod
    def setUpClass(cls):
        cls.scene = _scene(HOUSE_0034)
        cls.door = _opening(cls.scene, "door|2|4")
        cls.host = _wall(cls.scene, "wall|2|6.53|10.88|13.06|10.88")
        cls.duplicate = _wall(cls.scene, "wall|4|6.53|10.88|13.06|10.88")

    def test_door_is_linked_to_both_sides_of_the_physical_wall(self):
        self.assertEqual(self.door["wall_id"], self.host["id"])
        self.assertEqual(
            set(self.door["wall_ids"]), {self.host["id"], self.duplicate["id"]}
        )

    def test_opening_world_position_comes_from_host_wall_and_hole(self):
        self.assertAlmostEqual(self.door["position"][0], 8.7141, places=3)
        self.assertAlmostEqual(self.door["position"][1], 0.0, places=6)
        self.assertAlmostEqual(self.door["position"][2], 10.881, places=3)
        self.assertAlmostEqual(self.door["width"], 1.0815, places=3)
        self.assertAlmostEqual(self.door["height"], 2.1302, places=3)
        # assetPosition is wall-local in ProcTHOR and must not leak into world.
        self.assertNotAlmostEqual(self.door["position"][0], self.door["asset_position"]["x"])

    def test_host_wall_is_split_around_the_door(self):
        segs = self.host["segments"]
        self.assertEqual(len(segs), 3)
        left, lintel, right = segs
        self.assertAlmostEqual(left["start"], 0.0, places=4)
        self.assertAlmostEqual(left["end"], 1.6443, places=3)
        self.assertAlmostEqual(left["y0"], 0.0, places=6)
        self.assertAlmostEqual(left["y1"], self.host["height"], places=3)
        self.assertAlmostEqual(lintel["start"], 1.6443, places=3)
        self.assertAlmostEqual(lintel["end"], 2.7258, places=3)
        self.assertAlmostEqual(lintel["y0"], 2.1302, places=3)
        self.assertAlmostEqual(lintel["y1"], self.host["height"], places=3)
        self.assertAlmostEqual(right["start"], 2.7258, places=3)
        self.assertAlmostEqual(right["end"], self.host["length"], places=3)
        # the doorway itself is open floor-to-door-top
        self.assertFalse(_segments_cover(segs, 2.185, 0.9))
        self.assertFalse(_segments_cover(segs, 2.185, 2.13))

    def test_reversed_duplicate_wall_gets_the_same_world_opening(self):
        # wall|4 stores its base reversed; the cut must be mirrored accordingly.
        segs = self.duplicate["segments"]
        self.assertEqual(len(segs), 3)
        self.assertAlmostEqual(segs[0]["end"], 6.528 - 2.7258, places=3)
        self.assertAlmostEqual(segs[1]["start"], 6.528 - 2.7258, places=3)
        self.assertAlmostEqual(segs[1]["end"], 6.528 - 1.6443, places=3)
        # world gap of both records is identical: x in [8.1733, 9.2548]
        for wall in (self.host, self.duplicate):
            gap = [s for s in wall["segments"] if s["y0"] > 0.0]
            self.assertEqual(len(gap), 1)
            w0 = _local_to_world(wall, gap[0]["start"])[0]
            w1 = _local_to_world(wall, gap[0]["end"])[0]
            self.assertAlmostEqual(min(w0, w1), 8.1733, places=3)
            self.assertAlmostEqual(max(w0, w1), 9.2548, places=3)

    def test_collinear_duplicate_cannot_seal_the_doorway(self):
        x, _, z = DOOR_CROSSING_POINT
        for wall in (self.host, self.duplicate):
            u = _world_to_local(wall, x, z)
            self.assertFalse(_segments_cover(wall["segments"], u, 0.9),
                             f"{wall['id']} still seals the doorway")


class RealHouseWindowOpeningTests(unittest.TestCase):
    """house_0034 window|4|0: a partial-height window, never a door gap."""

    @classmethod
    def setUpClass(cls):
        cls.scene = _scene(HOUSE_0034)
        cls.window = _opening(cls.scene, "window|4|0")
        cls.host = _wall(cls.scene, "wall|4|6.53|15.23|13.06|15.23")

    def test_window_keeps_wall_below_sill_and_above_head(self):
        u0, u1 = self.window["local_x"]
        u_mid = (u0 + u1) / 2.0
        sill = self.window["position"][1]
        head = sill + self.window["height"]
        self.assertGreater(sill, 0.5)
        self.assertLess(head, self.host["height"])
        segs = self.host["segments"]
        self.assertEqual(len(segs), 4)
        # wall below the sill and above the head is preserved
        self.assertTrue(_segments_cover(segs, u_mid, sill - 0.1))
        self.assertTrue(_segments_cover(segs, u_mid, head + 0.1))
        # the hole itself is open
        self.assertFalse(_segments_cover(segs, u_mid, (sill + head) / 2))
        # side walls are full height
        self.assertTrue(_segments_cover(segs, max(u0 - 0.05, 0.0), 1.0))
        self.assertTrue(_segments_cover(segs, min(u1 + 0.05, self.host["length"]), 1.0))

    def test_window_world_position_is_hole_base_center(self):
        self.assertAlmostEqual(self.window["position"][0], 10.2179, places=3)
        self.assertAlmostEqual(self.window["position"][1], 0.8903, places=3)
        self.assertAlmostEqual(self.window["position"][2], 15.233, places=3)


class RegressionInvariantTests(unittest.TestCase):
    """Round 3 payload invariants that must not change with opening derivation."""

    @classmethod
    def setUpClass(cls):
        cls.house = _house(HOUSE_0034)
        cls.scene = build_scene3d(cls.house)

    def test_entity_counts_equal_raw_house(self):
        self.assertEqual(len(self.scene["rooms"]), len(self.house["rooms"]))
        self.assertEqual(len(self.scene["walls"]), len(self.house["walls"]))
        self.assertEqual(len(self.scene["doors"]), len(self.house["doors"]))
        self.assertEqual(len(self.scene["windows"]), len(self.house["windows"]))
        self.assertEqual(len(self.scene["objects"]), len(self.house["objects"]))

    def test_scene_bounds_unchanged(self):
        xs = [p["x"] for r in self.house["rooms"] for p in r["floorPolygon"]]
        zs = [p["z"] for r in self.house["rooms"] for p in r["floorPolygon"]]
        floor_ys = [p["y"] for r in self.house["rooms"] for p in r["floorPolygon"]]
        raw_heights = [
            round(max(p["y"] for p in w["polygon"]) - min(p["y"] for p in w["polygon"]), 4)
            for w in self.house["walls"]
        ]
        ys = floor_ys + raw_heights
        self.assertEqual(self.scene["bounds"], {
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
        })
        self.assertEqual(
            max(w["height"] for w in self.scene["walls"]), max(raw_heights)
        )

    def test_wall_canonical_geometry_unchanged(self):
        by_id = {w["id"]: w for w in self.house["walls"]}
        for wall in self.scene["walls"]:
            raw = by_id[wall["id"]]
            poly = raw["polygon"]
            ys = [p["y"] for p in poly]
            self.assertAlmostEqual(wall["height"], max(ys) - min(ys), places=4)
            expected_len = math.hypot(poly[1]["x"] - poly[0]["x"], poly[1]["z"] - poly[0]["z"])
            self.assertAlmostEqual(wall["length"], expected_len, places=4)

    def test_unrelated_walls_keep_exactly_one_full_segment(self):
        linked = set()
        for o in self.scene["doors"] + self.scene["windows"]:
            linked.update(str(x) for x in (o.get("wall_ids") or []))
        affected = {w["id"] for w in self.scene["walls"] if len(w["segments"]) != 1}
        # only the opening-linked (duplicated) walls are touched
        self.assertEqual(affected, linked)
        for wall in self.scene["walls"]:
            if wall["id"] in affected:
                continue
            self.assertEqual(wall["segments"], [{
                "start": 0.0,
                "end": round(wall["length"], 4),
                "y0": 0.0,
                "y1": round(wall["height"], 4),
            }])

    def test_no_fragmentation_or_micro_segments(self):
        self.assertEqual(sum(len(w["segments"]) for w in self.scene["walls"]), 106)
        self.assertLessEqual(max(len(w["segments"]) for w in self.scene["walls"]), 4)
        for wall in self.scene["walls"]:
            for seg in wall["segments"]:
                self.assertGreater(seg["end"] - seg["start"], 1e-3, wall["id"])
                self.assertGreater(seg["y1"] - seg["y0"], 1e-3, wall["id"])

    def test_objects_unchanged(self):
        by_id = {str(o["id"]): o for o in self.house["objects"]}
        for obj in self.scene["objects"]:
            raw = by_id[obj["id"]]
            pos = raw.get("position") or {}
            self.assertEqual(obj["position"], [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)])


class FixtureOpeningTests(unittest.TestCase):
    """The committed real ProcTHOR fixture also carries a door and a window."""

    @classmethod
    def setUpClass(cls):
        cls.scene = _scene(FIXTURE)

    def test_door_and_window_are_cut_into_their_walls(self):
        door = _opening(self.scene, "door|2|3")
        host = _wall(self.scene, door["wall_id"])
        self.assertGreater(len(host["segments"]), 1)
        u_mid = sum(door["local_x"]) / 2.0
        self.assertFalse(_segments_cover(host["segments"], u_mid, 1.0))

        window = _opening(self.scene, "window|2|0")
        host = _wall(self.scene, window["wall_id"])
        u_mid = sum(window["local_x"]) / 2.0
        sill = window["position"][1]
        self.assertTrue(_segments_cover(host["segments"], u_mid, sill - 0.1))
        self.assertFalse(_segments_cover(host["segments"], u_mid, sill + 0.5))

    def test_unlinked_collinear_duplicate_is_cut_geometrically(self):
        """A collinear duplicate that the door does not link must still open."""
        house = _house(FIXTURE)
        for door in house["doors"]:
            if door["id"] == "door|2|3":
                door.pop("wall1", None)  # break the linkage, keep the duplicate
        scene = build_scene3d(house)
        door = _opening(scene, "door|2|3")
        duplicate = _wall(scene, "wall|2|0.00|3.74|1.87|3.74")
        self.assertNotIn(duplicate["id"], door["wall_ids"])
        self.assertGreater(len(duplicate["segments"]), 1)
        u = _world_to_local(duplicate, door["position"][0], door["position"][2])
        self.assertFalse(_segments_cover(duplicate["segments"], u, 1.0))


class WallGeometryProbeTests(unittest.TestCase):
    """World-space rendering assertions through the shipped JS module."""

    @classmethod
    def setUpClass(cls):
        if NODE is None:
            raise unittest.SkipTest("node is not available for the wall geometry probe")
        cls.payload = {"scene": _scene(HOUSE_0034)}
        cls.probed = _run_probe(cls.payload)
        cls.boxes = {w["id"]: w["boxes"] for w in cls.probed["walls"]}

    def test_unrelated_wall_renders_as_the_same_full_box(self):
        wall = _wall(self.payload["scene"], "wall|3|13.06|10.88|17.41|10.88")
        boxes = self.boxes[wall["id"]]
        self.assertEqual(len(boxes), 1)
        box = boxes[0]
        self.assertAlmostEqual(box["size"][0], wall["length"], places=4)
        self.assertAlmostEqual(box["size"][1], wall["height"], places=4)
        self.assertAlmostEqual(box["center"][1], wall["height"] / 2, places=4)
        # the box's long axis endpoints coincide with the converted wall base
        theta = box["rotationY"]
        ux, uz = math.cos(theta), -math.sin(theta)
        half = box["size"][0] / 2
        e0 = [box["center"][0] - ux * half, box["center"][2] - uz * half]
        e1 = [box["center"][0] + ux * half, box["center"][2] + uz * half]
        for end, base in zip((e0, e1), wall["base"]):
            base3 = _to_three(base)
            self.assertAlmostEqual(end[0], base3[0], places=3)
            self.assertAlmostEqual(end[1], base3[2], places=3)

    def test_wall_box_is_aligned_with_the_wall_not_across_it(self):
        x_wall = _wall(self.payload["scene"], "wall|3|13.06|10.88|17.41|10.88")
        box = self.boxes[x_wall["id"]][0]
        # yaw 90 -> rotation 0: local x maps to Three x (world +X)
        self.assertAlmostEqual(box["rotationY"], 0.0, places=6)
        z_wall = _wall(self.payload["scene"], "wall|2|6.53|0.00|6.53|6.53")
        box = self.boxes[z_wall["id"]][0]
        # yaw 0 -> rotation +90deg: local x maps to Three -z (world +Z)
        self.assertAlmostEqual(box["rotationY"], math.pi / 2, places=6)

    def test_agent_doorway_crossing_never_intersects_a_solid_segment(self):
        crossing = _to_three(DOOR_CROSSING_POINT)
        for wall_id in (
            "wall|2|6.53|10.88|13.06|10.88",
            "wall|4|6.53|10.88|13.06|10.88",
        ):
            for box in self.boxes[wall_id]:
                self.assertFalse(
                    _point_in_box(crossing, box),
                    f"solid segment of {wall_id} blocks the agent doorway",
                )
        # sanity: without the derived segments the full wall *would* block it
        full = _full_wall_boxes(self.payload["scene"])
        blocked = [
            wid for wid, boxes in full.items()
            if any(_point_in_box(crossing, b) for b in boxes)
        ]
        self.assertIn("wall|2|6.53|10.88|13.06|10.88", blocked)
        self.assertIn("wall|4|6.53|10.88|13.06|10.88", blocked)

    def test_window_hole_is_open_but_sill_and_head_are_solid(self):
        window = _opening(self.payload["scene"], "window|4|0")
        cx, base_y, cz = _to_three(window["position"])
        centre = [cx, base_y + window["height"] / 2, cz]
        below = [cx, base_y - 0.2, cz]
        above = [cx, base_y + window["height"] + 0.2, cz]
        wall_boxes = self.boxes["wall|4|6.53|15.23|13.06|15.23"] + self.boxes[
            "wall|exterior|6.53|15.23|13.06|15.23"
        ]
        self.assertFalse(any(_point_in_box(centre, b) for b in wall_boxes))
        self.assertTrue(any(_point_in_box(below, b) for b in wall_boxes))
        self.assertTrue(any(_point_in_box(above, b) for b in wall_boxes))


@unittest.skipUnless(EPISODE_DIR.is_dir(), "real crossing episode not available")
class RealEpisodeCrossingTests(unittest.TestCase):
    """End-to-end: the canonical episode payload renders the real opening."""

    @classmethod
    def setUpClass(cls):
        from spatialforge.inspector.embodied_session import EmbodiedSession

        s = EmbodiedSession("deterministic", 0)
        s.set_episodes_roots([str(EPISODE_DIR.parent)])
        cls.payload = s.episode_scene3d("ep-bc-05ca85bf")
        cls.probed = _run_probe(cls.payload)

    def test_episode_house_is_house_0034_with_a_real_door_transition(self):
        self.assertIn("house_0034", self.payload["scene_meta"]["house_path"])
        timeline = [
            e for e in (self.payload["episode"]["trace_segments"] or [])
        ]
        self.assertTrue(timeline)
        pts = [tuple(p) for seg in timeline for p in seg]
        self.assertIn(tuple(DOOR_CROSSING[0]), pts)
        self.assertIn(tuple(DOOR_CROSSING[1]), pts)

    def test_real_crossing_point_is_open_in_the_rendered_scene(self):
        boxes = {w["id"]: w["boxes"] for w in self.probed["walls"]}
        crossing = _to_three(DOOR_CROSSING_POINT)
        for wall_id in (
            "wall|2|6.53|10.88|13.06|10.88",
            "wall|4|6.53|10.88|13.06|10.88",
        ):
            for box in boxes[wall_id]:
                self.assertFalse(_point_in_box(crossing, box), wall_id)


class SnapshotUpgradeTests(unittest.TestCase):
    """Persisted Round-3 `_scenes` snapshots must gain the opening truth."""

    @staticmethod
    def _stale_scene() -> dict:
        scene = _scene(FIXTURE)
        for wall in scene["walls"]:
            wall.pop("segments", None)
        for opening in scene["doors"] + scene["windows"]:
            for field in ("wall_id", "wall_ids", "local_x"):
                opening.pop(field, None)
            opening["position"] = [0.0, 0.0, 0.0]  # old wall-local value
        return scene

    def test_stale_snapshot_is_upgraded_in_place(self):
        from spatialforge.inspector.embodied_session import EmbodiedSession

        session = EmbodiedSession("deterministic", 0)
        upgraded = session._upgrade_scene_openings(self._stale_scene(), str(FIXTURE))
        door = _opening(upgraded, "door|2|3")
        self.assertIn("wall_id", door)
        self.assertNotEqual(door["position"], [0.0, 0.0, 0.0])
        self.assertGreater(len(_wall(upgraded, door["wall_id"])["segments"]), 1)

    def test_fresh_scene_is_not_modified(self):
        from spatialforge.inspector.embodied_session import EmbodiedSession

        scene = _scene(FIXTURE)
        before = json.dumps(scene, sort_keys=True)
        session = EmbodiedSession("deterministic", 0)
        after = session._upgrade_scene_openings(json.loads(before), str(FIXTURE))
        self.assertEqual(json.dumps(after, sort_keys=True), before)

    def test_scene_for_house_path_upgrades_persisted_snapshot(self):
        import tempfile
        from spatialforge.inspector.embodied_session import EmbodiedSession

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "_scenes").mkdir()
            (root / "_scenes" / f"{FIXTURE.stem}.json").write_text(
                json.dumps({"scene": self._stale_scene()}), encoding="utf-8"
            )
            session = EmbodiedSession("deterministic", 0)
            session.set_episodes_roots([root])
            scene = session._scene_for_house_path(str(FIXTURE))
        wall = _wall(scene, "wall|2|0.00|0.00|0.00|3.74")
        self.assertGreater(len(wall["segments"]), 1)


class FrontendWallGeometrySourceTests(unittest.TestCase):
    """The 3D renderer must consume the canonical segment truth."""

    @classmethod
    def setUpClass(cls):
        cls.god3d = (WEB / "god3d.js").read_text(encoding="utf-8")
        cls.wall_geometry = (WEB / "wall_geometry.js").read_text(encoding="utf-8")
        cls.server = (REPO / "spatialforge" / "inspector" / "embodied_server.py").read_text(
            encoding="utf-8"
        )

    def test_renderer_uses_shared_wall_geometry_module(self):
        self.assertIn('from "./wall_geometry.js"', self.god3d)
        self.assertIn("wallBoxes(w)", self.god3d)
        self.assertIn("openingBox(o)", self.god3d)
        self.assertIn("wall.segments", self.wall_geometry)
        self.assertNotIn("w.length, w.height, WALL_THICKNESS", self.god3d)

    def test_wall_box_orientation_is_wall_direction_not_wall_normal(self):
        # yaw is the Thor wall direction; rotation derives from world_space.
        self.assertIn("wallRotationY", self.wall_geometry)
        self.assertIn("worldToThree", self.wall_geometry)

    def test_module_is_routed(self):
        self.assertIn('"/wall_geometry.js"', self.server)


if __name__ == "__main__":
    unittest.main()
