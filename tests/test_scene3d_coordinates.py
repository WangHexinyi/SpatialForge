"""3D semantic coordinate-system consistency tests.

The 3D semantic God View must represent the same ProcTHOR world as the Unity
God View and the canonical 2D projection. AI2-THOR is left-handed (+X right,
+Y up, +Z forward) while Three.js is right-handed (+X right, +Y up, +Z toward
the viewer), so the renderer applies exactly one mirror to every entity:

    worldToThree([x, y, z]) -> [x, y, -z]

These tests use a real asymmetric ProcTHOR house (house_0034) plus a real
episode (ep-run-e46fe1fd on val_house_0) and prove, numerically and through the
shipped browser modules under node:

* the single conversion is applied to agent / target / door / walls / object;
* the 2D projection and a top-down 3D camera above the scene show the same
  plan orientation (no mirror, no under-floor viewing);
* agent yaw / FOV point the same way in 2D and 3D;
* the door opening stays on the correct wall;
* no wall/object fragmentation is introduced.
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
HOUSE_0034 = REPO / "outputs" / "embodied_bc" / "houses" / "house_0034.json"
UAT_EPISODE_DIR = REPO / "outputs" / "episodes_postfix" / "ep-run-e46fe1fd"
UAT_HOUSE = Path("/root/autodl-tmp/val_house_0.json")
WALL_PROBE = REPO / "tests" / "tools" / "wall_geometry_probe.mjs"
WORLD_PROBE = REPO / "tests" / "tools" / "world_space_probe.mjs"
SCENE2D_PROBE = REPO / "tests" / "tools" / "scene2d_probe.mjs"
NODE = shutil.which("node")

# Real, asymmetric points from house_0034 (Kitchen / LivingRoom / Bedroom split)
# and the real target of episode ep-bc-05ca85bf.
AGENT_POSE = [9.5, 0.95, 5.75]            # house_0034 agent_start, yaw 90 (east)
TARGET_POSE = [4.280856609344482, 0.9961651563644409, 12.262551307678223]
OBJECT_POSE = [8.989452058675672, 0.8416028022766113, 0.3321602284908295]  # Fridge|2|2
DOOR_ID = "door|2|4"
HOST_WALL = "wall|2|6.53|10.88|13.06|10.88"
DUP_WALL = "wall|4|6.53|10.88|13.06|10.88"
CROSS_WALL = "wall|2|6.53|0.00|6.53|6.53"


def _house(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _run_probe(script: Path, *docs) -> dict:
    if NODE is None:
        raise unittest.SkipTest("node is not available for the JS probes")
    with tempfile.TemporaryDirectory() as tmp:
        args = [NODE, str(script)]
        for i, doc in enumerate(docs):
            p = Path(tmp) / f"in{i}.json"
            p.write_text(json.dumps(doc), encoding="utf-8")
            args.append(str(p))
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise AssertionError(f"node probe {script.name} failed: {proc.stderr}")
    return json.loads(proc.stdout)


def _run_world_probe(points, dirs=None, yaws=None, bounds=None) -> dict:
    return _run_probe(WORLD_PROBE, {
        "points": points,
        "dirs": dirs or [],
        "yaws": yaws or [],
        "bounds": bounds,
    })


def _run_scene2d_probe(payload: dict, points) -> dict:
    return _run_probe(SCENE2D_PROBE, payload, {"points": points})


def _run_wall_probe(scene: dict) -> dict:
    return _run_probe(WALL_PROBE, {"scene": scene})


def _to_three(p: list) -> list:
    """Test-side mirror of the shipped canonical world_space.worldToThree."""
    return [p[0], p[1], -p[2]]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _normalize(a):
    length = math.sqrt(_dot(a, a))
    return [v / length for v in a]


def _camera_basis(position, target, up=(0.0, 1.0, 0.0)):
    """Three.js lookAt basis: (screen right, screen up, camera backward)."""
    z = _normalize(_sub(position, target))
    x = _normalize(_cross(list(up), z))
    y = _cross(z, x)
    return x, y, z


def _plan_up(up_screen):
    """Horizontal (xz) component of the screen-up vector, normalized."""
    horizontal = [up_screen[0], 0.0, up_screen[2]]
    if math.sqrt(_dot(horizontal, horizontal)) < 1e-9:
        return [0.0, 0.0, 0.0]
    return _normalize(horizontal)


def _point_in_box(point, box, eps=1e-6):
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


def _point_line_distance_xz(p, a, b):
    ux, uz = b[0] - a[0], b[2] - a[2]
    length = math.hypot(ux, uz)
    return abs((p[0] - a[0]) * (uz / length) - (p[2] - a[2]) * (ux / length))


class CanonicalConversionTests(unittest.TestCase):
    """The single shipped world->Three conversion, executed under node."""

    def test_world_to_three_is_the_single_z_mirror(self):
        points = [AGENT_POSE, TARGET_POSE, OBJECT_POSE, [0, 0, 0], [1, 2, 3]]
        out = _run_world_probe(points)
        self.assertEqual(out["points"], [_to_three(p) for p in points])
        self.assertEqual(out["roundtrip"], points)

    def test_directions_mirror_with_the_same_conversion(self):
        out = _run_world_probe([], dirs=[[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.assertEqual(out["dirs"], [[0, 0, -1], [1, 0, 0], [0, 1, 0]])

    def test_yaw_convention(self):
        out = _run_world_probe([], yaws=[0, 90, 180, 270])
        # Thor forward (sin, cos) -> Three (sin, -cos)
        expected = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        for got, want in zip(out["yaw_dirs"], expected):
            self.assertAlmostEqual(got[0], want[0], places=6)
            self.assertAlmostEqual(got[1], want[1], places=6)
        # rotation.y aligning a +X-length wall box with that direction
        expected_rot = [math.pi / 2, 0.0, -math.pi / 2, -math.pi]
        for got, want in zip(out["yaw_rotations"], expected_rot):
            self.assertAlmostEqual(math.atan2(math.sin(got), math.cos(got)),
                                   math.atan2(math.sin(want), math.cos(want)), places=6)

    def test_world_bounds_are_mirrored_and_swapped(self):
        out = _run_world_probe([], bounds={"min": [0, 0, 0], "max": [10, 3, 4]})
        self.assertEqual(out["bounds"], {"min": [0, 0, -4], "max": [10, 3, 0]})

    def test_fov_wedge_angle_points_along_the_converted_forward(self):
        """CircleGeometry angle phi -> Three (cos phi, 0, -sin phi); the FOV
        wedge uses wallRotationY(yaw), so its centre must be the agent forward."""
        yaws = [0, 90, 180, 270]
        out = _run_world_probe([], yaws=yaws)
        for forward, rot in zip(out["yaw_dirs"], out["yaw_rotations"]):
            wedge_dir = [math.cos(rot), -math.sin(rot)]
            self.assertAlmostEqual(wedge_dir[0], forward[0], places=6)
            self.assertAlmostEqual(wedge_dir[1], forward[1], places=6)

    def test_test_mirror_matches_the_shipped_conversion(self):
        points = [AGENT_POSE, TARGET_POSE, OBJECT_POSE]
        out = _run_world_probe(points)
        self.assertEqual(out["points"], [_to_three(p) for p in points])


class PlanOrientationConsistencyTests(unittest.TestCase):
    """2D projection and a 3D top-down camera above agree on the plan."""

    @classmethod
    def setUpClass(cls):
        cls.scene = build_scene3d(_house(HOUSE_0034))
        cls.door = _opening(cls.scene, DOOR_ID)
        cls.wall_a = _wall(cls.scene, HOST_WALL)
        cls.wall_b = _wall(cls.scene, CROSS_WALL)
        cls.points = [
            AGENT_POSE,
            TARGET_POSE,
            cls.door["position"],
            cls.wall_a["base"][0],
            cls.wall_a["base"][1],
            cls.wall_b["base"][0],
            cls.wall_b["base"][1],
            OBJECT_POSE,
        ]
        cls.names = ["agent", "target", "door", "wall_a0", "wall_a1",
                     "wall_b0", "wall_b1", "object"]
        cls.payload = episode_scene_payload(
            cls.scene,
            spawn={"position": AGENT_POSE, "yaw": 90.0, "horizon": 0.0},
            timeline=[
                {"idx": 1, "step": 1, "action": "MoveAhead", "action_origin": "model",
                 "agent": {"position": [AGENT_POSE[0] + 0.25, AGENT_POSE[1], AGENT_POSE[2]],
                           "rotation_yaw_deg": 90.0, "camera_horizon_deg": 0.0}},
            ],
            target_positions=[TARGET_POSE],
            target_object_ids=["Box|surface|7|61"],
            terminal=False,
        )
        cls.world = _run_world_probe(
            cls.points, bounds=cls.scene["bounds"], yaws=[0, 90, 180, 270]
        )
        cls.three = cls.world["points"]
        cls.scene2d = _run_scene2d_probe(cls.payload, cls.points)
        cls.top = cls.world["top"]
        cls.right, cls.up_screen, _ = _camera_basis(cls.top["position"], cls.top["target"])
        cls.up_plan = _plan_up(cls.up_screen)
        cls.scale2d = cls.scene2d["transform"]["scale"]

    def test_scene_is_asymmetric(self):
        """Guard: a symmetric layout could hide a mirror bug."""
        agent, target, obj = self.points[0], self.points[1], self.points[7]
        self.assertLess(target[0], agent[0])          # target is west of agent
        self.assertGreater(target[2], agent[2])       # target is north of agent
        self.assertLess(obj[2], agent[2])             # object is south of agent
        self.assertGreater(self.wall_a["length"], 6.0)
        self.assertAlmostEqual(self.wall_b["length"], 6.529, places=3)

    def test_three_positions_are_the_canonical_mirror(self):
        for name, p, t in zip(self.names, self.points, self.three):
            self.assertEqual(t, _to_three(p), name)

    def test_2d_projection_is_a_positive_scale_of_world_xz(self):
        for i in range(len(self.points)):
            for j in range(len(self.points)):
                p, q = self.points[i], self.points[j]
                pi = self.scene2d["projected_points"][i]
                pj = self.scene2d["projected_points"][j]
                self.assertAlmostEqual(
                    pj[0] - pi[0], (q[0] - p[0]) * self.scale2d, places=4,
                    msg=f"2D x: {self.names[i]} -> {self.names[j]}",
                )
                # canvas y grows downward; world +Z must point up on screen
                self.assertAlmostEqual(
                    -(pj[1] - pi[1]), (q[2] - p[2]) * self.scale2d, places=4,
                    msg=f"2D z: {self.names[i]} -> {self.names[j]}",
                )

    def test_3d_top_down_plan_matches_2d_for_every_pair(self):
        for i in range(len(self.points)):
            for j in range(len(self.points)):
                p, q = self.points[i], self.points[j]
                # 3D top camera: project canonical Three points on the camera basis
                di = _sub(self.three[i], self.top["position"])
                dj = _sub(self.three[j], self.top["position"])
                dsx = _dot(dj, self.right) - _dot(di, self.right)
                dsy = _dot(dj, self.up_plan) - _dot(di, self.up_plan)
                self.assertAlmostEqual(
                    dsx, q[0] - p[0], places=4,
                    msg=f"3D top x: {self.names[i]} -> {self.names[j]}",
                )
                self.assertAlmostEqual(
                    dsy, q[2] - p[2], places=4,
                    msg=f"3D top z: {self.names[i]} -> {self.names[j]}",
                )

    def test_target_relation_is_west_and_north_in_all_three_views(self):
        agent3, target3 = self.three[0], self.three[1]
        # world
        self.assertLess(target3[0], agent3[0])
        self.assertGreater(-target3[2], -agent3[2])  # world +Z north
        # 2D canvas (y down)
        agent2d = self.scene2d["projected_points"][0]
        target2d = self.scene2d["projected_points"][1]
        self.assertLess(target2d[0], agent2d[0])     # west = left
        self.assertLess(target2d[1], agent2d[1])     # north = up
        # 3D top camera
        d = _sub(target3, self.top["position"])
        a = _sub(agent3, self.top["position"])
        self.assertLess(_dot(d, self.right), _dot(a, self.right))
        self.assertGreater(_dot(d, self.up_plan), _dot(a, self.up_plan))


class TopDownCameraTests(unittest.TestCase):
    """The shipped top-view camera is above the scene and matches 2D axes."""

    @classmethod
    def setUpClass(cls):
        cls.scene = build_scene3d(_house(HOUSE_0034))
        cls.world = _run_world_probe(
            [AGENT_POSE, TARGET_POSE, OBJECT_POSE], bounds=cls.scene["bounds"]
        )
        cls.top = cls.world["top"]
        cls.right, cls.up_screen, _ = _camera_basis(cls.top["position"], cls.top["target"])
        cls.up_plan = _plan_up(cls.up_screen)

    def test_camera_is_above_the_scene(self):
        max_y = self.scene["bounds"]["max"][1]
        self.assertGreater(self.top["position"][1], max_y)
        self.assertGreater(self.top["position"][1], 0.0)
        # Three y equals world y (no vertical flip)
        for p, t in zip(self.world["points"], [_to_three(p) for p in [AGENT_POSE, TARGET_POSE, OBJECT_POSE]]):
            self.assertAlmostEqual(p[1], t[1], places=6)

    def test_screen_axes_match_2d(self):
        # world +X -> screen right
        self.assertGreater(_dot(self.right, _to_three([1, 0, 0])), 0.999)
        # world +Z -> screen up (Three -Z)
        self.assertGreater(_dot(self.up_plan, _to_three([0, 0, 1])), 0.999)

    def test_larger_world_z_is_higher_on_screen(self):
        low = _to_three([0.0, 0.0, 0.0])
        high = _to_three([0.0, 0.0, 1.0])
        base = self.top["position"]
        sy_low = _dot(_sub(low, base), self.up_plan)
        sy_high = _dot(_sub(high, base), self.up_plan)
        self.assertGreater(sy_high, sy_low)


class AgentHeadingConsistencyTests(unittest.TestCase):
    """Agent yaw / FOV direction is identical in world, 2D and 3D."""

    def test_house_agent_yaw_90_faces_east_everywhere(self):
        out = _run_world_probe([], yaws=[90])
        three_dir = out["yaw_dirs"][0]
        # Three xz -> plan (x, -z) must equal the 2D up-positive arrow (sin, cos)
        self.assertAlmostEqual(three_dir[0], 1.0, places=6)
        self.assertAlmostEqual(-three_dir[1], 0.0, places=6)

    def test_uat_episode_agent_yaw_and_movement_agree(self):
        if not UAT_EPISODE_DIR.is_dir():
            self.skipTest("UAT episode not available")
        from spatialforge.inspector.embodied_session import EmbodiedSession

        session = EmbodiedSession("deterministic", 0)
        session.set_episodes_roots([str(UAT_EPISODE_DIR.parent)])
        payload = session.episode_scene3d("ep-run-e46fe1fd")
        segments = payload["episode"]["trace_segments"] or []
        self.assertTrue(segments)
        seg = segments[0]
        self.assertGreaterEqual(len(seg), 2)
        p0, p1 = seg[0], seg[1]
        yaw = payload["episode"]["agent"]["rotation_yaw_deg"]
        out = _run_world_probe([], yaws=[yaw])
        three_dir = out["yaw_dirs"][0]
        # 3D forward plan direction
        fwd_plan = [three_dir[0], -three_dir[1]]
        # real movement plan direction
        dx, dz = p1[0] - p0[0], p1[2] - p0[2]
        length = math.hypot(dx, dz)
        self.assertGreater(length, 0.1)
        self.assertAlmostEqual(fwd_plan[0], dx / length, places=6)
        self.assertAlmostEqual(fwd_plan[1], dz / length, places=6)
        # 2D arrow uses the same world direction
        yaw_rad = math.radians(yaw)
        self.assertAlmostEqual(fwd_plan[0], math.sin(yaw_rad), places=6)
        self.assertAlmostEqual(fwd_plan[1], math.cos(yaw_rad), places=6)


class DoorwayPlacementTests(unittest.TestCase):
    """After the coordinate fix the door stays on its wall in Three space."""

    @classmethod
    def setUpClass(cls):
        cls.scene = build_scene3d(_house(HOUSE_0034))
        cls.door = _opening(cls.scene, DOOR_ID)
        cls.probed = _run_wall_probe(cls.scene)
        cls.boxes = {w["id"]: w["boxes"] for w in cls.probed["walls"]}

    def test_door_center_is_collinear_with_the_converted_host_wall(self):
        host = _wall(self.scene, HOST_WALL)
        a3 = _to_three(host["base"][0])
        b3 = _to_three(host["base"][1])
        door3 = _to_three(self.door["position"])
        self.assertLess(_point_line_distance_xz(door3, a3, b3), 0.01)

    def test_door_center_is_not_blocked_at_agent_height(self):
        door3 = _to_three(self.door["position"])
        probe_point = [door3[0], 0.9, door3[2]]
        for wall_id in (HOST_WALL, DUP_WALL):
            for box in self.boxes[wall_id]:
                self.assertFalse(
                    _point_in_box(probe_point, box),
                    f"{wall_id} seals the doorway in Three space",
                )


class NoFragmentationInvariantTests(unittest.TestCase):
    """The coordinate fix must not add boxes or change segment sizes."""

    @classmethod
    def setUpClass(cls):
        cls.scene = build_scene3d(_house(HOUSE_0034))
        cls.probed = _run_wall_probe(cls.scene)
        cls.boxes = {w["id"]: w["boxes"] for w in cls.probed["walls"]}

    def test_wall_box_count_equals_segment_count(self):
        total_segments = sum(len(w["segments"]) for w in self.scene["walls"])
        total_boxes = sum(len(w["boxes"]) for w in self.probed["walls"])
        self.assertEqual(total_boxes, total_segments)
        for wall in self.scene["walls"]:
            self.assertEqual(len(self.boxes[wall["id"]]), len(wall["segments"]))
            for seg, box in zip(wall["segments"], self.boxes[wall["id"]]):
                self.assertAlmostEqual(box["size"][0], seg["end"] - seg["start"], places=4)
                self.assertAlmostEqual(box["size"][1], seg["y1"] - seg["y0"], places=4)

    def test_scene_counts_unchanged(self):
        house = _house(HOUSE_0034)
        self.assertEqual(len(self.scene["rooms"]), len(house["rooms"]))
        self.assertEqual(len(self.scene["walls"]), len(house["walls"]))
        self.assertEqual(len(self.scene["doors"]), len(house["doors"]))
        self.assertEqual(len(self.scene["windows"]), len(house["windows"]))
        self.assertEqual(len(self.scene["objects"]), len(house["objects"]))


@unittest.skipUnless(
    UAT_EPISODE_DIR.is_dir() and UAT_HOUSE.is_file(),
    "UAT episode / house not available",
)
class UatEpisodeCoordinateTests(unittest.TestCase):
    """ep-run-e46fe1fd (val_house_0): real agent/target plan agreement."""

    @classmethod
    def setUpClass(cls):
        from spatialforge.inspector.embodied_session import EmbodiedSession

        session = EmbodiedSession("deterministic", 0)
        session.set_episodes_roots([str(UAT_EPISODE_DIR.parent)])
        cls.payload = session.episode_scene3d("ep-run-e46fe1fd")
        cls.scene = cls.payload["scene"]
        agent = cls.payload["episode"]["agent"]["position"]
        targets = [t["position"] for t in cls.payload["episode"]["targets"]]
        cls.points = [agent] + targets
        cls.world = _run_world_probe(cls.points, bounds=cls.scene["bounds"])
        cls.three = cls.world["points"]
        cls.scene2d = _run_scene2d_probe(cls.payload, cls.points)
        cls.top = cls.world["top"]
        cls.right, cls.up_screen, _ = _camera_basis(cls.top["position"], cls.top["target"])
        cls.up_plan = _plan_up(cls.up_screen)
        cls.scale2d = cls.scene2d["transform"]["scale"]

    def test_agent_and_target_use_the_same_mirror(self):
        for p, t in zip(self.points, self.three):
            self.assertEqual(t, _to_three(p))

    def test_agent_target_offsets_agree_between_2d_and_3d_top(self):
        self.assertGreaterEqual(len(self.points), 2)
        agent2d = self.scene2d["projected_points"][0]
        for i in range(1, len(self.points)):
            p = self.points[0]
            q = self.points[i]
            proj = self.scene2d["projected_points"][i]
            self.assertAlmostEqual(proj[0] - agent2d[0], (q[0] - p[0]) * self.scale2d, places=4)
            self.assertAlmostEqual(-(proj[1] - agent2d[1]), (q[2] - p[2]) * self.scale2d, places=4)
            di = _sub(self.three[0], self.top["position"])
            dj = _sub(self.three[i], self.top["position"])
            self.assertAlmostEqual(
                _dot(dj, self.right) - _dot(di, self.right), q[0] - p[0], places=4
            )
            self.assertAlmostEqual(
                _dot(dj, self.up_plan) - _dot(di, self.up_plan), q[2] - p[2], places=4
            )


class FrontendCoordinateSourceTests(unittest.TestCase):
    """Every 3D entity must flow through the single canonical conversion."""

    @classmethod
    def setUpClass(cls):
        cls.god3d = (WEB / "god3d.js").read_text(encoding="utf-8")
        cls.wall_geometry = (WEB / "wall_geometry.js").read_text(encoding="utf-8")
        cls.world_space = (WEB / "world_space.js").read_text(encoding="utf-8")
        cls.server = (REPO / "spatialforge" / "inspector" / "embodied_server.py").read_text(
            encoding="utf-8"
        )

    def test_single_conversion_module(self):
        self.assertIn("export function worldToThree", self.world_space)
        self.assertIn("return [Number(p[0]), Number(p[1]), -Number(p[2])]", self.world_space)
        self.assertIn('from "./world_space.js"', self.god3d)
        self.assertIn('from "./world_space.js"', self.wall_geometry)

    def test_renderer_converts_every_entity(self):
        for token in (
            "worldToThree",
            "worldDirToThree",
            "worldBoundsToThree",
            "topViewCamera",
            "threeToWorld",
        ):
            self.assertIn(token, self.god3d, token)
        # floors / walls / openings / objects / agent / target / trajectory / FOV
        self.assertIn(".map((p) => worldToThree(p))", self.god3d)   # floors
        self.assertIn("worldToThree(o.aabb.center)", self.god3d)    # object AABB
        self.assertIn("worldToThree(o.position)", self.god3d)       # object fallback
        self.assertIn("worldToThree(agent.position)", self.god3d)   # agent
        self.assertIn("worldToThree(t.position)", self.god3d)       # target
        self.assertIn("worldToThree(spawn.position)", self.god3d)   # spawn
        self.assertIn("worldToThree(p)", self.god3d)                # trajectory
        self.assertIn("worldToThree(wall.base[0])", self.wall_geometry)
        self.assertIn("worldToThree(opening.position)", self.wall_geometry)
        self.assertIn("wallRotationY(agent.rotation_yaw_deg", self.god3d)

    def test_no_scattered_sign_flips_remain(self):
        # old buggy patterns: raw world positions / yaw rotations in the renderer
        self.assertNotIn("mesh.position.set(o.position[0]", self.god3d)
        self.assertNotIn("lines.rotation.y = yaw", self.god3d)
        self.assertNotIn("wedge.rotation.z = -yaw", self.god3d)
        self.assertNotIn("mesh.rotation.y = ((w.yaw_deg || 0) * Math.PI) / 180", self.god3d)
        self.assertNotIn("mesh.rotation.y = ((o.yaw_deg || 0) * Math.PI) / 180", self.god3d)

    def test_world_space_module_is_routed(self):
        self.assertIn('"/world_space.js"', self.server)


if __name__ == "__main__":
    unittest.main()
