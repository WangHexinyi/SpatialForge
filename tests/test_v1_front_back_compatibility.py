import json
import unittest
from itertools import combinations
from pathlib import Path

from spatialforge.environment.camera import CameraPose
from spatialforge.environment.geometry import camera_depth


class V1FrontBackCompatibilityTests(unittest.TestCase):

    def test_v1_front_back_matches_camera_depth_on_reference_corpus(self):
        camera = CameraPose(
            position=(0.0, -6.0, 3.0),
            look_at=(0.0, 0.0, 4.0),
        )

        fixture_path = Path("tests/fixtures/v1_front_back_reference.json")
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        scenes = fixture["scenes"]
        self.assertEqual(len(scenes), 100)

        checked_pairs = 0

        for scene in scenes:
            objects = scene["objects"]

            for a, b in combinations(objects, 2):
                ay = a["location"][1]
                by = b["location"][1]

                if abs(ay - by) < 1e-12:
                    continue

                v1_a_front = ay < by

                depth_a = camera_depth(tuple(a["location"]), camera)
                depth_b = camera_depth(tuple(b["location"]), camera)

                v2_a_front = depth_a < depth_b

                self.assertEqual(
                    v1_a_front,
                    v2_a_front,
                    msg=(
                        f"front/back mismatch in {scene['scene_id']}: "
                        f"{a['name']} {a['location']} vs "
                        f"{b['name']} {b['location']}"
                    ),
                )

                checked_pairs += 1

        self.assertEqual(checked_pairs, 600)


if __name__ == "__main__":
    unittest.main()
