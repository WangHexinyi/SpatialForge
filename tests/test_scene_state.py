"""Tests for SceneObject, SceneState, and v1 JSON loading."""

import shutil
import tempfile
import unittest
from pathlib import Path

from spatialforge.environment.scene import SceneObject, SceneState, load_scene_state
from spatialforge.environment.camera import Vec3


class SceneStateTests(unittest.TestCase):

    def test_load_v1_scene_json(self):
        """An existing v1 JSON scene can be loaded into SceneState."""
        path = Path("examples/scenes/scene_000.json")
        self.assertTrue(path.exists(), f"missing reference file: {path}")

        state = load_scene_state(path)

        self.assertIsInstance(state, SceneState)
        self.assertEqual(state.scene_id, "scene_000")
        self.assertEqual(state.seed, 0)
        self.assertEqual(len(state.objects), 4)

    def test_object_count_and_fields_survive_loading(self):
        """Object count and semantic fields are preserved unchanged."""
        path = Path("examples/scenes/scene_000.json")
        state = load_scene_state(path)

        self.assertEqual(len(state.objects), 4)

        first = state.objects[0]
        self.assertIsInstance(first, SceneObject)
        self.assertEqual(first.name, "obj0")
        self.assertEqual(first.shape, "sphere")
        self.assertEqual(first.color, "red")
        self.assertIsInstance(first.location, tuple)
        self.assertEqual(len(first.location), 3)
        self.assertIsInstance(first.size, float)

    def test_scene_state_is_frozen(self):
        """SceneState is immutable after construction."""
        path = Path("examples/scenes/scene_000.json")
        state = load_scene_state(path)

        with self.assertRaises(AttributeError):
            state.seed = 999  # type: ignore[misc]

    def test_scene_object_is_frozen(self):
        """SceneObject is immutable after construction."""
        obj = SceneObject(
            name="test",
            shape="cube",
            color="blue",
            location=(0.0, 0.0, 0.0),
            size=1.0,
        )
        with self.assertRaises(AttributeError):
            obj.name = "changed"  # type: ignore[misc]

    def test_custom_scene_id(self):
        """load_scene_state accepts a custom scene_id override."""
        path = Path("examples/scenes/scene_000.json")
        state = load_scene_state(path, scene_id="custom_id")
        self.assertEqual(state.scene_id, "custom_id")

    def test_multiple_scenes_load(self):
        """At least 5 scene files can be loaded without error."""
        source = Path("examples/scenes/scene_000.json")

        with tempfile.TemporaryDirectory() as temp_dir:
            files = []
            for index in range(5):
                path = Path(temp_dir) / f"scene_{index:03d}.json"
                shutil.copyfile(source, path)
                files.append(path)

            self.assertGreaterEqual(len(files), 5)

            loaded = 0
            for path in files[:5]:
                state = load_scene_state(path)
                self.assertGreater(len(state.objects), 0)
                loaded += 1

            self.assertEqual(loaded, 5)


if __name__ == "__main__":
    unittest.main()
