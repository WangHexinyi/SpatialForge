"""Comprehensive tests for G2.0-E1 experiment dataset and scene foundation."""

import json
import random
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.export_v1_scenes import export_scenes, generate_scene_dict
from spatialforge.environment.scene import SceneObject, SceneState, load_scene_state
from spatialforge.environment.views import VIEW_IDS, generate_cardinal_views
from spatialforge.experiment.dataset import (
    INVERSE_ANSWER,
    NEG_LABELS,
    POS_LABELS,
    MultimodalSample,
    adapt_question_visual_refs,
    build_multimodal_sample,
    compute_dataset_hash,
    export_dataset_jsonl,
    format_single_question,
    get_camera_plan_for_scene,
    is_scene_visually_ambiguous,
    load_manifest_image_lookup,
    materialize_group_pool,
    sample_balanced_budget,
    visual_object_ref,
)


class TestV1SceneExport(unittest.TestCase):
    """Tests for exact reproduction of v1 scene generation."""

    def test_scene_000_matches_tracked_example(self):
        """Generated scene 0 matches tracked examples/scenes/scene_000.json."""
        tracked_path = Path("examples/scenes/scene_000.json")
        tracked_data = json.loads(tracked_path.read_text(encoding="utf-8"))

        gen_data = generate_scene_dict(0)
        self.assertEqual(gen_data["seed"], tracked_data["seed"])
        self.assertEqual(gen_data["camera"], tracked_data["camera"])
        self.assertEqual(len(gen_data["objects"]), len(tracked_data["objects"]))

        for gen_obj, track_obj in zip(gen_data["objects"], tracked_data["objects"]):
            self.assertEqual(gen_obj["name"], track_obj["name"])
            self.assertEqual(gen_obj["shape"], track_obj["shape"])
            self.assertEqual(gen_obj["color"], track_obj["color"])
            self.assertAlmostEqual(gen_obj["size"], track_obj["size"], places=9)
            for i in range(3):
                self.assertAlmostEqual(gen_obj["location"][i], track_obj["location"][i], places=9)

    def test_all_100_fixture_coordinates_match_exactly(self):
        """All 100 scenes match tests/fixtures/v1_front_back_reference.json coordinates."""
        fixture_path = Path("tests/fixtures/v1_front_back_reference.json")
        fixture_data = json.loads(fixture_path.read_text(encoding="utf-8"))
        scenes = fixture_data["scenes"]
        self.assertEqual(len(scenes), 100)

        for i, expected_scene in enumerate(scenes):
            gen_scene = generate_scene_dict(i)
            self.assertEqual(len(gen_scene["objects"]), len(expected_scene["objects"]))
            for gen_obj, exp_obj in zip(gen_scene["objects"], expected_scene["objects"]):
                self.assertEqual(gen_obj["name"], exp_obj["name"])
                for dim in range(3):
                    self.assertAlmostEqual(
                        gen_obj["location"][dim],
                        exp_obj["location"][dim],
                        places=9,
                        msg=f"Coordinate mismatch in scene {i} obj {gen_obj['name']}",
                    )

    def test_export_scenes_writes_deterministic_files(self):
        """export_scenes writes valid JSON files that can be loaded via load_scene_state."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            paths = export_scenes(tmp_path, count=5)
            self.assertEqual(len(paths), 5)
            for i, p in enumerate(paths):
                self.assertTrue(p.exists())
                state = load_scene_state(p)
                self.assertEqual(state.scene_id, f"scene_{i:03d}")
                self.assertEqual(len(state.objects), 4)

    def test_no_global_rng_mutation(self):
        """generate_scene_dict uses its own local Random instance and does not mutate global RNG."""
        random.seed(9999)
        val_before = random.random()

        random.seed(9999)
        generate_scene_dict(123)
        val_after = random.random()

        self.assertEqual(val_before, val_after)


class TestVisualReferenceAdapter(unittest.TestCase):
    """Tests for model-facing visual referring expressions and ambiguity filtering."""

    def _make_scene(self, descriptors):
        objects = []
        for i, (color, shape) in enumerate(descriptors):
            objects.append(
                SceneObject(
                    name=f"obj{i}",
                    shape=shape,
                    color=color,
                    location=(float(i), float(i), 0.4),
                    size=0.8,
                )
            )
        return SceneState(scene_id="test_scene", seed=0, objects=tuple(objects))

    def test_visual_object_ref_format(self):
        """Visual referring expression is 'the {color} {shape}'."""
        obj = SceneObject(name="obj0", shape="cube", color="red", location=(0.0, 0.0, 0.4), size=0.8)
        self.assertEqual(visual_object_ref(obj), "the red cube")

    def test_unambiguous_descriptors_accepted(self):
        """Distinct (color, shape) objects are not ambiguous."""
        scene = self._make_scene([("red", "cube"), ("blue", "sphere"), ("green", "cylinder"), ("yellow", "cube")])
        self.assertFalse(is_scene_visually_ambiguous(scene, 0, 1))
        self.assertFalse(is_scene_visually_ambiguous(scene, 0, 3))  # different colors

    def test_ambiguous_descriptors_rejected(self):
        """Duplicate (color, shape) objects in the same scene are detected as ambiguous."""
        scene = self._make_scene([("red", "cube"), ("red", "cube"), ("blue", "sphere"), ("yellow", "cylinder")])
        # Pair (0, 1) has two red cubes
        self.assertTrue(is_scene_visually_ambiguous(scene, 0, 1))
        # Pair (0, 2) references obj0 which shares descriptor with obj1
        self.assertTrue(is_scene_visually_ambiguous(scene, 0, 2))
        # Pair (2, 3) has unique blue sphere and yellow cylinder
        self.assertFalse(is_scene_visually_ambiguous(scene, 2, 3))

    def test_adapt_question_visual_refs_replaces_internal_names(self):
        """Question text internal references are cleanly rewritten to visual references."""
        scene = self._make_scene([("red", "sphere"), ("yellow", "cube")])
        raw_q = 'From this view, is object 0 ("obj0") left of or right of object 1 ("obj1")?'
        adapted_q = adapt_question_visual_refs(raw_q, scene, 0, 1)
        self.assertEqual(adapted_q, "From this view, is the red sphere left of or right of the yellow cube?")
        self.assertNotIn("obj0", adapted_q)
        self.assertNotIn("object 0", adapted_q)


class TestCameraPlans(unittest.TestCase):
    """Tests for primary camera plans A, C, and D."""

    def setUp(self):
        self.scene = SceneState(
            scene_id="scene_000",
            seed=0,
            objects=(SceneObject("obj0", "cube", "red", (0.0, 0.0, 0.4), 0.8),),
        )

    def test_plan_a_has_single_south_view(self):
        plan = get_camera_plan_for_scene("primary_a", self.scene)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0][0], "south")
        self.assertEqual(plan[0][1].position, (0.0, -6.0, 3.0))

    def test_plan_c_has_four_cardinals(self):
        plan = get_camera_plan_for_scene("primary_c", self.scene)
        self.assertEqual(len(plan), 4)
        view_ids = [vid for vid, _ in plan]
        self.assertEqual(view_ids, ["south", "east", "north", "west"])

    def test_plan_d_has_cardinals_plus_jitter(self):
        plan = get_camera_plan_for_scene("primary_d", self.scene)
        self.assertEqual(len(plan), 8)
        view_ids = [vid for vid, _ in plan]
        self.assertEqual(
            view_ids,
            [
                "south", "east", "north", "west",
                "south_jitter", "east_jitter", "north_jitter", "west_jitter",
            ],
        )

    def test_jitter_is_deterministic_and_seeded(self):
        """Camera plan D is 100% reproducible for the same scene seed."""
        plan1 = get_camera_plan_for_scene("primary_d", self.scene)
        plan2 = get_camera_plan_for_scene("primary_d", self.scene)
        for (_, c1), (_, c2) in zip(plan1, plan2):
            self.assertEqual(c1.position, c2.position)
            self.assertEqual(c1.look_at, c2.look_at)
            self.assertEqual(c1.up, c2.up)

    def test_jitter_differs_from_cardinal(self):
        """Jitter cameras are perturbed from the original cardinals."""
        plan = get_camera_plan_for_scene("primary_d", self.scene)
        for i in range(4):
            cardinal_cam = plan[i][1]
            jitter_cam = plan[i + 4][1]
            self.assertNotEqual(cardinal_cam.position, jitter_cam.position)


class TestMultimodalDatasetPipeline(unittest.TestCase):
    """Tests for dataset materialization, balancing, hashing, and isolation."""

    @classmethod
    def setUpClass(cls):
        # Generate 10 test scenes (0..7 for train, 8..9 for holdout)
        cls.train_scenes = [
            SceneState(
                scene_id=f"scene_{i:03d}",
                seed=i,
                objects=tuple(
                    SceneObject(
                        name=o["name"],
                        shape=o["shape"],
                        color=o["color"],
                        location=tuple(o["location"]),
                        size=o["size"],
                    )
                    for o in generate_scene_dict(i)["objects"]
                ),
            )
            for i in range(8)
        ]
        cls.holdout_scenes = [
            SceneState(
                scene_id=f"scene_{i:03d}",
                seed=i,
                objects=tuple(
                    SceneObject(
                        name=o["name"],
                        shape=o["shape"],
                        color=o["color"],
                        location=tuple(o["location"]),
                        size=o["size"],
                    )
                    for o in generate_scene_dict(i)["objects"]
                ),
            )
            for i in range(8, 10)
        ]

    def test_materialize_group_pool_a_c_d(self):
        """Pool materialization produces valid samples with expected counts."""
        pool_a, stats_a = materialize_group_pool(self.train_scenes, "primary_a")
        pool_c, stats_c = materialize_group_pool(self.train_scenes, "primary_c")
        pool_d, stats_d = materialize_group_pool(self.train_scenes, "primary_d")

        self.assertGreater(len(pool_a), 0)
        self.assertEqual(len(pool_c), len(pool_a) * 4)
        self.assertEqual(len(pool_d), len(pool_a) * 8)

        # In Group A, all views are 'south'
        self.assertTrue(all(s.view_id == "south" for s in pool_a))
        # Equal family distribution in Group A
        fam_counts = Counter(s.family for s in pool_a)
        self.assertEqual(len(set(fam_counts.values())), 1)

    def test_qasample_semantics_preserved_in_multimodal_sample(self):
        """MultimodalSample preserves exact QA fields while adapting surface text."""
        pool_a, _ = materialize_group_pool(self.train_scenes[:1], "primary_a")
        s = pool_a[0]
        self.assertTrue(s.sample_id.startswith("scene_000:single:south:"))
        self.assertIn(s.family, ("horizontal", "vertical", "depth", "near_far"))
        self.assertIn(s.answer, ("left", "right", "above", "below", "front", "behind", "nearer", "farther"))
        self.assertEqual(len(s.object_indices), 2)
        self.assertTrue(s.is_view_dependent)
        self.assertEqual(len(s.conversations), 2)
        self.assertEqual(s.conversations[0]["role"], "user")
        self.assertEqual(s.conversations[1]["role"], "assistant")
        self.assertEqual(s.conversations[1]["content"][0]["text"], s.answer)

    def test_sample_balanced_budget(self):
        """Balanced sampler enforces exact equal allocation across families."""
        pool_c, _ = materialize_group_pool(self.train_scenes, "primary_c")
        target_budget = 40  # 10 per family
        selected = sample_balanced_budget(pool_c, target_budget)
        self.assertEqual(len(selected), 40)

        fam_counts = Counter(s.family for s in selected)
        for fam in ("horizontal", "vertical", "depth", "near_far"):
            self.assertEqual(fam_counts[fam], 10)

    def test_dataset_hash_is_deterministic(self):
        """Identical sample selection yields identical SHA256 content hash."""
        pool_c, _ = materialize_group_pool(self.train_scenes, "primary_c")
        sel1 = sample_balanced_budget(pool_c, 40)
        sel2 = sample_balanced_budget(pool_c, 40)

        h1 = compute_dataset_hash(sel1)
        h2 = compute_dataset_hash(sel2)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_train_holdout_strict_isolation(self):
        """Train and holdout splits share zero scenes, sample IDs, or image paths."""
        train_pool, _ = materialize_group_pool(self.train_scenes, "primary_c")
        holdout_pool, _ = materialize_group_pool(self.holdout_scenes, "primary_c")

        train_scene_ids = {s.scene_id for s in train_pool}
        holdout_scene_ids = {s.scene_id for s in holdout_pool}
        self.assertTrue(train_scene_ids.isdisjoint(holdout_scene_ids))

        train_sample_ids = {s.sample_id for s in train_pool}
        holdout_sample_ids = {s.sample_id for s in holdout_pool}
        self.assertTrue(train_sample_ids.isdisjoint(holdout_sample_ids))

        train_images = {s.image_path for s in train_pool}
        holdout_images = {s.image_path for s in holdout_pool}
        self.assertTrue(train_images.isdisjoint(holdout_images))

    def test_export_dataset_jsonl(self):
        """JSONL export creates valid parseable JSON Lines."""
        pool_a, _ = materialize_group_pool(self.train_scenes[:2], "primary_a")
        with TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_dataset.jsonl"
            export_dataset_jsonl(pool_a, out_file)
            self.assertTrue(out_file.exists())

            lines = out_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), len(pool_a))
            first = json.loads(lines[0])
            self.assertEqual(first["id"], pool_a[0].sample_id)
            self.assertEqual(first["answer"], pool_a[0].answer)

    def test_load_manifest_image_lookup(self):
        """load_manifest_image_lookup correctly indexes (scene_id, view_id) -> image_path."""
        with TemporaryDirectory() as tmp_dir:
            manifest_dir = Path(tmp_dir)
            manifest = {
                "scene_id": "scene_000",
                "observations": [
                    {"view_id": "south", "image_path": "outputs/experiments/g2.0-e/rendered/scene_000/south.png"},
                    {"view_id": "east", "image_path": "outputs/experiments/g2.0-e/rendered/scene_000/east.png"},
                ],
            }
            scene_dir = manifest_dir / "scene_000"
            scene_dir.mkdir()
            (scene_dir / "manifest.json").write_text(json.dumps(manifest))

            lookup = load_manifest_image_lookup(manifest_dir)
            self.assertEqual(lookup[("scene_000", "south")], "outputs/experiments/g2.0-e/rendered/scene_000/south.png")
            self.assertEqual(lookup[("scene_000", "east")], "outputs/experiments/g2.0-e/rendered/scene_000/east.png")
            self.assertNotIn(("scene_000", "north"), lookup)

    def test_group_d_cell_balancing(self):
        """Balanced sampler across multiple views ensures max(cell) - min(cell) <= 1."""
        pool_d, _ = materialize_group_pool(self.train_scenes, "primary_d")
        budget = 32 * 3  # 3 per cell across 8 views x 4 families
        selected = sample_balanced_budget(pool_d, budget)
        self.assertEqual(len(selected), budget)

        cell_counts = Counter((s.view_id, s.family) for s in selected)
        # All 32 cells should be present
        self.assertEqual(len(cell_counts), 32)
        self.assertLessEqual(max(cell_counts.values()) - min(cell_counts.values()), 1)


class TestBlockerFixRegressions(unittest.TestCase):
    """Rigorous regression tests for blockers F-01 (scene truncation confound) and F-02 (directional answer label bias)."""

    def test_operand_inversion_mappings(self):
        """Strict mathematical bijection for all 8 directional terms."""
        expected = {
            "left": "right",
            "right": "left",
            "above": "below",
            "below": "above",
            "front": "behind",
            "behind": "front",
            "nearer": "farther",
            "farther": "nearer",
        }
        self.assertEqual(INVERSE_ANSWER, expected)
        for term, inv in expected.items():
            self.assertEqual(INVERSE_ANSWER[inv], term)

        for fam in ("horizontal", "vertical", "depth", "near_far"):
            pos = POS_LABELS[fam]
            neg = NEG_LABELS[fam]
            self.assertEqual(INVERSE_ANSWER[pos], neg)
            self.assertEqual(INVERSE_ANSWER[neg], pos)

    def test_format_single_question_structural(self):
        """format_single_question constructs natural questions matching options."""
        q_h = format_single_question("horizontal", "the red cube", "the green cylinder")
        self.assertEqual(q_h, "From this view, is the red cube left of or right of the green cylinder?")

        q_v = format_single_question("vertical", "the red cube", "the green cylinder")
        self.assertEqual(q_v, "From this view, is the red cube above or below the green cylinder?")

        q_d = format_single_question("depth", "the red cube", "the green cylinder")
        self.assertEqual(q_d, "From this view, is the red cube in front of or behind the green cylinder?")

        q_nf = format_single_question("near_far", "the red cube", "the green cylinder")
        self.assertEqual(
            q_nf,
            "From this view, is the red cube nearer to or farther from the camera than the green cylinder?",
        )

    def test_presentation_inversion_in_materialize_pool(self):
        """materialize_group_pool creates both canonical and inverted samples when normalize_presentation=True."""
        scene_dict = generate_scene_dict(0)
        scene = SceneState(
            scene_id="scene_000",
            seed=0,
            objects=tuple(
                SceneObject(
                    name=o["name"],
                    shape=o["shape"],
                    color=o["color"],
                    location=tuple(o["location"]),
                    size=o["size"],
                )
                for o in scene_dict["objects"]
            ),
        )
        pool_norm, _ = materialize_group_pool([scene], "primary_a", normalize_presentation=True)
        canonical = [s for s in pool_norm if s.presentation_order == 0]
        inverted = [s for s in pool_norm if s.presentation_order == 1]
        self.assertGreater(len(canonical), 0)
        self.assertGreater(len(inverted), 0)
        self.assertEqual(len(canonical) + len(inverted), len(pool_norm))

        for s in inverted:
            self.assertTrue(s.sample_id.endswith(":order_1"))
            self.assertFalse(s.source_sample_id.endswith(":order_1"))

    def test_serialized_multimodal_sample_preserves_new_fields(self):
        """MultimodalSample.to_dict preserves source_sample_id and presentation_order."""
        sample = MultimodalSample(
            sample_id="scene_000:single:south:0:1:horizontal:inv",
            source_sample_id="scene_000:single:south:0:1:horizontal",
            presentation_order=1,
            scene_id="scene_000",
            view_id="south",
            family="horizontal",
            image_path="outputs/experiments/g2.0-e/rendered/scene_000/south.png",
            question="From this view, is the blue cube left of or right of the red sphere?",
            answer="right",
            object_indices=(1, 0),
            tags=("horizontal", "view_dependent"),
            is_view_dependent=True,
            conversations=(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image_path": "outputs/experiments/g2.0-e/rendered/scene_000/south.png"},
                        {"type": "text", "text": "From this view, is the blue cube left of or right of the red sphere?"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "right"}]},
            ),
        )
        d = sample.to_dict()
        self.assertEqual(d["id"], sample.sample_id)
        self.assertEqual(d["source_sample_id"], "scene_000:single:south:0:1:horizontal")
        self.assertEqual(d["presentation_order"], 1)

    def test_generated_training_datasets_f01_scene_coverage(self):
        """Verify F-01 resolution: all three training groups cover all 78 eligible scenes."""
        exp_dir = Path("outputs/experiments/g2.0-e")
        manifest_path = exp_dir / "e1_dataset_manifest.json"
        if not manifest_path.exists():
            self.skipTest("Generated experiment datasets not present on disk")

        # 78 scenes: 000..079 excluding 007 and 056
        expected_scenes = {f"scene_{i:03d}" for i in range(80)} - {"scene_007", "scene_056"}
        self.assertEqual(len(expected_scenes), 78)

        for group in ("a", "c", "d"):
            jsonl_path = exp_dir / f"train_group_{group}.jsonl"
            self.assertTrue(jsonl_path.exists())
            records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines()]
            self.assertEqual(len(records), 1552, f"Group {group} budget should be 1552")

            scenes_in_group = {r["scene_id"] for r in records}
            self.assertEqual(
                scenes_in_group,
                expected_scenes,
                f"Group {group} does not cover all 78 eligible scenes; missing: {expected_scenes - scenes_in_group}",
            )

            scene_counts = Counter(r["scene_id"] for r in records)
            self.assertGreaterEqual(min(scene_counts.values()), 4, f"Group {group} minimum scene exposure too low")
            self.assertLessEqual(max(scene_counts.values()), 32, f"Group {group} maximum scene exposure too high")
            mean_exp = sum(scene_counts.values()) / len(scene_counts)
            self.assertAlmostEqual(mean_exp, 1552 / 78, places=2)

    def test_generated_training_datasets_f02_label_balance(self):
        """Verify F-02 resolution: all three training groups have exact 50/50 label parity."""
        exp_dir = Path("outputs/experiments/g2.0-e")
        manifest_path = exp_dir / "e1_dataset_manifest.json"
        if not manifest_path.exists():
            self.skipTest("Generated experiment datasets not present on disk")

        for group in ("a", "c", "d"):
            jsonl_path = exp_dir / f"train_group_{group}.jsonl"
            records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines()]

            fam_labels = defaultdict(Counter)
            for r in records:
                fam_labels[r["family"]][r["answer"]] += 1

            for fam in ("horizontal", "vertical", "depth", "near_far"):
                pos = POS_LABELS[fam]
                neg = NEG_LABELS[fam]
                pos_c = fam_labels[fam][pos]
                neg_c = fam_labels[fam][neg]
                self.assertEqual(
                    pos_c,
                    194,
                    f"Group {group} family {fam} positive label count {pos_c} != 194",
                )
                self.assertEqual(
                    neg_c,
                    194,
                    f"Group {group} family {fam} negative label count {neg_c} != 194",
                )
                self.assertEqual(pos_c + neg_c, 388)

    def test_holdout_datasets_disjoint_and_balanced(self):
        """Holdout S1 and S2 cover exactly 10 holdout scenes each and are strictly disjoint from train."""
        exp_dir = Path("outputs/experiments/g2.0-e")
        manifest_path = exp_dir / "e1_dataset_manifest.json"
        if not manifest_path.exists():
            self.skipTest("Generated experiment datasets not present on disk")

        train_scenes = {f"scene_{i:03d}" for i in range(80)}
        holdout_range = {f"scene_{i:03d}" for i in range(80, 100)}

        for name in ("s1_cardinal", "s2_jitter"):
            jsonl_path = exp_dir / f"holdout_{name}.jsonl"
            self.assertTrue(jsonl_path.exists())
            records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines()]
            self.assertEqual(len(records), 1136)

            scenes = {r["scene_id"] for r in records}
            self.assertTrue(scenes.issubset(holdout_range))
            self.assertTrue(scenes.isdisjoint(train_scenes))

            fam_counts = Counter(r["family"] for r in records)
            for fam in ("horizontal", "vertical", "depth", "near_far"):
                self.assertEqual(fam_counts[fam], 284)


if __name__ == "__main__":
    unittest.main()


