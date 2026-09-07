"""Tests for G2.0-E3.1 Evaluator Hardening and Experiment Protocol Freeze."""

import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import torch

from spatialforge.experiment.evaluation import (
    DIRECTIONAL_VOCABULARY,
    VSR_ALL_DIMS,
    VSR_BROADER_DIMS,
    VSR_DIRECT_DIMS,
    VSR_RELATION_TO_DIM,
    EvaluationPredictionRecord,
    aggregate_metrics,
    classify_vsr_relation,
    compute_slice_metrics,
    parse_directional_answer,
    parse_yesno_answer,
)
from spatialforge.experiment.protocol import (
    E1_MANIFEST_PATH,
    ENGINEERING_SEED,
    EVAL_DATASET_PATHS,
    EXPERIMENT_ID,
    FORMAL_SEEDS,
    FROZEN_E1_DATASET_HASHES,
    FROZEN_E1_GROUP_TO_FILE,
    GROUPS,
    HOLDOUT_SCENE_IDS,
    IS_VSR_TEST_ONLY,
    MODEL_ID,
    S1_SEMANTIC_LABEL,
    S2_SEMANTIC_LABEL,
    TRAIN_DATASET_PATHS,
    get_run_output_dir,
    recover_model_revision,
    seed_everything,
    validate_seed,
    validate_training_scenes,
    verify_frozen_e1_datasets,
)


class TestDirectionalParserHardening(unittest.TestCase):
    """Test strict directional parsing against adversarial and valid cases."""

    def test_directional_accepted_cases(self):
        """Valid unambiguous directional answers are correctly extracted."""
        self.assertEqual(parse_directional_answer("left"), "left")
        self.assertEqual(parse_directional_answer("Left."), "left")
        self.assertEqual(parse_directional_answer("The red sphere is left of the blue cube."), "left")
        self.assertEqual(parse_directional_answer("The object is in front of the other object."), "front")
        self.assertEqual(parse_directional_answer("It is farther from the camera."), "farther")
        self.assertEqual(parse_directional_answer("It is behind."), "behind")
        self.assertEqual(parse_directional_answer("above\n"), "above")
        self.assertEqual(parse_directional_answer("below"), "below")
        self.assertEqual(parse_directional_answer("nearer to the camera"), "nearer")
        self.assertEqual(parse_directional_answer("The cylinder is right of the cone."), "right")

    def test_directional_rejected_adversarial_cases(self):
        """Adversarial, ambiguous, negated, and multi-label inputs return None."""
        # Explicit negation
        self.assertIsNone(parse_directional_answer("not left"))
        self.assertIsNone(parse_directional_answer("is not left"))
        self.assertIsNone(parse_directional_answer("not in front"))
        self.assertIsNone(parse_directional_answer("not behind"))
        self.assertIsNone(parse_directional_answer("never left"))
        self.assertIsNone(parse_directional_answer("the left object is not right of it"))

        # Disjunction and conflict
        self.assertIsNone(parse_directional_answer("left or right"))
        self.assertIsNone(parse_directional_answer("either left or right"))
        self.assertIsNone(parse_directional_answer("front or behind"))
        self.assertIsNone(parse_directional_answer("both left and right"))
        self.assertIsNone(parse_directional_answer("The sphere is left, but could be right."))

        # Explicit uncertainty and hedging
        self.assertIsNone(parse_directional_answer("I think it might be right"))
        self.assertIsNone(parse_directional_answer("it might be left"))
        self.assertIsNone(parse_directional_answer("could be behind"))
        self.assertIsNone(parse_directional_answer("probably left"))
        self.assertIsNone(parse_directional_answer("probably right"))
        self.assertIsNone(parse_directional_answer("maybe behind"))
        self.assertIsNone(parse_directional_answer("I cannot tell whether it is left"))
        self.assertIsNone(parse_directional_answer("perhaps above"))
        self.assertIsNone(parse_directional_answer("unclear"))
        self.assertIsNone(parse_directional_answer("cannot determine whether it is left"))
        self.assertIsNone(parse_directional_answer("I am unsure"))
        self.assertIsNone(parse_directional_answer("no relation"))

        # Word boundary tests: substring collisions within longer words
        self.assertIsNone(parse_directional_answer("upright position"))  # contains 'right' inside upright
        self.assertIsNone(parse_directional_answer("bright red sphere"))  # contains 'right' inside bright
        self.assertIsNone(parse_directional_answer("confronting the camera"))  # contains 'front' inside confront

        # Empty / whitespace / non-string
        self.assertIsNone(parse_directional_answer(""))
        self.assertIsNone(parse_directional_answer("   "))
        self.assertIsNone(parse_directional_answer(None))  # type: ignore


class TestYesNoParserHardening(unittest.TestCase):
    """Test strict yes/no parsing for VSR evaluation."""

    def test_yesno_accepted_cases(self):
        """Unambiguous yes and no statements are correctly parsed."""
        self.assertEqual(parse_yesno_answer("yes"), "yes")
        self.assertEqual(parse_yesno_answer("Yes."), "yes")
        self.assertEqual(parse_yesno_answer("Yes, true."), "yes")
        self.assertEqual(parse_yesno_answer("Yes, this statement is true."), "yes")
        self.assertEqual(parse_yesno_answer("no"), "no")
        self.assertEqual(parse_yesno_answer("No."), "no")
        self.assertEqual(parse_yesno_answer("No, false."), "no")
        self.assertEqual(parse_yesno_answer("No, it is not."), "no")
        self.assertEqual(parse_yesno_answer("No, this statement is not true."), "no")

    def test_yesno_rejected_adversarial_cases(self):
        """Adversarial uncertainty, substring collisions, and disjunctions return None."""
        # Epistemic hedge and uncertainty patterns
        self.assertIsNone(parse_yesno_answer("probably yes"))
        self.assertIsNone(parse_yesno_answer("probably no"))
        self.assertIsNone(parse_yesno_answer("maybe yes"))
        self.assertIsNone(parse_yesno_answer("maybe no"))
        self.assertIsNone(parse_yesno_answer("no idea"))
        self.assertIsNone(parse_yesno_answer("not sure"))
        self.assertIsNone(parse_yesno_answer("cannot say"))
        self.assertIsNone(parse_yesno_answer("cannot determine"))
        self.assertIsNone(parse_yesno_answer("I am not sure"))
        self.assertIsNone(parse_yesno_answer("Cannot determine"))
        self.assertIsNone(parse_yesno_answer("maybe"))
        self.assertIsNone(parse_yesno_answer("perhaps"))
        self.assertIsNone(parse_yesno_answer("unclear"))

        # Substring / token boundary safety: standalone words that contain 'no'
        self.assertIsNone(parse_yesno_answer("unknown"))
        self.assertIsNone(parse_yesno_answer("nothing"))
        self.assertIsNone(parse_yesno_answer("notice"))
        self.assertIsNone(parse_yesno_answer("another"))
        self.assertIsNone(parse_yesno_answer("not"))
        self.assertIsNone(parse_yesno_answer("cannot"))

        # Disjunction and conflict
        self.assertIsNone(parse_yesno_answer("yes or no"))
        self.assertIsNone(parse_yesno_answer("yes and no"))
        self.assertIsNone(parse_yesno_answer("both yes and no"))
        self.assertIsNone(parse_yesno_answer("not yes"))
        self.assertIsNone(parse_yesno_answer("not no"))
        self.assertIsNone(parse_yesno_answer("Yes, false"))
        self.assertIsNone(parse_yesno_answer("No, true"))

        # Empty / whitespace
        self.assertIsNone(parse_yesno_answer(""))
        self.assertIsNone(parse_yesno_answer("   "))
        self.assertIsNone(parse_yesno_answer(None))  # type: ignore


class TestVSRTaxonomyFreeze(unittest.TestCase):
    """Test preservation of recovered historical VSR relation taxonomy."""

    def test_direct_vs_broader_disjointness(self):
        """Direct transfer dimensions and broader historical orientation are strictly disjoint."""
        self.assertEqual(VSR_DIRECT_DIMS, frozenset({"left_right", "front_back"}))
        self.assertEqual(VSR_BROADER_DIMS, frozenset({"orientation"}))
        self.assertTrue(VSR_DIRECT_DIMS.isdisjoint(VSR_BROADER_DIMS))

    def test_orientation_relations_do_not_contain_direct_relations(self):
        """Orientation relations strictly exclude direct horizontal and depth relations."""
        direct_relations = {
            "left of", "right of", "at the left side of", "at the right side of", "at the side of",
            "in front of", "behind", "ahead of", "at the back of",
        }
        for rel in direct_relations:
            dim = classify_vsr_relation(rel)
            self.assertIn(dim, VSR_DIRECT_DIMS)
            self.assertNotIn(dim, VSR_BROADER_DIMS)

        historical_orientation_relations = {
            "facing", "facing away from", "toward", "opposite to",
            "parallel to", "perpendicular to", "across from", "across", "along",
        }
        for rel in historical_orientation_relations:
            dim = classify_vsr_relation(rel)
            self.assertEqual(dim, "orientation")
            self.assertNotIn(dim, VSR_DIRECT_DIMS)

    def test_all_recovered_historical_dimensions_present(self):
        """All 8 historical VSR dimensions are preserved."""
        expected_dims = {
            "left_right", "front_back", "vertical", "near_far",
            "topology", "contact", "orientation", "part_whole",
        }
        self.assertEqual(set(VSR_ALL_DIMS), expected_dims)
        self.assertEqual(classify_vsr_relation("inside"), "topology")
        self.assertEqual(classify_vsr_relation("on"), "contact")
        self.assertEqual(classify_vsr_relation("part of"), "part_whole")
        self.assertEqual(classify_vsr_relation("nonexistent_relation"), "other")


class TestFormalExperimentProtocolFreeze(unittest.TestCase):
    """Test frozen scientific protocol, seeds, groups, and semantic labeling."""

    def test_frozen_constants(self):
        """Verify experiment identity, model ID, and groups."""
        self.assertEqual(EXPERIMENT_ID, "g2.0-e3")
        self.assertEqual(MODEL_ID, "Qwen/Qwen2.5-VL-3B-Instruct")
        self.assertEqual(GROUPS, ("B", "A", "C", "D"))
        self.assertEqual(FORMAL_SEEDS, (42, 123, 456))
        self.assertEqual(ENGINEERING_SEED, 42)

    def test_seed_validation_and_execution(self):
        """Verify formal seeds are accepted and exploratory seeds require explicit flag."""
        for s in (42, 123, 456):
            validate_seed(s)

        with self.assertRaises(ValueError):
            validate_seed(999)

        validate_seed(999, allow_exploratory=True)

        # Seed CPU PRNGs deterministically
        seed_everything(42)
        r1 = random.random()
        t1 = torch.rand(3)

        seed_everything(42)
        r2 = random.random()
        t2 = torch.rand(3)

        self.assertEqual(r1, r2)
        self.assertTrue(torch.equal(t1, t2))

        seed_everything(123)
        self.assertNotEqual(random.random(), r1)

    def test_s1_s2_semantic_labeling_invariant(self):
        """S1 and S2 descriptions adhere strictly to frozen wording."""
        self.assertEqual(S1_SEMANTIC_LABEL, "unseen scenes + canonical horizontal views")
        self.assertEqual(
            S2_SEMANTIC_LABEL,
            "unseen scenes + newly instantiated bounded jitter poses generated under the same jitter process",
        )
        self.assertNotIn("continuous ood", S2_SEMANTIC_LABEL.lower())
        self.assertTrue(IS_VSR_TEST_ONLY)

    def test_dataset_mappings_and_split_isolation(self):
        """Baseline B has no dataset; A/C/D map to frozen E1 JSONLs without holdout leakage."""
        self.assertIsNone(TRAIN_DATASET_PATHS["B"])
        self.assertTrue(TRAIN_DATASET_PATHS["A"].endswith("train_group_a.jsonl"))
        self.assertTrue(TRAIN_DATASET_PATHS["C"].endswith("train_group_c.jsonl"))
        self.assertTrue(TRAIN_DATASET_PATHS["D"].endswith("train_group_d.jsonl"))

        # Verify training files exist and exclude scenes 080-099
        for group in ("A", "C", "D"):
            p = Path(TRAIN_DATASET_PATHS[group])
            self.assertTrue(p.exists(), f"File missing: {p}")
            scenes_in_file = set()
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    scenes_in_file.add(d["scene_id"])

            # Assert complete exclusion of holdout scenes 080-099
            self.assertTrue(
                scenes_in_file.isdisjoint(HOLDOUT_SCENE_IDS),
                f"Holdout scenes leaked into {group}: {scenes_in_file.intersection(HOLDOUT_SCENE_IDS)}",
            )

    def test_training_leakage_guard_catches_holdout_scene(self):
        """validate_training_scenes raises ValueError when holdout scene is provided."""
        valid_records = [{"scene_id": "scene_001"}, {"scene_id": "scene_042"}]
        validate_training_scenes(valid_records)  # passes

        leaked_records = [{"scene_id": "scene_001"}, {"scene_id": "scene_085"}]
        with self.assertRaises(ValueError) as ctx:
            validate_training_scenes(leaked_records)
        self.assertIn("scene_085", str(ctx.exception))

    def test_real_frozen_datasets_pass_tracked_hashes(self):
        """A. Real frozen E1 datasets pass verification against tracked constants."""
        verified = verify_frozen_e1_datasets(E1_MANIFEST_PATH)
        self.assertEqual(len(verified), 5)
        for grp, fname in FROZEN_E1_GROUP_TO_FILE.items():
            self.assertIn(grp, verified)
            self.assertEqual(verified[grp], FROZEN_E1_DATASET_HASHES[fname])

    def test_changed_dataset_unchanged_manifest_fails(self):
        """B. Changed dataset with unchanged manifest fails hard against tracked constant."""
        with TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            mutated_file = td / "train_group_a.jsonl"
            mutated_file.write_text(
                '{"id":"mutated","source_sample_id":"0","presentation_order":0,'
                '"view_id":"south","family":"horizontal","question":"q","answer":"left","image_path":"img.png"}\n',
                encoding="utf-8",
            )
            fake_manifest = {
                "groups": {
                    "group_a": {"file": str(mutated_file), "sha256": FROZEN_E1_DATASET_HASHES["train_group_a.jsonl"]},
                    "group_c": {"file": "outputs/experiments/g2.0-e/train_group_c.jsonl", "sha256": FROZEN_E1_DATASET_HASHES["train_group_c.jsonl"]},
                    "group_d": {"file": "outputs/experiments/g2.0-e/train_group_d.jsonl", "sha256": FROZEN_E1_DATASET_HASHES["train_group_d.jsonl"]},
                    "holdout_s1": {"file": "outputs/experiments/g2.0-e/holdout_s1_cardinal.jsonl", "sha256": FROZEN_E1_DATASET_HASHES["holdout_s1_cardinal.jsonl"]},
                    "holdout_s2": {"file": "outputs/experiments/g2.0-e/holdout_s2_jitter.jsonl", "sha256": FROZEN_E1_DATASET_HASHES["holdout_s2_jitter.jsonl"]},
                }
            }
            fake_manifest_path = td / "manifest.json"
            fake_manifest_path.write_text(json.dumps(fake_manifest), encoding="utf-8")

            with self.assertRaises(RuntimeError) as ctx:
                verify_frozen_e1_datasets(fake_manifest_path)
            self.assertIn("hash mismatch", str(ctx.exception).lower())

    def test_simultaneous_tamper_changed_dataset_and_manifest_still_fails(self):
        """C. Changed dataset + matching malicious manifest STILL fails hard against tracked constants."""
        import hashlib
        with TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            mutated_file = td / "train_group_a.jsonl"
            mutated_file.write_text(
                '{"id":"malicious","source_sample_id":"0","presentation_order":0,'
                '"view_id":"south","family":"horizontal","question":"q","answer":"left","image_path":"img.png"}\n',
                encoding="utf-8",
            )
            hasher = hashlib.sha256()
            hasher.update(b"malicious|0|0|south|horizontal|q|left|img.png\n")
            malicious_hash = hasher.hexdigest()

            fake_manifest = {
                "groups": {
                    "group_a": {"file": str(mutated_file), "sha256": malicious_hash},
                    "group_c": {"file": "outputs/experiments/g2.0-e/train_group_c.jsonl", "sha256": FROZEN_E1_DATASET_HASHES["train_group_c.jsonl"]},
                    "group_d": {"file": "outputs/experiments/g2.0-e/train_group_d.jsonl", "sha256": FROZEN_E1_DATASET_HASHES["train_group_d.jsonl"]},
                    "holdout_s1": {"file": "outputs/experiments/g2.0-e/holdout_s1_cardinal.jsonl", "sha256": FROZEN_E1_DATASET_HASHES["holdout_s1_cardinal.jsonl"]},
                    "holdout_s2": {"file": "outputs/experiments/g2.0-e/holdout_s2_jitter.jsonl", "sha256": FROZEN_E1_DATASET_HASHES["holdout_s2_jitter.jsonl"]},
                }
            }
            fake_manifest_path = td / "manifest.json"
            fake_manifest_path.write_text(json.dumps(fake_manifest), encoding="utf-8")

            with self.assertRaises(RuntimeError) as ctx:
                verify_frozen_e1_datasets(fake_manifest_path)
            self.assertIn("hash mismatch", str(ctx.exception).lower())

    def test_wrong_tracked_hash_fails_hard(self):
        """D. Discrepancy between actual dataset content and tracked constant fails hard."""
        from unittest.mock import patch
        tampered_tracked = dict(FROZEN_E1_DATASET_HASHES)
        tampered_tracked["train_group_a.jsonl"] = "0" * 64
        with patch.dict("spatialforge.experiment.protocol.FROZEN_E1_DATASET_HASHES", tampered_tracked):
            with self.assertRaises(RuntimeError) as ctx:
                verify_frozen_e1_datasets()
            self.assertIn("hash mismatch", str(ctx.exception).lower())

    def test_all_five_tracked_hashes_equal_accepted_values(self):
        """E. All five tracked hashes exactly equal the authoritative accepted values."""
        accepted_hashes = {
            "train_group_a.jsonl": "27c41955d52830d8f4d0a2d58ef33ca03290c16d683c47ee6fec48f6524655e0",
            "train_group_c.jsonl": "f7a8a5f31d194f1ea2a585753c5013bcf9d61f2bc8c50d366a4852fdb443449b",
            "train_group_d.jsonl": "263cd2634798f1debd74fa0c2f27f662686221e14642e63b7e73e66837698456",
            "holdout_s1_cardinal.jsonl": "1561e5fd40f2256d7d0e57b2a9a11bfc5ebbdd11585f60608423ab9f611394a4",
            "holdout_s2_jitter.jsonl": "bc8d42aabe1e9d5520f0d3c1c5575209b8e11488f78892da58b0b87066fdd132",
        }
        self.assertEqual(FROZEN_E1_DATASET_HASHES, accepted_hashes)
        self.assertEqual(len(FROZEN_E1_DATASET_HASHES), 5)

    def test_run_output_dir_collision_free(self):
        """Artifact directory paths are distinct and follow the frozen layout."""
        b_dir = get_run_output_dir("baseline_b")
        a42_dir = get_run_output_dir("A", 42)
        c42_dir = get_run_output_dir("C", 42)
        d42_dir = get_run_output_dir("D", 42)
        a123_dir = get_run_output_dir("A", 123)

        self.assertEqual(str(b_dir), "outputs/experiments/g2.0-e3/baseline_b")
        self.assertEqual(str(a42_dir), "outputs/experiments/g2.0-e3/group_a/seed_42")
        self.assertEqual(str(c42_dir), "outputs/experiments/g2.0-e3/group_c/seed_42")
        self.assertEqual(str(d42_dir), "outputs/experiments/g2.0-e3/group_d/seed_42")
        self.assertEqual(str(a123_dir), "outputs/experiments/g2.0-e3/group_a/seed_123")

        dirs = {b_dir, a42_dir, c42_dir, d42_dir, a123_dir}
        self.assertEqual(len(dirs), 5)  # All 5 paths are collision-free

        with self.assertRaises(ValueError):
            get_run_output_dir("A")  # Missing required seed


class TestEvaluationRecordsAndAggregation(unittest.TestCase):
    """Test deterministic prediction record serialization and metric aggregation."""

    def test_evaluation_prediction_record_serialization(self):
        """Prediction records serialize and deserialize deterministically."""
        rec = EvaluationPredictionRecord(
            sample_id="test_sample_001",
            source="synthetic_s1",
            scene_id="scene_080",
            group="A",
            seed=42,
            family="horizontal",
            ground_truth="left",
            raw_prediction="Left.",
            parsed_prediction="left",
            is_valid_prediction=True,
            is_correct=True,
            model_id=MODEL_ID,
            adapter_path="outputs/experiments/g2.0-e3/group_a/seed_42/adapter",
            view_id="south",
        )

        d = rec.to_dict()
        self.assertEqual(d["sample_id"], "test_sample_001")
        self.assertTrue(d["is_correct"])

        json_str_1 = rec.to_json()
        json_str_2 = rec.to_json()
        self.assertEqual(json_str_1, json_str_2)

        reloaded = EvaluationPredictionRecord.from_dict(json.loads(json_str_1))
        self.assertEqual(rec, reloaded)

    def test_metric_aggregation_and_invalid_accounting(self):
        """Metrics aggregate accurately and count invalid predictions as incorrect for primary accuracy."""
        recs = [
            # horizontal: 2 valid correct, 1 valid incorrect, 1 invalid
            EvaluationPredictionRecord("s1", "synth", "sc", "A", 42, "horizontal", "left", "left", "left", True, True, MODEL_ID),
            EvaluationPredictionRecord("s2", "synth", "sc", "A", 42, "horizontal", "left", "left", "left", True, True, MODEL_ID),
            EvaluationPredictionRecord("s3", "synth", "sc", "A", 42, "horizontal", "left", "right", "right", True, False, MODEL_ID),
            EvaluationPredictionRecord("s4", "synth", "sc", "A", 42, "horizontal", "left", "not left", None, False, False, MODEL_ID),
            # vertical: 1 valid correct, 1 invalid
            EvaluationPredictionRecord("s5", "synth", "sc", "A", 42, "vertical", "above", "above", "above", True, True, MODEL_ID),
            EvaluationPredictionRecord("s6", "synth", "sc", "A", 42, "vertical", "above", "maybe above", None, False, False, MODEL_ID),
        ]

        metrics = aggregate_metrics(recs)

        # Overall: 6 total, 4 valid, 2 invalid, 3 correct
        overall = metrics["overall"]
        self.assertEqual(overall["total"], 6)
        self.assertEqual(overall["valid_predictions"], 4)
        self.assertEqual(overall["invalid_predictions"], 2)
        self.assertAlmostEqual(overall["invalid_rate"], 2 / 6, places=5)
        self.assertEqual(overall["correct"], 3)
        # Primary accuracy = correct / total = 3 / 6 = 0.5
        self.assertAlmostEqual(overall["accuracy"], 0.5, places=5)
        # Diagnostic valid_accuracy = correct / valid = 3 / 4 = 0.75
        self.assertAlmostEqual(overall["valid_accuracy"], 0.75, places=5)

        # Dimension breakdown
        by_dim = metrics["by_dimension"]
        self.assertIn("horizontal", by_dim)
        self.assertIn("vertical", by_dim)
        self.assertEqual(by_dim["horizontal"]["total"], 4)
        self.assertEqual(by_dim["horizontal"]["correct"], 2)
        self.assertAlmostEqual(by_dim["horizontal"]["accuracy"], 0.5)
        self.assertEqual(by_dim["horizontal"]["invalid_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
