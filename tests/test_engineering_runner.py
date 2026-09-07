"""Unit tests for G2.0-E3.2 engineering experiment orchestration logic."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.run_engineering_experiment import (
    compute_config_hash,
    compute_engineering_summary,
    freeze_run_configs,
    run_preflight,
)
from spatialforge.experiment.evaluation import EvaluationPredictionRecord


class TestEngineeringExperimentRunner(unittest.TestCase):
    """Test CPU-side orchestration, config hashing, and summary calculations."""

    def test_compute_config_hash_determinism(self):
        """Config hashing is deterministic and key-order invariant."""
        cfg1 = {"a": 1, "b": "test", "c": [1, 2, 3]}
        cfg2 = {"c": [1, 2, 3], "a": 1, "b": "test"}
        cfg3 = {"a": 2, "b": "test", "c": [1, 2, 3]}

        h1 = compute_config_hash(cfg1)
        h2 = compute_config_hash(cfg2)
        h3 = compute_config_hash(cfg3)

        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertEqual(len(h1), 64)

    def test_preflight_and_config_freeze(self):
        """Preflight passes and freezes all 4 run configs with unique hashes."""
        preflight = run_preflight()
        self.assertEqual(len(preflight["verified_hashes"]), 5)
        self.assertEqual(preflight["model_revision"], "66285546d2b821cf421d4f5eb2576359d3770cd3")
        self.assertEqual(preflight["opt_steps"], 194)

        with TemporaryDirectory() as tmp_dir:
            configs = freeze_run_configs(preflight, base_dir=Path(tmp_dir))
            self.assertEqual(set(configs.keys()), {"baseline_b", "group_a_seed_42", "group_c_seed_42", "group_d_seed_42"})

            hashes = {c["config_hash"] for c in configs.values()}
            self.assertEqual(len(hashes), 4, "All 4 run config hashes must be distinct")

    def test_compute_engineering_summary_deltas(self):
        """Summary delta computation accurately calculates pairwise differences."""
        # Create mock results for B, A, C, D
        sample_ids = [f"sample_{i:04d}" for i in range(1136)]
        gts = ["left" if i % 2 == 0 else "right" for i in range(1136)]

        all_results = {}
        for grp, acc_s1, acc_s2 in [("B", 0.50, 0.40), ("A", 0.60, 0.50), ("C", 0.70, 0.65), ("D", 0.80, 0.75)]:
            recs_s1 = [
                EvaluationPredictionRecord(
                    sample_id=sid,
                    source="synthetic_s1",
                    scene_id="scene_080",
                    group=grp,
                    seed=None if grp == "B" else 42,
                    family="horizontal",
                    ground_truth=gt,
                    raw_prediction="left",
                    parsed_prediction="left",
                    is_valid_prediction=True,
                    is_correct=True,
                    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                )
                for sid, gt in zip(sample_ids, gts)
            ]
            recs_s2 = [
                EvaluationPredictionRecord(
                    sample_id=sid,
                    source="synthetic_s2",
                    scene_id="scene_080",
                    group=grp,
                    seed=None if grp == "B" else 42,
                    family="horizontal",
                    ground_truth=gt,
                    raw_prediction="left",
                    parsed_prediction="left",
                    is_valid_prediction=True,
                    is_correct=True,
                    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                )
                for sid, gt in zip(sample_ids, gts)
            ]

            metrics_s1 = {
                "overall": {"accuracy": acc_s1, "invalid_predictions": 0, "invalid_rate": 0.0, "correct": int(acc_s1 * 1136), "total": 1136},
                "by_dimension": {"horizontal": {"accuracy": acc_s1}},
            }
            metrics_s2 = {
                "overall": {"accuracy": acc_s2, "invalid_predictions": 0, "invalid_rate": 0.0, "correct": int(acc_s2 * 1136), "total": 1136},
                "by_dimension": {"horizontal": {"accuracy": acc_s2}},
            }

            all_results[grp] = {
                "metrics_s1": metrics_s1,
                "metrics_s2": metrics_s2,
                "recs_s1": recs_s1,
                "recs_s2": recs_s2,
            }

        with TemporaryDirectory() as tmp_dir:
            temp_summary = Path(tmp_dir) / "summary.json"
            summary = compute_engineering_summary(all_results, output_path=temp_summary)
            deltas = summary["deltas"]

            self.assertAlmostEqual(deltas["A_minus_B"]["s1"]["overall"], 0.10, places=5)
            self.assertAlmostEqual(deltas["C_minus_B"]["s1"]["overall"], 0.20, places=5)
            self.assertAlmostEqual(deltas["D_minus_B"]["s1"]["overall"], 0.30, places=5)
            self.assertAlmostEqual(deltas["C_minus_A"]["s1"]["overall"], 0.10, places=5)
            self.assertAlmostEqual(deltas["D_minus_A"]["s1"]["overall"], 0.20, places=5)
            self.assertAlmostEqual(deltas["D_minus_C"]["s1"]["overall"], 0.10, places=5)
            self.assertTrue(temp_summary.exists())


if __name__ == "__main__":
    unittest.main()
