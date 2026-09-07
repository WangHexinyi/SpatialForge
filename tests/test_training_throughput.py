"""CPU-safe unit tests for training throughput optimization, pipeline caching, and telemetry.

Guaranteed NOT to download or load the 3B model.
Verifies:
1. Cache key stability
2. Cache invalidation on provenance change
3. Order preservation in prefetcher
4. Deterministic prefetcher delivery
5. Progress percentage calculation
6. ETA calculation and edge cases
7. Atomic progress.json write
8. Telemetry does not alter RNG states
9. Immutable FormalTrainingConfig
10. Optimized sample sequence equality
"""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from spatialforge.experiment.performance import (
    LiveTelemetryReporter,
    NVMLClient,
    StageTimer,
    write_progress_file_atomic,
)
from spatialforge.experiment.pipeline import (
    DecodedImageCache,
    FrozenVisionFeatureCache,
    PinnedPrefetchDataLoader,
    PreparedDatasetCache,
    compute_batched_per_example_loss,
    compute_dataset_content_hash,
    compute_image_content_hash,
    compute_pipeline_cache_key,
    compute_vision_cache_key,
)
from spatialforge.experiment.training import (
    CANDIDATE_P5_VISION_CACHE_CONFIG,
    FormalTrainingConfig,
    PERFORMANCE_FORMAL_CONFIG,
    PERFORMANCE_FORMAL_P1_CONFIG,
    PROFILE_CANDIDATE_P5_VISION_CACHE,
    PROFILE_PERFORMANCE_FORMAL,
    PROFILE_PERFORMANCE_FORMAL_P1,
    PROFILE_REFERENCE_REPRO,
    REFERENCE_REPRO_CONFIG,
    get_execution_profile,
)


class TestTrainingThroughputAndTelemetry(unittest.TestCase):
    """CPU-safe tests for training performance, pipeline, and telemetry."""

    def test_cache_key_stability(self):
        """Cache key must be deterministic for identical inputs."""
        key1 = compute_pipeline_cache_key(
            "train_group_a.jsonl",
            model_path="/dummy/path",
            dataset_hash="hash123",
        )
        key2 = compute_pipeline_cache_key(
            "train_group_a.jsonl",
            model_path="/dummy/path",
            dataset_hash="hash123",
        )
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 64)

    def test_cache_invalidation(self):
        """Changing provenance (dataset hash, model path, schema) must invalidate key."""
        base_key = compute_pipeline_cache_key("ds.jsonl", "/dummy/1", "hash_a", "v1")
        diff_hash = compute_pipeline_cache_key("ds.jsonl", "/dummy/1", "hash_b", "v1")
        diff_schema = compute_pipeline_cache_key("ds.jsonl", "/dummy/1", "hash_a", "v2")

        self.assertNotEqual(base_key, diff_hash)
        self.assertNotEqual(base_key, diff_schema)

    def test_order_preservation(self):
        """Prefetcher must deliver items in strictly identical sequence (0, 1, ..., N-1)."""
        num_items = 25
        samples = []
        for i in range(num_items):
            samples.append({
                "input_ids": torch.tensor([i, i + 1], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], dtype=torch.long),
                "labels": torch.tensor([-100, i], dtype=torch.long),
                "pixel_values": torch.zeros((4, 4), dtype=torch.float32),
                "image_grid_thw": torch.tensor([1, 2, 2], dtype=torch.long),
            })

        loader = PinnedPrefetchDataLoader(samples, device="cpu", queue_size=4, pin_memory=False)
        received_ids = []
        for batch in loader:
            received_ids.append(batch["input_ids"][0, 0].item())

        self.assertEqual(received_ids, list(range(num_items)))

    def test_deterministic_prefetch(self):
        """Repeated iterations must deliver identical tensor contents."""
        samples = [{
            "input_ids": torch.tensor([42, 100], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "labels": torch.tensor([-100, 100], dtype=torch.long),
            "pixel_values": torch.randn(2, 2),
            "image_grid_thw": torch.tensor([1, 1, 1], dtype=torch.long),
        }]

        loader1 = PinnedPrefetchDataLoader(samples, device="cpu", queue_size=2, pin_memory=False)
        batch1 = next(iter(loader1))

        loader2 = PinnedPrefetchDataLoader(samples, device="cpu", queue_size=2, pin_memory=False)
        batch2 = next(iter(loader2))

        self.assertTrue(torch.equal(batch1["input_ids"], batch2["input_ids"]))
        self.assertTrue(torch.equal(batch1["pixel_values"], batch2["pixel_values"]))

    def test_progress_calculation(self):
        """Progress percentage and microbatch tracking must be mathematically exact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = LiveTelemetryReporter(
                run_dir=tmpdir,
                group="A",
                seed=42,
                total_microbatches=1552,
                total_optimizer_steps=194,
                heartbeat_interval=50,
                enabled=True,
            )

            metrics = reporter.compute_progress_metrics(
                microbatch=776,
                optimizer_step=97,
                scheduler_step=97,
                latest_raw_loss=0.1234,
                learning_rate=5e-5,
            )

            self.assertEqual(metrics["microbatch"], 776)
            self.assertEqual(metrics["total_microbatches"], 1552)
            self.assertEqual(metrics["progress_pct"], 50.0)
            self.assertEqual(metrics["optimizer_step"], 97)
            self.assertEqual(metrics["total_optimizer_steps"], 194)
            self.assertEqual(metrics["latest_raw_loss"], 0.1234)

    def test_eta_behavior(self):
        """ETA must handle boundary cases (start, middle, end) gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = LiveTelemetryReporter(
                run_dir=tmpdir,
                group="A",
                seed=42,
                total_microbatches=100,
                total_optimizer_steps=12,
                enabled=True,
            )

            # At start (0 microbatches)
            m_start = reporter.compute_progress_metrics(0, 0, 0, 1.0, 1e-4)
            self.assertEqual(m_start["eta_seconds"], 0.0)
            self.assertEqual(m_start["progress_pct"], 0.0)

            # At completion (100 microbatches)
            m_end = reporter.compute_progress_metrics(100, 12, 12, 0.1, 0.0)
            self.assertEqual(m_end["eta_seconds"], 0.0)
            self.assertEqual(m_end["progress_pct"], 100.0)

    def test_atomic_progress_json_write(self):
        """progress.json must be written atomically without corruption or leftovers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "progress.json"
            data = {"status": "running", "microbatch": 100, "samples_per_sec": 1.25}
            write_progress_file_atomic(dest, data)

            self.assertTrue(dest.exists())
            with dest.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, data)

            # Ensure no stray temp files
            all_files = list(Path(tmpdir).iterdir())
            self.assertEqual(len(all_files), 1)
            self.assertEqual(all_files[0].name, "progress.json")

    def test_telemetry_does_not_alter_rng(self):
        """Telemetry computation and emission must not touch Python, NumPy, or Torch RNG."""
        py_state_before = random.getstate()
        np_state_before = np.random.get_state()
        th_state_before = torch.get_rng_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = LiveTelemetryReporter(
                run_dir=tmpdir,
                group="A",
                seed=42,
                total_microbatches=1552,
                total_optimizer_steps=194,
                heartbeat_interval=1,
                enabled=True,
            )
            reporter.step(
                microbatch=1,
                optimizer_step=0,
                scheduler_step=0,
                latest_raw_loss=1.5,
                learning_rate=1e-4,
            )

        py_state_after = random.getstate()
        np_state_after = np.random.get_state()
        th_state_after = torch.get_rng_state()

        self.assertEqual(py_state_before, py_state_after)
        self.assertEqual(np_state_before[0], np_state_after[0])
        self.assertTrue(np.array_equal(np_state_before[1], np_state_after[1]))
        self.assertTrue(torch.equal(th_state_before, th_state_after))

    def test_immutable_formal_training_config(self):
        """FormalTrainingConfig must be frozen and reject attribute mutation."""
        cfg = FormalTrainingConfig()
        with self.assertRaises(FrozenInstanceError):
            cfg.learning_rate = 2e-4
        with self.assertRaises(FrozenInstanceError):
            cfg.gradient_accumulation_steps = 16

    def test_optimized_sample_sequence_equality(self):
        """Tensors collated via prefetcher must match baseline collation."""
        item = {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([-100, -100, 3], dtype=torch.long),
            "pixel_values": torch.ones((2, 2), dtype=torch.float32),
            "image_grid_thw": torch.tensor([1, 2, 2], dtype=torch.long),
        }

        # Collate using baseline helper
        from spatialforge.experiment.training import collate_single_sample_batch
        baseline_batch = collate_single_sample_batch(item, device="cpu")

        # Collate using prefetcher
        prefetcher = PinnedPrefetchDataLoader([item], device="cpu", queue_size=2, pin_memory=False)
        prefetch_batch = next(iter(prefetcher))

        for k in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"):
            self.assertTrue(
                torch.equal(baseline_batch[k], prefetch_batch[k]),
                f"Mismatch in key {k}",
            )

    def test_execution_profiles(self):
        """Execution profiles must enforce frozen semantics and correct GC/VC settings."""
        ref = get_execution_profile(PROFILE_REFERENCE_REPRO)
        p1 = get_execution_profile(PROFILE_PERFORMANCE_FORMAL_P1)
        perf = get_execution_profile(PROFILE_PERFORMANCE_FORMAL)
        p5 = get_execution_profile(PROFILE_CANDIDATE_P5_VISION_CACHE)

        self.assertIs(ref, REFERENCE_REPRO_CONFIG)
        self.assertIs(p1, PERFORMANCE_FORMAL_P1_CONFIG)
        self.assertIs(perf, PERFORMANCE_FORMAL_CONFIG)
        self.assertIs(p5, CANDIDATE_P5_VISION_CACHE_CONFIG)

        self.assertTrue(ref.gradient_checkpointing)
        self.assertFalse(p1.gradient_checkpointing)
        self.assertFalse(perf.gradient_checkpointing)
        self.assertFalse(p5.gradient_checkpointing)

        self.assertFalse(ref.use_vision_cache)
        self.assertFalse(p1.use_vision_cache)
        self.assertFalse(perf.use_vision_cache)
        self.assertTrue(p5.use_vision_cache)

        # Scientific semantics must be completely invariant
        for prof in (ref, p1, perf, p5):
            self.assertEqual(prof.per_device_train_batch_size, 1)
            self.assertEqual(prof.gradient_accumulation_steps, 8)
            self.assertEqual(prof.learning_rate, 1e-4)
            self.assertEqual(prof.num_train_epochs, 1)
            self.assertEqual(prof.bf16, True)
            self.assertEqual(prof.warmup_steps, 0)
            self.assertEqual(prof.scheduler, "linear")

        # Aliases
        self.assertIs(get_execution_profile("reference"), REFERENCE_REPRO_CONFIG)
        self.assertIs(get_execution_profile("repro"), REFERENCE_REPRO_CONFIG)
        self.assertIs(get_execution_profile("p1"), PERFORMANCE_FORMAL_P1_CONFIG)
        self.assertIs(get_execution_profile("formal"), PERFORMANCE_FORMAL_CONFIG)
        self.assertIs(get_execution_profile("p5"), CANDIDATE_P5_VISION_CACHE_CONFIG)
        self.assertIs(get_execution_profile("vision_cache"), CANDIDATE_P5_VISION_CACHE_CONFIG)
        self.assertIs(get_execution_profile("vc"), CANDIDATE_P5_VISION_CACHE_CONFIG)

        with self.assertRaises(ValueError):
            get_execution_profile("non_existent_profile")

    def test_batched_per_example_loss_equivalence(self):
        """Batched per-example loss must equal the mean of individual losses, even with unequal lengths."""
        import torch.nn.functional as F

        torch.manual_seed(42)
        batch_size = 3
        seq_len = 8
        vocab_size = 32

        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Unequal answer lengths:
        # Sample 0: answer length 2 (positions 4..5)
        # Sample 1: answer length 4 (positions 2..5)
        # Sample 2: answer length 3 (positions 3..5)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[0, 4:6] = torch.tensor([5, 12])
        labels[1, 2:6] = torch.tensor([3, 7, 19, 21])
        labels[2, 3:6] = torch.tensor([1, 8, 14])

        # Compute individual losses (MB1 semantics)
        individual_losses = []
        for i in range(batch_size):
            shift_l = logits[i, :-1, :].contiguous().float()
            shift_y = labels[i, 1:].contiguous()
            loss_i = F.cross_entropy(shift_l, shift_y, ignore_index=-100, reduction="mean")
            individual_losses.append(loss_i.item())

        expected_mean = sum(individual_losses) / len(individual_losses)

        # Compute batched per-example loss
        per_sample_losses, batched_mean = compute_batched_per_example_loss(logits, labels)

        for i in range(batch_size):
            self.assertAlmostEqual(per_sample_losses[i].item(), individual_losses[i], places=5)
        self.assertAlmostEqual(batched_mean.item(), expected_mean, places=5)

        # Compare with global HF mean: prove that global token mean differs due to length weighting
        shift_logits = logits[:, :-1, :].contiguous().float().view(-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        global_hf_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="mean").item()
        # They should differ because lengths are unequal (2 vs 4 vs 3)
        self.assertNotAlmostEqual(batched_mean.item(), global_hf_loss, places=2)

    def test_padding_invariance(self):
        """Right-padding with ignore_index tokens must have ZERO effect on per-example losses."""
        torch.manual_seed(42)
        logits = torch.randn(2, 6, 16)
        labels = torch.full((2, 6), -100, dtype=torch.long)
        labels[0, 2:4] = torch.tensor([2, 5])
        labels[1, 1:4] = torch.tensor([3, 8, 11])

        losses_orig, mean_orig = compute_batched_per_example_loss(logits, labels)

        # Pad sequences from length 6 to 10 with random logits and -100 labels
        extra_logits = torch.randn(2, 4, 16)
        extra_labels = torch.full((2, 4), -100, dtype=torch.long)
        padded_logits = torch.cat([logits, extra_logits], dim=1)
        padded_labels = torch.cat([labels, extra_labels], dim=1)

        losses_pad, mean_pad = compute_batched_per_example_loss(padded_logits, padded_labels)

        self.assertAlmostEqual(losses_orig[0].item(), losses_pad[0].item(), places=6)
        self.assertAlmostEqual(losses_orig[1].item(), losses_pad[1].item(), places=6)
        self.assertAlmostEqual(mean_orig.item(), mean_pad.item(), places=6)

    def test_telemetry_sample_counting(self):
        """Telemetry must track processed_samples and total_samples for ETA accuracy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = LiveTelemetryReporter(
                run_dir=tmpdir,
                group="A",
                seed=42,
                total_microbatches=194, # e.g. MB8 steps
                total_optimizer_steps=194,
                total_samples=1552,
                heartbeat_interval=1,
                enabled=True,
            )

            metrics = reporter.compute_progress_metrics(
                microbatch=50,
                optimizer_step=50,
                scheduler_step=50,
                latest_raw_loss=1.234,
                learning_rate=1e-4,
                processed_samples=400,
            )

            self.assertEqual(metrics["processed_samples"], 400)
            self.assertEqual(metrics["total_samples"], 1552)
            self.assertEqual(metrics["microbatch"], 50)
            self.assertEqual(metrics["total_microbatches"], 194)
            # 400 / 1552 = 25.77%
            self.assertAlmostEqual(metrics["progress_pct"], 25.77, places=1)

    def test_vision_cache_provenance_and_keys(self):
        """Vision cache key must be deterministic and sensitive to provenance changes."""
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            f.write(b"fake_image_bytes_12345")
            f.flush()

            h1 = compute_image_content_hash(f.name)
            self.assertEqual(len(h1), 64)

            k1 = compute_vision_cache_key(f.name, model_path="/dummy/model", dtype="bfloat16")
            k2 = compute_vision_cache_key(f.name, model_path="/dummy/model", dtype="bfloat16")
            self.assertEqual(k1, k2)

            k_diff_dtype = compute_vision_cache_key(f.name, model_path="/dummy/model", dtype="float32")
            self.assertNotEqual(k1, k_diff_dtype)

    def test_saved_performance_artifacts(self):
        """Saved benchmark, profiling, and floor artifacts must exist and have valid schema."""
        artifacts_dir = Path("outputs/experiments/g2.0-e3-performance")
        self.assertTrue(artifacts_dir.exists(), "g2.0-e3-performance dir missing")

        # 1. Nondeterminism floor
        floor_file = artifacts_dir / "nondeterminism_floor.json"
        self.assertTrue(floor_file.exists())
        with floor_file.open("r", encoding="utf-8") as f:
            floor_data = json.load(f)
        self.assertIn("losses", floor_data)
        self.assertIn("gradients", floor_data)
        self.assertIn("post_step_parameters", floor_data)
        self.assertEqual(floor_data["losses"]["max_abs_error"], 0.0)

        # 2. P1 benchmark
        p1_file = artifacts_dir / "p1_benchmark.json"
        self.assertTrue(p1_file.exists())
        with p1_file.open("r", encoding="utf-8") as f:
            p1_data = json.load(f)
        self.assertIn("reference", p1_data)
        self.assertIn("p1_candidate", p1_data)
        self.assertIn("speedup", p1_data)
        self.assertGreater(p1_data["speedup"], 1.8)

        # 3. P5 Vision Cache benchmark & equivalence
        p5_bench = artifacts_dir / "p5_vision_cache_benchmark.json"
        self.assertTrue(p5_bench.exists())
        with p5_bench.open("r", encoding="utf-8") as f:
            p5_b_data = json.load(f)
        self.assertGreaterEqual(p5_b_data["samples_per_sec"], 3.0)
        self.assertLessEqual(p5_b_data["projected_1552_min"], 8.7)
        self.assertGreater(p5_b_data["speedup_vs_ref"], 3.0)

        p5_eq = artifacts_dir / "p5_vision_cache_equivalence.json"
        self.assertTrue(p5_eq.exists())
        with p5_eq.open("r", encoding="utf-8") as f:
            p5_e_data = json.load(f)
        self.assertEqual(p5_e_data["candidate_diffs"]["losses"]["max_abs_error"], 0.0)
        self.assertLess(
            p5_e_data["candidate_diffs"]["gradients"]["max_abs_error"],
            floor_data["gradients"]["max_abs_error"],
        )


if __name__ == "__main__":
    unittest.main()
