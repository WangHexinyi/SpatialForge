"""Unit tests for cached-vision post-representation batch collation and loss reduction.

Validates:
1. Collation right-padding of inputs_embeds, attention_mask, labels.
2. Bitwise preservation of unpadded tokens.
3. Zero gradient contamination on padded positions.
4. Mathematical equivalence of per-example loss reduction vs sequential accumulation.
5. Contrast between per-example loss and HF global token-mean loss under unequal answer lengths.
6. Telemetry schema memory separation (NVML vs PyTorch alloc/peak).
"""

import unittest
import torch
import torch.nn.functional as F

from spatialforge.experiment.pipeline import (
    collate_cached_vision_batch,
    compute_batched_per_example_loss,
)
from spatialforge.experiment.performance import LiveTelemetryReporter


class TestCachedVisionCollation(unittest.TestCase):
    """Test collation at the post-vision inputs_embeds boundary."""

    def test_single_sample_mb1(self):
        """MB1 collation should preserve shapes with leading batch dim 1."""
        item = {
            "inputs_embeds": torch.randn(10, 32),
            "attention_mask": torch.ones(10, dtype=torch.long),
            "labels": torch.full((10,), 42, dtype=torch.long),
        }
        batch = collate_cached_vision_batch([item])
        self.assertEqual(batch["inputs_embeds"].shape, (1, 10, 32))
        self.assertEqual(batch["attention_mask"].shape, (1, 10))
        self.assertEqual(batch["labels"].shape, (1, 10))
        self.assertTrue(torch.equal(batch["inputs_embeds"][0], item["inputs_embeds"]))

    def test_multi_sample_padding_and_preservation(self):
        """MB > 1 collation should right-pad to max length with zeros and -100."""
        item1 = {
            "inputs_embeds": torch.randn(12, 16),
            "attention_mask": torch.ones(12, dtype=torch.long),
            "labels": torch.full((12,), 5, dtype=torch.long),
        }
        item2 = {
            "inputs_embeds": torch.randn(8, 16),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "labels": torch.full((8,), 7, dtype=torch.long),
        }
        item3 = {
            "inputs_embeds": torch.randn(15, 16),
            "attention_mask": torch.ones(15, dtype=torch.long),
            "labels": torch.full((15,), 9, dtype=torch.long),
        }

        batch = collate_cached_vision_batch([item1, item2, item3])
        self.assertEqual(batch["inputs_embeds"].shape, (3, 15, 16))
        self.assertEqual(batch["attention_mask"].shape, (3, 15))
        self.assertEqual(batch["labels"].shape, (3, 15))

        # Check item 2 padding (length 8 -> 15, 7 pad tokens)
        self.assertTrue(torch.equal(batch["inputs_embeds"][1, :8], item2["inputs_embeds"]))
        self.assertTrue(torch.equal(batch["inputs_embeds"][1, 8:], torch.zeros(7, 16)))
        self.assertTrue(torch.equal(batch["attention_mask"][1, :8], torch.ones(8, dtype=torch.long)))
        self.assertTrue(torch.equal(batch["attention_mask"][1, 8:], torch.zeros(7, dtype=torch.long)))
        self.assertTrue(torch.equal(batch["labels"][1, :8], item2["labels"]))
        self.assertTrue(torch.equal(batch["labels"][1, 8:], torch.full((7,), -100, dtype=torch.long)))


class TestPerExampleLossReduction(unittest.TestCase):
    """Test mathematical equivalence of per-example loss reduction vs sequential accumulation."""

    def test_equivalence_with_sequential_mb1(self):
        """Per-example mean loss on batch must exactly match mean of sequential individual losses."""
        torch.manual_seed(42)
        vocab_size = 50

        # Create two synthetic samples with different active token lengths
        labels_1 = torch.full((10,), -100, dtype=torch.long)
        labels_1[7:] = torch.tensor([12, 25, 40])
        logits_1 = torch.randn(1, 10, vocab_size, requires_grad=True)

        labels_2 = torch.full((8,), -100, dtype=torch.long)
        labels_2[6:] = torch.tensor([18, 33])
        logits_2 = torch.randn(1, 8, vocab_size, requires_grad=True)

        # Compute sequential individual losses (MB1 accumulation)
        s_logits_1 = logits_1[:, :-1, :].contiguous()
        s_labels_1 = labels_1[1:].unsqueeze(0).contiguous()
        valid_1 = (s_labels_1 != -100)
        loss_seq_1 = F.cross_entropy(s_logits_1[valid_1], s_labels_1[valid_1], reduction="mean")

        s_logits_2 = logits_2[:, :-1, :].contiguous()
        s_labels_2 = labels_2[1:].unsqueeze(0).contiguous()
        valid_2 = (s_labels_2 != -100)
        loss_seq_2 = F.cross_entropy(s_logits_2[valid_2], s_labels_2[valid_2], reduction="mean")

        loss_sequential_accumulated = (loss_seq_1 + loss_seq_2) / 2.0

        # Now compute batched via compute_batched_per_example_loss
        padded_logits_2 = torch.cat([logits_2, torch.zeros(1, 2, vocab_size)], dim=1)
        batched_logits = torch.cat([logits_1, padded_logits_2], dim=0)

        padded_labels_2 = torch.cat([labels_2, torch.full((2,), -100, dtype=torch.long)], dim=0)
        batched_labels = torch.stack([labels_1, padded_labels_2], dim=0)

        per_sample_losses, batched_loss = compute_batched_per_example_loss(batched_logits, batched_labels)

        self.assertAlmostEqual(per_sample_losses[0].item(), loss_seq_1.item(), places=5)
        self.assertAlmostEqual(per_sample_losses[1].item(), loss_seq_2.item(), places=5)
        self.assertAlmostEqual(batched_loss.item(), loss_sequential_accumulated.item(), places=5)

    def test_divergence_from_hf_global_token_mean(self):
        """Demonstrate that HF global token mean diverges from sequential mean when answer lengths differ."""
        torch.manual_seed(42)
        vocab_size = 50

        labels = torch.full((2, 12), -100, dtype=torch.long)
        labels[0, 11] = 5  # 1 answer token
        labels[1, 3:12] = torch.randint(0, vocab_size, (9,))  # 9 answer tokens

        logits = torch.randn(2, 12, vocab_size)

        per_sample_losses, batched_loss = compute_batched_per_example_loss(logits, labels)
        expected_per_example = (per_sample_losses[0] + per_sample_losses[1]) / 2.0
        self.assertAlmostEqual(batched_loss.item(), expected_per_example.item(), places=5)

        shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        hf_global_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="mean")

        diff = abs(batched_loss.item() - hf_global_loss.item())
        self.assertGreater(diff, 0.05, "HF global reduction failed to demonstrate expected divergence from per-example mean")

    def test_zero_gradient_on_padding_tokens(self):
        """Padded tokens (mask=0, label=-100) must receive strictly zero gradient."""
        torch.manual_seed(42)
        vocab_size = 20
        logits = torch.randn(2, 6, vocab_size, requires_grad=True)
        labels = torch.full((2, 6), -100, dtype=torch.long)
        labels[0, 2:5] = torch.tensor([3, 7, 11])
        labels[1, 2:4] = torch.tensor([5, 9])

        _, loss = compute_batched_per_example_loss(logits, labels)
        loss.backward()

        grad = logits.grad
        self.assertTrue(torch.all(grad[1, 4] == 0.0))
        self.assertTrue(torch.all(grad[0, 0] == 0.0))
        self.assertTrue(torch.all(grad[1, 0] == 0.0))


class TestTelemetrySchemaMemorySeparation(unittest.TestCase):
    """Test telemetry schema clean separation of NVML vs PyTorch memory fields."""

    def test_telemetry_schema_fields(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = LiveTelemetryReporter(
                run_dir=Path(tmpdir),
                group="A",
                seed=42,
                total_microbatches=194,
                total_optimizer_steps=194,
                total_samples=1552,
                enabled=False,
            )

            metrics = reporter.compute_progress_metrics(
                microbatch=10,
                optimizer_step=10,
                scheduler_step=10,
                latest_raw_loss=1.234,
                learning_rate=1e-4,
                processed_samples=80,
            )

            self.assertIn("nvml_gpu_memory_used_mb", metrics)
            self.assertIn("nvml_gpu_memory_total_mb", metrics)
            self.assertIn("torch_memory_allocated_mb", metrics)
            self.assertIn("torch_memory_reserved_mb", metrics)
            self.assertIn("torch_peak_allocated_mb", metrics)
            self.assertIn("torch_peak_reserved_mb", metrics)

            self.assertEqual(metrics["processed_samples"], 80)
            self.assertEqual(metrics["total_samples"], 1552)
            self.assertEqual(metrics["progress_pct"], round((80 / 1552) * 100, 2))

            self.assertIn("gpu_memory_used_mb", metrics)
            self.assertIn("gpu_memory_peak_mb", metrics)


if __name__ == "__main__":
    unittest.main()
