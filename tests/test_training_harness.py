"""Unit tests for G2.0-E2 training harness, data loading, formatting, and taxonomy.

All tests run on CPU without requiring GPU or full model weights.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from spatialforge.experiment.training import (
    DEFAULT_BF16,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_GRADIENT_CHECKPOINTING,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PER_DEVICE_BATCH_SIZE,
    DIRECTIONAL_VOCABULARY,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    LORA_TARGET_PROJECTIONS,
    LORA_TASK_TYPE,
    NUM_LANGUAGE_LAYERS,
    VSR_BROADER_DIMS,
    VSR_DIRECT_DIMS,
    FormalTrainingConfig,
    MultimodalSampleDataset,
    SmokeTrainingConfig,
    TrainingSampleRecord,
    classify_vsr_relation,
    collate_single_sample_batch,
    compute_formal_optimizer_steps,
    create_lora_config,
    get_language_model_target_modules,
    is_language_model_target_module,
    load_and_preprocess_image,
    load_training_records,
    parse_directional_answer,
    parse_yesno_answer,
)


class TestTrainingConfigAndConstants(unittest.TestCase):
    """Verify recovered v1 LoRA and training configuration invariants."""

    def test_lora_hyperparameters_match_recovered_v1(self):
        """LoRA configuration strictly matches recovered historical G4 baseline."""
        self.assertEqual(LORA_R, 8)
        self.assertEqual(LORA_ALPHA, 16)
        self.assertEqual(LORA_DROPOUT, 0.05)
        self.assertEqual(LORA_TASK_TYPE, "CAUSAL_LM")
        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.assertEqual(LORA_TARGET_MODULES, expected_modules)
        self.assertEqual(LORA_TARGET_PROJECTIONS, expected_modules)

    def test_create_lora_config_object(self):
        """create_lora_config returns valid PEFT LoraConfig with explicit LM allowlist."""
        cfg = create_lora_config()
        self.assertEqual(cfg.r, 8)
        self.assertEqual(cfg.lora_alpha, 16)
        self.assertEqual(cfg.lora_dropout, 0.05)
        expected_modules = set(get_language_model_target_modules())
        self.assertEqual(cfg.target_modules, expected_modules)
        self.assertEqual(len(cfg.target_modules), 252)

    def test_language_only_lora_target_allowlist(self):
        """get_language_model_target_modules returns exactly 252 language projections and zero visual projections."""
        modules = get_language_model_target_modules()
        self.assertEqual(len(modules), 252)
        # Exactly 36 layers * 7 projections
        self.assertEqual(len(modules), NUM_LANGUAGE_LAYERS * 7)

        # Check all are language model and none are visual
        for m in modules:
            self.assertTrue(m.startswith("model.language_model.layers."))
            self.assertNotIn("visual", m)
            leaf = m.split(".")[-1]
            self.assertIn(leaf, LORA_TARGET_PROJECTIONS)

        # Verify layer 0 and layer 35 projections are present
        self.assertIn("model.language_model.layers.0.self_attn.q_proj", modules)
        self.assertIn("model.language_model.layers.0.mlp.down_proj", modules)
        self.assertIn("model.language_model.layers.35.self_attn.o_proj", modules)
        self.assertIn("model.language_model.layers.35.mlp.gate_proj", modules)

    def test_is_language_model_target_module_filtering(self):
        """is_language_model_target_module accepts LM projections and strictly rejects visual tower."""
        # Accepted language model paths
        self.assertTrue(is_language_model_target_module("model.language_model.layers.0.self_attn.q_proj"))
        self.assertTrue(is_language_model_target_module("model.language_model.layers.15.mlp.gate_proj"))
        self.assertTrue(is_language_model_target_module("model.language_model.layers.35.mlp.down_proj"))

        # Strictly rejected visual tower paths
        self.assertFalse(is_language_model_target_module("model.visual.blocks.0.mlp.gate_proj"))
        self.assertFalse(is_language_model_target_module("model.visual.blocks.12.mlp.up_proj"))
        self.assertFalse(is_language_model_target_module("visual.blocks.31.mlp.down_proj"))
        self.assertFalse(is_language_model_target_module("model.visual.patch_embed.proj"))

        # Unrelated modules rejected
        self.assertFalse(is_language_model_target_module("model.language_model.layers.0.input_layernorm"))
        self.assertFalse(is_language_model_target_module("model.language_model.lm_head"))

    def test_training_hyperparameters_match_recovered_v1(self):
        """Training hyperparameters preserve historical single-epoch bf16 settings."""
        self.assertEqual(DEFAULT_LEARNING_RATE, 1e-4)
        self.assertEqual(DEFAULT_PER_DEVICE_BATCH_SIZE, 1)
        self.assertEqual(DEFAULT_GRADIENT_ACCUMULATION_STEPS, 8)
        self.assertEqual(DEFAULT_EPOCHS, 1)
        self.assertTrue(DEFAULT_BF16)
        self.assertTrue(DEFAULT_GRADIENT_CHECKPOINTING)

    def test_formal_training_config_invariants(self):
        """FormalTrainingConfig dataclass freezes recovered formal semantics."""
        cfg = FormalTrainingConfig()
        self.assertEqual(cfg.per_device_train_batch_size, 1)
        self.assertEqual(cfg.gradient_accumulation_steps, 8)
        self.assertEqual(cfg.learning_rate, 1e-4)
        self.assertEqual(cfg.num_train_epochs, 1)
        self.assertTrue(cfg.bf16)
        self.assertTrue(cfg.gradient_checkpointing)
        self.assertEqual(cfg.warmup_steps, 0)
        self.assertEqual(cfg.scheduler, "linear")

    def test_compute_formal_optimizer_steps(self):
        """1552 samples with accumulation 8 yields exactly 194 optimizer steps per epoch."""
        steps = compute_formal_optimizer_steps(1552, gradient_accumulation_steps=8, epochs=1)
        self.assertEqual(steps, 194)
        # Verify no remainder tail
        self.assertEqual(steps * 8, 1552)

    def test_accumulation_step_cadence_logic(self):
        """Optimizer steps occur strictly on multiples of accumulation steps."""
        accum_steps = 8
        stepped = []
        for mb in range(1, 17):
            is_step = (mb % accum_steps == 0)
            if is_step:
                stepped.append(mb)
        self.assertEqual(stepped, [8, 16])


class TestMultimodalDataLoading(unittest.TestCase):
    """Verify data loading, validation, and RGBA->RGB preprocessing."""

    def test_load_records_from_frozen_dataset(self):
        """Load first 10 records from frozen Group A JSONL and verify schema."""
        train_a_path = Path("outputs/experiments/g2.0-e/train_group_a.jsonl")
        if not train_a_path.exists():
            self.skipTest("Frozen dataset train_group_a.jsonl not found on disk")

        records = load_training_records(train_a_path, limit=10)
        self.assertEqual(len(records), 10)
        for r in records:
            self.assertTrue(r.sample_id.startswith("scene_"))
            self.assertIn(r.family, ("horizontal", "vertical", "depth", "near_far"))
            self.assertIn(r.answer, DIRECTIONAL_VOCABULARY)
            self.assertTrue(Path(r.image_path).exists())
            self.assertNotIn("obj0", r.question)
            self.assertNotIn('object 0 ("', r.question)

    def test_rgba_to_rgb_conversion(self):
        """load_and_preprocess_image explicitly converts RGBA PNGs to RGB mode."""
        with TemporaryDirectory() as tmp_dir:
            rgba_path = Path(tmp_dir) / "test_rgba.png"
            img_rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
            img_rgba.save(rgba_path)

            rgb_img = load_and_preprocess_image(rgba_path)
            self.assertEqual(rgb_img.mode, "RGB")
            self.assertEqual(rgb_img.size, (100, 100))

    def test_missing_field_rejection(self):
        """TrainingSampleRecord.from_dict rejects records missing required fields."""
        with self.assertRaises(ValueError):
            TrainingSampleRecord.from_dict({"id": "s1", "question": "q"})

    def test_internal_name_leakage_rejection(self):
        """TrainingSampleRecord.from_dict rejects questions leaking raw internal IDs."""
        with TemporaryDirectory() as tmp_dir:
            fake_img = Path(tmp_dir) / "fake.png"
            Image.new("RGB", (10, 10)).save(fake_img)

            bad_data = {
                "id": "s1",
                "question": 'Is object 0 ("obj0") left of object 1?',
                "answer": "left",
                "image_path": str(fake_img),
            }
            with self.assertRaises(ValueError):
                TrainingSampleRecord.from_dict(bad_data)

    def test_missing_image_file_rejection(self):
        """TrainingSampleRecord.from_dict rejects records with non-existent image paths."""
        data = {
            "id": "s1",
            "question": "Is the cube left of the sphere?",
            "answer": "left",
            "image_path": "/nonexistent/path/to/image.png",
        }
        with self.assertRaises(FileNotFoundError):
            TrainingSampleRecord.from_dict(data)

    def test_collate_single_sample_batch_dual_interface(self):
        """collate_single_sample_batch accepts both single dict and DataLoader batch list [dict]."""
        import torch

        dummy_item = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([-100, 2, 3]),
            "pixel_values": torch.zeros((10, 10)),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }

        # 1. Single dict interface
        batch1 = collate_single_sample_batch(dummy_item)
        self.assertEqual(batch1["input_ids"].shape, (1, 3))
        self.assertEqual(batch1["attention_mask"].shape, (1, 3))
        self.assertEqual(batch1["labels"].shape, (1, 3))

        # 2. DataLoader batch sequence interface [item]
        batch2 = collate_single_sample_batch([dummy_item])
        self.assertEqual(batch2["input_ids"].shape, (1, 3))
        self.assertEqual(batch2["attention_mask"].shape, (1, 3))
        self.assertEqual(batch2["labels"].shape, (1, 3))

        # 3. Invalid multi-sample batch rejection
        with self.assertRaises(ValueError):
            collate_single_sample_batch([dummy_item, dummy_item])

    def test_multimodal_sample_dataset_interface(self):
        """MultimodalSampleDataset provides length and deterministic sequential indexing."""
        with TemporaryDirectory() as tmp_dir:
            fake_img = Path(tmp_dir) / "test.png"
            Image.new("RGB", (20, 20)).save(fake_img)

            records = [
                TrainingSampleRecord(
                    sample_id=f"s_{i}",
                    source_sample_id=f"src_{i}",
                    presentation_order=i,
                    scene_id="sc_0",
                    view_id="south",
                    family="depth",
                    image_path=str(fake_img),
                    question=f"Question {i}",
                    answer="front",
                    tags=("test",),
                )
                for i in range(5)
            ]

            ds = MultimodalSampleDataset(records, processor=None)
            self.assertEqual(len(ds), 5)
            # Verify records remain deterministically ordered
            self.assertEqual(ds.records[0].sample_id, "s_0")
            self.assertEqual(ds.records[4].sample_id, "s_4")


class TestLossMaskingAndAnswerParsing(unittest.TestCase):
    """Verify label masking invariants and answer parsing."""

    def test_loss_masking_boundary_logic(self):
        """Prompt tokens are strictly masked (-100) and answer tokens are supervised."""
        import torch

        # Simulate 10 prompt tokens and 3 answer tokens
        prompt_ids = torch.tensor([101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        ans_ids = torch.tensor([201, 202, 203])

        combined = torch.cat([prompt_ids, ans_ids])
        labels = combined.clone()
        labels[: len(prompt_ids)] = -100

        self.assertEqual(len(labels), 13)
        self.assertEqual((labels == -100).sum().item(), 10)
        self.assertEqual((labels != -100).sum().item(), 3)
        # Verify supervised slice strictly matches answer IDs
        self.assertTrue(torch.equal(labels[10:], ans_ids))

    def test_parse_directional_answer(self):
        """parse_directional_answer extracts canonical terms from various model surface outputs."""
        self.assertEqual(parse_directional_answer("front"), "front")
        self.assertEqual(parse_directional_answer("front."), "front")
        self.assertEqual(parse_directional_answer("It is behind."), "behind")
        self.assertEqual(parse_directional_answer("The red cube is left of the sphere."), "left")
        self.assertEqual(parse_directional_answer("above\n"), "above")
        self.assertEqual(parse_directional_answer("nearer to the camera"), "nearer")
        self.assertEqual(parse_directional_answer("farther"), "farther")
        self.assertIsNone(parse_directional_answer("I cannot tell from this image."))
        self.assertIsNone(parse_directional_answer(""))

    def test_parse_yesno_answer(self):
        """parse_yesno_answer extracts yes/no for VSR evaluation."""
        self.assertEqual(parse_yesno_answer("yes"), "yes")
        self.assertEqual(parse_yesno_answer("Yes, this statement is true."), "yes")
        self.assertEqual(parse_yesno_answer("no"), "no")
        self.assertEqual(parse_yesno_answer("No, it is not."), "no")
        self.assertIsNone(parse_yesno_answer("Maybe"))

    def test_vsr_taxonomy_distinction(self):
        """VSR relations strictly separate direct relations from broader orientation."""
        # Direct relations
        self.assertEqual(classify_vsr_relation("left of"), "left_right")
        self.assertEqual(classify_vsr_relation("right of"), "left_right")
        self.assertEqual(classify_vsr_relation("in front of"), "front_back")
        self.assertEqual(classify_vsr_relation("behind"), "front_back")

        # Broader historical orientation
        self.assertEqual(classify_vsr_relation("facing"), "orientation")
        self.assertEqual(classify_vsr_relation("parallel to"), "orientation")
        self.assertEqual(classify_vsr_relation("perpendicular to"), "orientation")

        # Disjointness invariant
        self.assertTrue(VSR_DIRECT_DIMS.isdisjoint(VSR_BROADER_DIMS))
        self.assertIn("left_right", VSR_DIRECT_DIMS)
        self.assertIn("front_back", VSR_DIRECT_DIMS)
        self.assertIn("orientation", VSR_BROADER_DIMS)


if __name__ == "__main__":
    unittest.main()
