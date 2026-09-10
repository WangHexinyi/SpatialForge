import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from spatialforge.models.vl_adapter import (
    ModelSpec,
    collate_training_batch,
    detect_family,
    resolve_spec,
)


class _Cfg:
    def __init__(self, model_type, architectures=None):
        self.model_type = model_type
        self.architectures = architectures or []


class FamilyDetectionTests(unittest.TestCase):
    def test_qwen2_5(self):
        self.assertEqual(detect_family(_Cfg("qwen2_5_vl")), "qwen2_5_vl")

    def test_qwen3(self):
        self.assertEqual(detect_family(_Cfg("qwen3_vl")), "qwen3_vl")

    def test_qwen3_moe_maps_to_qwen3(self):
        self.assertEqual(detect_family(_Cfg("qwen3_vl_moe")), "qwen3_vl")

    def test_unknown(self):
        self.assertEqual(detect_family(_Cfg("llava")), "unknown")


class ResolveSpecTests(unittest.TestCase):
    def test_resolve_from_config(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(json.dumps({
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
                "_commit_hash": "abc123",
                "text_config": {"max_position_embeddings": 262144},
            }))
            spec = resolve_spec(d, adapter="/tmp/adapter")
            self.assertEqual(spec.family, "qwen3_vl")
            self.assertEqual(spec.revision, "abc123")
            self.assertEqual(spec.adapter, "/tmp/adapter")
            self.assertEqual(spec.max_context, 262144)
            self.assertEqual(spec.to_dict()["model_id"], "Qwen/Qwen3-VL-8B-Instruct")

    def test_resolve_missing_config_is_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            spec = resolve_spec(d)
            self.assertEqual(spec.family, "unknown")
            self.assertIsInstance(spec, ModelSpec)


class CollateTests(unittest.TestCase):
    def _item(self, n, value):
        return {
            "input_ids": torch.arange(value, value + n),
            "attention_mask": torch.ones(n, dtype=torch.long),
            "labels": torch.full((n,), -100, dtype=torch.long),
            "pixel_values": torch.ones(2, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "mm_token_type_ids": torch.zeros(n, dtype=torch.long),
        }

    def test_right_padding_and_order(self):
        items = [self._item(3, 10), self._item(5, 20)]
        batch = collate_training_batch(items, pad_token_id=0)
        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertEqual(batch["input_ids"][0, :3].tolist(), [10, 11, 12])
        self.assertEqual(batch["input_ids"][0, 3:].tolist(), [0, 0])
        self.assertEqual(batch["input_ids"][1, :5].tolist(), [20, 21, 22, 23, 24])
        self.assertEqual(batch["attention_mask"][0].tolist(), [1, 1, 1, 0, 0])
        self.assertTrue((batch["labels"][0, 3:] == -100).all())
        self.assertEqual(batch["pixel_values"].shape[0], 4)
        self.assertEqual(batch["image_grid_thw"].shape[0], 2)
        self.assertEqual(batch["mm_token_type_ids"].shape, (2, 5))

    def test_single_item_keeps_shape(self):
        batch = collate_training_batch([self._item(4, 0)], pad_token_id=0)
        self.assertEqual(batch["input_ids"].shape, (1, 4))


class TrainabilityTargetTests(unittest.TestCase):
    def _fake_model(self):
        model = nn.Module()

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4)
                self.k_proj = nn.Linear(4, 4)
                self.v_proj = nn.Linear(4, 4)
                self.o_proj = nn.Linear(4, 4)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(4, 4)
                self.up_proj = nn.Linear(4, 4)
                self.down_proj = nn.Linear(4, 4)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attn()
                self.mlp = MLP()

        class LM(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Layer() for _ in range(3)])

        class Visual(nn.Module):
            def __init__(self):
                super().__init__()
                self.merger = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
                self.blocks = nn.ModuleList([Layer() for _ in range(6)])

        model.language_model = LM()
        model.visual = Visual()
        return model

    def test_profile_a_only_lm(self):
        from spatialforge.experiment.training import collect_trainability_targets

        targets = collect_trainability_targets(self._fake_model(), "A")
        self.assertTrue(targets)
        self.assertTrue(all("visual" not in t for t in targets))
        self.assertTrue(all("language_model" in t for t in targets))

    def test_profile_b_merger_and_upper_blocks(self):
        from spatialforge.experiment.training import collect_trainability_targets

        targets = collect_trainability_targets(self._fake_model(), "B")
        self.assertTrue(any("merger" in t for t in targets))
        visual_blocks = sorted({int(t.split(".blocks.")[1].split(".")[0])
                                for t in targets if ".blocks." in t})
        self.assertEqual(visual_blocks, [2, 3, 4, 5])

    def test_profile_c_all_blocks(self):
        from spatialforge.experiment.training import collect_trainability_targets

        targets = collect_trainability_targets(self._fake_model(), "C")
        visual_blocks = sorted({int(t.split(".blocks.")[1].split(".")[0])
                                for t in targets if ".blocks." in t})
        self.assertEqual(visual_blocks, [0, 1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
