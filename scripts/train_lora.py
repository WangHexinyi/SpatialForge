r"""G4：合成训练集 LoRA 微调（bf16，batch1 + 累积8）。

用法（仓库根目录下）：
    python scripts/train_lora.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                          Trainer, TrainingArguments)

from spatialforge.schema import load_samples

MODEL_DIR = Path("models/Qwen2.5-VL-3B-Instruct")
TRAIN = Path("data/processed/synth_train.jsonl")
ADAPTER_OUT = Path("outputs/lora_adapter")


class SynthDataset(Dataset):
    """将统一 schema 样本转为训练张量；labels 仅监督答案部分。"""

    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s.images[0]).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": s.question}]}]
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[img],
                                padding=False, return_tensors="pt")
        prompt_ids = inputs["input_ids"][0]
        ans_ids = self.processor.tokenizer(
            s.gt + self.processor.tokenizer.eos_token,
            add_special_tokens=False)["input_ids"]
        input_ids = torch.cat([prompt_ids, torch.tensor(ans_ids)])
        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100
        return {"input_ids": input_ids, "labels": labels,
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"]}


def collate(batch):
    """batch_size 恒为 1，无需填充。"""
    b = batch[0]
    return {"input_ids": b["input_ids"].unsqueeze(0),
            "attention_mask": torch.ones_like(b["input_ids"]).unsqueeze(0),
            "labels": b["labels"].unsqueeze(0),
            "pixel_values": b["pixel_values"],
            "image_grid_thw": b["image_grid_thw"]}


def main():
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda")
    model.enable_input_require_grads()

    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                     task_type=TaskType.CAUSAL_LM,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    ds = SynthDataset(load_samples(TRAIN), processor)
    args = TrainingArguments(
        output_dir="outputs/lora_run", num_train_epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-4, bf16=True, gradient_checkpointing=True,
        logging_steps=20, save_strategy="no", report_to=[],
        dataloader_num_workers=0, remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=collate).train()

    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_OUT)
    print("[ok] adapter saved ->", ADAPTER_OUT)


if __name__ == "__main__":
    main()