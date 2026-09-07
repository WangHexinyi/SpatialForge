"""Controlled training harness and pipeline verification for G2.0-E2.

Implements the recovered historical Qwen2.5-VL-3B LoRA training pipeline,
loss masking, data loading with RGBA->RGB conversion, LoRA attachment,
training loop, adapter save/reload, inference, and evaluation interfaces.

Semantics & Invariants:
- Model: Qwen/Qwen2.5-VL-3B-Instruct (bf16)
- LoRA: r=8, alpha=16, dropout=0.05, 7 projection modules
- Loss policy: Prompt tokens masked (-100), answer + EOS supervised
- Single-sample batch (batch size 1, gradient accumulation 8)
- External evaluation: Distinguishes direct relation transfer (left_right, front_back)
  from broader historical orientation transfer.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

# Default model location on AutoDL persistent storage
DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Recovered historical LoRA configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TASK_TYPE = "CAUSAL_LM"
LORA_TARGET_PROJECTIONS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_TARGET_MODULES = LORA_TARGET_PROJECTIONS  # Preserved for backward compatibility
NUM_LANGUAGE_LAYERS = 36


def get_language_model_target_modules(num_layers: int = NUM_LANGUAGE_LAYERS) -> List[str]:
    """Return explicit fully-qualified module paths for language model projections only.

    Restricts LoRA adaptation to the 36 transformer layers of Qwen2.5-VL language model:
      - self_attn: q_proj, k_proj, v_proj, o_proj
      - mlp: gate_proj, up_proj, down_proj
    Total modules: 36 * 7 = 252 modules.
    Explicitly excludes all model.visual.* paths to keep the vision tower 100% frozen.
    """
    modules = []
    for layer in range(num_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            modules.append(f"model.language_model.layers.{layer}.self_attn.{proj}")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            modules.append(f"model.language_model.layers.{layer}.mlp.{proj}")
    return modules


def is_language_model_target_module(module_name: str) -> bool:
    """Check whether a module path belongs to the language model target projections.

    Must lie under language_model and end in one of the 7 allowed projection names.
    Must NOT contain 'visual'.
    """
    if "visual" in module_name:
        return False
    leaf = module_name.split(".")[-1]
    if leaf not in LORA_TARGET_PROJECTIONS:
        return False
    return "language_model" in module_name or module_name.startswith("model.layers.")


# Recovered historical training hyperparameters
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_PER_DEVICE_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_EPOCHS = 1
DEFAULT_BF16 = True
DEFAULT_GRADIENT_CHECKPOINTING = True
DEFAULT_WARMUP_STEPS = 0
DEFAULT_SCHEDULER = "linear"


@dataclass(frozen=True)
class FormalTrainingConfig:
    """Frozen configuration for formal G2.0-E controlled training."""

    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_BATCH_SIZE
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    learning_rate: float = DEFAULT_LEARNING_RATE
    num_train_epochs: int = DEFAULT_EPOCHS
    bf16: bool = DEFAULT_BF16
    gradient_checkpointing: bool = DEFAULT_GRADIENT_CHECKPOINTING
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    scheduler: str = DEFAULT_SCHEDULER


@dataclass(frozen=True)
class SmokeTrainingConfig:
    """Configuration for short pipeline smoke verification."""

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = DEFAULT_LEARNING_RATE
    num_train_epochs: int = 1
    bf16: bool = True
    gradient_checkpointing: bool = True
    warmup_steps: int = 0
    scheduler: str = "linear"
    num_smoke_microbatches: int = 8  # 1 full accumulation window


def compute_formal_optimizer_steps(
    num_samples: int,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    epochs: int = DEFAULT_EPOCHS,
) -> int:
    """Calculate exact optimizer steps for formal training (e.g. 1552 * 1 / 8 = 194)."""
    return (num_samples * epochs) // gradient_accumulation_steps

from spatialforge.experiment.evaluation import (
    DIRECTIONAL_VOCABULARY,
    VSR_ALL_DIMS,
    VSR_BROADER_DIMS,
    VSR_DIRECT_DIMS,
    VSR_RELATION_TO_DIM,
    classify_vsr_relation,
    parse_directional_answer,
    parse_yesno_answer,
)


@dataclass(frozen=True)
class TrainingSampleRecord:
    """Parsed, validated multimodal training record from G2.0-E1 JSONL."""

    sample_id: str
    source_sample_id: str
    presentation_order: int
    scene_id: str
    view_id: str
    family: str
    image_path: str
    question: str
    answer: str
    tags: Tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[Path] = None) -> "TrainingSampleRecord":
        """Validate and construct record from JSON dictionary."""
        required = ("id", "question", "answer", "image_path")
        for key in required:
            if key not in data or not data[key]:
                raise ValueError(f"Record missing required field: {key!r}")

        # Leakage guard: ensure raw internal names are not exposed in question
        q = data["question"]
        if 'object 0 ("' in q or 'object 1 ("' in q or '"obj0"' in q or '"obj1"' in q:
            raise ValueError(f"Raw internal object identifier leaked in question: {q!r}")

        img_path = data["image_path"]
        if base_dir is not None:
            full_path = (Path(base_dir) / img_path).resolve()
        else:
            full_path = Path(img_path).resolve()

        if not full_path.exists():
            raise FileNotFoundError(f"Image not found at path: {full_path}")

        return cls(
            sample_id=str(data["id"]),
            source_sample_id=str(data.get("source_sample_id", data["id"])),
            presentation_order=int(data.get("presentation_order", 0)),
            scene_id=str(data.get("scene_id", "")),
            view_id=str(data.get("view_id", "")),
            family=str(data.get("family", "")),
            image_path=str(full_path),
            question=str(data["question"]),
            answer=str(data["answer"]),
            tags=tuple(data.get("tags", [])),
        )


def load_training_records(
    jsonl_path: Union[str, Path],
    base_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[TrainingSampleRecord]:
    """Load and validate multimodal training records from JSONL."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                rec = TrainingSampleRecord.from_dict(data, base_dir=base_dir)
                records.append(rec)
            except Exception as err:
                raise ValueError(f"Failed to parse line {line_idx + 1} in {path}: {err}") from err

            if limit is not None and len(records) >= limit:
                break

    return records


def load_and_preprocess_image(image_path: Union[str, Path]) -> Image.Image:
    """Load an image and explicitly convert RGBA to RGB for Qwen2.5-VL."""
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {img_path}")
    img = Image.open(img_path)
    # Explicit conversion to RGB (handles RGBA, Palette, Grayscale, etc.)
    return img.convert("RGB")


def create_lora_config(
    r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
    target_modules: Optional[Sequence[str]] = None,
) -> Any:
    """Create PEFT LoraConfig adhering to historical v1 specifications with strictly scoped LM modules."""
    from peft import LoraConfig, TaskType

    if target_modules is None:
        target_modules = get_language_model_target_modules()

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(target_modules),
    )


def format_chat_prompt(processor: Any, image: Image.Image, question: str) -> str:
    """Apply Qwen2.5-VL chat template to format user question with image."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_training_tensors(
    processor: Any,
    image: Image.Image,
    question: str,
    answer: str,
) -> Dict[str, Any]:
    """Construct multimodal input tensors with exact historical loss masking.

    Prompt tokens: labels = -100
    Answer tokens + EOS: supervised (labels = token_ids)
    """
    import torch

    prompt_text = format_chat_prompt(processor, image, question)
    inputs = processor(text=[prompt_text], images=[image], padding=False, return_tensors="pt")

    prompt_ids = inputs["input_ids"][0]
    ans_text = answer + processor.tokenizer.eos_token
    ans_ids = processor.tokenizer(ans_text, add_special_tokens=False)["input_ids"]

    combined_input_ids = torch.cat([prompt_ids, torch.tensor(ans_ids, dtype=torch.long)])
    labels = combined_input_ids.clone()
    labels[: len(prompt_ids)] = -100

    attention_mask = torch.ones_like(combined_input_ids)

    return {
        "input_ids": combined_input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
        "prompt_length": len(prompt_ids),
        "answer_length": len(ans_ids),
    }


def collate_single_sample_batch(
    batch: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Collate a single sample into a batch of size 1, unsqueezing tensor dimensions.

    Accepts either a single sample dict or a batch sequence [item] as provided
    by standard PyTorch DataLoader(..., batch_size=1).
    """
    import torch

    if isinstance(batch, (list, tuple)):
        if len(batch) != 1:
            raise ValueError(f"collate_single_sample_batch requires batch size 1, got {len(batch)}")
        item = batch[0]
    else:
        item = batch

    collated = {
        "input_ids": item["input_ids"].unsqueeze(0),
        "attention_mask": item["attention_mask"].unsqueeze(0),
        "labels": item["labels"].unsqueeze(0),
        "pixel_values": item["pixel_values"],
        "image_grid_thw": item["image_grid_thw"],
    }

    if device is not None:
        collated = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in collated.items()
        }
        # pixel_values should be bfloat16 to match model
        if "pixel_values" in collated and collated["pixel_values"].dtype == torch.float32:
            collated["pixel_values"] = collated["pixel_values"].to(torch.bfloat16)

    return collated


class MultimodalSampleDataset:
    """Deterministic sequential dataset wrapper over TrainingSampleRecords."""

    def __init__(self, records: Sequence[TrainingSampleRecord], processor: Any):
        self.records = list(records)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img = load_and_preprocess_image(rec.image_path)
        return build_training_tensors(self.processor, img, rec.question, rec.answer)



def run_single_forward_step(
    model: Any,
    batch: Dict[str, Any],
) -> Tuple[float, Any]:
    """Execute a single forward step and return loss value and model outputs."""
    import torch

    with torch.set_grad_enabled(True):
        outputs = model(**batch)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered in forward pass: {loss.item()}")
        return float(loss.item()), outputs


def run_single_backward_step(
    loss_tensor: Any,
    optimizer: Any,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    """Execute a backward pass and optimizer step; verify gradients."""
    import torch

    loss_tensor.backward()

    # Check for NaN / Inf gradients
    grad_norms = []
    has_nonzero = False
    for param in optimizer.param_groups[0]["params"]:
        if param.grad is not None:
            norm = param.grad.norm().item()
            if not torch.isfinite(param.grad).all():
                raise ValueError("Non-finite gradient encountered during backward pass")
            grad_norms.append(norm)
            if norm > 0:
                has_nonzero = True

    if not has_nonzero:
        raise ValueError("No trainable parameter received a non-zero gradient")

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], max_grad_norm)

    optimizer.step()
    optimizer.zero_grad()

    return {
        "mean_grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
        "max_grad_norm": max(grad_norms) if grad_norms else 0.0,
    }


def run_inference_greedy(
    model: Any,
    processor: Any,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 16,
    device: str = "cuda",
) -> str:
    """Execute single-sample greedy inference and return raw generated text."""
    import torch

    prompt_text = format_chat_prompt(processor, image, question)
    inputs = processor(text=[prompt_text], images=[image], padding=False, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding
        )

    output_ids = generated_ids[0][prompt_len:]
    answer = processor.decode(output_ids, skip_special_tokens=True)
    return answer.strip()


@dataclass
class TrainingStepState:
    """State record for a single microbatch / optimizer step."""

    microbatch_count: int
    optimizer_step_count: int
    epoch: int
    raw_loss: float
    scaled_loss: float
    learning_rate: float
    is_optimizer_step: bool


def run_formal_training_loop(
    model: Any,
    processor: Any,
    records: Sequence[TrainingSampleRecord],
    output_dir: Union[str, Path],
    config: Optional[FormalTrainingConfig] = None,
    max_microbatches: Optional[int] = None,
    device: str = "cuda",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Reusable formal training loop with exact gradient accumulation semantics.

    Semantics:
    - Test-set leakage guard: validates records do not contain holdout scenes.
    - Seed policy: seeds PRNGs if seed is provided.
    - optimizer.zero_grad()
    - For each microbatch:
        raw_loss = forward()
        scaled_loss = raw_loss / gradient_accumulation_steps
        scaled_loss.backward()
    - Every gradient_accumulation_steps:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    """
    import torch
    from transformers import get_linear_schedule_with_warmup
    from spatialforge.experiment.protocol import seed_everything, validate_training_scenes

    validate_training_scenes(records)
    if seed is not None:
        seed_everything(seed)

    if config is None:
        config = FormalTrainingConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enable input require grads and gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.train()

    # Trainable parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    total_samples = len(records) if max_microbatches is None else min(len(records), max_microbatches)
    total_optimizer_steps = compute_formal_optimizer_steps(
        total_samples,
        config.gradient_accumulation_steps,
        config.num_train_epochs,
    )
    sched_steps = max(total_optimizer_steps, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=sched_steps,
    )

    optimizer.zero_grad()

    step_records: List[TrainingStepState] = []
    raw_losses: List[float] = []
    microbatch_count = 0
    optimizer_step_count = 0
    t0 = time.time()
    initial_vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    for epoch in range(config.num_train_epochs):
        for rec in records:
            if max_microbatches is not None and microbatch_count >= max_microbatches:
                break

            img = load_and_preprocess_image(rec.image_path)
            tensors = build_training_tensors(processor, img, rec.question, rec.answer)
            batch = collate_single_sample_batch(tensors, device=device)

            raw_loss_val, outputs = run_single_forward_step(model, batch)
            raw_losses.append(raw_loss_val)

            scaled_loss = outputs.loss / config.gradient_accumulation_steps
            scaled_loss.backward()

            microbatch_count += 1
            is_step = (microbatch_count % config.gradient_accumulation_steps == 0)

            current_lr = optimizer.param_groups[0]["lr"]

            if is_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step_count += 1

            step_records.append(
                TrainingStepState(
                    microbatch_count=microbatch_count,
                    optimizer_step_count=optimizer_step_count,
                    epoch=epoch,
                    raw_loss=raw_loss_val,
                    scaled_loss=float(scaled_loss.item()),
                    learning_rate=current_lr,
                    is_optimizer_step=is_step,
                )
            )

        if max_microbatches is not None and microbatch_count >= max_microbatches:
            break

    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    wall_time = time.time() - t0

    # Save adapter
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)

    return {
        "total_microbatches": microbatch_count,
        "total_optimizer_steps": optimizer_step_count,
        "initial_loss": raw_losses[0] if raw_losses else 0.0,
        "final_loss": raw_losses[-1] if raw_losses else 0.0,
        "raw_losses": raw_losses,
        "step_records": step_records,
        "initial_vram_gb": initial_vram,
        "peak_vram_gb": peak_vram,
        "wall_time_sec": wall_time,
        "mean_microbatch_time_sec": wall_time / max(microbatch_count, 1),
        "adapter_dir": str(adapter_dir),
    }


def run_tiny_smoke_training(
    model: Any,
    processor: Any,
    records: Sequence[TrainingSampleRecord],
    output_dir: Path,
    num_steps: int = 8,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run an 8-microbatch (1 accumulation window) training smoke test."""
    config = FormalTrainingConfig(
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        gradient_checkpointing=True,
    )
    result = run_formal_training_loop(
        model=model,
        processor=processor,
        records=records,
        output_dir=output_dir,
        config=config,
        max_microbatches=num_steps,
        device=device,
    )
    return {
        "num_steps": result["total_microbatches"],
        "total_optimizer_steps": result["total_optimizer_steps"],
        "initial_loss": result["initial_loss"],
        "final_loss": result["final_loss"],
        "all_losses": result["raw_losses"],
        "mean_step_time_sec": result["mean_microbatch_time_sec"],
        "initial_vram_gb": result["initial_vram_gb"],
        "peak_vram_gb": result["peak_vram_gb"],
        "adapter_dir": result["adapter_dir"],
        "step_records": result["step_records"],
    }
