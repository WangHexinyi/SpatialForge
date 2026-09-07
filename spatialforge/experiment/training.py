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
from dataclasses import asdict, dataclass
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


# =====================================================================
# Authoritative Training Profiles for G2.0-E3.3A
# =====================================================================

PROFILE_REFERENCE: str = "reference"
PROFILE_BALANCED: str = "balanced"
PROFILE_MAX_PERFORMANCE: str = "max_performance"

# SpatialForge development / engineering execution default
DEFAULT_TRAINING_PROFILE: str = PROFILE_MAX_PERFORMANCE


class ProfileCapabilityError(RuntimeError):
    """Raised when the requested training profile exceeds verified device capabilities.

    Strictly enforces the NO SILENT FALLBACK rule.
    """

    def __init__(
        self,
        requested_profile: str,
        required_memory_estimate: float,
        available_memory: float,
        recommended_profile: str,
        message: Optional[str] = None,
    ):
        self.requested_profile = requested_profile
        self.required_memory_estimate = required_memory_estimate
        self.available_memory = available_memory
        self.recommended_profile = recommended_profile
        if message is None:
            rec_suffix = (
                f"{recommended_profile.capitalize()} is compatible."
                if recommended_profile
                else "No known execution profile fits within available memory."
            )
            message = (
                f"{requested_profile} cannot safely start on the current device "
                f"(requires ~{required_memory_estimate:.0f} MB, available: {available_memory:.0f} MB). "
                f"{rec_suffix}"
            )
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": "ProfileCapabilityError",
            "requested_profile": self.requested_profile,
            "required_memory_estimate": self.required_memory_estimate,
            "available_memory": self.available_memory,
            "recommended_profile": self.recommended_profile,
            "message": str(self),
        }


@dataclass(frozen=True)
class TrainingProfile:
    """Authoritative execution profile for SpatialForge training.

    Guarantees:
    - microbatch_size * gradient_accumulation_steps == effective_batch_size
    - effective_batch_size == 8
    """

    profile_id: str
    display_name: str
    microbatch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    gradient_checkpointing: bool
    use_vision_cache: bool
    intended_use: str
    memory_estimate_mb: float
    description: str

    @property
    def vision_cache_enabled(self) -> bool:
        return self.use_vision_cache

    def __post_init__(self):
        if self.microbatch_size * self.gradient_accumulation_steps != self.effective_batch_size:
            raise ValueError(
                f"Profile {self.profile_id!r} invariant violated: "
                f"microbatch_size ({self.microbatch_size}) * gradient_accumulation_steps ({self.gradient_accumulation_steps}) "
                f"!= effective_batch_size ({self.effective_batch_size})"
            )
        if self.effective_batch_size != 8:
            raise ValueError(
                f"Profile {self.profile_id!r} invariant violated: effective_batch_size must be 8, got {self.effective_batch_size}"
            )


REFERENCE_PROFILE = TrainingProfile(
    profile_id=PROFILE_REFERENCE,
    display_name="Reference",
    microbatch_size=1,
    gradient_accumulation_steps=8,
    effective_batch_size=8,
    gradient_checkpointing=False,
    use_vision_cache=True,
    intended_use="strict reproduction / debugging / numerical regression",
    memory_estimate_mb=19900.0,
    description="P5 semantics: Single-sample microbatching with frozen vision caching, ensuring bitwise identical forward/backward accumulation.",
)

BALANCED_PROFILE = TrainingProfile(
    profile_id=PROFILE_BALANCED,
    display_name="Balanced",
    microbatch_size=2,
    gradient_accumulation_steps=4,
    effective_batch_size=8,
    gradient_checkpointing=False,
    use_vision_cache=True,
    intended_use="performance with substantial VRAM headroom",
    memory_estimate_mb=23550.0,
    description="P6 semantics: 2-sample post-vision microbatching with substantial (~9.2 GB) VRAM headroom and 1.86x speedup.",
)

MAX_PERFORMANCE_PROFILE = TrainingProfile(
    profile_id=PROFILE_MAX_PERFORMANCE,
    display_name="Max Performance",
    microbatch_size=4,
    gradient_accumulation_steps=2,
    effective_batch_size=8,
    gradient_checkpointing=False,
    use_vision_cache=True,
    intended_use="maximize iteration speed and productive GPU utilization on validated ~32 GB devices",
    memory_estimate_mb=30900.0,
    description="P7 semantics: 4-sample post-vision microbatching maximizing compute saturation (>8.3 sps, ~96% GPU util) on 32 GB devices.",
)

FIRST_CLASS_PROFILES: Dict[str, TrainingProfile] = {
    PROFILE_REFERENCE: REFERENCE_PROFILE,
    PROFILE_BALANCED: BALANCED_PROFILE,
    PROFILE_MAX_PERFORMANCE: MAX_PERFORMANCE_PROFILE,
}


def get_profile(profile_id: Optional[Union[str, TrainingProfile]] = None) -> TrainingProfile:
    """Retrieve authoritative TrainingProfile by ID or alias.

    If profile_id is None, returns the development default ('max_performance').
    """
    if isinstance(profile_id, TrainingProfile):
        return profile_id
    if profile_id is None:
        return MAX_PERFORMANCE_PROFILE

    key = str(profile_id).strip().lower()
    if key in ("max_performance", "max", "p7", "performance"):
        return MAX_PERFORMANCE_PROFILE
    if key in ("balanced", "p6", "perf_balanced"):
        return BALANCED_PROFILE
    if key in ("reference", "p5", "repro", "reference_repro", "candidate_p5_vision_cache"):
        return REFERENCE_PROFILE

    raise ValueError(
        f"Unknown training profile: {profile_id!r}. "
        f"Available profiles: {[PROFILE_REFERENCE, PROFILE_BALANCED, PROFILE_MAX_PERFORMANCE]}"
    )


def list_profiles() -> List[Dict[str, Any]]:
    """Return structured metadata for all first-class training profiles for callers and future God View UI."""
    return [
        asdict(REFERENCE_PROFILE),
        asdict(BALANCED_PROFILE),
        asdict(MAX_PERFORMANCE_PROFILE),
    ]


def validate_profile_capability(
    profile_or_id: Union[str, TrainingProfile],
    available_memory_mb: Optional[float] = None,
) -> None:
    """Validate that target device satisfies profile requirements without silent fallback.

    Args:
        profile_or_id: Profile instance or string ID to validate.
        available_memory_mb: Optional override for available memory in MB.
                             If None, dynamically queries NVML or PyTorch.
    Raises:
        ProfileCapabilityError: If available memory is less than required memory estimate.
    """
    profile = get_profile(profile_or_id) if isinstance(profile_or_id, str) else profile_or_id

    if available_memory_mb is None:
        import torch
        if not torch.cuda.is_available():
            # In CPU environments, capability validation passes unless available_memory_mb is explicitly provided
            return

        try:
            from spatialforge.experiment.performance import NVMLClient
            client = NVMLClient(0)
            stats = client.query()
            client.close()
            available_memory_mb = stats.get("nvml_gpu_memory_total_mb", 0.0)
            if available_memory_mb <= 0.0:
                available_memory_mb = float(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        except Exception:
            available_memory_mb = float(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))

    if available_memory_mb < profile.memory_estimate_mb:
        # Actionable recommendation: find highest profile that fits
        if available_memory_mb >= BALANCED_PROFILE.memory_estimate_mb:
            rec = PROFILE_BALANCED
        elif available_memory_mb >= REFERENCE_PROFILE.memory_estimate_mb:
            rec = PROFILE_REFERENCE
        else:
            rec = None

        raise ProfileCapabilityError(
            requested_profile=profile.profile_id,
            required_memory_estimate=profile.memory_estimate_mb,
            available_memory=available_memory_mb,
            recommended_profile=rec,
        )

    return True


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
    use_vision_cache: bool = False
    training_profile_id: str = "reference"

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_profile(
        cls,
        profile_or_id: Optional[Union[str, TrainingProfile]] = None,
        validate_capability: bool = True,
        available_memory_mb: Optional[float] = None,
        **overrides: Any,
    ) -> "FormalTrainingConfig":
        """Create FormalTrainingConfig from an authoritative TrainingProfile.

        Defaults to development default ('max_performance').
        """
        profile = get_profile(profile_or_id)
        if validate_capability:
            validate_profile_capability(profile, available_memory_mb=available_memory_mb)

        kwargs: Dict[str, Any] = {
            "training_profile_id": profile.profile_id,
            "per_device_train_batch_size": profile.microbatch_size,
            "gradient_accumulation_steps": profile.gradient_accumulation_steps,
            "gradient_checkpointing": profile.gradient_checkpointing,
            "use_vision_cache": profile.use_vision_cache,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "num_train_epochs": DEFAULT_EPOCHS,
            "bf16": DEFAULT_BF16,
            "warmup_steps": DEFAULT_WARMUP_STEPS,
            "scheduler": DEFAULT_SCHEDULER,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def to_manifest_dict(self) -> Dict[str, Any]:
        """Serialize exact execution profile and hyperparameters for provenance."""
        return {
            "training_profile_id": self.training_profile_id,
            "microbatch_size": self.per_device_train_batch_size,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "gradient_checkpointing": self.gradient_checkpointing,
            "vision_cache_enabled": self.use_vision_cache,
            "use_vision_cache": self.use_vision_cache,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "bf16": self.bf16,
            "warmup_steps": self.warmup_steps,
            "scheduler": self.scheduler,
        }


# =====================================================================
# Historical and Named Execution Profiles for G2.0-E3
# =====================================================================

PROFILE_REFERENCE_REPRO: str = "REFERENCE_REPRO"
PROFILE_PERFORMANCE_FORMAL_P1: str = "PERFORMANCE_FORMAL_P1"
PROFILE_PERFORMANCE_FORMAL: str = "PERFORMANCE_FORMAL"
PROFILE_CANDIDATE_P5_VISION_CACHE: str = "CANDIDATE_P5_VISION_CACHE"

REFERENCE_REPRO_CONFIG = FormalTrainingConfig(
    gradient_checkpointing=True,
    use_vision_cache=False,
    training_profile_id="reference_repro",
)

PERFORMANCE_FORMAL_P1_CONFIG = FormalTrainingConfig(
    gradient_checkpointing=False,
    use_vision_cache=False,
    training_profile_id="performance_formal_p1",
)

PERFORMANCE_FORMAL_CONFIG = PERFORMANCE_FORMAL_P1_CONFIG

CANDIDATE_P5_VISION_CACHE_CONFIG = FormalTrainingConfig(
    gradient_checkpointing=False,
    use_vision_cache=True,
    training_profile_id="reference",
)

EXECUTION_PROFILES: Dict[str, FormalTrainingConfig] = {
    PROFILE_REFERENCE_REPRO: REFERENCE_REPRO_CONFIG,
    PROFILE_PERFORMANCE_FORMAL_P1: PERFORMANCE_FORMAL_P1_CONFIG,
    PROFILE_PERFORMANCE_FORMAL: PERFORMANCE_FORMAL_CONFIG,
    PROFILE_CANDIDATE_P5_VISION_CACHE: CANDIDATE_P5_VISION_CACHE_CONFIG,
    PROFILE_BALANCED: FormalTrainingConfig.from_profile(BALANCED_PROFILE, validate_capability=False),
    PROFILE_MAX_PERFORMANCE: FormalTrainingConfig.from_profile(MAX_PERFORMANCE_PROFILE, validate_capability=False),
}


def get_execution_profile(profile_name: Optional[str] = None) -> FormalTrainingConfig:
    """Retrieve frozen execution profile by name or alias."""
    if profile_name is None:
        profile_name = DEFAULT_TRAINING_PROFILE
    key = str(profile_name).upper().strip()
    if key in EXECUTION_PROFILES:
        return EXECUTION_PROFILES[key]
    if key in ("REFERENCE", "REPRO", "REFERENCE_REPRO"):
        return REFERENCE_REPRO_CONFIG
    if key in ("PERFORMANCE", "FORMAL", "OPT", "P1", "PERFORMANCE_FORMAL_P1"):
        return PERFORMANCE_FORMAL_P1_CONFIG
    if key in ("P5", "VISION_CACHE", "VC", "CANDIDATE_VISION_CACHE"):
        return CANDIDATE_P5_VISION_CACHE_CONFIG
    if key in ("P6", "BALANCED"):
        return FormalTrainingConfig.from_profile(BALANCED_PROFILE, validate_capability=False)
    if key in ("P7", "MAX", "MAX_PERFORMANCE"):
        return FormalTrainingConfig.from_profile(MAX_PERFORMANCE_PROFILE, validate_capability=False)
    raise ValueError(
        f"Unknown execution profile: {profile_name!r}. "
        f"Must be one of {list(EXECUTION_PROFILES.keys())}"
    )


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
    num_samples: Optional[int] = None,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    epochs: int = DEFAULT_EPOCHS,
    microbatch_size: int = 1,
    num_records: Optional[int] = None,
    num_train_epochs: Optional[int] = None,
) -> int:
    """Calculate exact optimizer steps for formal training (e.g. 1552 * 1 / 8 = 194)."""
    n = num_samples if num_samples is not None else (num_records if num_records is not None else 0)
    ep = num_train_epochs if num_train_epochs is not None else epochs
    effective_batch = microbatch_size * gradient_accumulation_steps
    return (n * ep) // effective_batch

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
    """Execute a single forward step and return loss value and model outputs.

    Uses compute_batched_per_example_loss when a multi-sample cached vision batch
    is provided to guarantee exact per-example weighting (1/B sum L_i).
    """
    import torch
    from spatialforge.experiment.pipeline import compute_batched_per_example_loss

    with torch.set_grad_enabled(True):
        if "inputs_embeds" in batch and batch.get("labels") is not None and batch["inputs_embeds"].shape[0] > 1:
            outputs = model(
                inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch.get("attention_mask"),
            )
            _, loss = compute_batched_per_example_loss(outputs.logits, batch["labels"])
            outputs.loss = loss
        else:
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
    group: str = "A",
    use_optimized_pipeline: bool = True,
    enable_telemetry: bool = True,
    heartbeat_interval: int = 50,
    cache_dir: Optional[Union[str, Path]] = None,
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
    - Live progress telemetry and heartbeat emitted every heartbeat_interval microbatches.
    """
    import torch
    from transformers import get_linear_schedule_with_warmup
    from spatialforge.experiment.protocol import seed_everything, validate_training_scenes
    from spatialforge.experiment.performance import GpuTelemetrySampler, LiveTelemetryReporter
    from spatialforge.experiment.pipeline import (
        PreparedDatasetCache,
        PinnedPrefetchDataLoader,
        compute_pipeline_cache_key,
    )

    validate_training_scenes(records)
    if seed is not None:
        seed_everything(seed)

    if config is None:
        config = FormalTrainingConfig.from_profile(DEFAULT_TRAINING_PROFILE)

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

    microbatch_size = config.per_device_train_batch_size
    total_samples = len(records) if max_microbatches is None else min(len(records), max_microbatches * microbatch_size)
    total_microbatches = (total_samples + microbatch_size - 1) // microbatch_size
    total_optimizer_steps = compute_formal_optimizer_steps(
        total_samples,
        config.gradient_accumulation_steps,
        config.num_train_epochs,
        microbatch_size=microbatch_size,
    )
    sched_steps = max(total_optimizer_steps, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=sched_steps,
    )

    optimizer.zero_grad()

    # Start telemetry if enabled
    sampler = None
    if enable_telemetry and torch.cuda.is_available() and device.startswith("cuda"):
        sampler = GpuTelemetrySampler(device_index=0, sample_interval_sec=0.2).start()

    reporter = LiveTelemetryReporter(
        run_dir=output_dir,
        group=group,
        seed=seed,
        total_microbatches=total_microbatches,
        total_optimizer_steps=total_optimizer_steps,
        heartbeat_interval=heartbeat_interval,
        sampler=sampler,
        enabled=enable_telemetry,
        total_samples=total_samples,
    )

    step_records: List[TrainingStepState] = []
    raw_losses: List[float] = []
    microbatch_count = 0
    processed_samples = 0
    optimizer_step_count = 0
    t0 = time.time()
    initial_vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    target_records = records[:total_samples] if max_microbatches is not None else records

    if use_optimized_pipeline:
        if config.use_vision_cache:
            from spatialforge.experiment.pipeline import FrozenVisionFeatureCache, DecodedImageCache
            vc_cache = FrozenVisionFeatureCache(cache_dir=cache_dir)
            img_cache = DecodedImageCache()
            base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
            prepared_samples = []
            with torch.no_grad():
                for r in target_records:
                    c_feat = vc_cache.get_or_compute(
                        r.image_path, model, processor, device=device, dtype=torch.bfloat16
                    ).to(device)
                    img = img_cache.get(r.image_path)
                    t = build_training_tensors(processor, img, r.question, r.answer)
                    in_emb = base_model.model.get_input_embeddings()(t["input_ids"].unsqueeze(0).to(device))
                    mask, _ = base_model.model.get_placeholder_mask(
                        t["input_ids"].unsqueeze(0).to(device), inputs_embeds=in_emb, image_features=c_feat
                    )
                    in_emb = in_emb.masked_scatter(mask, c_feat).squeeze(0).cpu()
                    prepared_samples.append({
                        "inputs_embeds": in_emb,
                        "attention_mask": t["attention_mask"],
                        "labels": t["labels"],
                    })
        else:
            cache_key = compute_pipeline_cache_key(f"group_{group.lower()}", dataset_hash=f"formal_{group}_{len(target_records)}")
            cache = PreparedDatasetCache(cache_dir=cache_dir)
            prepared_samples = cache.get_or_build(target_records, processor, cache_key=cache_key)

        for epoch in range(config.num_train_epochs):
            prefetcher = PinnedPrefetchDataLoader(
                prepared_samples,
                device=device,
                batch_size=config.per_device_train_batch_size,
                queue_size=8,
            )
            for batch in prefetcher:
                current_bs = batch["inputs_embeds"].shape[0] if "inputs_embeds" in batch else 1
                raw_loss_val, outputs = run_single_forward_step(model, batch)
                raw_losses.append(raw_loss_val)

                scaled_loss = outputs.loss / config.gradient_accumulation_steps
                scaled_loss.backward()

                microbatch_count += 1
                processed_samples += current_bs
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

                reporter.step(
                    microbatch=microbatch_count,
                    optimizer_step=optimizer_step_count,
                    scheduler_step=optimizer_step_count,
                    latest_raw_loss=raw_loss_val,
                    learning_rate=current_lr,
                    processed_samples=processed_samples,
                )

                if max_microbatches is not None and microbatch_count >= max_microbatches:
                    break
    else:
        # Reference execution path (synchronous, on-the-fly)
        for epoch in range(config.num_train_epochs):
            for rec in target_records:
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
                processed_samples += 1
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

                reporter.step(
                    microbatch=microbatch_count,
                    optimizer_step=optimizer_step_count,
                    scheduler_step=optimizer_step_count,
                    latest_raw_loss=raw_loss_val,
                    learning_rate=current_lr,
                    processed_samples=processed_samples,
                )

    # Final telemetry update
    if raw_losses:
        final_lr = optimizer.param_groups[0]["lr"]
        reporter.step(
            microbatch=microbatch_count,
            optimizer_step=optimizer_step_count,
            scheduler_step=optimizer_step_count,
            latest_raw_loss=raw_losses[-1],
            learning_rate=final_lr,
            force=True,
            status="completed",
            processed_samples=processed_samples,
        )

    telemetry_summary = sampler.stop() if sampler is not None else None

    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    wall_time = time.time() - t0

    # Save adapter
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)

    return {
        "training_profile_id": config.training_profile_id,
        "microbatch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.effective_batch_size,
        "gradient_checkpointing": config.gradient_checkpointing,
        "vision_cache_enabled": config.use_vision_cache,
        "total_microbatches": microbatch_count,
        "total_samples": processed_samples,
        "total_optimizer_steps": optimizer_step_count,
        "initial_loss": raw_losses[0] if raw_losses else 0.0,
        "final_loss": raw_losses[-1] if raw_losses else 0.0,
        "raw_losses": raw_losses,
        "step_records": step_records,
        "initial_vram_gb": initial_vram,
        "peak_vram_gb": peak_vram,
        "wall_time_sec": wall_time,
        "mean_microbatch_time_sec": wall_time / max(microbatch_count, 1),
        "mean_sample_time_sec": wall_time / max(processed_samples, 1),
        "samples_per_sec": processed_samples / max(wall_time, 0.001),
        "adapter_dir": str(adapter_dir),
        "telemetry": telemetry_summary,
        "vision_cache_stats": vc_cache.get_stats() if config.use_vision_cache else None,
        "vision_cache_disk_bytes": vc_cache.get_disk_size_bytes() if config.use_vision_cache else None,
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
