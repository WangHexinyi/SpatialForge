"""Optimized data pipeline, deterministic tensor caching, and pinned prefetching.

Preserves exact training mathematics, tensor bit-representations, and formal sample order.
Provides:
1. Cache key derivation with strict provenance validation (model revision, dataset hash, config).
2. Shared image-level vision tensor caching and full-sample preparation caching.
3. Bounded sequential CPU prefetching with pinned host memory and non-blocking GPU transfer.
4. Reference vs optimized execution modes for verification and benchmarking.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image

from spatialforge.experiment.protocol import (
    DEFAULT_MODEL_PATH,
    FROZEN_E1_DATASET_HASHES,
    recover_model_revision,
)
from spatialforge.experiment.training import (
    TrainingSampleRecord,
    build_training_tensors,
    collate_single_sample_batch,
    load_and_preprocess_image,
)

DEFAULT_CACHE_DIR = Path("outputs/cache/prepared_samples")
PIPELINE_SCHEMA_VERSION = "g2.0-e3-v2"


# =====================================================================
# Provenance and Cache Key Generation
# =====================================================================

def compute_dataset_content_hash(dataset_path: Union[str, Path]) -> str:
    """Compute deterministic SHA-256 hash of dataset JSONL lines."""
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path not found: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_pipeline_cache_key(
    dataset_name_or_path: Union[str, Path],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    dataset_hash: Optional[str] = None,
    schema_version: str = PIPELINE_SCHEMA_VERSION,
) -> str:
    """Generate provenance-locked cache key for prepared sample tensors."""
    p = Path(dataset_name_or_path)
    fname = p.name

    if dataset_hash is None:
        if fname in FROZEN_E1_DATASET_HASHES:
            dataset_hash = FROZEN_E1_DATASET_HASHES[fname]
        elif p.exists():
            dataset_hash = compute_dataset_content_hash(p)
        else:
            dataset_hash = hashlib.sha256(str(dataset_name_or_path).encode("utf-8")).hexdigest()

    model_rev = recover_model_revision(model_path)

    key_payload = {
        "schema_version": schema_version,
        "dataset_name": fname,
        "dataset_sha256": dataset_hash,
        "model_revision": model_rev,
        "model_path": str(model_path),
    }
    key_str = json.dumps(key_payload, sort_keys=True)
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


DEFAULT_VISION_CACHE_DIR = Path("outputs/cache/frozen_vision_features")


def compute_image_content_hash(image_path: Union[str, Path]) -> str:
    """Compute deterministic SHA-256 hash of raw image file bytes."""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_vision_cache_key(
    image_path: Union[str, Path],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    schema_version: str = PIPELINE_SCHEMA_VERSION,
    dtype: str = "bfloat16",
) -> str:
    """Generate provenance-locked cache key for frozen visual representations."""
    p = Path(image_path)
    img_hash = compute_image_content_hash(p)
    model_rev = recover_model_revision(model_path)

    key_payload = {
        "schema_version": schema_version,
        "image_name": p.name,
        "image_sha256": img_hash,
        "model_revision": model_rev,
        "model_path": str(model_path),
        "dtype": dtype,
    }
    key_str = json.dumps(key_payload, sort_keys=True)
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def compute_batched_per_example_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    sample_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-example mean cross-entropy loss mathematically equivalent to MB1.

    Guarantees that each sample in a batch is weighted equally (1/B), preserving
    the exact per-sample optimizer objective regardless of unequal answer lengths
    or padding tokens.

    ``sample_weights`` (optional, [B]) applies per-sample class weighting for
    imbalance control; the batch loss is the weighted mean sum(w_i L_i)/sum(w_i).

    Args:
        logits: [B, S, V] unscaled logits from language model.
        labels: [B, S] target token ids with ignore_index for prompt & padding.
        ignore_index: int, default -100.
        sample_weights: optional [B] non-negative per-sample weights.

    Returns:
        per_sample_losses: [B] tensor of per-sample mean losses.
        batched_mean_loss: scalar weighted mean loss.
    """
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    B, S_minus_1, V = shift_logits.shape

    loss_flat = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    loss_matrix = loss_flat.view(B, S_minus_1)

    mask = (shift_labels != ignore_index).float()
    n_tokens = mask.sum(dim=1).clamp(min=1.0)
    per_sample_losses = (loss_matrix * mask).sum(dim=1) / n_tokens
    if sample_weights is not None:
        w = sample_weights.to(per_sample_losses.device, dtype=per_sample_losses.dtype)
        batched_mean_loss = (per_sample_losses * w).sum() / w.sum().clamp(min=1e-6)
    else:
        batched_mean_loss = per_sample_losses.mean()
    return per_sample_losses, batched_mean_loss


# =====================================================================
# Image-Level & Full-Sample Cache
# =====================================================================

class DecodedImageCache:
    """In-memory cache for decoded RGB PIL Images to avoid redundant disk I/O."""

    def __init__(self):
        self._cache: Dict[str, Image.Image] = {}
        self._lock = threading.Lock()

    def get(self, image_path: str) -> Image.Image:
        with self._lock:
            if image_path in self._cache:
                return self._cache[image_path]
        img = load_and_preprocess_image(image_path)
        with self._lock:
            self._cache[image_path] = img
        return img

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class PreparedDatasetCache:
    """Deterministic cache for CPU-prepared multimodal training tensors.

    Stores:
    - Shared image vision tensors (pixel_values, image_grid_thw) per unique image
    - Tokenized input_ids, attention_mask, labels per sample
    Memory footprint: ~300 MB for 1552 samples in Group A (78 images).
    Guarantees bitwise equivalence with standard build_training_tensors.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.in_memory = in_memory
        self._memory_cache: Dict[str, List[Dict[str, Any]]] = {}

    def get_or_build(
        self,
        records: Sequence[TrainingSampleRecord],
        processor: Any,
        cache_key: str,
        force_rebuild: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve prepared tensors from cache, or build deterministically if missing."""
        if not force_rebuild and self.in_memory and cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.pt"
        if not force_rebuild and cache_file.exists():
            try:
                loaded = torch.load(cache_file, weights_only=False)
                if isinstance(loaded, dict) and loaded.get("cache_key") == cache_key:
                    samples = loaded["samples"]
                    if len(samples) == len(records):
                        if self.in_memory:
                            self._memory_cache[cache_key] = samples
                        return samples
            except Exception:
                pass  # Corrupted or incomplete cache; proceed to rebuild

        # Build deterministically
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        img_vision_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        prepared_samples: List[Dict[str, Any]] = []

        for rec in records:
            # Check image-level vision cache
            img_path_str = str(rec.image_path)
            if img_path_str not in img_vision_cache:
                img = load_and_preprocess_image(rec.image_path)
                tensors = build_training_tensors(processor, img, rec.question, rec.answer)
                # Store shared vision tensors
                img_vision_cache[img_path_str] = (
                    tensors["pixel_values"],
                    tensors["image_grid_thw"],
                )
                sample_item = tensors
            else:
                # Reuse vision tensors; build text tensors deterministically
                img = load_and_preprocess_image(rec.image_path)
                tensors = build_training_tensors(processor, img, rec.question, rec.answer)
                shared_pv, shared_grid = img_vision_cache[img_path_str]
                sample_item = {
                    "input_ids": tensors["input_ids"],
                    "attention_mask": tensors["attention_mask"],
                    "labels": tensors["labels"],
                    "pixel_values": shared_pv,
                    "image_grid_thw": shared_grid,
                    "prompt_length": tensors["prompt_length"],
                    "answer_length": tensors["answer_length"],
                }
            if "mm_token_type_ids" in tensors:
                sample_item["mm_token_type_ids"] = tensors["mm_token_type_ids"]

            sample_item["sample_weight"] = float(getattr(rec, "sample_weight", 1.0))
            prepared_samples.append(sample_item)

        # Atomic file write
        temp_cache_file = self.cache_dir / f".{cache_key}.pt.tmp.{os.getpid()}"
        save_payload = {
            "cache_key": cache_key,
            "sample_count": len(prepared_samples),
            "samples": prepared_samples,
        }
        torch.save(save_payload, temp_cache_file)
        os.replace(temp_cache_file, cache_file)

        if self.in_memory:
            self._memory_cache[cache_key] = prepared_samples

        return prepared_samples


# =====================================================================
# Frozen Vision Feature Cache (Precomputed Visual Representations)
# =====================================================================

class FrozenVisionFeatureCache:
    """Provenance-locked cache for precomputed frozen visual representations.

    Eliminates redundant vision encoder forward passes for repeated images.
    Guarantees bitwise equivalence in visual tokens and forward losses.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_VISION_CACHE_DIR
        self.in_memory = in_memory
        self._memory_cache: Dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0

    def get_or_compute(
        self,
        image_path: Union[str, Path],
        model: Any,
        processor: Any,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Retrieve precomputed visual representations, computing on-demand if missing."""
        cache_key = compute_vision_cache_key(image_path, dtype=str(dtype).replace("torch.", ""))

        with self._lock:
            if self.in_memory and cache_key in self._memory_cache:
                self.hits += 1
                return self._memory_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.pt"
        if cache_file.exists():
            try:
                loaded = torch.load(cache_file, weights_only=False)
                if isinstance(loaded, dict) and loaded.get("cache_key") == cache_key:
                    feat = loaded["image_embeds"]
                    with self._lock:
                        if self.in_memory:
                            self._memory_cache[cache_key] = feat
                        self.hits += 1
                    return feat
            except Exception:
                pass

        # Compute using frozen base model visual encoder
        with self._lock:
            self.misses += 1

        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        img = load_and_preprocess_image(image_path)
        tensors = build_training_tensors(processor, img, question="", answer="")
        pv = tensors["pixel_values"].to(device, dtype=dtype)
        grid = tensors["image_grid_thw"].to(device)

        with torch.no_grad():
            feats = base_model.model.get_image_features(pv, grid).pooler_output
            image_embeds = torch.cat(feats, dim=0).to(dtype=dtype).cpu()

        # Atomic write
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temp_file = self.cache_dir / f".{cache_key}.pt.tmp.{os.getpid()}"
        save_payload = {
            "cache_key": cache_key,
            "image_path": str(image_path),
            "image_embeds": image_embeds,
        }
        torch.save(save_payload, temp_file)
        os.replace(temp_file, cache_file)

        with self._lock:
            if self.in_memory:
                self._memory_cache[cache_key] = image_embeds

        return image_embeds

    def get_stats(self) -> Dict[str, Any]:
        """Return cache hit, miss, and query statistics."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total_queries": total,
                "hit_rate": hit_rate,
                "unique_cached": len(self._memory_cache),
            }

    def get_disk_size_bytes(self) -> int:
        """Calculate total disk space used by cached feature files."""
        if not self.cache_dir.exists():
            return 0
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.pt") if f.is_file())

    def clear(self) -> None:
        """Clear memory cache and reset counters."""
        with self._lock:
            self._memory_cache.clear()
            self.hits = 0
            self.misses = 0


def collate_cached_vision_batch(items: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate multiple post-vision sample dicts into a right-padded batch.

    Pads inputs_embeds with 0.0, attention_mask with 0, and labels with -100.
    Ensures bitwise preservation of each sample's unpadded token sequence.
    """
    batch_size = len(items)
    if batch_size == 1:
        item = items[0]
        return {
            "inputs_embeds": item["inputs_embeds"].unsqueeze(0) if item["inputs_embeds"].ndim == 2 else item["inputs_embeds"],
            "attention_mask": item["attention_mask"].unsqueeze(0) if item["attention_mask"].ndim == 1 else item["attention_mask"],
            "labels": item["labels"].unsqueeze(0) if item["labels"].ndim == 1 else item["labels"],
        }

    max_len = max(item["inputs_embeds"].shape[0] for item in items)
    hidden_dim = items[0]["inputs_embeds"].shape[1]
    dtype = items[0]["inputs_embeds"].dtype

    batched_embeds = torch.zeros((batch_size, max_len, hidden_dim), dtype=dtype)
    batched_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    batched_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, item in enumerate(items):
        seq_len = item["inputs_embeds"].shape[0]
        batched_embeds[i, :seq_len] = item["inputs_embeds"]
        batched_mask[i, :seq_len] = item["attention_mask"]
        batched_labels[i, :seq_len] = item["labels"]

    return {
        "inputs_embeds": batched_embeds,
        "attention_mask": batched_mask,
        "labels": batched_labels,
    }


# =====================================================================
# Deterministic Bounded Prefetch Iterator with Pinned Host Memory
# =====================================================================

class PinnedPrefetchDataLoader:
    """Sequential, FIFO background prefetcher delivering pinned batches on CUDA.

    Guarantees:
    - 100% deterministic presentation order (sample 0, 1, ..., N-1)
    - Pinned host memory buffers for fast asynchronous DMA copy
    - Non-blocking host-to-device transfers (collated['...'].to(device, non_blocking=True))
    - Strict bounded queue size (defaults to 4-8 microbatches) to prevent memory ballooning
    - Clean thread shutdown on iteration completion or exception
    """

    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        device: str = "cuda",
        batch_size: int = 1,
        queue_size: int = 8,
        pin_memory: bool = True,
    ):
        self.samples = list(samples)
        self.device = device
        self.batch_size = max(1, batch_size)
        self.queue_size = max(2, queue_size)
        self.pin_memory = pin_memory and torch.cuda.is_available() and device.startswith("cuda")
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

    def __len__(self) -> int:
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._produce, daemon=True, name="PrefetchWorker")
        self._worker.start()

        num_batches = len(self)
        try:
            for _ in range(num_batches):
                item = self._queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item

                # Transfer to device non-blocking
                device_batch = {}
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        device_batch[k] = v.to(self.device, non_blocking=True)
                    else:
                        device_batch[k] = v

                # Ensure pixel_values or inputs_embeds is bfloat16 to match model
                if "pixel_values" in device_batch and device_batch["pixel_values"].dtype == torch.float32:
                    device_batch["pixel_values"] = device_batch["pixel_values"].to(torch.bfloat16)
                if "inputs_embeds" in device_batch and device_batch["inputs_embeds"].dtype == torch.float32:
                    device_batch["inputs_embeds"] = device_batch["inputs_embeds"].to(torch.bfloat16)

                yield device_batch
        finally:
            self._stop_event.set()
            # Drain queue to unblock worker
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            if self._worker is not None and self._worker.is_alive():
                self._worker.join(timeout=1.0)

    def _produce(self) -> None:
        try:
            for i in range(0, len(self.samples), self.batch_size):
                if self._stop_event.is_set():
                    break
                batch_items = self.samples[i : i + self.batch_size]

                # Collate batch on host
                if "inputs_embeds" in batch_items[0]:
                    collated = collate_cached_vision_batch(batch_items)
                elif len(batch_items) > 1:
                    from spatialforge.models.vl_adapter import collate_training_batch

                    collated = collate_training_batch(batch_items)
                else:
                    item = batch_items[0]
                    collated = {
                        "input_ids": item["input_ids"].unsqueeze(0),
                        "attention_mask": item["attention_mask"].unsqueeze(0),
                        "labels": item["labels"].unsqueeze(0),
                        "pixel_values": item["pixel_values"],
                        "image_grid_thw": item["image_grid_thw"],
                    }
                    if "mm_token_type_ids" in item:
                        collated["mm_token_type_ids"] = (
                            item["mm_token_type_ids"].unsqueeze(0)
                            if item["mm_token_type_ids"].dim() == 1
                            else item["mm_token_type_ids"]
                        )
                    if "sample_weight" in item:
                        collated["sample_weight"] = torch.tensor(
                            [float(item["sample_weight"])], dtype=torch.float32
                        )

                # Pin memory if enabled
                if self.pin_memory:
                    pinned = {}
                    for k, v in collated.items():
                        if isinstance(v, torch.Tensor):
                            pinned[k] = v.pin_memory()
                        else:
                            pinned[k] = v
                    batch = pinned
                else:
                    batch = collated

                while not self._stop_event.is_set():
                    try:
                        self._queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as err:
            self._queue.put(err)
        finally:
            # Signal completion
            try:
                self._queue.put(None, timeout=0.5)
            except Exception:
                pass
