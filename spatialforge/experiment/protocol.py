"""Frozen formal experiment protocol and integrity guards for G2.0-E3.

Defines the authoritative frozen specifications for:
- Experiment identity: g2.0-e3
- Model: Qwen/Qwen2.5-VL-3B-Instruct
- Groups: B (zero-shot baseline), A (fixed south), C (4 canonical), D (4 canonical + 4 jitter)
- Formal seeds: (42, 123, 456); engineering seed: 42
- Generation settings: greedy decoding (do_sample=False)
- Dataset SHA-256 verification against frozen E1 manifest
- Test-set leakage guards (holdout scenes 080-099 prohibited from training)
- VSR test-only guard (external VSR is final transfer test data only, never for checkpoint selection)
- Collision-free run artifact directories and provenance capture
- Frozen evaluation set semantic definitions (S1 and S2)
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


# =====================================================================
# Frozen Experiment Specification Constants
# =====================================================================

EXPERIMENT_ID: str = "g2.0-e3"
MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MODEL_PATH: str = "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"

# Experimental conditions
GROUPS: Tuple[str, ...] = ("B", "A", "C", "D")

# Formal experimental seeds
FORMAL_SEEDS: Tuple[int, ...] = (42, 123, 456)
ENGINEERING_SEED: int = 42

# Training dataset mappings (B has no adapter / no training data)
TRAIN_DATASET_PATHS: Dict[str, Optional[str]] = {
    "A": "outputs/experiments/g2.0-e/train_group_a.jsonl",
    "C": "outputs/experiments/g2.0-e/train_group_c.jsonl",
    "D": "outputs/experiments/g2.0-e/train_group_d.jsonl",
    "B": None,
}

# Evaluation dataset paths
EVAL_DATASET_PATHS: Dict[str, str] = {
    "synthetic_s1": "outputs/experiments/g2.0-e/holdout_s1_cardinal.jsonl",
    "synthetic_s2": "outputs/experiments/g2.0-e/holdout_s2_jitter.jsonl",
    "vsr": "data/processed/vsr_test.jsonl",
}

E1_MANIFEST_PATH: str = "outputs/experiments/g2.0-e/e1_dataset_manifest.json"

# Authoritative frozen E1 dataset content SHA-256 hashes (PRIMARY immutable source of truth)
FROZEN_E1_DATASET_HASHES: Dict[str, str] = {
    "train_group_a.jsonl": "27c41955d52830d8f4d0a2d58ef33ca03290c16d683c47ee6fec48f6524655e0",
    "train_group_c.jsonl": "f7a8a5f31d194f1ea2a585753c5013bcf9d61f2bc8c50d366a4852fdb443449b",
    "train_group_d.jsonl": "263cd2634798f1debd74fa0c2f27f662686221e14642e63b7e73e66837698456",
    "holdout_s1_cardinal.jsonl": "1561e5fd40f2256d7d0e57b2a9a11bfc5ebbdd11585f60608423ab9f611394a4",
    "holdout_s2_jitter.jsonl": "bc8d42aabe1e9d5520f0d3c1c5575209b8e11488f78892da58b0b87066fdd132",
}

FROZEN_E1_GROUP_TO_FILE: Dict[str, str] = {
    "group_a": "train_group_a.jsonl",
    "group_c": "train_group_c.jsonl",
    "group_d": "train_group_d.jsonl",
    "holdout_s1": "holdout_s1_cardinal.jsonl",
    "holdout_s2": "holdout_s2_jitter.jsonl",
}

# Semantic descriptions for held-out evaluation sets
# CRITICAL INVARIANT: DO NOT call S2 "continuous OOD viewpoint generalization".
S1_SEMANTIC_LABEL: str = "unseen scenes + canonical horizontal views"
S2_SEMANTIC_LABEL: str = "unseen scenes + newly instantiated bounded jitter poses generated under the same jitter process"

# Generation settings
GENERATION_DO_SAMPLE: bool = False  # Greedy decoding
SYNTHETIC_MAX_NEW_TOKENS: int = 16  # Fixed tokens for directional QA
VSR_MAX_NEW_TOKENS: int = 8         # Recovered historical VSR value (scripts/eval_vsr.py)

# Parser policies
DIRECTIONAL_PARSER_POLICY: str = "strict_directional_v1"
YESNO_PARSER_POLICY: str = "strict_yesno_v1"
INVALID_PREDICTION_POLICY: str = "invalid_counts_as_incorrect_for_primary_accuracy"

# VSR External Test Guard: VSR is strictly for final transfer evaluation.
# It must NEVER be used for checkpoint selection, early stopping, hyperparameter tuning,
# or seed selection. Formal training runs 1 epoch and produces final adapter only.
IS_VSR_TEST_ONLY: bool = True

# Holdout scenes strictly reserved for evaluation diagnostics (scene_080..scene_099)
HOLDOUT_SCENE_IDS: frozenset = frozenset(f"scene_{i:03d}" for i in range(80, 100))


# =====================================================================
# Seed Policy & Execution Helper
# =====================================================================

def validate_seed(seed: int, allow_exploratory: bool = False) -> None:
    """Validate that seed is one of the frozen formal seeds or explicitly marked exploratory."""
    if seed not in FORMAL_SEEDS and not allow_exploratory:
        raise ValueError(
            f"Seed {seed} is not in frozen formal seeds {FORMAL_SEEDS}. "
            "Pass allow_exploratory=True only for non-formal exploratory runs."
        )


def seed_everything(seed: int) -> None:
    """Seed Python random, NumPy (if present), and PyTorch (CPU and CUDA).

    Scientific claim: Deterministic dataset order, explicit training seed, recorded seed,
    reproducible stochastic initialization and dropout policy.
    Does not force global deterministic algorithms that could destabilize CUDA ops without evidence.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# =====================================================================
# Dataset Hash & Split Guards
# =====================================================================

def compute_jsonl_content_hash(jsonl_path: Union[str, Path]) -> str:
    """Compute the deterministic content SHA-256 for a dataset JSONL.

    Matches exact G2.0-E1 dataset hashing specification:
    hash(f"{id}|{source_sample_id}|{presentation_order}|{view_id}|{family}|{question}|{answer}|{image_path}\n")
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset JSONL not found: {path}")

    hasher = hashlib.sha256()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            record_bytes = (
                f"{data['id']}|{data['source_sample_id']}|{data['presentation_order']}|"
                f"{data['view_id']}|{data['family']}|{data['question']}|{data['answer']}|{data['image_path']}\n"
            ).encode("utf-8")
            hasher.update(record_bytes)
    return hasher.hexdigest()


def verify_frozen_e1_datasets(
    manifest_path: Optional[Union[str, Path]] = E1_MANIFEST_PATH,
    base_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Verify SHA-256 hashes of frozen E1 datasets against tracked authoritative constants.

    Tracked constants in FROZEN_E1_DATASET_HASHES are the PRIMARY source of truth.
    - Actual dataset JSONL content SHA-256 MUST equal the tracked constant.
    - If manifest_path is provided, it is cross-checked secondarily:
      manifest expected hash must also match the tracked constant.
    - Manifest cannot authorize or override a changed dataset.
    - Fails hard (FileNotFoundError or RuntimeError) on:
      * missing dataset file
      * hash mismatch between actual dataset and tracked constant
      * manifest hash mismatch against tracked constant
    """
    target_keys = {
        "group_a": "train_group_a.jsonl",
        "group_c": "train_group_c.jsonl",
        "group_d": "train_group_d.jsonl",
        "holdout_s1": "holdout_s1_cardinal.jsonl",
        "holdout_s2": "holdout_s2_jitter.jsonl",
    }

    groups_manifest: Dict[str, Any] = {}
    if manifest_path is not None:
        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise FileNotFoundError(f"E1 dataset manifest not found: {manifest_file}")

        try:
            with manifest_file.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            groups_manifest = manifest.get("groups", {})
        except Exception as e:
            raise RuntimeError(f"Failed to parse E1 dataset manifest {manifest_file}: {e}") from e

    verified_hashes: Dict[str, str] = {}
    for key, expected_filename in target_keys.items():
        tracked_hash = FROZEN_E1_DATASET_HASHES.get(expected_filename)
        if not tracked_hash:
            raise RuntimeError(f"No tracked authoritative hash defined for {expected_filename}")

        rel_file = f"outputs/experiments/g2.0-e/{expected_filename}"
        if key in groups_manifest:
            group_info = groups_manifest[key]
            rel_file = group_info.get("file", rel_file)
            manifest_hash = group_info.get("sha256")
            if manifest_hash != tracked_hash:
                raise RuntimeError(
                    f"E1 manifest hash mismatch against tracked constant for {key}: "
                    f"manifest={manifest_hash}, tracked={tracked_hash}"
                )

        file_path = (Path(base_dir) / rel_file) if base_dir else Path(rel_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Frozen dataset file missing: {file_path}")

        actual_hash = compute_jsonl_content_hash(file_path)
        if actual_hash != tracked_hash:
            raise RuntimeError(
                f"Frozen E1 dataset hash mismatch for {file_path} ({key}): "
                f"expected tracked {tracked_hash}, got {actual_hash}"
            )
        verified_hashes[key] = actual_hash

    return verified_hashes


def validate_training_scenes(records: Sequence[Any]) -> None:
    """Ensure no holdout scene (scene_080..scene_099) leaks into training records.

    Accepts sequence of objects with 'scene_id' attribute or dicts with 'scene_id' key.
    Fails hard (ValueError) on any leakage.
    """
    leaked_scenes: Set[str] = set()
    for r in records:
        scene_id = getattr(r, "scene_id", None) or (r.get("scene_id") if isinstance(r, dict) else None)
        if scene_id in HOLDOUT_SCENE_IDS:
            leaked_scenes.add(scene_id)

    if leaked_scenes:
        raise ValueError(
            f"Test-set leakage detected! Training data contains holdout scenes: {sorted(leaked_scenes)}"
        )


# =====================================================================
# Run Artifact Contract & Path Helpers
# =====================================================================

def get_run_output_dir(
    group: str,
    seed: Optional[int] = None,
    experiment_id: str = EXPERIMENT_ID,
    base_dir: Union[str, Path] = Path("outputs/experiments"),
) -> Path:
    """Return collision-free artifact directory path for a run.

    Structure:
        outputs/experiments/<experiment_id>/baseline_b/
        outputs/experiments/<experiment_id>/group_a/seed_<seed>/
        outputs/experiments/<experiment_id>/group_c/seed_<seed>/
        outputs/experiments/<experiment_id>/group_d/seed_<seed>/
    """
    group_norm = group.strip().lower()
    if group_norm in ("b", "baseline_b", "group_b"):
        return Path(base_dir) / experiment_id / "baseline_b"

    canonical_group = f"group_{group_norm[-1]}" if len(group_norm) == 1 else group_norm
    if canonical_group not in ("group_a", "group_c", "group_d"):
        raise ValueError(f"Unknown group: {group!r}")

    if seed is None:
        raise ValueError(f"Seed required for trained group {canonical_group}")

    return Path(base_dir) / experiment_id / canonical_group / f"seed_{seed}"


@dataclass(frozen=True)
class RunManifest:
    """Complete provenance and artifact manifest for a formal run."""

    experiment_id: str
    group: str
    seed: Optional[int]
    model_id: str
    model_revision: Optional[str]
    dataset_hash: Optional[str]
    training_config: Optional[Dict[str, Any]]
    dependency_versions: Dict[str, str]
    gpu: str
    git_commit: Optional[str]
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str = "initialized"
    artifacts: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "group": self.group,
            "seed": self.seed,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dataset_hash": self.dataset_hash,
            "training_config": self.training_config,
            "dependency_versions": self.dependency_versions,
            "gpu": self.gpu,
            "git_commit": self.git_commit,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "artifacts": self.artifacts or {},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)


# =====================================================================
# Model Revision & Provenance Helpers
# =====================================================================

def recover_model_revision(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Optional[str]:
    """Recover the exact Hugging Face snapshot revision hash for the local model."""
    path = Path(model_path)
    # Check 1: model path .cache/huggingface/trees/*.json
    tree_dir = path / ".cache" / "huggingface" / "trees"
    if tree_dir.exists():
        json_files = list(tree_dir.glob("*.json"))
        if json_files:
            return json_files[0].stem

    # Check 2: ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/refs/main
    user_hub_ref = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "refs" / "main"
    if user_hub_ref.exists():
        commit = user_hub_ref.read_text(encoding="utf-8").strip()
        if commit:
            return commit

    return None


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash if in a git repository."""
    try:
        import subprocess
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def get_environment_provenance() -> Dict[str, str]:
    """Record runtime dependency versions and hardware info."""
    import platform

    provenance = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    for pkg in ("torch", "transformers", "peft", "qwen_vl_utils", "PIL"):
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            provenance[pkg] = version
        except ImportError:
            provenance[pkg] = "not_installed"

    try:
        import torch
        provenance["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            provenance["cuda_version"] = str(torch.version.cuda)
            provenance["gpu_device"] = torch.cuda.get_device_name(0)
            provenance["gpu_count"] = str(torch.cuda.device_count())
        else:
            provenance["cuda_version"] = "none"
            provenance["gpu_device"] = "none"
    except ImportError:
        pass

    commit = get_git_commit_hash()
    if commit:
        provenance["git_commit"] = commit

    return provenance
