"""Multimodal dataset materialization and experiment formatting for G2.0-E.

Bridges:
    SceneState
    +
    QASample (G2.0-D)
    +
    ObservationMetadata / render manifest
        ↓
    visually grounded multimodal SFT record (MultimodalSample)

Semantics & Visual Grounding:
- QASample uses internal object identifiers: object 0 ("obj0").
- The rendered images do not show internal variable names.
- Following recovered v1, model-facing referring expressions are "the {color} {shape}".
- If either referenced object has an ambiguous (non-unique) color+shape in the scene,
  the sample is strictly excluded from multimodal training/evaluation.
- Surface text is adapted while preserving QASample identity, family, answer,
  relation truth, object indices, view IDs, and curriculum tags.
- Pure Python; does not require Blender.
"""

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from spatialforge.environment.camera import CameraPose
from spatialforge.environment.qa import QASample, _SINGLE_OPTION_PHRASE, generate_single_view_qa
from spatialforge.environment.relation import compute_spatial_truth
from spatialforge.environment.scene import SceneObject, SceneState
from spatialforge.environment.views import VIEW_IDS, generate_cardinal_views
from spatialforge.environment.sampling import jitter_pose

# Strict mathematical inverse mappings for directional answers
INVERSE_ANSWER: Dict[str, str] = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "front": "behind",
    "behind": "front",
    "nearer": "farther",
    "farther": "nearer",
}

POS_LABELS: Dict[str, str] = {
    "horizontal": "left",
    "vertical": "above",
    "depth": "front",
    "near_far": "nearer",
}

NEG_LABELS: Dict[str, str] = {
    "horizontal": "right",
    "vertical": "below",
    "depth": "behind",
    "near_far": "farther",
}


@dataclass(frozen=True)
class MultimodalSample:
    """One visually grounded multimodal VLM training or evaluation sample."""

    sample_id: str
    source_sample_id: str
    presentation_order: int
    scene_id: str
    view_id: str
    family: str
    image_path: str
    question: str
    answer: str
    object_indices: Tuple[int, ...]
    tags: Tuple[str, ...]
    is_view_dependent: bool
    conversations: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic JSON-compatible serialization."""
        return {
            "id": self.sample_id,
            "source_sample_id": self.source_sample_id,
            "presentation_order": self.presentation_order,
            "scene_id": self.scene_id,
            "view_id": self.view_id,
            "family": self.family,
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "object_indices": list(self.object_indices),
            "tags": list(self.tags),
            "is_view_dependent": self.is_view_dependent,
            "conversations": list(self.conversations),
        }


def visual_object_ref(obj: SceneObject) -> str:
    """Create a natural model-facing visual referring expression."""
    return f"the {obj.color} {obj.shape}"


def format_single_question(family: str, ref_a: str, ref_b: str) -> str:
    """Structurally construct the model-facing question without brittle string replacement."""
    return f"From this view, is {ref_a} {_SINGLE_OPTION_PHRASE[family]} {ref_b}?"


def is_scene_visually_ambiguous(state: SceneState, index_a: int, index_b: int) -> bool:
    """True if either object lacks a unique color+shape descriptor in the scene."""
    descriptors = [visual_object_ref(obj) for obj in state.objects]
    counts = Counter(descriptors)
    desc_a = descriptors[index_a]
    desc_b = descriptors[index_b]
    return counts[desc_a] > 1 or counts[desc_b] > 1


def adapt_question_visual_refs(
    question: str,
    state: SceneState,
    index_a: int,
    index_b: int,
) -> str:
    """Rewrite raw internal QASample object references into visual expressions.

    Maintained for backwards-compatibility; callers are encouraged to use
    format_single_question directly.
    """
    obj_a = state.objects[index_a]
    obj_b = state.objects[index_b]

    internal_ref_a = f'object {index_a} ("{obj_a.name}")'
    internal_ref_b = f'object {index_b} ("{obj_b.name}")'

    visual_ref_a = visual_object_ref(obj_a)
    visual_ref_b = visual_object_ref(obj_b)

    adapted = question.replace(internal_ref_a, visual_ref_a)
    adapted = adapted.replace(internal_ref_b, visual_ref_b)
    return adapted


def build_multimodal_sample(
    qa: QASample,
    state: SceneState,
    image_path: str,
    presentation_order: int = 0,
) -> Optional[MultimodalSample]:
    """Convert a single-view QASample into a MultimodalSample.

    Args:
        qa: Underlying G2.0-D QASample.
        state: Authoritative SceneState.
        image_path: Path to rendered view image.
        presentation_order: 0 for canonical (A relative to B), 1 for inverted (B relative to A).

    Returns None if the referenced objects are visually ambiguous in the scene.
    """
    if len(qa.object_indices) != 2:
        return None

    idx_a, idx_b = qa.object_indices[0], qa.object_indices[1]
    if is_scene_visually_ambiguous(state, idx_a, idx_b):
        return None

    if presentation_order == 0:
        presented_indices = (idx_a, idx_b)
        answer = qa.answer
    elif presentation_order == 1:
        presented_indices = (idx_b, idx_a)
        answer = INVERSE_ANSWER[qa.answer]
    else:
        raise ValueError(f"Invalid presentation_order {presentation_order}; expected 0 or 1.")

    obj_first = state.objects[presented_indices[0]]
    obj_second = state.objects[presented_indices[1]]
    ref_first = visual_object_ref(obj_first)
    ref_second = visual_object_ref(obj_second)

    question = format_single_question(qa.family, ref_first, ref_second)
    view_id = qa.view_ids[0] if qa.view_ids else "unknown"
    sample_id = f"{qa.sample_id}:order_{presentation_order}"

    conversations = (
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    )

    return MultimodalSample(
        sample_id=sample_id,
        source_sample_id=qa.sample_id,
        presentation_order=presentation_order,
        scene_id=qa.scene_id,
        view_id=view_id,
        family=qa.family,
        image_path=image_path,
        question=question,
        answer=answer,
        object_indices=presented_indices,
        tags=qa.tags,
        is_view_dependent=qa.is_view_dependent,
        conversations=conversations,
    )


def load_manifest_image_lookup(rendered_dir: Path) -> Dict[Tuple[str, str], str]:
    """Load all scene manifests in a directory and return a (scene_id, view_id) -> image_path map.

    Supports:
        - per-scene manifest: rendered_dir / scene_XXX / manifest.json
        - legacy manifests dir: rendered_dir / manifests / scene_XXX_manifest.json
        - flat manifests: rendered_dir / scene_XXX_manifest.json
    """
    lookup: Dict[Tuple[str, str], str] = {}
    rendered_dir = Path(rendered_dir)
    if not rendered_dir.exists():
        return lookup

    manifest_paths = list(rendered_dir.glob("scene_*/manifest.json")) + list(
        rendered_dir.glob("*/manifest.json")
    )
    if not manifest_paths:
        manifest_paths = list(rendered_dir.glob("manifests/*_manifest.json")) + list(
            rendered_dir.glob("*_manifest.json")
        )

    for path in sorted(set(manifest_paths)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            scene_id = data.get("scene_id", "")
            for obs in data.get("observations", []):
                view_id = obs.get("view_id", "")
                img_path = obs.get("image_path", "")
                if scene_id and view_id and img_path:
                    lookup[(scene_id, view_id)] = img_path
        except Exception:
            continue
    return lookup


def get_camera_plan_for_scene(
    plan_name: str,
    scene: SceneState,
) -> List[Tuple[str, CameraPose]]:
    """Return deterministic (view_id, CameraPose) for primary plans."""
    cardinal_cams = generate_cardinal_views()
    cardinal_pairs = list(zip(VIEW_IDS, cardinal_cams))

    if plan_name == "primary_a":
        return [cardinal_pairs[0]]  # south

    elif plan_name == "primary_c":
        return cardinal_pairs  # south, east, north, west

    elif plan_name == "primary_d":
        plan = list(cardinal_pairs)
        for idx, (view_id, cam) in enumerate(cardinal_pairs):
            jitter_seed = scene.seed * 1000 + idx * 17 + 42
            j_cam = jitter_pose(
                cam,
                seed=jitter_seed,
                angular_deg=15.0,
                radius_frac=0.10,
                target_radius=0.0,
            )
            plan.append((f"{view_id}_jitter", j_cam))
        return plan

    elif plan_name == "holdout_s2_jitter":
        # 4 novel jitter views from primary_d on holdout scenes
        plan = []
        for idx, (view_id, cam) in enumerate(cardinal_pairs):
            jitter_seed = scene.seed * 1000 + idx * 17 + 42
            j_cam = jitter_pose(
                cam,
                seed=jitter_seed,
                angular_deg=15.0,
                radius_frac=0.10,
                target_radius=0.0,
            )
            plan.append((f"{view_id}_jitter", j_cam))
        return plan

    else:
        raise ValueError(f"Unknown camera plan: {plan_name!r}")


def materialize_group_pool(
    scenes: Sequence[SceneState],
    plan_name: str,
    image_lookup: Optional[Dict[Tuple[str, str], str]] = None,
    default_image_pattern: str = "outputs/experiments/g2.0-e/rendered/{scene_id}/{view_id}.png",
    normalize_presentation: bool = True,
) -> Tuple[List[MultimodalSample], Dict[str, Any]]:
    """Materialize the full eligible single-view pool for a group and return comprehensive counts.

    If normalize_presentation is True, deterministic alternating operand ordering is applied
    per (view_id, family) across sorted scenes and pairs to balance directional answer labels.
    """
    raw_by_cell: Dict[Tuple[str, str], List[Tuple[SceneState, QASample, str]]] = defaultdict(list)
    family_counts: Counter = Counter()
    view_counts: Counter = Counter()
    scene_counts: Counter = Counter()
    exclusion_counts: Counter = Counter()

    for scene in scenes:
        camera_plan = get_camera_plan_for_scene(plan_name, scene)
        for view_id, cam in camera_plan:
            key = (scene.scene_id, view_id)
            if image_lookup and key in image_lookup:
                img_path = image_lookup[key]
            else:
                img_path = default_image_pattern.format(scene_id=scene.scene_id, view_id=view_id)

            truth = compute_spatial_truth(scene, cam)
            qa_samples = generate_single_view_qa(truth, view_id)

            for qa in qa_samples:
                idx_a, idx_b = qa.object_indices
                if is_scene_visually_ambiguous(scene, idx_a, idx_b):
                    exclusion_counts["ambiguous_visual_reference"] += 1
                    continue
                raw_by_cell[(view_id, qa.family)].append((scene, qa, img_path))

    pool: List[MultimodalSample] = []

    for (view_id, fam), items in sorted(raw_by_cell.items()):
        items.sort(key=lambda x: (x[0].scene_id, x[1].object_indices))
        for k, (scene, qa, img_path) in enumerate(items):
            if normalize_presentation:
                target_label = POS_LABELS[fam] if k % 2 == 0 else NEG_LABELS[fam]
                order = 0 if qa.answer == target_label else 1
            else:
                order = 0

            sample = build_multimodal_sample(qa, scene, img_path, presentation_order=order)
            if sample is None:
                exclusion_counts["build_failure"] += 1
                continue

            pool.append(sample)
            family_counts[sample.family] += 1
            view_counts[sample.view_id] += 1
            scene_counts[sample.scene_id] += 1

    stats = {
        "plan_name": plan_name,
        "total_scenes": len(scenes),
        "total_eligible": len(pool),
        "by_family": dict(sorted(family_counts.items())),
        "by_view": dict(sorted(view_counts.items())),
        "by_scene": dict(sorted(scene_counts.items())),
        "exclusions": dict(sorted(exclusion_counts.items())),
    }
    return pool, stats


def sample_balanced_budget(
    pool: Sequence[MultimodalSample],
    target_budget: int,
    seed: int = 42,
) -> List[MultimodalSample]:
    """Deterministically sample a balanced subset across families, viewpoints, and scenes.

    Ensures:
      1. Equal allocation across the 4 relation families (target_budget // 4 each).
      2. Balanced allocation across viewpoints (cell max-min <= 1).
      3. Uniform exposure across scenes via round-robin over available scene IDs.
      4. Balanced binary answer labels per cell and per family (diff <= 1).
      5. Strictly deterministic execution with zero global RNG mutation.
    """
    if target_budget > len(pool):
        raise ValueError(
            f"Target budget {target_budget} exceeds available eligible pool {len(pool)}"
        )
    if target_budget % 4 != 0:
        raise ValueError(f"Target budget must be divisible by 4 (families), got {target_budget}")

    per_family_budget = target_budget // 4

    by_family: Dict[str, List[MultimodalSample]] = defaultdict(list)
    for s in pool:
        by_family[s.family].append(s)

    selected: List[MultimodalSample] = []

    for fam in ("horizontal", "vertical", "depth", "near_far"):
        fam_samples = by_family[fam]
        if len(fam_samples) < per_family_budget:
            raise ValueError(
                f"Family {fam} has only {len(fam_samples)} samples, cannot satisfy {per_family_budget}"
            )

        by_view: Dict[str, List[MultimodalSample]] = defaultdict(list)
        for s in fam_samples:
            by_view[s.view_id].append(s)

        views = sorted(by_view.keys())
        num_views = len(views)
        base_per_view = per_family_budget // num_views
        remainder = per_family_budget % num_views
        view_targets = {vid: base_per_view + (1 if i < remainder else 0) for i, vid in enumerate(views)}

        fam_selected: List[MultimodalSample] = []

        for v_idx, vid in enumerate(views):
            target_k = view_targets[vid]
            candidates = by_view[vid]
            if target_k >= len(candidates):
                fam_selected.extend(candidates)
                continue

            by_scene: Dict[str, List[MultimodalSample]] = defaultdict(list)
            for c in candidates:
                by_scene[c.scene_id].append(c)

            scenes_list = sorted(by_scene.keys())
            num_scenes = len(scenes_list)
            offset = (v_idx * (num_scenes // num_views if num_views > 1 else 0)) % num_scenes
            ordered_scenes = scenes_list[offset:] + scenes_list[:offset]

            target_pos = target_k // 2 + (1 if (target_k % 2 == 1 and v_idx % 2 == 0) else 0)
            target_neg = target_k - target_pos

            pos_label = POS_LABELS.get(fam, "")
            neg_label = NEG_LABELS.get(fam, "")

            scene_pos: Dict[str, List[MultimodalSample]] = {}
            scene_neg: Dict[str, List[MultimodalSample]] = {}
            for sc in ordered_scenes:
                p_list = [c for c in by_scene[sc] if c.answer == pos_label]
                n_list = [c for c in by_scene[sc] if c.answer == neg_label]
                if p_list:
                    shift = v_idx % len(p_list)
                    p_list = p_list[shift:] + p_list[:shift]
                if n_list:
                    shift = v_idx % len(n_list)
                    n_list = n_list[shift:] + n_list[:shift]
                scene_pos[sc] = p_list
                scene_neg[sc] = n_list

            chosen_pos: List[MultimodalSample] = []
            chosen_neg: List[MultimodalSample] = []

            while len(chosen_pos) < target_pos:
                progress = False
                for sc in ordered_scenes:
                    if len(chosen_pos) >= target_pos:
                        break
                    if scene_pos[sc]:
                        chosen_pos.append(scene_pos[sc].pop(0))
                        progress = True
                if not progress:
                    break

            while len(chosen_neg) < target_neg:
                progress = False
                for sc in ordered_scenes:
                    if len(chosen_neg) >= target_neg:
                        break
                    if scene_neg[sc]:
                        chosen_neg.append(scene_neg[sc].pop(0))
                        progress = True
                if not progress:
                    break

            chosen = chosen_pos + chosen_neg
            if len(chosen) < target_k:
                remaining = [c for sc in ordered_scenes for c in (scene_pos[sc] + scene_neg[sc])]
                chosen.extend(remaining[: target_k - len(chosen)])

            fam_selected.extend(chosen)

        selected.extend(fam_selected)

    selected.sort(key=lambda x: x.sample_id)
    return selected


def compute_dataset_hash(samples: Sequence[MultimodalSample]) -> str:
    """Compute a deterministic SHA-256 hash representing the exact dataset contents."""
    hasher = hashlib.sha256()
    for s in samples:
        record_bytes = (
            f"{s.sample_id}|{s.source_sample_id}|{s.presentation_order}|"
            f"{s.view_id}|{s.family}|{s.question}|{s.answer}|{s.image_path}\n"
        ).encode("utf-8")
        hasher.update(record_bytes)
    return hasher.hexdigest()


def export_dataset_jsonl(samples: Sequence[MultimodalSample], out_path: Path) -> Path:
    """Export multimodal samples to a JSONL file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    return out_path


def build_and_export_e1_datasets(
    scenes_dir: Path = Path("outputs/scenes"),
    rendered_dir: Path = Path("outputs/experiments/g2.0-e/rendered"),
    out_dir: Path = Path("outputs/experiments/g2.0-e"),
) -> Dict[str, Any]:
    """Materialize and export the frozen primary datasets for G2.0-E1.

    Exports:
        - train_group_a.jsonl (N=1552)
        - train_group_c.jsonl (N=1552)
        - train_group_d.jsonl (N=1552)
        - holdout_s1_cardinal.jsonl (N=1136)
        - holdout_s2_jitter.jsonl (N=1136)
        - e1_dataset_manifest.json
    """
    from spatialforge.environment.scene import load_scene_state

    scenes_dir = Path(scenes_dir)
    rendered_dir = Path(rendered_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 80 train scenes and 20 holdout scenes
    train_scenes = [load_scene_state(scenes_dir / f"scene_{i:03d}.json") for i in range(80)]
    holdout_scenes = [load_scene_state(scenes_dir / f"scene_{i:03d}.json") for i in range(80, 100)]

    image_lookup = load_manifest_image_lookup(rendered_dir)

    # 1. Materialize training pools with operand-order normalization
    pool_a, stats_a = materialize_group_pool(train_scenes, "primary_a", image_lookup, normalize_presentation=True)
    pool_c, stats_c = materialize_group_pool(train_scenes, "primary_c", image_lookup, normalize_presentation=True)
    pool_d, stats_d = materialize_group_pool(train_scenes, "primary_d", image_lookup, normalize_presentation=True)

    # 2. Frozen common budget: N=1552 (the maximum common balanced budget across A, C, D)
    common_budget = len(pool_a)
    sel_a = sample_balanced_budget(pool_a, common_budget)
    sel_c = sample_balanced_budget(pool_c, common_budget)
    sel_d = sample_balanced_budget(pool_d, common_budget)

    # 3. Materialize holdout pools
    s1_pool, s1_stats = materialize_group_pool(holdout_scenes, "primary_c", image_lookup, normalize_presentation=True)
    s2_pool, s2_stats = materialize_group_pool(holdout_scenes, "holdout_s2_jitter", image_lookup, normalize_presentation=True)

    # Export JSONL files
    path_a = export_dataset_jsonl(sel_a, out_dir / "train_group_a.jsonl")
    path_c = export_dataset_jsonl(sel_c, out_dir / "train_group_c.jsonl")
    path_d = export_dataset_jsonl(sel_d, out_dir / "train_group_d.jsonl")
    path_s1 = export_dataset_jsonl(s1_pool, out_dir / "holdout_s1_cardinal.jsonl")
    path_s2 = export_dataset_jsonl(s2_pool, out_dir / "holdout_s2_jitter.jsonl")

    manifest = {
        "milestone": "G2.0-E1",
        "common_training_budget": common_budget,
        "splits": {
            "train_scenes": [f"scene_{i:03d}" for i in range(80)],
            "holdout_scenes": [f"scene_{i:03d}" for i in range(80, 100)],
            "eligible_train_scenes": sorted(list(set(s.scene_id for s in sel_a))),
        },
        "groups": {
            "group_a": {
                "description": "Fixed-view single-view control (south only)",
                "plan": "primary_a",
                "pool_eligible": stats_a["total_eligible"],
                "selected_samples": len(sel_a),
                "unique_scenes": len(set(s.scene_id for s in sel_a)),
                "sha256": compute_dataset_hash(sel_a),
                "families": dict(Counter(s.family for s in sel_a)),
                "views": dict(Counter(s.view_id for s in sel_a)),
                "answers": {fam: dict(Counter(s.answer for s in sel_a if s.family == fam)) for fam in POS_LABELS},
                "exclusions": stats_a["exclusions"],
                "file": str(path_a),
            },
            "group_c": {
                "description": "Canonical multi-view treatment (south, east, north, west)",
                "plan": "primary_c",
                "pool_eligible": stats_c["total_eligible"],
                "selected_samples": len(sel_c),
                "unique_scenes": len(set(s.scene_id for s in sel_c)),
                "sha256": compute_dataset_hash(sel_c),
                "families": dict(Counter(s.family for s in sel_c)),
                "views": dict(Counter(s.view_id for s in sel_c)),
                "answers": {fam: dict(Counter(s.answer for s in sel_c if s.family == fam)) for fam in POS_LABELS},
                "exclusions": stats_c["exclusions"],
                "file": str(path_c),
            },
            "group_d": {
                "description": "Canonical + jitter treatment (4 cardinal + 4 bounded jitter)",
                "plan": "primary_d",
                "pool_eligible": stats_d["total_eligible"],
                "selected_samples": len(sel_d),
                "unique_scenes": len(set(s.scene_id for s in sel_d)),
                "sha256": compute_dataset_hash(sel_d),
                "families": dict(Counter(s.family for s in sel_d)),
                "views": dict(Counter(s.view_id for s in sel_d)),
                "answers": {fam: dict(Counter(s.answer for s in sel_d if s.family == fam)) for fam in POS_LABELS},
                "exclusions": stats_d["exclusions"],
                "file": str(path_d),
            },
            "holdout_s1": {
                "description": "Unseen scenes (80-99), cardinal canonical viewpoints",
                "pool_eligible": len(s1_pool),
                "selected_samples": len(s1_pool),
                "unique_scenes": len(set(s.scene_id for s in s1_pool)),
                "sha256": compute_dataset_hash(s1_pool),
                "families": dict(Counter(s.family for s in s1_pool)),
                "answers": {fam: dict(Counter(s.answer for s in s1_pool if s.family == fam)) for fam in POS_LABELS},
                "file": str(path_s1),
            },
            "holdout_s2": {
                "description": "Unseen scenes (80-99), novel evaluation jitter viewpoints",
                "pool_eligible": len(s2_pool),
                "selected_samples": len(s2_pool),
                "unique_scenes": len(set(s.scene_id for s in s2_pool)),
                "sha256": compute_dataset_hash(s2_pool),
                "families": dict(Counter(s.family for s in s2_pool)),
                "answers": {fam: dict(Counter(s.answer for s in s2_pool if s.family == fam)) for fam in POS_LABELS},
                "file": str(path_s2),
            },
        },
    }

    manifest_path = out_dir / "e1_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = build_and_export_e1_datasets()
    print("[ok] G2.0-E1 datasets successfully materialized and exported.")
    print("     Common training budget N =", m["common_training_budget"])
    print("     Group A hash:", m["groups"]["group_a"]["sha256"][:16])
    print("     Group C hash:", m["groups"]["group_c"]["sha256"][:16])
    print("     Group D hash:", m["groups"]["group_d"]["sha256"][:16])

