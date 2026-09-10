"""Structured procedural world generator for SpatialForge G2.1-C P1.

STATUS: DEPRECATED AS TRAINING SOURCE (v3.1 architecture revision).
This procedural toy world (scene_challenge_*, S0-S3, cube/sphere/cylinder/torus)
is **no longer a training or product-environment source**. The formal training
mainline is the Embodied ProcTHOR Agent (see ``spatialforge.embodied`` and
docs/PROJECT_PLAN_v3.md).

This module is retained ONLY as synthetic geometry fixtures for:
- unit tests / deterministic regression
- camera math / projection verification

It must not be presented as a formal Environment in the active training/product
UI, and must not be wrapped as training data.

Pure CPU Python. No bpy import and no third-party dependencies.

Scientific Boundaries:
- Does NOT claim authoritative visibility or occlusion truth.
- Uses "projected-overlap candidates", "depth layering", and "clutter / packing summary".
- Produces synthetic, deterministic scenes under seed compatible with SceneState contracts.
- Strictly adheres to Object Identity Constraint: no duplicate exact (color, shape) combinations.
- Upgrades world structure: support surfaces, object-on-object stacking, compound objects,
  controlled orientation variation, and expanded shape vocabulary.
"""

from dataclasses import asdict, dataclass, field
import json
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from spatialforge.environment.camera import CameraPose, Vec3
from spatialforge.environment.scene import SceneObject, SceneState
from spatialforge.environment.views import generate_cardinal_views, VIEW_IDS

# Canonical vocabulary
BASE_SHAPES = ("cube", "sphere", "cylinder")
EXPANDED_SHAPES = (
    "cone",
    "torus",
    "octahedron",
    "pyramid",
    "tetrahedron",
    "prism",
    "capsule",
)
SHAPES = BASE_SHAPES + EXPANDED_SHAPES

COLORS = ("red", "blue", "green", "yellow", "purple", "cyan")
SIZES = (0.6, 0.8, 1.0)

# Allowed orientation rules for shape families
ALLOWED_ORIENTATIONS = {
    "cube": {"yaw": True, "sideways": False, "tilt": True},
    "cylinder": {"yaw": True, "sideways": True, "tilt": False},
    "cone": {"yaw": True, "sideways": True, "tilt": False},
    "sphere": {"yaw": False, "sideways": False, "tilt": False},
    "torus": {"yaw": True, "upright": True, "tilt": False},
    "octahedron": {"yaw": True, "sideways": False, "tilt": True},
    "pyramid": {"yaw": True, "sideways": False, "tilt": False},
    "tetrahedron": {"yaw": True, "sideways": False, "tilt": True},
    "prism": {"yaw": True, "sideways": True, "tilt": False},
    "capsule": {"yaw": True, "sideways": True, "tilt": False},
}

# Perceptually similar / confusable color pairings (for distractors)
SIMILAR_COLOR_PAIRS = {
    frozenset({"blue", "cyan"}),
    frozenset({"green", "cyan"}),
    frozenset({"red", "purple"}),
    frozenset({"purple", "blue"}),
    frozenset({"red", "yellow"}),
}

# Depth layer definitions along canonical viewing axis Y (world Z-up, camera south at -Y looking +Y)
# Foreground: closer to South camera
# Midground: central region
# Background: farther from South camera
DEPTH_LAYERS = {
    "foreground": (-1.8, -0.6),
    "midground": (-0.5, 0.5),
    "background": (0.6, 1.8),
}

# Standard difficulty tier specifications for structured procedural micro-worlds
TIER_SPECS = {
    "S0": {
        "tier": "S0",
        "description": "Legacy Sparse Baseline (~4 objects)",
        "min_objects": 4,
        "max_objects": 4,
        "target_depth_layers": 1,
        "clutter_factor": 0.10,
        "min_clearance": 0.35,
        "target_overlap_candidates": 0,
        "target_distractor_pairs": 0,
        "allow_size_distance_confounds": False,
        "allowed_shapes": BASE_SHAPES,
        "has_platforms": False,
        "max_stacks": 0,
        "max_compounds": 0,
        "has_orientations": False,
    },
    "S1": {
        "tier": "S1",
        "description": "Mild Structured Challenge (~6–8 objects)",
        "min_objects": 6,
        "max_objects": 8,
        "target_depth_layers": 2,
        "clutter_factor": 0.40,
        "min_clearance": 0.18,
        "target_overlap_candidates": 1,
        "target_distractor_pairs": 2,
        "allow_size_distance_confounds": False,
        "allowed_shapes": ("cube", "sphere", "cylinder", "cone", "torus", "octahedron", "pyramid"),
        "has_platforms": True,
        "max_stacks": 1,
        "max_compounds": 0,
        "has_orientations": True,
    },
    "S2": {
        "tier": "S2",
        "description": "Strong Structured Challenge (~8–10 objects)",
        "min_objects": 8,
        "max_objects": 10,
        "target_depth_layers": 3,
        "clutter_factor": 0.65,
        "min_clearance": 0.10,
        "target_overlap_candidates": 2,
        "target_distractor_pairs": 4,
        "allow_size_distance_confounds": True,
        "allowed_shapes": SHAPES,
        "has_platforms": True,
        "max_stacks": 2,
        "max_compounds": 1,
        "has_orientations": True,
    },
    "S3": {
        "tier": "S3",
        "description": "Dense Structured Challenge (~10–12 objects)",
        "min_objects": 10,
        "max_objects": 12,
        "target_depth_layers": 3,
        "clutter_factor": 0.85,
        "min_clearance": 0.08,
        "target_overlap_candidates": 3,
        "target_distractor_pairs": 6,
        "allow_size_distance_confounds": True,
        "allowed_shapes": SHAPES,
        "has_platforms": True,
        "max_stacks": 3,
        "max_compounds": 2,
        "has_orientations": True,
    },
}


@dataclass(frozen=True)
class SceneChallengeConfig:
    """Deterministic configuration for challenge scene generation."""

    tier: str = "S2"
    seed: int = 0
    object_count: Optional[int] = None
    min_clearance: Optional[float] = None
    clutter_factor: Optional[float] = None
    allow_size_distance_confounds: Optional[bool] = None
    scene_id: Optional[str] = None

    def get_tier_spec(self) -> Dict[str, Any]:
        tier_upper = self.tier.upper()
        if tier_upper not in TIER_SPECS:
            raise ValueError(f"Unknown tier: {self.tier!r}. Expected one of {list(TIER_SPECS.keys())}")
        return TIER_SPECS[tier_upper]


@dataclass(frozen=True)
class SceneChallengeMetadata:
    """Inspectable metadata summarizing scene difficulty axes and vertical structure."""

    difficulty_tier: str
    object_count: int
    depth_layer_count: int
    distractor_pair_count: int
    overlap_candidate_count: int
    confound_pair_count: int
    clutter_score: float
    depth_layers: Dict[str, int]
    distractor_pairs: List[Dict[str, Any]]
    overlap_candidates: List[Dict[str, Any]]
    confound_pairs: List[Dict[str, Any]]
    packing_summary: Dict[str, Any]
    # G2.1-C P1 Structured World additions:
    support_relation_count: int = 0
    stacked_object_count: int = 0
    elevated_object_count: int = 0
    orientation_diverse_object_count: int = 0
    compound_object_count: int = 0
    placement_modes: Dict[str, int] = field(default_factory=dict)
    shape_distribution: Dict[str, int] = field(default_factory=dict)
    support_relations: List[Dict[str, Any]] = field(default_factory=list)
    stacked_objects: List[str] = field(default_factory=list)
    elevated_objects: List[str] = field(default_factory=list)
    orientation_diverse_objects: List[str] = field(default_factory=list)
    compound_objects: List[Dict[str, Any]] = field(default_factory=list)
    scientific_disclaimer: str = (
        "Projected-overlap candidates and clutter metrics are geometric heuristics for "
        "challenge benchmarking and do NOT constitute authoritative visibility or occlusion truth."
    )

    @property
    def compound_assemblies(self) -> List[Dict[str, Any]]:
        return self.compound_objects

    def to_dict(self) -> Dict[str, Any]:
        return {
            "difficulty_tier": self.difficulty_tier,
            "object_count": self.object_count,
            "depth_layer_count": self.depth_layer_count,
            "distractor_pair_count": self.distractor_pair_count,
            "overlap_candidate_count": self.overlap_candidate_count,
            "confound_pair_count": self.confound_pair_count,
            "clutter_score": self.clutter_score,
            "support_relation_count": self.support_relation_count,
            "stacked_object_count": self.stacked_object_count,
            "elevated_object_count": self.elevated_object_count,
            "orientation_diverse_object_count": self.orientation_diverse_object_count,
            "compound_object_count": self.compound_object_count,
            "placement_modes": self.placement_modes,
            "shape_distribution": self.shape_distribution,
            "support_relations": self.support_relations,
            "stacked_objects": self.stacked_objects,
            "elevated_objects": self.elevated_objects,
            "orientation_diverse_objects": self.orientation_diverse_objects,
            "compound_objects": self.compound_objects,
            "compound_assemblies": self.compound_objects,
            "depth_layers": self.depth_layers,
            "distractor_pairs": self.distractor_pairs,
            "overlap_candidates": self.overlap_candidates,
            "confound_pairs": self.confound_pairs,
            "packing_summary": self.packing_summary,
            "scientific_disclaimer": self.scientific_disclaimer,
        }


@dataclass(frozen=True)
class ChallengeScene:
    """Result container bundling generated SceneState and challenge metadata."""

    scene_state: SceneState
    metadata: SceneChallengeMetadata

    def to_scene_dict(self) -> Dict[str, Any]:
        """Produce dictionary directly serializable to scene JSON format."""
        objects_data = []
        for obj in self.scene_state.objects:
            entry: Dict[str, Any] = {
                "name": obj.name,
                "shape": obj.shape,
                "color": obj.color,
                "location": [round(coord, 4) for coord in obj.location],
                "size": round(obj.size, 3),
            }
            rot = getattr(obj, "rotation", (0.0, 0.0, 0.0))
            if any(abs(r) > 1e-4 for r in rot):
                entry["rotation"] = [round(r, 2) for r in rot]
            role = getattr(obj, "role", "object")
            if role != "object":
                entry["role"] = role
            supp = getattr(obj, "support_parent", None)
            if supp is not None:
                entry["support_parent"] = supp
            pm = getattr(obj, "placement_mode", "on_floor")
            if pm != "on_floor":
                entry["placement_mode"] = pm
            cid = getattr(obj, "compound_id", None)
            if cid is not None:
                entry["compound_id"] = cid
            cpart = getattr(obj, "compound_part", None)
            if cpart is not None:
                entry["compound_part"] = cpart
            svar = getattr(obj, "shape_variant", None)
            if svar is not None:
                entry["shape_variant"] = svar
            objects_data.append(entry)

        return {
            "scene_id": self.scene_state.scene_id,
            "seed": self.scene_state.seed,
            "camera": [0.0, -6.0, 3.0],
            "objects": objects_data,
            "challenge_metadata": self.metadata.to_dict(),
        }


def _select_descriptors(
    count: int,
    tier: str,
    rng: random.Random,
    allowed_shapes: Sequence[str] = SHAPES,
) -> List[Tuple[str, str]]:
    """Select count unique (color, shape) tuples enforcing Object Identity Constraint.

    Guarantees:
    - Zero exact (color, shape) duplicate pairings.
    - S0 draws from BASE_SHAPES.
    - S1-S3 inject distractor pairings and expanded shapes.
    """
    shapes_pool = [s for s in allowed_shapes if s in SHAPES]
    all_combos: List[Tuple[str, str]] = [
        (c, s) for c in COLORS for s in shapes_pool
    ]
    if count > len(all_combos):
        raise ValueError(
            f"Requested count {count} exceeds total unique (color, shape) capacity {len(all_combos)}"
        )

    if tier == "S0":
        rng.shuffle(all_combos)
        return all_combos[:count]

    # For S1, S2, S3: prioritize intentional distractor pairings and expanded shapes
    selected: List[Tuple[str, str]] = []
    selected_set: Set[Tuple[str, str]] = set()

    def add_combo(c: str, s: str) -> bool:
        if (c, s) not in selected_set and len(selected) < count:
            selected.append((c, s))
            selected_set.add((c, s))
            return True
        return False

    # 1. Anchor distractor 1: Cool tone cluster (blue / cyan)
    anchor_color = rng.choice(["blue", "cyan"])
    partner_color = "cyan" if anchor_color == "blue" else "blue"
    anchor_shape = rng.choice(shapes_pool)
    other_shape = rng.choice([s for s in shapes_pool if s != anchor_shape])

    add_combo(anchor_color, anchor_shape)
    add_combo(partner_color, anchor_shape)  # Same shape, similar color
    add_combo(anchor_color, other_shape)    # Same color, different shape

    # 2. Anchor distractor 2: Green / Cyan or Red / Purple
    if len(selected) < count:
        pair_choice = rng.choice([("green", "cyan"), ("red", "purple"), ("yellow", "red")])
        c1, c2 = pair_choice
        sh1 = rng.choice(shapes_pool)
        sh2 = rng.choice([s for s in shapes_pool if s != sh1])
        add_combo(c1, sh1)
        add_combo(c2, sh1)
        add_combo(c1, sh2)

    # 3. Ensure expanded shapes are represented in S1-S3
    expanded_candidates = [s for s in shapes_pool if s in EXPANDED_SHAPES]
    if expanded_candidates:
        rng.shuffle(expanded_candidates)
        for exp_sh in expanded_candidates:
            if len(selected) >= count:
                break
            available_colors = [c for c in COLORS if (c, exp_sh) not in selected_set]
            if available_colors:
                chosen_c = rng.choice(available_colors)
                add_combo(chosen_c, exp_sh)

    # 4. Fill remaining from pool deterministically
    remaining = [combo for combo in all_combos if combo not in selected_set]
    rng.shuffle(remaining)
    for combo in remaining:
        if len(selected) >= count:
            break
        add_combo(combo[0], combo[1])

    # 5. Guarantee at least two flat-topped bases (cube, cylinder, prism) exist in S1-S3
    flat_count = sum(1 for (c, s) in selected if s in ("cube", "cylinder", "prism"))
    replace_idx = len(selected) - 1
    while flat_count < 2 and replace_idx >= 0:
        if selected[replace_idx][1] not in ("cube", "cylinder", "prism"):
            for s_pick in ("cube", "cylinder", "prism"):
                cand = (selected[replace_idx][0], s_pick)
                if cand not in selected_set:
                    selected_set.remove(selected[replace_idx])
                    selected[replace_idx] = cand
                    selected_set.add(cand)
                    flat_count += 1
                    break
        replace_idx -= 1

    return selected


def _check_collision_2d(
    x: float,
    y: float,
    radius: float,
    placed_objects: Sequence[Dict[str, Any]],
    min_clearance: float,
) -> bool:
    """Return True if placing at (x, y) with radius violates clearance with any placed object."""
    for obj in placed_objects:
        ox, oy = obj["location"][0], obj["location"][1]
        orad = obj["size"] / 2.0
        dist = math.hypot(x - ox, y - oy)
        if dist < (radius + orad + min_clearance):
            return True
    return False


def _check_collision_3d(
    x: float,
    y: float,
    z: float,
    size: float,
    placed_objects: Sequence[Dict[str, Any]],
    min_clearance: float,
    support_parent_name: Optional[str] = None,
    supported_child_names: Sequence[str] = (),
) -> bool:
    """Return True if (x, y, z, size) collides illegally with any placed object in 3D.

    - If two objects overlap in Z by more than 0.02m, they must have horizontal clearance.
    - If one object rests on top of another in a valid support relationship, direct vertical contact is permitted.
    - Objects strictly separated in Z do not collide horizontally.
    """
    z_min_self = z - size / 2.0
    z_max_self = z + size / 2.0
    r_self = size / 2.0

    for other in placed_objects:
        other_name = other["name"]
        if other_name == support_parent_name or other_name in supported_child_names:
            continue
        if other.get("support_parent") == other_name:
            continue

        ox, oy, oz = other["location"]
        osize = other["size"]
        z_min_other = oz - osize / 2.0
        z_max_other = oz + osize / 2.0
        r_other = osize / 2.0

        z_overlap = min(z_max_self, z_max_other) - max(z_min_self, z_min_other)
        if z_overlap > 0.02:
            d_xy = math.hypot(x - ox, y - oy)
            if d_xy < (r_self + r_other + min_clearance):
                return True
        else:
            d_xy = math.hypot(x - ox, y - oy)
            if d_xy < (r_self + r_other):
                if z_overlap > -1e-4:
                    return True
    return False


def generate_challenge_scene(config: SceneChallengeConfig) -> ChallengeScene:
    """Deterministically generate a structured challenge scene under seed and tier specifications."""
    spec = config.get_tier_spec()
    rng = random.Random(config.seed)

    tier = spec["tier"]
    min_clearance = (
        config.min_clearance if config.min_clearance is not None else spec["min_clearance"]
    )
    clutter_factor = (
        config.clutter_factor if config.clutter_factor is not None else spec["clutter_factor"]
    )
    allow_confounds = (
        config.allow_size_distance_confounds
        if config.allow_size_distance_confounds is not None
        else spec["allow_size_distance_confounds"]
    )

    if config.object_count is not None:
        count = config.object_count
    else:
        count = rng.randint(spec["min_objects"], spec["max_objects"])

    scene_id = config.scene_id or f"scene_challenge_{tier.lower()}_{config.seed:03d}"

    # 1. Object descriptor selection
    allowed_shapes = spec.get("allowed_shapes", BASE_SHAPES)
    descriptors = _select_descriptors(count, tier, rng, allowed_shapes)

    # 2. Depth layer quota allocation (for ground placement anchors)
    layer_names = ["foreground", "midground", "background"]
    if tier == "S0":
        quotas = {"foreground": 1, "midground": 2, "background": 1}
    elif tier == "S1":
        quotas = {"foreground": count // 2, "midground": count - (count // 2), "background": 0}
        if count >= 7:
            quotas = {"foreground": 2, "midground": 3, "background": count - 5}
    else:
        base = count // 3
        rem = count % 3
        quotas = {
            "foreground": base + (1 if rem > 0 else 0),
            "midground": base + (1 if rem > 1 else 0),
            "background": base,
        }

    assigned_layers: List[str] = []
    for l_name in layer_names:
        assigned_layers.extend([l_name] * quotas.get(l_name, 0))
    while len(assigned_layers) < count:
        assigned_layers.append("midground")
    assigned_layers = assigned_layers[:count]
    rng.shuffle(assigned_layers)

    # 3. Size assignment
    sizes: List[float] = []
    for i in range(count):
        if tier == "S0":
            sizes.append(rng.choice(SIZES))
        else:
            layer = assigned_layers[i]
            if allow_confounds and i == 0 and layer == "foreground":
                sizes.append(0.6)  # Intentional small-near
            elif allow_confounds and i == 1 and layer == "background":
                sizes.append(1.0)  # Intentional large-far
            else:
                sizes.append(rng.choice(SIZES))

    # 4. Multi-attempt structured placement under seed
    placed: List[Dict[str, Any]] = []
    max_scene_attempts = 25

    for scene_attempt in range(max_scene_attempts):
        place_rng = random.Random(config.seed * 1000 + scene_attempt)
        placed = []
        scene_success = True

        # Plan structured placement assignments
        # S0: 100% on_floor
        # S1: 1 platform block + 1 on_platform + 1 on_top_of stack
        # S2: 1 platform + 1 on_platform + 1 on_top_of stack + 1 compound entity (base + top)
        # S3: 1 platform + 2 on_platform + 1 stacked chain (3 tiers) + 1 compound entity
        placement_plan: List[Dict[str, Any]] = []
        for i in range(count):
            color, shape = descriptors[i]
            placement_plan.append({
                "index": i,
                "name": f"obj{i}_{color}_{shape}",
                "shape": shape,
                "color": color,
                "size": sizes[i],
                "layer": assigned_layers[i],
                "role": "object",
                "placement_mode": "on_floor",
                "support_parent": None,
                "compound_id": None,
                "compound_part": None,
                "shape_variant": None,
                "rotation": (0.0, 0.0, 0.0),
            })

        if tier in ("S1", "S2", "S3"):
            # A. Support surface platform
            plat_candidates = [
                item for item in placement_plan
                if item["shape"] in ("cube", "cylinder") and item["size"] >= 0.8
            ]
            platform_item = plat_candidates[0] if plat_candidates else placement_plan[0]
            platform_item["role"] = "support_surface"
            platform_item["size"] = max(platform_item["size"], 0.8)

            # B. Object on platform
            avail_elev = [
                item for item in placement_plan
                if item["index"] != platform_item["index"]
            ]
            used_parents: Set[str] = set()

            if avail_elev:
                on_plat_item = avail_elev[0]
                on_plat_item["placement_mode"] = "on_platform"
                on_plat_item["support_parent"] = platform_item["name"]
                on_plat_item["size"] = min(on_plat_item["size"], platform_item["size"])
                used_parents.add(platform_item["name"])

            # C. Stacking pair (object-on-object)
            remain_for_stack = [
                item for item in placement_plan
                if item["index"] not in (platform_item["index"], on_plat_item["index"] if avail_elev else -1)
                and item["name"] not in used_parents
            ]
            flat_bases = [
                it for it in remain_for_stack
                if it["shape"] in ("cube", "cylinder", "prism")
            ]
            if flat_bases and len(remain_for_stack) >= 2:
                stack_base = flat_bases[0]
                stack_base["size"] = max(stack_base["size"], 0.8)
                used_parents.add(stack_base["name"])
                candidates_top = [it for it in remain_for_stack if it["index"] != stack_base["index"] and it["name"] not in used_parents]
                if candidates_top:
                    stack_top = candidates_top[0]
                    stack_top["placement_mode"] = "on_top_of"
                    stack_top["support_parent"] = stack_base["name"]
                    stack_top["size"] = min(stack_top["size"], stack_base["size"])

                    # In S3, create a stacked chain (3rd tier) if items available
                    if tier == "S3" and len(candidates_top) >= 3 and stack_top["shape"] in ("cube", "cylinder"):
                        candidates_chain = [
                            it for it in candidates_top
                            if it["index"] != stack_top["index"] and it["name"] not in used_parents
                        ]
                        if candidates_chain:
                            chain_top = candidates_chain[0]
                            chain_top["placement_mode"] = "stacked_chain"
                            chain_top["support_parent"] = stack_top["name"]
                            chain_top["size"] = min(chain_top["size"], stack_top["size"])
                            used_parents.add(stack_top["name"])

            # D. Compound Object (S2, S3)
            if tier in ("S2", "S3"):
                unassigned = [
                    it for it in placement_plan
                    if it["support_parent"] is None and it["role"] == "object" and it["name"] not in used_parents
                ]
                if len(unassigned) >= 2:
                    cmp_base = unassigned[0]
                    cmp_top = unassigned[1]
                    cmp_base["role"] = "compound_base"
                    cmp_base["compound_id"] = "compound_0"
                    cmp_base["compound_part"] = "base"
                    cmp_top["role"] = "compound_part"
                    cmp_top["compound_id"] = "compound_0"
                    cmp_top["compound_part"] = "top"
                    cmp_top["placement_mode"] = "compound_assembly"
                    cmp_top["support_parent"] = cmp_base["name"]
                    cmp_top["size"] = min(cmp_top["size"], cmp_base["size"])
                    used_parents.add(cmp_base["name"])

                if tier == "S3":
                    unassigned_s3 = [
                        it for it in placement_plan
                        if it["support_parent"] is None and it["role"] == "object" and it["name"] not in used_parents
                    ]
                    if len(unassigned_s3) >= 2:
                        cmp_base2 = unassigned_s3[0]
                        cmp_top2 = unassigned_s3[1]
                        cmp_base2["role"] = "compound_base"
                        cmp_base2["compound_id"] = "compound_1"
                        cmp_base2["compound_part"] = "base"
                        cmp_top2["role"] = "compound_part"
                        cmp_top2["compound_id"] = "compound_1"
                        cmp_top2["compound_part"] = "top"
                        cmp_top2["placement_mode"] = "compound_assembly"
                        cmp_top2["support_parent"] = cmp_base2["name"]
                        cmp_top2["size"] = min(cmp_top2["size"], cmp_base2["size"])
                        used_parents.add(cmp_base2["name"])

                # Optional leaning pose assignment for an unassigned ground object
                unassigned_leaning = [
                    it for it in placement_plan
                    if it["support_parent"] is None and it["role"] == "object" and it["name"] not in used_parents
                    and it["shape"] in ("cylinder", "cone", "prism", "capsule", "cube")
                ]
                if unassigned_leaning:
                    lean_item = unassigned_leaning[-1]
                    lean_item["placement_mode"] = "leaning"

            # E. Controlled Orientation Variation
            for item in placement_plan:
                sh = item["shape"]
                rules = ALLOWED_ORIENTATIONS.get(sh, {"yaw": True, "sideways": False, "tilt": False})

                # Broad yaw for non-spheres
                if rules.get("yaw"):
                    yaw_deg = float(place_rng.choice([0, 15, 30, 45, 60, 90, 120, 135, 180, 225, 270]))
                else:
                    yaw_deg = 0.0

                pitch_deg = 0.0
                roll_deg = 0.0

                # Leaning pose: intentional tilt
                if item["placement_mode"] == "leaning":
                    pitch_deg = 20.0
                # Sideways pose (only if not supporting another object and not leaning)
                else:
                    has_dependents = any(it["support_parent"] == item["name"] for it in placement_plan)
                    if not has_dependents and rules.get("sideways") and place_rng.random() < 0.35:
                        if sh == "cylinder":
                            pitch_deg = 90.0
                        elif sh == "cone":
                            pitch_deg = 70.0
                        elif sh in ("prism", "capsule"):
                            pitch_deg = 90.0

                    # Upright ring for torus
                    if sh == "torus" and rules.get("upright") and place_rng.random() < 0.50:
                        pitch_deg = 90.0

                # Tilted twist for stacked top box
                if item["placement_mode"] in ("on_top_of", "compound_assembly") and sh == "cube":
                    yaw_deg = (yaw_deg + 30.0) % 360.0

                item["rotation"] = (pitch_deg, roll_deg, yaw_deg)

        # Place objects in dependency order: base objects first, then elevated/supported
        base_items = [it for it in placement_plan if it["support_parent"] is None]
        supported_items = [it for it in placement_plan if it["support_parent"] is not None]

        # Ensure base objects cover all depth layers in S2 and S3
        if tier in ("S2", "S3") and len(base_items) >= 3:
            req_layers = ["foreground", "midground", "background"]
            for idx, req_layer in enumerate(req_layers):
                base_items[idx]["layer"] = req_layer
            for idx in range(len(req_layers), len(base_items)):
                base_items[idx]["layer"] = place_rng.choice(req_layers)

        # 4a. Place base objects on floor
        for item in base_items:
            layer = item["layer"]
            y_min, y_max = DEPTH_LAYERS[layer]
            size = item["size"]
            z = size / 2.0  # resting on ground plane

            best_loc: Optional[Tuple[float, float, float]] = None

            for attempt in range(250):
                if place_rng.random() < clutter_factor and len(placed) > 0 and attempt < 120:
                    anchor = place_rng.choice(placed)
                    anchor_x, anchor_y = anchor["location"][0], anchor["location"][1]
                    anchor_r = anchor["size"] / 2.0
                    cluster_dist = place_rng.uniform(
                        anchor_r + size / 2.0 + min_clearance,
                        anchor_r + size / 2.0 + min_clearance + 0.35,
                    )
                    angle = place_rng.uniform(0, 2 * math.pi)
                    target_x = anchor_x + cluster_dist * math.cos(angle)
                    target_y = anchor_y + cluster_dist * math.sin(angle)
                    target_y = max(y_min, min(y_max, target_y))
                else:
                    target_x = place_rng.uniform(-1.85, 1.85)
                    target_y = place_rng.uniform(y_min, y_max)

                target_x = max(-2.0 + size / 2.0, min(2.0 - size / 2.0, target_x))
                target_y = max(-2.0 + size / 2.0, min(2.0 - size / 2.0, target_y))

                if not _check_collision_3d(target_x, target_y, z, size, placed, min_clearance):
                    best_loc = (target_x, target_y, z)
                    break

            if best_loc is None:
                # Systematic workspace search within assigned depth layer
                grid_steps = 35
                for xi in range(-grid_steps, grid_steps + 1):
                    gx = (xi / grid_steps) * (1.95 - size / 2.0)
                    for yi in range(grid_steps + 1):
                        gy = y_min + (yi / grid_steps) * (y_max - y_min)
                        gy = max(-2.0 + size / 2.0, min(2.0 - size / 2.0, gy))
                        if not _check_collision_3d(gx, gy, z, size, placed, min_clearance=0.03):
                            best_loc = (gx, gy, z)
                            break
                    if best_loc is not None:
                        break

            if best_loc is None:
                scene_success = False
                break

            item["location"] = best_loc
            placed.append(item)

        if not scene_success:
            continue

        # 4b. Place supported / elevated objects
        placed_by_name = {it["name"]: it for it in placed}
        # Group supported items by parent
        children_by_parent: Dict[str, List[Dict[str, Any]]] = {}
        for item in supported_items:
            children_by_parent.setdefault(item["support_parent"], []).append(item)

        for p_name, children in children_by_parent.items():
            parent = placed_by_name.get(p_name)
            if not parent:
                scene_success = False
                break

            px, py, pz = parent["location"]
            psize = parent["size"]
            z_top = pz + psize / 2.0

            for c_idx, item in enumerate(children):
                size = item["size"]
                z_child = z_top + size / 2.0
                best_loc = None

                # If parent has multiple children on its surface, offset them
                if len(children) > 1:
                    nom_angle = (2 * math.pi * c_idx) / len(children)
                    nom_dist = 0.20 * psize
                    base_ox = nom_dist * math.cos(nom_angle)
                    base_oy = nom_dist * math.sin(nom_angle)
                else:
                    base_ox, base_oy = 0.0, 0.0

                for attempt in range(60):
                    jitter = 0.06 * psize
                    ox = base_ox + place_rng.uniform(-jitter, jitter)
                    oy = base_oy + place_rng.uniform(-jitter, jitter)
                    cx = px + ox
                    cy = py + oy

                    # Verify 3D clearance against all other placed objects
                    if not _check_collision_3d(
                        cx, cy, z_child, size, placed, min_clearance=0.03,
                        support_parent_name=parent["name"],
                    ):
                        best_loc = (cx, cy, z_child)
                        break

                if best_loc is None:
                    # Fallback centered directly above parent
                    if not _check_collision_3d(
                        px + base_ox, py + base_oy, z_child, size, placed, min_clearance=0.01,
                        support_parent_name=parent["name"],
                    ):
                        best_loc = (px + base_ox, py + base_oy, z_child)

                if best_loc is None:
                    scene_success = False
                    break

                item["location"] = best_loc
                placed.append(item)
                placed_by_name[item["name"]] = item

        if scene_success and len(placed) == count:
            break

    if not placed or len(placed) < count:
        raise RuntimeError(f"Could not place all {count} objects collision-free in scene {scene_id}")

    # Build SceneState objects
    scene_objects = tuple(
        SceneObject(
            name=p["name"],
            shape=p["shape"],
            color=p["color"],
            location=p["location"],
            size=p["size"],
            rotation=p.get("rotation", (0.0, 0.0, 0.0)),
            role=p.get("role", "object"),
            support_parent=p.get("support_parent", None),
            placement_mode=p.get("placement_mode", "on_floor"),
            compound_id=p.get("compound_id", None),
            compound_part=p.get("compound_part", None),
            shape_variant=p.get("shape_variant", None),
        )
        for p in placed
    )
    scene_state = SceneState(
        scene_id=scene_id,
        seed=config.seed,
        objects=scene_objects,
    )

    # 5. Compute authoritative challenge metadata
    metadata = analyze_scene_challenge(scene_state, tier=tier)

    return ChallengeScene(
        scene_state=scene_state,
        metadata=metadata,
    )


def analyze_scene_challenge(
    scene_state: SceneState,
    tier: Optional[str] = None,
    evaluation_camera: Optional[CameraPose] = None,
) -> SceneChallengeMetadata:
    """Compute inspectable difficulty axes and heuristic metrics for any scene."""
    objects = scene_state.objects
    count = len(objects)

    # Infer tier if not explicitly provided
    if tier is None:
        if count <= 5:
            tier = "S0"
        elif count <= 7:
            tier = "S1"
        elif count <= 9:
            tier = "S2"
        else:
            tier = "S3"

    # 1. Depth Layering Analysis
    layer_counts = {"foreground": 0, "midground": 0, "background": 0}
    for obj in objects:
        y = obj.location[1]
        if y < -0.5:
            layer_counts["foreground"] += 1
        elif y <= 0.5:
            layer_counts["midground"] += 1
        else:
            layer_counts["background"] += 1

    depth_layer_count = sum(1 for v in layer_counts.values() if v > 0)

    # 2. Similar Distractor Analysis
    distractor_pairs: List[Dict[str, Any]] = []
    for i in range(count):
        for j in range(i + 1, count):
            oi, oj = objects[i], objects[j]
            pair_type = None

            if oi.color == oj.color and oi.shape != oj.shape:
                pair_type = "same_color_different_shape"
            elif oi.shape == oj.shape and frozenset({oi.color, oj.color}) in SIMILAR_COLOR_PAIRS:
                pair_type = "similar_color_same_shape"
            elif oi.shape == oj.shape and oi.color != oj.color:
                pair_type = "same_shape_different_color"

            if pair_type:
                distractor_pairs.append({
                    "object_index_a": i,
                    "object_index_b": j,
                    "name_a": oi.name,
                    "name_b": oj.name,
                    "color_a": oi.color,
                    "color_b": oj.color,
                    "shape_a": oi.shape,
                    "shape_b": oj.shape,
                    "distractor_type": pair_type,
                })

    # 3. Projected-Overlap Candidate Heuristic
    # Evaluated against standard canonical South camera (0, -6, 3) and cardinal directions
    if evaluation_camera is not None:
        cams = [("eval_cam", evaluation_camera)]
    else:
        cardinals = generate_cardinal_views()
        cams = list(zip(VIEW_IDS, cardinals))

    overlap_candidates: List[Dict[str, Any]] = []
    seen_overlap_pairs: Set[Tuple[int, int]] = set()

    for view_id, cam in cams:
        c_pos = cam.position
        for i in range(count):
            for j in range(i + 1, count):
                pair_key = (i, j)
                if pair_key in seen_overlap_pairs:
                    continue

                oi, oj = objects[i], objects[j]
                pi, pj = oi.location, oj.location

                # Vectors from camera to objects
                vi = (pi[0] - c_pos[0], pi[1] - c_pos[1], pi[2] - c_pos[2])
                vj = (pj[0] - c_pos[0], pj[1] - c_pos[1], pj[2] - c_pos[2])
                di = math.sqrt(vi[0] ** 2 + vi[1] ** 2 + vi[2] ** 2)
                dj = math.sqrt(vj[0] ** 2 + vj[1] ** 2 + vj[2] ** 2)

                if di < 1e-4 or dj < 1e-4:
                    continue

                dot = (vi[0] * vj[0] + vi[1] * vj[1] + vi[2] * vj[2]) / (di * dj)
                dot = max(-1.0, min(1.0, dot))
                theta_deg = math.degrees(math.acos(dot))

                # Angular radii
                ri = oi.size / 2.0
                rj = oj.size / 2.0
                alpha_i_deg = math.degrees(math.atan2(ri, di))
                alpha_j_deg = math.degrees(math.atan2(rj, dj))
                total_angular_span = alpha_i_deg + alpha_j_deg

                # Overlap candidate condition: angular separation is less than 1.25x their combined angular radii
                # and there is a meaningful depth difference (at least 0.45m)
                depth_diff = abs(dj - di)
                if theta_deg < 1.25 * total_angular_span and depth_diff >= 0.45:
                    near_obj = oi.name if di < dj else oj.name
                    far_obj = oj.name if di < dj else oi.name
                    overlap_ratio = round(max(0.0, 1.0 - (theta_deg / max(1e-4, total_angular_span))), 3)

                    overlap_candidates.append({
                        "object_index_a": i,
                        "object_index_b": j,
                        "pair": [oi.name, oj.name],
                        "view_id": view_id,
                        "near_object": near_obj,
                        "far_object": far_obj,
                        "angular_separation_deg": round(theta_deg, 2),
                        "angular_span_deg": round(total_angular_span, 2),
                        "depth_difference_m": round(depth_diff, 2),
                        "projected_overlap_ratio": overlap_ratio,
                    })
                    seen_overlap_pairs.add(pair_key)

    # 4. Size-Distance Confounds Analysis
    confound_pairs: List[Dict[str, Any]] = []
    # Use South camera for canonical size-distance confound evaluation
    south_cam = CameraPose(position=(0.0, -6.0, 3.0), look_at=(0.0, 0.0, 0.4), up=(0.0, 0.0, 1.0), fov_deg=60.0)
    sc_pos = south_cam.position

    for i in range(count):
        for j in range(i + 1, count):
            oi, oj = objects[i], objects[j]
            size_diff = abs(oi.size - oj.size)
            if size_diff < 0.2:
                continue

            # Identify smaller vs larger object
            if oi.size < oj.size:
                small_obj, large_obj = oi, oj
            else:
                small_obj, large_obj = oj, oi

            # Distances to camera
            d_small = math.hypot(
                math.hypot(small_obj.location[0] - sc_pos[0], small_obj.location[1] - sc_pos[1]),
                small_obj.location[2] - sc_pos[2],
            )
            d_large = math.hypot(
                math.hypot(large_obj.location[0] - sc_pos[0], large_obj.location[1] - sc_pos[1]),
                large_obj.location[2] - sc_pos[2],
            )

            # Confound condition: physically smaller object is closer to camera than physically larger object,
            # resulting in similar apparent angular sizes (projected size ratio within 0.75 - 1.33)
            if d_small < d_large:
                apparent_ratio = (small_obj.size / d_small) / (large_obj.size / d_large)
                if 0.75 <= apparent_ratio <= 1.33:
                    confound_pairs.append({
                        "small_object": small_obj.name,
                        "large_object": large_obj.name,
                        "small_physical_size": round(small_obj.size, 3),
                        "large_physical_size": round(large_obj.size, 3),
                        "near_distance_m": round(d_small, 2),
                        "far_distance_m": round(d_large, 2),
                        "projected_size_ratio": round(apparent_ratio, 3),
                    })

    # 5. Structured World Metrics (G2.1-C P1 additions)
    obj_by_name = {o.name: o for o in objects}
    support_relations: List[Dict[str, Any]] = []
    stacked_objects: List[str] = []
    elevated_objects: List[str] = []
    orientation_diverse_objects: List[str] = []
    placement_modes: Dict[str, int] = {}
    shape_distribution: Dict[str, int] = {}

    for obj in objects:
        # Shape distribution
        sh = obj.shape
        shape_distribution[sh] = shape_distribution.get(sh, 0) + 1

        # Placement mode distribution
        pm = getattr(obj, "placement_mode", "on_floor")
        placement_modes[pm] = placement_modes.get(pm, 0) + 1

        # Elevated object condition (bottom of object is elevated above table surface z > 0.05m)
        bottom_z = obj.location[2] - obj.size / 2.0
        if bottom_z > 0.05:
            elevated_objects.append(obj.name)

        # Stacked objects
        if pm in ("on_top_of", "stacked_chain", "compound_assembly"):
            stacked_objects.append(obj.name)

        # Support parent relation
        supp_name = getattr(obj, "support_parent", None)
        if supp_name and supp_name in obj_by_name:
            parent_obj = obj_by_name[supp_name]
            contact_z = parent_obj.location[2] + parent_obj.size / 2.0
            elevation_m = max(0.0, obj.location[2] - obj.size / 2.0)
            support_relations.append({
                "supported_object": obj.name,
                "supporting_object": supp_name,
                "child_name": obj.name,
                "parent_name": supp_name,
                "support_type": pm,
                "relation_type": pm,
                "contact_z": round(contact_z, 3),
                "contact_z_m": round(contact_z, 3),
                "elevation_m": round(elevation_m, 3),
            })
        elif bottom_z > 0.05 and not supp_name:
            # Fallback heuristic for unannotated scenes: detect if resting on an object below
            for other in objects:
                if other.name == obj.name:
                    continue
                other_top = other.location[2] + other.size / 2.0
                if abs(bottom_z - other_top) < 0.06:
                    d_xy = math.hypot(obj.location[0] - other.location[0], obj.location[1] - other.location[1])
                    if d_xy <= (other.size / 2.0 + 0.05):
                        support_relations.append({
                            "supported_object": obj.name,
                            "supporting_object": other.name,
                            "child_name": obj.name,
                            "parent_name": other.name,
                            "support_type": "inferred_support",
                            "relation_type": "inferred_support",
                            "contact_z": round(other_top, 3),
                            "contact_z_m": round(other_top, 3),
                            "elevation_m": round(bottom_z, 3),
                        })
                        stacked_objects.append(obj.name)
                        break

        # Orientation diversity
        rot = getattr(obj, "rotation", (0.0, 0.0, 0.0))
        if any(abs(r) > 1e-2 for r in rot):
            orientation_diverse_objects.append(obj.name)

    # Compound objects grouping
    compound_groups: Dict[str, List[SceneObject]] = {}
    for obj in objects:
        cid = getattr(obj, "compound_id", None)
        if cid:
            compound_groups.setdefault(cid, []).append(obj)

    compound_objects: List[Dict[str, Any]] = []
    for cid, parts in compound_groups.items():
        base = next((p for p in parts if getattr(p, "compound_part", None) == "base"), parts[0])
        top = next((p for p in parts if getattr(p, "compound_part", None) != "base"), parts[-1])
        c_type = f"{base.shape}_{top.shape}"
        compound_objects.append({
            "compound_id": cid,
            "compound_type": c_type,
            "assembly_type": c_type,
            "parts": [
                {"name": p.name, "part": getattr(p, "compound_part", "part")}
                for p in parts
            ],
            "part_names": [p.name for p in parts],
            "base_object": base.name,
            "top_object": top.name,
        })

    # 6. Clutter & Packing Summary
    if count >= 2:
        min_clearance_found = float("inf")
        nearest_neighbor_dists: List[float] = []

        for i in range(count):
            closest_nn = float("inf")
            for j in range(count):
                if i == j:
                    continue
                d_3d = math.sqrt(
                    (objects[i].location[0] - objects[j].location[0]) ** 2 +
                    (objects[i].location[1] - objects[j].location[1]) ** 2 +
                    (objects[i].location[2] - objects[j].location[2]) ** 2
                )
                clearance = d_3d - (objects[i].size / 2.0 + objects[j].size / 2.0)
                if clearance < min_clearance_found:
                    min_clearance_found = clearance
                if d_3d < closest_nn:
                    closest_nn = d_3d
            nearest_neighbor_dists.append(closest_nn)

        mean_nn_dist = sum(nearest_neighbor_dists) / float(count)
        min_clearance_val = max(0.0, min_clearance_found)
    else:
        min_clearance_val = 1.0
        mean_nn_dist = 2.0

    # Footprint area and density
    footprint_area = sum(math.pi * ((obj.size / 2.0) ** 2) for obj in objects)
    xs = [obj.location[0] for obj in objects]
    ys = [obj.location[1] for obj in objects]
    bbox_w = (max(xs) - min(xs) + 0.8) if xs else 2.0
    bbox_h = (max(ys) - min(ys) + 0.8) if ys else 2.0
    bbox_area = bbox_w * bbox_h
    packing_density = footprint_area / max(1.0, bbox_area)

    # Clutter score: calibrated continuous heuristic in [0.0, 1.0]
    base_count_score = min(0.50, max(0.0, (count - 4) * 0.055))
    proximity_score = min(0.30, max(0.0, (1.8 - min(1.8, mean_nn_dist)) / 1.8 * 0.30))
    structure_bonus = min(0.20, (len(support_relations) * 0.04 + len(elevated_objects) * 0.03))
    raw_clutter = 0.15 + base_count_score + proximity_score + structure_bonus
    clutter_score = round(max(0.0, min(1.0, raw_clutter)), 3)

    packing_summary = {
        "min_surface_clearance_m": round(min_clearance_val, 3),
        "mean_nearest_neighbor_dist_m": round(mean_nn_dist, 3),
        "footprint_area_m2": round(footprint_area, 3),
        "bounding_area_m2": round(bbox_area, 3),
        "packing_density": round(packing_density, 3),
    }

    return SceneChallengeMetadata(
        difficulty_tier=tier,
        object_count=count,
        depth_layer_count=depth_layer_count,
        distractor_pair_count=len(distractor_pairs),
        overlap_candidate_count=len(overlap_candidates),
        confound_pair_count=len(confound_pairs),
        clutter_score=clutter_score,
        depth_layers=layer_counts,
        distractor_pairs=distractor_pairs,
        overlap_candidates=overlap_candidates,
        confound_pairs=confound_pairs,
        packing_summary=packing_summary,
        support_relation_count=len(support_relations),
        stacked_object_count=len(stacked_objects),
        elevated_object_count=len(elevated_objects),
        orientation_diverse_object_count=len(orientation_diverse_objects),
        compound_object_count=len(compound_objects),
        placement_modes=placement_modes,
        shape_distribution=shape_distribution,
        support_relations=support_relations,
        stacked_objects=stacked_objects,
        elevated_objects=elevated_objects,
        orientation_diverse_objects=orientation_diverse_objects,
        compound_objects=compound_objects,
    )
