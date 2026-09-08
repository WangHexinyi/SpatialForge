"""Targeted tests for SpatialForge G2.1-C P0 Scene Challenge Prototype.

Pure CPU tests. Verifies:
- Deterministic generation under seed
- Tier-dependent object-count ranges (S0-S3)
- Strict Object Identity Constraint: no duplicate exact color+shape combinations
- Valid object geometries, ground placement, and 3D collision clearance
- Increased depth-layer diversity in harder tiers
- Higher clutter / packing relative to sparse baseline
- Presence of similar distractor candidates
- Presence of projected-overlap candidates metadata
- Presence of size-distance confounds
- JSON serializability and SceneState contract compatibility
"""

import json
import math
from pathlib import Path
import tempfile
import unittest

from spatialforge.environment.challenge import (
    BASE_SHAPES,
    COLORS,
    DEPTH_LAYERS,
    EXPANDED_SHAPES,
    SHAPES,
    SIMILAR_COLOR_PAIRS,
    SIZES,
    TIER_SPECS,
    ChallengeScene,
    SceneChallengeConfig,
    SceneChallengeMetadata,
    analyze_scene_challenge,
    generate_challenge_scene,
)
from spatialforge.environment.scene import SceneObject, SceneState, load_scene_state


class TestSceneChallenge(unittest.TestCase):
    def test_deterministic_generation_under_seed(self):
        """Identical (tier, seed) produces 100% bit-exact SceneState and metadata."""
        for tier in ("S0", "S1", "S2", "S3"):
            for seed in (0, 42, 123):
                cfg1 = SceneChallengeConfig(tier=tier, seed=seed)
                cfg2 = SceneChallengeConfig(tier=tier, seed=seed)
                ch1 = generate_challenge_scene(cfg1)
                ch2 = generate_challenge_scene(cfg2)

                self.assertEqual(ch1.scene_state, ch2.scene_state)
                self.assertEqual(ch1.metadata, ch2.metadata)
                self.assertEqual(ch1.to_scene_dict(), ch2.to_scene_dict())

    def test_tier_dependent_object_count_ranges(self):
        """Generated object counts strictly obey prototype tier envelopes."""
        tier_bounds = {
            "S0": (4, 4),
            "S1": (6, 8),
            "S2": (8, 10),
            "S3": (10, 12),
        }
        for tier, (min_c, max_c) in tier_bounds.items():
            for seed in range(20):
                cfg = SceneChallengeConfig(tier=tier, seed=seed)
                ch = generate_challenge_scene(cfg)
                count = len(ch.scene_state.objects)
                self.assertGreaterEqual(
                    count,
                    min_c,
                    f"Tier {tier} seed {seed} count {count} < min {min_c}",
                )
                self.assertLessEqual(
                    count,
                    max_c,
                    f"Tier {tier} seed {seed} count {count} > max {max_c}",
                )
                self.assertEqual(ch.metadata.object_count, count)

        # Explicit count override
        cfg_custom = SceneChallengeConfig(tier="S2", seed=99, object_count=9)
        ch_custom = generate_challenge_scene(cfg_custom)
        self.assertEqual(len(ch_custom.scene_state.objects), 9)

    def test_no_duplicate_exact_color_shape_combinations(self):
        """CRITICAL: Zero duplicate exact (color, shape) pairs in any scene."""
        for tier in ("S0", "S1", "S2", "S3"):
            for seed in range(30):
                cfg = SceneChallengeConfig(tier=tier, seed=seed)
                ch = generate_challenge_scene(cfg)
                combos = [(o.color, o.shape) for o in ch.scene_state.objects]
                unique_combos = set(combos)
                self.assertEqual(
                    len(combos),
                    len(unique_combos),
                    f"Duplicate descriptor in {tier} seed {seed}: {combos}",
                )

    def test_valid_object_identities_and_clearance(self):
        """Objects have valid shapes, colors, valid elevations, and zero 3D collision penetrations."""
        for tier in ("S0", "S1", "S2", "S3"):
            for seed in range(25):
                cfg = SceneChallengeConfig(tier=tier, seed=seed)
                ch = generate_challenge_scene(cfg)
                objs = ch.scene_state.objects

                for i, obj in enumerate(objs):
                    self.assertIn(obj.shape, SHAPES)
                    self.assertIn(obj.color, COLORS)
                    self.assertIn(obj.size, SIZES)
                    if tier == "S0" or obj.support_parent is None:
                        # Ground contact: z coordinate equals radius = size / 2
                        self.assertAlmostEqual(obj.location[2], obj.size / 2.0, places=3)
                    else:
                        # Elevated object: bottom of object >= ground level
                        self.assertGreaterEqual(obj.location[2] - obj.size / 2.0, -1e-4)

                    self.assertTrue(obj.name.startswith("obj"))

                    # 3D collision clearance against all other objects
                    for j in range(i + 1, len(objs)):
                        other = objs[j]
                        d_xy = math.hypot(
                            obj.location[0] - other.location[0],
                            obj.location[1] - other.location[1],
                        )
                        min_req_xy = (obj.size + other.size) / 2.0
                        # If horizontally overlapping, must be vertically separated
                        if d_xy < min_req_xy - 1e-4:
                            min_req_z = (obj.size + other.size) / 2.0
                            dz = abs(obj.location[2] - other.location[2])
                            self.assertGreaterEqual(
                                dz,
                                min_req_z - 1e-4,
                                f"3D collision penetration in {tier} seed {seed} between {obj.name} and {other.name}: d_xy={d_xy} < {min_req_xy}, dz={dz} < {min_req_z}",
                            )

    def test_increased_depth_layer_diversity_in_harder_tiers(self):
        """S2 and S3 distribute objects across all 3 depth layers (foreground, midground, background)."""
        for tier in ("S2", "S3"):
            for seed in range(15):
                cfg = SceneChallengeConfig(tier=tier, seed=seed)
                ch = generate_challenge_scene(cfg)
                meta = ch.metadata

                self.assertEqual(
                    meta.depth_layer_count,
                    3,
                    f"{tier} seed {seed} should populate all 3 depth layers: {meta.depth_layers}",
                )
                self.assertGreater(meta.depth_layers["foreground"], 0)
                self.assertGreater(meta.depth_layers["midground"], 0)
                self.assertGreater(meta.depth_layers["background"], 0)

    def test_higher_clutter_packing_relative_to_sparse_baseline(self):
        """S2 and S3 exhibit significantly higher clutter score and packing density than S0."""
        s0_clutter = []
        s3_clutter = []
        s0_density = []
        s3_density = []

        for seed in range(15):
            ch_s0 = generate_challenge_scene(SceneChallengeConfig(tier="S0", seed=seed))
            ch_s3 = generate_challenge_scene(SceneChallengeConfig(tier="S3", seed=seed))

            s0_clutter.append(ch_s0.metadata.clutter_score)
            s3_clutter.append(ch_s3.metadata.clutter_score)
            s0_density.append(ch_s0.metadata.packing_summary["packing_density"])
            s3_density.append(ch_s3.metadata.packing_summary["packing_density"])

        mean_s0_clutter = sum(s0_clutter) / len(s0_clutter)
        mean_s3_clutter = sum(s3_clutter) / len(s3_clutter)
        mean_s0_density = sum(s0_density) / len(s0_density)
        mean_s3_density = sum(s3_density) / len(s3_density)

        self.assertGreater(mean_s3_clutter, mean_s0_clutter + 0.35)
        self.assertGreater(mean_s3_density, mean_s0_density)

    def test_presence_of_distractor_candidates(self):
        """S1, S2, S3 contain multiple distractor pairs and scale with difficulty."""
        ch_s0 = generate_challenge_scene(SceneChallengeConfig(tier="S0", seed=42))
        ch_s2 = generate_challenge_scene(SceneChallengeConfig(tier="S2", seed=42))
        ch_s3 = generate_challenge_scene(SceneChallengeConfig(tier="S3", seed=42))

        self.assertGreaterEqual(ch_s2.metadata.distractor_pair_count, 4)
        self.assertGreater(ch_s2.metadata.distractor_pair_count, ch_s0.metadata.distractor_pair_count)
        self.assertGreater(ch_s3.metadata.distractor_pair_count, ch_s2.metadata.distractor_pair_count)

        # Verify distractor records structure
        for d in ch_s2.metadata.distractor_pairs:
            self.assertIn("object_index_a", d)
            self.assertIn("object_index_b", d)
            self.assertIn("name_a", d)
            self.assertIn("name_b", d)
            self.assertIn("distractor_type", d)
            self.assertIn(
                d["distractor_type"],
                (
                    "same_color_different_shape",
                    "similar_color_same_shape",
                    "same_shape_different_color",
                ),
            )

    def test_presence_of_overlap_candidates_metadata(self):
        """Harder tiers produce projected-overlap candidates without claiming occlusion truth."""
        ch_s2 = generate_challenge_scene(SceneChallengeConfig(tier="S2", seed=42))
        self.assertGreater(ch_s2.metadata.overlap_candidate_count, 0)
        self.assertIn("scientific_disclaimer", ch_s2.metadata.to_dict())
        self.assertIn("NOT constitute authoritative", ch_s2.metadata.scientific_disclaimer)

        for cand in ch_s2.metadata.overlap_candidates:
            self.assertIn("near_object", cand)
            self.assertIn("far_object", cand)
            self.assertIn("angular_separation_deg", cand)
            self.assertIn("depth_difference_m", cand)
            self.assertIn("projected_overlap_ratio", cand)
            self.assertGreater(cand["depth_difference_m"], 0.4)

    def test_size_distance_confounds_optional_support(self):
        """When enabled, size-distance confounds are detected and recorded with projected ratios."""
        ch = generate_challenge_scene(
            SceneChallengeConfig(tier="S2", seed=42, allow_size_distance_confounds=True)
        )
        self.assertGreater(ch.metadata.confound_pair_count, 0)
        confound = ch.metadata.confound_pairs[0]
        self.assertIn("small_object", confound)
        self.assertIn("large_object", confound)
        self.assertLess(confound["small_physical_size"], confound["large_physical_size"])
        self.assertLess(confound["near_distance_m"], confound["far_distance_m"])
        self.assertGreaterEqual(confound["projected_size_ratio"], 0.75)
        self.assertLessEqual(confound["projected_size_ratio"], 1.33)

    def test_json_serializable_metadata_and_scene_compatibility(self):
        """to_scene_dict produces clean JSON that can be re-loaded by standard load_scene_state."""
        for tier in ("S0", "S1", "S2", "S3"):
            ch = generate_challenge_scene(SceneChallengeConfig(tier=tier, seed=123))
            scene_dict = ch.to_scene_dict()

            with tempfile.TemporaryDirectory() as tmpdir:
                p = Path(tmpdir) / f"{ch.scene_state.scene_id}.json"
                p.write_text(json.dumps(scene_dict, indent=2), encoding="utf-8")

                # Verify load_scene_state reads it without error
                loaded_state = load_scene_state(p)
                self.assertEqual(loaded_state.scene_id, ch.scene_state.scene_id)
                self.assertEqual(len(loaded_state.objects), len(ch.scene_state.objects))
                self.assertEqual(loaded_state.seed, ch.scene_state.seed)

                for orig_obj, loaded_obj in zip(ch.scene_state.objects, loaded_state.objects):
                    self.assertEqual(orig_obj.name, loaded_obj.name)
                    self.assertEqual(orig_obj.shape, loaded_obj.shape)
                    self.assertEqual(orig_obj.color, loaded_obj.color)
                    self.assertAlmostEqual(orig_obj.size, loaded_obj.size, places=3)
                    for c1, c2 in zip(orig_obj.location, loaded_obj.location):
                        self.assertAlmostEqual(c1, c2, places=3)

    def test_analyze_legacy_scene(self):
        """analyze_scene_challenge analyzes legacy scene_000 without crashing."""
        legacy_path = Path("outputs/scenes/scene_000.json")
        if legacy_path.exists():
            sc = load_scene_state(legacy_path)
            meta = analyze_scene_challenge(sc, tier="S0")
            self.assertEqual(meta.object_count, 4)
            self.assertEqual(meta.difficulty_tier, "S0")
            self.assertIn("clutter_score", meta.to_dict())
            self.assertIn("packing_summary", meta.to_dict())

    def test_p1_expanded_shape_vocabulary(self):
        """S0 uses only base shapes; S1-S3 exhibit expanded shapes."""
        # S0: only cube, sphere, cylinder
        for seed in range(15):
            ch_s0 = generate_challenge_scene(SceneChallengeConfig(tier="S0", seed=seed))
            for obj in ch_s0.scene_state.objects:
                self.assertIn(obj.shape, BASE_SHAPES)

        # S1-S3: expanded shapes appear
        expanded_seen = set()
        for tier in ("S1", "S2", "S3"):
            for seed in range(25):
                ch = generate_challenge_scene(SceneChallengeConfig(tier=tier, seed=seed))
                for obj in ch.scene_state.objects:
                    if obj.shape in EXPANDED_SHAPES:
                        expanded_seen.add(obj.shape)

        for es in EXPANDED_SHAPES:
            self.assertIn(
                es,
                expanded_seen,
                f"Expanded shape {es} was not generated across S1-S3 suites",
            )

    def test_p1_support_surfaces_and_stacking(self):
        """S1-S3 contain elevated objects, support relations, and stacking."""
        ch_s0 = generate_challenge_scene(SceneChallengeConfig(tier="S0", seed=0))
        self.assertEqual(ch_s0.metadata.support_relation_count, 0)
        self.assertEqual(ch_s0.metadata.elevated_object_count, 0)

        for tier in ("S1", "S2", "S3"):
            for seed in range(15):
                ch = generate_challenge_scene(SceneChallengeConfig(tier=tier, seed=seed))
                meta = ch.metadata
                self.assertGreaterEqual(meta.support_relation_count, 1)
                self.assertGreaterEqual(meta.elevated_object_count, 1)
                if tier in ("S2", "S3"):
                    self.assertGreaterEqual(meta.stacked_object_count, 1)

                # Check support relation record integrity
                obj_map = {o.name: o for o in ch.scene_state.objects}
                for rel in meta.support_relations:
                    self.assertIn(rel["parent_name"], obj_map)
                    self.assertIn(rel["child_name"], obj_map)
                    parent = obj_map[rel["parent_name"]]
                    child = obj_map[rel["child_name"]]
                    self.assertEqual(child.support_parent, parent.name)
                    # Child center Z must be strictly above parent center Z
                    self.assertGreater(child.location[2], parent.location[2])

    def test_p1_controlled_orientation_variation(self):
        """S0 is upright only; S1-S3 exhibit orientation diversity with valid Euler angles."""
        ch_s0 = generate_challenge_scene(SceneChallengeConfig(tier="S0", seed=0))
        for obj in ch_s0.scene_state.objects:
            # pitch and roll must be 0 in S0
            self.assertAlmostEqual(obj.rotation[0], 0.0)
            self.assertAlmostEqual(obj.rotation[1], 0.0)

        diverse_counts = []
        for tier in ("S1", "S2", "S3"):
            for seed in range(15):
                ch = generate_challenge_scene(SceneChallengeConfig(tier=tier, seed=seed))
                diverse_counts.append(ch.metadata.orientation_diverse_object_count)
                for obj in ch.scene_state.objects:
                    rot = obj.rotation
                    self.assertEqual(len(rot), 3)
                    # All angles within 0 to 360 degrees
                    for angle in rot:
                        self.assertGreaterEqual(angle, 0.0)
                        self.assertLessEqual(angle, 360.0)

        self.assertGreater(sum(diverse_counts), 0)

    def test_p1_compound_assemblies(self):
        """S2 and S3 generate compound assemblies with inspectable metadata."""
        for tier in ("S2", "S3"):
            found_assembly = False
            for seed in range(15):
                ch = generate_challenge_scene(SceneChallengeConfig(tier=tier, seed=seed))
                if ch.metadata.compound_object_count > 0:
                    found_assembly = True
                    for asm in ch.metadata.compound_assemblies:
                        self.assertTrue(asm["compound_id"].startswith("compound_"))
                        self.assertGreaterEqual(len(asm["parts"]), 2)
                        part_names = [p["name"] for p in asm["parts"]]
                        obj_map = {o.name: o for o in ch.scene_state.objects}
                        for pn in part_names:
                            self.assertIn(pn, obj_map)
                            o = obj_map[pn]
                            self.assertEqual(o.compound_id, asm["compound_id"])
                            self.assertIn(o.compound_part, ("base", "top", "part"))
            self.assertTrue(found_assembly, f"No compound assembly found in {tier}")

    def test_p1_enriched_challenge_metadata_schema(self):
        """Metadata includes all new P1 structural metrics in dictionary export."""
        ch = generate_challenge_scene(SceneChallengeConfig(tier="S2", seed=42))
        d = ch.metadata.to_dict()
        required_p1_keys = [
            "support_relation_count",
            "stacked_object_count",
            "elevated_object_count",
            "orientation_diverse_object_count",
            "compound_object_count",
            "placement_modes",
            "shape_distribution",
            "support_relations",
            "compound_assemblies",
        ]
        for k in required_p1_keys:
            self.assertIn(k, d, f"Missing key {k} in challenge metadata dict")

    def test_p1_scene_json_roundtrip_all_attributes(self):
        """to_scene_dict exports and load_scene_state restores all P1 object attributes."""
        ch = generate_challenge_scene(SceneChallengeConfig(tier="S3", seed=7))
        scene_dict = ch.to_scene_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / f"{ch.scene_state.scene_id}.json"
            p.write_text(json.dumps(scene_dict, indent=2), encoding="utf-8")

            loaded = load_scene_state(p)
            for orig, rec in zip(ch.scene_state.objects, loaded.objects):
                self.assertEqual(orig.name, rec.name)
                self.assertEqual(orig.shape, rec.shape)
                self.assertEqual(orig.color, rec.color)
                self.assertEqual(orig.size, rec.size)
                self.assertEqual(orig.rotation, rec.rotation)
                self.assertEqual(orig.role, rec.role)
                self.assertEqual(orig.support_parent, rec.support_parent)
                self.assertEqual(orig.placement_mode, rec.placement_mode)
                self.assertEqual(orig.compound_id, rec.compound_id)
                self.assertEqual(orig.compound_part, rec.compound_part)


if __name__ == "__main__":
    unittest.main()
