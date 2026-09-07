"""Unit tests for G2.0-E3.3A5 training execution profiles and capability verification."""

import unittest
from unittest.mock import patch

from spatialforge.experiment.training import (
    DEFAULT_TRAINING_PROFILE,
    EXECUTION_PROFILES,
    FormalTrainingConfig,
    ProfileCapabilityError,
    REFERENCE_PROFILE,
    BALANCED_PROFILE,
    MAX_PERFORMANCE_PROFILE,
    TrainingProfile,
    compute_formal_optimizer_steps,
    get_profile,
    list_profiles,
    validate_profile_capability,
)


class TestTrainingProfiles(unittest.TestCase):
    """Test suite for first-class execution profiles, invariants, and preflight guards."""

    def test_default_profile_is_max_performance(self):
        """Development default must be max_performance for maximum resource efficiency."""
        self.assertEqual(DEFAULT_TRAINING_PROFILE, "max_performance")
        default_prof = get_profile()
        self.assertEqual(default_prof.profile_id, "max_performance")
        self.assertEqual(default_prof.microbatch_size, 4)
        self.assertEqual(default_prof.gradient_accumulation_steps, 2)

    def test_all_three_first_class_profiles_exist(self):
        """All three required first-class profiles must be defined and accessible."""
        profiles = list_profiles()
        profile_ids = {p["profile_id"] for p in profiles}
        self.assertIn("reference", profile_ids)
        self.assertIn("balanced", profile_ids)
        self.assertIn("max_performance", profile_ids)
        self.assertEqual(len(profiles), 3)

    def test_effective_batch_size_invariants(self):
        """All profiles must enforce effective_batch_size == 8 and MB * ACC == 8."""
        for prof_id in ("reference", "balanced", "max_performance"):
            prof = get_profile(prof_id)
            self.assertEqual(prof.effective_batch_size, 8)
            self.assertEqual(
                prof.microbatch_size * prof.gradient_accumulation_steps,
                8,
                f"Profile {prof_id} violates MB * ACC == 8 invariant",
            )
            # Vision cache must be enabled for all production execution profiles
            self.assertTrue(prof.vision_cache_enabled)
            # Gradient checkpointing must be False for cached-vision speed
            self.assertFalse(prof.gradient_checkpointing)

    def test_profile_lookup_and_aliases(self):
        """Profile lookup supports canonical names, aliases, and instances."""
        ref = get_profile("reference")
        self.assertIs(ref, REFERENCE_PROFILE)
        self.assertIs(get_profile("p5"), REFERENCE_PROFILE)
        self.assertIs(get_profile("P5"), REFERENCE_PROFILE)

        bal = get_profile("balanced")
        self.assertIs(bal, BALANCED_PROFILE)
        self.assertIs(get_profile("p6"), BALANCED_PROFILE)
        self.assertIs(get_profile("P6"), BALANCED_PROFILE)

        max_perf = get_profile("max_performance")
        self.assertIs(max_perf, MAX_PERFORMANCE_PROFILE)
        self.assertIs(get_profile("p7"), MAX_PERFORMANCE_PROFILE)
        self.assertIs(get_profile("P7"), MAX_PERFORMANCE_PROFILE)

        # Lookup by instance returns itself
        self.assertIs(get_profile(max_perf), max_perf)

    def test_invalid_profile_rejection(self):
        """Invalid profile identifier raises ValueError with helpful message."""
        with self.assertRaises(ValueError) as ctx:
            get_profile("non_existent_profile")
        self.assertIn("Unknown training profile", str(ctx.exception))

    def test_no_silent_fallback_capability_check(self):
        """Capability check enforces NO SILENT FALLBACK when memory is insufficient."""
        # 1. Sufficient memory: passes
        prof = get_profile("max_performance")
        self.assertTrue(validate_profile_capability(prof, available_memory_mb=32000.0))

        # 2. Insufficient memory for max_performance (~30,900 MB req):
        # Must raise ProfileCapabilityError, NEVER silently fallback to balanced/reference
        with self.assertRaises(ProfileCapabilityError) as ctx:
            validate_profile_capability("max_performance", available_memory_mb=25000.0)

        err = ctx.exception
        self.assertEqual(err.requested_profile, "max_performance")
        self.assertGreater(err.required_memory_estimate, 25000.0)
        self.assertEqual(err.available_memory, 25000.0)
        # Recommended profile should be balanced (requires ~23,550 MB <= 25,000 MB)
        self.assertEqual(err.recommended_profile, "balanced")

        # 3. Insufficient memory even for balanced (e.g. 21,000 MB):
        with self.assertRaises(ProfileCapabilityError) as ctx2:
            validate_profile_capability("balanced", available_memory_mb=21000.0)
        self.assertEqual(ctx2.exception.recommended_profile, "reference")

        # 4. Insufficient memory even for reference (e.g. 15,000 MB):
        with self.assertRaises(ProfileCapabilityError) as ctx3:
            validate_profile_capability("reference", available_memory_mb=15000.0)
        self.assertIsNone(ctx3.exception.recommended_profile)

    def test_formal_training_config_from_profile(self):
        """FormalTrainingConfig.from_profile initializes correct hyperparams."""
        cfg = FormalTrainingConfig.from_profile("max_performance")
        self.assertEqual(cfg.training_profile_id, "max_performance")
        self.assertEqual(cfg.per_device_train_batch_size, 4)
        self.assertEqual(cfg.gradient_accumulation_steps, 2)
        self.assertEqual(cfg.effective_batch_size, 8)
        self.assertFalse(cfg.gradient_checkpointing)
        self.assertTrue(cfg.use_vision_cache)

        manifest = cfg.to_manifest_dict()
        self.assertEqual(manifest["training_profile_id"], "max_performance")
        self.assertEqual(manifest["per_device_train_batch_size"], 4)
        self.assertEqual(manifest["gradient_accumulation_steps"], 2)
        self.assertEqual(manifest["effective_batch_size"], 8)
        self.assertEqual(manifest["gradient_checkpointing"], False)
        self.assertEqual(manifest["use_vision_cache"], True)

    def test_sample_based_progress_and_optimizer_step_accounting(self):
        """1552 samples always produces exactly 194 optimizer steps across all profiles."""
        total_samples = 1552

        for prof_id, expected_mb_size, expected_mb_count in [
            ("reference", 1, 1552),
            ("balanced", 2, 776),
            ("max_performance", 4, 388),
        ]:
            prof = get_profile(prof_id)
            self.assertEqual(prof.microbatch_size, expected_mb_size)
            mb_count = (total_samples + prof.microbatch_size - 1) // prof.microbatch_size
            self.assertEqual(mb_count, expected_mb_count)

            opt_steps = compute_formal_optimizer_steps(
                num_records=total_samples,
                gradient_accumulation_steps=prof.gradient_accumulation_steps,
                num_train_epochs=1,
                microbatch_size=prof.microbatch_size,
            )
            self.assertEqual(
                opt_steps,
                194,
                f"Profile {prof_id} produced {opt_steps} optimizer steps instead of 194",
            )


if __name__ == "__main__":
    unittest.main()
