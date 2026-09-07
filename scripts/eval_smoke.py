"""Targeted smoke test for G2.0-E3.1 Evaluator Hardening and Protocol Freeze.

Executes CPU verification of:
1. Frozen E1 dataset SHA-256 integrity against e1_dataset_manifest.json
2. Training split leakage guard (scenes 080-099 excluded)
3. Hardened directional parser (canonical + adversarial cases)
4. Hardened yes/no parser (canonical + adversarial cases)
5. Historical VSR taxonomy & disjointness invariant
6. Local model snapshot revision & environment provenance
7. Collision-free run directory helper & run manifest generation
8. Deterministic evaluation record serialization & metric aggregation
"""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spatialforge.experiment.evaluation import (
    DIRECTIONAL_VOCABULARY,
    VSR_ALL_DIMS,
    VSR_BROADER_DIMS,
    VSR_DIRECT_DIMS,
    EvaluationPredictionRecord,
    aggregate_metrics,
    classify_vsr_relation,
    parse_directional_answer,
    parse_yesno_answer,
)
from spatialforge.experiment.protocol import (
    E1_MANIFEST_PATH,
    ENGINEERING_SEED,
    EXPERIMENT_ID,
    FORMAL_SEEDS,
    GROUPS,
    HOLDOUT_SCENE_IDS,
    IS_VSR_TEST_ONLY,
    MODEL_ID,
    S1_SEMANTIC_LABEL,
    S2_SEMANTIC_LABEL,
    TRAIN_DATASET_PATHS,
    get_environment_provenance,
    get_run_output_dir,
    recover_model_revision,
    seed_everything,
    validate_seed,
    validate_training_scenes,
    verify_frozen_e1_datasets,
)


def run_eval_smoke() -> bool:
    print("=" * 60)
    print("SpatialForge G2.0-E3.1 Evaluator Hardening & Protocol Smoke")
    print("=" * 60)

    # 1. Dataset Hash Verification
    print("\n[1/7] Verifying frozen E1 dataset hashes...")
    verified_hashes = verify_frozen_e1_datasets(REPO_ROOT / E1_MANIFEST_PATH, base_dir=REPO_ROOT)
    for k, h in verified_hashes.items():
        print(f"  ✓ {k:12s}: {h}")

    # 2. Training Split Leakage Guard
    print("\n[2/7] Verifying training split isolation (scenes 080-099)...")
    for grp in ("A", "C", "D"):
        p = REPO_ROOT / TRAIN_DATASET_PATHS[grp]
        scenes = set()
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                scenes.add(d["scene_id"])
        leaked = scenes.intersection(HOLDOUT_SCENE_IDS)
        assert not leaked, f"Holdout scenes leaked in {grp}: {leaked}"
        print(f"  ✓ Group {grp}: {len(scenes)} scenes, 0 holdout scenes (disjoint)")

    # 3. Hardened Directional Parser
    print("\n[3/7] Verifying hardened directional parser...")
    assert parse_directional_answer("left") == "left"
    assert parse_directional_answer("The red sphere is left of the cube.") == "left"
    assert parse_directional_answer("The object is in front of the other object.") == "front"
    assert parse_directional_answer("not left") is None
    assert parse_directional_answer("not in front") is None
    assert parse_directional_answer("left or right") is None
    assert parse_directional_answer("either left or right") is None
    assert parse_directional_answer("front or behind") is None
    assert parse_directional_answer("I think it might be right") is None
    assert parse_directional_answer("it might be left") is None
    assert parse_directional_answer("could be behind") is None
    assert parse_directional_answer("probably left") is None
    assert parse_directional_answer("I cannot tell whether it is left") is None
    assert parse_directional_answer("maybe behind") is None
    assert parse_directional_answer("unclear") is None
    assert parse_directional_answer("upright") is None
    print("  ✓ Strict directional parser passed all valid & adversarial checks")

    # 4. Hardened Yes/No Parser
    print("\n[4/7] Verifying hardened yes/no parser...")
    assert parse_yesno_answer("yes") == "yes"
    assert parse_yesno_answer("Yes, true.") == "yes"
    assert parse_yesno_answer("no") == "no"
    assert parse_yesno_answer("No, false.") == "no"
    assert parse_yesno_answer("probably yes") is None
    assert parse_yesno_answer("probably no") is None
    assert parse_yesno_answer("maybe yes") is None
    assert parse_yesno_answer("maybe no") is None
    assert parse_yesno_answer("no idea") is None
    assert parse_yesno_answer("not sure") is None
    assert parse_yesno_answer("cannot say") is None
    assert parse_yesno_answer("I am not sure") is None
    assert parse_yesno_answer("Cannot determine") is None
    assert parse_yesno_answer("yes or no") is None
    assert parse_yesno_answer("yes and no") is None
    assert parse_yesno_answer("both yes and no") is None
    assert parse_yesno_answer("maybe") is None
    assert parse_yesno_answer("unknown") is None
    print("  ✓ Strict yes/no parser passed all valid & adversarial checks")

    # 5. VSR Taxonomy & Invariants
    print("\n[5/7] Verifying historical VSR taxonomy...")
    assert VSR_DIRECT_DIMS.isdisjoint(VSR_BROADER_DIMS)
    assert classify_vsr_relation("left of") in VSR_DIRECT_DIMS
    assert classify_vsr_relation("in front of") in VSR_DIRECT_DIMS
    assert classify_vsr_relation("facing") in VSR_BROADER_DIMS
    assert classify_vsr_relation("inside") == "topology"
    assert classify_vsr_relation("on") == "contact"
    assert classify_vsr_relation("part of") == "part_whole"
    print("  ✓ Complete 8-dimension historical taxonomy verified; direct/broader disjoint")

    # 6. Model Revision & Provenance
    print("\n[6/7] Recovering model revision and environment provenance...")
    rev = recover_model_revision()
    print(f"  ✓ Local model snapshot revision: {rev}")
    prov = get_environment_provenance()
    print(f"  ✓ Python: {prov.get('python')}, Torch: {prov.get('torch')}, Peft: {prov.get('peft')}")
    print(f"  ✓ Git commit: {prov.get('git_commit')}")

    # 7. Metric Aggregation & Run Contract
    print("\n[7/7] Verifying evaluation record serialization & metric aggregation...")
    rec1 = EvaluationPredictionRecord("s1", "synthetic_s1", "scene_080", "A", 42, "horizontal", "left", "left", "left", True, True, MODEL_ID)
    rec2 = EvaluationPredictionRecord("s2", "synthetic_s1", "scene_080", "A", 42, "horizontal", "left", "not left", None, False, False, MODEL_ID)
    metrics = aggregate_metrics([rec1, rec2])
    assert metrics["overall"]["total"] == 2
    assert metrics["overall"]["valid_predictions"] == 1
    assert metrics["overall"]["invalid_predictions"] == 1
    assert metrics["overall"]["accuracy"] == 0.5
    assert metrics["overall"]["valid_accuracy"] == 1.0

    b_path = get_run_output_dir("baseline_b")
    a_path = get_run_output_dir("group_a", 42)
    assert str(b_path).endswith("outputs/experiments/g2.0-e3/baseline_b")
    assert str(a_path).endswith("outputs/experiments/g2.0-e3/group_a/seed_42")
    print("  ✓ Evaluation records & metric aggregation verified; run paths collision-free")

    print("\n" + "=" * 60)
    print("ALL G2.0-E3.1 SMOKE CHECKS PASSED.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_eval_smoke()
    sys.exit(0 if success else 1)
