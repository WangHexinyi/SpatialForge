"""Export controllable prototype challenge scenes for SpatialForge G2.1-C P0.

STATUS: DEPRECATED AS TRAINING SOURCE. This exporter writes procedural toy-world
fixtures that are NOT used as formal training data (see docs/PROJECT_PLAN_v3.md).
It is retained only for deterministic geometry/regression fixtures.

Usage:
    python scripts/generate_challenge_scenes.py --tier S2 --count 5
    python scripts/generate_challenge_scenes.py --export-reference-suite

Pure Python. No bpy or GPU dependencies.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spatialforge.environment.challenge import (
    SceneChallengeConfig,
    generate_challenge_scene,
    TIER_SPECS,
)


def export_challenge_scene(
    tier: str,
    seed: int,
    out_dir: Path,
    object_count: int | None = None,
) -> Path:
    """Generate and write a single challenge scene JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SceneChallengeConfig(
        tier=tier,
        seed=seed,
        object_count=object_count,
        scene_id=f"scene_challenge_{tier.lower()}_{seed:03d}",
    )
    ch = generate_challenge_scene(cfg)
    scene_dict = ch.to_scene_dict()

    target_path = out_dir / f"{cfg.scene_id}.json"
    target_path.write_text(
        json.dumps(scene_dict, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target_path


def export_reference_suite(out_dir: Path) -> List[Path]:
    """Generate canonical reference scenes across all difficulty tiers (S0, S1, S2, S3)."""
    suite_specs = [
        ("S0", 0, 4),
        ("S1", 0, 7),
        ("S2", 0, 9),
        ("S3", 0, 11),
        ("S2", 1, 10),
        ("S3", 1, 12),
    ]
    exported = []
    for tier, seed, count in suite_specs:
        p = export_challenge_scene(tier=tier, seed=seed, out_dir=out_dir, object_count=count)
        exported.append(p)
    return exported


def main():
    parser = argparse.ArgumentParser(
        description="Generate SpatialForge G2.1-C P0 challenge scenes."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/scenes"),
        help="Destination directory for exported scene JSONs (default: outputs/scenes)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="S2",
        choices=list(TIER_SPECS.keys()),
        help="Difficulty tier: S0, S1, S2, S3 (default: S2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of scenes to export starting from seed (default: 1)",
    )
    parser.add_argument(
        "--export-reference-suite",
        action="store_true",
        help="Export canonical S0, S1, S2, S3 prototype reference suite for God View inspection",
    )

    args = parser.parse_args()

    if args.export_reference_suite:
        paths = export_reference_suite(args.out_dir)
        print(f"Exported {len(paths)} reference challenge scenes to {args.out_dir}:")
        for p in paths:
            print(f"  * {p.name}")
    else:
        for i in range(args.count):
            seed = args.seed + i
            p = export_challenge_scene(tier=args.tier, seed=seed, out_dir=args.out_dir)
            print(f"Exported {args.tier} scene: {p}")


if __name__ == "__main__":
    main()
