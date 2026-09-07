"""Deterministic export of complete v1 scene JSON files.

Reproduces the exact procedural generation logic recovered from historical
v1 (scripts/scene_generator.py in commit a47b031^).

Pure Python, no bpy or third-party dependencies required.

Verification:
- For seed 0, matches examples/scenes/scene_000.json semantically.
- For seeds 0..99, object locations exactly match
  tests/fixtures/v1_front_back_reference.json to numerical precision.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

SLOTS = [(-1.2, 0.6), (1.1, 0.9), (-0.4, -0.8), (0.9, -1.1)]
SIZES = [0.6, 0.8, 1.0]
COLORS = ["red", "blue", "green", "yellow", "purple", "cyan"]
SHAPES = ["cube", "sphere", "cylinder"]


def generate_scene_dict(seed: int) -> Dict[str, Any]:
    """Generate the complete dictionary for one scene using exact v1 RNG order.

    Call order per slot in v1 (left-to-right argument evaluation):
      1. jx = slot[0] + rng.uniform(-0.25, 0.25)
      2. jy = slot[1] + rng.uniform(-0.25, 0.25)
      3. shape = rng.choice(SHAPES)
      4. color = rng.choice(COLORS)
      5. size = rng.choice(SIZES)
      6. location = [jx, jy, size / 2]
    """
    rng = random.Random(seed)
    objects: List[Dict[str, Any]] = []

    for i, slot in enumerate(SLOTS):
        jx = slot[0] + rng.uniform(-0.25, 0.25)
        jy = slot[1] + rng.uniform(-0.25, 0.25)
        shape = rng.choice(SHAPES)
        color = rng.choice(COLORS)
        size = rng.choice(SIZES)

        objects.append(
            {
                "name": f"obj{i}",
                "shape": shape,
                "color": color,
                "location": [jx, jy, size / 2],
                "size": size,
            }
        )

    return {
        "seed": seed,
        "camera": [0, -6, 3],
        "objects": objects,
    }


def export_scenes(out_dir: Path, count: int = 100) -> List[Path]:
    """Export complete scene_000.json .. scene_{count-1}.json files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i in range(count):
        scene_dict = generate_scene_dict(i)
        scene_path = out_dir / f"scene_{i:03d}.json"
        scene_path.write_text(
            json.dumps(scene_dict, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        paths.append(scene_path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export complete v1 scene JSON files."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/scenes"),
        help="Destination directory for exported scene JSONs (default: outputs/scenes)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of scenes to export (default: 100)",
    )
    args = parser.parse_args()

    paths = export_scenes(args.out_dir, args.count)
    print(f"[ok] exported {len(paths)} scene JSON files to {args.out_dir}")


if __name__ == "__main__":
    main()
