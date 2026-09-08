"""CPU-side scene representation independent of Blender."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from spatialforge.environment.camera import Vec3


@dataclass(frozen=True)
class SceneObject:
    """Immutable representation of a single scene primitive with optional structure."""

    name: str
    shape: str
    color: str
    location: Vec3
    size: float
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (pitch, roll, yaw) in degrees
    role: str = "object"  # "object", "support_surface", "compound_base", "compound_part"
    support_parent: Optional[str] = None  # name of supporting object or None
    placement_mode: str = "on_floor"  # "on_floor", "on_top_of", "on_platform", "stacked_chain", "tight_cluster", "leaning", "compound_assembly"
    compound_id: Optional[str] = None  # compound object group id
    compound_part: Optional[str] = None  # part label in compound
    shape_variant: Optional[str] = None  # optional variant (e.g. truncated, hollow, ring_stand)


@dataclass(frozen=True)
class SceneState:
    """Immutable CPU-side description of an entire scene."""

    scene_id: str
    seed: int
    objects: tuple  # tuple of SceneObject


def load_scene_state(path: Path, scene_id: str | None = None) -> SceneState:
    """Load a scene JSON file into a SceneState.

    Consumes legacy v1 format as well as upgraded structured formats.
    If scene_id is not provided, it is derived from the filename stem.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    if scene_id is None:
        scene_id = path.stem

    seed = data.get("seed", 0)

    objects = []
    for obj in data["objects"]:
        loc = obj["location"]
        rot_data = obj.get("rotation", (0.0, 0.0, 0.0))
        rot = (
            (float(rot_data[0]), float(rot_data[1]), float(rot_data[2]))
            if isinstance(rot_data, (list, tuple)) and len(rot_data) == 3
            else (0.0, 0.0, 0.0)
        )
        objects.append(
            SceneObject(
                name=obj["name"],
                shape=obj["shape"],
                color=obj["color"],
                location=(loc[0], loc[1], loc[2]),
                size=obj["size"],
                rotation=rot,
                role=obj.get("role", "object"),
                support_parent=obj.get("support_parent", None),
                placement_mode=obj.get("placement_mode", "on_floor"),
                compound_id=obj.get("compound_id", None),
                compound_part=obj.get("compound_part", None),
                shape_variant=obj.get("shape_variant", None),
            )
        )

    return SceneState(
        scene_id=scene_id,
        seed=seed,
        objects=tuple(objects),
    )
