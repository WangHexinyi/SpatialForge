"""CPU-side scene representation independent of Blender."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from spatialforge.environment.camera import Vec3


@dataclass(frozen=True)
class SceneObject:
    """Immutable representation of a single scene primitive."""

    name: str
    shape: str
    color: str
    location: Vec3
    size: float


@dataclass(frozen=True)
class SceneState:
    """Immutable CPU-side description of an entire scene."""

    scene_id: str
    seed: int
    objects: tuple  # tuple of SceneObject


def load_scene_state(path: Path, scene_id: str | None = None) -> SceneState:
    """Load a v1 scene JSON file into a SceneState.

    The loader consumes existing v1 JSON format without modifying it.
    If scene_id is not provided, it is derived from the filename stem.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    if scene_id is None:
        scene_id = path.stem

    seed = data.get("seed", 0)

    objects = []
    for obj in data["objects"]:
        loc = obj["location"]
        objects.append(
            SceneObject(
                name=obj["name"],
                shape=obj["shape"],
                color=obj["color"],
                location=(loc[0], loc[1], loc[2]),
                size=obj["size"],
            )
        )

    return SceneState(
        scene_id=scene_id,
        seed=seed,
        objects=tuple(objects),
    )
