"""View-conditioned spatial truth engine for SpatialForge G2.0-C.

Pure CPU Python. No bpy import and no third-party dependencies.

Scope
-----
This module computes deterministic *camera-relative* geometric truth from a
``SceneState`` and a ``CameraPose``. It is deliberately NOT an
image-projection, occlusion, visibility, or QA layer.

Semantics
---------
All relations operate on ``SceneObject.location`` (object centres) only.
Object extents are never inferred from ``size``; no bbox / surface / contact /
overlap / projection / occlusion semantics live here.

Relations are derived from camera-frame coordinates returned by
``world_to_camera``, i.e. ``(camera_right_x, camera_up_y, camera_depth)`` with
``camera_depth`` signed along the camera forward axis. For a pair (A, B) with
frame coordinates ``a`` and ``b``:

    left / right      dx = a.x - b.x
                      dx < -eps -> A LEFT of B
                      dx > +eps -> A RIGHT of B
                      otherwise  -> ALIGNED

    above / below     dy = a.y - b.y        (camera-relative +Y)
                      dy > +eps -> A ABOVE B
                      dy < -eps -> A BELOW B
                      otherwise  -> ALIGNED

    front / behind    dz = a.z - b.z
                      dz < -eps -> A FRONT of B   (shallower depth)
                      dz > +eps -> A BEHIND B
                      otherwise  -> SAME_DEPTH

    near / far        dr = metric(A) - metric(B), where metric is the
                      Euclidean distance from camera.position to the object
                      centre. Camera depth is NOT used for near/far.
                      dr < -eps -> A NEARER than B
                      dr > +eps -> A FARTHER than B
                      otherwise  -> EQUIDISTANT

All relations are signed pair orderings from the perspective of A (the first
object in the pair) relative to B. Neutral states (ALIGNED / SAME_DEPTH /
EQUIDISTANT) are part of the truth and are never silently collapsed.

Behind-camera objects are kept. Objects with ``camera_depth <= eps`` get
``in_front_of_camera = False`` but their camera-frame coordinates and pairwise
geometric relations are still emitted. Visibility / filtering belongs to a
later QA / occlusion layer.

Identity
--------
``SceneObject.name`` is NOT assumed unique. The stable order of
``SceneState.objects`` is the authoritative identity for this gate, exposed as
``object_index``. Pair ordering is the deterministic index combination order
``(0, 1), (0, 2), ... (1, 2), ...``.
"""

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Tuple

from spatialforge.environment.camera import CameraPose, Vec3
from spatialforge.environment.geometry import metric_distance, world_to_camera
from spatialforge.environment.scene import SceneState

# Default tie tolerance for all scalar comparisons.
_DEFAULT_EPS = 1e-6

# left / right
REL_LEFT = "left"
REL_RIGHT = "right"
REL_ALIGNED = "aligned"

# above / below
REL_ABOVE = "above"
REL_BELOW = "below"

# front / behind
REL_FRONT = "front"
REL_BEHIND = "behind"
REL_SAME_DEPTH = "same_depth"

# near / far
REL_NEARER = "nearer"
REL_FARTHER = "farther"
REL_EQUIDISTANT = "equidistant"


def _euclidean(a: Vec3, b: Vec3) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _classify(delta: float, eps: float, negative: str, positive: str, neutral: str) -> str:
    """Map a signed scalar delta onto a three-way relation label."""
    if delta < -eps:
        return negative
    if delta > eps:
        return positive
    return neutral


@dataclass(frozen=True)
class ObjectTruth:
    """Camera-relative truth for a single scene object."""

    object_index: int
    name: str
    shape: str
    color: str
    size: float
    world_location: Vec3
    camera_right_x: float
    camera_up_y: float
    camera_depth: float
    camera_metric_distance: float
    in_front_of_camera: bool

    def to_dict(self) -> dict:
        return {
            "object_index": self.object_index,
            "name": self.name,
            "shape": self.shape,
            "color": self.color,
            "size": self.size,
            "world_location": list(self.world_location),
            "camera_right_x": self.camera_right_x,
            "camera_up_y": self.camera_up_y,
            "camera_depth": self.camera_depth,
            "camera_metric_distance": self.camera_metric_distance,
            "in_front_of_camera": self.in_front_of_camera,
        }


@dataclass(frozen=True)
class PairTruth:
    """Camera-relative pairwise spatial truth for an ordered object pair."""

    object_index_a: int
    object_index_b: int
    name_a: str
    name_b: str
    left_right: str
    above_below: str
    front_behind: str
    near_far: str
    metric_distance: float

    def to_dict(self) -> dict:
        return {
            "object_index_a": self.object_index_a,
            "object_index_b": self.object_index_b,
            "name_a": self.name_a,
            "name_b": self.name_b,
            "left_right": self.left_right,
            "above_below": self.above_below,
            "front_behind": self.front_behind,
            "near_far": self.near_far,
            "metric_distance": self.metric_distance,
        }


@dataclass(frozen=True)
class SpatialTruthRecord:
    """Deterministic camera-relative spatial truth for one SceneState + CameraPose.

    ``objects`` mirrors the authoritative ``SceneState.objects`` order via
    ``object_index``. ``pairs`` are ordered deterministically as index
    combinations ``(0, 1), (0, 2), ... (1, 2), ...``.

    World facts (object identity metadata, world location, metric
    object-to-object distance) are preserved verbatim and are never mutated or
    reinterpreted based on ``CameraPose``.
    """

    scene_id: str
    camera: CameraPose
    eps: float
    objects: Tuple[ObjectTruth, ...]
    pairs: Tuple[PairTruth, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "camera": {
                "position": list(self.camera.position),
                "look_at": list(self.camera.look_at),
                "up": list(self.camera.up),
                "fov_deg": self.camera.fov_deg,
            },
            "eps": self.eps,
            "objects": [obj.to_dict() for obj in self.objects],
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


def compute_spatial_truth(
    state: SceneState,
    camera: CameraPose,
    eps: float = _DEFAULT_EPS,
) -> SpatialTruthRecord:
    """Compute deterministic camera-relative spatial truth for a scene.

    Args:
        state: The immutable scene to evaluate.
        camera: The authoritative camera pose.
        eps: Tie tolerance for scalar comparisons. Must be finite and >= 0.

    Returns:
        A frozen ``SpatialTruthRecord`` with per-object and per-pair truth.
    """
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError(f"eps must be a finite value >= 0, got {eps}")

    frames = []
    metric_distances = []
    for obj in state.objects:
        frame = world_to_camera(obj.location, camera)
        frames.append(frame)
        metric_distances.append(metric_distance(obj.location, camera))

    object_truths = []
    for index, obj in enumerate(state.objects):
        frame = frames[index]
        depth = frame[2]
        object_truths.append(
            ObjectTruth(
                object_index=index,
                name=obj.name,
                shape=obj.shape,
                color=obj.color,
                size=obj.size,
                world_location=obj.location,
                camera_right_x=frame[0],
                camera_up_y=frame[1],
                camera_depth=depth,
                camera_metric_distance=metric_distances[index],
                in_front_of_camera=depth > eps,
            )
        )

    pair_truths = []
    for index_a, index_b in combinations(range(len(state.objects)), 2):
        a = state.objects[index_a]
        b = state.objects[index_b]
        frame_a = frames[index_a]
        frame_b = frames[index_b]

        pair_truths.append(
            PairTruth(
                object_index_a=index_a,
                object_index_b=index_b,
                name_a=a.name,
                name_b=b.name,
                left_right=_classify(
                    frame_a[0] - frame_b[0], eps, REL_LEFT, REL_RIGHT, REL_ALIGNED
                ),
                above_below=_classify(
                    frame_a[1] - frame_b[1], eps, REL_BELOW, REL_ABOVE, REL_ALIGNED
                ),
                front_behind=_classify(
                    frame_a[2] - frame_b[2], eps, REL_FRONT, REL_BEHIND, REL_SAME_DEPTH
                ),
                near_far=_classify(
                    metric_distances[index_a] - metric_distances[index_b],
                    eps,
                    REL_NEARER,
                    REL_FARTHER,
                    REL_EQUIDISTANT,
                ),
                metric_distance=_euclidean(a.location, b.location),
            )
        )

    return SpatialTruthRecord(
        scene_id=state.scene_id,
        camera=camera,
        eps=eps,
        objects=tuple(object_truths),
        pairs=tuple(pair_truths),
    )
