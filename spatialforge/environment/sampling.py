"""Camera sampling system for SpatialForge G2.0-B.1.

Pure CPU Python. No bpy import and no third-party dependencies.

Canonical direction convention
-----------------------------
A canonical *direction* describes the CAMERA OFFSET FROM THE TARGET, not the
camera forward vector. For a normalised direction ``d``::

    position = target + radius * normalize(d)
    look_at  = target

This mirrors the semantic naming of ``generate_cardinal_views()`` where the
view name describes the camera position (e.g. "south" camera sits south of the
target looking north).

Axes (world is Z-up):
    east  = (+1, 0,  0)
    west  = (-1, 0,  0)
    north = ( 0,+1,  0)
    south = ( 0,-1,  0)
    up    = ( 0, 0, +1)
    down  = ( 0, 0, -1)

Set composition / deterministic order
--------------------------------------
6  = 6 axes
14 = 6 axes + 8 cube corners            (all of {-1,0,1}^3 with |d| == 3)
26 = 6 axes + 12 edge dirs + 8 corners  (all nonzero of {-1,0,1}^3)

Edges are the 12 weight-2 directions; corners the 8 weight-3 directions.
Within each weight class directions are produced in a fixed, documented order
(see the build helpers below), so IDs and order are fully deterministic.

Vertical up-vector policy
-------------------------
World up is normally (0, 0, 1). If the camera viewing direction is parallel or
numerically near-parallel to world Z, the deterministic fallback up vector
(0, 1, 0) is used so the pose is always valid under ``camera_basis``.
"""

import math
import random
from typing import List, Tuple

from spatialforge.environment.camera import CameraPose, Vec3
from spatialforge.environment.geometry import camera_basis

_EPS = 1e-12

# Near-parallel up threshold: if the viewing direction is within roughly
# 1.0e-3 radians (~0.08 degrees) of world Z, fall back to up=(0, 1, 0) so the
# pose basis stays numerically well-conditioned instead of only catching the
# exactly-parallel case.
_UP_PARALLEL_EPS = 1e-6

# Per-coordinate sign -> human readable id segment.
_DIM_SEGMENTS = (
    {1: "east", -1: "west"},   # x
    {1: "north", -1: "south"},  # y
    {1: "up", -1: "down"},      # z
)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: Vec3) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: Vec3) -> Vec3:
    length = _norm(v)
    if length < _EPS:
        raise ValueError("cannot normalize a zero-length vector")
    return (v[0] / length, v[1] / length, v[2] / length)


def _scaled(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _direction_id(direction: Vec3) -> str:
    """Lower_snake_case id for a canonical {-1,0,1} direction vector."""
    segments = []
    for dim in range(3):
        sign = int(direction[dim])
        if sign != 0:
            segments.append(_DIM_SEGMENTS[dim][sign])
    return "_".join(segments)


def _build_axes() -> Tuple[Tuple[str, Vec3], ...]:
    out = []
    for dim in range(3):
        for sign in (1, -1):
            vec = [0.0, 0.0, 0.0]
            vec[dim] = float(sign)
            vec = tuple(vec)  # type: ignore[assignment]
            out.append((_direction_id(vec), vec))
    return tuple(out)


def _build_edges() -> Tuple[Tuple[str, Vec3], ...]:
    out = []
    # Pair (i, j) iterated in fixed order: xy, xz, yz. Within each pair the
    # sign of coordinate i is swept before coordinate j, both +1 then -1.
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for si in (1, -1):
            for sj in (1, -1):
                vec = [0.0, 0.0, 0.0]
                vec[i] = float(si)
                vec[j] = float(sj)
                vec = tuple(vec)  # type: ignore[assignment]
                out.append((_direction_id(vec), vec))
    return tuple(out)


def _build_corners() -> Tuple[Tuple[str, Vec3], ...]:
    out = []
    # Signs swept x then y then z, each +1 then -1.
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                vec = (float(sx), float(sy), float(sz))
                out.append((_direction_id(vec), vec))
    return tuple(out)


DIRECTIONS_AXIS = _build_axes()
DIRECTIONS_EDGE = _build_edges()
DIRECTIONS_CORNER = _build_corners()

DIRECTIONS_6 = DIRECTIONS_AXIS
DIRECTIONS_14 = DIRECTIONS_AXIS + DIRECTIONS_CORNER
DIRECTIONS_26 = DIRECTIONS_AXIS + DIRECTIONS_EDGE + DIRECTIONS_CORNER

_DIRECTION_SETS = {6: DIRECTIONS_6, 14: DIRECTIONS_14, 26: DIRECTIONS_26}


def canonical_directions(kind: int) -> Tuple[Tuple[str, Vec3], ...]:
    """Return the deterministic canonical direction set for ``kind`` in (6, 14, 26).

    Returns an immutable tuple of ``(view_id, direction)`` pairs where
    ``direction`` is a (dx, dy, dz) vector in {-1, 0, 1} describing the camera
    offset from the target (before normalisation).
    """
    if kind not in _DIRECTION_SETS:
        raise ValueError(f"unknown canonical set kind: {kind!r} (expected 6, 14 or 26)")
    return _DIRECTION_SETS[kind]


def _view_up(position: Vec3, look_at: Vec3) -> Vec3:
    """Deterministic up vector that is valid under ``camera_basis``.

    World up is (0, 0, 1); if the viewing direction is parallel or numerically
    near-parallel to world Z (see ``_UP_PARALLEL_EPS``), fall back to (0, 1, 0).
    """
    forward = _normalize(_sub(look_at, position))
    world_up = (0.0, 0.0, 1.0)
    if abs(_dot(forward, world_up)) >= 1.0 - _UP_PARALLEL_EPS:
        return (0.0, 1.0, 0.0)
    return world_up


def _pose(look_at: Vec3, position: Vec3, fov_deg: float) -> CameraPose:
    """Build a CameraPose whose up vector is chosen by the up-vector policy."""
    return CameraPose(
        position=position,
        look_at=look_at,
        up=_view_up(position, look_at),
        fov_deg=fov_deg,
    )


def canonical_camera(
    target: Vec3,
    radius: float,
    direction: Vec3,
    fov_deg: float = 60.0,
) -> CameraPose:
    """Build a canonical CameraPose for a direction offset from the target.

    ``direction`` need not be pre-normalised (e.g. (1, 1, 1) is accepted).
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    direction = _normalize(tuple(direction))
    position = _add(target, _scaled(direction, radius))
    return _pose(target, position, fov_deg)


def canonical_views(
    kind: int,
    target: Vec3 = (0.0, 0.0, 0.4),
    radius: float = 6.0,
    fov_deg: float = 60.0,
) -> List[CameraPose]:
    """Return CameraPose values for every direction in a canonical set.

    ``kind`` is 6, 14 or 26. Output order matches ``canonical_directions``.
    """
    return [canonical_camera(target, radius, d, fov_deg) for _, d in canonical_directions(kind)]


def jitter_pose(
    camera: CameraPose,
    seed: int,
    *,
    angular_deg: float = 0.0,
    radius_frac: float = 0.0,
    target_radius: float = 0.0,
) -> CameraPose:
    """Perturb a camera pose with a local seeded RNG (never the global RNG).

    Perturbation modes (all disabled by default, so a zero configuration
    returns the input pose unchanged):
      * ``angular_deg``  - true maximum angular deviation (degrees) of the
                           viewing direction. The new direction is drawn from
                           a bounded cone: it is rotated toward a random unit
                           tangent in the camera right/up plane by an angle
                           theta sampled uniformly in ``[0, angular_deg]``, so
                           the resulting deviation never exceeds angular_deg.
      * ``radius_frac``  - relative perturbation of the target distance,
                           drawn in ``[-radius_frac, radius_frac]``; must be
                           in ``[0, 1)`` so the camera radius stays positive.
      * ``target_radius`` - true 3D radial bound on the look-at target
                           displacement: the new target is drawn uniformly
                           inside a sphere of this radius centred on the
                           original target, so
                           ``norm(target_j - target) <= target_radius``.
                           Zero leaves the target exactly unchanged.

    Identical ``camera`` + ``seed`` + parameters reproduce identical output.
    The returned pose is always valid under ``camera_basis``.
    """
    if angular_deg < 0.0:
        raise ValueError(f"angular_deg must be >= 0, got {angular_deg}")
    if radius_frac < 0.0:
        raise ValueError(f"radius_frac must be >= 0, got {radius_frac}")
    if radius_frac >= 1.0:
        raise ValueError(f"radius_frac must be < 1, got {radius_frac}")
    if target_radius < 0.0:
        raise ValueError(f"target_radius must be >= 0, got {target_radius}")
    if angular_deg == 0.0 and radius_frac == 0.0 and target_radius == 0.0:
        return camera

    rng = random.Random(seed)

    right, up, forward = camera_basis(camera)
    target = camera.look_at
    radius = _norm(_sub(target, camera.position))

    factor = 1.0 + rng.uniform(-radius_frac, radius_frac)
    radius_j = radius * factor

    if angular_deg == 0.0:
        perturbed = forward
    else:
        max_angle = math.radians(angular_deg)
        azimuth = rng.uniform(0.0, 2.0 * math.pi)
        theta = rng.uniform(0.0, max_angle)
        tangent = _add(
            _scaled(right, math.cos(azimuth)),
            _scaled(up, math.sin(azimuth)),
        )
        perturbed = _normalize(
            _add(
                _scaled(forward, math.cos(theta)),
                _scaled(tangent, math.sin(theta)),
            )
        )

    delta_target: Vec3 = (0.0, 0.0, 0.0)
    if target_radius > 0.0:
        # Uniform-in-volume sphere sample: isotropic unit direction times a
        # radial magnitude proportional to U^(1/3), scaled by target_radius.
        direction = (0.0, 0.0, 0.0)
        for _ in range(20):
            raw = (rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
            if _norm(raw) >= _EPS:
                direction = _normalize(raw)
                break
        else:
            raise RuntimeError("target jitter failed to produce a nonzero direction")
        radial = target_radius * (rng.random() ** (1.0 / 3.0))
        delta_target = _scaled(direction, radial)
    target_j = _add(target, delta_target)

    position = _sub(target_j, _scaled(perturbed, radius_j))
    return _pose(target_j, position, camera.fov_deg)


def sample_camera(
    target: Vec3 = (0.0, 0.0, 0.4),
    radius: float = 6.0,
    seed: int = 0,
    fov_deg: float = 60.0,
) -> CameraPose:
    """Sample a camera from the full continuous sphere of directions.

    The viewing/camera-offset direction is drawn uniformly on the unit sphere
    via three independent Gaussians (avoids the pole bias of naive yaw/pitch
    uniform sampling) using a local seeded RNG. ``target`` and ``radius`` are
    applied deterministically. Reproducible for a given ``seed``.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    rng = random.Random(seed)

    direction = (0.0, 0.0, 0.0)
    for _ in range(20):
        sample = (rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
        if _norm(sample) >= _EPS:
            direction = _normalize(sample)
            break
    else:
        raise RuntimeError("continuous sampling failed to produce a nonzero direction")

    position = _add(target, _scaled(direction, radius))
    return _pose(target, position, fov_deg)
