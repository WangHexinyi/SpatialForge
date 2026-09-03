import math
from typing import Tuple

from spatialforge.environment.camera import CameraPose, Vec3


_EPS = 1e-12


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Vec3) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: Vec3) -> Vec3:
    length = _norm(v)
    if length < _EPS:
        raise ValueError("cannot normalize a zero-length vector")
    return (v[0] / length, v[1] / length, v[2] / length)


def camera_basis(camera: CameraPose) -> Tuple[Vec3, Vec3, Vec3]:
    """Return orthonormal (right, up, forward) vectors in world space."""

    forward = _normalize(_sub(camera.look_at, camera.position))

    right_raw = _cross(forward, camera.up)
    if _norm(right_raw) < _EPS:
        raise ValueError("camera up vector cannot be parallel to viewing direction")

    right = _normalize(right_raw)
    true_up = _normalize(_cross(right, forward))

    return right, true_up, forward


def world_to_camera(point: Vec3, camera: CameraPose) -> Vec3:
    """Transform a world-space point into camera-local coordinates.

    Returns:
        (x_right, y_up, z_forward)
    """

    right, up, forward = camera_basis(camera)
    delta = _sub(point, camera.position)

    return (
        _dot(delta, right),
        _dot(delta, up),
        _dot(delta, forward),
    )


def metric_distance(point: Vec3, camera: CameraPose) -> float:
    """Euclidean distance between camera centre and a world-space point."""

    return _norm(_sub(point, camera.position))


def camera_depth(point: Vec3, camera: CameraPose) -> float:
    """Signed distance along the camera forward axis."""

    return world_to_camera(point, camera)[2]
