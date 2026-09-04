"""Deterministic multi-view camera set generator.

Pure CPU Python. No bpy import.
"""

from typing import List

from spatialforge.environment.camera import CameraPose, Vec3


# Distance from scene centre along each cardinal axis.
_RADIUS = 6.0

# Camera height above ground plane.
_HEIGHT = 3.0

# All cameras converge on this target near the scene centre.
_TARGET: Vec3 = (0.0, 0.0, 0.4)


def generate_cardinal_views(
    radius: float = _RADIUS,
    height: float = _HEIGHT,
    target: Vec3 = _TARGET,
    fov_deg: float = 60.0,
) -> List[CameraPose]:
    """Return four cardinal camera poses: south, east, north, west.

    Each camera looks toward *target* at the given radius.
    Camera IDs and ordering are deterministic.

    Convention (world Z-up):
        south = camera south of centre, looking north (+Y)
        east  = camera east of centre,  looking west  (-X)
        north = camera north of centre, looking south (-Y)
        west  = camera west of centre,  looking east  (+X)
    """
    views = [
        # south: position at -Y, look toward +Y
        CameraPose(position=(0.0, -radius, height), look_at=target, fov_deg=fov_deg),
        # east: position at +X, look toward -X
        CameraPose(position=(radius, 0.0, height), look_at=target, fov_deg=fov_deg),
        # north: position at +Y, look toward -Y
        CameraPose(position=(0.0, radius, height), look_at=target, fov_deg=fov_deg),
        # west: position at -X, look toward +X
        CameraPose(position=(-radius, 0.0, height), look_at=target, fov_deg=fov_deg),
    ]
    return views


VIEW_IDS = ("south", "east", "north", "west")
