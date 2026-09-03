from dataclasses import dataclass
from typing import Tuple


Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class CameraPose:
    """Authoritative camera pose for SpatialForge v2.

    Coordinate conventions:
    - World space is Z-up.
    - Camera-local +X is right.
    - Camera-local +Y is up.
    - Camera-local +Z is forward.
    """

    position: Vec3
    look_at: Vec3
    up: Vec3 = (0.0, 0.0, 1.0)
    fov_deg: float = 60.0

    def __post_init__(self) -> None:
        if self.position == self.look_at:
            raise ValueError("camera position and look_at must differ")

        if not 0.0 < self.fov_deg < 180.0:
            raise ValueError("fov_deg must be between 0 and 180 degrees")
