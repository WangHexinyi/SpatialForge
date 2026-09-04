"""Metadata describing one rendered observation."""

from dataclasses import dataclass

from spatialforge.environment.camera import CameraPose


@dataclass(frozen=True)
class ObservationMetadata:
    """CPU-side metadata for a single rendered view.

    Every field is preserved to enable downstream reconstruction.
    """

    scene_id: str
    view_id: str
    image_path: str
    camera: CameraPose

    def to_dict(self) -> dict:
        """Deterministic JSON-compatible serialization.

        CameraPose fields are stored as explicit nested dicts, not opaque strings.
        """
        return {
            "scene_id": self.scene_id,
            "view_id": self.view_id,
            "image_path": self.image_path,
            "camera": {
                "position": list(self.camera.position),
                "look_at": list(self.camera.look_at),
                "up": list(self.camera.up),
                "fov_deg": self.camera.fov_deg,
            },
        }
