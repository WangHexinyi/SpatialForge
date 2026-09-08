"""SpatialForge God View / Research Inspector.

Provides read-only 3D scene inspection, observation validation, projection provenance,
question-model-prediction alignment, and live telemetry for spatial reasoning research.
"""

from spatialforge.inspector.contracts import (
    SCHEMA_VERSION,
    InspectorSnapshot,
    adapt_presentation_order,
    validate_model_input_isolation,
)
from spatialforge.inspector.projection import (
    ProjectionContract,
    reconstruct_legacy_projection,
    make_unknown_projection,
    run_blender_projection_probe,
)
from spatialforge.inspector.artifacts import ArtifactRepository, resolve_safe_path
from spatialforge.inspector.snapshot import build_inspector_snapshot

__all__ = [
    "SCHEMA_VERSION",
    "InspectorSnapshot",
    "ProjectionContract",
    "ArtifactRepository",
    "build_inspector_snapshot",
    "adapt_presentation_order",
    "validate_model_input_isolation",
    "reconstruct_legacy_projection",
    "make_unknown_projection",
    "run_blender_projection_probe",
    "resolve_safe_path",
]
