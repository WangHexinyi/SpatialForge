"""Deterministic versioned InspectorSnapshot contract (spatialforge.inspector.v1).

Strict Scientific Boundary:
- Separates privileged Environment Truth, Camera & Observation, Model Input,
  Supervision, Model Prediction, and Telemetry into isolated channels.
- Model input channel adheres to strict allowlist: image reference + question ONLY.
- Presentation-order adapter ensures relevant 3D spatial truth matches the actual
  order of objects in the question without running a second geometry engine.
"""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from spatialforge.environment.camera import CameraPose
from spatialforge.environment.relation import (
    PairTruth,
    REL_ABOVE,
    REL_ALIGNED,
    REL_BEHIND,
    REL_BELOW,
    REL_EQUIDISTANT,
    REL_FARTHER,
    REL_FRONT,
    REL_LEFT,
    REL_NEARER,
    REL_RIGHT,
    REL_SAME_DEPTH,
    SpatialTruthRecord,
)
from spatialforge.environment.scene import SceneState
from spatialforge.inspector.projection import ProjectionContract

SCHEMA_VERSION = "spatialforge.inspector.v1"

# Pure deterministic relation inversions for presentation-order adaptation
INVERSE_RELATION = {
    # Horizontal
    REL_LEFT: REL_RIGHT,
    REL_RIGHT: REL_LEFT,
    REL_ALIGNED: REL_ALIGNED,
    # Vertical
    REL_ABOVE: REL_BELOW,
    REL_BELOW: REL_ABOVE,
    # Depth (signed forward axis distance)
    REL_FRONT: REL_BEHIND,
    REL_BEHIND: REL_FRONT,
    REL_SAME_DEPTH: REL_SAME_DEPTH,
    # Near / Far (Euclidean camera distance)
    REL_NEARER: REL_FARTHER,
    REL_FARTHER: REL_NEARER,
    REL_EQUIDISTANT: REL_EQUIDISTANT,
}


def adapt_presentation_order(
    pair_truth: PairTruth,
    presentation_order: int,
) -> Dict[str, Any]:
    """Adapt canonical PairTruth (index_a < index_b) to the question's actual operand order.

    Args:
        pair_truth: Canonical PairTruth from SpatialTruthRecord (a=index_a, b=index_b).
        presentation_order: 0 for canonical (A relative to B), 1 for inverted (B relative to A).

    Returns:
        Deterministic dict describing (first_object) relative to (second_object) in the
        exact perspective presented to the model/user.
    """
    if presentation_order == 0:
        return {
            "object_index_first": pair_truth.object_index_a,
            "object_index_second": pair_truth.object_index_b,
            "name_first": pair_truth.name_a,
            "name_second": pair_truth.name_b,
            "presentation_order": 0,
            "left_right": pair_truth.left_right,
            "above_below": pair_truth.above_below,
            "front_behind": pair_truth.front_behind,
            "near_far": pair_truth.near_far,
            "metric_distance": pair_truth.metric_distance,
            "description": (
                f"object {pair_truth.object_index_a} (\"{pair_truth.name_a}\") relative to "
                f"object {pair_truth.object_index_b} (\"{pair_truth.name_b}\")"
            ),
        }
    elif presentation_order == 1:
        return {
            "object_index_first": pair_truth.object_index_b,
            "object_index_second": pair_truth.object_index_a,
            "name_first": pair_truth.name_b,
            "name_second": pair_truth.name_a,
            "presentation_order": 1,
            "left_right": INVERSE_RELATION[pair_truth.left_right],
            "above_below": INVERSE_RELATION[pair_truth.above_below],
            "front_behind": INVERSE_RELATION[pair_truth.front_behind],
            "near_far": INVERSE_RELATION[pair_truth.near_far],
            "metric_distance": pair_truth.metric_distance,
            "description": (
                f"object {pair_truth.object_index_b} (\"{pair_truth.name_b}\") relative to "
                f"object {pair_truth.object_index_a} (\"{pair_truth.name_a}\") [inverted order]"
            ),
        }
    else:
        raise ValueError(f"Invalid presentation_order: {presentation_order}; expected 0 or 1")


@dataclass(frozen=True)
class InspectorIdentity:
    scene_id: str
    view_id: str
    sample_id: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "view_id": self.view_id,
            "sample_id": self.sample_id,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class EnvironmentTruthChannel:
    """Privileged ground truth channel for research inspection only.

    NEVER exposed to model input or evaluation inference.
    """

    scene_id: str
    seed: int
    objects: List[Dict[str, Any]]
    camera_object_truths: List[Dict[str, Any]]
    relevant_pair_truth: Optional[Dict[str, Any]] = None
    all_pairs_truth: Optional[List[Dict[str, Any]]] = None
    challenge_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "scene_id": self.scene_id,
            "seed": self.seed,
            "objects": self.objects,
            "camera_object_truths": self.camera_object_truths,
            "relevant_pair_truth": self.relevant_pair_truth,
            "all_pairs_truth": self.all_pairs_truth,
            "semantic_note": "in_front_of_camera is front half-space only; NOT visibility/FOV/occlusion",
        }
        if self.challenge_metadata is not None:
            d["challenge_metadata"] = self.challenge_metadata
        return d


@dataclass(frozen=True)
class CameraChannel:
    pose: Dict[str, Any]
    projection: Dict[str, Any]
    provenance: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pose": self.pose,
            "projection": self.projection,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class ObservationChannel:
    image_path: str
    image_url: str
    scene_id: str
    view_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_url": self.image_url,
            "scene_id": self.scene_id,
            "view_id": self.view_id,
        }


@dataclass(frozen=True)
class ModelInputChannel:
    """STRICT ALLOWLIST: Only sensor observation image and natural language question.

    Contains ZERO privileged 3D coordinates, object labels, or scene graphs.
    """

    image_path: str
    question: str
    image_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "image_path": self.image_path,
            "question": self.question,
        }
        if self.image_url is not None:
            d["image_url"] = self.image_url
        return d


@dataclass(frozen=True)
class SupervisionChannel:
    ground_truth: Optional[str]
    family: Optional[str]
    is_view_dependent: Optional[bool] = None
    tags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ground_truth": self.ground_truth,
            "family": self.family,
            "is_view_dependent": self.is_view_dependent,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class PredictionChannel:
    status: str  # "available", "unavailable", "alignment_error"
    raw_prediction: Optional[str] = None
    parsed_prediction: Optional[str] = None
    is_valid_prediction: Optional[bool] = None
    is_correct: Optional[bool] = None
    provenance: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "raw_prediction": self.raw_prediction,
            "parsed_prediction": self.parsed_prediction,
            "is_valid_prediction": self.is_valid_prediction,
            "is_correct": self.is_correct,
            "provenance": self.provenance,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class RuntimeChannel:
    training_profile: Optional[Dict[str, Any]]
    progress: Optional[Dict[str, Any]]
    telemetry: Optional[Dict[str, Any]]
    source: str
    freshness: str  # "fresh", "completed", "stale", "missing", "malformed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "training_profile": self.training_profile,
            "progress": self.progress,
            "telemetry": self.telemetry,
            "source": self.source,
            "freshness": self.freshness,
        }


@dataclass(frozen=True)
class InspectorSnapshot:
    """Deterministic, versioned God View snapshot container."""

    identity: InspectorIdentity
    environment_truth: EnvironmentTruthChannel
    camera: CameraChannel
    observation: ObservationChannel
    model_input: ModelInputChannel
    supervision: SupervisionChannel
    prediction: PredictionChannel
    runtime: RuntimeChannel
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert full snapshot to versioned dictionary with explicit channel separation."""
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "environment_truth": self.environment_truth.to_dict(),
            "camera": self.camera.to_dict(),
            "observation": self.observation.to_dict(),
            "model_input": self.model_input.to_dict(),
            "supervision": self.supervision.to_dict(),
            "prediction": self.prediction.to_dict(),
            "runtime": self.runtime.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Deterministic JSON string serialization."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def validate_model_input_isolation(snapshot_dict: Dict[str, Any]) -> None:
    """Strict test guard verifying that model_input contains no privileged truth.

    Raises:
        AssertionError: If any privileged or leaked keys appear in model_input.
    """
    model_input = snapshot_dict.get("model_input")
    assert isinstance(model_input, dict), "model_input must be a dict"

    allowed_keys = {"image_path", "image_url", "question"}
    actual_keys = set(model_input.keys())
    extra_keys = actual_keys - allowed_keys
    assert not extra_keys, f"model_input leaks disallowed keys: {extra_keys}"

    # Verify no coordinates or answers sneak into values
    forbidden_terms = (
        "world_location",
        "camera_depth",
        "object_index",
        "ground_truth",
        "spatial_truth",
        "supervision",
    )
    for k, v in model_input.items():
        if isinstance(v, str):
            for term in forbidden_terms:
                assert term not in v, f"Forbidden term {term!r} found in model_input[{k!r}]"
