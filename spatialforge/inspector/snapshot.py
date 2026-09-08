"""Builder for versioned InspectorSnapshot assembling isolated channels.

Scope:
- Synthesizes authoritative SceneState, camera pose, projection contract,
  observation reference, model input, supervision, predictions, and runtime telemetry.
- Strictly protects the scientific boundary: model_input receives ONLY image reference and question.
- Presentation-order adaptation ensures the displayed pair truth mirrors the question's actual perspective.
"""

from typing import Any, Dict, List, Optional, Tuple

from spatialforge.environment.relation import compute_spatial_truth
from spatialforge.environment.trajectory import adjust_camera_distance
from spatialforge.inspector.artifacts import ArtifactRepository
from spatialforge.inspector.contracts import (
    CameraChannel,
    EnvironmentTruthChannel,
    InspectorIdentity,
    InspectorSnapshot,
    ModelInputChannel,
    ObservationChannel,
    PredictionChannel,
    RuntimeChannel,
    SupervisionChannel,
    adapt_presentation_order,
    validate_model_input_isolation,
)
from spatialforge.inspector.projection import (
    make_preview_projection,
    make_unknown_projection,
    reconstruct_legacy_projection,
)


def build_inspector_snapshot(
    repo: ArtifactRepository,
    scene_id: str,
    view_id: str,
    sample_id: Optional[str] = None,
    run_id: Optional[str] = None,
    camera_distance: Optional[float] = None,
) -> InspectorSnapshot:
    """Build a complete, versioned InspectorSnapshot for a scene and viewpoint."""
    scene_state = repo.load_scene(scene_id)
    camera_pose = repo.get_observation_camera(scene_id, view_id)
    if camera_distance is not None and camera_distance > 0.0:
        camera_pose = adjust_camera_distance(camera_pose, float(camera_distance))

    # Resolve rendered image path
    rel_img_path = repo.get_image_path(scene_id, view_id)
    img_url = f"/api/image?path={rel_img_path}" if rel_img_path else ""

    # Resolve projection contract
    if rel_img_path and "outputs/experiments/g2.0-e/rendered" in rel_img_path:
        projection = reconstruct_legacy_projection(camera_pose)
    elif not rel_img_path:
        projection = make_preview_projection(camera_pose)
    else:
        projection = make_unknown_projection(camera_pose)

    # Compute privileged 3D spatial truth (object centers only)
    truth = compute_spatial_truth(scene_state, camera_pose)
    camera_object_truths = [obj.to_dict() for obj in truth.objects]
    all_pairs = [p.to_dict() for p in truth.pairs]

    raw_scene_objects = [
        {
            "object_index": i,
            "name": obj.name,
            "shape": obj.shape,
            "color": obj.color,
            "size": obj.size,
            "location": list(obj.location),
            "rotation": list(getattr(obj, "rotation", (0.0, 0.0, 0.0))),
            "role": getattr(obj, "role", "object"),
            "support_parent": getattr(obj, "support_parent", None),
            "placement_mode": getattr(obj, "placement_mode", "on_floor"),
            "compound_id": getattr(obj, "compound_id", None),
            "compound_part": getattr(obj, "compound_part", None),
            "shape_variant": getattr(obj, "shape_variant", None),
        }
        for i, obj in enumerate(scene_state.objects)
    ]

    # Resolve sample
    sample_data: Optional[Dict[str, Any]] = None
    if sample_id:
        sample_data = repo.get_sample(sample_id)
    else:
        candidates = repo.list_samples(scene_id=scene_id, view_id=view_id)
        if candidates:
            sample_data = candidates[0]
            sample_id = sample_data.get("id") or sample_data.get("sample_id")

    relevant_pair_truth = None
    if sample_data:
        question = sample_data.get("question", "")
        ground_truth = sample_data.get("answer")
        family = sample_data.get("family")
        is_view_dep = sample_data.get("is_view_dependent", True)
        tags = tuple(sample_data.get("tags", []))
        obj_indices = sample_data.get("object_indices", [])

        if len(obj_indices) == 2:
            idx_a, idx_b = obj_indices[0], obj_indices[1]
            c_a, c_b = min(idx_a, idx_b), max(idx_a, idx_b)
            canonical_pair = None
            for p in truth.pairs:
                if p.object_index_a == c_a and p.object_index_b == c_b:
                    canonical_pair = p
                    break
            if canonical_pair:
                order = 0 if (idx_a == c_a and idx_b == c_b) else 1
                relevant_pair_truth = adapt_presentation_order(canonical_pair, order)
    else:
        question = f"Visual observation from {view_id} viewpoint of {scene_id}."
        ground_truth = None
        family = None
        is_view_dep = None
        tags = ()

    # Match prediction
    if sample_id and ground_truth:
        prediction = repo.match_prediction(
            sample_id=sample_id,
            expected_scene_id=scene_id,
            expected_view_id=view_id,
            expected_ground_truth=ground_truth,
            run_id=run_id,
        )
    else:
        prediction = PredictionChannel(
            status="unavailable",
            provenance="no_sample_or_ground_truth",
        )

    runtime = repo.load_runtime_telemetry(run_id=run_id)

    challenge_metadata = repo.load_challenge_metadata(scene_id)

    snapshot = InspectorSnapshot(
        identity=InspectorIdentity(
            scene_id=scene_id,
            view_id=view_id,
            sample_id=sample_id,
            run_id=run_id,
        ),
        environment_truth=EnvironmentTruthChannel(
            scene_id=scene_id,
            seed=scene_state.seed,
            objects=raw_scene_objects,
            camera_object_truths=camera_object_truths,
            relevant_pair_truth=relevant_pair_truth,
            all_pairs_truth=all_pairs,
            challenge_metadata=challenge_metadata,
        ),
        camera=CameraChannel(
            pose={
                "position": list(camera_pose.position),
                "look_at": list(camera_pose.look_at),
                "up": list(camera_pose.up),
                "fov_deg": camera_pose.fov_deg,
            },
            projection=projection.to_dict(),
            provenance=projection.provenance,
        ),
        observation=ObservationChannel(
            image_path=rel_img_path or "",
            image_url=img_url,
            scene_id=scene_id,
            view_id=view_id,
        ),
        model_input=ModelInputChannel(
            image_path=rel_img_path or "",
            image_url=img_url if img_url else None,
            question=question,
        ),
        supervision=SupervisionChannel(
            ground_truth=ground_truth,
            family=family,
            is_view_dependent=is_view_dep,
            tags=tags,
        ),
        prediction=prediction,
        runtime=runtime,
    )

    # Enforce boundary: verify no leaks in model_input
    validate_model_input_isolation(snapshot.to_dict())

    return snapshot
