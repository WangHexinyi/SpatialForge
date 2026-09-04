"""Blender rendering entrypoint for multi-view scene rendering.

Usage (from repo root):
    blender -b -P scripts/render_multiview.py

Environment variables:
    SYNTH_SEED   random seed (default: 0)
    SYNTH_IDX    scene index like 003 (default: 000)
    SCENE_JSON   path to a v1 scene JSON (overrides SYNTH_SEED/SYNTH_IDX)

This script may import bpy. It is NOT part of the CPU-side environment core.

Coordinate adapter note:
    SpatialForge camera-local convention:
        +X = right, +Y = up, +Z = forward
    Blender camera convention:
        local -Z = forward (the direction the camera looks)
        local +Y = up

    We convert by computing SpatialForge's (right, up, forward) basis from
    CameraPose, then deriving Blender Euler rotation from those axes.
"""

import json
import os
import sys
from pathlib import Path

import bpy
import mathutils

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so we can import spatialforge modules.
# This allows the Blender subprocess to reach the CPU-side environment code.
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from spatialforge.environment.scene import load_scene_state  # noqa: E402
from spatialforge.environment.views import generate_cardinal_views, VIEW_IDS  # noqa: E402
from spatialforge.environment.geometry import camera_basis  # noqa: E402
from spatialforge.environment.camera import CameraPose  # noqa: E402
from spatialforge.environment.observation import ObservationMetadata  # noqa: E402


# ---------------------------------------------------------------------------
# Colour map matching v1 convention
# ---------------------------------------------------------------------------
COLORS = {
    "red": (0.8, 0.05, 0.05, 1),
    "blue": (0.05, 0.1, 0.8, 1),
    "green": (0.05, 0.6, 0.1, 1),
    "yellow": (0.9, 0.8, 0.05, 1),
    "purple": (0.5, 0.1, 0.7, 1),
    "cyan": (0.05, 0.7, 0.7, 1),
}


def _clear_scene():
    """Remove all objects from the default scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _build_scene(scene_state):
    """Create Blender mesh objects from a SceneState."""
    _clear_scene()

    # Ground plane
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    bpy.context.active_object.color = (0.85, 0.85, 0.85, 1)

    for obj in scene_state.objects:
        loc = obj.location
        s = obj.size
        if obj.shape == "cube":
            bpy.ops.mesh.primitive_cube_add(size=s, location=(loc[0], loc[1], loc[2]))
        elif obj.shape == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=s / 2, location=(loc[0], loc[1], loc[2])
            )
        else:
            bpy.ops.mesh.primitive_cylinder_add(
                radius=s / 2, depth=s, location=(loc[0], loc[1], loc[2])
            )
        blender_obj = bpy.context.active_object
        blender_obj.name = obj.name
        blender_obj.color = COLORS.get(obj.color, (0.5, 0.5, 0.5, 1))


def _pose_to_blender_rotation(camera: CameraPose):
    """Convert a CameraPose to a Blender Euler rotation.

    SpatialForge basis (from camera_basis):
        right  = local +X
        up     = local +Y
        forward = local +Z

    Blender camera orientation:
        local -Z = forward
        local +Y = up

    We build a 3x3 rotation matrix whose columns are the world-space
    vectors that Blender's local axes should align to:
        column 0 (Blender local +X) = SpatialForge right
        column 1 (Blender local +Y) = SpatialForge up  (same in both)
        column 2 (Blender local +Z) = -SpatialForge forward  (Blender -Z = forward)

    Then convert to Euler angles.
    """
    right, up, forward = camera_basis(camera)

    # Blender's camera looks along -Z, so local +Z = -forward
    blender_forward = (-forward[0], -forward[1], -forward[2])

    # Build 3x3 rotation matrix (columns = local axes in world space)
    rot_matrix = mathutils.Matrix((
        (right[0], up[0], blender_forward[0]),
        (right[1], up[1], blender_forward[1]),
        (right[2], up[2], blender_forward[2]),
    ))

    return rot_matrix.to_euler("XYZ")


def _configure_render():
    """Set up Workbench rendering (matching v1 convention)."""
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.color_type = "OBJECT"
    sc.render.resolution_x = 512
    sc.render.resolution_y = 512


def main():
    # Determine scene source
    scene_json_env = os.environ.get("SCENE_JSON")
    if scene_json_env:
        scene_path = Path(scene_json_env)
    else:
        seed = int(os.environ.get("SYNTH_SEED", "0"))
        idx = os.environ.get("SYNTH_IDX", "000")
        scene_path = Path(_REPO_ROOT) / "outputs" / "synth" / f"scene_{idx}.json"

    if not scene_path.exists():
        print(f"[error] scene file not found: {scene_path}", file=sys.stderr)
        sys.exit(1)

    scene_state = load_scene_state(scene_path)
    print(f"[info] loaded scene {scene_state.scene_id} with {len(scene_state.objects)} objects")

    # Build Blender objects once
    _build_scene(scene_state)
    _configure_render()

    # Camera set
    cameras = generate_cardinal_views()

    # Output directory
    out_dir = Path(_REPO_ROOT) / "outputs" / "synth"
    out_dir.mkdir(parents=True, exist_ok=True)

    observations = []

    for view_id, camera in zip(VIEW_IDS, cameras):
        # Create a Blender camera object
        cam_data = bpy.data.cameras.new(f"cam_{view_id}")
        cam_data.lens = 35  # focal length (arbitrary for workbench)
        cam_obj = bpy.data.objects.new(f"cam_{view_id}", cam_data)
        bpy.context.collection.objects.link(cam_obj)

        # Apply SpatialForge CameraPose to Blender
        # Position
        cam_obj.location = mathutils.Vector(camera.position)
        # Orientation (derived from CameraPose, not hand-guessed)
        cam_obj.rotation_euler = _pose_to_blender_rotation(camera)

        # Set as active camera
        bpy.context.scene.camera = cam_obj

        # Render
        image_name = f"{scene_state.scene_id}_view_{view_id}.png"
        image_path = out_dir / image_name
        sc = bpy.context.scene
        sc.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)

        obs = ObservationMetadata(
            scene_id=scene_state.scene_id,
            view_id=view_id,
            image_path=Path(image_path).relative_to(_REPO_ROOT).as_posix(),
            camera=camera,
        )
        observations.append(obs)

        # Remove the camera object before next iteration
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data)

        print(f"[rendered] {image_path}")

    # Write metadata manifest to manifests/ subdirectory
    manifests_dir = out_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / f"{scene_state.scene_id}_manifest.json"
    manifest = {
        "scene_id": scene_state.scene_id,
        "seed": scene_state.seed,
        "observations": [obs.to_dict() for obs in observations],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
