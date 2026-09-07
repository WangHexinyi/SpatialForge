"""Blender rendering entrypoint for G2.0-E curriculum camera plans.

Usage (from repo root):
    # Single scene:
    /root/autodl-tmp/tools/blender/blender -b -P scripts/render_curriculum.py -- \
        --scene outputs/scenes/scene_000.json --plan primary_d

    # Batch render all scenes:
    /root/autodl-tmp/tools/blender/blender -b -P scripts/render_curriculum.py -- \
        --scene-dir outputs/scenes --plan primary_d --count 100

Layout:
    outputs/experiments/g2.0-e/rendered/{scene_id}/
        {view_id}.png
        manifest.json

Supported plans:
    primary_a: south only (1 view)
    primary_c: 4 horizontal cardinal cameras (south, east, north, west)
    primary_d: 4 cardinal cameras + 4 deterministic bounded jitter cameras (8 views)

Leaves scripts/render_multiview.py completely unchanged as the closed G2.0-B
regression reference.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import bpy
import mathutils

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from spatialforge.environment.camera import CameraPose  # noqa: E402
from spatialforge.environment.geometry import camera_basis  # noqa: E402
from spatialforge.environment.observation import ObservationMetadata  # noqa: E402
from spatialforge.environment.sampling import jitter_pose  # noqa: E402
from spatialforge.environment.scene import load_scene_state  # noqa: E402
from spatialforge.environment.views import generate_cardinal_views, VIEW_IDS  # noqa: E402

# Colour map matching v1 convention exactly
COLORS = {
    "red": (0.8, 0.05, 0.05, 1),
    "blue": (0.05, 0.1, 0.8, 1),
    "green": (0.05, 0.6, 0.1, 1),
    "yellow": (0.9, 0.8, 0.05, 1),
    "purple": (0.5, 0.1, 0.7, 1),
    "cyan": (0.05, 0.7, 0.7, 1),
}


def build_camera_plan(plan_name: str, scene_seed: int = 0) -> List[Tuple[str, CameraPose]]:
    """Construct deterministic (view_id, CameraPose) sequence for a named plan."""
    cardinal_cams = generate_cardinal_views()
    cardinal_pairs = list(zip(VIEW_IDS, cardinal_cams))  # south, east, north, west

    if plan_name == "primary_a":
        # Group A: South view only
        return [cardinal_pairs[0]]

    elif plan_name == "primary_c":
        # Group C: 4 horizontal cardinal views
        return cardinal_pairs

    elif plan_name == "primary_d":
        # Group D: 4 cardinal views + 4 deterministic bounded jitter views
        # angular_deg = 15, radius_frac = 0.10, target_radius = 0.0
        # Deterministic view/scene-specific seed: scene_seed * 1000 + view_index * 17 + 42
        plan = list(cardinal_pairs)
        for idx, (view_id, cam) in enumerate(cardinal_pairs):
            jitter_seed = scene_seed * 1000 + idx * 17 + 42
            j_cam = jitter_pose(
                cam,
                seed=jitter_seed,
                angular_deg=15.0,
                radius_frac=0.10,
                target_radius=0.0,
            )
            plan.append((f"{view_id}_jitter", j_cam))
        return plan

    else:
        raise ValueError(f"Unknown camera plan: {plan_name!r}. Expected primary_a, primary_c, primary_d.")


def _clear_scene() -> None:
    """Remove all objects from the default scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _build_scene(scene_state) -> None:
    """Create Blender mesh objects from a SceneState matching G2.0-B setup."""
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

    Matches G2.0-B implementation:
        Blender camera looks along -Z (local +Z = -forward)
        Blender local +Y = up
    """
    right, up, forward = camera_basis(camera)
    blender_forward = (-forward[0], -forward[1], -forward[2])

    rot_matrix = mathutils.Matrix((
        (right[0], up[0], blender_forward[0]),
        (right[1], up[1], blender_forward[1]),
        (right[2], up[2], blender_forward[2]),
    ))
    return rot_matrix.to_euler("XYZ")


def _configure_render() -> None:
    """Set up Workbench rendering matching G2.0-B."""
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.color_type = "OBJECT"
    sc.render.resolution_x = 512
    sc.render.resolution_y = 512


def render_scene_cameras(
    scene_path: Path,
    camera_plan: List[Tuple[str, CameraPose]],
    out_dir: Path,
) -> Path:
    """Render a scene under a specified camera plan into an isolated scene subdirectory."""
    scene_state = load_scene_state(scene_path)
    scene_dir = out_dir / scene_state.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    _build_scene(scene_state)
    _configure_render()

    sc = bpy.context.scene
    observations = []

    for view_id, camera in camera_plan:
        cam_data = bpy.data.cameras.new(f"cam_{view_id}")
        cam_data.lens = 35
        cam_obj = bpy.data.objects.new(f"cam_{view_id}", cam_data)
        bpy.context.collection.objects.link(cam_obj)

        cam_obj.location = mathutils.Vector(camera.position)
        cam_obj.rotation_euler = _pose_to_blender_rotation(camera)
        sc.camera = cam_obj

        image_name = f"{view_id}.png"
        image_path = scene_dir / image_name
        sc.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)

        rel_path = Path(image_path).resolve().relative_to(Path(_REPO_ROOT).resolve()).as_posix()
        obs = ObservationMetadata(
            scene_id=scene_state.scene_id,
            view_id=view_id,
            image_path=rel_path,
            camera=camera,
        )
        observations.append(obs)

        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data)

    manifest_path = scene_dir / "manifest.json"

    # Merge or update existing manifest if present
    existing_observations = {}
    if manifest_path.exists():
        try:
            old_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in old_data.get("observations", []):
                existing_observations[item["view_id"]] = item
        except Exception:
            pass

    for obs in observations:
        existing_observations[obs.view_id] = obs.to_dict()

    manifest_data = {
        "scene_id": scene_state.scene_id,
        "seed": scene_state.seed,
        "observations": list(existing_observations.values()),
    }
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args():
    raw_args = sys.argv
    if "--" in raw_args:
        args_to_parse = raw_args[raw_args.index("--") + 1 :]
    else:
        args_to_parse = []

    parser = argparse.ArgumentParser(description="Render curriculum camera plans.")
    parser.add_argument("--scene", type=Path, default=None, help="Path to single scene JSON")
    parser.add_argument("--scene-dir", type=Path, default=None, help="Directory of scene JSONs for batch rendering")
    parser.add_argument("--count", type=int, default=100, help="Number of scenes to render if scene-dir is specified")
    parser.add_argument("--start", type=int, default=0, help="Starting scene index for batch render")
    parser.add_argument("--plan", type=str, default=None, choices=["primary_a", "primary_c", "primary_d"])
    parser.add_argument("--out-dir", type=Path, default=None, help="Root rendered output directory")
    parsed = parser.parse_args(args_to_parse)

    # Fallbacks to environment variables
    scene_env = os.environ.get("SCENE_JSON")
    scene_dir_env = os.environ.get("SCENE_DIR")
    plan_env = os.environ.get("PLAN_NAME", "primary_d")
    out_dir_env = os.environ.get("OUT_DIR", "outputs/experiments/g2.0-e/rendered")

    plan_name = parsed.plan or plan_env
    out_dir = parsed.out_dir or Path(out_dir_env)

    scene_path = parsed.scene or (Path(scene_env) if scene_env else None)
    scene_dir = parsed.scene_dir or (Path(scene_dir_env) if scene_dir_env else None)

    return scene_path, scene_dir, parsed.start, parsed.count, plan_name, out_dir


def main():
    scene_path, scene_dir, start_idx, count, plan_name, out_dir = parse_args()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if scene_dir:
        # Batch rendering mode
        scene_dir = Path(scene_dir)
        scene_files = sorted(scene_dir.glob("scene_*.json"))[start_idx : start_idx + count]
        if not scene_files:
            print(f"[error] no scene files found in {scene_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"[info] batch rendering {len(scene_files)} scenes with plan {plan_name} into {out_dir}")
        for i, s_path in enumerate(scene_files):
            scene_state = load_scene_state(s_path)
            camera_plan = build_camera_plan(plan_name, scene_seed=scene_state.seed)
            manifest_path = render_scene_cameras(s_path, camera_plan, out_dir)
            if (i + 1) % 10 == 0 or i == len(scene_files) - 1:
                print(f"[{i+1}/{len(scene_files)}] rendered {scene_state.scene_id} ({len(camera_plan)} views)")
    else:
        # Single scene mode
        s_path = scene_path or Path(_REPO_ROOT) / "outputs" / "scenes" / "scene_000.json"
        if not s_path.exists():
            print(f"[error] scene file not found: {s_path}", file=sys.stderr)
            sys.exit(1)

        scene_state = load_scene_state(s_path)
        camera_plan = build_camera_plan(plan_name, scene_seed=scene_state.seed)
        print(f"[info] rendering scene {scene_state.scene_id} with plan {plan_name} ({len(camera_plan)} views)")
        manifest_path = render_scene_cameras(s_path, camera_plan, out_dir)
        print(f"[ok] rendered {len(camera_plan)} views -> manifest {manifest_path}")


if __name__ == "__main__":
    main()
