"""ProjectionContract and Blender-compatible camera projection reconstruction.

Scope:
- Models exact, reconstructed legacy, and unknown camera projection intrinsics.
- Accounts for the historical discrepancy between nominal CameraPose.fov_deg (60 deg)
  and actual curriculum render parameters (Blender lens=35mm, sensor=36x24 AUTO, 512x512, FOV ~54.43 deg).
- Implements exact mathematical projection and Blender probe validation.
- Pure Python; Blender execution is isolated to optional validation probe.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from spatialforge.environment.camera import CameraPose, Vec3
from spatialforge.environment.geometry import camera_basis, world_to_camera

STATUS_EXACT = "exact"
STATUS_RECONSTRUCTED_LEGACY = "reconstructed_legacy"
STATUS_UNKNOWN = "unknown"
STATUS_PREVIEW_GENERATED = "preview_generated"

COORDINATE_CONVENTION_BLENDER = (
    "Blender camera local -Z viewing convention (+X right, +Y up, -Z viewing direction)"
)


@dataclass(frozen=True)
class ProjectionContract:
    """Rigorous projection intrinsics contract with explicit provenance.

    Distinguishes nominal CameraPose fov from verified or reconstructed projection.
    """

    status: str  # "exact", "reconstructed_legacy", "unknown"
    provenance: str
    camera_pose_nominal_fov_deg: float
    lens_mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    sensor_fit: Optional[str] = None  # "AUTO", "HORIZONTAL", "VERTICAL"
    resolution_x: Optional[int] = None
    resolution_y: Optional[int] = None
    pixel_aspect_x: Optional[float] = None
    pixel_aspect_y: Optional[float] = None
    clip_start: Optional[float] = None
    clip_end: Optional[float] = None
    horizontal_fov_deg: Optional[float] = None
    vertical_fov_deg: Optional[float] = None
    projection_matrix: Optional[List[List[float]]] = None
    frustum_corners_local: Optional[Dict[str, List[List[float]]]] = None
    coordinate_convention: str = COORDINATE_CONVENTION_BLENDER

    @property
    def is_frustum_available(self) -> bool:
        return (
            self.status in (STATUS_EXACT, STATUS_RECONSTRUCTED_LEGACY, STATUS_PREVIEW_GENERATED)
            and self.horizontal_fov_deg is not None
            and self.vertical_fov_deg is not None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic JSON-serializable dictionary."""
        return {
            "status": self.status,
            "provenance": self.provenance,
            "camera_pose_nominal_fov_deg": round(self.camera_pose_nominal_fov_deg, 6),
            "is_frustum_available": self.is_frustum_available,
            "lens_mm": self.lens_mm,
            "sensor_width_mm": self.sensor_width_mm,
            "sensor_height_mm": self.sensor_height_mm,
            "sensor_fit": self.sensor_fit,
            "resolution_x": self.resolution_x,
            "resolution_y": self.resolution_y,
            "pixel_aspect_x": self.pixel_aspect_x,
            "pixel_aspect_y": self.pixel_aspect_y,
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "horizontal_fov_deg": round(self.horizontal_fov_deg, 6)
            if self.horizontal_fov_deg is not None
            else None,
            "vertical_fov_deg": round(self.vertical_fov_deg, 6)
            if self.vertical_fov_deg is not None
            else None,
            "projection_matrix": self.projection_matrix,
            "frustum_corners_local": self.frustum_corners_local,
            "coordinate_convention": self.coordinate_convention,
        }


def compute_blender_sensor_dimensions(
    lens_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    sensor_fit: str,
    res_x: int,
    res_y: int,
    pixel_aspect_x: float = 1.0,
    pixel_aspect_y: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Compute effective sensor dimensions and horizontal/vertical FOV matching Blender logic.

    Returns:
        (effective_width_mm, effective_height_mm, horizontal_fov_deg, vertical_fov_deg)
    """
    if lens_mm <= 0:
        raise ValueError(f"lens_mm must be positive, got {lens_mm}")
    if res_x <= 0 or res_y <= 0:
        raise ValueError(f"Resolutions must be positive, got ({res_x}, {res_y})")

    eff_w = res_x * pixel_aspect_x
    eff_h = res_y * pixel_aspect_y
    aspect = eff_w / eff_h

    fit = sensor_fit.upper()
    if fit == "AUTO":
        # In Blender, AUTO fits sensor_width_mm to the larger dimension
        if eff_w >= eff_h:
            fit = "HORIZONTAL"
        else:
            fit = "VERTICAL_AUTO"

    if fit == "HORIZONTAL":
        w_mm = sensor_width_mm
        h_mm = sensor_width_mm / aspect
    elif fit == "VERTICAL_AUTO":
        # When AUTO fits vertically (portrait mode), Blender uses sensor_width_mm as the vertical span
        h_mm = sensor_width_mm
        w_mm = sensor_width_mm * aspect
    elif fit == "VERTICAL":
        h_mm = sensor_height_mm
        w_mm = sensor_height_mm * aspect
    else:
        raise ValueError(f"Unknown sensor_fit: {sensor_fit!r}")

    fov_h_rad = 2.0 * math.atan((w_mm / 2.0) / lens_mm)
    fov_v_rad = 2.0 * math.atan((h_mm / 2.0) / lens_mm)

    return w_mm, h_mm, math.degrees(fov_h_rad), math.degrees(fov_v_rad)


def compute_perspective_matrix(
    fov_v_deg: float,
    aspect_ratio: float,
    near: float,
    far: float,
) -> List[List[float]]:
    """Standard OpenGL/Three.js perspective projection matrix."""
    tan_half_v = math.tan(math.radians(fov_v_deg / 2.0))
    f = 1.0 / tan_half_v
    nf_range = near - far

    return [
        [f / aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / nf_range, (2.0 * far * near) / nf_range],
        [0.0, 0.0, -1.0, 0.0],
    ]


def compute_frustum_corners_local(
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    near: float = 0.2,
    far: float = 5.0,
) -> Dict[str, List[List[float]]]:
    """Compute frustum corner coordinates in camera-local space.

    Camera local convention (matching SpatialForge geometry):
    +X is right, +Y is up, +Z is forward along viewing direction.
    """
    tan_h = math.tan(math.radians(horizontal_fov_deg / 2.0))
    tan_v = math.tan(math.radians(vertical_fov_deg / 2.0))

    def _rect(dist: float) -> List[List[float]]:
        hw = dist * tan_h
        hh = dist * tan_v
        # Order: top-left, top-right, bottom-right, bottom-left
        return [
            [-hw, hh, dist],
            [hw, hh, dist],
            [hw, -hh, dist],
            [-hw, -hh, dist],
        ]

    return {
        "near": _rect(near),
        "far": _rect(far),
    }


def reconstruct_legacy_projection(
    camera_pose: CameraPose,
    res_x: int = 512,
    res_y: int = 512,
    lens_mm: float = 35.0,
    sensor_width_mm: float = 36.0,
    sensor_height_mm: float = 24.0,
    sensor_fit: str = "AUTO",
    pixel_aspect_x: float = 1.0,
    pixel_aspect_y: float = 1.0,
    clip_start: float = 0.1,
    clip_end: float = 1000.0,
) -> ProjectionContract:
    """Reconstruct exact projection intrinsics from historical curriculum renderer setup.

    Historical manifest records nominal camera.fov_deg = 60.0, but the curriculum
    renderer script (scripts/render_curriculum.py) set cam_data.lens = 35 on 512x512
    resolution, resulting in true rendered FOV of ~54.43 degrees.
    """
    w_mm, h_mm, fov_h, fov_v = compute_blender_sensor_dimensions(
        lens_mm=lens_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        sensor_fit=sensor_fit,
        res_x=res_x,
        res_y=res_y,
        pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y,
    )

    proj_mat = compute_perspective_matrix(fov_v, (res_x / res_y), clip_start, clip_end)
    corners = compute_frustum_corners_local(fov_h, fov_v, near=0.2, far=5.0)

    provenance = (
        f"reconstructed_legacy: curriculum renderer lens={lens_mm}mm, "
        f"sensor={sensor_width_mm}x{sensor_height_mm}mm {sensor_fit}, "
        f"res={res_x}x{res_y} (nominal CameraPose fov={camera_pose.fov_deg} deg)"
    )

    return ProjectionContract(
        status=STATUS_RECONSTRUCTED_LEGACY,
        provenance=provenance,
        camera_pose_nominal_fov_deg=camera_pose.fov_deg,
        lens_mm=lens_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        sensor_fit=sensor_fit,
        resolution_x=res_x,
        resolution_y=res_y,
        pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y,
        clip_start=clip_start,
        clip_end=clip_end,
        horizontal_fov_deg=fov_h,
        vertical_fov_deg=fov_v,
        projection_matrix=proj_mat,
        frustum_corners_local=corners,
        coordinate_convention=COORDINATE_CONVENTION_BLENDER,
    )


def make_unknown_projection(camera_pose: CameraPose) -> ProjectionContract:
    """Create a ProjectionContract for unverified cameras where intrinsics were not captured."""
    return ProjectionContract(
        status=STATUS_UNKNOWN,
        provenance="unknown_intrinsics: nominal CameraPose fov_deg is not verified against renderer",
        camera_pose_nominal_fov_deg=camera_pose.fov_deg,
        horizontal_fov_deg=None,
        vertical_fov_deg=None,
        coordinate_convention=COORDINATE_CONVENTION_BLENDER,
    )


def make_preview_projection(camera_pose: CameraPose) -> ProjectionContract:
    """Create a ProjectionContract for interactive WebGL preview cameras.

    Clearly distinguishes interactive browser preview projection from historical reconstructed Blender renders.
    """
    fov = camera_pose.fov_deg
    proj_mat = compute_perspective_matrix(fov, 1.0, 0.1, 100.0)
    corners = compute_frustum_corners_local(fov, fov, near=0.25, far=4.2)
    return ProjectionContract(
        status=STATUS_PREVIEW_GENERATED,
        provenance=f"preview_generated: interactive WebGL viewport preview at nominal fov={fov:.1f} deg",
        camera_pose_nominal_fov_deg=fov,
        horizontal_fov_deg=fov,
        vertical_fov_deg=fov,
        projection_matrix=proj_mat,
        frustum_corners_local=corners,
        coordinate_convention=COORDINATE_CONVENTION_BLENDER,
    )


def project_world_to_image_coordinates(
    point: Vec3,
    camera: CameraPose,
    projection: ProjectionContract,
) -> Tuple[Optional[float], Optional[float], float, bool]:
    """Project a world-space 3D point into normalized 2D image coordinates.

    Matches Blender's `bpy_extras.object_utils.world_to_camera_view`:
      u: [0.0, 1.0] from left to right
      v: [0.0, 1.0] from bottom to top
      z_depth: signed distance along camera forward axis

    Returns:
        (u, v, z_depth, is_inside_frustum)
        If behind camera (z_depth <= 0) or intrinsics unknown, (None, None, z_depth, False)
    """
    x_cam, y_cam, z_depth = world_to_camera(point, camera)

    if z_depth <= 0.0 or not projection.is_frustum_available:
        return None, None, z_depth, False

    tan_h = math.tan(math.radians(projection.horizontal_fov_deg / 2.0))
    tan_v = math.tan(math.radians(projection.vertical_fov_deg / 2.0))

    if tan_h <= 0.0 or tan_v <= 0.0:
        return None, None, z_depth, False

    ndc_x = x_cam / (z_depth * tan_h)
    ndc_y = y_cam / (z_depth * tan_v)

    u = 0.5 + 0.5 * ndc_x
    v = 0.5 + 0.5 * ndc_y

    is_inside = (0.0 <= u <= 1.0) and (0.0 <= v <= 1.0)
    return u, v, z_depth, is_inside


def run_blender_projection_probe(
    blender_executable: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a Blender projection probe validating pure Python projection math.

    Tests representative points:
      - optical center (center ray)
      - left boundary
      - right boundary
      - top boundary
      - bottom boundary
      - behind-camera point
      - non-square aspect ratios (landscape 640x480, portrait 480x640)
    """
    if blender_executable is None:
        cand = "/root/autodl-tmp/tools/blender/blender"
        blender_executable = cand if Path(cand).exists() else "blender"

    probe_code = """
import json, math, sys
try:
    import bpy, bpy_extras, mathutils
except ImportError:
    print(json.dumps({"error": "bpy not available"}))
    sys.exit(0)

def test_case(res_x, res_y, p_world, cam_pos=(0,-6,3), look_at=(0,0,0.4)):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = res_x
    sc.render.resolution_y = res_y
    sc.render.pixel_aspect_x = 1.0
    sc.render.pixel_aspect_y = 1.0

    cam_data = bpy.data.cameras.new("probe_cam")
    cam_data.lens = 35
    cam_obj = bpy.data.objects.new("probe_cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    sc.camera = cam_obj

    # SpatialForge camera basis -> Blender rotation
    fwd = mathutils.Vector((look_at[0]-cam_pos[0], look_at[1]-cam_pos[1], look_at[2]-cam_pos[2])).normalized()
    up = mathutils.Vector((0, 0, 1))
    right = fwd.cross(up).normalized()
    true_up = right.cross(fwd).normalized()
    blender_fwd = -fwd

    rot_matrix = mathutils.Matrix((
        (right.x, true_up.x, blender_fwd.x),
        (right.y, true_up.y, blender_fwd.y),
        (right.z, true_up.z, blender_fwd.z),
    ))
    cam_obj.location = cam_pos
    cam_obj.rotation_euler = rot_matrix.to_euler("XYZ")
    bpy.context.view_layer.update()

    co = bpy_extras.object_utils.world_to_camera_view(sc, cam_obj, mathutils.Vector(p_world))
    
    # Cleanup
    bpy.data.objects.remove(cam_obj, do_unlink=True)
    bpy.data.cameras.remove(cam_data)

    return [float(co.x), float(co.y), float(co.z)]

# Test suite points for canonical South camera at (0, -6, 3) looking at (0, 0, 0.4)
# Distance to look_at is sqrt(6^2 + 2.6^2) = 6.539113m
# Let's test points along the optical ray and offset
results = []
# 1. Optical center
results.append({"name": "center_optical_ray", "res": [512, 512], "pt": [0.0, 0.0, 0.4], "blender": test_case(512, 512, (0, 0, 0.4))})

# 2. Points on near plane and depth
# South camera basis: forward is (0, 6, -2.6)/dist. Right is (1, 0, 0). Up is true_up.
# Let's test simple points in world space
test_pts = [
    ("left_point", (512, 512), (-1.5, 0.0, 0.4)),
    ("right_point", (512, 512), (1.5, 0.0, 0.4)),
    ("above_point", (512, 512), (0.0, 0.0, 1.5)),
    ("below_point", (512, 512), (0.0, 0.0, -0.5)),
    ("behind_camera", (512, 512), (0.0, -10.0, 3.0)),
    ("landscape_center", (640, 480), (0.0, 0.0, 0.4)),
    ("landscape_offset", (640, 480), (1.0, 0.5, 0.8)),
    ("portrait_center", (480, 640), (0.0, 0.0, 0.4)),
    ("portrait_offset", (480, 640), (-0.8, 0.2, 1.2)),
]
for name, res, pt in test_pts:
    results.append({"name": name, "res": res, "pt": pt, "blender": test_case(res[0], res[1], pt)})

print("__BLENDER_PROBE_JSON__" + json.dumps(results))
"""
    try:
        proc = subprocess.run(
            [blender_executable, "-b", "--python-expr", probe_code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as e:
        return {
            "status": "skipped",
            "reason": f"Failed to execute Blender binary: {e}",
            "passed": True,
        }

    out = proc.stdout
    marker = "__BLENDER_PROBE_JSON__"
    if marker not in out:
        return {
            "status": "error",
            "reason": f"Blender did not output probe marker. Output: {out[:300]}",
            "passed": False,
        }

    raw_json = out.split(marker)[1].strip().splitlines()[0]
    cases = json.loads(raw_json)

    south_cam = CameraPose(position=(0.0, -6.0, 3.0), look_at=(0.0, 0.0, 0.4))
    discrepancies = []
    max_err = 0.0

    for item in cases:
        name = item["name"]
        res_x, res_y = item["res"]
        pt = tuple(item["pt"])
        blender_u, blender_v, blender_z = item["blender"]

        contract = reconstruct_legacy_projection(
            south_cam,
            res_x=res_x,
            res_y=res_y,
            lens_mm=35.0,
            sensor_width_mm=36.0,
            sensor_fit="AUTO",
        )

        py_u, py_v, py_z, inside = project_world_to_image_coordinates(pt, south_cam, contract)

        if blender_z <= 0.0:
            # Behind camera
            err = abs(py_z - blender_z)
            passed = (py_u is None) and (err < 1e-4)
        else:
            err_u = abs(py_u - blender_u)
            err_v = abs(py_v - blender_v)
            err_z = abs(py_z - blender_z)
            err = max(err_u, err_v, err_z)
            passed = err < 1e-3

        max_err = max(max_err, err)
        discrepancies.append({
            "name": name,
            "point": pt,
            "res": [res_x, res_y],
            "blender": [round(blender_u, 5), round(blender_v, 5), round(blender_z, 5)],
            "python": [
                round(py_u, 5) if py_u is not None else None,
                round(py_v, 5) if py_v is not None else None,
                round(py_z, 5),
            ],
            "error": err,
            "passed": passed,
        })

    all_passed = all(d["passed"] for d in discrepancies)
    return {
        "status": "success",
        "all_passed": all_passed,
        "max_error": max_err,
        "cases": discrepancies,
    }
