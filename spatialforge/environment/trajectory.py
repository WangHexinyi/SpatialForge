"""Pure deterministic experimental camera utilities for G2.1-P0.

Research Prototype:
- Adaptive Camera Framing (scene-aware bounding extent, conservative margins, occupancy presets)
- Precessing Continuous Observation Trajectory (smooth 3D multi-frequency multi-axial coverage)
- Prototype Trajectory Diagnostics (path length, azimuth/elevation/radius spans, near-repeat check)

Strict Scientific Boundary:
- Experimental prototype poses only; historical camera/render/data semantics remain immutable.
- Pure CPU Python: no bpy, zero third-party dependencies.
- Reuses existing CameraPose contracts and camera_basis / up-vector policy.
"""

from dataclasses import dataclass
import json
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from spatialforge.environment.camera import CameraPose, Vec3
from spatialforge.environment.geometry import camera_basis
from spatialforge.environment.scene import SceneObject, SceneState

_EPS = 1e-12
_UP_PARALLEL_EPS = 1e-6

# Framing presets
FRAMING_PRESET_TIGHT = "tight"
FRAMING_PRESET_BALANCED = "balanced"
FRAMING_PRESET_LOOSE = "loose"
FRAMING_PRESET_HISTORICAL = "historical_fixed"

PRESET_OCCUPANCIES: Dict[str, float] = {
    FRAMING_PRESET_LOOSE: 0.55,
    FRAMING_PRESET_BALANCED: 0.70,
    FRAMING_PRESET_TIGHT: 0.85,
}

HISTORICAL_FIXED_RADIUS = 6.0

# Distance modes
DISTANCE_MODE_HISTORICAL = "historical_fixed"
DISTANCE_MODE_ADAPTIVE = "adaptive"
DISTANCE_MODE_MANUAL = "manual"
DISTANCE_MODE_ADAPTIVE_SCALE = "adaptive_scale"

# Trajectory Families
TRAJECTORY_FAMILY_SURVEY_ORBIT = "survey_orbit"
TRAJECTORY_FAMILY_HELICAL_SWEEP = "helical_sweep"
TRAJECTORY_FAMILY_FIGURE8_CROSS = "figure8_cross"
TRAJECTORY_FAMILY_RADIAL_PROBE = "radial_probe"
TRAJECTORY_FAMILY_CUSTOM_WAYPOINTS = "custom_waypoints"
TRAJECTORY_FAMILY_PRECESSING_REFERENCE = "precessing_reference"

ALL_TRAJECTORY_FAMILIES = (
    TRAJECTORY_FAMILY_SURVEY_ORBIT,
    TRAJECTORY_FAMILY_HELICAL_SWEEP,
    TRAJECTORY_FAMILY_FIGURE8_CROSS,
    TRAJECTORY_FAMILY_RADIAL_PROBE,
    TRAJECTORY_FAMILY_CUSTOM_WAYPOINTS,
    TRAJECTORY_FAMILY_PRECESSING_REFERENCE,
)


def adjust_camera_distance(
    pose: CameraPose,
    requested_distance: float,
) -> CameraPose:
    """Move camera along target-camera ray to requested_distance while preserving look_at and ray direction.

    Mathematical formulation:
        p_new = target + normalize(p_old - target) * requested_distance
    """
    if not math.isfinite(requested_distance) or requested_distance <= 0.0:
        raise ValueError(f"requested_distance must be a finite positive number, got {requested_distance}")

    ray = _sub(pose.position, pose.look_at)
    old_dist = _norm(ray)
    if old_dist < _EPS:
        raise ValueError("camera position and look_at are coincident; ray direction is undefined")

    direction = _normalize(ray)
    new_pos = _add(
        pose.look_at,
        (direction[0] * requested_distance, direction[1] * requested_distance, direction[2] * requested_distance),
    )

    new_pose = CameraPose(
        position=new_pos,
        look_at=pose.look_at,
        up=pose.up,
        fov_deg=pose.fov_deg,
    )
    _ = camera_basis(new_pose)
    return new_pose


def parse_waypoint_text(text: str) -> Tuple[List[Vec3], Optional[List[Vec3]]]:
    """Parse multiline waypoint string into lists of positions and optional look-at targets.

    Supports lines with either:
      x, y, z
    or
      x, y, z, tx, ty, tz  (camera position + observation look-at target)
    Separated by commas, spaces, or tabs.
    Ignores blank lines and comments starting with '#'.
    Validates that coordinates are finite numbers.
    Ensures at least 2 waypoints are provided.
    """
    positions: List[Vec3] = []
    targets: List[Vec3] = []
    has_targets = False

    lines = text.strip().splitlines()
    expected_dim = None
    for line_idx, raw_line in enumerate(lines, start=1):
        line = raw_line.split("#")[0].strip()
        if not line:
            continue
        tokens = line.replace(",", " ").split()
        if len(tokens) not in (3, 6):
            raise ValueError(
                f"Line {line_idx}: expected 3 coordinates (x, y, z) or 6 (x, y, z, tx, ty, tz), got {len(tokens)} tokens: {raw_line!r}"
            )
        if expected_dim is None:
            expected_dim = len(tokens)
        elif len(tokens) != expected_dim:
            raise ValueError(
                f"Line {line_idx}: waypoint specifies {len(tokens)} coordinates, but earlier waypoints specified {expected_dim} coordinates. All waypoints must be consistent (all 3 or all 6)."
            )
        try:
            coords = [float(tok) for tok in tokens]
        except ValueError as e:
            raise ValueError(f"Line {line_idx}: could not parse float coordinate: {e}")

        for val in coords:
            if not math.isfinite(val):
                raise ValueError(f"Line {line_idx}: coordinate must be finite, got {val}")

        pos: Vec3 = (coords[0], coords[1], coords[2])
        positions.append(pos)
        if len(coords) == 6:
            target: Vec3 = (coords[3], coords[4], coords[5])
            targets.append(target)
            has_targets = True

    if len(positions) < 2:
        raise ValueError(f"At least 2 waypoints required, got {len(positions)}")

    return positions, (targets if has_targets else None)


def _catmull_rom_centripetal_segment(
    p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, alpha: float = 0.5
):
    """Factory returning an evaluator function c(u) for u in [0, 1] for a centripetal Catmull-Rom segment."""
    def _dist_alpha(a: Vec3, b: Vec3) -> float:
        d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
        return max(d2 ** (alpha * 0.5), _EPS)

    t0 = 0.0
    t1 = t0 + _dist_alpha(p0, p1)
    t2 = t1 + _dist_alpha(p1, p2)
    t3 = t2 + _dist_alpha(p2, p3)

    def _eval(u: float) -> Vec3:
        t = t1 + u * (t2 - t1)

        def _lin(va: Vec3, vb: Vec3, ta: float, tb: float) -> Vec3:
            denom = tb - ta
            if abs(denom) < _EPS:
                return va
            fa = (tb - t) / denom
            fb = (t - ta) / denom
            return (fa * va[0] + fb * vb[0], fa * va[1] + fb * vb[1], fa * va[2] + fb * vb[2])

        a1 = _lin(p0, p1, t0, t1)
        a2 = _lin(p1, p2, t1, t2)
        a3 = _lin(p2, p3, t2, t3)

        b1 = _lin(a1, a2, t0, t2)
        b2 = _lin(a2, a3, t1, t3)

        return _lin(b1, b2, t1, t2)

    return _eval


def interpolate_waypoints(
    waypoints: Sequence[Vec3],
    waypoint_targets: Optional[Sequence[Vec3]] = None,
    frame_count: int = 60,
    interpolation_mode: str = "catmull_rom",
    fov_deg: float = 60.0,
    default_target: Optional[Vec3] = None,
) -> Tuple[List[CameraPose], List[float], float]:
    """Generate smooth open camera path passing through ordered control points.

    - Uses centripetal Catmull-Rom (closed = false, curveType = 'centripetal') or linear interpolation.
    - Path start matches waypoints[0], path end matches waypoints[-1].
    - Does NOT close the path.
    - Resamples by arc length so frames have uniform spatial separation.
    - If waypoint_targets is provided, smoothly interpolates look-at targets per frame;
      otherwise targets default_target.
    Returns:
        (poses, arc_distances, total_path_length)
    """
    if len(waypoints) < 2:
        raise ValueError(f"At least 2 waypoints required, got {len(waypoints)}")
    if frame_count < 2:
        raise ValueError(f"frame_count must be >= 2, got {frame_count}")
    if waypoint_targets is not None and len(waypoint_targets) != len(waypoints):
        raise ValueError(
            f"waypoint_targets count ({len(waypoint_targets)}) must match waypoints count ({len(waypoints)})"
        )

    if default_target is None:
        default_target = (0.0, 0.0, 0.4)

    m = len(waypoints)

    def _eval_pos(u: float) -> Vec3:
        if u <= 0.0:
            return waypoints[0]
        if u >= m - 1:
            return waypoints[-1]

        seg_idx = int(u)
        if seg_idx >= m - 1:
            seg_idx = m - 2
        seg_u = u - seg_idx

        if interpolation_mode == "linear" or m == 2:
            p1 = waypoints[seg_idx]
            p2 = waypoints[seg_idx + 1]
            return (
                p1[0] + seg_u * (p2[0] - p1[0]),
                p1[1] + seg_u * (p2[1] - p1[1]),
                p1[2] + seg_u * (p2[2] - p1[2]),
            )
        else:
            p0 = (
                (2.0 * waypoints[0][0] - waypoints[1][0], 2.0 * waypoints[0][1] - waypoints[1][1], 2.0 * waypoints[0][2] - waypoints[1][2])
                if seg_idx == 0
                else waypoints[seg_idx - 1]
            )
            p1 = waypoints[seg_idx]
            p2 = waypoints[seg_idx + 1]
            p3 = (
                (2.0 * waypoints[-1][0] - waypoints[-2][0], 2.0 * waypoints[-1][1] - waypoints[-2][1], 2.0 * waypoints[-1][2] - waypoints[-2][2])
                if seg_idx + 1 == m - 1
                else waypoints[seg_idx + 2]
            )
            evaluator = _catmull_rom_centripetal_segment(p0, p1, p2, p3)
            return evaluator(seg_u)

    def _eval_target(u: float) -> Vec3:
        if waypoint_targets is None:
            return default_target
        if u <= 0.0:
            return waypoint_targets[0]
        if u >= m - 1:
            return waypoint_targets[-1]
        seg_idx = int(u)
        if seg_idx >= m - 1:
            seg_idx = m - 2
        seg_u = u - seg_idx
        t1 = waypoint_targets[seg_idx]
        t2 = waypoint_targets[seg_idx + 1]
        return (
            t1[0] + seg_u * (t2[0] - t1[0]),
            t1[1] + seg_u * (t2[1] - t1[1]),
            t1[2] + seg_u * (t2[2] - t1[2]),
        )

    # 1. Dense sampling for arc length calculation
    dense_samples_per_seg = 60
    total_dense = max(120, (m - 1) * dense_samples_per_seg)
    dense_pts: List[Vec3] = []
    dense_u: List[float] = []
    cum_dists: List[float] = [0.0]

    for step in range(total_dense + 1):
        u_val = (step / float(total_dense)) * (m - 1)
        pt = _eval_pos(u_val)
        dense_pts.append(pt)
        dense_u.append(u_val)
        if step > 0:
            d = _norm(_sub(pt, dense_pts[-2]))
            cum_dists.append(cum_dists[-1] + d)

    total_len = cum_dists[-1]
    if total_len < _EPS:
        total_len = 0.01

    # 2. Resample equidistant arc length points
    poses: List[CameraPose] = []
    arc_distances: List[float] = []
    cur_dense_idx = 0

    for i in range(frame_count):
        target_arc = (i / float(frame_count - 1)) * total_len
        arc_distances.append(target_arc)

        if i == 0:
            pos = waypoints[0]
            tgt = _eval_target(0.0)
        elif i == frame_count - 1:
            pos = waypoints[-1]
            tgt = _eval_target(m - 1.0)
        else:
            while cur_dense_idx < len(cum_dists) - 2 and cum_dists[cur_dense_idx + 1] < target_arc:
                cur_dense_idx += 1
            d_prev = cum_dists[cur_dense_idx]
            d_next = cum_dists[cur_dense_idx + 1]
            span = d_next - d_prev
            frac = (target_arc - d_prev) / span if span > _EPS else 0.0
            u_interp = dense_u[cur_dense_idx] + frac * (dense_u[cur_dense_idx + 1] - dense_u[cur_dense_idx])
            pos = _eval_pos(u_interp)
            tgt = _eval_target(u_interp)

        up = _view_up(pos, tgt)
        pose = CameraPose(position=pos, look_at=tgt, up=up, fov_deg=fov_deg)
        _ = camera_basis(pose)
        poses.append(pose)

    return poses, arc_distances, total_len


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: Vec3) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: Vec3) -> Vec3:
    length = _norm(v)
    if length < _EPS:
        raise ValueError("cannot normalize a zero-length vector")
    return (v[0] / length, v[1] / length, v[2] / length)


def _view_up(position: Vec3, look_at: Vec3) -> Vec3:
    """Deterministic up vector matching canonical camera sampling policy.

    World up is (0, 0, 1); falls back to (0, 1, 0) if viewing direction is near-parallel to world Z.
    """
    forward = _normalize(_sub(look_at, position))
    world_up = (0.0, 0.0, 1.0)
    if abs(_dot(forward, world_up)) >= 1.0 - _UP_PARALLEL_EPS:
        return (0.0, 1.0, 0.0)
    return world_up


def primitive_size_margin(shape: str, size: float) -> float:
    """Conservative bounding radius margin for standard geometric primitives.

    - Cube (edge s): circumscribed sphere radius = s * sqrt(3) / 2 ~= 0.8660 * s
    - Cylinder (radius s/2, height s): circumscribed sphere radius = s * sqrt(2) / 2 ~= 0.7071 * s
    - Sphere (diameter s, radius s/2): exact circumscribed radius = s * 0.5
    - Expanded shapes (cone, torus, octahedron, pyramid, tetrahedron, prism, capsule)
    """
    s = float(size)
    sh = shape.lower()
    if sh == "cube":
        return s * (math.sqrt(3.0) / 2.0)
    elif sh == "cylinder":
        return s * (math.sqrt(2.0) / 2.0)
    elif sh == "sphere":
        return s * 0.5
    elif sh in ("cone", "pyramid"):
        return s * (math.sqrt(2.0) / 2.0)
    elif sh in ("torus", "octahedron", "capsule"):
        return s * 0.5
    elif sh in ("tetrahedron", "prism"):
        return s * 0.65
    return s * (math.sqrt(3.0) / 2.0)


def compute_scene_extent(scene_state: SceneState) -> Tuple[Vec3, float, str]:
    """Derive scene center and conservative scene radius from authoritative SceneState.

    Empty scene handling policy:
    If scene has 0 objects, falls back to canonical center (0.0, 0.0, 0.4) with nominal radius 1.0.
    """
    if not scene_state.objects:
        return (0.0, 0.0, 0.4), 1.0, "empty_scene_fallback (center=(0.0, 0.0, 0.4), radius=1.0m)"

    n = len(scene_state.objects)
    cx = sum(obj.location[0] for obj in scene_state.objects) / n
    cy = sum(obj.location[1] for obj in scene_state.objects) / n
    cz = sum(obj.location[2] for obj in scene_state.objects) / n
    center: Vec3 = (cx, cy, cz)

    max_extent = 0.0
    for obj in scene_state.objects:
        dx = obj.location[0] - cx
        dy = obj.location[1] - cy
        dz = obj.location[2] - cz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        margin = primitive_size_margin(obj.shape, obj.size)
        extent = dist + margin
        if extent > max_extent:
            max_extent = extent

    scene_radius = max(max_extent, 0.1)
    margin_note = (
        "Conservative shape-aware circumscribed bounding radius: "
        "cube=s*sqrt(3)/2 (~0.866s), cylinder=s*sqrt(2)/2 (~0.707s), sphere=s*0.5"
    )
    return center, scene_radius, margin_note


def distance_for_occupancy(scene_radius: float, occupancy: float, fov_deg: float = 60.0) -> float:
    """Solve camera distance D given bounding radius R and target occupancy eta.

    D = R / (eta * tan(fov / 2))
    """
    if occupancy <= 0.0 or occupancy > 1.0:
        raise ValueError(f"occupancy must be in (0.0, 1.0], got {occupancy}")
    half_fov_rad = math.radians(fov_deg / 2.0)
    return scene_radius / (occupancy * math.tan(half_fov_rad))


@dataclass(frozen=True)
class AdaptiveFramingResult:
    """Diagnostic and numerical result of scene-aware adaptive camera framing."""

    scene_center: Vec3
    scene_radius: float
    framing_mode: str
    target_occupancy: float
    camera_distance: float
    fov_deg: float
    margin_assumptions: str
    presets_comparison: Dict[str, float]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_center": [round(c, 4) for c in self.scene_center],
            "scene_radius": round(self.scene_radius, 4),
            "framing_mode": self.framing_mode,
            "target_occupancy": round(self.target_occupancy, 4),
            "camera_distance": round(self.camera_distance, 4),
            "fov_deg": round(self.fov_deg, 2),
            "margin_assumptions": self.margin_assumptions,
            "presets_comparison": {k: round(v, 4) for k, v in self.presets_comparison.items()},
            "diagnostics": self.diagnostics,
        }


def compute_adaptive_framing(
    scene_state: SceneState,
    framing_mode: str = FRAMING_PRESET_BALANCED,
    target_occupancy: Optional[float] = None,
    fov_deg: float = 60.0,
) -> AdaptiveFramingResult:
    """Compute adaptive framing camera distance and diagnostics for a scene."""
    center, scene_radius, margin_note = compute_scene_extent(scene_state)

    half_fov_rad = math.radians(fov_deg / 2.0)
    tan_half_fov = math.tan(half_fov_rad)

    # Compute comparison table across all standard presets
    presets_comparison: Dict[str, float] = {
        FRAMING_PRESET_HISTORICAL: HISTORICAL_FIXED_RADIUS,
        FRAMING_PRESET_LOOSE: distance_for_occupancy(scene_radius, PRESET_OCCUPANCIES[FRAMING_PRESET_LOOSE], fov_deg),
        FRAMING_PRESET_BALANCED: distance_for_occupancy(scene_radius, PRESET_OCCUPANCIES[FRAMING_PRESET_BALANCED], fov_deg),
        FRAMING_PRESET_TIGHT: distance_for_occupancy(scene_radius, PRESET_OCCUPANCIES[FRAMING_PRESET_TIGHT], fov_deg),
    }

    if framing_mode == FRAMING_PRESET_HISTORICAL:
        effective_distance = HISTORICAL_FIXED_RADIUS
        effective_occupancy = scene_radius / (effective_distance * tan_half_fov)
    elif target_occupancy is not None:
        effective_occupancy = float(target_occupancy)
        effective_distance = distance_for_occupancy(scene_radius, effective_occupancy, fov_deg)
    elif framing_mode in PRESET_OCCUPANCIES:
        effective_occupancy = PRESET_OCCUPANCIES[framing_mode]
        effective_distance = presets_comparison[framing_mode]
    else:
        # Default fallback to balanced
        effective_occupancy = PRESET_OCCUPANCIES[FRAMING_PRESET_BALANCED]
        effective_distance = presets_comparison[FRAMING_PRESET_BALANCED]

    diagnostics = {
        "object_count": len(scene_state.objects),
        "scene_radius_m": round(scene_radius, 4),
        "tan_half_fov": round(tan_half_fov, 4),
        "historical_radius_m": HISTORICAL_FIXED_RADIUS,
        "historical_occupancy": round(scene_radius / (HISTORICAL_FIXED_RADIUS * tan_half_fov), 4),
        "distance_reduction_ratio": round(effective_distance / HISTORICAL_FIXED_RADIUS, 4),
    }

    return AdaptiveFramingResult(
        scene_center=center,
        scene_radius=scene_radius,
        framing_mode=framing_mode,
        target_occupancy=effective_occupancy,
        camera_distance=effective_distance,
        fov_deg=fov_deg,
        margin_assumptions=margin_note,
        presets_comparison=presets_comparison,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class TrajectoryConfig:
    """Deterministic configuration for continuous observation trajectory."""

    frame_count: int = 60
    seed: int = 0
    framing_mode: str = FRAMING_PRESET_BALANCED
    target_occupancy: Optional[float] = None
    base_radius: Optional[float] = None
    azimuth_cycles: float = 1.0
    azimuth_start_deg: float = 0.0
    elevation_base_deg: float = 25.0
    elevation_amplitude_deg: float = 15.0
    elevation_cycles: float = 2.5
    radius_amplitude_frac: float = 0.15
    radius_cycles: float = 1.5
    target_bias_fraction: float = 0.10
    target_bias_cycles: float = 1.0
    fov_deg: float = 60.0
    elevation_envelope: Tuple[float, float] = (-20.0, 50.0)
    trajectory_family: str = TRAJECTORY_FAMILY_PRECESSING_REFERENCE
    distance_mode: str = DISTANCE_MODE_ADAPTIVE
    manual_distance: Optional[float] = None
    distance_scale: float = 1.0
    speed_m_per_s: float = 1.0
    speed_m_s: Optional[float] = None
    waypoints: Optional[Tuple[Vec3, ...]] = None
    waypoint_targets: Optional[Tuple[Vec3, ...]] = None
    interpolation_mode: str = "catmull_rom"

    def __post_init__(self):
        if self.speed_m_s is not None:
            object.__setattr__(self, "speed_m_per_s", float(self.speed_m_s))
        else:
            object.__setattr__(self, "speed_m_s", float(self.speed_m_per_s))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "seed": self.seed,
            "framing_mode": self.framing_mode,
            "target_occupancy": self.target_occupancy,
            "base_radius": self.base_radius,
            "azimuth_cycles": self.azimuth_cycles,
            "azimuth_start_deg": self.azimuth_start_deg,
            "elevation_base_deg": self.elevation_base_deg,
            "elevation_amplitude_deg": self.elevation_amplitude_deg,
            "elevation_cycles": self.elevation_cycles,
            "radius_amplitude_frac": self.radius_amplitude_frac,
            "radius_cycles": self.radius_cycles,
            "target_bias_fraction": self.target_bias_fraction,
            "target_bias_cycles": self.target_bias_cycles,
            "fov_deg": self.fov_deg,
            "elevation_envelope": list(self.elevation_envelope),
            "trajectory_family": self.trajectory_family,
            "distance_mode": self.distance_mode,
            "manual_distance": self.manual_distance,
            "distance_scale": self.distance_scale,
            "speed_m_per_s": self.speed_m_per_s,
            "waypoints": [list(w) for w in self.waypoints] if self.waypoints else None,
            "waypoint_targets": [list(t) for t in self.waypoint_targets] if self.waypoint_targets else None,
            "interpolation_mode": self.interpolation_mode,
        }


@dataclass(frozen=True)
class TrajectoryFrame:
    """A single deterministic observation frame along a continuous trajectory."""

    frame_index: int
    normalized_time: float
    pose: CameraPose
    azimuth_deg: float
    elevation_deg: float
    distance_to_target: float
    timestamp_sec: float = 0.0
    arc_distance_m: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "normalized_time": round(self.normalized_time, 4),
            "timestamp_sec": round(self.timestamp_sec, 4),
            "arc_distance_m": round(self.arc_distance_m, 4),
            "pose": {
                "position": [round(x, 4) for x in self.pose.position],
                "look_at": [round(x, 4) for x in self.pose.look_at],
                "up": [round(x, 4) for x in self.pose.up],
                "fov_deg": round(self.pose.fov_deg, 2),
            },
            "azimuth_deg": round(self.azimuth_deg, 2),
            "elevation_deg": round(self.elevation_deg, 2),
            "distance_to_target": round(self.distance_to_target, 4),
        }


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    """Prototype diagnostics analyzing geometric coverage and near-repeat behavior."""

    frame_count: int
    azimuth_span_deg: float
    elevation_min_deg: float
    elevation_max_deg: float
    radius_min: float
    radius_max: float
    trajectory_path_length: float
    min_separation_window_frames: int
    min_position_distance_m: float
    min_angular_difference_deg: float
    near_repeat_detected: bool
    diagnostics_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "azimuth_span_deg": round(self.azimuth_span_deg, 2),
            "elevation_min_deg": round(self.elevation_min_deg, 2),
            "elevation_max_deg": round(self.elevation_max_deg, 2),
            "radius_min": round(self.radius_min, 4),
            "radius_max": round(self.radius_max, 4),
            "trajectory_path_length": round(self.trajectory_path_length, 4),
            "min_separation_window_frames": self.min_separation_window_frames,
            "min_position_distance_m": round(self.min_position_distance_m, 4),
            "min_angular_difference_deg": round(self.min_angular_difference_deg, 2),
            "near_repeat_detected": self.near_repeat_detected,
            "diagnostics_summary": self.diagnostics_summary,
        }


@dataclass(frozen=True)
class CameraTrajectory:
    """Complete container for a deterministic continuous observation trajectory."""

    trajectory_id: str
    scene_id: str
    config: TrajectoryConfig
    framing: AdaptiveFramingResult
    frames: List[TrajectoryFrame]
    diagnostics: TrajectoryDiagnostics
    trajectory_family: str = TRAJECTORY_FAMILY_PRECESSING_REFERENCE
    total_path_length_m: float = 0.0
    total_duration_sec: float = 0.0
    scientific_speed_m_s: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "scene_id": self.scene_id,
            "trajectory_family": self.trajectory_family,
            "total_path_length_m": round(self.total_path_length_m, 4),
            "total_duration_sec": round(self.total_duration_sec, 4),
            "scientific_speed_m_s": round(self.scientific_speed_m_s, 4),
            "config": self.config.to_dict(),
            "framing": self.framing.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
            "diagnostics": self.diagnostics.to_dict(),
        }


def compute_trajectory_diagnostics(
    frames: List[TrajectoryFrame],
    base_radius: float,
) -> TrajectoryDiagnostics:
    """Compute prototype trajectory diagnostics including near-repeat detection."""
    n = len(frames)
    if n == 0:
        return TrajectoryDiagnostics(
            frame_count=0,
            azimuth_span_deg=0.0,
            elevation_min_deg=0.0,
            elevation_max_deg=0.0,
            radius_min=0.0,
            radius_max=0.0,
            trajectory_path_length=0.0,
            min_separation_window_frames=0,
            min_position_distance_m=0.0,
            min_angular_difference_deg=0.0,
            near_repeat_detected=False,
            diagnostics_summary="empty trajectory",
        )

    # Elevation and radius ranges
    elevations = [f.elevation_deg for f in frames]
    radii = [f.distance_to_target for f in frames]
    elev_min = min(elevations)
    elev_max = max(elevations)
    r_min = min(radii)
    r_max = max(radii)

    # Azimuth total swept span
    az_span = 0.0
    for i in range(n - 1):
        d_az = abs(frames[i + 1].azimuth_deg - frames[i].azimuth_deg)
        if d_az > 180.0:
            d_az = 360.0 - d_az
        az_span += d_az

    # Path length
    path_len = 0.0
    for i in range(n - 1):
        p1 = frames[i].pose.position
        p2 = frames[i + 1].pose.position
        path_len += _norm(_sub(p2, p1))

    # Near-repeat analysis comparing sufficiently separated frames
    # Separation window: at least 15% of frames or 5 frames
    k_sep = max(5, int(n * 0.15))
    min_pos_dist = float("inf")
    min_ang_diff = float("inf")
    near_repeat_detected = False

    # Precompute viewing directions
    view_dirs = []
    for f in frames:
        vd = _normalize(_sub(f.pose.look_at, f.pose.position))
        view_dirs.append(vd)

    # Near-repeat thresholds: position distance < 0.25 * r_0 AND view angle < 15 deg
    pos_thresh = 0.25 * base_radius
    ang_thresh = 15.0

    for i in range(n):
        for j in range(i + k_sep, n):
            d_pos = _norm(_sub(frames[j].pose.position, frames[i].pose.position))
            dot_val = max(-1.0, min(1.0, _dot(view_dirs[i], view_dirs[j])))
            ang_diff = math.degrees(math.acos(dot_val))

            if d_pos < min_pos_dist:
                min_pos_dist = d_pos
            if ang_diff < min_ang_diff:
                min_ang_diff = ang_diff

            if d_pos < pos_thresh and ang_diff < ang_thresh:
                near_repeat_detected = True

    if min_pos_dist == float("inf"):
        min_pos_dist = 0.0
    if min_ang_diff == float("inf"):
        min_ang_diff = 0.0

    summary = (
        f"Trajectory N={n}: path={path_len:.2f}m, az_span={az_span:.1f}°, "
        f"elev=[{elev_min:.1f}°, {elev_max:.1f}°], r=[{r_min:.2f}, {r_max:.2f}]m, "
        f"near_repeat={'DETECTED' if near_repeat_detected else 'NONE'} "
        f"(min_sep_window={k_sep} frames, min_d={min_pos_dist:.2f}m, min_ang={min_ang_diff:.1f}°)"
    )

    return TrajectoryDiagnostics(
        frame_count=n,
        azimuth_span_deg=az_span,
        elevation_min_deg=elev_min,
        elevation_max_deg=elev_max,
        radius_min=r_min,
        radius_max=r_max,
        trajectory_path_length=path_len,
        min_separation_window_frames=k_sep,
        min_position_distance_m=min_pos_dist,
        min_angular_difference_deg=min_ang_diff,
        near_repeat_detected=near_repeat_detected,
        diagnostics_summary=summary,
    )


def generate_trajectory(
    scene_state: SceneState,
    config: Optional[TrajectoryConfig] = None,
) -> CameraTrajectory:
    """Generate a deterministic continuous observation trajectory across selected topological families.

    Supported Families:
    1. survey_orbit: broad horizontal survey circumscribing scene at stable elevation
    2. helical_sweep: clear elevation spiral rising continuously around scene
    3. figure8_cross: crossing side viewpoints and relation reversals across center
    4. radial_probe: deliberate approach (close probe), lateral repositioning, and retreat
    5. custom_waypoints: arbitrary user-specified open spline interpolated through 3D control points
    6. precessing_reference: legacy multi-frequency sinusoidal prototype

    Distance Modes:
    - historical_fixed: 6.0m
    - adaptive: scene-aware adaptive framing
    - manual: explicit distance in meters (config.manual_distance)
    - adaptive_scale: adaptive distance scaled by config.distance_scale
    """
    if config is None:
        config = TrajectoryConfig()

    if config.frame_count < 2:
        raise ValueError(f"frame_count must be >= 2, got {config.frame_count}")

    # Solve adaptive framing
    framing = compute_adaptive_framing(
        scene_state=scene_state,
        framing_mode=config.framing_mode,
        target_occupancy=config.target_occupancy,
        fov_deg=config.fov_deg,
    )

def resolve_camera_distance(
    distance_mode: str,
    base_adaptive_distance: float,
    manual_distance: Optional[float] = None,
    distance_scale: Optional[float] = None,
) -> float:
    """Resolve camera distance based on distance mode, base adaptive distance, and parameters."""
    if distance_mode == DISTANCE_MODE_HISTORICAL:
        return HISTORICAL_FIXED_RADIUS
    elif distance_mode == DISTANCE_MODE_MANUAL:
        if manual_distance is not None:
            if not math.isfinite(manual_distance) or manual_distance <= 0.0:
                raise ValueError(f"manual_distance must be a finite positive number, got {manual_distance}")
            return float(manual_distance)
        return float(base_adaptive_distance)
    elif distance_mode == DISTANCE_MODE_ADAPTIVE_SCALE:
        scale = distance_scale if (distance_scale is not None and distance_scale > 0.0) else 1.0
        return float(base_adaptive_distance * scale)
    else:
        return float(base_adaptive_distance)


def generate_trajectory(
    scene_state: SceneState,
    config: Optional[TrajectoryConfig] = None,
) -> CameraTrajectory:
    """Generate a continuous deterministic camera trajectory around a scene."""
    if config is None:
        config = TrajectoryConfig()

    if config.frame_count < 2:
        raise ValueError(f"frame_count must be >= 2, got {config.frame_count}")

    # Solve adaptive framing
    framing = compute_adaptive_framing(
        scene_state=scene_state,
        framing_mode=config.framing_mode,
        target_occupancy=config.target_occupancy,
        fov_deg=config.fov_deg,
    )

    # Determine base radius r_0 from distance_mode
    if config.distance_mode == DISTANCE_MODE_MANUAL and config.base_radius is not None and config.manual_distance is None:
        r_0 = float(config.base_radius)
    elif config.base_radius is not None and config.distance_mode == DISTANCE_MODE_ADAPTIVE:
        r_0 = float(config.base_radius)
    else:
        r_0 = resolve_camera_distance(
            distance_mode=config.distance_mode,
            base_adaptive_distance=framing.camera_distance,
            manual_distance=config.manual_distance,
            distance_scale=config.distance_scale,
        )

    if r_0 <= 0.05:
        r_0 = 0.05

    center = framing.scene_center
    scene_radius = framing.scene_radius
    n = config.frame_count
    v_speed = config.speed_m_per_s if (config.speed_m_per_s and config.speed_m_per_s > 0.0) else 1.0

    # Local seeded RNG
    rng = random.Random(config.seed)
    if config.seed == 0:
        delta_phi = 0.0
        delta_r = 0.0
        delta_bx = 0.0
        delta_by = math.pi / 2.0
        delta_bz = 0.0
        theta_0 = math.radians(config.azimuth_start_deg)
    else:
        theta_0 = math.radians(config.azimuth_start_deg) + rng.uniform(0.0, 2.0 * math.pi)
        delta_phi = rng.uniform(0.0, 2.0 * math.pi)
        delta_r = rng.uniform(0.0, 2.0 * math.pi)
        delta_bx = rng.uniform(0.0, 2.0 * math.pi)
        delta_by = rng.uniform(0.0, 2.0 * math.pi)
        delta_bz = rng.uniform(0.0, 2.0 * math.pi)

    elev_min_rad = math.radians(config.elevation_envelope[0])
    elev_max_rad = math.radians(config.elevation_envelope[1])
    target_bias_max = config.target_bias_fraction * scene_radius

    family = config.trajectory_family

    frames: List[TrajectoryFrame] = []

    if family == TRAJECTORY_FAMILY_CUSTOM_WAYPOINTS:
        # User-specified waypoints
        pts: List[Vec3] = list(config.waypoints) if config.waypoints else []
        targets: Optional[List[Vec3]] = list(config.waypoint_targets) if config.waypoint_targets else None
        if len(pts) < 2:
            # Fallback to a default survey waypoint sequence around scene
            p_phi = math.radians(25.0)
            pts = [
                (
                    center[0] + r_0 * math.cos(p_phi) * math.cos(ang),
                    center[1] + r_0 * math.cos(p_phi) * math.sin(ang),
                    center[2] + r_0 * math.sin(p_phi),
                )
                for ang in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0, 2.0 * math.pi)
            ]

        interp_poses, arc_dists, total_len = interpolate_waypoints(
            waypoints=pts,
            waypoint_targets=targets,
            frame_count=n,
            interpolation_mode=config.interpolation_mode,
            fov_deg=config.fov_deg,
            default_target=center,
        )

        for i, (pose, arc_d) in enumerate(zip(interp_poses, arc_dists)):
            t = i / float(n - 1)
            d_vec = _sub(pose.position, pose.look_at)
            dist_to_tgt = _norm(d_vec)
            az_rad = math.atan2(d_vec[1], d_vec[0])
            elev_rad = math.asin(max(-1.0, min(1.0, d_vec[2] / max(dist_to_tgt, _EPS))))
            t_sec = arc_d / v_speed
            frames.append(
                TrajectoryFrame(
                    frame_index=i,
                    normalized_time=t,
                    pose=pose,
                    azimuth_deg=math.degrees(az_rad) % 360.0,
                    elevation_deg=math.degrees(elev_rad),
                    distance_to_target=dist_to_tgt,
                    timestamp_sec=t_sec,
                    arc_distance_m=arc_d,
                )
            )

    else:
        # Analytic parametric paths
        for i in range(n):
            t = i / float(n - 1)

            if family == TRAJECTORY_FAMILY_SURVEY_ORBIT:
                # 1. Broad scene survey: pure circular orbit at stable elevation
                theta = theta_0 + 2.0 * math.pi * config.azimuth_cycles * t
                theta_deg = math.degrees(theta) % 360.0
                phi = math.radians(config.elevation_base_deg)
                phi = max(elev_min_rad, min(elev_max_rad, phi))
                phi_deg = math.degrees(phi)
                r = r_0
                target = center

            elif family == TRAJECTORY_FAMILY_HELICAL_SWEEP:
                # 2. Helical sweep: clear elevation progression rising around scene (spiral)
                # Azimuth sweeps 2 full revolutions (720 deg)
                theta = theta_0 + 4.0 * math.pi * t
                theta_deg = math.degrees(theta) % 360.0
                # Elevation progression from -5 deg up to +45 deg (or clamped to envelope)
                elev_span = (elev_max_rad - elev_min_rad) * 0.7
                phi = elev_min_rad * 0.3 + elev_span * t
                phi = max(elev_min_rad, min(elev_max_rad, phi))
                phi_deg = math.degrees(phi)
                r = r_0
                target = center

            elif family == TRAJECTORY_FAMILY_FIGURE8_CROSS:
                # 3. Figure-8 cross: crossing side viewpoints and relation reversals
                # Azimuth oscillates back and forth around center (+/- 60 deg)
                theta = theta_0 + math.radians(60.0) * math.sin(2.0 * math.pi * t)
                theta_deg = math.degrees(theta) % 360.0
                # Elevation oscillates at 2x frequency (+/- 16 deg) crossing center
                phi_base = math.radians(22.0)
                phi = phi_base + math.radians(16.0) * math.sin(4.0 * math.pi * t)
                phi = max(elev_min_rad, min(elev_max_rad, phi))
                phi_deg = math.degrees(phi)
                r = r_0
                target = center

            elif family == TRAJECTORY_FAMILY_RADIAL_PROBE:
                # 4. Radial probe: deliberate approach (close probe) and retreat
                theta = theta_0 + math.radians(120.0) * t
                theta_deg = math.degrees(theta) % 360.0
                phi = math.radians(config.elevation_base_deg) + math.radians(8.0) * math.sin(2.0 * math.pi * t)
                phi = max(elev_min_rad, min(elev_max_rad, phi))
                phi_deg = math.degrees(phi)
                # Radial oscillation: close approach (0.5 * r0) to retreat (1.5 * r0)
                r = r_0 * (1.0 - 0.5 * math.cos(2.0 * math.pi * 2.0 * t))
                if r <= 0.05:
                    r = 0.05
                target = center

            else:
                # 6. Precessing Reference (legacy multi-frequency sinusoidal prototype)
                theta = theta_0 + 2.0 * math.pi * config.azimuth_cycles * t
                theta_deg = math.degrees(theta) % 360.0
                phi_0 = math.radians(config.elevation_base_deg)
                a_phi = math.radians(config.elevation_amplitude_deg)
                phi = phi_0 + a_phi * math.sin(2.0 * math.pi * config.elevation_cycles * t + delta_phi)
                phi = max(elev_min_rad, min(elev_max_rad, phi))
                phi_deg = math.degrees(phi)

                a_r = config.radius_amplitude_frac
                r = r_0 * (1.0 + a_r * math.sin(2.0 * math.pi * config.radius_cycles * t + delta_r))
                if r <= 0.05:
                    r = 0.05

                kb = config.target_bias_cycles
                bias_x = target_bias_max * math.sin(2.0 * math.pi * kb * t + delta_bx)
                bias_y = target_bias_max * math.cos(2.0 * math.pi * kb * 1.3 * t + delta_by)
                bias_z = 0.5 * target_bias_max * math.sin(2.0 * math.pi * kb * 0.7 * t + delta_bz)
                target = (center[0] + bias_x, center[1] + bias_y, center[2] + bias_z)

            # Compute Cartesian coordinates relative to target
            dx = r * math.cos(phi) * math.cos(theta)
            dy = r * math.cos(phi) * math.sin(theta)
            dz = r * math.sin(phi)
            pos: Vec3 = (target[0] + dx, target[1] + dy, target[2] + dz)

            up = _view_up(pos, target)
            pose = CameraPose(
                position=pos,
                look_at=target,
                up=up,
                fov_deg=config.fov_deg,
            )
            _ = camera_basis(pose)
            actual_dist = _norm(_sub(target, pos))

            frames.append(
                TrajectoryFrame(
                    frame_index=i,
                    normalized_time=t,
                    pose=pose,
                    azimuth_deg=theta_deg,
                    elevation_deg=phi_deg,
                    distance_to_target=actual_dist,
                    timestamp_sec=0.0,
                    arc_distance_m=0.0,
                )
            )

        # Compute cumulative path arc distances and timestamps
        cum_dist = 0.0
        new_frames = []
        for i, f in enumerate(frames):
            if i > 0:
                p_prev = frames[i - 1].pose.position
                p_curr = f.pose.position
                cum_dist += _norm(_sub(p_curr, p_prev))
            t_sec = cum_dist / v_speed
            new_frames.append(
                TrajectoryFrame(
                    frame_index=f.frame_index,
                    normalized_time=f.normalized_time,
                    pose=f.pose,
                    azimuth_deg=f.azimuth_deg,
                    elevation_deg=f.elevation_deg,
                    distance_to_target=f.distance_to_target,
                    timestamp_sec=t_sec,
                    arc_distance_m=cum_dist,
                )
            )
        frames = new_frames

    diagnostics = compute_trajectory_diagnostics(frames, base_radius=r_0)
    total_len = frames[-1].arc_distance_m if frames else 0.0
    total_dur = total_len / v_speed if v_speed > 0.0 else 0.0
    traj_id = f"traj_{scene_state.scene_id}_{family}_{config.framing_mode}_s{config.seed}_n{n}"

    return CameraTrajectory(
        trajectory_id=traj_id,
        scene_id=scene_state.scene_id,
        config=config,
        framing=framing,
        frames=frames,
        diagnostics=diagnostics,
        trajectory_family=family,
        total_path_length_m=total_len,
        total_duration_sec=total_dur,
        scientific_speed_m_s=v_speed,
    )
