// Canonical ProcTHOR / AI2-THOR world -> Three.js render-space conversion.
//
// AI2-THOR (Unity) is left-handed: +X right, +Y up, +Z forward/north. Three.js
// is right-handed: +X right, +Y up, +Z toward the viewer. Placing Thor (x, z)
// directly into Three mirrors the floor plan: a top-down Three camera (screen
// up = -ThreeZ) then shows +Z down, while the canonical 2D projection and the
// Unity God camera both show +Z up. That is why the 3D semantic view previously
// looked mirrored and only matched from underneath the floor.
//
// There is exactly ONE conversion, applied to every entity and direction:
//
//     worldToThree([x, y, z]) -> [x, y, -z]
//
// With this mirror, a top-down Three camera above the scene shows +X right and
// +Z up, i.e. the same plan orientation as 2D and Unity. `threeToWorld` is the
// inverse; `worldDirToThree` applies the same mirror to a direction;
// `wallRotationY` / `topViewCamera` / `worldBoundsToThree` derive from it so no
// renderer scatters its own sign flips.

/** Canonical ProcTHOR world point -> Three.js position. */
export function worldToThree(p) {
  return [Number(p[0]), Number(p[1]), -Number(p[2])];
}

/** Inverse of worldToThree (Three.js position -> ProcTHOR world point). */
export function threeToWorld(p) {
  return [Number(p[0]), Number(p[1]), -Number(p[2])];
}

/** Canonical ProcTHOR world direction -> Three.js direction (same mirror). */
export function worldDirToThree(d) {
  return [Number(d[0]), Number(d[1]), -Number(d[2])];
}

/** Thor yaw forward (sin, cos) as a Three-space xz unit direction. */
export function worldYawToThree(yawDeg) {
  const r = ((Number(yawDeg) || 0) * Math.PI) / 180;
  return [Math.sin(r), -Math.cos(r)];
}

/**
 * `rotation.y` that aligns a geometry whose length runs along local +X (wall
 * box / opening plane) with a Thor wall of the given yaw. Derived from the
 * converted wall direction: local +X maps to (cos, -sin) under rotation.y, so
 * matching (sin yaw, -cos yaw) gives PI/2 - yaw.
 */
export function wallRotationY(yawDeg) {
  const r = ((Number(yawDeg) || 0) * Math.PI) / 180;
  return Math.PI / 2 - r;
}

/** World AABB -> Three-space AABB (z mirrored and min/max swapped). */
export function worldBoundsToThree(bounds) {
  const a = worldToThree(bounds.min);
  const b = worldToThree(bounds.max);
  return {
    min: [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2])],
    max: [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.max(a[2], b[2])],
  };
}

/**
 * Top-down camera pose for Three-space bounds. The camera sits above the scene
 * with a tiny +Z offset so `up=(0,1,0)` resolves to screen-up = -ThreeZ =
 * +worldZ, matching the 2D / Unity plan orientation.
 */
export function topViewCamera(bounds) {
  const cx = (bounds.min[0] + bounds.max[0]) / 2;
  const cz = (bounds.min[2] + bounds.max[2]) / 2;
  const size = Math.max(
    bounds.max[0] - bounds.min[0],
    bounds.max[2] - bounds.min[2],
    4
  );
  return {
    position: [cx, size * 1.2, cz + 0.01],
    target: [cx, 0, cz],
  };
}
