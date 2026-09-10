// Pure wall / opening placement math for the God View 3D renderer.
//
// The canonical scene payload gives every wall as a `base` pair (world xz
// endpoints), a `height`, and a derived `segments` list in wall-local
// coordinates: `start`/`end` along the wall from `base[0]`, `y0`/`y1` the
// vertical band. `segments` already has the door / window openings cut out, so
// rendering them renders the actual opening instead of a solid wall.
//
// All output here is in Three.js render space: every world point goes through
// the single canonical `worldToThree` conversion in `world_space.js` (Thor is
// left-handed, Three.js is right-handed), and every rotation derives from the
// converted wall direction via `wallRotationY`. Wall boxes are BoxGeometry
// whose *length* runs along local +x; Plane openings use the same convention.
//
// Keeping this math dependency-free (beyond the world-space module) makes it
// executable under node (see tests/tools/wall_geometry_probe.mjs).

import { worldToThree, wallRotationY } from "./world_space.js";

export const WALL_THICKNESS = 0.06;

const EPS = 1e-4;

/** Solid wall-local bands for one wall (falls back to one full-wall segment). */
export function wallSegments(wall) {
  if (!wall || !wall.base) return [];
  const segs = Array.isArray(wall.segments) ? wall.segments : null;
  if (segs && segs.length) return segs;
  const length = Number(wall.length);
  const height = Number(wall.height);
  if (!(length > EPS) || !(height > EPS)) return [];
  return [{ start: 0, end: length, y0: 0, y1: height }];
}

/** Three.js box placement for one wall-local segment. */
export function wallSegmentBox(wall, seg) {
  const a = worldToThree(wall.base[0]);
  const b = worldToThree(wall.base[1]);
  const dx = b[0] - a[0];
  const dz = b[2] - a[2];
  const length = Math.hypot(dx, dz) || Number(wall.length) || 1;
  const ux = dx / length;
  const uz = dz / length;
  const mid = (Number(seg.start) + Number(seg.end)) / 2;
  const y0 = Number(seg.y0);
  const y1 = Number(seg.y1);
  return {
    center: [a[0] + ux * mid, (y0 + y1) / 2, a[2] + uz * mid],
    size: [Number(seg.end) - Number(seg.start), y1 - y0, WALL_THICKNESS],
    rotationY: wallRotationY(wall.yaw_deg),
  };
}

/** All Three.js boxes for one wall. */
export function wallBoxes(wall) {
  return wallSegments(wall).map((s) => wallSegmentBox(wall, s));
}

/** Three.js plane placement for one door / window opening. */
export function openingBox(opening) {
  if (!opening || !opening.position) return null;
  const width = Number(opening.width);
  const height = Number(opening.height);
  if (!(width > 0) || !(height > 0)) return null;
  const [x, baseY, z] = worldToThree(opening.position);
  return {
    center: [x, baseY + height / 2, z],
    size: [width, height, 0],
    rotationY: wallRotationY(opening.yaw_deg),
  };
}
